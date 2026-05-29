"""P4 dashboard JSON API: GET /api/v1/state and POST /api/v1/escalate.

These cover the operator-facing contract — external monitors will scrape
/api/v1/state and the empty-response-guard manual escalation path is
the documented recovery handle for stalled sessions."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(tmp_path: Path) -> FastAPI:
    from tinyctx import dashboard
    app = FastAPI()
    dashboard.register(app, tmp_path)
    return app


@pytest.fixture(autouse=True)
def _reset_modules():
    from tinyctx import (
        empty_response_guard,
        exec_resume,
        pending_input,
        request_phase,
        session_state,
        stuck_loop,
        synthetic_continue,
    )
    session_state.reset_all()
    request_phase.reset_state()
    empty_response_guard.reset_state()
    stuck_loop.reset_state()
    synthetic_continue.reset_state()
    exec_resume.reset_state()
    yield
    session_state.reset_all()
    request_phase.reset_state()
    empty_response_guard.reset_state()
    stuck_loop.reset_state()
    synthetic_continue.reset_state()
    exec_resume.reset_state()


# ─── GET /api/v1/state ────────────────────────────────────────────────────


def test_state_empty_shape(tmp_path: Path):
    """No active sessions: arrays/dicts present and counts are zero."""
    app = _make_app(tmp_path)
    client = TestClient(app)

    r = client.get("/api/v1/state")
    assert r.status_code == 200
    body = r.json()

    assert "generated_at" in body
    assert isinstance(body["generated_at"], str)
    assert body["counts"] == {
        "active_sessions": 0,
        "force_frontier_flagged": 0,
        "pending_inputs": 0,
    }
    assert body["active"] == []
    assert body["force_frontier_flags"] == {}
    assert body["stuck_loop_state"] == {}
    assert body["exec_resume_history"] == []
    assert body["synthetic_continue_state"] == {}
    assert isinstance(body["exec_resume_state"], dict)
    assert isinstance(body["integrations"], dict)
    assert body["pending_inputs"] == {}


def test_state_with_one_active_session(tmp_path: Path):
    """One phase recorded → one entry in `active` with the correct shape."""
    from tinyctx.request_phase import RequestPhase, set_phase

    set_phase("sid-active", RequestPhase.backend_streaming, "rq_test_1")

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/v1/state")
    assert r.status_code == 200
    body = r.json()

    assert body["counts"]["active_sessions"] == 1
    assert len(body["active"]) == 1
    entry = body["active"][0]
    assert entry["proj_sid"] == "sid-active"
    assert entry["phase"] == "backend_streaming"
    assert entry["request_id"] == "rq_test_1"
    assert isinstance(entry["since_ts"], float)
    assert entry["age_s"] >= 0.0

    # stuck_loop and synthetic_continue surface per-sid even for sessions
    # that haven't tripped them yet — values just default.
    assert "sid-active" in body["stuck_loop_state"]
    assert "sid-active" in body["synthetic_continue_state"]


def test_state_includes_force_frontier_flags(tmp_path: Path):
    """A flagged session shows up in the `force_frontier_flags` map."""
    from tinyctx import empty_response_guard as _erg

    _erg.force_next_to_frontier("sid-flagged", "test_reason")

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/v1/state")
    body = r.json()

    assert body["counts"]["force_frontier_flagged"] == 1
    assert "sid-flagged" in body["force_frontier_flags"]
    assert "manual: test_reason" in body["force_frontier_flags"]["sid-flagged"]["reason"]


# ─── POST /api/v1/escalate ────────────────────────────────────────────────


def test_escalate_sets_force_frontier_flag(tmp_path: Path):
    """Posting a proj_sid sets the empty-response-guard flag with
    reason='manual: manual_api'. The dashboard's GET endpoint then
    surfaces it so operators see the action took effect."""
    from tinyctx import empty_response_guard as _erg

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/v1/escalate", json={"proj_sid": "sid-escalate"})
    assert r.status_code == 202
    assert r.json() == {"escalated": "sid-escalate"}

    flag = _erg.peek_force_frontier("sid-escalate")
    assert flag is not None
    assert "manual_api" in flag["reason"]


def test_escalate_missing_body_returns_400(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/v1/escalate", content=b"")
    assert r.status_code == 400
    assert "error" in r.json()


def test_escalate_missing_proj_sid_returns_400(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/v1/escalate", json={"other_field": "x"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_escalate_non_object_body_returns_400(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/v1/escalate", json=["not", "an", "object"])
    assert r.status_code == 400


# ─── pending input API ────────────────────────────────────────────────────


def test_pending_input_status_and_submit(tmp_path: Path):
    from tinyctx import pending_input

    req = pending_input.create_request(
        "conv-api",
        fields=[{"name": "api_key", "type": "password"}],
        prompt="Need API key",
        ttl_s=60.0,
    )

    app = _make_app(tmp_path)
    client = TestClient(app)

    status = client.get(f"/api/v1/pending-input/{req['request_id']}")
    assert status.status_code == 200
    body = status.json()
    assert body["request_id"] == req["request_id"]
    assert body["submitted"] is False
    assert "value" not in body["fields"][0]

    submitted = client.post(
        f"/api/v1/pending-input/{req['request_id']}",
        json={"values": {"api_key": "sk-secret"}},
    )
    assert submitted.status_code == 202
    assert submitted.json()["submitted"] is True
    assert "sk-secret" not in submitted.text

    consumed = pending_input.consume_submitted("conv-api")
    assert consumed is not None
    assert consumed["values"] == {"api_key": "sk-secret"}


def test_pending_input_submit_pokes_exec_resume_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace
    from tinyctx import exec_resume, pending_input

    calls = []

    async def fake_poke(cwd: str, **kwargs):
        calls.append((cwd, kwargs))
        return SimpleNamespace(
            status="spawned",
            reason="",
            pid=123,
            session_id="session-1",
            log_path="/tmp/resume.log",
        )

    monkeypatch.setattr(exec_resume, "poke", fake_poke)
    req = pending_input.create_request(
        "conv-api",
        fields=[{"name": "api_key", "type": "password"}],
        prompt="Need API key",
        ttl_s=60.0,
        cwd=str(tmp_path),
    )

    app = _make_app(tmp_path)
    client = TestClient(app)
    submitted = client.post(
        f"/api/v1/pending-input/{req['request_id']}",
        json={"values": {"api_key": "sk-secret"}},
    )

    assert submitted.status_code == 202
    assert calls
    assert calls[0][0] == str(tmp_path)
    prompt = calls[0][1]["prompt"]
    assert "pending input" in prompt.lower()
    assert "sk-secret" not in prompt
    assert submitted.json()["resume"]["status"] == "spawned"


def test_state_includes_pending_input_snapshot(tmp_path: Path):
    from tinyctx import pending_input

    req = pending_input.create_request(
        "conv-api",
        fields=[{"name": "api_key", "type": "password"}],
        prompt="Need API key",
        ttl_s=60.0,
    )

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/v1/state")
    body = r.json()

    assert body["counts"]["pending_inputs"] == 1
    assert req["request_id"] in body["pending_inputs"]
    pending = body["pending_inputs"][req["request_id"]]
    assert pending["fields"][0]["type"] == "password"
    assert "value" not in pending["fields"][0]


def test_dashboard_html_contains_pending_input_form_mount(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)

    r = client.get("/dashboard")

    assert r.status_code == 200
    assert 'id="pending-input-list"' in r.text
    assert "/api/v1/pending-input/" in r.text


def test_pending_input_unknown_request_returns_404(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/v1/pending-input/missing").status_code == 404
    assert client.post(
        "/api/v1/pending-input/missing",
        json={"values": {"api_key": "sk-secret"}},
    ).status_code == 404


def test_pending_input_submit_requires_values_object(tmp_path: Path):
    from tinyctx import pending_input

    req = pending_input.create_request(
        "conv-api", fields=[{"name": "api_key", "type": "password"}])
    app = _make_app(tmp_path)
    client = TestClient(app)

    r = client.post(
        f"/api/v1/pending-input/{req['request_id']}",
        json={"values": ["not", "object"]},
    )
    assert r.status_code == 400


def test_pending_input_submit_rejects_empty_values_without_hiding_request(
    tmp_path: Path,
):
    from tinyctx import pending_input

    req = pending_input.create_request(
        "conv-api", fields=[{"name": "api_key", "type": "password"}])
    app = _make_app(tmp_path)
    client = TestClient(app)

    r = client.post(
        f"/api/v1/pending-input/{req['request_id']}",
        json={"values": {}},
    )

    assert r.status_code == 400
    status = pending_input.status(req["request_id"])
    assert status is not None
    assert status["submitted"] is False


# ─── /dashboard/state surfaces request_phase too ──────────────────────────


def test_dashboard_state_includes_request_phase(tmp_path: Path):
    """The legacy /dashboard/state endpoint also surfaces the new
    request_phase block — ensures the dashboard HTML's existing JSON
    consumer can render it without a separate fetch."""
    from tinyctx.request_phase import RequestPhase, set_phase

    set_phase("sid-dash", RequestPhase.injecting, "rq_dash")

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/state")
    assert r.status_code == 200
    body = r.json()
    assert "request_phase" in body
    assert "sid-dash" in body["request_phase"]
    assert body["request_phase"]["sid-dash"]["phase"] == "injecting"
    assert body["request_phase"]["sid-dash"]["age_s"] >= 0.0
