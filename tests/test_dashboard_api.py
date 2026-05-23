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
        request_phase,
        stuck_loop,
        synthetic_continue,
    )
    request_phase.reset_state()
    empty_response_guard.reset_state()
    stuck_loop.reset_state()
    synthetic_continue.reset_state()
    exec_resume.reset_state()
    yield
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
    assert body["counts"] == {"active_sessions": 0, "force_frontier_flagged": 0}
    assert body["active"] == []
    assert body["force_frontier_flags"] == {}
    assert body["stuck_loop_state"] == {}
    assert body["exec_resume_history"] == []
    assert body["synthetic_continue_state"] == {}
    assert isinstance(body["exec_resume_state"], dict)
    assert isinstance(body["integrations"], dict)


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
