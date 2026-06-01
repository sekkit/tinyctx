"""Web dashboard endpoints: HTML page, SSE stream, state, aggregates, recent."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_jsonl(p: Path, events: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _today_path(log_dir: Path) -> Path:
    return log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"


def _make_app(tmp_path: Path):
    from fastapi import FastAPI
    from tinyctx import dashboard
    app = FastAPI()
    dashboard.register(app, tmp_path)
    return app


def test_dashboard_html_renders(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "no-store" in r.headers["cache-control"]
    assert "tinyctx dashboard" in r.text
    assert "EventSource" in r.text
    # All 4 dashboard endpoints referenced in the page JS:
    assert "/dashboard/stream" in r.text
    assert "/dashboard/state" in r.text
    assert "/dashboard/aggregates" in r.text
    assert "/dashboard/recent" in r.text
    assert "/dashboard/self-improvement" in r.text
    assert "integration-table" in r.text
    assert "self-improvement-table" in r.text
    assert "request phase" in r.text
    assert "token-stats" in r.text


def test_state_endpoint_returns_snapshot(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/state")
    assert r.status_code == 200
    assert "no-store" in r.headers["cache-control"]
    body = r.json()
    assert "uptime_s" in body
    assert "proxy_pid" in body
    assert "stuck_loop" in body
    assert "soft_completion" in body
    assert "session_error_streaks" in body
    assert "integrations" in body


def test_self_improvement_endpoint_returns_workspace_summary(tmp_path: Path):
    from tinyctx import frontier, trajectory, workspace

    workspace.save_context_profile({"commands": ["pytest"]}, root=tmp_path)
    trajectory.record_event(
        "abc",
        "route",
        root=tmp_path,
        phase="router",
        metrics={"passed": True, "tokens_saved": 10},
    )
    frontier.add_candidate(
        "abc",
        frontier.Candidate(
            candidate_id="router-v1",
            kind="router",
            payload={"threshold": 0.5},
            metrics={"quality": 0.9, "tokens_saved": 0.4},
        ),
        root=tmp_path,
    )

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get(
        "/dashboard/self-improvement",
        params={"session": "abc", "kind": "router"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["commands"] == ["pytest"]
    assert body["sessions"] == ["abc"]
    assert body["trajectory"]["summary"]["by_phase"]["router"] == 1
    assert body["frontier"]["best"]["candidate_id"] == "router-v1"


def test_integrations_endpoint_returns_snapshot(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/integrations")
    assert r.status_code == 200
    assert "no-store" in r.headers["cache-control"]
    body = r.json()
    for name in [
        "context_mode", "gitnexus", "graphify", "serena",
        "advisor", "scout_hook", "caveman",
    ]:
        assert name in body
        assert "installed" in body[name]
        assert "registered" in body[name]
        assert "ready" in body[name]


def test_integrations_endpoint_advisor_requires_agent_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tinyctx import advisor_bootstrap as ab

    app = _make_app(tmp_path)
    client = TestClient(app)

    monkeypatch.setattr(
        ab,
        "detect_state",
        lambda *_args, **_kwargs: ab.State(
            python_path="C:/fake/python.exe",
            python_exists=True,
            codex_config_exists=True,
            codex_config_has_advisor=True,
            codex_config_has_advisor_agent=False,
            advisor_agent_path="C:/Users/test/.codex/agents/advisor.toml",
            advisor_agent_file_exists=False,
        ),
    )

    r = client.get("/dashboard/integrations")
    assert r.status_code == 200
    advisor = r.json()["advisor"]
    assert advisor["installed"] is True
    assert advisor["registered"] is False
    assert advisor["ready"] is False
    assert advisor["details"]["mcp_registered"] is True
    assert advisor["details"]["agent_registered"] is False
    assert advisor["details"]["agent_config_exists"] is False


def test_aggregates_empty_when_no_log(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/aggregates?since_s=900")
    assert r.status_code == 200
    body = r.json()
    assert body["turns_real"] == 0
    assert body["since_s"] == 900


def test_aggregates_clamps_extreme_values(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/aggregates?since_s=10")
    assert r.json()["since_s"] == 60  # clamped up
    r = client.get("/dashboard/aggregates?since_s=999999")
    assert r.json()["since_s"] == 86400  # clamped down


def test_aggregates_rolls_up_real_traces(tmp_path: Path):
    """Two real-codex traces + one test trace — aggregates count only real."""
    now = time.time()
    events = [
        {"t": now - 60, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "local", "status": 200, "elapsed_s": 5.0,
         "forwarded_bytes": 1000, "bytes_out": 100, "turn_count": 10,
         "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100,
         "keepalives_emitted": 0, "error_streak": 0},
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "frontier", "status": 200, "elapsed_s": 12.5,
         "forwarded_bytes": 5000, "bytes_out": 500, "turn_count": 11,
         "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600,
         "keepalives_emitted": 2, "error_streak": 0},
        {"t": now - 20, "event": "request_trace",
         "forced_by_client_model": True, "requested_model": "tinyctx-local",
         "route": "local", "status": 200, "elapsed_s": 0.1,
         "forwarded_bytes": 50, "bytes_out": 10, "turn_count": 0,
         "keepalives_emitted": 0, "error_streak": 0},
        {"t": now - 18, "event": "tool_result_shrink", "shrunk": 2,
         "call_ids": ["c1", "c2"]},
        {"t": now - 10, "event": "stuck_reminder_injected",
         "turn_count": 100, "proj_sid": "global"},
    ]
    _write_jsonl(_today_path(tmp_path), events)

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/aggregates?since_s=900")
    body = r.json()
    assert body["turns_real"] == 2  # test trace excluded
    assert body["by_route"] == {"local": 1, "frontier": 1}
    assert body["stuck_reminders"] == 1
    assert body["tool_result_shrinks"] == 2
    assert body["keepalive_saves"] == 1
    assert body["prompt_cache_hit_tokens"] == 1300
    assert body["prompt_cache_miss_tokens"] == 700
    assert body["prompt_cache_hit_ratio"] == 0.65
    assert body["p50_elapsed_s"] > 0


def test_recent_endpoint_returns_formatted_events(tmp_path: Path):
    now = time.time()
    events = [
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "local", "status": 200, "elapsed_s": 5.0,
         "forwarded_bytes": 1000, "bytes_out": 100, "turn_count": 10,
         "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100,
         "keepalives_emitted": 0, "error_streak": 0},
        {"t": now - 20, "event": "soft_completion_classified",
         "soft_punt": True, "p": 0.95, "reason": "asks user what next"},
        # uninteresting events filtered out
        {"t": now - 15, "event": "mutation_gate", "fire": False},
    ]
    _write_jsonl(_today_path(tmp_path), events)

    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/recent?limit=10")
    body = r.json()
    assert isinstance(body, list)
    # mutation_gate filtered out
    kinds = [e["event"] for e in body]
    assert "request_trace" in kinds
    assert "soft_completion_classified" in kinds
    assert "mutation_gate" not in kinds
    req = next(e for e in body if e["event"] == "request_trace")
    assert req["prompt_cache_hit_ratio"] == 0.9


def test_recent_request_trace_includes_execution_mode(tmp_path: Path):
    now = time.time()
    events = [
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "local", "status": 200, "elapsed_s": 5.0,
         "forwarded_bytes": 1000, "bytes_out": 100, "turn_count": 10,
         "keepalives_emitted": 0, "error_streak": 0,
         "orchestrator_injected": True,
         "orchestrator_task_type": "review",
         "orchestrator_confidence": 0.93,
         "orchestrator_execution_mode": "parallel_subagents",
         "orchestrator_execution_reason": "independent review lanes",
         "orchestrator_parallel_subtasks": [
             {"title": "API pass", "agent": "reviewer", "prompt": "review API"},
         ]},
    ]
    _write_jsonl(_today_path(tmp_path), events)

    app = _make_app(tmp_path)
    client = TestClient(app)
    body = client.get("/dashboard/recent?limit=10").json()

    item = body[0]
    assert item["orchestrator_execution_mode"] == "parallel_subagents"
    assert item["orchestrator_parallel_subtasks"][0]["title"] == "API pass"


def test_recent_filters_out_pytest_traffic(tmp_path: Path):
    now = time.time()
    events = [
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": True, "requested_model": "tinyctx-local",
         "route": "local", "status": 200, "elapsed_s": 0.1,
         "forwarded_bytes": 50, "bytes_out": 10, "turn_count": 0,
         "keepalives_emitted": 0, "error_streak": 0},
        # Advisor traffic IS kept
        {"t": now - 20, "event": "request_trace",
         "forced_by_client_model": True, "requested_model": "tinyctx-frontier",
         "route": "frontier", "status": 200, "elapsed_s": 3.0,
         "forwarded_bytes": 8000, "bytes_out": 30000, "turn_count": 0,
         "keepalives_emitted": 0, "error_streak": 0},
    ]
    _write_jsonl(_today_path(tmp_path), events)
    app = _make_app(tmp_path)
    client = TestClient(app)
    body = client.get("/dashboard/recent?limit=10").json()
    # tinyctx-local pytest-style trace should be filtered; advisor kept
    advisor = [e for e in body if e.get("kind") == "advisor"]
    assert len(advisor) == 1
    main = [e for e in body if e.get("kind") == "main"]
    assert len(main) == 0


def test_format_event_filters_uninteresting():
    """Module-level filter excludes events the dashboard doesn't render."""
    from tinyctx.dashboard import _format_event_for_dashboard
    # Uninteresting noise event types
    assert _format_event_for_dashboard({"event": "mutation_gate"}) is None
    assert _format_event_for_dashboard({"event": "stream_done"}) is None
    assert _format_event_for_dashboard({"event": "route"}) is None
    # Interesting types pass through
    out = _format_event_for_dashboard({
        "event": "stuck_reminder_injected", "turn_count": 50, "proj_sid": "x"})
    assert out is not None
    assert out["turn_count"] == 50


def test_format_event_keeps_failure_signal_escalation():
    from tinyctx.dashboard import _format_event_for_dashboard

    out = _format_event_for_dashboard({
        "event": "failure_signal_escalated_to_frontier",
        "score": 2,
        "signals": [{"kind": "tool_call_storm", "tool_name": "ls", "count": 3}],
    })
    assert out is not None
    assert out["score"] == 2
    assert out["signals"][0]["kind"] == "tool_call_storm"


def test_format_event_keeps_tool_result_shrink():
    from tinyctx.dashboard import _format_event_for_dashboard

    out = _format_event_for_dashboard({
        "event": "tool_result_shrink",
        "shrunk": 2,
        "call_ids": ["c1", "c2"],
    })
    assert out is not None
    assert out["shrunk"] == 2
    assert out["call_ids"] == ["c1", "c2"]


def test_state_includes_in_memory_dicts(tmp_path: Path):
    """State endpoint reads live module-level dicts. Inject a value and
    verify it appears."""
    from tinyctx import stuck_loop, soft_completion
    stuck_loop.reset_state()
    soft_completion.reset_state()
    stuck_loop._LAST_REMINDER_TURN["test:abc"] = 99
    stuck_loop._LAST_ADVISOR_TS["test:abc"] = time.time()
    soft_completion._set_flag_for_test("test:abc", reason="asks", p=0.9)

    app = _make_app(tmp_path)
    client = TestClient(app)
    body = client.get("/dashboard/state").json()
    assert body["stuck_loop"]["last_reminder_turn"]["test:abc"] == 99
    assert "test:abc" in body["soft_completion"]["active_flags"]


def test_autoresearch_scan_is_cached(monkeypatch):
    """Regression: state_snapshot's autoresearch scan must not os.walk the
    home tree on every call — repeated polls used to freeze /dashboard/state."""
    import os as _os

    from tinyctx import dashboard

    calls = {"n": 0}

    def counting_walk(base, *a, **k):
        calls["n"] += 1
        return iter([(base, [], [])])

    monkeypatch.setattr(_os, "walk", counting_walk)
    dashboard.reset_autoresearch_cache()
    dashboard._scan_autoresearch_runs()          # cold: walks ~ and /tmp
    cold = calls["n"]
    assert cold == 2
    dashboard._scan_autoresearch_runs()          # warm: cached, no new walk
    assert calls["n"] == cold
    dashboard.reset_autoresearch_cache()
    dashboard._scan_autoresearch_runs()          # reset: walks again
    assert calls["n"] == cold + 2


def test_autoresearch_scan_prunes_and_caps():
    """The walk prunes known-huge subtrees and is hard-capped."""
    from tinyctx import dashboard

    assert {"node_modules", "Library", "site-packages"} <= dashboard._AR_PRUNE
    assert dashboard._AR_MAX_DIRS > 0
    dashboard.reset_autoresearch_cache()
    assert dashboard._AR_CACHE["runs"] == {}
