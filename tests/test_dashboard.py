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
    assert "tinyctx dashboard" in r.text
    assert "EventSource" in r.text
    # All 4 dashboard endpoints referenced in the page JS:
    assert "/dashboard/stream" in r.text
    assert "/dashboard/state" in r.text
    assert "/dashboard/aggregates" in r.text
    assert "/dashboard/recent" in r.text


def test_state_endpoint_returns_snapshot(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/dashboard/state")
    assert r.status_code == 200
    body = r.json()
    assert "uptime_s" in body
    assert "proxy_pid" in body
    assert "stuck_loop" in body
    assert "soft_completion" in body
    assert "session_error_streaks" in body


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
         "keepalives_emitted": 0, "error_streak": 0},
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "frontier", "status": 200, "elapsed_s": 12.5,
         "forwarded_bytes": 5000, "bytes_out": 500, "turn_count": 11,
         "keepalives_emitted": 2, "error_streak": 0},
        {"t": now - 20, "event": "request_trace",
         "forced_by_client_model": True, "requested_model": "tinyctx-local",
         "route": "local", "status": 200, "elapsed_s": 0.1,
         "forwarded_bytes": 50, "bytes_out": 10, "turn_count": 0,
         "keepalives_emitted": 0, "error_streak": 0},
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
    assert body["keepalive_saves"] == 1
    assert body["p50_elapsed_s"] > 0


def test_recent_endpoint_returns_formatted_events(tmp_path: Path):
    now = time.time()
    events = [
        {"t": now - 30, "event": "request_trace",
         "forced_by_client_model": False,
         "route": "local", "status": 200, "elapsed_s": 5.0,
         "forwarded_bytes": 1000, "bytes_out": 100, "turn_count": 10,
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
