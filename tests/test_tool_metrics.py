"""Tool-call frequency tracking — verify mining + classification +
dedup contract. Live trace 2026-05-10: needed visibility into which
MCP servers / built-in tools the agent actually calls; existing
request_trace event recorded counts but not names."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tinyctx import tool_metrics
    tool_metrics.reset_state()
    yield
    tool_metrics.reset_state()


# ─── classification ───────────────────────────────────────────────────────


def test_classify_mcp_server_namespace():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("mcp__advisor__ask_advisor") == ("mcp:advisor", "ask_advisor")
    assert _classify_tool("mcp__gitnexus__find_definition") == ("mcp:gitnexus", "find_definition")
    assert _classify_tool("mcp__playwright__navigate") == ("mcp:playwright", "navigate")


def test_classify_agent_protocol():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("spawn_agent") == ("agent_protocol", "spawn_agent")
    assert _classify_tool("wait_agent") == ("agent_protocol", "wait_agent")
    assert _classify_tool("close_agent") == ("agent_protocol", "close_agent")
    assert _classify_tool("request_user_input") == ("agent_protocol", "request_user_input")


def test_classify_builtin_tools():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("shell") == ("builtin", "shell")
    assert _classify_tool("apply_patch") == ("builtin", "apply_patch")
    assert _classify_tool("container.exec") == ("builtin", "container.exec")


def test_classify_tracker():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("update_plan") == ("tracker", "update_plan")
    assert _classify_tool("TodoWrite") == ("tracker", "TodoWrite")


def test_classify_unknown_falls_into_other():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("some_random_thing") == ("other", "some_random_thing")


def test_classify_handles_invalid():
    from tinyctx.tool_metrics import _classify_tool
    assert _classify_tool("") == ("invalid", "")
    assert _classify_tool(None) == ("invalid", "")  # type: ignore[arg-type]


# ─── recording ────────────────────────────────────────────────────────────


def test_record_increments_count_for_unique_call_ids():
    from tinyctx import tool_metrics as tm
    body = {
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "shell"},
            {"type": "function_call", "call_id": "c2",
             "name": "mcp__advisor__ask_advisor"},
            {"type": "function_call", "call_id": "c3", "name": "spawn_agent"},
        ]
    }
    n = tm.record_from_body(body)
    assert n == 3
    snap = tm.snapshot()
    assert snap["total_calls"] == 3
    assert snap["by_namespace"]["builtin"] == 1
    assert snap["by_namespace"]["mcp:advisor"] == 1
    assert snap["by_namespace"]["agent_protocol"] == 1


def test_record_dedups_repeated_call_ids():
    from tinyctx import tool_metrics as tm
    body1 = {
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "shell"},
        ]
    }
    body2 = {
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "shell"},  # repeat
            {"type": "function_call", "call_id": "c2", "name": "shell"},  # new
        ]
    }
    assert tm.record_from_body(body1) == 1
    # body2 brings 1 new (c2); c1 dedupes
    assert tm.record_from_body(body2) == 1
    snap = tm.snapshot()
    assert snap["total_calls"] == 2  # c1 + c2, not 3


def test_record_skips_calls_without_call_id():
    """No call_id → can't dedup → must skip (rather than over-count)."""
    from tinyctx import tool_metrics as tm
    body = {
        "input": [
            {"type": "function_call", "name": "shell"},  # missing call_id
            {"type": "function_call", "call_id": "", "name": "shell"},  # empty
            {"type": "function_call", "call_id": "c1", "name": "shell"},  # valid
        ]
    }
    assert tm.record_from_body(body) == 1


def test_record_ignores_non_function_call_items():
    from tinyctx import tool_metrics as tm
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "function_call", "call_id": "c2", "name": "shell"},
        ]
    }
    assert tm.record_from_body(body) == 1


def test_record_handles_malformed_body():
    """Hot-path safety: never raise."""
    from tinyctx import tool_metrics as tm
    assert tm.record_from_body(None) == 0  # type: ignore[arg-type]
    assert tm.record_from_body({}) == 0
    assert tm.record_from_body({"input": "not a list"}) == 0
    assert tm.record_from_body({"input": [None, "string", 42]}) == 0


# ─── snapshot ─────────────────────────────────────────────────────────────


def test_snapshot_sorts_by_call_count_descending():
    from tinyctx import tool_metrics as tm
    body = {
        "input": [
            {"type": "function_call", "call_id": "a1", "name": "shell"},
            {"type": "function_call", "call_id": "a2", "name": "shell"},
            {"type": "function_call", "call_id": "a3", "name": "shell"},
            {"type": "function_call", "call_id": "b1", "name": "apply_patch"},
            {"type": "function_call", "call_id": "c1", "name": "spawn_agent"},
        ]
    }
    tm.record_from_body(body)
    snap = tm.snapshot()
    by_tool = snap["by_tool"]
    # Most-called first
    assert by_tool[0]["tool"] == "shell"
    assert by_tool[0]["calls"] == 3


def test_snapshot_includes_last_seen_age():
    from tinyctx import tool_metrics as tm
    body = {"input": [{"type": "function_call", "call_id": "c1",
                        "name": "shell"}]}
    tm.record_from_body(body)
    snap = tm.snapshot()
    e = snap["by_tool"][0]
    assert e["last_seen_age_s"] is not None
    assert e["last_seen_age_s"] >= 0
    assert e["last_seen_age_s"] < 5  # must be very recent
