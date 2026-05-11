"""Empty-response detection + force-frontier guard."""
from __future__ import annotations

import time

import pytest


# ─── usage extraction ─────────────────────────────────────────────────────


def test_extract_chat_completion_tokens():
    """Chat-Completions emits `completion_tokens` in usage block."""
    from tinyctx.empty_response_guard import _extract_tail_usage
    buf = ('data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],'
           '"usage":{"prompt_tokens":1000,"completion_tokens":1,'
           '"total_tokens":1001}}\n\n'
           'data: [DONE]\n\n')
    completion, finish = _extract_tail_usage(buf)
    assert completion == 1
    assert finish == "stop"


def test_extract_responses_api_output_tokens():
    """Responses-API emits `output_tokens` instead, status field too."""
    from tinyctx.empty_response_guard import _extract_tail_usage
    buf = ('event: response.completed\n'
           'data: {"type":"response.completed","response":{"status":"completed",'
           '"usage":{"input_tokens":1000,"output_tokens":3}}}\n\n')
    completion, finish = _extract_tail_usage(buf)
    assert completion == 3
    assert finish == "completed"


def test_extract_returns_none_when_no_usage():
    from tinyctx.empty_response_guard import _extract_tail_usage
    assert _extract_tail_usage("") == (None, "")
    assert _extract_tail_usage("data: not json") == (None, "")


def test_extract_finish_reason_length():
    from tinyctx.empty_response_guard import _extract_tail_usage
    buf = '"completion_tokens": 4096, "finish_reason": "length"'
    completion, finish = _extract_tail_usage(buf)
    assert completion == 4096
    assert finish == "length"


# ─── flag detection ───────────────────────────────────────────────────────


def test_flag_set_on_empty_response():
    """Live trace pattern: 1 token + finish_reason=stop → flag set."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = ('data: {"choices":[{"delta":{"content":""},'
           '"finish_reason":"stop"}],'
           '"usage":{"prompt_tokens":700000,"completion_tokens":1}}\n\n'
           'data: [DONE]\n\n')
    info = empty_response_guard.maybe_flag_empty_response("p1", buf)
    assert info is not None
    assert info["completion_tokens"] == 1
    assert info["finish_reason"] == "stop"
    assert empty_response_guard.peek_force_frontier("p1") is not None


def test_flag_not_set_on_normal_response():
    """100 tokens + finish_reason=stop → no flag (substantive response)."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = ('"usage":{"completion_tokens":100,"prompt_tokens":1000}'
           '"finish_reason":"stop"')
    info = empty_response_guard.maybe_flag_empty_response("p1", buf)
    assert info is None
    assert empty_response_guard.peek_force_frontier("p1") is None


def test_flag_not_set_on_tool_calls():
    """1 token + finish_reason=tool_calls → no flag (agent IS acting)."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = ('"usage":{"completion_tokens":1}'
           '"finish_reason":"tool_calls"')
    info = empty_response_guard.maybe_flag_empty_response("p1", buf)
    assert info is None  # tool_calls is not a "normal stop"


def test_flag_set_on_length_truncation():
    """finish_reason=length is also a normal "model intended to be done"
    case (it ran out of budget). 1 token + length → flag."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = ('"usage":{"completion_tokens":2}'
           '"finish_reason":"length"')
    info = empty_response_guard.maybe_flag_empty_response("p1", buf)
    assert info is not None
    assert info["finish_reason"] == "length"


def test_flag_threshold_configurable():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = ('"usage":{"completion_tokens":10}'
           '"finish_reason":"stop"')
    # Default threshold 5 — 10 tokens passes through
    assert empty_response_guard.maybe_flag_empty_response(
        "p1", buf) is None
    # Stricter threshold 20 — 10 tokens flagged
    assert empty_response_guard.maybe_flag_empty_response(
        "p1", buf, min_completion_tokens=20) is not None


# ─── flag consume ────────────────────────────────────────────────────────


def test_consume_returns_info_then_clears():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    buf = '"usage":{"completion_tokens":1}"finish_reason":"stop"'
    empty_response_guard.maybe_flag_empty_response("p1", buf)

    info = empty_response_guard.consume_force_frontier("p1")
    assert info is not None
    assert info["completion_tokens"] == 1
    # Now consumed — second call returns None
    info2 = empty_response_guard.consume_force_frontier("p1")
    assert info2 is None


def test_peek_does_not_consume():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    empty_response_guard.force_next_to_frontier("p1", "test")
    assert empty_response_guard.peek_force_frontier("p1") is not None
    # Second peek still sees it
    assert empty_response_guard.peek_force_frontier("p1") is not None
    # Then consume
    assert empty_response_guard.consume_force_frontier("p1") is not None
    assert empty_response_guard.consume_force_frontier("p1") is None


# ─── manual trigger (for testing / recovery) ──────────────────────────────


def test_manual_force_for_recovery():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    empty_response_guard.force_next_to_frontier(
        "global", "user-requested manual recovery")
    info = empty_response_guard.consume_force_frontier("global")
    assert info is not None
    assert "manual" in info["reason"]


# ─── per-session isolation ────────────────────────────────────────────────


def test_isolation_by_proj_sid():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    empty_response_guard.force_next_to_frontier("projA", "x")
    assert empty_response_guard.consume_force_frontier("projA") is not None
    assert empty_response_guard.consume_force_frontier("projB") is None


def test_state_snapshot_for_dashboard():
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    empty_response_guard.force_next_to_frontier("projA", "test1")
    empty_response_guard.force_next_to_frontier("projB", "test2")
    snap = empty_response_guard.state_snapshot()
    assert "projA" in snap
    assert "projB" in snap
    assert "test1" in snap["projA"]["reason"]


# ─── default config ──────────────────────────────────────────────────────


def test_default_config_enabled_with_sane_threshold():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.empty_response_guard_enabled is True
    # Threshold low enough to catch the 1-token failure mode but high
    # enough to allow brief acks ("OK", "Done.")
    assert 1 <= cfg.empty_response_min_completion_tokens <= 20


# ─── multi-conversation isolation (conv_sid scoping) ─────────────────────


def test_force_frontier_isolated_by_conversation():
    """Flag set under one conv_sid must NOT be consumed by a different
    conv_sid in the same project. Observed bug: user opened a fresh
    conversation asking for gpt-5.4-mini and got force-routed to gpt-5.5
    because an old flag from the previous conversation was still set."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    proj = "cwd-hash:global"
    conv_a = f"{proj}:019e0741-aaaa"
    conv_b = f"{proj}:019e14c1-bbbb"
    empty_response_guard.force_next_to_frontier(conv_a, "test")
    assert empty_response_guard.consume_force_frontier(conv_b) is None
    info = empty_response_guard.consume_force_frontier(conv_a)
    assert info is not None


def test_advisor_sub_thread_force_frontier_isolated():
    """Force-frontier flag on advisor sub-thread (its own prompt_cache_key)
    must NOT trigger force-frontier on the parent thread's next turn."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    main = "proj:main-uuid"
    advisor = "proj:advisor-uuid"
    empty_response_guard.force_next_to_frontier(advisor, "advisor empty")
    assert empty_response_guard.consume_force_frontier(main) is None


def test_back_compat_when_no_conversation_id():
    """When proxy falls back to proj_sid (no prompt_cache_key in body),
    the flag still sets + consumes correctly — old usage keeps working."""
    from tinyctx import empty_response_guard
    empty_response_guard.reset_state()
    empty_response_guard.force_next_to_frontier("proj-only", "manual")
    info = empty_response_guard.consume_force_frontier("proj-only")
    assert info is not None


# ─── Bug B: consuming a conv_sid flag clears dangling proj_sid flag ──────


def test_consume_conv_sid_flag_also_clears_proj_sid_flag():
    """Mirrors the proxy.py consume path: when conv_sid flag is consumed
    (returns truthy) we also unconditionally clear the proj_sid flag for
    the same project. Without this, a dangling proj_sid flag set by
    e.g. exec_resume / stall escalation that happened AFTER the conv_sid
    flag would be picked up later by a DIFFERENT conversation's fallback
    and force-route it for no reason."""
    from tinyctx import empty_response_guard as _erg
    _erg.reset_state()
    proj_sid = "global"
    conv_sid = f"{proj_sid}:conv-1"
    # Both flags set: stall under conv_sid, then exec_resume under proj_sid.
    _erg.force_next_to_frontier(conv_sid, "stall under conv")
    _erg.force_next_to_frontier(proj_sid, "exec_resume under proj")
    # Simulate proxy.py consume sequence.
    force_info = _erg.consume_force_frontier(conv_sid)
    if force_info is None and conv_sid != proj_sid:
        force_info = _erg.consume_force_frontier(proj_sid)
    elif force_info is not None and conv_sid != proj_sid:
        _erg.reset_state(proj_sid)
    assert force_info is not None
    # The proj_sid flag is gone now — a different conversation's fallback
    # must NOT find it.
    other_conv = f"{proj_sid}:conv-2"
    leftover = _erg.consume_force_frontier(proj_sid)
    assert leftover is None
    # And from the perspective of conv-2's exact same fallback chain:
    assert _erg.consume_force_frontier(other_conv) is None
    assert _erg.consume_force_frontier(proj_sid) is None
