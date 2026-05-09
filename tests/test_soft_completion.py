"""Soft-completion gate: stream sniffer detects "soft punt to user"
patterns; the next request to that session gets a `<system-reminder>`
forcing the agent to route the would-be question through advisor.
"""
from __future__ import annotations

import pytest


# ─── pattern detection (chunk-level) ────────────────────────────────────────


def test_detect_what_would_you_like():
    """Canonical English pattern from live trace."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    matched = soft_completion.scan_chunk(
        "p1", b"...some text...\nWhat would you like to work on next?")
    assert matched == "en_what_would_you_like"
    assert soft_completion.get_flag("p1") is not None


def test_detect_options_list():
    """`Some options:` followed by newline is a strong soft-punt signal."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    matched = soft_completion.scan_chunk(
        "p1", b"...analysis done.\n\nSome options:\n- continue\n- stop")
    assert matched == "en_options_list"


def test_detect_zh_what_next():
    """Chinese soft-completion ("你想接下来做什么")."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    matched = soft_completion.scan_chunk(
        "p1", "总结完成。你想接下来做什么？".encode("utf-8"))
    assert matched == "zh_what_next"


def test_no_match_on_legit_content():
    """Legitimate technical content must not false-fire."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    legit = (b"def what_to_do(): return 'process'\n"
             b"# returns the next step based on input\n"
             b"queue.peek() to inspect the next item")
    matched = soft_completion.scan_chunk("p1", legit)
    assert matched is None
    assert soft_completion.get_flag("p1") is None


def test_pattern_split_across_chunks():
    """SSE chunks can split a pattern across two `scan_chunk` calls.
    The per-session ring buffer must survive across calls so the regex
    sees the full phrase."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    # Split "What would you like to work on next?" across 3 chunks
    m1 = soft_completion.scan_chunk("p1", b"...What would")
    m2 = soft_completion.scan_chunk("p1", b" you like ")
    m3 = soft_completion.scan_chunk("p1", b"to work on next?")
    assert m1 is None
    assert m2 is None
    assert m3 == "en_what_would_you_like"


def test_fires_only_once_per_stream():
    """Once matched, subsequent chunks in the same stream are no-ops."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    m1 = soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")
    assert m1 == "en_what_would_you_like"
    # Another matchable phrase later in the same stream → no re-match
    m2 = soft_completion.scan_chunk(
        "p1", b"\n\nWhat would you like to work on next?")
    assert m2 is None  # already flagged, sniffer short-circuits


def test_reset_stream_clears_buffer_not_flag():
    """`reset_stream` is called at start of each stream — clears the
    accumulated text buffer but leaves the flag intact (the flag must
    survive until the gate consumes it on the next request)."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")
    assert soft_completion.get_flag("p1") is not None
    soft_completion.reset_stream("p1")
    # flag still active
    assert soft_completion.get_flag("p1") is not None
    snap = soft_completion.state_snapshot("p1")
    assert snap["buffer_chars"] == 0


# ─── per-session isolation ─────────────────────────────────────────────────


def test_state_isolated_by_proj_sid():
    """Project A flag must not propagate to project B."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "projA", b"What would you like to work on next?")
    assert soft_completion.get_flag("projA") is not None
    assert soft_completion.get_flag("projB") is None


# ─── gate injection ────────────────────────────────────────────────────────


def test_gate_no_inject_when_flag_unset():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    body = {"input": [{"role": "user", "content": "hi"}]}
    out, gated, pat = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is False
    assert pat == ""
    assert out is body


def test_gate_injects_when_flag_set():
    """After detection, the next request gets the advisor-vet reminder
    appended to body.input."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")

    body = {"input": [{"role": "user", "content": "thanks for the summary"}]}
    out, gated, pat = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is True
    assert pat == "en_what_would_you_like"
    assert len(out["input"]) == 2
    last = out["input"][-1]
    assert last["role"] == "user"
    text = last["content"][0]["text"]
    assert "<system-reminder>" in text
    assert "soft-completion gate" in text
    assert "spawn_agent(role=\"advisor\"" in text
    assert "ask:" in text and "work:" in text
    # pattern is interpolated for traceability
    assert "en_what_would_you_like" in text


def test_gate_fires_once_then_clears_flag():
    """After injection, the flag is consumed — second call returns no-op."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")

    body = {"input": [{"role": "user", "content": "x"}]}
    _, gated1, _ = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated1 is True

    # Same body, second call → flag already consumed
    body2 = {"input": [{"role": "user", "content": "y"}]}
    _, gated2, _ = soft_completion.maybe_inject_soft_completion_gate(
        body2, "p1")
    assert gated2 is False


def test_gate_does_not_mutate_original_body():
    """Original body and its input list must remain untouched."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")

    original = {"input": [{"role": "user", "content": "x"}]}
    items_id = id(original["input"])
    out, gated, _ = soft_completion.maybe_inject_soft_completion_gate(
        original, "p1")
    assert gated is True
    assert len(original["input"]) == 1
    assert id(original["input"]) == items_id
    assert len(out["input"]) == 2


def test_gate_skips_malformed_body():
    """Bodies without an input array → no-op without exception."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.scan_chunk(
        "p1", b"What would you like to work on next?")
    body = {"messages": [{"role": "user", "content": "hi"}]}  # chat-style
    out, gated, _ = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is False


# ─── default config ────────────────────────────────────────────────────────


def test_config_default_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.soft_completion_gate_enabled is True


def test_trace_fields_default_off():
    from tinyctx.trace import RequestTrace
    t = RequestTrace()
    assert t.soft_completion_detected is False
    assert t.soft_completion_pattern == ""
    assert t.soft_completion_gate_injected is False
    assert t.soft_completion_gate_pattern == ""


# ─── error tolerance ──────────────────────────────────────────────────────


def test_scan_chunk_tolerates_malformed_utf8():
    """Partial UTF-8 sequences (mid-codepoint chunks) must not raise."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    # 0xE4 starts a 3-byte sequence; alone it's malformed
    m = soft_completion.scan_chunk("p1", b"\xe4 incomplete utf8")
    # No match (not the pattern), and no exception either
    assert m is None
