"""Tests for sanitize.proactive_compact — proxy-side history truncation
that fires when est_tokens crosses a danger threshold.

This is the safety net for codex.app 0.128's "Codex ran out of room" error
when codex's own auto-compact didn't fire (because gpt-5.5 ships with
auto_compact_token_limit=null and the user's profile-scoped override only
applies to one profile, not the default profile being run).
"""
from __future__ import annotations

import json

from tinyctx.sanitize import (
    clear_proactive_cache,
    proactive_compact,
)


def _make_body(n_turns: int, *, with_system_head: bool = False,
               with_codex_compact: bool = False) -> dict:
    """Build a minimal Responses-API body with `n_turns` user/assistant
    pairs in `input`."""
    body: dict = {
        "model": "tinyctx-auto",
        "instructions": (
            "You are performing a CONTEXT CHECKPOINT COMPACTION. "
            "Create a handoff summary for another LLM that will resume the task."
            if with_codex_compact
            else "You are a coding agent."
        ),
        "input": [],
    }
    if with_system_head:
        body["input"].append({
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "system bytes"}],
        })
    for i in range(n_turns):
        body["input"].append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"user turn {i}"}],
        })
        body["input"].append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"assistant reply {i}"}],
        })
    return body


def test_proactive_compact_below_threshold_noop():
    body = _make_body(20)
    out, info = proactive_compact(
        body,
        session_id="s1",
        est_tokens=50_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "below_threshold"
    # body unchanged
    assert out is body or out == body


def test_proactive_compact_skips_codex_compaction_request():
    body = _make_body(50, with_codex_compact=True)
    out, info = proactive_compact(
        body,
        session_id="s2",
        est_tokens=300_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "skip_codex_compaction"


def test_proactive_compact_skips_when_no_input_array():
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    out, info = proactive_compact(
        body,
        session_id="s3",
        est_tokens=300_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "no_input_array"


def test_proactive_compact_skips_when_too_few_items():
    body = _make_body(2)  # 4 items < recent_keep(8) + 3
    out, info = proactive_compact(
        body,
        session_id="s4",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is False
    assert "too_few_items" in info["reason"]


def test_proactive_compact_truncates_middle_default_placeholder():
    clear_proactive_cache()
    body = _make_body(30)  # 60 items
    n_before = len(body["input"])
    out, info = proactive_compact(
        body,
        session_id="s5",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is True
    assert info["items_before"] == n_before
    # head=0 (no system items) + 1 summary + 8 recent = 9
    assert info["items_after"] == 9
    assert info["middle_items_compacted"] == n_before - 8
    assert "compacted" in info["reason"]
    assert info["cached"] is False

    # Last 8 items should be the same as the original last 8.
    assert out["input"][-8:] == body["input"][-8:]
    # The summary item is in the middle.
    summary_item = out["input"][-9]
    assert summary_item["role"] == "user"
    assert summary_item["type"] == "message"
    text = summary_item["content"][0]["text"]
    assert "tinyctx auto-compact" in text
    assert "older turns" in text or "omitted to fit context" in text


def test_proactive_compact_keeps_system_head_items():
    clear_proactive_cache()
    body = _make_body(20, with_system_head=True)
    out, info = proactive_compact(
        body,
        session_id="s6",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is True
    # First item should still be the system message we put in
    first = out["input"][0]
    assert first["role"] == "system"
    # Then summary
    assert out["input"][1]["role"] == "user"
    text = out["input"][1]["content"][0]["text"]
    assert "tinyctx auto-compact" in text
    # Then 8 recent
    assert len(out["input"]) == 1 + 1 + 8


def test_proactive_compact_caches_summary_for_same_middle():
    clear_proactive_cache()
    body = _make_body(30)
    out1, info1 = proactive_compact(
        body, session_id="s7", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info1["applied"] is True
    assert info1["cached"] is False

    # Identical body again — should hit cache
    out2, info2 = proactive_compact(
        body, session_id="s7", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info2["applied"] is True
    assert info2["cached"] is True


def test_proactive_compact_uses_summarizer_when_provided():
    clear_proactive_cache()
    body = _make_body(30)
    captured_blob: dict = {}

    def fake_summarizer(blob: str) -> str:
        captured_blob["blob"] = blob
        return "FAKE SUMMARY: 30 turns about adding compact support."

    out, info = proactive_compact(
        body, session_id="s8", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=fake_summarizer,
    )
    assert info["applied"] is True
    summary_text = out["input"][-9]["content"][0]["text"]
    assert "FAKE SUMMARY" in summary_text
    # Summarizer was given a blob with the middle turns
    assert "user turn 0" in captured_blob["blob"]
    assert "assistant reply 0" in captured_blob["blob"]


def test_proactive_compact_summarizer_failure_falls_back_to_placeholder():
    clear_proactive_cache()
    body = _make_body(30)

    def crashing_summarizer(blob: str) -> str:
        raise RuntimeError("backend dead")

    out, info = proactive_compact(
        body, session_id="s9", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=crashing_summarizer,
    )
    # NEVER fail the request — only quality regression
    assert info["applied"] is True
    summary_text = out["input"][-9]["content"][0]["text"]
    assert "backend dead" in summary_text or "summarizer failed" in summary_text


def test_proactive_compact_does_not_mutate_input_body():
    clear_proactive_cache()
    body = _make_body(30)
    snapshot = json.dumps(body, sort_keys=True)
    proactive_compact(
        body, session_id="s10", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    # original body must be untouched (deepcopy semantics)
    assert json.dumps(body, sort_keys=True) == snapshot


def test_clear_proactive_cache_session_scoped():
    clear_proactive_cache()
    body = _make_body(30)

    def s1_summarizer(blob: str) -> str:
        return "s1-specific summary"

    out1, info1 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info1["cached"] is False
    # cache hit
    out2, info2 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info2["cached"] is True

    # Clear only sA
    clear_proactive_cache("sA")
    out3, info3 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info3["cached"] is False
