"""Tests for choice_arbiter — judge parsing, verdict storage, injection,
and the full intercept pipeline."""
from __future__ import annotations

import pytest

from tinyctx.choice_arbiter import (
    JudgeResult,
    Verdict,
    _parse_judge_response,
    _strip_thinking,
    consume_verdict,
    inject_verdict_into_body,
    reset_state,
    store_verdict,
)


# ─── _strip_thinking ────────────────────────────────────────────────────

@pytest.mark.parametrize("inp,expected", [
    ("hello", "hello"),
    ("<think>reasoning</think> answer", " answer"),
    ("<THINK>blah</THINK>done", "done"),
    ("", ""),
])
def test_strip_thinking(inp, expected):
    assert _strip_thinking(inp) == expected


# ─── _parse_judge_response ──────────────────────────────────────────────

def test_parse_choice_ask_yes():
    raw = '{"is_choice_ask": true, "question": "A or B?", "options": ["A", "B"], "context_summary": "picking lib"}'
    r = _parse_judge_response(raw)
    assert r is not None
    assert r.is_choice_ask is True
    assert r.question == "A or B?"
    assert r.options == ["A", "B"]
    assert r.context_summary == "picking lib"


def test_parse_choice_ask_no():
    raw = '{"is_choice_ask": false, "question": "", "options": [], "context_summary": ""}'
    r = _parse_judge_response(raw)
    assert r is not None
    assert r.is_choice_ask is False
    assert r.options == []


def test_parse_with_thinking():
    raw = '<think>hmm</think> {"is_choice_ask": true, "question": "X?", "options": ["X", "Y"], "context_summary": "test"}'
    r = _parse_judge_response(raw)
    assert r is not None
    assert r.is_choice_ask is True


def test_parse_garbage_returns_none():
    assert _parse_judge_response("not json") is None
    assert _parse_judge_response("") is None
    assert _parse_judge_response('{"other": "stuff"}') is None


def test_parse_salvage_fallback():
    # When we have is_choice_ask + question but no valid JSON object
    raw = 'blah "is_choice_ask": true, "question": "pick one", "options": ["opt1", "opt2"]'
    r = _parse_judge_response(raw)
    assert r is not None
    assert r.is_choice_ask is True
    assert r.question == "pick one"
    assert r.options == ["opt1", "opt2"]


# ─── verdict store / consume ────────────────────────────────────────────

def test_store_and_consume():
    reset_state("test_sid")
    v = Verdict(
        advisor_choice="CHOSEN: option A",
        question="A or B?",
        options=["A", "B"],
        ts=1234.5,
    )
    store_verdict("test_sid", v)
    got = consume_verdict("test_sid")
    assert got is not None
    assert got.advisor_choice == "CHOSEN: option A"
    assert got.question == "A or B?"
    assert got.options == ["A", "B"]
    assert got.ts == 1234.5


def test_consume_clears():
    reset_state("test_sid")
    store_verdict("test_sid", Verdict("x", "q", ["a"], 1.0))
    assert consume_verdict("test_sid") is not None
    assert consume_verdict("test_sid") is None  # consumed


def test_consume_missing_returns_none():
    reset_state("test_sid")
    assert consume_verdict("nonexistent") is None


# ─── inject_verdict_into_body ───────────────────────────────────────────

def test_inject_into_body_with_input():
    body = {"input": [{"type": "message", "role": "user", "content": "hi"}]}
    v = Verdict("pick A", "A or B?", ["A", "B"], 1.0)
    new_body, ok = inject_verdict_into_body(body, v)
    assert ok is True
    assert new_body is not body  # not mutated
    items = new_body["input"]
    assert len(items) == 2
    last = items[-1]
    assert last["role"] == "user"
    assert "pick A" in last["content"][0]["text"]


def test_inject_into_body_no_input():
    body = {"model": "gpt-5.5"}
    v = Verdict("pick A", "Q", ["A"], 1.0)
    new_body, ok = inject_verdict_into_body(body, v)
    assert ok is False
    assert new_body is body


def test_inject_preserves_other_fields():
    body = {
        "model": "test",
        "instructions": "keep me",
        "input": [{"type": "message", "role": "assistant", "content": "hello"}],
    }
    v = Verdict("CHOSEN: B", "A or B?", ["A", "B"], 1.0)
    new_body, ok = inject_verdict_into_body(body, v)
    assert ok is True
    assert new_body["model"] == "test"
    assert new_body["instructions"] == "keep me"
    assert len(new_body["input"]) == 2


# ─── reset_state ────────────────────────────────────────────────────────

def test_reset_state_specific():
    reset_state("sid_a")
    store_verdict("sid_a", Verdict("x", "q", ["a"], 1.0))
    store_verdict("sid_b", Verdict("y", "r", ["b"], 1.0))
    reset_state("sid_a")
    assert consume_verdict("sid_a") is None
    assert consume_verdict("sid_b") is not None
    reset_state("sid_b")  # cleanup


def test_reset_state_all():
    store_verdict("sid_a", Verdict("x", "q", ["a"], 1.0))
    store_verdict("sid_b", Verdict("y", "r", ["b"], 1.0))
    reset_state(None)  # wipe all
    assert consume_verdict("sid_a") is None
    assert consume_verdict("sid_b") is None
