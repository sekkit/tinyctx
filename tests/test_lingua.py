"""Tests for tinyctx.lingua.

llmlingua may or may not be installed. The tests work in both modes:
when missing, compress_for_frontier is a no-op; when present, we still
mock the PromptCompressor to keep tests fast (the real model is ~hundreds
of MB and slow to load)."""
from __future__ import annotations

import json
from unittest import mock

import pytest

from tinyctx import lingua as ling


def test_is_available_falsy_when_missing(monkeypatch):
    """When llmlingua isn't installed the module reports unavailable
    rather than crashing."""
    import importlib
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "llmlingua" else real(name))
    assert ling.is_available() is False


def test_compress_for_frontier_noop_when_unavailable(monkeypatch):
    monkeypatch.setattr(ling, "is_available", lambda: False)
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1",
         "output": "x" * 5000},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["applied"] is False
    assert "llmlingua_not_installed" in info["skipped"]
    assert out["input"][0]["output"] == "x" * 5000  # unchanged


def test_compress_skips_below_min_bytes(monkeypatch):
    """Items shorter than MIN_BYTES bypass the compressor — overhead
    would dominate."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: object())
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": "tiny"},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["applied"] is False
    assert info["items_examined"] == 1
    assert info["items_compressed"] == 0


def test_compress_only_targets_tool_results(monkeypatch):
    """User messages, assistant messages, function_call items, and
    instructions must be left alone — they're either prompt-cache-
    critical or already minimal."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: object())
    monkeypatch.setattr(ling, "_compress_one",
                        lambda text, **kw: ("compressed", {
                            "compressed": True, "compressed_chars": 10,
                            "original_chars": len(text), "reason": "ok"}))
    long = "x" * 5000
    body = {
        "instructions": long,                          # must NOT be touched
        "input": [
            {"role": "user", "content": long},          # NOT touched
            {"type": "function_call", "call_id": "c1",
             "name": "shell", "arguments": long},      # NOT touched
            {"type": "function_call_output", "call_id": "c1",
             "output": long},                          # WILL be compressed
            {"role": "assistant", "content": long},     # NOT touched
        ],
        "tools": [{"type": "function", "name": "shell", "description": long}],
    }
    out, info = ling.compress_for_frontier(body)
    assert info["items_compressed"] == 1
    # Untouched fields verbatim
    assert out["instructions"] == long
    assert out["input"][0]["content"] == long  # user msg
    assert out["input"][1]["arguments"] == long  # function_call args
    assert out["input"][3]["content"] == long  # assistant msg
    assert out["tools"][0]["description"] == long
    # tool_result swapped
    assert out["input"][2]["output"] == "compressed"


def test_compress_preserves_list_form_output(monkeypatch):
    """codex 0.128 output as list of content items — re-encode in the
    same shape so the wire stays valid."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_compress_one",
                        lambda text, **kw: ("c", {
                            "compressed": True, "compressed_chars": 1,
                            "original_chars": len(text), "reason": "ok"}))
    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: object())
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "output_text", "text": "x" * 5000}
        ]},
    ]}
    out, _ = ling.compress_for_frontier(body)
    new_out = out["input"][0]["output"]
    assert isinstance(new_out, list), "must keep list-form"
    assert new_out[0]["type"] == "output_text"
    assert new_out[0]["text"] == "c"


def test_compress_keeps_original_when_saving_below_budget(monkeypatch):
    """When the compressor's output is nearly as big as the input,
    keep original — re-write doesn't compress, only churns the cache."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: object())
    src = "x" * 1000

    class _FakePC:
        def compress_prompt(self, *a, **kw):
            return {"compressed_prompt": "x" * 950}  # 95% of original

    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: _FakePC())
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": src},
    ]}
    out, info = ling.compress_for_frontier(body, ratio=0.5)
    assert info["items_compressed"] == 0
    assert "saving_below_budget" in info["skipped"]
    assert out["input"][0]["output"] == src


def test_compress_skips_below_floor_bytes(monkeypatch):
    """If compressor wants to drop below FLOOR_BYTES we keep original —
    too much information loss."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "FLOOR_BYTES", 500)
    src = "x" * 1000

    class _FakePC:
        def compress_prompt(self, *a, **kw):
            return {"compressed_prompt": "tiny"}

    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: _FakePC())
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": src},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["items_compressed"] == 0
    assert "below_floor_bytes" in info["skipped"]


def test_compress_runtime_error_falls_through(monkeypatch):
    """A crash inside PromptCompressor must not break the request —
    leave the body untouched."""
    monkeypatch.setattr(ling, "is_available", lambda: True)

    class _FakePC:
        def compress_prompt(self, *a, **kw):
            raise RuntimeError("model OOM")

    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: _FakePC())
    src = "x" * 5000
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": src},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["items_compressed"] == 0
    assert any("runtime_error" in r for r in info["skipped"])
    assert out["input"][0]["output"] == src


def test_compress_returns_byte_counts(monkeypatch):
    """Verify chars_before/chars_after wired correctly."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: object())
    monkeypatch.setattr(ling, "_compress_one",
                        lambda text, **kw: ("short", {
                            "compressed": True, "compressed_chars": 5,
                            "original_chars": len(text), "reason": "ok"}))
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": "x" * 5000},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["chars_before"] == 5000
    assert info["chars_after"] == 5
    assert info["applied"] is True


# ---------------------------------------------------- new behavioral tests


def test_lazy_import_does_not_crash_when_llmlingua_missing(monkeypatch):
    """`_get_compressor` returns None (no exception) when the llmlingua
    package isn't importable — exercises the lazy-import fallback path."""
    # Reset module-level cache so the import attempt actually happens.
    monkeypatch.setattr(ling, "_COMPRESSOR_CACHE", None)
    monkeypatch.setattr(ling, "_COMPRESSOR_INIT_FAILED", False)
    # Force the llmlingua import to fail by removing it from sys.modules
    # and shadowing it with None.
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "llmlingua", None)
    pc = ling._get_compressor()
    assert pc is None
    # And the module remembers the failure so subsequent calls short-circuit.
    assert ling._COMPRESSOR_INIT_FAILED is True


def test_compress_one_below_min_bytes_returns_text_unchanged():
    """Sub-MIN_BYTES inputs bypass compression entirely; `_compress_one`
    returns (text, info with reason='below_min_bytes')."""
    src = "short"
    out, info = ling._compress_one(src)
    assert out == src
    assert info["compressed"] is False
    assert info["reason"] == "below_min_bytes"
    assert info["original_chars"] == len(src)


def test_compress_one_compressor_unavailable_path(monkeypatch):
    """When `_get_compressor` returns None, a long input is returned
    untouched with reason='compressor_unavailable'."""
    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: None)
    src = "x" * (ling.MIN_BYTES + 100)
    out, info = ling._compress_one(src)
    assert out == src
    assert info["compressed"] is False
    assert info["reason"] == "compressor_unavailable"


def test_compress_one_clamps_ratio_to_valid_range(monkeypatch):
    """ratio is clamped to [0.1, 0.95] before being passed to the
    underlying PromptCompressor — extreme values must not raise."""
    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: _RecordingPC())
    src = "x" * (ling.MIN_BYTES + 500)
    # ratio=0.0 should be clamped to 0.1
    _out, info = ling._compress_one(src, ratio=0.0)
    assert _RecordingPC.last_rate == 0.1
    # ratio=1.0 should be clamped to 0.95
    _out2, info2 = ling._compress_one(src, ratio=1.0)
    assert _RecordingPC.last_rate == 0.95
    # default ratio passes through unchanged within range
    _out3, info3 = ling._compress_one(src, ratio=0.5)
    assert _RecordingPC.last_rate == 0.5


class _RecordingPC:
    """Records the rate kwarg the production code passes."""
    last_rate: float | None = None

    def compress_prompt(self, text, *, rate, **kw):
        type(self).last_rate = rate
        # Halve the input so compression "succeeds"
        return {"compressed_prompt": text[: max(1, len(text) // 2)]}


def test_compress_one_empty_compressor_result_falls_through(monkeypatch):
    """If the compressor returns an empty / non-string `compressed_prompt`,
    we keep the original."""
    class _EmptyPC:
        def compress_prompt(self, text, **kw):
            return {"compressed_prompt": ""}

    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: _EmptyPC())
    src = "x" * (ling.MIN_BYTES + 100)
    out, info = ling._compress_one(src)
    assert out == src
    assert info["compressed"] is False
    assert info["reason"] == "empty_result"


def test_compress_one_idempotent_on_already_short_output(monkeypatch):
    """A second compression pass on already-compressed text (now under
    MIN_BYTES) must not run the model again — it short-circuits on the
    size threshold. This validates the "idempotent on already-compressed"
    contract."""
    pass_count = {"n": 0}

    # Compress to ~half the input — large enough to clear FLOOR_BYTES on
    # the first pass, but second-pass output ends up below MIN_BYTES.
    class _HalfPC:
        def compress_prompt(self, text, **kw):
            pass_count["n"] += 1
            half = max(ling.FLOOR_BYTES + 50, len(text) // 2 - 10)
            return {"compressed_prompt": "y" * half}

    monkeypatch.setattr(ling, "_get_compressor",
                        lambda *a, **kw: _HalfPC())
    # Pick a size where input >= MIN_BYTES but half < MIN_BYTES.
    src = "x" * (ling.MIN_BYTES + 200)  # 1000 chars
    first, info1 = ling._compress_one(src)
    assert info1["compressed"] is True, f"first pass info: {info1}"
    assert pass_count["n"] == 1
    # Second pass: first output is now under MIN_BYTES → bypass entirely.
    assert len(first) < ling.MIN_BYTES
    second, info2 = ling._compress_one(first)
    assert second == first  # unchanged
    assert info2["reason"] == "below_min_bytes"
    assert pass_count["n"] == 1  # compressor was NOT invoked again


def test_flatten_to_text_string():
    """String payload returns (string, 'string')."""
    text, shape = ling._flatten_to_text("hello world")
    assert text == "hello world"
    assert shape == "string"


def test_flatten_to_text_list_of_text_items():
    """codex 0.128 list-of-content-items shape is concatenated by '\\n'."""
    payload = [
        {"type": "output_text", "text": "alpha"},
        {"type": "input_text", "text": "beta"},
        {"type": "image", "url": "ignored"},  # non-text type → skipped
        {"type": "text", "text": "gamma"},
    ]
    text, shape = ling._flatten_to_text(payload)
    assert shape == "list-text-items"
    assert text == "alpha\nbeta\ngamma"


def test_flatten_to_text_arbitrary_dict_falls_to_json():
    """Dict / other payloads serialize to JSON (shape='json')."""
    payload = {"foo": 1, "bar": [1, 2]}
    text, shape = ling._flatten_to_text(payload)
    assert shape == "json"
    parsed = json.loads(text)
    assert parsed == payload


def test_compress_for_frontier_uses_messages_container(monkeypatch):
    """When body has `messages` instead of `input`, the same compression
    logic applies (mirrors OpenAI chat.completions vs Responses API)."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: object())
    monkeypatch.setattr(ling, "_compress_one",
                        lambda text, **kw: ("zip", {
                            "compressed": True, "compressed_chars": 3,
                            "original_chars": len(text), "reason": "ok"}))
    body = {"messages": [
        {"type": "tool_result", "content": "y" * 5000},
        {"type": "function_call_output", "output": "y" * 5000},
    ]}
    out, info = ling.compress_for_frontier(body)
    assert info["items_compressed"] == 2
    assert out["messages"][0]["content"] == "zip"
    assert out["messages"][1]["output"] == "zip"


def test_compress_for_frontier_handles_missing_input_gracefully(monkeypatch):
    """If body has neither `input` nor `messages` (or a non-list value),
    the body returns unchanged with no compression."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    body = {"some": "other shape"}
    out, info = ling.compress_for_frontier(body)
    assert out == body
    assert info["items_compressed"] == 0
    assert info["items_examined"] == 0


def test_compress_for_frontier_does_not_mutate_input_body(monkeypatch):
    """The function operates on a deep copy — the caller's body must be
    untouched even when compression is applied."""
    monkeypatch.setattr(ling, "is_available", lambda: True)
    monkeypatch.setattr(ling, "_get_compressor", lambda *a, **kw: object())
    monkeypatch.setattr(ling, "_compress_one",
                        lambda text, **kw: ("c", {
                            "compressed": True, "compressed_chars": 1,
                            "original_chars": len(text), "reason": "ok"}))
    src = "a" * 5000
    original = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": src},
    ]}
    out, _ = ling.compress_for_frontier(original)
    # Caller's dict still holds the long original.
    assert original["input"][0]["output"] == src
    # Returned dict is the compressed one.
    assert out["input"][0]["output"] == "c"
    # And it's a distinct object.
    assert out is not original
    assert out["input"] is not original["input"]


def test_cli_status_prints_availability_and_defaults(capsys, monkeypatch):
    """`python -m tinyctx.lingua status` prints availability + defaults
    and returns rc=0 even when llmlingua isn't installed."""
    monkeypatch.setattr(ling, "is_available", lambda: False)
    rc = ling.main(["status"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "llmlingua importable" in captured.out
    assert "no" in captured.out
    assert "Install:" in captured.out
    assert ling.DEFAULT_MODEL in captured.out


def test_cli_warmup_without_llmlingua_returns_1(capsys, monkeypatch):
    """`warmup` with the dep missing should exit 1 with a clear stderr
    message, not crash."""
    monkeypatch.setattr(ling, "is_available", lambda: False)
    rc = ling.main(["warmup"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not installed" in captured.err


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
