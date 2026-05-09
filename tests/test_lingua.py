"""Tests for tinyctx.lingua.

llmlingua may or may not be installed. The tests work in both modes:
when missing, compress_for_frontier is a no-op; when present, we still
mock the PromptCompressor to keep tests fast (the real model is ~hundreds
of MB and slow to load)."""
from __future__ import annotations

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
