"""Unit tests for tinyctx.conv_id.resolve_conv_key.

# Bug context
The proxy used `prompt_cache_key` alone for per-conversation state scoping.
OpenAI prompt-cache invalidation can regenerate that UUID mid-thread, which
silently reset the synthetic_continue counter and prevented the P2 budget
cap from firing. `resolve_conv_key` returns a fingerprint-based key that
stays stable across pck drift.

The new key shape is one of:
  - `<proj_sid>:fp:<8-hex>`  — when any structural signal is present
  - `<proj_sid>:<pck>`       — legacy fallback for older clients
  - `<proj_sid>`             — bare fallback for empty / compaction bodies
"""
from __future__ import annotations

import pytest

from tinyctx.conv_id import resolve_conv_key, _stable_fingerprint, _developer_block_text


def _body(*, pck="pck-1", install="i-1", model="gpt-5.5", dev="<perm>x</perm>"):
    return {
        "model": model,
        "prompt_cache_key": pck,
        "client_metadata": {"x-codex-installation-id": install},
        "input": [{"type": "message", "role": "developer",
                   "content": [{"type": "input_text", "text": dev}]}],
    }


def test_returns_proj_sid_for_non_dict_body():
    assert resolve_conv_key("p", None) == "p"  # type: ignore[arg-type]
    assert resolve_conv_key("p", "string") == "p"  # type: ignore[arg-type]


def test_returns_proj_sid_for_empty_dict():
    assert resolve_conv_key("p", {}) == "p"


def test_fingerprint_path_when_install_id_present():
    out = resolve_conv_key("proj", _body())
    assert out.startswith("proj:fp:")
    assert len(out.split(":")[-1]) == 8


def test_fingerprint_stable_across_pck_drift():
    a = resolve_conv_key("proj", _body(pck="A"))
    b = resolve_conv_key("proj", _body(pck="B"))
    c = resolve_conv_key("proj", _body(pck="C"))
    assert a == b == c


def test_fingerprint_changes_when_model_changes():
    a = resolve_conv_key("proj", _body(model="gpt-5.5"))
    b = resolve_conv_key("proj", _body(model="tinyctx-frontier"))
    assert a != b


def test_fingerprint_changes_when_install_changes():
    a = resolve_conv_key("proj", _body(install="a"))
    b = resolve_conv_key("proj", _body(install="b"))
    assert a != b


def test_fingerprint_changes_when_dev_block_changes():
    a = resolve_conv_key("proj", _body(dev="permA"))
    b = resolve_conv_key("proj", _body(dev="permB"))
    assert a != b


def test_legacy_pck_fallback_when_no_stable_signals():
    """No client_metadata, no model, no developer input → fall through to pck."""
    body = {"prompt_cache_key": "uuid-xyz"}
    assert resolve_conv_key("proj", body) == "proj:uuid-xyz"


def test_empty_pck_yields_bare_proj_sid():
    body = {"prompt_cache_key": ""}
    assert resolve_conv_key("proj", body) == "proj"


def test_developer_block_text_extraction_handles_string_content():
    body = {"input": [{"role": "developer", "content": "raw string content"}]}
    assert _developer_block_text(body) == "raw string content"


def test_developer_block_text_returns_empty_when_no_developer_role():
    body = {"input": [{"role": "user", "content": "hi"}]}
    assert _developer_block_text(body) == ""


def test_developer_block_text_only_scans_head_items():
    """Performance: don't walk the full input list — first 3 items is enough."""
    items = [{"role": "user", "content": "x"}] * 50
    items.append({"role": "developer", "content": [{"type": "input_text",
                                                     "text": "should-not-find"}]})
    assert _developer_block_text({"input": items}) == ""


def test_stable_fingerprint_returns_empty_when_no_signals():
    assert _stable_fingerprint({}) == ""
    assert _stable_fingerprint({"prompt_cache_key": "abc"}) == ""


def test_fingerprint_is_8_hex_chars():
    import re
    fp = _stable_fingerprint(_body())
    assert re.fullmatch(r"[0-9a-f]{8}", fp), f"unexpected fingerprint shape: {fp!r}"
