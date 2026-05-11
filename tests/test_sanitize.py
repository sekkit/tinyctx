"""Tests for sanitize.rewrite_input_roles — the local responses-path role
normalizer that rewrites codex 0.128's `developer` role to `system` for
strict OpenAI-compat backends (older LMStudio builds, vLLM responses
adapters, etc.) that 400 on the newer role name.
"""
from __future__ import annotations

from copy import deepcopy

from tinyctx.sanitize import (
    _DEFAULT_ROLE_REWRITE_MAP,
    rewrite_input_roles,
)


def test_rewrites_developer_to_system():
    body = {
        "model": "qwen3.6-27b",
        "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "AGENTS.md ..."}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ],
    }
    snap = deepcopy(body)
    out, n = rewrite_input_roles(body, rewrite_map=_DEFAULT_ROLE_REWRITE_MAP)
    assert n == 1
    assert out["input"][0]["role"] == "system"
    assert out["input"][1]["role"] == "user"
    # input not mutated
    assert body == snap


def test_mixed_roles_only_developer_rewritten():
    body = {
        "input": [
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": "rules"},
            {"role": "assistant", "content": "ack"},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "developer", "content": "more rules"},
            {"role": "system", "content": "already system"},
        ],
    }
    out, n = rewrite_input_roles(body, rewrite_map=_DEFAULT_ROLE_REWRITE_MAP)
    assert n == 2
    roles = [it["role"] for it in out["input"]]
    assert roles == ["user", "system", "assistant", "tool", "system", "system"]


def test_empty_rewrite_map_is_noop():
    body = {"input": [{"role": "developer", "content": "x"}]}
    out, n = rewrite_input_roles(body, rewrite_map={})
    assert n == 0
    assert out is body  # exact same reference — no copy made


def test_none_rewrite_map_is_noop():
    body = {"input": [{"role": "developer", "content": "x"}]}
    out, n = rewrite_input_roles(body, rewrite_map=None)
    assert n == 0
    assert out is body


def test_missing_input_array_is_noop():
    """Chat-completions style body uses `messages` not `input`; the
    responses-path-only rewrite must leave such bodies untouched (the chat
    path is handled by normalize_for_chat which already maps developer)."""
    body = {"messages": [{"role": "developer", "content": "x"}]}
    out, n = rewrite_input_roles(body, rewrite_map=_DEFAULT_ROLE_REWRITE_MAP)
    assert n == 0
    assert out is body


def test_no_match_returns_original_without_copy():
    """Cheap pre-scan avoids deepcopy when no role matches."""
    body = {"input": [{"role": "user", "content": "x"},
                      {"role": "assistant", "content": "y"}]}
    out, n = rewrite_input_roles(body, rewrite_map=_DEFAULT_ROLE_REWRITE_MAP)
    assert n == 0
    assert out is body


def test_custom_rewrite_map():
    """User can override the default map via config.local_role_rewrite_map."""
    body = {
        "input": [
            {"role": "developer", "content": "a"},
            {"role": "function", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    }
    out, n = rewrite_input_roles(
        body, rewrite_map={"developer": "system", "function": "tool"})
    assert n == 2
    assert [it["role"] for it in out["input"]] == ["system", "tool", "user"]


def test_non_dict_items_are_skipped():
    body = {"input": [None, "string", {"role": "developer", "content": "x"}]}
    out, n = rewrite_input_roles(body, rewrite_map=_DEFAULT_ROLE_REWRITE_MAP)
    assert n == 1
    assert out["input"][2]["role"] == "system"
