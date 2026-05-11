"""Tests for sanitize.rewrite_input_roles — the local responses-path role
normalizer that rewrites codex 0.128's `developer` role to `system` for
strict OpenAI-compat backends (older LMStudio builds, vLLM responses
adapters, etc.) that 400 on the newer role name.
"""
from __future__ import annotations

from copy import deepcopy

from tinyctx.sanitize import (
    _DEFAULT_ROLE_REWRITE_MAP,
    drop_orphan_tool_outputs,
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


# --- drop_orphan_tool_outputs ---

def test_drop_orphan_matched_call_and_output_unchanged():
    body = {
        "input": [
            {"type": "function_call", "call_id": "call_a", "name": "f",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_a",
             "output": "ok"},
        ],
    }
    snap = deepcopy(body)
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is False
    assert info["dropped"] == 0
    assert out["input"] == snap["input"]


def test_drop_orphan_drops_function_call_output_without_call():
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "call_orphan",
             "output": "stale"},
            {"type": "message", "role": "assistant", "content": "yo"},
        ],
    }
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is True
    assert info["dropped"] == 1
    assert info["call_ids"] == ["call_orphan"]
    types = [it.get("type") for it in out["input"]]
    assert "function_call_output" not in types


def test_drop_orphan_drops_tool_search_output_without_call():
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": "x"},
            {"type": "tool_search_output",
             "call_id": "call_MlwwII5Pdis9RismoaSO7eC0",
             "status": "completed", "tools": []},
            {"type": "message", "role": "user", "content": "go"},
        ],
    }
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is True
    assert info["dropped"] == 1
    assert info["call_ids"] == ["call_MlwwII5Pdis9RismoaSO7eC0"]
    assert all(it.get("type") != "tool_search_output" for it in out["input"])


def test_drop_orphan_keeps_tool_search_output_with_matching_call():
    body = {
        "input": [
            {"type": "tool_search_call", "call_id": "call_X",
             "arguments": "{}"},
            {"type": "tool_search_output", "call_id": "call_X",
             "status": "completed", "tools": []},
        ],
    }
    snap = deepcopy(body)
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is False
    assert out["input"] == snap["input"]


def test_drop_orphan_multiple_orphans_all_dropped():
    body = {
        "input": [
            {"type": "function_call_output", "call_id": "a", "output": "1"},
            {"type": "tool_result", "call_id": "b", "output": "2"},
            {"type": "tool_search_output", "call_id": "c",
             "status": "completed", "tools": []},
            {"type": "message", "role": "user", "content": "ok"},
        ],
    }
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is True
    assert info["dropped"] == 3
    assert set(info["call_ids"]) == {"a", "b", "c"}
    assert out["input"] == [{"type": "message", "role": "user", "content": "ok"}]


def test_drop_orphan_keeps_unpaired_call_without_output():
    # OpenAI semantics: a function_call without a matching output is benign
    # (e.g. the output is yet to come this turn). Must NOT be dropped.
    body = {
        "input": [
            {"type": "function_call", "call_id": "call_pending",
             "name": "f", "arguments": "{}"},
        ],
    }
    snap = deepcopy(body)
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is False
    assert out["input"] == snap["input"]


def test_drop_orphan_no_input_field_is_noop():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is False
    assert info["dropped"] == 0
    assert out == body


def test_drop_orphan_non_dict_items_do_not_crash():
    body = {
        "input": [
            None,
            "string",
            42,
            {"type": "function_call_output", "call_id": "z", "output": "x"},
        ],
    }
    out, info = drop_orphan_tool_outputs(body)
    assert info["applied"] is True
    assert info["dropped"] == 1
    assert out["input"] == [None, "string", 42]
