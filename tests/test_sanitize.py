"""Tests for sanitize.rewrite_input_roles — the local responses-path role
normalizer that rewrites codex 0.128's `developer` role to `system` for
strict OpenAI-compat backends (older LMStudio builds, vLLM responses
adapters, etc.) that 400 on the newer role name.
"""
from __future__ import annotations

from copy import deepcopy

from tinyctx.sanitize import (
    _DEFAULT_ROLE_REWRITE_MAP,
    collect_failure_signals,
    detect_tool_call_storm,
    drop_orphan_tool_outputs,
    hoist_input_messages_to_instructions,
    rewrite_input_roles,
    strip_encrypted_content,
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


def test_hoists_developer_and_system_input_messages_into_instructions():
    body = {
        "instructions": "base guidance",
        "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "repo rules"}]},
            {"type": "message", "role": "system",
             "content": [{"type": "input_text", "text": "resume prior plan"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "resume"}]},
        ],
    }
    snap = deepcopy(body)
    out, n = hoist_input_messages_to_instructions(body)
    assert n == 2
    assert out["instructions"] == (
        "repo rules\n\nresume prior plan\n\nbase guidance"
    )
    assert [it["role"] for it in out["input"]] == ["user"]
    assert body == snap


def test_hoist_skips_non_message_items_and_empty_content():
    body = {
        "input": [
            {"type": "function_call", "role": "system",
             "name": "noop", "arguments": "{}"},
            {"type": "message", "role": "system", "content": []},
            {"type": "message", "role": "user", "content": "hello"},
        ],
    }
    out, n = hoist_input_messages_to_instructions(body)
    assert n == 0
    assert out is body


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


# --- strip_encrypted_content: reasoning content→summary fold (codex 0.128 wire fix) ---
#
# Reproduces the wire-shape rejection that aborted the Tetris/Battle-City runs:
# both LMStudio and chatgpt.com return HTTP 400
#   "Invalid 'input[0].content': array too long. Expected an array with
#    maximum length 0, but got an array with length 1 instead."
# when codex 0.128 ships reasoning items as
#   {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", ...}]}.
# The proxy now folds content.text into summary.text and drops content so
# both backends accept the body.


def test_strip_encrypted_folds_reasoning_content_into_summary():
    body = {
        "input": [
            {"type": "reasoning", "summary": [],
             "content": [{"type": "reasoning_text",
                          "text": "user wants tetris"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "go"}]},
        ],
    }
    out = strip_encrypted_content(body)
    reasoning = out["input"][0]
    assert "content" not in reasoning
    assert reasoning["summary"] == [
        {"type": "summary_text", "text": "user wants tetris"}]
    # id is NOT synthesized — synthetic IDs cause "item not found" 400
    # on subsequent turns when store=false.  The upstream auto-assigns ids.
    assert "id" not in reasoning
    # Non-reasoning items (and their content arrays) are untouched.
    assert out["input"][1]["content"] == [
        {"type": "input_text", "text": "go"}]


def test_strip_encrypted_preserves_existing_non_tinyctx_id_and_strips_synthetic():
    """Non-tinyctx IDs are preserved; rs_tinyctx_ IDs are stripped so the
    upstream treats items as new rather than failing a store lookup."""
    body = {
        "input": [
            {"id": "rs_real_id_42", "type": "reasoning",
             "summary": [{"type": "summary_text", "text": "x"}]},
            {"id": "rs_tinyctx_deadbeef", "type": "reasoning",
             "summary": [{"type": "summary_text", "text": "was synthetic"}]},
        ],
    }
    out = strip_encrypted_content(body)
    assert out["input"][0]["id"] == "rs_real_id_42"
    assert "id" not in out["input"][1]


def test_strip_encrypted_strips_rs_tinyctx_ids_from_all_item_types():
    """rs_tinyctx_ IDs are stripped from ANY item type, not just reasoning."""
    body = {
        "input": [
            {"id": "rs_tinyctx_abc123", "type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
            {"id": "rs_tinyctx_def456", "type": "function_call",
             "name": "f", "arguments": "{}"},
            {"id": "rs_tinyctx_ghi789", "type": "function_call_output",
             "call_id": "x", "output": "ok"},
        ],
    }
    out = strip_encrypted_content(body)
    for it in out["input"]:
        assert "id" not in it, f"{it.get('type')} should have no id"


def test_strip_encrypted_preserves_non_tinyctx_ids():
    """IDs without rs_tinyctx_ prefix are left alone."""
    body = {
        "input": [
            {"id": "call_abc", "type": "function_call",
             "name": "f", "arguments": "{}"},
            {"id": "resp_xyz", "type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "ok"}]},
        ],
    }
    out = strip_encrypted_content(body)
    assert out["input"][0]["id"] == "call_abc"
    assert out["input"][1]["id"] == "resp_xyz"


# --- detect_tool_call_storm threshold ---


def test_detect_tool_call_storm_rejects_below_threshold():
    """4 identical calls should NOT trigger storm (new threshold is 5+)."""
    body = {"input": [
        {"type": "function_call", "name": "exec_command",
         "arguments": '{"cmd":"echo hi"}'}
        for _ in range(4)
    ]}
    result = detect_tool_call_storm(body)
    assert result["triggered"] is False, f"4 calls should not trigger, got {result}"


def test_detect_tool_call_storm_triggers_above_threshold():
    """5 identical calls SHOULD trigger storm."""
    body = {"input": [
        {"type": "function_call", "name": "exec_command",
         "arguments": '{"cmd":"echo hi"}'}
        for _ in range(5)
    ]}
    result = detect_tool_call_storm(body)
    assert result["triggered"] is True
    assert result["tool_name"] == "exec_command"
    assert result["count"] == 5


def test_detect_tool_call_storm_different_args_dont_trigger():
    """Same tool name with different args is NOT a storm."""
    body = {"input": [
        {"type": "function_call", "name": "write",
         "arguments": '{"path":"a.txt"}'},
        {"type": "function_call", "name": "write",
         "arguments": '{"path":"b.txt"}'},
        {"type": "function_call", "name": "write",
         "arguments": '{"path":"c.txt"}'},
    ]}
    result = detect_tool_call_storm(body)
    assert result["triggered"] is False


def test_collect_failure_signals_ignores_below_threshold_storm():
    """2 identical calls → score should NOT include storm (threshold 5)."""
    body = {"input": [
        {"type": "function_call", "name": "exec_command",
         "arguments": '{"cmd":"echo hi"}'},
        {"type": "function_call", "name": "exec_command",
         "arguments": '{"cmd":"echo hi"}'},
    ]}
    result = collect_failure_signals(body)
    assert result["score"] == 0
    assert len(result["signals"]) == 0


def test_strip_encrypted_preserves_existing_summary_and_appends_folded():
    body = {
        "input": [
            {"type": "reasoning",
             "summary": [{"type": "summary_text", "text": "prior"}],
             "content": [{"type": "reasoning_text", "text": "new"}]},
        ],
    }
    out = strip_encrypted_content(body)
    assert "content" not in out["input"][0]
    assert out["input"][0]["summary"] == [
        {"type": "summary_text", "text": "prior"},
        {"type": "summary_text", "text": "new"},
    ]


def test_strip_encrypted_drops_empty_content_array_with_no_summary_change():
    body = {
        "input": [
            {"type": "reasoning", "summary": [], "content": []},
        ],
    }
    out = strip_encrypted_content(body)
    assert "content" not in out["input"][0]
    assert out["input"][0]["summary"] == []


def test_strip_encrypted_drops_unknown_content_types_without_folding():
    body = {
        "input": [
            {"type": "reasoning", "summary": [],
             "content": [{"type": "future_unknown_kind", "text": "hmm"}]},
        ],
    }
    out = strip_encrypted_content(body)
    assert "content" not in out["input"][0]
    # Unknown content shapes don't pollute summary — upstream would reject
    # them anyway. The drop keeps the body shape strictly spec-compliant.
    assert out["input"][0]["summary"] == []


def test_strip_encrypted_concatenates_multiple_reasoning_text_entries():
    body = {
        "input": [
            {"type": "reasoning", "summary": [],
             "content": [
                 {"type": "reasoning_text", "text": "part 1"},
                 {"type": "reasoning_text", "text": "part 2"},
             ]},
        ],
    }
    out = strip_encrypted_content(body)
    assert out["input"][0]["summary"] == [
        {"type": "summary_text", "text": "part 1"},
        {"type": "summary_text", "text": "part 2"},
    ]


def test_strip_encrypted_still_removes_encrypted_content():
    """The original behavior — strip encrypted_content — must keep working
    alongside the new content→summary fold."""
    body = {
        "input": [
            {"type": "reasoning",
             "encrypted_content": "ZW5jcnlwdGVk",
             "summary": [],
             "content": [{"type": "reasoning_text", "text": "thought"}]},
        ],
    }
    out = strip_encrypted_content(body)
    assert "encrypted_content" not in out["input"][0]
    assert "content" not in out["input"][0]
    assert out["input"][0]["summary"] == [
        {"type": "summary_text", "text": "thought"}]


def test_strip_encrypted_does_not_mutate_input_body():
    body = {
        "input": [
            {"type": "reasoning", "summary": [],
             "content": [{"type": "reasoning_text", "text": "x"}]},
        ],
    }
    snap = deepcopy(body)
    strip_encrypted_content(body)
    assert body == snap


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
