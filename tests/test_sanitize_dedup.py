"""Tests for the DCP-inspired history-hygiene transforms in sanitize.py."""
from __future__ import annotations

import json
from copy import deepcopy

from tinyctx.sanitize import (
    _DEDUP_PLACEHOLDER,
    _FAILED_PURGE_PLACEHOLDER,
    dedup_tool_calls,
    purge_failed_tool_inputs,
    _tool_call_signature,
)


def test_signature_matches_identical_calls():
    a = {"type": "function_call", "name": "ls", "arguments": '{"path": "/tmp"}'}
    b = {"type": "function_call", "name": "ls", "arguments": '{"path": "/tmp"}'}
    c = {"type": "function_call", "name": "ls", "arguments": '{"path": "/etc"}'}
    assert _tool_call_signature(a) == _tool_call_signature(b)
    assert _tool_call_signature(a) != _tool_call_signature(c)


def test_signature_handles_dict_args_stably():
    """dict arguments should hash deterministically regardless of key order."""
    a = {"type": "function_call", "name": "f", "arguments": {"a": 1, "b": 2}}
    b = {"type": "function_call", "name": "f", "arguments": {"b": 2, "a": 1}}
    assert _tool_call_signature(a) == _tool_call_signature(b)


def test_dedup_replaces_earlier_duplicates_keeps_latest():
    body = {
        "input": [
            {"role": "user", "content": "list /tmp"},
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/tmp"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "<huge ls output 50KB>"},
            {"role": "assistant", "content": "found foo"},
            {"role": "user", "content": "list /tmp again"},
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/tmp"}', "call_id": "c2"},
            {"type": "function_call_output", "call_id": "c2",
             "output": "<another huge ls output>"},
        ]
    }
    out = dedup_tool_calls(body)
    items = out["input"]
    # First call's args replaced
    assert items[1]["arguments"] == _DEDUP_PLACEHOLDER
    # First call's output replaced
    assert items[2]["output"] == _DEDUP_PLACEHOLDER
    # Latest call left intact
    assert items[5]["arguments"] == '{"path": "/tmp"}'
    assert items[6]["output"] == "<another huge ls output>"
    # User messages untouched
    assert items[0]["content"] == "list /tmp"
    assert items[4]["content"] == "list /tmp again"


def test_dedup_does_not_touch_distinct_calls():
    body = {
        "input": [
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/tmp"}', "call_id": "c1"},
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/etc"}', "call_id": "c2"},
        ]
    }
    out = dedup_tool_calls(body)
    assert out["input"][0]["arguments"] == '{"path": "/tmp"}'
    assert out["input"][1]["arguments"] == '{"path": "/etc"}'


def test_dedup_does_not_mutate_input():
    body = {
        "input": [
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/tmp"}', "call_id": "c1"},
            {"type": "function_call", "name": "ls",
             "arguments": '{"path": "/tmp"}', "call_id": "c2"},
        ]
    }
    snapshot = deepcopy(body)
    _ = dedup_tool_calls(body)
    assert body == snapshot, "dedup_tool_calls must not mutate input"


def test_purge_failed_tool_inputs_after_threshold():
    body = {
        "input": [
            {"type": "function_call", "name": "apply_patch",
             "arguments": "<huge patch text>", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "Error: patch did not apply"},
            # 5 assistant turns follow
            {"role": "assistant", "content": "let me try again"},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "...."},
        ]
    }
    out = purge_failed_tool_inputs(body, after_turns=4)
    items = out["input"]
    assert items[0]["arguments"] == _FAILED_PURGE_PLACEHOLDER
    # error result remains visible
    assert "Error" in items[1]["output"]


def test_purge_keeps_recent_failed_input():
    """If only 1 turn has elapsed, keep the original input — agent might retry."""
    body = {
        "input": [
            {"type": "function_call", "name": "apply_patch",
             "arguments": "<huge patch text>", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "Error: patch did not apply"},
            {"role": "assistant", "content": "let me try"},
        ]
    }
    out = purge_failed_tool_inputs(body, after_turns=4)
    assert out["input"][0]["arguments"] == "<huge patch text>"


def test_purge_skips_when_no_errors():
    body = {
        "input": [
            {"type": "function_call", "name": "ls",
             "arguments": '{"path":"/tmp"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "ok"},
        ]
    }
    out = purge_failed_tool_inputs(body, after_turns=1)
    assert out["input"][0]["arguments"] == '{"path":"/tmp"}'


def test_cache_aware_mutator_first_turn_defers():
    """First request for a session shouldn't fire mutations — the cache
    just got populated, no point breaking it immediately."""
    from tinyctx.sanitize import CacheAwareMutator
    m = CacheAwareMutator(ttl_seconds=300, threshold=0.65)
    fire, reason = m.should_apply("sess-A", est_tokens=10_000,
                                  max_tokens=400_000, now=1000.0)
    assert fire is False
    assert "first_turn" in reason


def test_cache_aware_mutator_fires_on_threshold():
    from tinyctx.sanitize import CacheAwareMutator
    m = CacheAwareMutator(ttl_seconds=300, threshold=0.65)
    # seed first turn so we're past the first-turn-defer rule
    m.should_apply("sess-A", est_tokens=1, max_tokens=400_000, now=1000.0)
    fire, reason = m.should_apply("sess-A", est_tokens=300_000,
                                  max_tokens=400_000, now=1010.0)
    assert fire is True
    assert "context_usage" in reason


def test_cache_aware_mutator_fires_on_ttl():
    from tinyctx.sanitize import CacheAwareMutator
    m = CacheAwareMutator(ttl_seconds=300, threshold=0.65)
    m.should_apply("sess-A", est_tokens=1, max_tokens=400_000, now=1000.0)
    # 6 minutes later, low usage — TTL trigger fires.
    fire, reason = m.should_apply("sess-A", est_tokens=10_000,
                                  max_tokens=400_000, now=1000.0 + 360)
    assert fire is True
    assert "queue_age" in reason


def test_cache_aware_mutator_defers_within_ttl_and_under_threshold():
    from tinyctx.sanitize import CacheAwareMutator
    m = CacheAwareMutator(ttl_seconds=300, threshold=0.65)
    m.should_apply("sess-A", est_tokens=1, max_tokens=400_000, now=1000.0)
    # 30 seconds later, low usage — keep deferring.
    fire, reason = m.should_apply("sess-A", est_tokens=20_000,
                                  max_tokens=400_000, now=1030.0)
    assert fire is False
    assert "deferred" in reason


def test_cache_aware_mutator_mark_applied_resets_clock():
    from tinyctx.sanitize import CacheAwareMutator
    m = CacheAwareMutator(ttl_seconds=300, threshold=0.65)
    m.should_apply("sess-A", est_tokens=1, max_tokens=400_000, now=1000.0)
    m.mark_applied("sess-A", now=1500.0)
    fire, reason = m.should_apply("sess-A", est_tokens=10_000,
                                  max_tokens=400_000, now=1500.0 + 30)
    # Within TTL of the mark, low usage → defer.
    assert fire is False


def test_scrub_unsupported_tools_keeps_only_allowed():
    from tinyctx.sanitize import scrub_unsupported_tools
    body = {"tools": [
        {"type": "function", "name": "f1"},
        {"type": "web_search"},
        {"type": "image_generation"},
        {"type": "namespace", "name": "mcp__codex_apps__figma"},
        {"type": "function", "name": "f2"},
    ]}
    out = scrub_unsupported_tools(body, supported_types={"function"})
    types = [t["type"] for t in out["tools"]]
    assert types == ["function", "function"]
    # input untouched
    assert len(body["tools"]) == 5


def test_scrub_unsupported_tools_empty_set_means_pass_through():
    from tinyctx.sanitize import scrub_unsupported_tools
    body = {"tools": [{"type": "namespace"}, {"type": "function"}]}
    out = scrub_unsupported_tools(body, supported_types=())
    assert len(out["tools"]) == 2


def test_scrub_unsupported_tools_no_op_when_no_tools():
    from tinyctx.sanitize import scrub_unsupported_tools
    body = {"input": [{"role": "user", "content": "hi"}]}
    out = scrub_unsupported_tools(body)
    assert out == body


def test_flatten_tool_output_strips_input_image():
    """codex 0.128+ tool outputs are lists mixing text and base64 PNGs;
    DeepSeek's chat-completions API rejects `input_image` with HTTP 400.
    We must flatten to plain text + a placeholder for each image."""
    from tinyctx.sanitize import _flatten_tool_output
    out = _flatten_tool_output([
        {"type": "input_text", "text": "Build succeeded."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR..."},
        {"type": "output_text", "text": "exit 0"},
    ])
    assert "Build succeeded." in out
    assert "exit 0" in out
    assert "[image attached" in out
    assert "base64" not in out
    assert "iVBOR" not in out


def test_flatten_tool_output_passes_string_through():
    from tinyctx.sanitize import _flatten_tool_output
    assert _flatten_tool_output("plain stdout") == "plain stdout"
    assert _flatten_tool_output(None) == ""


def test_normalize_for_chat_handles_function_call_output_with_image():
    """End-to-end: a function_call_output whose `output` is a list of
    content items (text + image) must end up as a single tool message
    with a flat string content, no `input_image` leaking through."""
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "screenshot",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1",
             "output": [
                 {"type": "input_text", "text": "screenshot saved"},
                 {"type": "input_image",
                  "image_url": "data:image/png;base64,XXX"},
             ]},
        ],
    }
    out = normalize_for_chat(body)
    tool_msgs = [m for m in out["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert isinstance(content, str), \
        "function_call_output content must be a plain string for chat APIs"
    assert "input_image" not in content
    assert "screenshot saved" in content
    assert "[image attached" in content


def test_expand_mcp_namespaces_unwraps_advisor_namespace():
    """codex 0.128+ packages MCP function tools inside a `type=namespace`
    shell; tinyctx must unwrap them into top-level `type=function` entries
    so scrub_unsupported_tools doesn't drop them and the executor can see
    them. Names get the `mcp__<server>__<tool>` prefix codex uses for
    dispatch."""
    from tinyctx.sanitize import expand_mcp_namespaces
    body = {
        "tools": [
            {"type": "function", "name": "exec_command",
             "description": "x", "parameters": {"type": "object"}},
            {"type": "namespace", "name": "mcp__advisor__",
             "description": "Tools in mcp__advisor__.",
             "tools": [
                 {"type": "function", "name": "ask_advisor",
                  "description": "Consult frontier.",
                  "parameters": {"type": "object",
                                 "properties": {"question": {"type": "string"}},
                                 "required": ["question"]},
                  "strict": False},
             ]},
            {"type": "namespace", "name": "mcp__computer_use__",
             "description": "Computer use tools.",
             "tools": [
                 {"type": "function", "name": "click",
                  "description": "click ui.", "parameters": {"type": "object"}},
                 {"type": "function", "name": "drag",
                  "description": "drag ui.", "parameters": {"type": "object"}},
             ]},
        ],
    }
    out = expand_mcp_namespaces(body)
    fn_names = sorted(t["name"] for t in out["tools"]
                      if t.get("type") == "function")
    assert fn_names == ["exec_command",
                        "mcp__advisor__ask_advisor",
                        "mcp__computer_use__click",
                        "mcp__computer_use__drag"]
    # No namespace shells remain (they should be fully expanded).
    assert not any(t.get("type") == "namespace" for t in out["tools"])
    # Inner tool's parameters/description preserved on the rewritten entry.
    advisor = next(t for t in out["tools"]
                   if t["name"] == "mcp__advisor__ask_advisor")
    assert advisor["description"] == "Consult frontier."
    assert advisor["parameters"]["required"] == ["question"]
    assert advisor["strict"] is False
    # Original input is not mutated.
    assert body["tools"][1]["type"] == "namespace"


def test_expand_mcp_namespaces_no_op_without_namespace():
    from tinyctx.sanitize import expand_mcp_namespaces
    body = {"tools": [{"type": "function", "name": "f"}]}
    assert expand_mcp_namespaces(body) is body


def test_expand_then_scrub_keeps_advisor_drops_codex_specials():
    """Composition test: the real proxy pipeline runs expand then scrub.
    After expand, scrub should keep the advisor tool (now type=function)
    but still drop web_search/image_generation/the empty namespace."""
    from tinyctx.sanitize import expand_mcp_namespaces, scrub_unsupported_tools
    body = {
        "tools": [
            {"type": "function", "name": "exec_command"},
            {"type": "web_search", "external_web_access": False},
            {"type": "image_generation", "output_format": "png"},
            {"type": "namespace", "name": "mcp__advisor__",
             "tools": [{"type": "function", "name": "ask_advisor",
                        "description": "x", "parameters": {"type": "object"}}]},
        ],
    }
    expanded = expand_mcp_namespaces(body)
    scrubbed = scrub_unsupported_tools(expanded)  # default: function only
    names = sorted(t["name"] for t in scrubbed["tools"])
    assert names == ["exec_command", "mcp__advisor__ask_advisor"]




def test_inject_responses_defaults_sets_missing_dotted_path():
    from tinyctx.sanitize import inject_responses_defaults
    body = {"text": {"verbosity": "low"}, "model": "x"}
    out = inject_responses_defaults(body, {"text.format.type": "text"})
    assert out["text"]["format"]["type"] == "text"
    assert out["text"]["verbosity"] == "low"  # preserved
    # input untouched
    assert "format" not in body["text"]


def test_inject_responses_defaults_does_not_overwrite_existing():
    from tinyctx.sanitize import inject_responses_defaults
    body = {"text": {"format": {"type": "json_object"}}}
    out = inject_responses_defaults(body, {"text.format.type": "text"})
    assert out["text"]["format"]["type"] == "json_object"  # NOT overwritten


def test_inject_responses_defaults_empty_map_is_noop():
    from tinyctx.sanitize import inject_responses_defaults
    body = {"input": []}
    assert inject_responses_defaults(body, {}) == body


def test_inject_responses_defaults_creates_intermediate_dicts():
    from tinyctx.sanitize import inject_responses_defaults
    body = {"input": []}
    out = inject_responses_defaults(body, {"text.format.type": "text"})
    assert out["text"]["format"]["type"] == "text"


def test_strip_unsupported_responses_fields_drops_specified():
    from tinyctx.sanitize import strip_unsupported_responses_fields
    body = {
        "model": "x", "input": [], "tools": [],
        "client_metadata": {"a": 1},
        "prompt_cache_key": "abc123",
        "instructions": "stay",
    }
    out = strip_unsupported_responses_fields(body)
    assert "client_metadata" not in out
    assert "prompt_cache_key" not in out
    assert out["instructions"] == "stay"
    # input untouched
    assert "client_metadata" in body


def test_purge_handles_is_error_flag():
    body = {
        "input": [
            {"type": "function_call", "name": "f",
             "arguments": "x", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "200 OK", "is_error": True},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
    }
    out = purge_failed_tool_inputs(body, after_turns=4)
    assert out["input"][0]["arguments"] == _FAILED_PURGE_PLACEHOLDER


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
