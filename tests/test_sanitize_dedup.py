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


def test_inject_advisor_hint_appends_to_instructions():
    """inject_advisor_hint must append the advisor usage guidance to
    body.instructions so the executor model sees it on every turn (this
    is the workaround for AGENTS.md not reaching mid-thread requests)."""
    from tinyctx.sanitize import inject_advisor_hint
    body = {
        "model": "gpt-5.5",
        "instructions": "You are codex. Be helpful.",
        "input": [{"role": "user", "content": "hi"}],
    }
    out = inject_advisor_hint(body)
    assert out is not body, "must return a new body, not mutate input"
    assert out["instructions"].startswith("You are codex. Be helpful.")
    assert 'spawn_agent(role="advisor"' in out["instructions"]
    # Original body untouched
    assert "spawn_agent" not in body["instructions"]


def test_inject_advisor_hint_aligns_with_anthropic_advisor_strategy():
    """Anthropic's official Advisor Strategy doc lists FOUR concrete
    triggers + a treatment-of-advice protocol. tinyctx's hint must
    contain all of them verbatim (or near-verbatim) so the executor
    model gets the full benefit Anthropic's SWE-bench numbers
    measured. Source:
    platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool

    User directive: "需要完整对齐, 不可以省略".
    """
    from tinyctx.sanitize import _ADVISOR_HINT
    txt = _ADVISOR_HINT

    # 1. Conciseness rule (Anthropic claims 35-45% cost reduction).
    assert "under 100 words" in txt
    assert "enumerated steps" in txt
    assert "not explanations" in txt

    # 2. The four trigger rules.
    # (a) BEFORE substantive work
    assert "BEFORE substantive work" in txt
    assert "before writing" in txt
    assert "before committing to an interpretation" in txt
    assert "before building on an assumption" in txt
    # (b) Orientation vs substantive distinction
    assert "Orientation is not substantive work" in txt
    # (c) BEFORE declaring done
    assert "task is complete" in txt or "declaring done" in txt
    assert "make your deliverable durable" in txt
    # (d) When stuck
    assert "stuck" in txt.lower()
    assert "errors recurring" in txt
    assert "approach not converging" in txt
    # (e) When changing approach
    assert "change of approach" in txt

    # 3. Frequency guidance.
    assert "longer than a few steps" in txt
    assert "before committing to an approach and once before declaring done" in txt
    assert "first call" in txt  # adds most of its value on the first call
    assert "approach crystallizes" in txt

    # 4. Treatment-of-advice protocol.
    assert "give it serious weight" in txt.lower()
    assert "fails empirically" in txt
    assert "primary-source evidence" in txt
    assert "passing self-test is not evidence" in txt

    # 5. Reconcile-call pattern (data conflicts with advice).
    assert "don't silently switch" in txt.lower()
    assert "reconcile" in txt or "Surface the conflict" in txt
    assert "which constraint breaks the tie" in txt

    # 6. Mechanism-specific note: spawn_agent only forwards `task`.
    # tinyctx's reality differs from Anthropic's parameterless advisor()
    # tool; the hint must call this out so the executor packs context.
    assert "only forwards the `task` field" in txt
    assert "pack the task" in txt.lower() or "Pack the task" in txt


def test_inject_advisor_hint_includes_output_format_guidance():
    """The hint must also tell the model NOT to wrap final answers in
    `cat << EOF` heredoc — observed in real Codex.app session: model
    treated long final summaries as exec_command output, which Codex.app
    UI then folded into a collapsed tool-call block making the conclusion
    invisible to the user."""
    from tinyctx.sanitize import inject_advisor_hint
    body = {
        "model": "gpt-5.5",
        "instructions": "You are codex.",
        "input": [{"role": "user", "content": "hi"}],
    }
    out = inject_advisor_hint(body)
    inst = out["instructions"]
    # Must mention the anti-pattern by name
    assert "cat" in inst.lower() and "EOF" in inst
    assert "apply_patch" in inst  # tells model to use apply_patch for files
    assert "exec_command" in inst  # tells model what exec_command IS for
    # Must distinguish prose vs files vs side-effects
    assert "assistant message" in inst.lower()


def test_inject_advisor_hint_skips_advisor_sub_thread():
    """The advisor sub-thread itself uses model=tinyctx-frontier; injecting
    advisor guidance into its prompt would loop and waste budget. Skip."""
    from tinyctx.sanitize import inject_advisor_hint
    body = {
        "model": "tinyctx-frontier",
        "instructions": "You are an expert advisor...",
        "input": [{"role": "user", "content": "Q: ..."}],
    }
    out = inject_advisor_hint(body)
    assert out is body, "must return original body unchanged for advisor sub-thread"
    assert "spawn_agent(role=\"advisor\"" not in out["instructions"]


def test_inject_advisor_hint_idempotent():
    """If instructions already contain the advisor hint (e.g. AGENTS.md
    loaded it on a fresh thread), don't double-add."""
    from tinyctx.sanitize import inject_advisor_hint, _ADVISOR_HINT, _ADVISOR_HINT_MARKER
    body = {
        "model": "gpt-5.5",
        "instructions": "Codex base...\n\n" + _ADVISOR_HINT,
        "input": [{"role": "user", "content": "go"}],
    }
    out = inject_advisor_hint(body)
    # Should be unchanged (not appended again) — the original-instructions
    # marker count is preserved.
    expected = body["instructions"].count(_ADVISOR_HINT_MARKER)
    assert out["instructions"].count(_ADVISOR_HINT_MARKER) == expected
    assert out is body  # no copy when nothing to do


def test_inject_advisor_hint_disabled_via_env():
    import os as _os
    from tinyctx.sanitize import inject_advisor_hint
    saved = _os.environ.get("TINYCTX_INJECT_ADVISOR_HINT")
    _os.environ["TINYCTX_INJECT_ADVISOR_HINT"] = "0"
    try:
        body = {"model": "gpt-5.5", "instructions": "x", "input": []}
        out = inject_advisor_hint(body)
        assert out is body
        assert "spawn_agent" not in out["instructions"]
    finally:
        if saved is None:
            _os.environ.pop("TINYCTX_INJECT_ADVISOR_HINT", None)
        else:
            _os.environ["TINYCTX_INJECT_ADVISOR_HINT"] = saved


def test_inject_advisor_hint_skips_when_no_instructions():
    from tinyctx.sanitize import inject_advisor_hint
    body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "hi"}]}
    out = inject_advisor_hint(body)
    assert out is body  # nothing to append to


def test_normalize_for_chat_stubs_assistant_reasoning_content():
    """codex 0.128+ ships empty `type=reasoning` items (real thinking text
    is server-only), so we can't reconstruct reasoning_content. DeepSeek's
    thinking mode then 400s on the next turn unless every assistant
    message carries some `reasoning_content` field. Stubbing it to "" is
    the honest minimum — we forward nothing because we have nothing."""
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hello"}]},
        ],
    }
    out = normalize_for_chat(body)
    asst_msgs = [m for m in out["messages"] if m.get("role") == "assistant"]
    assert asst_msgs, "expected at least one assistant message"
    for m in asst_msgs:
        assert "reasoning_content" in m, \
            "every assistant message must carry reasoning_content (even if empty)"


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


def test_normalize_for_chat_translates_max_output_tokens_to_max_tokens():
    """When the body comes in with `max_output_tokens` (Responses API)
    and chat-completions normalization runs, `max_tokens` (chat API)
    must be set to the same value. This is critical for runaway-cap
    enforcement: tinyctx injects `max_output_tokens=16000` into local
    requests to prevent 80-second / 1.25 MB DeepSeek thinking loops
    that cause "peer closed connection" stream errors.

    User reported pattern of session interruption traced to this.
    """
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "max_output_tokens": 16000,
        "input": [{"type":"message","role":"user",
                   "content":[{"type":"input_text","text":"hi"}]}],
    }
    out = normalize_for_chat(body)
    assert out.get("max_tokens") == 16000, (
        f"max_output_tokens must translate to max_tokens for chat backends; "
        f"got out={out!r}"
    )


def test_normalize_for_chat_max_tokens_explicit_takes_precedence():
    """If the caller already set `max_tokens` explicitly, it wins over
    the `max_output_tokens` translation."""
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "max_tokens": 4000,
        "max_output_tokens": 16000,
        "input": [{"type":"message","role":"user","content":"hi"}],
    }
    out = normalize_for_chat(body)
    assert out.get("max_tokens") == 4000


def test_cap_responses_fields_lowers_excessive_values():
    """Force-cap mechanism. Unlike inject_responses_defaults, this MUST
    override an existing value when it exceeds the cap. Codex.app sends
    max_output_tokens=128000 by default; without this cap, runaway
    DeepSeek thinking loops produce 1.6 MB / 86s streams that the
    upstream cuts mid-flight, manifesting as session interruptions."""
    from tinyctx.sanitize import cap_responses_fields
    body = {"model": "x", "max_output_tokens": 128000}
    out = cap_responses_fields(body, {"max_output_tokens": 16000})
    assert out["max_output_tokens"] == 16000


def test_cap_responses_fields_leaves_lower_values_alone():
    """If the existing value is BELOW the cap, leave it. We want to
    cap excess, not raise floors."""
    from tinyctx.sanitize import cap_responses_fields
    body = {"max_output_tokens": 4000}
    out = cap_responses_fields(body, {"max_output_tokens": 16000})
    assert out["max_output_tokens"] == 4000


def test_cap_responses_fields_skips_missing_fields():
    """If the field isn't present, leave the body alone (cap doesn't
    inject; use inject_responses_defaults for that)."""
    from tinyctx.sanitize import cap_responses_fields
    body = {"model": "x"}
    out = cap_responses_fields(body, {"max_output_tokens": 16000})
    assert "max_output_tokens" not in out


def test_cap_responses_fields_does_not_mutate_input():
    """Defensive copy semantics like the rest of sanitize."""
    import json
    from tinyctx.sanitize import cap_responses_fields
    body = {"max_output_tokens": 200000}
    snap = json.dumps(body, sort_keys=True)
    cap_responses_fields(body, {"max_output_tokens": 16000})
    assert json.dumps(body, sort_keys=True) == snap


def test_local_backend_caps_max_output_tokens_in_default_config():
    """E2E: Config().local must set cap_fields with max_output_tokens
    so codex.app's default 128000 gets lowered. This is the production
    knob that prevents runaway-induced session interruptions."""
    from tinyctx.config import Config
    cfg = Config()
    assert "max_output_tokens" in cfg.local.cap_fields
    assert cfg.local.cap_fields["max_output_tokens"] >= 4000
    assert cfg.local.cap_fields["max_output_tokens"] <= 64000
    # Frontier intentionally has no cap (chatgpt backend rejects field)
    assert "max_output_tokens" not in cfg.frontier.cap_fields


def test_inject_responses_defaults_caps_runaway_output_for_local():
    """End-to-end: the default config's local.inject_defaults must
    contain `max_output_tokens` so runaway local-model output is
    capped. This is the production protection for the interruption
    pattern."""
    from tinyctx.config import Config
    cfg = Config()
    assert "max_output_tokens" in cfg.local.inject_defaults
    assert cfg.local.inject_defaults["max_output_tokens"] >= 4000
    assert cfg.local.inject_defaults["max_output_tokens"] <= 64000
    # frontier should NOT have this (codex's chatgpt backend rejects
    # max_output_tokens; we let it run with its own server-side limit)
    assert "max_output_tokens" not in cfg.frontier.inject_defaults


def test_normalize_for_chat_keeps_orphan_function_call_with_placeholder_result():
    """Regression: previously normalize_for_chat silently DROPPED any
    function_call whose call_id had no matching function_call_output in
    the same body. That hid history from the model — model would then
    "forget" tools it had called.

    New behavior: emit the function_call AND synthesize a placeholder
    tool result so chat-completions structure stays valid.
    """
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type":"input_text","text":"do thing"}]},
            # Orphan function_call: no matching function_call_output
            {"type": "function_call", "call_id": "c_orphan",
             "name": "shell", "arguments": '{"cmd":["ls"]}'},
        ],
    }
    out = normalize_for_chat(body)
    msgs = out["messages"]

    # The function_call must be emitted
    asst_with_tc = [m for m in msgs
                    if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(asst_with_tc) == 1, (
        f"function_call must be emitted, not silently dropped. messages={msgs}"
    )
    assert asst_with_tc[0]["tool_calls"][0]["id"] == "c_orphan"

    # And a placeholder tool result must follow it
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, (
        f"orphan call should get a synthesized placeholder result. messages={msgs}"
    )
    assert tool_msgs[0]["tool_call_id"] == "c_orphan"
    assert "tinyctx" in tool_msgs[0]["content"].lower()
    assert "not present" in tool_msgs[0]["content"]


def test_normalize_for_chat_widens_tool_result_types():
    """`output_ids` previously only scanned for `function_call_output`,
    silently dropping function_calls whose results were emitted as
    `tool_result` or `mcp_result` (legacy MCP / older codex paths).
    Now widened to all _TOOL_RESULT_TYPES."""
    from tinyctx.sanitize import normalize_for_chat
    body = {
        "model": "x",
        "input": [
            {"type": "function_call", "call_id": "cm1",
             "name": "mcp__server__tool", "arguments": "{}"},
            # MCP-style result (not function_call_output)
            {"type": "mcp_result", "call_id": "cm1",
             "output": "result from mcp server"},
        ],
    }
    out = normalize_for_chat(body)
    msgs = out["messages"]

    # Function call should be emitted (not dropped)
    asst_tc = [m for m in msgs
               if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(asst_tc) == 1
    # Tool result message should be emitted with the right cid
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "cm1"
    assert "result from mcp server" in tool_msgs[0]["content"]


def test_proactive_compact_cache_uses_bucket_keying_not_per_turn():
    """Regression: cache key was previously hash(middle_items), which
    changed every single turn (middle grows by one each turn). That
    caused a cache miss + summarizer call EVERY turn AND the model
    saw a slightly different summary each turn (information drift,
    "subtle forgetting").

    New design: bucket key. Same bucket → same cached summary across
    ~20 turns. Cache miss only on bucket flip.
    """
    from tinyctx.sanitize import (
        proactive_compact, clear_proactive_cache, _PROACTIVE_CACHE_BUCKET_SIZE,
    )
    clear_proactive_cache()

    summarizer_call_count = {"n": 0}

    def fake_summarizer(blob: str) -> str:
        summarizer_call_count["n"] += 1
        return f"summary-call-{summarizer_call_count['n']}"

    sid = "bucket-cache-test"
    # Build a body with enough items that proactive_compact applies and
    # `middle` falls within bucket 0 first, then crosses into bucket 1.
    def _body(n_items):
        items = [{"type":"message","role":"user",
                  "content":[{"type":"input_text","text":f"u{i}"}]}
                 for i in range(n_items)]
        return {"model":"x", "instructions":"y", "input": items}

    # First call: middle has ~30 items. bucket 0 (30//20=1, actually).
    # Since recent_keep=8, middle = first n-8 items.
    body1 = _body(40)   # 40-8 = 32 middle, bucket 32//20 = 1
    out1, info1 = proactive_compact(body1, session_id=sid,
                                    est_tokens=300_000, threshold_tokens=200_000,
                                    recent_keep=8, summarizer=fake_summarizer)
    assert info1["applied"]
    assert summarizer_call_count["n"] == 1

    # Second call: middle has 33 items (one more). Same bucket (33//20=1).
    # Should HIT cache, no new summarizer call.
    body2 = _body(41)
    out2, info2 = proactive_compact(body2, session_id=sid,
                                    est_tokens=300_000, threshold_tokens=200_000,
                                    recent_keep=8, summarizer=fake_summarizer)
    assert info2["applied"]
    assert info2["cached"], "same bucket should hit cache"
    assert summarizer_call_count["n"] == 1, (
        f"summarizer should NOT have run again for same bucket "
        f"(was called {summarizer_call_count['n']} times)"
    )

    # Third call: pad enough items to flip the bucket boundary.
    # Need middle to reach next bucket (40 items), so input total = 40+8 = 48.
    body3 = _body(48)
    out3, info3 = proactive_compact(body3, session_id=sid,
                                    est_tokens=300_000, threshold_tokens=200_000,
                                    recent_keep=8, summarizer=fake_summarizer)
    assert info3["applied"]
    assert not info3["cached"], "bucket flip should miss cache"
    assert summarizer_call_count["n"] == 2, (
        f"summarizer should have run once on bucket flip "
        f"(was called {summarizer_call_count['n']} times)"
    )


def test_proactive_compact_incremental_seeds_with_previous_bucket_summary():
    """When bucket flips, the new summary generation should be SEEDED
    with the previous bucket's summary, not start from scratch. This
    keeps continuity across bucket boundaries — the new summary is a
    refined version of the old one + new fall-off content."""
    from tinyctx.sanitize import proactive_compact, clear_proactive_cache
    clear_proactive_cache()

    captured_blobs: list[str] = []

    def capturing_summarizer(blob: str) -> str:
        captured_blobs.append(blob)
        return f"summary-v{len(captured_blobs)}"

    sid = "incremental-test"
    def _body(n):
        items = [{"type":"message","role":"user",
                  "content":[{"type":"input_text","text":f"item-{i}"}]}
                 for i in range(n)]
        return {"model":"x","instructions":"y","input": items}

    # Bucket 1
    proactive_compact(_body(40), session_id=sid, est_tokens=300_000,
                      threshold_tokens=200_000, recent_keep=8,
                      summarizer=capturing_summarizer)
    # Force bucket 2 (middle ≥ 40 items)
    proactive_compact(_body(48), session_id=sid, est_tokens=300_000,
                      threshold_tokens=200_000, recent_keep=8,
                      summarizer=capturing_summarizer)

    # Second blob should be SEEDED with first summary
    assert len(captured_blobs) == 2
    second = captured_blobs[1]
    assert "Previous handoff summary" in second
    assert "summary-v1" in second, (
        f"second summarizer call must include previous bucket's summary "
        f"as seed; got: {second[:300]}"
    )
    assert "Additional turns since that summary" in second


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
