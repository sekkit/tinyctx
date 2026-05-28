"""Cross-model sanitizer + history hygiene.

Reasoning items in the Responses API carry `encrypted_content` that's bound
to the originating model. When we route a turn to a different model than the
last assistant turn used, those payloads become undecryptable garbage and
codex crashes (openai/codex#17541). The simplest correct fix is to strip them.

In addition we offer two history-hygiene transforms inspired by the DCP
plugin (Opencode-DCP/opencode-dynamic-context-pruning, AGPL — patterns only,
no code copied):

  - dedup_tool_calls(body): when a tool is invoked multiple times with the
    *same* arguments earlier in the session, replace older occurrences with a
    short placeholder that points at the latest. Saves tokens while keeping
    every call locatable.

  - purge_failed_tool_inputs(body, after_turns=4): after a tool call returns
    an error and N more assistant turns have elapsed, replace the original
    input arguments with a placeholder. Keeps the error visible (so the agent
    knows the path was tried) but drops the bulky input payload.

Both transforms are byte-stable: the placeholder text is fixed, so cache
prefixes don't churn beyond the affected items themselves. Neither transform
modifies user messages.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


_REASONING_ITEM_TYPES = {"reasoning", "reasoning_summary", "thinking"}
_TOOL_CALL_TYPES = {"function_call", "tool_use", "mcp_call"}
_TOOL_RESULT_TYPES = {"function_call_output", "tool_result", "mcp_result"}

# Output-style items the Responses API requires to be paired with a prior
# matching call-style item carrying the same `call_id`. Superset of
# `_TOOL_RESULT_TYPES` — adds the Codex 0.128+ tool-search shape, which
# we don't synthesize stubs for (we just drop orphans). See
# `drop_orphan_tool_outputs` below.
_ORPHAN_PAIR_OUTPUT_TYPES = (
    _TOOL_RESULT_TYPES | {"tool_search_output"})
_ORPHAN_PAIR_CALL_TYPES = (
    _TOOL_CALL_TYPES | {"tool_search_call"})

_DEDUP_PLACEHOLDER = "[tinyctx: identical call deduped — see later turn]"
_FAILED_PURGE_PLACEHOLDER = "[tinyctx: failed input purged after N turns]"
_RESULT_SHRINK_MARKER = "[tinyctx: tool result summarized]"


def strip_encrypted_content(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `body` with `encrypted_content` removed from every
    reasoning-style item. Also folds codex 0.128's `content[reasoning_text]`
    payload back into the spec-compliant `summary[summary_text]` slot.

    Why the content→summary fold lives here, not as a separate function:
    both transforms walk the same reasoning items and both exist to keep
    cross-model requests upstream-compatible. The Responses API spec gives
    reasoning items a `summary` array; `content` on a reasoning item is
    rejected by strict validators (LMStudio and chatgpt.com both return
    `array_above_max_length` HTTP 400 when content has length > 0). The
    400 then escalates to frontier, which in turn fails — zero-byte
    response and codex aborts the session. See forensics dumps from
    2026-05-23 (rq_33b8a7e... and the upstream_400 entries).

    Cheap deepcopy - request bodies are small."""
    out = deepcopy(body)
    for key in ("input", "messages"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            # Strip proxy-synthesized IDs from ANY item type.
            # Reasoning items get `rs_tinyctx_<hash>` from
            # _fold_reasoning_content_into_summary.  When store=false
            # the upstream looks up every id-bearing item and 400s on
            # "not found."  Removing the id tells the upstream the item
            # is new (not a stored-item reference).
            existing_id = it.get("id")
            if isinstance(existing_id, str) and existing_id.startswith("rs_tinyctx_"):
                it.pop("id", None)
            t = it.get("type") or it.get("role") or ""
            if t in _REASONING_ITEM_TYPES:
                it.pop("encrypted_content", None)
                _fold_reasoning_content_into_summary(it)
            # nested content arrays
            content = it.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in _REASONING_ITEM_TYPES:
                        c.pop("encrypted_content", None)
    # `include` may explicitly request encrypted_content - drop that flag too.
    inc = out.get("include")
    if isinstance(inc, list):
        out["include"] = [x for x in inc if x != "reasoning.encrypted_content"]
    return out


def _fold_reasoning_content_into_summary(item: dict[str, Any]) -> None:
    """Move text from `item.content[reasoning_text]` into
    `item.summary[summary_text]`, drop the `content` field, and ensure
    the item carries an `id` (synthesizing one if missing).

    Mutates `item` in place. Idempotent: if `content` is missing/empty or
    every entry is empty after extraction, the field is just removed.

    Why the `id` synthesis: LMStudio (and any strict OpenAI-Responses
    validator) requires `id` on reasoning items in `input[]` —
    `ResponseReasoningItemParam` has it as a required field. Codex 0.128
    emits reasoning items WITHOUT `id`. After stripping `content` the
    item would still be rejected without an `id`. We synthesize a stable
    `rs_tinyctx_<8hex>` prefix so the upstream's pydantic validator
    accepts the union variant. The synthetic id doesn't need to match
    anything in the upstream's cache — it just satisfies the schema.

    The fold is conservative — it only extracts entries with
    `type == "reasoning_text"`. Other shapes (e.g. unknown future
    content types) are dropped silently because the upstream would reject
    them anyway. If `summary` already has entries, the folded text is
    appended after them so existing summary order is preserved."""
    content = item.get("content")
    folded_texts: list[str] = []
    if isinstance(content, list) and content:
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") != "reasoning_text":
                continue
            text = c.get("text")
            if isinstance(text, str) and text:
                folded_texts.append(text)
    item.pop("content", None)
    if folded_texts:
        summary = item.get("summary")
        if not isinstance(summary, list):
            summary = []
        for t in folded_texts:
            summary.append({"type": "summary_text", "text": t})
        item["summary"] = summary
    # Strip synthetic IDs the proxy added on a previous pass.
    # These IDs satisfy the schema validator but also cause the upstream
    # to treat the item as a reference to a stored item.  When store=false
    # the item lookup fails with 400 "Item not found."  Removing the id
    # lets the upstream (re-)create the item as new.
    existing_id = item.get("id")
    if isinstance(existing_id, str) and existing_id.startswith("rs_tinyctx_"):
        item.pop("id", None)


def rewrite_model(body: dict[str, Any], target_model: str) -> dict[str, Any]:
    body["model"] = target_model
    return body


# Default rewrite map for `body.input[*].role` values that older / community
# OpenAI-compat backends (LMStudio < 0.4.x, vLLM responses adapter, llama.cpp
# OAI server) reject with HTTP 400 "Unexpected message role.". codex 0.128+
# emits `role="developer"` for system-level instructions (AGENTS.md etc.);
# `system` is the universally-accepted equivalent.
_DEFAULT_ROLE_REWRITE_MAP: dict[str, str] = {"developer": "system"}


def hoist_input_messages_to_instructions(
    body: dict[str, Any],
    *,
    roles: tuple[str, ...] = ("developer", "system"),
) -> tuple[dict[str, Any], int]:
    """Move `body.input[*]` message items with roles in `roles` into the
    top-level `instructions` string.

    Some OpenAI-compatible `/v1/responses` backends accept system guidance only
    via `instructions` and reject `input` items whose role is `system`. Hoisting
    developer/system guidance preserves semantics better than rewriting those
    items to `user`.
    """
    items = body.get("input")
    if not isinstance(items, list) or not items:
        return body, 0

    extracted: list[str] = []
    keep: list[Any] = []
    for it in items:
        if not isinstance(it, dict):
            keep.append(it)
            continue
        if it.get("role") not in roles:
            keep.append(it)
            continue
        if it.get("type") not in (None, "message"):
            keep.append(it)
            continue
        text = _input_message_text(it.get("content"))
        if not text.strip():
            keep.append(it)
            continue
        extracted.append(text.strip())

    if not extracted:
        return body, 0

    out = deepcopy(body)
    out["input"] = keep
    inst = out.get("instructions")
    hoisted = "\n\n".join(part for part in extracted if part).strip()
    if isinstance(inst, str) and inst.strip():
        out["instructions"] = hoisted + "\n\n" + inst
    else:
        out["instructions"] = hoisted
    return out, len(extracted)


def rewrite_input_roles(
    body: dict[str, Any],
    *,
    rewrite_map: dict[str, str] | None = None,
) -> tuple[dict[str, Any], int]:
    """Walk `body.input` (Responses API) and rewrite items whose `role` is a
    key in `rewrite_map`. Returns (new_body, rewritten_count). When nothing
    matches, returns the original body and 0 (no copy made).

    Only touches the `input` array (Responses API). `messages` is the chat-
    completions shape and `normalize_for_chat` already handles
    `developer -> system` there.
    """
    if not rewrite_map:
        return body, 0
    items = body.get("input")
    if not isinstance(items, list) or not items:
        return body, 0
    # Cheap pre-scan to avoid deepcopy when there's nothing to rewrite.
    if not any(isinstance(it, dict) and it.get("role") in rewrite_map
               for it in items):
        return body, 0
    out = deepcopy(body)
    n = 0
    for it in out["input"]:
        if not isinstance(it, dict):
            continue
        r = it.get("role")
        if r in rewrite_map:
            it["role"] = rewrite_map[r]
            n += 1
    return out, n


def compact_input_messages_for_local_responses(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Reduce `input` message items to the most conservative shape for strict
    local `/responses` backends.

    Strategy:
      - hoist all non-user message items into `instructions`
      - merge user message items into a single trailing user message
      - keep non-message structural items (function_call, function_call_output,
        reasoning, etc.) unchanged and in order
    """
    items = body.get("input")
    if not isinstance(items, list) or not items:
        return body, {"hoisted": 0, "merged_user_messages": 0}

    instructions_parts: list[str] = []
    user_parts: list[str] = []
    kept: list[Any] = []
    hoisted = 0
    merged_user_messages = 0

    for it in items:
        if not isinstance(it, dict):
            kept.append(it)
            continue
        if it.get("type") not in (None, "message"):
            kept.append(it)
            continue
        role = it.get("role")
        text = _input_message_text(it.get("content")).strip()
        if not text:
            continue
        if role == "user":
            user_parts.append(text)
            merged_user_messages += 1
            continue
        instructions_parts.append(text)
        hoisted += 1

    if hoisted == 0 and merged_user_messages <= 1:
        return body, {"hoisted": 0, "merged_user_messages": merged_user_messages}

    out = deepcopy(body)
    new_input = [it for it in kept]
    if user_parts:
        new_input.append({
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "\n\n".join(part for part in user_parts if part),
            }],
        })
    out["input"] = new_input

    inst = out.get("instructions")
    hoisted_text = "\n\n".join(part for part in instructions_parts if part).strip()
    if hoisted_text:
        if isinstance(inst, str) and inst.strip():
            out["instructions"] = hoisted_text + "\n\n" + inst
        else:
            out["instructions"] = hoisted_text
    return out, {"hoisted": hoisted, "merged_user_messages": merged_user_messages}


def embed_instructions_into_user_message(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Move top-level `instructions` into the first user message and remove
    the top-level field.

    Useful for strict local `/responses` backends that reject any system-style
    instruction channel but still accept plain user text.
    """
    instructions = body.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        return body, False
    items = body.get("input")
    if not isinstance(items, list):
        return body, False

    out = deepcopy(body)
    out.pop("instructions", None)
    first_user_idx: int | None = None
    for idx, it in enumerate(out["input"]):
        if isinstance(it, dict) and it.get("type") in (None, "message") and it.get("role") == "user":
            first_user_idx = idx
            break

    if first_user_idx is None:
        out["input"].insert(0, {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": instructions.strip(),
            }],
        })
        return out, True

    msg = out["input"][first_user_idx]
    text = _input_message_text(msg.get("content")).strip()
    merged = instructions.strip() if not text else instructions.strip() + "\n\n" + text
    msg["content"] = [{
        "type": "input_text",
        "text": merged,
    }]
    return out, True


def condense_instructions_for_local_responses(
    body: dict[str, Any],
    *,
    max_chars: int = 1200,
) -> tuple[dict[str, Any], bool]:
    """Replace a very large `instructions` block with a compact local-safe
    summary while preserving the most important guardrails.

    This is specifically for strict local `/responses` backends that require
    `instructions` to exist but degrade or reject overly complex prompt
    structure.
    """
    instructions = body.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        return body, False
    if len(instructions) <= max_chars:
        return body, False

    parts: list[str] = []
    lowered = instructions.lower()
    if "danger-full-access" in lowered:
        parts.append("Execution environment: danger-full-access filesystem; network enabled.")
    if "approval policy is currently never" in lowered:
        parts.append("Approval mode: never ask for approval; do not request escalations.")
    if "default mode" in lowered or "collaboration mode: default" in lowered:
        parts.append("Collaboration mode: default; make reasonable assumptions and execute directly.")
    if "update_plan" in instructions:
        parts.append("Use update_plan to track multi-step work and mark steps complete.")
    if "context-mode" in lowered:
        parts.append("If context-mode tools are available, prefer them for large searches and processing.")
    parts.append("Respect higher-priority system/developer/project instructions. Keep outputs concise and actionable.")
    condensed = "\n".join(parts)
    budget = max(0, max_chars - len(condensed) - 120)
    if budget:
        head_budget = max(0, budget // 2)
        tail_budget = max(0, budget - head_budget)
        head = instructions[:head_budget].strip()
        tail = instructions[-tail_budget:].strip() if tail_budget else ""
        snippets = []
        if head:
            snippets.append("[trimmed original instructions: head]\n" + head)
        if tail and tail != head:
            snippets.append("[trimmed original instructions: tail]\n" + tail)
        if snippets:
            condensed = condensed + "\n\n" + "\n\n".join(snippets)
    out = deepcopy(body)
    out["instructions"] = condensed[:max_chars]
    return out, True


def _input_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


# Codex's Responses-API requests carry tool entries with codex-specific
# `type` values that strict schemas (LMStudio's, vLLM's older versions, etc.)
# reject with HTTP 400. The universal type every OpenAI-compat backend
# accepts is "function". We default to keeping only that; users with a
# more permissive backend can override via [local].supported_tool_types.
_DEFAULT_SUPPORTED_TOOL_TYPES = {"function"}

# Codex sends fields like client_metadata, prompt_cache_key, parallel_tool_calls
# that some backends will accept but others reject. We drop the known-rejected
# ones by default; everything else passes through.
_DEFAULT_STRIP_FIELDS = ("client_metadata", "prompt_cache_key")


# ── advisor hint injection ──────────────────────────────────────────────
# When a thread can't be restarted (so updated AGENTS.md never reaches
# its base instructions), the executor never learns about the advisor
# sub-agent and therefore never spawns it. The proxy injects this hint
# into `instructions` for every main-thread request, so the executor
# sees it on every turn even mid-thread. Idempotent — won't double-add.
_ADVISOR_HINT = """

## advisor sub-agent — Anthropic Advisor Strategy (full alignment)

The advisor should respond in under 100 words and use enumerated steps, not explanations.

You have a `spawn_agent(role="advisor", task="...")` sub-agent backed by a stronger reviewer model (gpt-5.5 / Opus-class). NOTE the spawn_agent mechanism only forwards the `task` field — the advisor does NOT automatically see your full conversation. Pack the task with the concrete question + the context the advisor needs to weigh in + what you're currently leaning toward.

Call advisor BEFORE substantive work — before writing, before committing to an interpretation, before building on an assumption. If the task requires orientation first (finding files, fetching a source, seeing what's there), do that, then call advisor. Orientation is not substantive work. Writing, editing, and declaring an answer are.

Also call advisor:
- When you believe the task is complete. BEFORE this call, make your deliverable durable: write the file, save the result, commit the change. The advisor call takes time; if the session ends during it, a durable result persists and an unwritten one doesn't.
- When stuck — errors recurring, approach not converging, results that don't fit.
- When considering a change of approach.

On tasks longer than a few steps, call advisor at least once before committing to an approach and once before declaring done. On short reactive tasks where the next action is dictated by tool output you just read, you don't need to keep calling — the advisor adds most of its value on the first call, before the approach crystallizes.

How to treat the advice: give it serious weight. If you follow a step and it fails empirically, or you have primary-source evidence that contradicts a specific claim (the file says X, the paper states Y), adapt. A passing self-test is not evidence the advice is wrong — it's evidence your test doesn't check what the advice is checking.

If you've already retrieved data pointing one way and the advisor points another: don't silently switch. Surface the conflict in one more advisor call — "I found X, you suggest Y, which constraint breaks the tie?" The advisor saw your evidence in the task you sent and may have underweighted it; a reconcile call is cheaper than committing to the wrong branch.

Pattern:
  spawn_agent(role="advisor", task=<concrete question + tight context + what you're leaning toward>)
  wait_agent(<the agent id>)
  # Then act on the advice, applying the rules above

Each call costs ~5-10K frontier tokens. Budget ~2-3 advisor calls per task.

## Final-answer output format

When you give the user the final summary / conclusion / report at the end of a task, write it as a **plain assistant message** (markdown is fine). Do **not** wrap it in `exec_command` with `cat << 'EOF' ... EOF` heredoc to simulate a "long output." That pattern makes the UI fold the message into a collapsed tool-call block so the user can't read the conclusion without expanding it.

If you genuinely need to write content to a file, use `apply_patch` (creates the file directly) — never `cat heredoc | tee` and never `cat << EOF > file`. Reserve `exec_command` for shell side-effects (`gradle build`, `pytest`, file inspection), not for prose output.

In short: **prose answers go in the assistant message, files go through apply_patch, shell side-effects go through exec_command.**

## When you need the user to choose between options

If you find yourself wanting to write something like:

```
请选择：
A → IMU 自实现 6DoF
B → RemoteLoader 初始化修复
```

…stop. Don't put a multiple-choice prompt as plain text in the assistant message. The user has the `request_user_input` tool wired up; **call that tool** with the question and options, and the system will route the choice (either to the user's UI or to an automated frontier advisor, depending on configuration). Plain-text "请选择 A or B" gets the session stuck waiting on a human; the tool gets it routed and answered cleanly.
"""

# Marker the injector looks for to avoid double-injection.
_ADVISOR_HINT_MARKER = 'spawn_agent(role="advisor"'
_CTX_TOOL_UNAVAILABLE_MARKER = "<!-- tinyctx-context-mode-unavailable -->"
_CTX_TOOL_UNAVAILABLE_NOTE = f"""

{_CTX_TOOL_UNAVAILABLE_MARKER}
## Session capability note

This request does not expose `context-mode` MCP tools to the model.
Do NOT call `ctx_batch_execute`, `ctx_search`, `ctx_execute`,
`ctx_execute_file`, or any `mcp__context_*` / `mcp__context-mode__*`
tool names unless they are explicitly present in the tool list for this
request.

If those tools are unavailable, continue with the tools that are
actually available in the current session instead of retrying the same
unsupported call.
"""


def inject_advisor_hint(body: dict[str, Any]) -> dict[str, Any]:
    """Append the advisor sub-agent usage hint to `body["instructions"]`.

    Skips:
      - the advisor sub-thread itself (model == "tinyctx-frontier") so we
        don't recursively prompt the advisor to spawn another advisor;
      - bodies that already contain the hint (e.g. AGENTS.md was loaded);
      - bodies where instructions is missing or non-string.

    Disabled with `TINYCTX_INJECT_ADVISOR_HINT=0`.

    Returns a new body when injecting, the original body otherwise."""
    if os.environ.get("TINYCTX_INJECT_ADVISOR_HINT", "1") == "0":
        return body
    model = (body.get("model") or "").lower()
    if model == "tinyctx-frontier":
        # the advisor itself; injecting would loop / waste prompt budget
        return body
    inst = body.get("instructions")
    if not isinstance(inst, str) or not inst:
        return body
    if _ADVISOR_HINT_MARKER in inst:
        return body
    out = deepcopy(body)
    out["instructions"] = inst + _ADVISOR_HINT
    return out


def inject_context_mode_unavailable_note(body: dict[str, Any]) -> dict[str, Any]:
    """Append a capability note when `context-mode` tools are absent.

    Repo-level instructions may strongly prefer `ctx_*` helpers. When the
    active Codex client did not expose those MCP tools for this turn, the
    model can otherwise loop on unsupported calls.
    """
    inst = body.get("instructions")
    if not isinstance(inst, str) or not inst:
        return body
    if _CTX_TOOL_UNAVAILABLE_MARKER in inst:
        return body
    tools = body.get("tools")
    if not isinstance(tools, list):
        tools = []
    names = [
        str(tool.get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    ]
    has_context_mode_tool = any(
        name.startswith("ctx_")
        or "context_mode" in name
        or "context-mode" in name
        for name in names
    )
    if has_context_mode_tool:
        return body
    out = deepcopy(body)
    out["instructions"] = _CTX_TOOL_UNAVAILABLE_NOTE.strip() + "\n\n" + inst
    return out


def expand_mcp_namespaces(body: dict[str, Any], *,
                          prefix_inner: bool = True,
                          no_prefix_namespaces: set[str] | None = None) -> dict[str, Any]:
    """Codex 0.128+ wraps MCP-server tools in a `type: "namespace"` shell:

        {"type": "namespace", "name": "mcp__advisor__",
         "description": "...",
         "tools": [
             {"type": "function", "name": "ask_advisor",
              "description": "...", "parameters": {...}, "strict": false},
             ...
         ]}

    Without expansion the namespace shell is dropped by `scrub_unsupported_tools`
    (chat-completions backends only accept `type=function`), so MCP tools like
    `ask_advisor` never reach the executor — the Advisor Strategy never fires.

    This rewrites every namespace into N function entries, prefixing each
    inner tool's name with the namespace name so codex's tool dispatcher
    can still route the call back to the right MCP server. Codex itself
    documents the convention `mcp__<server>__<tool>` in its base prompt, so
    the round-trip remains correct.

    Returns a new body. No-op when `tools` is absent or has no namespaces.
    """
    if "tools" not in body or not isinstance(body["tools"], list):
        return body
    if not any(isinstance(t, dict) and t.get("type") == "namespace"
               for t in body["tools"]):
        return body
    out = deepcopy(body)
    new_tools: list[dict[str, Any]] = []
    raw_no_prefix = {str(item) for item in (no_prefix_namespaces or set())}
    for t in out["tools"]:
        if not isinstance(t, dict):
            new_tools.append(t)
            continue
        if t.get("type") != "namespace":
            new_tools.append(t)
            continue
        ns_name = t.get("name") or ""
        for inner in t.get("tools") or []:
            if not isinstance(inner, dict):
                continue
            if inner.get("type") != "function":
                # Don't try to handle nested non-function tools — they'd
                # need their own translation. Leave unchanged for now;
                # they'll be dropped downstream by scrub if not whitelisted.
                new_tools.append(inner)
                continue
            inner_name = inner.get("name") or ""
            # codex uses "mcp__<server>__<tool>"; namespace name already
            # ends with "__" so concatenation just works. If a future
            # codex version drops the trailing "__", we add it explicitly.
            if prefix_inner and _should_prefix_namespace_tool(
                    ns_name, inner_name, raw_no_prefix):
                if ns_name and not ns_name.endswith("__"):
                    ns_name = ns_name + "__"
                qualified = (ns_name + inner_name) if ns_name else inner_name
            else:
                # Some codex builds expect the inner tool NAME unchanged
                # at the wire level — they reverse-lookup namespace from
                # an internal table instead of parsing the prefix. Toggle
                # via TINYCTX_MCP_NAME_NO_PREFIX=1 (read by the proxy).
                qualified = inner_name
            rewritten = dict(inner)
            rewritten["name"] = qualified
            new_tools.append(rewritten)
    out["tools"] = new_tools
    return out


def _should_prefix_namespace_tool(
    namespace_name: str,
    inner_name: str,
    no_prefix_namespaces: set[str],
) -> bool:
    if namespace_name in no_prefix_namespaces:
        return False
    normalized = namespace_name.replace("-", "_").lower()
    if normalized in {
        "mcp__context_mode__",
        "mcp__plugin_context_mode_context_mode__",
    } and inner_name.startswith("ctx_"):
        return False
    return True


def scrub_unsupported_tools(
    body: dict[str, Any],
    *,
    supported_types: set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Filter `body['tools']` to keep only entries whose `type` is in
    `supported_types`. An empty set/tuple means "keep all" (no filtering).
    No-op when `tools` is absent.

    Returns a new body (does not mutate input)."""
    if "tools" not in body or not isinstance(body["tools"], list):
        return body
    if supported_types is None:
        keep = _DEFAULT_SUPPORTED_TOOL_TYPES
    elif not supported_types:
        return body  # empty = pass through
    else:
        keep = set(supported_types)
    out = deepcopy(body)
    out["tools"] = [t for t in out["tools"]
                    if isinstance(t, dict) and t.get("type") in keep]
    return out


def strip_unsupported_responses_fields(
    body: dict[str, Any],
    *,
    drop: tuple[str, ...] = _DEFAULT_STRIP_FIELDS,
) -> dict[str, Any]:
    """Drop top-level fields a strict OpenAI-compat backend may reject.
    Returns a new body (does not mutate input)."""
    out = deepcopy(body)
    for k in drop:
        out.pop(k, None)
    return out


def inject_responses_defaults(
    body: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Inject default values for fields that codex omits but a strict
    backend (e.g. LMStudio) requires. `defaults` is a dotted-path map; each
    leaf is set ONLY if missing — never overwrites user-provided values.

    Example:
        inject_responses_defaults(body, {"text.format.type": "text"})
    ensures `body["text"]["format"]["type"] == "text"` when not already set.
    """
    if not defaults:
        return body
    out = deepcopy(body)
    for path, value in defaults.items():
        keys = path.split(".")
        cur = out
        for k in keys[:-1]:
            v = cur.get(k)
            if not isinstance(v, dict):
                cur[k] = {}
            cur = cur[k]
        leaf = keys[-1]
        if leaf not in cur:
            cur[leaf] = value
    return out


def cap_responses_fields(
    body: dict[str, Any],
    caps: dict[str, int],
) -> dict[str, Any]:
    """Force-cap numeric fields: set body[path] = min(existing, cap).
    Unlike `inject_responses_defaults`, this OVERRIDES existing values
    when they exceed the cap — necessary for hard limits like
    max_output_tokens that callers set higher than is safe.

    Real bug found while diagnosing 1.6 MB / 86s DeepSeek runaway today
    (21:11): codex.app sends `max_output_tokens=128000` in its Responses
    requests. The `inject_responses_defaults({"max_output_tokens":16000})`
    we added in commit 5715d40 was a no-op because the field was always
    already present from codex. With this helper we cap from above,
    landing the actually-effective max at min(128000, 16000) = 16000.

    `caps` is a dotted-path → integer cap map. Caps a value if greater
    than the cap. Leaves missing values alone (use inject + cap for both
    "set if missing" and "lower if too high").
    """
    if not caps:
        return body
    out = deepcopy(body)
    for path, cap in caps.items():
        if not isinstance(cap, int) or cap <= 0:
            continue
        keys = path.split(".")
        cur = out
        for k in keys[:-1]:
            v = cur.get(k)
            if not isinstance(v, dict):
                break
            cur = cur[k]
        else:
            leaf = keys[-1]
            cur_val = cur.get(leaf)
            if isinstance(cur_val, int) and cur_val > cap:
                cur[leaf] = cap
    return out


def _flatten_tool_output(output: Any) -> str:
    """Codex 0.128+ tool outputs (function_call_output.output) can be:
      - a plain string ("ran ok")
      - a list of content items, mixing text and input_image:
          [{"type": "input_text", "text": "..."},
           {"type": "input_image", "image_url": "data:image/png;base64,..."}]

    Chat-completions backends (DeepSeek, Ollama) reject `input_image` with
    HTTP 400 ("unknown variant `input_image`, expected `text`"). Vision is
    a model capability anyway, and the local executor isn't a vision model,
    so we flatten to plain text and replace each image with a `[image
    attached]` placeholder so the executor still knows an image was there.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                t = item.get("type")
                if t in ("text", "input_text", "output_text"):
                    txt = item.get("text", "")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
                elif t in ("input_image", "image", "image_url"):
                    parts.append("[image attached: vision content omitted "
                                 "for local executor]")
                # silently drop other unknown types
        return "\n".join(parts)
    # Fallback: stringify dicts/etc.
    try:
        import json as _j
        return _j.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(output)


def normalize_for_chat(body: dict[str, Any], *, strip_tools: bool = False) -> dict[str, Any]:
    """Convert a Responses-style body to a chat-completions body for backends
    that only speak chat (LMStudio default endpoint, Ollama, etc.).

    If *strip_tools* is True, tools and tool_choice are omitted entirely.
    Use this for backends like vLLM without --enable-auto-tool-choice.

    This is intentionally minimal and lossy - keeps user/assistant text and
    tool calls but drops reasoning items. Use only when the local backend
    really needs chat-format.
    """
    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": [],
        # Default to non-stream to match _run_forward's is_stream default of
        # False. Mismatched defaults caused upstream to return SSE while the
        # non-stream forward path tried to parse it as JSON (clients like
        # Vercel AI SDK's generateObject hit /v1/responses without a stream
        # field).
        "stream": body.get("stream", False),
    }
    # carry over a few common knobs if present
    for k in ("temperature", "top_p", "max_tokens", "stop", "features"):
        if k in body:
            out[k] = body[k]
    # tool_choice and tools: only forward when strip_tools is False.
    # vLLM without --enable-auto-tool-choice rejects any request that
    # contains "tools", so we strip both by default for chat backends.
    # The caller (proxy) can override via strip_tools=False.
    if not strip_tools:
        tc = body.get("tool_choice")
        if tc is not None and tc != "auto":
            out["tool_choice"] = tc
    # Responses-API uses `max_output_tokens`; chat-completions uses
    # `max_tokens`. Translate when the chat field is absent so any cap
    # set via `inject_responses_defaults` (e.g. local.inject_defaults
    # max_output_tokens=16000 to prevent runaway DeepSeek output) is
    # actually honored on the chat wire.
    if "max_tokens" not in out and isinstance(body.get("max_output_tokens"), int):
        out["max_tokens"] = body["max_output_tokens"]
    # Convert tools from Responses API format to chat-completions format.
    # Responses: {"type": "function", "name": "x", "parameters": {...}, "description": "..."}
    # Chat:      {"type": "function", "function": {"name": "x", "parameters": {...}, "description": "..."}}
    # Skip entirely when strip_tools=True (vLLM without --enable-auto-tool-choice).
    if not strip_tools and "tools" in body and isinstance(body["tools"], list):
        chat_tools = []
        for t in body["tools"]:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function" and "function" not in t:
                chat_tools.append({
                    "type": "function",
                    "function": {
                        k: v for k, v in t.items()
                        if k in ("name", "description", "parameters", "strict")
                    },
                })
            else:
                chat_tools.append(t)
        if chat_tools:
            out["tools"] = chat_tools
    # Build messages from instructions + input
    instr = body.get("instructions")
    if isinstance(instr, str) and instr.strip():
        out["messages"].append({"role": "system", "content": instr})
    src = body.get("input") or body.get("messages") or []
    # Responses API allows `input` to be a plain string (shorthand for a
    # single user message). Normalize to a list so the rest of the function
    # can iterate uniformly.
    if isinstance(src, str):
        out["messages"].append({"role": "user", "content": src})
        return out
    if not isinstance(src, list):
        return out

    # Collect call_ids that have a matching output so we only emit paired calls.
    # Widen the scan to all _TOOL_RESULT_TYPES (function_call_output, tool_result,
    # mcp_result) — older codex builds and some MCP paths emit non-default
    # result types and the previous narrow check silently dropped their
    # paired function_call items, hiding history from the model.
    output_ids: set[str] = set()
    for it in src:
        if isinstance(it, dict) and it.get("type") in _TOOL_RESULT_TYPES:
            cid = it.get("call_id") or it.get("id") or ""
            if cid:
                output_ids.add(cid)

    # Two-pass: first linearize into (role, payload) tuples, then merge.
    raw_msgs: list[dict[str, Any]] = []
    pending_reasoning: str = ""
    for it in src:
        if not isinstance(it, dict):
            continue
        role = it.get("role")
        t = it.get("type")
        content = it.get("content")
        if role == "developer":
            role = "system"

        if t in _REASONING_ITEM_TYPES:
            # Extract reasoning text so it can be passed back as
            # reasoning_content on the next assistant message (required by
            # DeepSeek thinking mode).
            summary = it.get("summary")
            if isinstance(summary, list):
                for s in summary:
                    if isinstance(s, dict):
                        pending_reasoning += s.get("text", "")
            elif isinstance(content, str):
                pending_reasoning += content
            continue
        if role and content is not None:
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content
                              if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text")]
                text = "\n".join(p for p in text_parts if p)
                if text:
                    msg: dict[str, Any] = {"role": role, "content": text}
                    if role == "assistant" and pending_reasoning:
                        msg["reasoning_content"] = pending_reasoning
                        pending_reasoning = ""
                    raw_msgs.append(msg)
            else:
                msg = {"role": role, "content": content}
                if role == "assistant" and pending_reasoning:
                    msg["reasoning_content"] = pending_reasoning
                    pending_reasoning = ""
                raw_msgs.append(msg)
        elif t == "function_call":
            cid = it.get("call_id") or it.get("id") or "call"
            tc_msg: dict[str, Any] = {
                "_tc": True,
                "role": "assistant",
                "tool_calls": [{
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": it.get("name") or "",
                        "arguments": it.get("arguments") or "",
                    },
                }],
            }
            if pending_reasoning:
                tc_msg["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            raw_msgs.append(tc_msg)
            # If the call has no matching output anywhere in src, the
            # chat-completions backend will reject the request (every
            # tool_calls message MUST be paired with a tool result).
            # Was previously a silent `continue` that DROPPED the call
            # entirely — that hid history from the model and caused
            # subtle "I don't remember calling X" forgetting. Now we
            # emit the call AND synthesize a placeholder result so the
            # chat structure is valid and the model knows the call
            # happened.
            if cid not in output_ids:
                raw_msgs.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": (
                        "[tinyctx: tool result not present in transcript "
                        "(call elided by codex history mgmt or in flight); "
                        "call kept for context continuity]"
                    ),
                })
        elif t in _TOOL_RESULT_TYPES:
            # Widened from `function_call_output` only — see output_ids
            # comment above. `output` is the typical Responses-API field;
            # `content` is what tool_result/mcp_result use.
            raw_msgs.append({
                "role": "tool",
                "tool_call_id": it.get("call_id") or it.get("id") or "call",
                "content": _flatten_tool_output(
                    it.get("output") if it.get("output") is not None else it.get("content")
                ),
            })

    # Merge pass: consecutive assistant text + tool_calls → one message;
    # consecutive tool_calls → one message with multiple tool_calls.
    # reasoning_content is preserved on the merged message.
    merged: list[dict[str, Any]] = []
    for msg in raw_msgs:
        is_tc = msg.pop("_tc", False)
        rc = msg.pop("reasoning_content", None)
        if is_tc and merged and merged[-1].get("role") == "assistant":
            prev = merged[-1]
            if "tool_calls" in prev:
                prev["tool_calls"].extend(msg["tool_calls"])
            else:
                prev["tool_calls"] = msg["tool_calls"]
            # Attach reasoning_content to the merged assistant message
            if rc and "reasoning_content" not in prev:
                prev["reasoning_content"] = rc
        else:
            if rc:
                msg["reasoning_content"] = rc
            merged.append(msg)

    system_msgs = [msg for msg in out["messages"] if msg.get("role") == "system"]
    system_msgs.extend(msg for msg in merged if msg.get("role") == "system")
    non_system_msgs = [msg for msg in merged if msg.get("role") != "system"]
    out["messages"] = []
    if system_msgs:
        system_parts: list[str] = []
        for msg in system_msgs:
            content = msg.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(json.dumps(content, ensure_ascii=False, default=str))
        out["messages"].append({
            "role": "system",
            "content": "\n\n".join(part for part in system_parts if part),
        })
    out["messages"].extend(non_system_msgs)

    # ── DeepSeek thinking-mode reasoning_content compat ──
    # codex 0.128+ ships `type=reasoning` items with empty content/summary/
    # encrypted_content (the actual thinking text is server-only). With
    # nothing to reconstruct, our normalize loop never attaches
    # `reasoning_content` to assistant messages. DeepSeek's thinking-mode
    # endpoints then 400 with:
    #     The `reasoning_content` in the thinking mode must be passed back
    #     to the API.
    # Empty-string reasoning_content satisfies the strict check while
    # signalling honestly that we have no thinking text to forward.
    # Toggle off by setting TINYCTX_FORCE_REASONING_STUB=0.
    import os as _os
    if _os.environ.get("TINYCTX_FORCE_REASONING_STUB", "1") == "1":
        for _m in out["messages"]:
            if _m.get("role") == "assistant" and "reasoning_content" not in _m:
                _m["reasoning_content"] = ""
    return out


def _tool_call_signature(item: dict[str, Any]) -> str | None:
    """Hash the (name, arguments) of a tool-call item. Returns None if the
    item is not a tool call."""
    t = item.get("type")
    if t not in _TOOL_CALL_TYPES:
        return None
    name = item.get("name") or item.get("tool_name") or ""
    args = item.get("arguments")
    if args is None:
        args = item.get("input") or item.get("arguments_text") or ""
    if isinstance(args, (dict, list)):
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
    else:
        args_str = str(args)
    sig = hashlib.sha256(f"{name}\0{args_str}".encode("utf-8", "replace")).hexdigest()[:16]
    return sig


def flatten_tool_schemas(
    body: dict[str, Any],
    *,
    leaf_threshold: int = 10,
    depth_threshold: int = 2,
) -> tuple[dict[str, Any], dict[str, set[str]], dict[str, Any]]:
    """Flatten complex nested tool parameter schemas into dot-notation.

    Returns:
      - rewritten body
      - tool_name -> flattened dotted keys
      - info dict for trace/logging
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body, {}, {"applied": False, "flattened_tools": []}

    out = deepcopy(body)
    flattened: dict[str, set[str]] = {}
    flattened_tools: list[str] = []

    for entry in out.get("tools", []):
        if not isinstance(entry, dict):
            continue
        tool_name, params = _tool_name_and_parameters(entry)
        if not tool_name or not isinstance(params, dict):
            continue
        rewritten, flat_keys = _flatten_parameter_schema(
            params,
            leaf_threshold=leaf_threshold,
            depth_threshold=depth_threshold,
        )
        if not flat_keys:
            continue
        _set_tool_parameters(entry, rewritten)
        flattened[tool_name] = flat_keys
        flattened_tools.append(tool_name)

    return out, flattened, {
        "applied": bool(flattened_tools),
        "flattened_tools": flattened_tools,
    }


def renest_tool_arguments(
    tool_name: str,
    arguments: str | dict[str, Any],
    flattened_tool_keys: dict[str, set[str]] | None,
) -> str | dict[str, Any]:
    """Reconstruct nested JSON args from dot-notation keys for one tool."""
    if not flattened_tool_keys or tool_name not in flattened_tool_keys:
        return arguments
    flat_keys = flattened_tool_keys.get(tool_name) or set()
    if not flat_keys:
        return arguments

    parsed: Any = arguments
    as_string = isinstance(arguments, str)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            return arguments
    if not isinstance(parsed, dict):
        return arguments

    nested: dict[str, Any] = {}
    changed = False
    for key, value in parsed.items():
        if isinstance(key, str) and key in flat_keys and "." in key:
            _set_dotted(nested, key, value)
            changed = True
        else:
            nested[key] = value
    if not changed:
        return arguments
    if as_string:
        return json.dumps(nested, ensure_ascii=False)
    return nested


def _tool_name_and_parameters(entry: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    name = entry.get("name")
    params = entry.get("parameters")
    if isinstance(name, str) and isinstance(params, dict):
        return name, params
    fn = entry.get("function")
    if isinstance(fn, dict):
        fname = fn.get("name")
        fparams = fn.get("parameters")
        if isinstance(fname, str) and isinstance(fparams, dict):
            return fname, fparams
    return "", None


def _set_tool_parameters(entry: dict[str, Any], rewritten: dict[str, Any]) -> None:
    if isinstance(entry.get("parameters"), dict):
        entry["parameters"] = rewritten
        return
    fn = entry.get("function")
    if isinstance(fn, dict):
        fn["parameters"] = rewritten


def _flatten_parameter_schema(
    schema: dict[str, Any],
    *,
    leaf_threshold: int,
    depth_threshold: int,
) -> tuple[dict[str, Any], set[str]]:
    leaf_count, max_depth = _schema_complexity(schema)
    if leaf_count <= leaf_threshold and max_depth <= depth_threshold:
        return schema, set()
    if str(schema.get("type") or "object") != "object":
        return schema, set()
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return schema, set()

    flat_props: dict[str, Any] = {}
    flat_required: list[str] = []
    flat_keys: set[str] = set()
    top_required = set(schema.get("required") or [])
    for key, sub in props.items():
        _flatten_schema_node(
            name=str(key),
            schema=sub,
            out_props=flat_props,
            out_required=flat_required,
            flat_keys=flat_keys,
            parent_required=(key in top_required),
        )
    if not flat_keys:
        return schema, set()
    rewritten = dict(schema)
    rewritten["type"] = "object"
    rewritten["properties"] = flat_props
    if flat_required:
        rewritten["required"] = flat_required
    else:
        rewritten.pop("required", None)
    return rewritten, flat_keys


def _flatten_schema_node(
    *,
    name: str,
    schema: Any,
    out_props: dict[str, Any],
    out_required: list[str],
    flat_keys: set[str],
    parent_required: bool,
) -> None:
    if not isinstance(schema, dict):
        out_props[name] = {"type": "string"}
        return
    props = schema.get("properties")
    is_object = str(schema.get("type") or "object") == "object"
    if is_object and isinstance(props, dict) and props:
        required = set(schema.get("required") or [])
        for child_name, child_schema in props.items():
            dotted = f"{name}.{child_name}"
            _flatten_schema_node(
                name=dotted,
                schema=child_schema,
                out_props=out_props,
                out_required=out_required,
                flat_keys=flat_keys,
                parent_required=(parent_required and child_name in required),
            )
        return
    leaf = dict(schema)
    if "description" in leaf and isinstance(leaf["description"], str):
        leaf["description"] = f"[flattened from {name}] {leaf['description']}"
    else:
        leaf["description"] = f"[flattened from {name}]"
    out_props[name] = leaf
    if "." in name:
        flat_keys.add(name)
    if parent_required:
        out_required.append(name)


def _schema_complexity(schema: Any, *, depth: int = 0) -> tuple[int, int]:
    if not isinstance(schema, dict):
        return 1, depth
    props = schema.get("properties")
    is_object = str(schema.get("type") or "object") == "object"
    if is_object and isinstance(props, dict) and props:
        leaves = 0
        max_depth = depth
        for child in props.values():
            child_leaves, child_depth = _schema_complexity(child, depth=depth + 1)
            leaves += child_leaves
            max_depth = max(max_depth, child_depth)
        return leaves, max_depth
    return 1, depth


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def detect_tool_call_storm(
    body: dict[str, Any],
    *,
    recent_window: int = 8,
    repeat_threshold: int = 5,
) -> dict[str, Any]:
    """Detect repeated identical tool calls in the recent transcript.

    This is a proxy-side signal that the current model may be stuck in a
    low-value loop repeatedly attempting the same tool with the same args.
    """
    items = body.get("input") or body.get("messages")
    if not isinstance(items, list):
        return {"triggered": False, "count": 0, "tool_name": "", "signature": ""}
    recent_calls: list[tuple[int, str, str]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        sig = _tool_call_signature(it)
        if sig is None:
            continue
        name = str(it.get("name") or it.get("tool_name") or "")
        recent_calls.append((idx, name, sig))
    if not recent_calls:
        return {"triggered": False, "count": 0, "tool_name": "", "signature": ""}
    recent_calls = recent_calls[-max(1, recent_window):]
    counts = Counter(sig for _, _, sig in recent_calls)
    signature, count = counts.most_common(1)[0]
    if count < max(2, repeat_threshold):
        return {"triggered": False, "count": count, "tool_name": "", "signature": signature}
    tool_name = next(
        (name for _, name, sig in reversed(recent_calls) if sig == signature),
        "",
    )
    indices = [idx for idx, _, sig in recent_calls if sig == signature]
    return {
        "triggered": True,
        "count": count,
        "tool_name": tool_name,
        "signature": signature,
        "indices": indices,
        "recent_window": recent_window,
    }


def collect_failure_signals(
    body: dict[str, Any],
    *,
    recent_window: int = 8,
    storm_repeat_threshold: int = 5,
) -> dict[str, Any]:
    """Collect conservative preflight failure signals from visible history.

    Score meanings:
      - tool_call_storm: 2 points (strong signal)
      - multiple recent tool errors: 1 point
      - repeated dedup placeholders in recent history: 1 point
    """
    items = body.get("input") or body.get("messages")
    if not isinstance(items, list):
        return {"score": 0, "signals": []}

    score = 0
    signals: list[dict[str, Any]] = []

    storm = detect_tool_call_storm(
        body,
        recent_window=recent_window,
        repeat_threshold=storm_repeat_threshold,
    )
    if storm.get("triggered"):
        score += 2
        signals.append({
            "kind": "tool_call_storm",
            "tool_name": storm.get("tool_name", ""),
            "count": storm.get("count", 0),
        })

    recent_errors = _extract_errors(items, last_n=recent_window)
    if len(recent_errors) >= 2:
        score += 1
        signals.append({
            "kind": "recent_tool_errors",
            "count": len(recent_errors),
        })

    recent_tail = items[-max(1, recent_window):]
    placeholder_hits = 0
    for it in recent_tail:
        if not isinstance(it, dict):
            continue
        blob = json.dumps(it, ensure_ascii=False, default=str)
        if _DEDUP_PLACEHOLDER in blob:
            placeholder_hits += 1
    if placeholder_hits >= 2:
        score += 1
        signals.append({
            "kind": "dedup_placeholder_history",
            "count": placeholder_hits,
        })

    return {"score": score, "signals": signals}


def dedup_tool_calls(body: dict[str, Any]) -> dict[str, Any]:
    """Replace earlier identical tool-call+result pairs with a placeholder.
    Keeps the *last* occurrence intact so the agent's most recent attempt is
    fully visible. Pairs are detected by (name, arguments) hash; the result
    item directly following a deduped call (matched by call_id when present)
    is also collapsed.
    """
    out = deepcopy(body)
    items = out.get("input") or out.get("messages")
    if not isinstance(items, list) or len(items) < 4:
        return out

    # First pass: find indices of the LATEST occurrence per signature.
    last_idx_by_sig: dict[str, int] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        sig = _tool_call_signature(it)
        if sig is None:
            continue
        last_idx_by_sig[sig] = i

    # Build a map of call_id -> bool (is this tool call the latest)
    keep_call_ids: set[str] = set()
    drop_call_ids: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        sig = _tool_call_signature(it)
        if sig is None:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        if not cid:
            continue
        if last_idx_by_sig[sig] == i:
            keep_call_ids.add(cid)
        else:
            drop_call_ids.add(cid)

    # Second pass: rewrite earlier occurrences and their immediate results.
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        sig = _tool_call_signature(it)
        cid = it.get("call_id") or it.get("id") or ""
        if sig is not None and last_idx_by_sig[sig] != i:
            # earlier dup of a tool call — replace arguments with placeholder.
            if "arguments" in it:
                it["arguments"] = _DEDUP_PLACEHOLDER
            elif "input" in it:
                it["input"] = _DEDUP_PLACEHOLDER
            else:
                it["arguments"] = _DEDUP_PLACEHOLDER
            continue
        # tool result whose call was deduped — also collapse output
        if it.get("type") in _TOOL_RESULT_TYPES and cid in drop_call_ids:
            if "output" in it:
                it["output"] = _DEDUP_PLACEHOLDER
            elif "content" in it:
                it["content"] = _DEDUP_PLACEHOLDER
    return out


def purge_failed_tool_inputs(body: dict[str, Any],
                             *, after_turns: int = 4) -> dict[str, Any]:
    """Once a tool call has produced an error and `after_turns` assistant
    messages have followed, replace the original tool-call's arguments with a
    fixed placeholder. The error itself stays so the agent knows the path was
    tried. Useful for codex's pattern of retrying a path (`apply_patch`
    failures, etc.) without keeping every long stderr in context.
    """
    out = deepcopy(body)
    items = out.get("input") or out.get("messages")
    if not isinstance(items, list) or len(items) < 2:
        return out

    # Find tool-result items that look like errors.
    failed_call_ids: dict[str, int] = {}    # call_id -> result index
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_RESULT_TYPES:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        if not cid:
            continue
        text_blob = ""
        for k in ("output", "content", "result", "error"):
            v = it.get(k)
            if isinstance(v, str):
                text_blob += v
            elif isinstance(v, (list, dict)):
                text_blob += json.dumps(v, ensure_ascii=False)[:2000]
        if it.get("is_error") or it.get("error") or "error" in text_blob.lower()[:400]:
            failed_call_ids[cid] = i

    if not failed_call_ids:
        return out

    # Count assistant turns that followed each failed result.
    assistant_turns_after: dict[str, int] = {}
    for cid, ridx in failed_call_ids.items():
        n = 0
        for j in range(ridx + 1, len(items)):
            if not isinstance(items[j], dict):
                continue
            role = items[j].get("role")
            t = items[j].get("type")
            if role == "assistant" or t == "message" or t in _TOOL_CALL_TYPES:
                n += 1
        assistant_turns_after[cid] = n

    # Now rewrite the original tool-call args for sufficiently old failures.
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_CALL_TYPES:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        if cid in failed_call_ids and assistant_turns_after.get(cid, 0) >= after_turns:
            if "arguments" in it:
                it["arguments"] = _FAILED_PURGE_PLACEHOLDER
            elif "input" in it:
                it["input"] = _FAILED_PURGE_PLACEHOLDER
            else:
                it["arguments"] = _FAILED_PURGE_PLACEHOLDER
    return out


def shrink_large_tool_results(
    body: dict[str, Any],
    *,
    after_turns: int = 1,
    min_bytes: int = 12_000,
    signal_lines: int = 8,
    head_chars: int = 400,
    tail_chars: int = 800,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace old oversized tool results with a deterministic summary.

    Applies only to prior tool-result items that are:
      - at least `min_bytes` once flattened to text, and
      - followed by `after_turns` or more assistant turns.
    """
    out = deepcopy(body)
    items = out.get("input") or out.get("messages")
    if not isinstance(items, list) or len(items) < 2:
        return out, {"applied": False, "shrunk": 0, "call_ids": []}

    call_name_by_id: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict) or it.get("type") not in _TOOL_CALL_TYPES:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        if not cid:
            continue
        call_name_by_id[cid] = str(it.get("name") or it.get("tool_name") or "")

    assistant_turns_after: dict[int, int] = {}
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or it.get("type") not in _TOOL_RESULT_TYPES:
            continue
        n = 0
        for j in range(idx + 1, len(items)):
            nxt = items[j]
            if not isinstance(nxt, dict):
                continue
            role = nxt.get("role")
            t = nxt.get("type")
            if role == "assistant" or t == "message" or t in _TOOL_CALL_TYPES:
                n += 1
        assistant_turns_after[idx] = n

    shrunk = 0
    call_ids: list[str] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or it.get("type") not in _TOOL_RESULT_TYPES:
            continue
        if assistant_turns_after.get(idx, 0) < after_turns:
            continue
        text = _flatten_tool_output(it.get("output") if "output" in it else it.get("content"))
        if len(text.encode("utf-8")) < min_bytes:
            continue
        cid = str(it.get("call_id") or it.get("id") or "")
        summary = _summarize_tool_result(
            text,
            tool_name=call_name_by_id.get(cid, ""),
            signal_lines=signal_lines,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        if "output" in it:
            it["output"] = summary
        elif "content" in it:
            it["content"] = summary
        else:
            it["output"] = summary
        shrunk += 1
        if cid:
            call_ids.append(cid)

    return out, {
        "applied": shrunk > 0,
        "shrunk": shrunk,
        "call_ids": call_ids,
    }


def _summarize_tool_result(
    text: str,
    *,
    tool_name: str,
    signal_lines: int,
    head_chars: int,
    tail_chars: int,
) -> str:
    original_chars = len(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    signal_markers = (
        "error", "exception", "traceback", "failed", "failure",
        "warning", "warn", "pass", "passed", "diff", "+++",
        "---", "@@", "build", "test", "pytest",
    )
    picked_signals: list[str] = []
    for line in lines:
        low = line.lower()
        if any(marker in low for marker in signal_markers):
            picked_signals.append(line)
        if len(picked_signals) >= signal_lines:
            break

    head = text[:head_chars].strip()
    tail = text[-tail_chars:].strip() if original_chars > tail_chars else ""
    parts = [
        f"{_RESULT_SHRINK_MARKER} tool={tool_name or 'unknown'} chars={original_chars} lines={len(lines)}",
    ]
    if picked_signals:
        parts.append("signals:")
        parts.extend(f"- {line[:240]}" for line in picked_signals)
    if head:
        parts.append("head:")
        parts.append(head)
    if tail and tail != head:
        parts.append("tail:")
        parts.append(tail)
    return "\n".join(parts)


# ----------------------------------------------------- cache-aware deferral

@dataclass
class CacheAwareMutator:
    """Gate for history-mutating transforms (dedup_tool_calls,
    purge_failed_tool_inputs). Inspired by cortexkit/magic-context's
    cache-aware deferred operations.

    The problem: dedup/purge change history bytes, which invalidates the
    upstream prompt-cache prefix. Anthropic's cache TTL is 5 min and reads
    cost 0.1× base — so blindly mutating on every turn destroys the dominant
    cost-saving lever.

    The fix: only fire mutations when at least one is true:
        (a) the cache prefix is probably stale anyway (TTL elapsed), OR
        (b) the context is filling up and we'd be forced to compact soon.

    Otherwise leave history untouched and let the cache earn its discount.
    """

    ttl_seconds: float = 300.0      # match Anthropic 5-min cache TTL
    threshold: float = 0.65         # context-usage trigger
    last_applied: dict[str, float] = field(default_factory=dict)

    def should_apply(
        self,
        session_id: str,
        *,
        est_tokens: int,
        max_tokens: int,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """Return (apply_now, reason). The reason is human-readable and
        suitable for logging."""
        t = now if now is not None else time.time()
        usage = est_tokens / max(1, max_tokens)
        if usage >= self.threshold:
            return True, f"context_usage={usage:.0%} >= {self.threshold:.0%}"
        last = self.last_applied.get(session_id, 0.0)
        # First-ever request for a session: don't fire (stay in cache window).
        if last == 0.0:
            self.last_applied[session_id] = t
            return False, f"first_turn_in_session deferred (usage={usage:.0%})"
        age = t - last
        if age >= self.ttl_seconds:
            return True, f"queue_age={age:.0f}s >= {self.ttl_seconds:.0f}s"
        return False, (f"deferred (usage={usage:.0%}, "
                       f"age={age:.0f}s < {self.ttl_seconds:.0f}s)")

    def mark_applied(self, session_id: str, *, now: float | None = None) -> None:
        self.last_applied[session_id] = now if now is not None else time.time()


# ----------------------------------------------- proactive history truncation
#
# Why this exists: codex.app 0.128 hard-codes `context_window: 272000` for
# gpt-5.5 and ships `auto_compact_token_limit: null` (auto-compact disabled
# by default). When tinyctx routes to a 1M-context backend like DeepSeek,
# codex's client-side history can grow to ~850k before it gives up with
# "Codex ran out of room in the model's context window. Start a new thread
# or clear earlier history before retrying." — and tinyctx never gets to
# influence that error path because codex aborts client-side.
#
# Two layers of defense:
#   (1) Set `model_auto_compact_token_limit = 200000` at the TOP LEVEL of
#       ~/.codex/config.toml so codex itself triggers compact before its
#       internal 272k ceiling. (Profile-scoped settings don't apply when
#       running the default profile; this is a known foot-gun.)
#   (2) Proxy-side fallback (this function): when codex DOESN'T self-compact
#       in time, tinyctx detects est_tokens above a danger threshold and
#       silently rewrites the forwarded body — keeps system prompt + most
#       recent N turns + a single tinyctx-generated summary item replacing
#       the older middle. The upstream model sees a slim body that fits;
#       codex's client-side history is unchanged so the UI still shows
#       every turn.
#
# Tradeoff: codex re-sends the bloated history every turn, so we'd compact
# on every subsequent request. We cache the summary keyed by (session_id,
# pre-recent-turn-count) so back-to-back turns reuse it cheaply.

# Cache: {(session_id, bucket_index): summary_text}
# bucket_index = len(middle_items) // _PROACTIVE_CACHE_BUCKET_SIZE
# Same bucket → same cached summary, reused for ~_PROACTIVE_CACHE_BUCKET_SIZE
# turns. See the long comment in proactive_compact() for the rationale.
_PROACTIVE_SUMMARY_CACHE: dict[tuple[str, int], str] = {}
_PROACTIVE_CACHE_BUCKET_SIZE: int = 20

# Tool names whose latest call+output should be pinned through compaction.
# `update_plan` is codex's built-in plan tracker; others (TodoWrite, etc.)
# carry the same "what is the agent doing" semantic and would also bleed
# the agent if compacted away. Add new ones here if downstream agents
# adopt their own tracker tool. Order does not matter; the lookup scans
# backwards for the latest call whose name is in this set.
_TRACKER_TOOL_NAMES: frozenset[str] = frozenset({
    "update_plan",
    "TodoWrite",
})

# Tool names recognized as "shell-like" — outputs are exec results.
# Used by _extract_shell_history to surface recent shell activity into
# the compaction summary.
_SHELL_TOOL_NAMES: frozenset[str] = frozenset({
    "exec_command",
    "shell",
    "Bash",
    "local_shell_call",
})

# Tool names recognized as "edit-like" — calls represent file writes.
# Used by _extract_edits to surface recent file changes into the summary.
_EDIT_TOOL_NAMES: frozenset[str] = frozenset({
    "apply_patch",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "str_replace_editor",
})

# Substrings (case-insensitive) that flag a tool result as an error.
# Used by _extract_errors to surface recent failures into the summary.
_ERROR_MARKERS: tuple[str, ...] = (
    "error", "failed", "traceback", "exception",
    "assertionerror", "fatal:", "panic:", "non-zero exit",
)

# Substrings (case-insensitive) that flag a shell command as "interesting"
# enough to surface even when it didn't happen recently — build/test/git/
# package-manager invocations and similar high-signal commands. Used by
# _extract_shell_history so heavy build commands buried 200+ turns back
# still reach the post-compact summary.
_HIGH_SIGNAL_SHELL_MARKERS: tuple[str, ...] = (
    "build", "test", "pytest", "cargo", "gradle", "make", "npm ", "yarn ",
    "pnpm ", "uv run", "pip install", "git ", "curl ", "wget ", "ssh ",
    "docker ", "kubectl ", "terraform ", "ansible-", "go build", "go test",
    "rustc", "javac", "mvn ", "swift build", "xcodebuild", "deploy",
    "migrate", "psql", "mysql", "sqlite3",
)


def _hash_items(items: list) -> str:
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _is_compaction_request(body: dict[str, Any]) -> bool:
    """Cheap fingerprint match — same patterns as router.is_compaction_request
    but without importing router (avoid circular import)."""
    inst = body.get("instructions") or ""
    if not isinstance(inst, str):
        return False
    s = inst.lower()
    return ("create a handoff summary" in s
            or "another llm that will resume the task" in s
            or "seamlessly continue the work" in s)


def _flatten_history_for_summary(items: list, *, max_chars: int = 60_000) -> str:
    """Mini version of compactor._flatten_history — no role drafts, just a
    plain transcript blob for a single-pass compactor call."""
    out: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role") or it.get("type") or ""
        content = it.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t in ("text", "input_text", "output_text"):
                        parts.append(str(c.get("text", "")))
            text = "\n".join(parts)
        elif role in ("function_call", "tool_use"):
            args = it.get("arguments") or it.get("input") or ""
            text = f"[tool call: {it.get('name','?')}({str(args)[:400]})]"
        elif role in ("function_call_output", "tool_result"):
            v = it.get("output") or it.get("content") or ""
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)[:1500]
            text = f"[tool result: {str(v)[:1500]}]"
        if text and text.strip():
            out.append(f"<{role}>\n{text.strip()}\n</{role}>")
    blob = "\n\n".join(out)
    if len(blob) > max_chars:
        head = blob[: max_chars // 4]
        tail = blob[-(3 * max_chars // 4):]
        blob = head + "\n\n... [middle truncated by tinyctx proactive_compact] ...\n\n" + tail
    return blob


def _tool_result_text(item: dict[str, Any]) -> str:
    """Best-effort flatten of a tool-result item's output for extraction."""
    v = item.get("output")
    if v is None:
        v = item.get("content")
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(v)
    return str(v)


def _short_args(args: Any, *, limit: int = 200) -> str:
    """Compact one-line preview of a tool call's arguments."""
    if isinstance(args, (dict, list)):
        try:
            s = json.dumps(args, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            s = str(args)
    else:
        s = str(args) if args is not None else ""
    s = s.replace("\n", " ").strip()
    return s[:limit] + ("..." if len(s) > limit else "")


def _extract_shell_history(items: list, *, last_n: int = 15,
                           high_signal_extra: int = 10) -> list[str]:
    """Return up to last_n one-line summaries of RECENT shell-like tool
    calls + their outputs PLUS up to high_signal_extra HIGH-SIGNAL entries
    (build/test/git/etc.) that may sit further back in history.

    The post-compact agent needs both:
      - recency (what just happened, regardless of importance), AND
      - importance (the cargo build / pytest invocation 200 turns ago is
        load-bearing even if 200 echo/ls calls happened since).

    Each entry looks like: "`cmd preview`: output_preview"
    Output is chronological (newest last), de-duplicated by command line.
    """
    # Index outputs by call_id for pairing.
    outputs: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") in _TOOL_RESULT_TYPES:
            cid = it.get("call_id") or it.get("id") or ""
            if cid:
                outputs[cid] = _tool_result_text(it)

    # Collect every shell call (with original index for chronological ordering).
    every: list[tuple[int, str, str]] = []  # (index, cmd_preview, line)
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_CALL_TYPES:
            continue
        if it.get("name") not in _SHELL_TOOL_NAMES:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        args = it.get("arguments")
        cmd_preview = ""
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                cmd = parsed.get("command") if isinstance(parsed, dict) else None
                if isinstance(cmd, list):
                    cmd_preview = " ".join(str(c) for c in cmd)
                elif isinstance(cmd, str):
                    cmd_preview = cmd
            except Exception:  # noqa: BLE001 — tool args may be non-JSON
                # Why: shell tool args usually JSON but legacy callers
                # send raw strings. Fall through to `_short_args` below
                # so we still produce a preview rather than blank.
                pass
        if not cmd_preview:
            cmd_preview = _short_args(args, limit=160)
        out_preview = (outputs.get(cid, "") or "").replace("\n", " ").strip()
        if len(out_preview) > 200:
            out_preview = out_preview[:200] + "..."
        line = (f"`{cmd_preview[:160]}`: {out_preview}" if out_preview
                else f"`{cmd_preview[:160]}`")
        every.append((idx, cmd_preview, line))

    if not every:
        return []

    # Recent slice (last N).
    recent = every[-last_n:]
    recent_idx_set = {x[0] for x in recent}

    # High-signal slice: scan all, prefer those NOT already in recent.
    high_signal: list[tuple[int, str, str]] = []
    for entry in every:
        idx, cmd_preview, _line = entry
        if idx in recent_idx_set:
            continue
        low = cmd_preview.lower()
        if any(m in low for m in _HIGH_SIGNAL_SHELL_MARKERS):
            high_signal.append(entry)
    # Keep the LATEST `high_signal_extra` high-signal entries.
    high_signal = high_signal[-high_signal_extra:]

    # Merge, sort chronologically, dedup by command line.
    merged = sorted(recent + high_signal, key=lambda x: x[0])
    seen: set[str] = set()
    out_lines: list[str] = []
    for _idx, _cmd, line in merged:
        if line in seen:
            continue
        seen.add(line)
        out_lines.append(line)
    return out_lines


def _extract_edits(items: list, *, last_n: int = 10) -> list[str]:
    """Return up to last_n one-line summaries of recent file-edit calls."""
    found: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_CALL_TYPES:
            continue
        if it.get("name") not in _EDIT_TOOL_NAMES:
            continue
        name = it.get("name") or ""
        args = it.get("arguments")
        preview = _short_args(args, limit=200)
        found.append(f"{name}: {preview}")
    return found[-last_n:]


def _extract_errors(items: list, *, last_n: int = 8) -> list[str]:
    """Return up to last_n distinct error/failure snippets from tool outputs."""
    found: list[str] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_RESULT_TYPES:
            continue
        text = _tool_result_text(it)
        if not text:
            continue
        low = text.lower()
        if not any(m in low for m in _ERROR_MARKERS):
            continue
        # Pick a short signature for dedup — first 80 chars after the marker.
        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        key = snippet[:80]
        if key in seen:
            continue
        seen.add(key)
        found.append(snippet)
    return found[-last_n:]


def detect_tool_call_storm(
    body: dict[str, Any],
    *,
    recent_window: int = 8,
    repeat_threshold: int = 5,
) -> dict[str, Any]:
    """Detect repeated identical tool calls in the recent transcript."""
    items = body.get("input") or body.get("messages")
    if not isinstance(items, list):
        return {"triggered": False, "count": 0, "tool_name": "", "signature": ""}
    recent_calls: list[tuple[int, str, str]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        sig = _tool_call_signature(it)
        if sig is None:
            continue
        name = str(it.get("name") or it.get("tool_name") or "")
        recent_calls.append((idx, name, sig))
    if not recent_calls:
        return {"triggered": False, "count": 0, "tool_name": "", "signature": ""}
    recent_calls = recent_calls[-max(1, recent_window):]
    signature, count = Counter(sig for _, _, sig in recent_calls).most_common(1)[0]
    if count < max(2, repeat_threshold):
        return {
            "triggered": False,
            "count": count,
            "tool_name": "",
            "signature": signature,
        }
    tool_name = next(
        (name for _, name, sig in reversed(recent_calls) if sig == signature),
        "",
    )
    indices = [idx for idx, _, sig in recent_calls if sig == signature]
    return {
        "triggered": True,
        "count": count,
        "tool_name": tool_name,
        "signature": signature,
        "indices": indices,
        "recent_window": recent_window,
    }


def collect_failure_signals(
    body: dict[str, Any],
    *,
    recent_window: int = 8,
    storm_repeat_threshold: int = 5,
) -> dict[str, Any]:
    """Collect conservative preflight failure signals from visible history."""
    items = body.get("input") or body.get("messages")
    if not isinstance(items, list):
        return {"score": 0, "signals": []}

    score = 0
    signals: list[dict[str, Any]] = []

    storm = detect_tool_call_storm(
        body,
        recent_window=recent_window,
        repeat_threshold=storm_repeat_threshold,
    )
    if storm.get("triggered"):
        score += 2
        signals.append({
            "kind": "tool_call_storm",
            "tool_name": storm.get("tool_name", ""),
            "count": storm.get("count", 0),
        })

    recent_errors = _extract_errors(items, last_n=recent_window)
    if len(recent_errors) >= 2:
        score += 1
        signals.append({
            "kind": "recent_tool_errors",
            "count": len(recent_errors),
        })

    recent_tail = items[-max(1, recent_window):]
    placeholder_hits = 0
    for it in recent_tail:
        if not isinstance(it, dict):
            continue
        blob = json.dumps(it, ensure_ascii=False, default=str)
        if _DEDUP_PLACEHOLDER in blob:
            placeholder_hits += 1
    if placeholder_hits >= 2:
        score += 1
        signals.append({
            "kind": "dedup_placeholder_history",
            "count": placeholder_hits,
        })

    return {"score": score, "signals": signals}


def _build_signals_section(items: list) -> str:
    """Render shell/edit/error extractions into a markdown preamble.

    Returns "" when nothing was extracted; otherwise a multi-line block
    that the summarizer can quote or rewrite into its handoff. Both the
    summarizer blob AND the fallback placeholder prepend this text so the
    information survives even when the LM call fails.
    """
    shells = _extract_shell_history(items, last_n=15)
    edits = _extract_edits(items, last_n=10)
    errs = _extract_errors(items, last_n=8)
    if not (shells or edits or errs):
        return ""
    parts: list[str] = ["## Pre-extracted execution signals"]
    if edits:
        parts.append("### Recent file edits")
        parts.extend(f"- {e}" for e in edits)
    if shells:
        parts.append("### Recent shell activity")
        parts.extend(f"- {s}" for s in shells)
    if errs:
        parts.append("### Errors / failures observed")
        parts.extend(f"- {e}" for e in errs)
    return "\n".join(parts)


def proactive_compact(
    body: dict[str, Any],
    *,
    session_id: str,
    est_tokens: int,
    threshold_tokens: int,
    recent_keep: int = 8,
    summarizer: Callable[[str], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """If `est_tokens >= threshold_tokens`, rewrite `body.input` to:
        [system items kept as-is]
        + [single message item with a tinyctx summary of older turns]
        + [last `recent_keep` items kept as-is]

    Returns (new_body, info_dict). `info_dict` always includes:
        applied: bool         — whether truncation actually happened
        reason: str           — human-readable for trace
        items_before: int
        items_after: int

    `summarizer` takes the flattened history blob and returns a summary
    string. If None or it raises, we fall back to a deterministic
    "[tinyctx: N older turns omitted to fit context]" placeholder so we
    NEVER fail the request — quality regression beats hard error.

    Skips when:
      - est_tokens < threshold_tokens
      - body is itself a codex compaction request (don't compact a compact)
      - history has fewer items than `recent_keep + 3` (nothing to drop)
      - body has no `input` array (chat-completions style or odd shape)
    """
    info: dict[str, Any] = {"applied": False, "reason": "below_threshold",
                            "items_before": 0, "items_after": 0}

    if est_tokens < threshold_tokens:
        return body, info

    if _is_compaction_request(body):
        info["reason"] = "skip_codex_compaction"
        return body, info

    items = body.get("input")
    if not isinstance(items, list):
        info["reason"] = "no_input_array"
        return body, info

    info["items_before"] = len(items)
    if len(items) < recent_keep + 3:
        info["reason"] = f"too_few_items ({len(items)} < {recent_keep + 3})"
        return body, info

    # Split: keep system-ish items at the head, summarize middle, keep tail.
    # Codex puts system bytes in `instructions` (top-level), not in `input`,
    # so most input items are conversational. We still defensively keep any
    # leading "system" or "developer" role items at the head.
    head: list = []
    rest: list = list(items)
    while rest:
        first = rest[0]
        if isinstance(first, dict):
            r = first.get("role") or first.get("type") or ""
            if r in ("system", "developer"):
                head.append(rest.pop(0))
                continue
        break

    if len(rest) <= recent_keep:
        info["reason"] = f"recent_keep covers all rest ({len(rest)} <= {recent_keep})"
        return body, info

    middle = rest[:-recent_keep]
    tail = rest[-recent_keep:]

    # ── Goal + tracker pinning ────────────────────────────────────────
    # The old logic compacted EVERYTHING in `middle` into a summary item.
    # Two specific item types must NOT be lost regardless of position,
    # otherwise the post-compact agent loses grounding:
    #
    #   1. The first `role: "user"` message — the original task goal.
    #      Without it the agent has no idea what it was asked to do; the
    #      summary alone (especially when summarizer failed) is just
    #      "[N older turns omitted]" with zero intent recovered.
    #   2. The latest `update_plan` function_call + its function_call_output
    #      — the agent's plan/tracker. Without it the agent either
    #      re-plans blindly or surfaces to the user.
    #
    # Pinning algorithm:
    #   - identify pin candidates by walking `middle` (head and tail items
    #     are already preserved verbatim, no pinning work needed there)
    #   - first user message: forward scan, first item with role=="user"
    #   - latest tracker: backward scan, latest function_call with name
    #     in _TRACKER_TOOL_NAMES; ALSO grab its paired function_call_output
    #     (by call_id) so we don't orphan either half
    #   - extract those items from middle (preserving order) and insert
    #     them right after the summary item (before the tail).
    #
    # If a pin candidate is already in head or tail, we don't re-extract
    # it from middle (it's already preserved).
    pinned_from_middle: list = []
    pinned_indices: set[int] = set()
    # First user message = the user's original goal. We pin from middle
    # ONLY when no user msg lives in head (head is preserved in original
    # order, so a user msg there IS the first one). User msgs in tail are
    # recent follow-ups, not the original goal — they don't count as
    # already-preserving the goal.
    user_in_head = any(
        isinstance(it, dict) and it.get("role") == "user"
        for it in head
    )
    if not user_in_head:
        for i, it in enumerate(middle):
            if isinstance(it, dict) and it.get("role") == "user":
                pinned_indices.add(i)
                break

    # Latest tracker call in middle
    latest_tracker_call_idx: int | None = None
    latest_tracker_call_id: str = ""
    for i in range(len(middle) - 1, -1, -1):
        it = middle[i]
        if (isinstance(it, dict)
                and it.get("type") in _TOOL_CALL_TYPES
                and it.get("name") in _TRACKER_TOOL_NAMES):
            latest_tracker_call_idx = i
            latest_tracker_call_id = it.get("call_id") or it.get("id") or ""
            break

    if latest_tracker_call_idx is not None:
        # Check head/tail for an existing tracker call — if a fresher one
        # is already there, no need to pin from middle.
        tracker_in_head_or_tail = any(
            isinstance(it, dict)
            and it.get("type") in _TOOL_CALL_TYPES
            and it.get("name") in _TRACKER_TOOL_NAMES
            for it in head + tail
        )
        if not tracker_in_head_or_tail:
            pinned_indices.add(latest_tracker_call_idx)
            # Find its paired function_call_output (search after the call)
            if latest_tracker_call_id:
                for j in range(latest_tracker_call_idx + 1, len(middle)):
                    it = middle[j]
                    if (isinstance(it, dict)
                            and it.get("type") in _TOOL_RESULT_TYPES
                            and (it.get("call_id") or it.get("id"))
                                == latest_tracker_call_id):
                        pinned_indices.add(j)
                        break

    # Build pinned_from_middle in original order so the LLM sees the
    # goal before the tracker (matches conversation chronology).
    # We keep `middle` itself intact — the pinned items still appear in
    # the summarizer blob (belt + suspenders: the goal/tracker show up
    # both verbatim AND inside the summary). The pinned items are
    # inserted into the FINAL out["input"] as their own verbatim copy.
    if pinned_indices:
        for i in sorted(pinned_indices):
            pinned_from_middle.append(middle[i])

    # Tool-call/output pairing repair. The Responses API requires every
    # `function_call_output` to have a matching `function_call` somewhere
    # earlier in the input. Slicing the middle away can leave orphan
    # outputs in tail whose calls were dropped — chatgpt.com 400s with:
    #   "No tool call found for function call output with call_id ..."
    #
    # Strategy: for every orphan output in tail (i.e. output whose call_id
    # is not present as a function_call in head ∪ tail), synthesize a stub
    # `function_call` item with that call_id immediately before the output.
    # The stub carries an obvious tinyctx marker so the upstream model
    # sees the call as "already happened in the compacted history".
    head_call_ids = {
        it.get("call_id") or it.get("id")
        for it in head if isinstance(it, dict)
        and it.get("type") in _TOOL_CALL_TYPES
    }
    tail_call_ids = {
        it.get("call_id") or it.get("id")
        for it in tail if isinstance(it, dict)
        and it.get("type") in _TOOL_CALL_TYPES
    }
    repaired_tail: list = []
    synthetic_calls = 0
    for it in tail:
        if (isinstance(it, dict)
                and it.get("type") in _TOOL_RESULT_TYPES):
            cid = it.get("call_id") or it.get("id") or ""
            if cid and cid not in head_call_ids and cid not in tail_call_ids:
                # Synthesize a matching function_call stub right before this output.
                repaired_tail.append({
                    "type": "function_call",
                    "call_id": cid,
                    "name": "tinyctx_compacted_call",
                    "arguments": json.dumps({
                        "note": "original call was elided by tinyctx "
                                "proactive_compact; see summary item above"
                    }),
                })
                synthetic_calls += 1
        repaired_tail.append(it)
    tail = repaired_tail

    # Cache key strategy: BUCKET-based, not per-turn-hash.
    #
    # Old design hashed `middle` directly. Since middle grows by ~1 item
    # per turn, the hash changed every turn → cache miss every turn →
    # summarizer ran every turn AND the model saw a slightly different
    # summary on each turn (information drift, "subtle forgetting").
    #
    # New design: bucket the middle by length. Same bucket → same cached
    # summary, reused across the bucket window (default 20 turns).
    # When middle grows enough to flip buckets, regenerate ONCE and use
    # for the next 20 turns. Trade-off: the summary lags by up to 20
    # turns of fall-off content, which is fine because that content is
    # ALSO in the recent_keep tail (still shown verbatim).
    #
    # The session_id stays in the key so different conversations don't
    # cross-contaminate. _PROACTIVE_CACHE_BUCKET_SIZE controls the
    # refresh granularity; smaller = more frequent regeneration =
    # higher cost but fresher summary.
    bucket = len(middle) // _PROACTIVE_CACHE_BUCKET_SIZE
    cache_key = (session_id, bucket)
    summary_text = _PROACTIVE_SUMMARY_CACHE.get(cache_key)
    cached = summary_text is not None

    # Pre-extract execution signals (shell / edits / errors). This block
    # is deterministically appended to the final summary regardless of
    # whether the LM summarizer succeeds — so the post-compact agent ALWAYS
    # sees recent shell activity, file edits, and error markers. The same
    # block is also prepended to the summarizer blob so the LM can quote
    # or refine it within its narrative.
    signals_section = _build_signals_section(middle)

    if summary_text is None:
        blob = _flatten_history_for_summary(middle)
        # Incremental seed: when we have a previous bucket's summary,
        # prepend it to the blob so the summarizer can extend it rather
        # than re-summarize from scratch. This keeps continuity across
        # bucket refreshes — the new summary is a refined version of the
        # old one + the new fall-off content, not an independent view.
        prev_summary = _PROACTIVE_SUMMARY_CACHE.get((session_id, bucket - 1))
        if prev_summary and bucket > 0:
            blob = (
                "## Previous handoff summary (carry forward and refine)\n\n"
                + prev_summary
                + "\n\n## Additional turns since that summary\n\n"
                + blob
            )
        if signals_section:
            blob = (
                signals_section
                + "\n\n## Raw transcript of older turns\n\n"
                + blob
            )
        if summarizer is not None:
            try:
                summary_text = summarizer(blob)
            except Exception as e:  # noqa: BLE001
                summary_text = (
                    f"[tinyctx proactive_compact: summarizer failed ({e!s}); "
                    f"{len(middle)} older turns omitted to fit context]"
                )
        if not summary_text:
            summary_text = (
                f"[tinyctx proactive_compact: {len(middle)} older turns "
                "omitted to fit context. Recent turns and system prompt "
                "remain. Ask the user if you need details from earlier.]"
            )
        # Always append the pre-extracted signals to the summary text so
        # they survive even when the LM summary is terse or the fallback
        # placeholder fires. This is the load-bearing line for execution-
        # state preservation: shell/edit/error markers ALWAYS reach the
        # post-compact agent.
        if signals_section:
            summary_text = summary_text.rstrip() + "\n\n" + signals_section
        _PROACTIVE_SUMMARY_CACHE[cache_key] = summary_text

    summary_item = {
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": (
                "[tinyctx auto-compact: the conversation is approaching the "
                "context window limit. Earlier turns have been replaced with "
                "this summary. The most recent turns follow this message "
                "verbatim.]\n\n## Handoff summary of earlier turns\n\n"
                + summary_text
            ),
        }],
    }

    out = deepcopy(body)
    # Pinned items (first user goal + latest tracker pair) go between the
    # summary item and the tail — i.e. they sit in chronological order
    # AFTER the summary explains "these older turns got compacted, but
    # these specific items are kept verbatim because they carry the goal
    # and current tracker state."
    out["input"] = head + [summary_item] + pinned_from_middle + tail

    info["applied"] = True
    info["reason"] = (f"est_tokens={est_tokens} >= {threshold_tokens}, "
                      f"compacted {len(middle)} middle items "
                      f"({'cached' if cached else 'fresh'} summary"
                      + (f", {synthetic_calls} synthetic call stubs"
                         if synthetic_calls else "")
                      + (f", {len(pinned_from_middle)} pinned items"
                         if pinned_from_middle else "")
                      + ")")
    info["items_after"] = len(out["input"])
    info["middle_items_compacted"] = len(middle)
    info["cached"] = cached
    info["synthetic_call_stubs"] = synthetic_calls
    info["pinned_items"] = len(pinned_from_middle)
    return out, info


def trim_tools_for_frontier(
    body: dict[str, Any],
    *,
    recent_window: int = 30,
    essentials: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce `body["tools"]` to a working set before forwarding to frontier.

    Codex 0.128 sends ~50 tools per request (~10k tokens). Most sessions
    only call a handful. This filter keeps:
      (a) any tool whose name appears as a function_call.name within the
          last `recent_window` items of body.input — i.e. tools the model
          ACTUALLY USED recently, so it can repeat the pattern;
      (b) any tool whose name is in `essentials` — guaranteed to remain
          available even on a fresh turn (shell, apply_patch, advisor, etc.).

    Returns (new_body, info_dict) with info:
        applied: bool
        reason: str
        tools_before, tools_after: int
        kept_names: list[str]   (sorted, for trace logging)
        dropped_names: list[str]

    Defensive: if body has no tools, no input, or fewer tools than 5,
    returns the body unchanged (no point trimming a small list).
    """
    info: dict[str, Any] = {"applied": False, "reason": "no_tools",
                            "tools_before": 0, "tools_after": 0,
                            "kept_names": [], "dropped_names": []}

    tools = body.get("tools")
    if not isinstance(tools, list) or len(tools) < 5:
        info["reason"] = f"few_tools ({len(tools) if isinstance(tools, list) else 0} < 5)"
        info["tools_before"] = len(tools) if isinstance(tools, list) else 0
        info["tools_after"] = info["tools_before"]
        return body, info

    info["tools_before"] = len(tools)

    # Collect tool names actually used in the recent window.
    items = body.get("input")
    used_names: set[str] = set()
    if isinstance(items, list) and items:
        for it in items[-recent_window:]:
            if not isinstance(it, dict):
                continue
            if it.get("type") in _TOOL_CALL_TYPES:
                name = it.get("name")
                if isinstance(name, str):
                    used_names.add(name)

    keep_set = set(essentials) | used_names

    kept: list = []
    dropped_names: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            kept.append(t)
            continue
        name = t.get("name") or ""
        if name in keep_set:
            kept.append(t)
        else:
            dropped_names.append(name)

    if len(kept) == len(tools):
        info["reason"] = "all_tools_in_keep_set"
        info["tools_after"] = len(kept)
        info["kept_names"] = sorted({t.get("name","") for t in kept if isinstance(t, dict)})
        return body, info

    out = deepcopy(body)
    out["tools"] = kept

    info["applied"] = True
    info["reason"] = (
        f"recent={len(used_names)} essentials={len(essentials)} -> "
        f"kept {len(kept)}/{len(tools)}"
    )
    info["tools_after"] = len(kept)
    info["kept_names"] = sorted({t.get("name","") for t in kept if isinstance(t, dict)})
    info["dropped_names"] = sorted(dropped_names)
    return out, info


def drop_orphan_tool_outputs(
    body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Final preflight: remove orphan tool-output items from `body.input`.

    The Responses API rejects an output item (function_call_output,
    tool_result, mcp_result, tool_search_output) whose `call_id` has no
    matching call item earlier in the input. chatgpt.com codex backend
    returns HTTP 400:
        "No tool call found for tool search output with call_id ..."

    Orphans can arise from several upstream sources we don't fully
    control: proactive_compact eliding the matching call from the
    middle (handled for function_call types via stub synthesis; not
    handled for tool_search_call), codex client reordering, history
    dedup placeholders, or upstream client bugs. This pass is a
    belt-and-suspenders defense that runs after every other sanitize
    step so it catches orphans regardless of cause.

    Asymmetry note: an orphan CALL without a matching output is benign
    per OpenAI semantics (means the call didn't produce output yet, or
    the output is in this turn's response). We do NOT drop those.

    Returns (new_body, info). info keys:
        applied: bool   — at least one orphan dropped
        dropped: int    — count of orphan items removed
        call_ids: list  — orphan call_ids dropped (for logging)
    """
    info: dict[str, Any] = {"applied": False, "dropped": 0, "call_ids": []}
    items = body.get("input")
    if not isinstance(items, list):
        return body, info
    call_ids = {
        it.get("call_id") or it.get("id")
        for it in items
        if isinstance(it, dict)
        and it.get("type") in _ORPHAN_PAIR_CALL_TYPES
    }
    call_ids.discard(None)
    call_ids.discard("")
    new_items: list = []
    dropped_call_ids: list[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") in _ORPHAN_PAIR_OUTPUT_TYPES:
            cid = it.get("call_id") or it.get("id") or ""
            if cid and cid not in call_ids:
                dropped_call_ids.append(cid)
                continue
        new_items.append(it)
    if not dropped_call_ids:
        return body, info
    out = dict(body)
    out["input"] = new_items
    info["applied"] = True
    info["dropped"] = len(dropped_call_ids)
    info["call_ids"] = dropped_call_ids
    return out, info


def clear_proactive_cache(session_id: str | None = None) -> None:
    """Test helper: clear the proactive_compact summary cache. Pass a
    session_id to clear only that session, or None to clear everything."""
    if session_id is None:
        _PROACTIVE_SUMMARY_CACHE.clear()
        return
    keys_to_drop = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == session_id]
    for k in keys_to_drop:
        _PROACTIVE_SUMMARY_CACHE.pop(k, None)
