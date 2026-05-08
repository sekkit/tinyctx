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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


_REASONING_ITEM_TYPES = {"reasoning", "reasoning_summary", "thinking"}
_TOOL_CALL_TYPES = {"function_call", "tool_use", "mcp_call"}
_TOOL_RESULT_TYPES = {"function_call_output", "tool_result", "mcp_result"}

_DEDUP_PLACEHOLDER = "[tinyctx: identical call deduped — see later turn]"
_FAILED_PURGE_PLACEHOLDER = "[tinyctx: failed input purged after N turns]"


def strip_encrypted_content(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `body` with `encrypted_content` removed from every
    reasoning-style item. Cheap deepcopy - request bodies are small."""
    out = deepcopy(body)
    for key in ("input", "messages"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("type") or it.get("role") or ""
            if t in _REASONING_ITEM_TYPES:
                it.pop("encrypted_content", None)
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


def rewrite_model(body: dict[str, Any], target_model: str) -> dict[str, Any]:
    body["model"] = target_model
    return body


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


def expand_mcp_namespaces(body: dict[str, Any], *,
                          prefix_inner: bool = True) -> dict[str, Any]:
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
            if prefix_inner:
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


def normalize_for_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses-style body to a chat-completions body for backends
    that only speak chat (LMStudio default endpoint, Ollama, etc.).

    This is intentionally minimal and lossy - keeps user/assistant text and
    tool calls but drops reasoning items. Use only when the local backend
    really needs chat-format.
    """
    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": [],
        "stream": body.get("stream", True),
    }
    # carry over a few common knobs if present
    for k in ("temperature", "top_p", "max_tokens", "tool_choice", "stop"):
        if k in body:
            out[k] = body[k]
    # Convert tools from Responses API format to chat-completions format.
    # Responses: {"type": "function", "name": "x", "parameters": {...}, "description": "..."}
    # Chat:      {"type": "function", "function": {"name": "x", "parameters": {...}, "description": "..."}}
    #
    if "tools" in body and isinstance(body["tools"], list):
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
    if not isinstance(src, list):
        return out

    # Collect call_ids that have a matching output so we only emit paired calls.
    output_ids: set[str] = set()
    for it in src:
        if isinstance(it, dict) and it.get("type") == "function_call_output":
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
            if cid not in output_ids:
                continue
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
        elif t == "function_call_output":
            raw_msgs.append({
                "role": "tool",
                "tool_call_id": it.get("call_id") or it.get("id") or "call",
                "content": _flatten_tool_output(it.get("output")),
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

    out["messages"].extend(merged)

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

# Cache: {(session_id, n_pre_recent_items_hash): summary_text}
_PROACTIVE_SUMMARY_CACHE: dict[tuple[str, str], str] = {}


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

    cache_key = (session_id, _hash_items(middle))
    summary_text = _PROACTIVE_SUMMARY_CACHE.get(cache_key)
    cached = summary_text is not None

    if summary_text is None:
        blob = _flatten_history_for_summary(middle)
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
    out["input"] = head + [summary_item] + tail

    info["applied"] = True
    info["reason"] = (f"est_tokens={est_tokens} >= {threshold_tokens}, "
                      f"compacted {len(middle)} middle items "
                      f"({'cached' if cached else 'fresh'} summary"
                      + (f", {synthetic_calls} synthetic call stubs"
                         if synthetic_calls else "")
                      + ")")
    info["items_after"] = len(out["input"])
    info["middle_items_compacted"] = len(middle)
    info["cached"] = cached
    info["synthetic_call_stubs"] = synthetic_calls
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


def clear_proactive_cache(session_id: str | None = None) -> None:
    """Test helper: clear the proactive_compact summary cache. Pass a
    session_id to clear only that session, or None to clear everything."""
    if session_id is None:
        _PROACTIVE_SUMMARY_CACHE.clear()
        return
    keys_to_drop = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == session_id]
    for k in keys_to_drop:
        _PROACTIVE_SUMMARY_CACHE.pop(k, None)
