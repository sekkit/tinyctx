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
                "content": it.get("output") or "",
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
