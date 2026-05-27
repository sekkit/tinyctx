"""Stream rewriting: intercept upstream `response.completed` and inject
a synthetic `function_call` when the soft_completion classifier
returns PUNT with high confidence, so the agent resumes work instead
of stopping prematurely.

Why this exists
───────────────
The rule-based gate (soft_completion + AGENTS.md PATH A) depends on
agent self-discipline. Live trace 2026-05-10: agent saw the gate
twice in a row and kept emitting `plan-without-tool-call`. Loop.

This module bypasses agent discipline by directly forging the SSE
event sequence codex parses for a function_call. When the classifier
says PUNT with p ≥ threshold, the proxy holds the upstream's
`response.completed` event, emits synthetic events for a function_call
that codex will actually dispatch (see `synthetic_continue.STRATEGIES`
— `shell`/`local_shell`/`update_plan` no-ops), then emits the held
`response.completed`.

What about advisor? The original design tried to inject a synthetic
`spawn_agent(role="advisor")` here. Binary analysis + 0 observed
ADVISOR routes proved codex silently drops it (probable schema
mismatch / origin validation). `synthetic_advisor_call_events` is
kept for tests and future re-enable, but the live proxy uses
`synthetic_continue.build_continue_injection` instead — those are
codex builtins that DO get dispatched. Real advisor sub-agent calls
happen only when the executor itself emits `model="tinyctx-frontier"`.

Codex sees:
  <whatever the agent emitted>
  + (synthetic) function_call to a codex builtin (shell true / update_plan / …)
  + response.completed

→ codex executes the builtin, returns the output to the model on
the next turn. Agent resumes with new context instead of stopping.

Risk profile
────────────
Relies on codex parsing a function_call by name. If codex's
namespace dispatcher returns "unsupported call" (the user's config.toml
note), the rewrite surfaces as an error in codex chat. Opt-in via
`soft_completion_stream_rewrite_enabled = true` in
`~/.tinyctx/config.toml [global]`. Easy rollback.
"""
from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4


# SSE event marker we intercept. Both the upstream-passthrough path
# and the chat→responses translator emit this exact event name when
# the upstream stream completes. Cap-sensitive.
_COMPLETED_MARKER = b"event: response.completed"


def looks_like_response_completed(chunk: bytes) -> bool:
    """Cheap pre-check: does the chunk contain a response.completed
    SSE event header? Used to short-circuit the heavier split."""
    return _COMPLETED_MARKER in chunk


def split_at_completed(chunk: bytes) -> tuple[bytes, bytes]:
    """Split a chunk into (pre, completed_event_and_rest). The chunk
    might bundle multiple SSE events; we want everything BEFORE the
    completed event yielded immediately, and the completed event held
    for synthesis injection.

    Returns `(b"", chunk)` if the marker isn't present (caller checks
    via `looks_like_response_completed` first, but this is a safe
    fallback)."""
    idx = chunk.find(_COMPLETED_MARKER)
    if idx < 0:
        return b"", chunk
    return chunk[:idx], chunk[idx:]


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    """Format an SSE event the way codex's parser expects.
    `event:` line + `data:` JSON line + blank line."""
    return (f"event: {event_name}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n").encode()


# Build the task body the synthetic advisor call will pass. Keeps the
# advisor's input self-contained (it sees only what we pack here, not
# the full conversation).

_TASK_TEMPLATE = """\
The executor turn just ended with finish_reason=stop, emitting a plan/text but NO tool call. Per tinyctx soft-completion gate, the executor must NOT pause for user input on a plan-without-action. Decide whether to:

  1. Tell the executor to keep working (plan is good, just execute it).
  2. Or flag a real reason it should stop here (genuine ambiguity, missing input, irreversible decision).

Classifier said: `{reason}` (p={p:.2f}).
finish_reason: stop (no tool call followed)
text length: {text_chars} chars
text excerpt (last 1500 chars):
---
{text_excerpt}
---

Reply with EXACTLY ONE line, no prose:
  - work: <bullet list — concrete next tool calls the executor should make to advance the plan>
  - ask: <one-line reason a human is genuinely needed>

The executor will execute `work:` items via tools (apply_patch / shell / update_plan / etc.) without further checkin."""


def build_task_body(
        text_excerpt: str,
        classifier_reason: str,
        classifier_p: float,
) -> str:
    """Compose the task field for the synthetic advisor call. Caps
    text_excerpt at 1500 chars (advisor sees only what we pack)."""
    excerpt = (text_excerpt or "")[-1500:]
    return _TASK_TEMPLATE.format(
        reason=classifier_reason or "plan without tool call",
        p=classifier_p,
        text_chars=len(text_excerpt or ""),
        text_excerpt=excerpt,
    )


def synthetic_advisor_call_events(
        task: str,
        tool_name: str = "spawn_agent",
        output_index: int = 99,
        extra_args: dict[str, Any] | None = None,
) -> list[bytes]:
    """Synthesize the SSE event sequence that codex parses as a
    function_call to advisor. Sequence (Responses-API):

      1. response.output_item.added         — item is a function_call
      2. response.function_call_arguments.delta  — streams the args JSON
      3. response.function_call_arguments.done   — finalizes args
      4. response.output_item.done          — item complete

    `tool_name` defaults to `spawn_agent` (codex's native sub-agent
    dispatcher; routes via `[agents.advisor]` config when role=advisor).
    Override to `mcp__advisor__ask_advisor` for codex < 0.128 / MCP-only
    setups.

    `extra_args` is merged into the call's arguments JSON alongside
    `task` — e.g. `{"role": "advisor"}` for spawn_agent. None defaults
    to no extra args (caller's task is the sole arg).

    `output_index` defaults to 99 to avoid colliding with upstream's
    own indexed items (which start at 0)."""
    item_id = "fc_tinyctx_" + uuid4().hex[:16]
    args_obj: dict[str, Any] = {"task": task}
    if extra_args:
        # Caller's keys take precedence over `task` if collision (rare).
        args_obj = {**args_obj, **extra_args}
        args_obj["task"] = task  # ensure task always present
    args_json = json.dumps(args_obj, ensure_ascii=False)

    item_skeleton: dict[str, Any] = {
        "type": "function_call",
        "id": item_id,
        "name": tool_name,
        "arguments": "",
    }
    item_done: dict[str, Any] = {**item_skeleton, "arguments": args_json}

    return [
        _sse_event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": item_skeleton,
        }),
        _sse_event("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": output_index,
            "delta": args_json,
        }),
        _sse_event("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": output_index,
            "arguments": args_json,
        }),
        _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item_done,
        }),
    ]
