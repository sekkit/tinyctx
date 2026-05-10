"""Synthetic tool-call injection strategies — break the codex
"finish_reason=stop pause for user" loop by injecting a function_call
event that codex's tool dispatcher will actually execute.

Why
───
Stream-rewrite tried `spawn_agent` first; binary analysis + 0 observed
ADVISOR routes proved codex silently drops it (likely schema mismatch
or origin validation). This module provides a ranked list of
ALTERNATIVE synthetic tool calls — each one a real codex builtin
that's far more likely to be dispatched.

When a strategy is dispatched by codex, codex returns the
function_call_output to the model. Model resumes generation with that
result in context — turn does NOT pause for user input. Loop broken.

Strategies (ranked safest → most invasive)
──────────────────────────────────────────
1. `shell` no-op — `["true"]` does nothing on POSIX, returns instantly.
   The result is empty stdout/stderr, exit 0. Model gets a confirmation
   the shell ran; can decide what to do next.
2. `local_shell` variant — same but uses the alternate registered name
   we saw in the binary (some codex versions prefer this).
3. `update_plan` no-op — sends an empty plan update with an
   explanation. Codex's plan tool always accepts this and routes back
   to model.

Per-session strategy rotation
─────────────────────────────
On EACH soft-completion-driven injection for a session, pick the NEXT
untried strategy. After cycling through all, start over (maybe a later
attempt works; conditions can change). Tracker is `_NEXT_STRATEGY_IDX`.

Per-session injection budget
────────────────────────────
`_INJECTION_COUNT_PER_SESSION` caps how many synthetic continues we
inject for one session. When over budget, `build_continue_injection`
returns the `budget_exhausted` sentinel; the caller is expected to
escalate (force frontier on next turn) and to inject a one-shot
`<system-reminder>` warning the agent that auto-continue ran out — a
genuine "agent finished" outcome should be reviewed manually instead
of nudged forever.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from uuid import uuid4


# Per-session counter of which strategy to try next. Rotates 0..N-1
# then back to 0. A future enhancement could detect which strategies
# successfully dispatched and prefer those, but rotation is a fine
# default.
_NEXT_STRATEGY_IDX: dict[str, int] = defaultdict(int)


# Per-session count of synthetic-continue injections. Caps runaway
# loops where soft_completion mis-classifies and we keep injecting
# forever. Default cap is the caller-provided `max_injections`.
_INJECTION_COUNT_PER_SESSION: dict[str, int] = defaultdict(int)


# Per-session flag tracking whether the budget-exhausted system-
# reminder has already been injected. One-shot per exhaustion event so
# we don't append the warning every turn.
_LAST_BUDGET_REMINDER_FIRED: dict[str, bool] = defaultdict(bool)


# Strategy registry. Each entry: tool_name codex dispatches + the
# arguments dict to inject. The synthetic event chain (output_item.added,
# function_call_arguments.delta + done, output_item.done) is built by
# the same SSE event builder used for spawn_agent — we just vary the
# `name` and `arguments` payloads.

STRATEGIES: list[dict[str, Any]] = [
    {
        "label": "shell_noop",
        "tool_name": "shell",
        "args": {"command": ["true"]},
    },
    {
        "label": "local_shell_noop",
        "tool_name": "local_shell",
        "args": {"command": ["true"]},
    },
    {
        "label": "update_plan_noop",
        "tool_name": "update_plan",
        "args": {
            "explanation": ("tinyctx auto-continue: tracker unchanged, "
                             "agent should resume executing the plan."),
            "plan": [],
        },
    },
]


def pick_next_strategy(proj_sid: str) -> dict[str, Any]:
    """Return the next strategy for this session, rotating through the
    list. Always returns a strategy (rotates back to 0 after exhausting)."""
    idx = _NEXT_STRATEGY_IDX[proj_sid] % len(STRATEGIES)
    _NEXT_STRATEGY_IDX[proj_sid] = (idx + 1) % len(STRATEGIES)
    return STRATEGIES[idx]


def reset_strategy_index(proj_sid: str) -> None:
    """Reset rotation back to strategy 0 — call when a strategy is
    confirmed working (e.g. observed function_call_output from codex
    matching one of our synthetic ids in the next request)."""
    _NEXT_STRATEGY_IDX[proj_sid] = 0


# ─── injection budget ────────────────────────────────────────────────────


def injection_count(proj_sid: str) -> int:
    return _INJECTION_COUNT_PER_SESSION.get(proj_sid, 0)


def is_over_budget(proj_sid: str, max_injections: int) -> bool:
    return _INJECTION_COUNT_PER_SESSION.get(proj_sid, 0) >= max_injections


# ─── synthetic SSE event builder ──────────────────────────────────────────

def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    return (f"event: {event_name}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n").encode()


def synthetic_tool_call_events(
        tool_name: str,
        args: dict[str, Any],
        output_index: int = 99,
) -> list[bytes]:
    """Synthesize the 4-event SSE sequence codex parses as a function_call:
      response.output_item.added (item type=function_call)
      response.function_call_arguments.delta (args streaming)
      response.function_call_arguments.done
      response.output_item.done

    `output_index` defaults to 99 to avoid colliding with upstream's
    own indexed items.
    """
    item_id = "fc_tinyctx_" + uuid4().hex[:16]
    args_json = json.dumps(args, ensure_ascii=False)

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


def build_continue_injection(
        proj_sid: str,
        max_injections: int = 20,
) -> tuple[list[bytes], dict[str, Any]]:
    """High-level: pick next strategy + build its SSE events. Returns
    (events, strategy_used). The proxy yields events into the response
    stream; the strategy dict goes into log/forensics.

    When `_INJECTION_COUNT_PER_SESSION[proj_sid] >= max_injections`,
    returns `([], {"label": "budget_exhausted", ...})` and does not
    increment the counter. Caller should escalate to frontier and inject
    `build_budget_exhausted_reminder` text on the next request.
    """
    if is_over_budget(proj_sid, max_injections):
        return [], {"label": "budget_exhausted", "tool_name": "", "args": {}}
    strategy = pick_next_strategy(proj_sid)
    _INJECTION_COUNT_PER_SESSION[proj_sid] += 1
    events = synthetic_tool_call_events(
        strategy["tool_name"], strategy["args"])
    return events, strategy


# ─── budget-exhausted reminder ───────────────────────────────────────────

_BUDGET_REMINDER_TEMPLATE = """\
<system-reminder>
[NOT USER INPUT — tinyctx proxy injection-budget watchdog]

This session has been auto-continued **{count} times** by tinyctx after detecting `finish_reason=stop` while the tracker still appeared open. The synthetic-continue budget is now exhausted.

If you are GENUINELY DONE with the user's request:
- Say so explicitly in one short sentence so the user can verify and route the next ask.

If you are NOT done:
- Either invoke `spawn_agent(role="advisor", task=...)` to plan the next concrete action, OR surface a clear blocker to the user (one specific question that would unblock you).

The next `finish_reason=stop` for this session will route to the frontier model instead of being auto-continued. This warning is one-shot — it will not repeat next turn.
</system-reminder>"""


def build_budget_exhausted_reminder(proj_sid: str, count: int) -> str:
    """Return the system-reminder text shown to the agent on the request
    AFTER `build_continue_injection` returned `budget_exhausted`."""
    return _BUDGET_REMINDER_TEMPLATE.format(count=count)


def maybe_inject_budget_reminder(
        body: dict[str, Any],
        proj_sid: str,
        count: int,
) -> tuple[dict[str, Any], bool]:
    """Append `build_budget_exhausted_reminder` text to `body.input` once
    per exhaustion. Subsequent calls for the same session return
    `(body, False)` until `reset_state(proj_sid)` clears the flag.
    Mirrors stuck_loop's API."""
    if _LAST_BUDGET_REMINDER_FIRED.get(proj_sid):
        return body, False
    items = body.get("input")
    if not isinstance(items, list):
        return body, False
    new_items = list(items)
    new_items.append({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": build_budget_exhausted_reminder(proj_sid, count),
        }],
    })
    out = dict(body)
    out["input"] = new_items
    _LAST_BUDGET_REMINDER_FIRED[proj_sid] = True
    return out, True


# ─── test helpers ────────────────────────────────────────────────────────


def reset_state(proj_sid: str | None = None) -> None:
    if proj_sid is None:
        _NEXT_STRATEGY_IDX.clear()
        _INJECTION_COUNT_PER_SESSION.clear()
        _LAST_BUDGET_REMINDER_FIRED.clear()
        return
    _NEXT_STRATEGY_IDX.pop(proj_sid, None)
    _INJECTION_COUNT_PER_SESSION.pop(proj_sid, None)
    _LAST_BUDGET_REMINDER_FIRED.pop(proj_sid, None)


def state_snapshot(proj_sid: str,
                    max_injections: int = 20) -> dict[str, Any]:
    count = _INJECTION_COUNT_PER_SESSION.get(proj_sid, 0)
    return {
        "next_strategy_idx": _NEXT_STRATEGY_IDX.get(proj_sid, 0),
        "available_strategies": [s["label"] for s in STRATEGIES],
        "injection_count": count,
        "over_budget": count >= max_injections,
        "budget_reminder_fired": _LAST_BUDGET_REMINDER_FIRED.get(proj_sid, False),
    }
