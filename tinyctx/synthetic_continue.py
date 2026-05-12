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
attempt works; conditions can change). Tracker lives in SessionState
under namespace ``synthetic_continue`` key ``strategy_idx``.

Per-session injection budget
────────────────────────────
SessionState namespace ``synthetic_continue`` key ``injection_count``
caps how many synthetic continues we inject for one session. When over
budget, `build_continue_injection` returns the `budget_exhausted`
sentinel; the caller is expected to escalate (force frontier on next
turn) and to inject a one-shot `<system-reminder>` warning the agent
that auto-continue ran out — a genuine "agent finished" outcome should
be reviewed manually instead of nudged forever.

State storage
─────────────
This module is the P1 pilot for the unified `tinyctx.session_state`
container. The legacy `_INJECTION_COUNT_PER_SESSION` and
`_LAST_BUDGET_REMINDER_FIRED` module attributes are kept (live views
over `session_state` keyed by `synthetic_continue.injection_count` /
`budget_reminder_fired`) so external tests that wrote to them continue
to work.
"""
from __future__ import annotations

import json
from typing import Any, Iterator
from uuid import uuid4

from . import session_state

# ─── SessionState namespace + compaction reset policy ────────────────────

_NS = "synthetic_continue"
_K_STRATEGY_IDX = "strategy_idx"
_K_INJECTION_COUNT = "injection_count"
_K_BUDGET_REMINDER_FIRED = "budget_reminder_fired"

# Compaction clears injection_count + budget_reminder_fired but NOT
# strategy_idx (rotation is independent of conversation length).
session_state.register_compaction_reset(
    _NS, [_K_INJECTION_COUNT, _K_BUDGET_REMINDER_FIRED]
)


# ─── live dict views over session_state ──────────────────────────────────
# The previous implementation used module-level `dict[str, int]` /
# `dict[str, bool]` containers. Some tests (and possibly other modules)
# write to them or inspect them by key. `_SessionStateDictView` is a
# minimal dict-shaped proxy that forwards every operation to
# `session_state` so the legacy attributes keep working without
# duplicating state.


class _SessionStateDictView:
    """Read/write proxy that maps `view[conv_sid]` to a single
    SessionState key. Only the operations actually used by the legacy
    code paths and the existing tests are implemented."""

    __slots__ = ("_key", "_default")

    def __init__(self, key: str, default: Any) -> None:
        self._key = key
        self._default = default

    def __iter__(self) -> Iterator[str]:
        # Iterate every conv_sid that has a non-default value for this
        # key under the synthetic_continue namespace.
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if self._key in by_ns.get(_NS, {}):
                yield sid

    def keys(self) -> list[str]:
        return list(iter(self))

    def __contains__(self, conv_sid: Any) -> bool:
        snap_ns = session_state.snapshot(conv_sid).get(_NS, {})
        return self._key in snap_ns

    def __getitem__(self, conv_sid: Any) -> Any:
        val = session_state.get(conv_sid, _NS, self._key)
        return self._default if val is None else val

    def __setitem__(self, conv_sid: Any, value: Any) -> None:
        session_state.set(conv_sid, _NS, self._key, value)

    def __delitem__(self, conv_sid: Any) -> None:
        session_state.clear(conv_sid, _NS, self._key)

    def get(self, conv_sid: Any, default: Any = None) -> Any:
        val = session_state.get(conv_sid, _NS, self._key)
        return default if val is None else val

    def pop(self, conv_sid: Any, default: Any = None) -> Any:
        existing = session_state.get(conv_sid, _NS, self._key)
        session_state.clear(conv_sid, _NS, self._key)
        return existing if existing is not None else default

    def clear(self) -> None:
        # Walk every conv that has a value for this key and drop it.
        for sid in list(iter(self)):
            session_state.clear(sid, _NS, self._key)


_NEXT_STRATEGY_IDX = _SessionStateDictView(_K_STRATEGY_IDX, default=0)
_INJECTION_COUNT_PER_SESSION = _SessionStateDictView(
    _K_INJECTION_COUNT, default=0
)
_LAST_BUDGET_REMINDER_FIRED = _SessionStateDictView(
    _K_BUDGET_REMINDER_FIRED, default=False
)


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
    idx = int(session_state.get(proj_sid, _NS, _K_STRATEGY_IDX, 0)) % len(STRATEGIES)
    session_state.set(proj_sid, _NS, _K_STRATEGY_IDX, (idx + 1) % len(STRATEGIES))
    return STRATEGIES[idx]


def reset_strategy_index(proj_sid: str) -> None:
    """Reset rotation back to strategy 0 — call when a strategy is
    confirmed working (e.g. observed function_call_output from codex
    matching one of our synthetic ids in the next request)."""
    session_state.set(proj_sid, _NS, _K_STRATEGY_IDX, 0)


# ─── injection budget ────────────────────────────────────────────────────


def injection_count(proj_sid: str) -> int:
    return int(session_state.get(proj_sid, _NS, _K_INJECTION_COUNT, 0))


def is_over_budget(proj_sid: str, max_injections: int) -> bool:
    return injection_count(proj_sid) >= max_injections


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

    When the per-session injection_count has reached `max_injections`,
    returns `([], {"label": "budget_exhausted", ...})` and does not
    increment the counter. Caller should escalate to frontier and inject
    `build_budget_exhausted_reminder` text on the next request.
    """
    if is_over_budget(proj_sid, max_injections):
        return [], {"label": "budget_exhausted", "tool_name": "", "args": {}}
    strategy = pick_next_strategy(proj_sid)
    session_state.increment(proj_sid, _NS, _K_INJECTION_COUNT)
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
    if session_state.get(proj_sid, _NS, _K_BUDGET_REMINDER_FIRED, False):
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
    session_state.set(proj_sid, _NS, _K_BUDGET_REMINDER_FIRED, True)
    return out, True


# ─── test helpers ────────────────────────────────────────────────────────


def reset_state(proj_sid: str | None = None) -> None:
    """Clear all synthetic_continue state. When `proj_sid` is None, wipe
    every conv_sid's state under this namespace; otherwise clear only
    that session's three keys."""
    if proj_sid is None:
        # Clear this namespace across all conv_sids tracked in
        # SessionState. We don't call reset_all() because other
        # namespaces (other modules) may have unrelated state.
        for sid in list(_NEXT_STRATEGY_IDX.keys()):
            session_state.clear(sid, _NS, _K_STRATEGY_IDX)
        for sid in list(_INJECTION_COUNT_PER_SESSION.keys()):
            session_state.clear(sid, _NS, _K_INJECTION_COUNT)
        for sid in list(_LAST_BUDGET_REMINDER_FIRED.keys()):
            session_state.clear(sid, _NS, _K_BUDGET_REMINDER_FIRED)
        return
    session_state.clear(proj_sid, _NS, _K_STRATEGY_IDX)
    session_state.clear(proj_sid, _NS, _K_INJECTION_COUNT)
    session_state.clear(proj_sid, _NS, _K_BUDGET_REMINDER_FIRED)


def reset_compaction_state(conv_sid: str | None,
                             proj_sid: str | None = None) -> None:
    """Clear injection budget + budget-reminder flag for a conversation
    after tinyctx observes a compaction boundary. The strategy index is
    NOT cleared — strategy rotation is independent of conversation length
    and there's no benefit to resetting it.

    When `proj_sid` is supplied AND differs from `conv_sid`, also sweep
    every per-conv key prefixed by `f"{proj_sid}:"` — necessary because
    codex's compaction handoff request may omit `prompt_cache_key`,
    degrading `conv_sid` to just `proj_sid`, while normal-turn keys for
    the SAME project look like `"proj_sid:cache_key"`. Without the sweep,
    pre-compaction counters survive across the boundary.

    No-op when `conv_sid` is falsy so callers that lost conv_sid scope
    can still call this without guards.
    """
    if not conv_sid:
        return
    # Per-namespace compaction reset for the exact conv_sid.
    session_state.reset_compaction(conv_sid)

    # When the compaction-handoff request degraded `conv_sid` to
    # `proj_sid`, also sweep all sibling per-conv keys.
    if proj_sid and proj_sid == conv_sid:
        prefix = f"{proj_sid}:"
        for sid in session_state.keys_with_prefix(prefix):
            session_state.reset_compaction(sid)


def state_snapshot(proj_sid: str,
                    max_injections: int = 20) -> dict[str, Any]:
    count = injection_count(proj_sid)
    return {
        "next_strategy_idx": int(
            session_state.get(proj_sid, _NS, _K_STRATEGY_IDX, 0)),
        "available_strategies": [s["label"] for s in STRATEGIES],
        "injection_count": count,
        "over_budget": count >= max_injections,
        "budget_reminder_fired": bool(
            session_state.get(proj_sid, _NS, _K_BUDGET_REMINDER_FIRED, False)),
    }
