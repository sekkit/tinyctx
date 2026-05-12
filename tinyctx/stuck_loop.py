"""Stuck-loop watchdog: detect when a codex session has been hammering
the same sub-problem for many turns and inject a `<system-reminder>`
into the request input asking the agent to either consult advisor or
surface its blocker to the user.

Why this exists
───────────────
Live trace 2026-05-10: a codex session ran `turn_count=1323` over
~25 minutes trying to debug an X3 Pro rendering issue, context grew
to 553K tokens, and codex.app eventually terminated the session
without the agent ever reaching a "Final summary" — so the §4
collection-discipline rule and the advisor completion-gate (added in
`tinyctx/templates/AGENTS.md`) never fired, because both trigger on
the agent declaring done. The agent was stuck, not premature-completing.

Strategy (Anthropic Advisor Strategy aligned)
─────────────────────────────────────────────
When the watchdog fires, instead of "letting the model figure it out",
push it through a hard pivot point: either spawn an advisor with a
self-honest "what am I missing" framing, or stop and surface to the
user. Recency-position injection at the tail of `body.input` exploits
attention recency so the reminder is hard to ignore even in a 500K-
token context where rule blocks at position 0 get drowned out.

Trigger calibration (defaults — tune in config)
───────────────────────────────────────────────
- `turn_trigger=80`  — most healthy tasks finish well below 80 turns.
                       The earlier failed task at 154 turns is borderline;
                       1323 is way over. 80 catches the latter without
                       false-firing on the former's tail.
- `turn_gap=50`      — between consecutive reminders. Don't nag every
                       turn; give the model 50 turns to act on advice.
- `advisor_grace_s=600` — if the agent already called advisor in the
                       last 10 minutes, skip the nudge. The agent is
                       already doing the right thing.

Idempotency
───────────
Injection is keyed off `_LAST_REMINDER_TURN[proj_sid]`. A retry of the
same turn won't re-inject (same `turn_count`). State is per-project-
session-key for multi-project isolation, same as `_SESSION_ERROR_STREAK`.

State storage
─────────────
P2 migration: state lives in `tinyctx.session_state` under namespace
`stuck_loop` keys `last_reminder_turn` and `last_advisor_ts`. The
legacy `_LAST_REMINDER_TURN` and `_LAST_ADVISOR_TS` module attributes
are preserved as `_SessionStateDictView` shims so tests that poke them
directly continue to work. NEITHER key is registered for compaction
reset (per 09b1946 saga fix: clearing reminder-turn caused rapid
double-nag; advisor activity remains relevant across compaction).
"""
from __future__ import annotations

import time
from typing import Any, Iterator

from . import session_state


# ─── SessionState namespace + compaction reset policy ─────────────────────

_NS = "stuck_loop"
_K_LAST_REMINDER_TURN = "last_reminder_turn"
_K_LAST_ADVISOR_TS = "last_advisor_ts"

# Intentionally EMPTY: per 09b1946 saga fix, neither last_reminder_turn
# nor last_advisor_ts is cleared on compaction. Codex's monotonic turn
# counter survives compaction, so the gate semantics still hold; advisor
# activity also remains relevant. Clearing either causes rapid double-nag.
session_state.register_compaction_reset(_NS, [])


# ─── legacy dict views ────────────────────────────────────────────────────


class _SessionStateDictView:
    """Read/write proxy that maps `view[proj_sid]` to a single SessionState
    key under namespace `stuck_loop`. Returns `_default` (typed-zero) for
    missing keys to preserve the original defaultdict semantics."""

    __slots__ = ("_key", "_default")

    def __init__(self, key: str, default: Any) -> None:
        self._key = key
        self._default = default

    def __iter__(self) -> Iterator[str]:
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if self._key in by_ns.get(_NS, {}):
                yield sid

    def keys(self) -> list[str]:
        return list(iter(self))

    def items(self) -> list[tuple[str, Any]]:
        snap = session_state.snapshot()
        out: list[tuple[str, Any]] = []
        for sid, by_ns in snap.items():
            if self._key in by_ns.get(_NS, {}):
                out.append((sid, by_ns[_NS][self._key]))
        return out

    def values(self) -> list[Any]:
        return [v for _, v in self.items()]

    def __contains__(self, proj_sid: Any) -> bool:
        snap_ns = session_state.snapshot(proj_sid).get(_NS, {})
        return self._key in snap_ns

    def __getitem__(self, proj_sid: Any) -> Any:
        val = session_state.get(proj_sid, _NS, self._key)
        return self._default if val is None else val

    def __setitem__(self, proj_sid: Any, value: Any) -> None:
        session_state.set(proj_sid, _NS, self._key, value)

    def __delitem__(self, proj_sid: Any) -> None:
        session_state.clear(proj_sid, _NS, self._key)

    def get(self, proj_sid: Any, default: Any = None) -> Any:
        val = session_state.get(proj_sid, _NS, self._key)
        return default if val is None else val

    def pop(self, proj_sid: Any, default: Any = None) -> Any:
        existing = session_state.consume(proj_sid, _NS, self._key)
        return existing if existing is not None else default

    def clear(self) -> None:
        for sid in list(iter(self)):
            session_state.clear(sid, _NS, self._key)


# Module-level dicts; per-process (proxy is single-process). Keys are
# `proj_sid` (composite project + session key) so projects don't share state.

_LAST_REMINDER_TURN = _SessionStateDictView(_K_LAST_REMINDER_TURN, default=0)
_LAST_ADVISOR_TS = _SessionStateDictView(_K_LAST_ADVISOR_TS, default=0.0)


# ─── reminder text ─────────────────────────────────────────────────────────
# Wrapped in <system-reminder> tags following codex.app's own internal
# reminder pattern. The "[NOT USER INPUT]" header makes it explicit so
# the model doesn't mistake the proxy nudge for a fresh user instruction.

_REMINDER_TEMPLATE = """\
<system-reminder>
[NOT USER INPUT — tinyctx proxy stuck-loop watchdog]

This session has accumulated **{turn_count} turns** without convergence on the current sub-problem. You may be in a debugging loop. Stop the current path NOW and do ONE of:

**A. Consult advisor** (preferred when you have ≥3 distinct attempts to describe):
```
spawn_agent(role="advisor", task=\"\"\"
Question: I've been stuck on <one-sentence problem statement> for {turn_count}+ turns.
Tried (be specific):
  1. <approach 1> — failed because <one-line reason>
  2. <approach 2> — failed because <one-line reason>
  3. <approach 3> — failed because <one-line reason>
What angle or hypothesis am I missing? Is the problem actually solvable from here?
\"\"\")
wait_agent(...)
```

**B. Surface to the user** (when the situation is genuinely under-specified):
- Summarize the exact sub-problem you're stuck on
- List the distinct paths you've tried and the concrete failure mode of each
- Ask the user the one specific question that would unblock you

Do NOT continue another iteration of the same debugging loop. This watchdog won't nag again for {turn_gap}+ turns.
</system-reminder>"""


# ─── public API ────────────────────────────────────────────────────────────


def maybe_inject_stuck_reminder(body: dict[str, Any],
                                 proj_sid: str,
                                 turn_count: int,
                                 *,
                                 turn_trigger: int = 80,
                                 turn_gap: int = 50,
                                 advisor_grace_s: float = 600.0,
                                 advisor_scope_sid: str | None = None,
                                 ) -> tuple[dict[str, Any], bool]:
    """Append a stuck-loop `<system-reminder>` to `body.input` when the
    session has run long without convergence. No-op when:
      - `turn_count < turn_trigger`
      - last reminder was less than `turn_gap` turns ago
      - an advisor was invoked within `advisor_grace_s` seconds
      - body has no `input` array (malformed; skip silently)

    `proj_sid` here is the SCOPE for the reminder gate — the caller
    decides whether it's per-project or per-conversation. Pass a
    conversation-scoped key so a new codex thread (whose `turn_count`
    resets to 0) is not blocked by a stale `_LAST_REMINDER_TURN` from
    the previous thread.

    `advisor_scope_sid` (optional) decouples the advisor grace lookup
    from the reminder gate. Pass the project-scoped key so advisor
    activity in any sub-thread quiets nudges across the project; if
    None, falls back to `proj_sid` (back-compat).

    Returns `(new_body, was_injected)`. The original body is not mutated.
    Trace records `stuck_reminder_injected` + `stuck_turn_count_at_inject`
    on injection so we can correlate effectiveness later.
    """
    advisor_key = advisor_scope_sid if advisor_scope_sid is not None else proj_sid
    if turn_count < turn_trigger:
        return body, False
    last_turn = int(session_state.get(proj_sid, _NS, _K_LAST_REMINDER_TURN, 0))
    if turn_count - last_turn < turn_gap:
        return body, False
    last_advisor_ts = float(session_state.get(
        advisor_key, _NS, _K_LAST_ADVISOR_TS, 0.0))
    if time.time() - last_advisor_ts < advisor_grace_s:
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
            "text": _REMINDER_TEMPLATE.format(
                turn_count=turn_count, turn_gap=turn_gap),
        }],
    })
    out = dict(body)
    out["input"] = new_items
    session_state.set(proj_sid, _NS, _K_LAST_REMINDER_TURN, turn_count)
    return out, True


def mark_advisor_call(proj_sid: str) -> None:
    """Note that an advisor sub-thread fired for this session. Resets
    the grace window so the watchdog won't nag right after the agent
    already escalated. Called by the proxy when it sees a request with
    `requested_model == "tinyctx-frontier"` (advisor sub-agent route)
    or any other explicit frontier-route signal."""
    session_state.set(proj_sid, _NS, _K_LAST_ADVISOR_TS, time.time())


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that session's reminder + advisor timing."""
    if proj_sid is None:
        for sid in list(_LAST_REMINDER_TURN.keys()):
            session_state.clear(sid, _NS, _K_LAST_REMINDER_TURN)
        for sid in list(_LAST_ADVISOR_TS.keys()):
            session_state.clear(sid, _NS, _K_LAST_ADVISOR_TS)
        return
    session_state.clear(proj_sid, _NS, _K_LAST_REMINDER_TURN)
    session_state.clear(proj_sid, _NS, _K_LAST_ADVISOR_TS)


def reset_compaction_state(conv_sid: str | None,
                             proj_sid: str | None = None) -> None:
    """Hook called when tinyctx observes a compaction boundary.

    Originally this cleared `_LAST_REMINDER_TURN[conv_sid]` on the
    rationale that codex's turn_count keeps climbing post-compaction and
    a clean baseline gives the watchdog a fresh window. Live observation
    proved that wrong: turn 322 fired a reminder, then a compaction
    landed mid-stream, and turn 324 re-fired (gap=2, far below the
    configured 50). The agent perceived rapid double-nag.

    Resolution: do NOT reset `_LAST_REMINDER_TURN` here. Codex's
    absolute turn counter is fine as a reference — the gate
    `turn_count - _LAST_REMINDER_TURN[scope] >= turn_gap` still holds
    its intended semantics across compaction. Worst case the first
    post-compaction reminder fires "as soon as the gap allows" instead
    of "as soon as turn_trigger allows" — but that's the right behavior
    for an agent that's been stuck through a compaction.

    Advisor grace timestamp is also preserved (advisor activity remains
    relevant regardless of compaction). The function is kept as a stable
    hook in case future tracker state needs compaction-boundary handling.

    No-op when `conv_sid` is falsy. `proj_sid` accepted but unused —
    kept for back-compat with proxy.py call sites.
    """
    if not conv_sid:
        return
    # Intentionally no state mutation. See docstring for rationale.
    return


def state_snapshot(proj_sid: str) -> dict[str, Any]:
    """Test/dev helper to inspect per-session watchdog state."""
    last_turn = int(session_state.get(
        proj_sid, _NS, _K_LAST_REMINDER_TURN, 0))
    last_advisor_ts = float(session_state.get(
        proj_sid, _NS, _K_LAST_ADVISOR_TS, 0.0))
    return {
        "last_reminder_turn": last_turn,
        "last_advisor_ts": last_advisor_ts,
        "seconds_since_advisor": (
            time.time() - last_advisor_ts if last_advisor_ts else None),
    }
