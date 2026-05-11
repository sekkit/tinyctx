"""Mid-stream stall watchdog: detect when an upstream backend HANGS during
SSE relay — no events, no error, just silence.

Why this exists
───────────────
`empty_response_guard` catches near-empty responses POST-stream, after the
backend ended cleanly with 1-2 completion tokens. It does NOT catch the
other interruption mode: the backend opened the SSE stream, emitted some
prefix, then stopped emitting events and never closed the connection.
Codex.app sits there waiting for `response.completed` while the proxy
sits there waiting for the next byte. Without this watchdog, the session
hangs until codex's idle timeout fires (often minutes) or the user kills
it manually.

Inspired by openai/symphony SPEC §8.5 Part A and codex's own
`stall_timeout_ms`. Fires when no upstream events arrive within a
configurable threshold (default 180s).

Mechanism
─────────
Per-session last-event state lives in `_LAST_EVENT` keyed by proj_sid;
each entry stores both the monotonic timestamp and (optionally) the
conv_sid the caller supplied at `mark_event` time so the stall callback
can scope force-frontier escalation to one conversation rather than the
whole project. The proxy's SSE relay calls `mark_event(proj_sid, conv_sid)`
each time it sees a parsed upstream event. A background asyncio task polls every
`check_interval_s` seconds and fires `on_stall(proj_sid)` for any
session whose last event was longer than `threshold_s` ago. After
firing, the session is removed from the dict to prevent re-firing on
the next poll.

`time.monotonic()` is used throughout — wall-clock can jump (NTP, DST,
suspend/resume) and would fire spurious stalls.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Union


_LOG = logging.getLogger("tinyctx.stall_watchdog")

# Per-session state: {sid: {"ts": monotonic, "conv_sid": str | None}}.
# `sid` is the proj_sid the watchdog loop iterates over; `conv_sid` is
# the per-conversation key stored by callers that have body access at
# `mark_event` time. The stall callback prefers conv_sid (precise — only
# that conversation's next request gets force-routed) and falls back to
# the iterated sid when conv_sid wasn't supplied.
_LAST_EVENT: dict[str, dict[str, Any]] = {}


# ─── public API ────────────────────────────────────────────────────────────


def mark_event(proj_sid: str, conv_sid: str | None = None) -> None:
    """Record that `proj_sid` just received an upstream event. Called
    from the SSE relay loop on each parsed event. Cheap dict write.

    `conv_sid` (optional) is the per-conversation key derived from
    `prompt_cache_key`. When provided, the stall callback will scope
    force-frontier escalation to that conversation instead of the whole
    project — preventing a stall in conversation A from leaking into
    conversation B's next request. When omitted, the previous-stored
    conv_sid is preserved (a later call without conv_sid doesn't erase
    an earlier one)."""
    if not proj_sid:
        return
    entry = _LAST_EVENT.get(proj_sid)
    now = time.monotonic()
    if entry is None:
        _LAST_EVENT[proj_sid] = {"ts": now, "conv_sid": conv_sid}
        return
    entry["ts"] = now
    if conv_sid is not None:
        entry["conv_sid"] = conv_sid


def check_stalled(proj_sid: str, threshold_s: float) -> bool:
    """True iff `proj_sid` has a recorded event AND it was more than
    `threshold_s` seconds ago. Sessions never seen are NOT stalled
    (no false positive on cold sessions)."""
    if not proj_sid:
        return False
    entry = _LAST_EVENT.get(proj_sid)
    if entry is None:
        return False
    return (time.monotonic() - entry["ts"]) > threshold_s


def seconds_since_event(proj_sid: str) -> float | None:
    """Elapsed seconds since the last event for this session; None if
    no event recorded. For dashboard display."""
    entry = _LAST_EVENT.get(proj_sid)
    if entry is None:
        return None
    return time.monotonic() - entry["ts"]


def get_conv_sid(proj_sid: str) -> str | None:
    """Return the conv_sid associated with `proj_sid`'s last event,
    or None if either no event was recorded or the caller never supplied
    a conv_sid. Used by the stall callback to scope escalation."""
    entry = _LAST_EVENT.get(proj_sid)
    if entry is None:
        return None
    return entry.get("conv_sid")


def clear(proj_sid: str) -> None:
    """Drop tracking for `proj_sid`. Called when the stream legitimately
    ends (success or expected error path) so the next poll doesn't see
    a stale entry and fire a spurious stall."""
    if not proj_sid:
        return
    _LAST_EVENT.pop(proj_sid, None)


# ─── background loop ──────────────────────────────────────────────────────


StallCallback = Callable[..., Union[None, Awaitable[None]]]


def start_watchdog(check_interval_s: float,
                   threshold_s: float,
                   on_stall: StallCallback,
                   ) -> asyncio.Task:
    """Spawn the watchdog as an asyncio.Task. Caller keeps the handle
    so it can `task.cancel()` on shutdown. The task sleeps
    `check_interval_s`, snapshots the live dict's keys, and fires
    `on_stall` for each stalled sid. Sync and async callbacks are both
    supported — the loop awaits coroutine returns and ignores plain ones.

    Per-callback exceptions are caught and logged; one stalled session
    can never crash the loop. After firing, the sid is removed from the
    state dict so the same stall doesn't re-fire on the next iteration.
    """

    # Detect callback arity ONCE at watchdog start. The previous
    # try/except TypeError around the call site would also swallow
    # TypeErrors raised DEEP inside the callback body (e.g. `int(None)`
    # in a log helper) and silently retry with one arg — masking real
    # bugs AND firing side effects twice (the first invocation already
    # partially ran before the deep TypeError bubbled).
    try:
        sig = inspect.signature(on_stall)
        n_params = len([p for p in sig.parameters.values()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        accepts_conv_sid = n_params >= 2
    except (TypeError, ValueError):
        accepts_conv_sid = False

    async def _loop() -> None:
        while True:
            try:
                await asyncio.sleep(check_interval_s)
            except asyncio.CancelledError:
                raise
            try:
                # Snapshot keys to avoid mutation-during-iteration when
                # mark_event / clear fires concurrently.
                sids = list(_LAST_EVENT.keys())
                for sid in sids:
                    try:
                        if not check_stalled(sid, threshold_s):
                            continue
                        # Snapshot conv_sid BEFORE pop so the callback can
                        # scope escalation to one conversation. Pop next
                        # so a callback that touches state doesn't re-fire.
                        entry = _LAST_EVENT.get(sid) or {}
                        conv_sid = entry.get("conv_sid")
                        _LAST_EVENT.pop(sid, None)
                        try:
                            result = (on_stall(sid, conv_sid)
                                      if accepts_conv_sid
                                      else on_stall(sid))
                            if inspect.isawaitable(result):
                                await result
                        except Exception as exc:  # noqa: BLE001
                            _LOG.warning(
                                "stall_watchdog_callback_error sid=%s err=%s",
                                sid, exc)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        _LOG.warning(
                            "stall_watchdog_iter_error sid=%s err=%s",
                            sid, exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("stall_watchdog_loop_error err=%s", exc)

    return asyncio.create_task(_loop())


# ─── inspection helpers ───────────────────────────────────────────────────


def state_snapshot() -> dict[str, dict[str, Any]]:
    """All tracked sessions with elapsed-since-last-event + the stored
    conv_sid (None when caller didn't supply one). For dashboard."""
    now = time.monotonic()
    return {sid: {"seconds_since_event": now - entry["ts"],
                  "conv_sid": entry.get("conv_sid")}
            for sid, entry in _LAST_EVENT.items()}


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that one."""
    if proj_sid is None:
        _LAST_EVENT.clear()
        return
    _LAST_EVENT.pop(proj_sid, None)
