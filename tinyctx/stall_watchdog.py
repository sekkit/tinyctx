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
Per-session last-event timestamps live in `_LAST_EVENT_TS` keyed by
proj_sid. The proxy's SSE relay calls `mark_event(proj_sid)` each time
it sees a parsed upstream event. A background asyncio task polls every
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

_LAST_EVENT_TS: dict[str, float] = {}


# ─── public API ────────────────────────────────────────────────────────────


def mark_event(proj_sid: str) -> None:
    """Record that `proj_sid` just received an upstream event. Called
    from the SSE relay loop on each parsed event. Cheap dict write."""
    if not proj_sid:
        return
    _LAST_EVENT_TS[proj_sid] = time.monotonic()


def check_stalled(proj_sid: str, threshold_s: float) -> bool:
    """True iff `proj_sid` has a recorded event AND it was more than
    `threshold_s` seconds ago. Sessions never seen are NOT stalled
    (no false positive on cold sessions)."""
    if not proj_sid:
        return False
    last = _LAST_EVENT_TS.get(proj_sid)
    if last is None:
        return False
    return (time.monotonic() - last) > threshold_s


def seconds_since_event(proj_sid: str) -> float | None:
    """Elapsed seconds since the last event for this session; None if
    no event recorded. For dashboard display."""
    last = _LAST_EVENT_TS.get(proj_sid)
    if last is None:
        return None
    return time.monotonic() - last


def clear(proj_sid: str) -> None:
    """Drop tracking for `proj_sid`. Called when the stream legitimately
    ends (success or expected error path) so the next poll doesn't see
    a stale entry and fire a spurious stall."""
    if not proj_sid:
        return
    _LAST_EVENT_TS.pop(proj_sid, None)


# ─── background loop ──────────────────────────────────────────────────────


StallCallback = Callable[[str], Union[None, Awaitable[None]]]


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

    async def _loop() -> None:
        while True:
            try:
                await asyncio.sleep(check_interval_s)
            except asyncio.CancelledError:
                raise
            try:
                # Snapshot keys to avoid mutation-during-iteration when
                # mark_event / clear fires concurrently.
                sids = list(_LAST_EVENT_TS.keys())
                for sid in sids:
                    try:
                        if not check_stalled(sid, threshold_s):
                            continue
                        # Remove BEFORE firing so a callback that itself
                        # touches state doesn't re-fire.
                        _LAST_EVENT_TS.pop(sid, None)
                        try:
                            result = on_stall(sid)
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
    """All tracked sessions with elapsed-since-last-event. For dashboard."""
    now = time.monotonic()
    return {sid: {"seconds_since_event": now - last}
            for sid, last in _LAST_EVENT_TS.items()}


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that one."""
    if proj_sid is None:
        _LAST_EVENT_TS.clear()
        return
    _LAST_EVENT_TS.pop(proj_sid, None)
