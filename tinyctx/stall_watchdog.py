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
Per-session last-event state lives under SessionState namespace
`stall_watchdog` key `last_event` keyed by proj_sid; each entry stores
both the monotonic timestamp and (optionally) the conv_sid the caller
supplied at `mark_event` time so the stall callback can scope force-
frontier escalation to one conversation rather than the whole project.
The proxy's SSE relay calls `mark_event(proj_sid, conv_sid)` each time
it sees a parsed upstream event. A background asyncio task polls every
`check_interval_s` seconds and fires `on_stall(proj_sid)` for any
session whose last event was longer than `threshold_s` ago. After
firing, the session is removed so it doesn't re-fire on the next poll.

`time.monotonic()` is used throughout — wall-clock can jump (NTP, DST,
suspend/resume) and would fire spurious stalls.

Cancel-and-retry (2026-05-11)
─────────────────────────────
The flag-for-next-turn fallback alone is insufficient when codex.app is
itself wedged waiting on `response.completed` — codex won't send a next
turn until the current one closes. To honour the user directive "凡是
中断了都要加重试", `_ACTIVE_TASKS` tracks the in-flight relay producer
task per proj_sid via `register_task` / `unregister_task`. When the
watchdog fires, the stall callback cancels that task — sending
`CancelledError` into its httpx await — which makes the producer push a
synthetic `StallCancelledError` onto its consumer queue. The consumer
then emits a clean SSE terminator (status=incomplete) so codex's SSE
parser sees a structurally valid close and re-prompts; meanwhile the
existing `force_next_to_frontier` flag stays set as belt+suspenders so
the codex-driven follow-up routes to frontier.

State storage
─────────────
P2 migration: `_LAST_EVENT` lives in `tinyctx.session_state` under
namespace `stall_watchdog` key `last_event` (one dict value per
proj_sid). `_ACTIVE_TASKS` stays a module-local dict — asyncio.Task
handles are not serializable state. The legacy `_LAST_EVENT` attribute
is preserved as a `_SessionStateDictView` shim so tests that poke
`_LAST_EVENT[sid]["ts"]` directly continue to work (the shim returns
the underlying dict by reference so in-place mutation propagates).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Iterator, Union

from . import session_state


_LOG = logging.getLogger("tinyctx.stall_watchdog")


# ─── SessionState namespace + compaction reset policy ─────────────────────

_NS = "stall_watchdog"
_K_LAST_EVENT = "last_event"

# Compaction clears stale stall data — a stale `last_event` shouldn't
# carry over across the boundary since the post-compaction request is a
# fresh stream and the watchdog will re-mark it.
session_state.register_compaction_reset(_NS, [_K_LAST_EVENT])


# ─── legacy dict view ──────────────────────────────────────────────────────


class _LastEventDictView:
    """Read/write proxy over SessionState for the legacy `_LAST_EVENT`
    attribute. `view[sid]` returns the underlying entry dict by
    reference so callers (and tests) that mutate `view[sid]["ts"] = ...`
    in place propagate the change back to SessionState storage.
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[str]:
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if _K_LAST_EVENT in by_ns.get(_NS, {}):
                yield sid

    def keys(self) -> list[str]:
        return list(iter(self))

    def __contains__(self, sid: Any) -> bool:
        return session_state.get(sid, _NS, _K_LAST_EVENT) is not None

    def __getitem__(self, sid: Any) -> dict[str, Any]:
        val = session_state.get(sid, _NS, _K_LAST_EVENT)
        if val is None:
            raise KeyError(sid)
        return val  # by-reference: mutation propagates to storage

    def __setitem__(self, sid: Any, value: dict[str, Any]) -> None:
        session_state.set(sid, _NS, _K_LAST_EVENT, value)

    def __delitem__(self, sid: Any) -> None:
        session_state.clear(sid, _NS, _K_LAST_EVENT)

    def get(self, sid: Any, default: Any = None) -> Any:
        val = session_state.get(sid, _NS, _K_LAST_EVENT)
        return default if val is None else val

    def pop(self, sid: Any, default: Any = None) -> Any:
        existing = session_state.consume(sid, _NS, _K_LAST_EVENT)
        return existing if existing is not None else default

    def clear(self) -> None:
        for sid in list(iter(self)):
            session_state.clear(sid, _NS, _K_LAST_EVENT)


# Per-session state: {sid: {"ts": monotonic, "conv_sid": str | None}}.
# `sid` is the proj_sid the watchdog loop iterates over; `conv_sid` is
# the per-conversation key stored by callers that have body access at
# `mark_event` time. The stall callback prefers conv_sid (precise — only
# that conversation's next request gets force-routed) and falls back to
# the iterated sid when conv_sid wasn't supplied.
_LAST_EVENT = _LastEventDictView()


# Per-session in-flight relay task handle. The proxy's `_stream_proxy`
# registers ITS OWN producer task at relay start (via `register_task`)
# and unregisters in `finally`. The stall callback consults this dict
# to actually cancel the wedged upstream task — without it, the watchdog
# can only flag the NEXT turn, leaving the current request hung until
# codex's own idle timeout fires (observed: ~9 minutes).
#
# NOT migrated to SessionState: asyncio.Task is a runtime handle, not
# serializable state. Stays a plain module-level dict.
_ACTIVE_TASKS: dict[str, "asyncio.Task"] = {}


class StallCancelledError(Exception):
    """Synthetic exception the proxy's relay producer pushes onto its
    consumer queue when the stall watchdog cancels the in-flight task.

    Carries enough context for forensics + retry classification:
      - elapsed_silent_s: seconds since last upstream event
      - conv_sid: per-conversation scope key (None for legacy callers)
      - proj_sid: project/session key the cancel was scoped to

    The proxy's `_stream_proxy` catches this in its consumer error
    handler, emits a clean `_terminator_event(status=incomplete)`, and
    sets `force_next_to_frontier` on the conv scope so codex's
    follow-up turn routes to frontier (retry-escalate equivalent).
    """

    def __init__(self, message: str, *,
                 proj_sid: str = "",
                 conv_sid: str | None = None,
                 elapsed_silent_s: float | None = None) -> None:
        super().__init__(message)
        self.proj_sid = proj_sid
        self.conv_sid = conv_sid
        self.elapsed_silent_s = elapsed_silent_s


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
    now = time.monotonic()
    entry = session_state.get(proj_sid, _NS, _K_LAST_EVENT)
    if entry is None:
        session_state.set(proj_sid, _NS, _K_LAST_EVENT,
                          {"ts": now, "conv_sid": conv_sid})
        return
    entry["ts"] = now
    if conv_sid is not None:
        entry["conv_sid"] = conv_sid
    # entry is held by reference inside SessionState — in-place mutation
    # already persisted; no re-set needed.


def check_stalled(proj_sid: str, threshold_s: float) -> bool:
    """True iff `proj_sid` has a recorded event AND it was more than
    `threshold_s` seconds ago. Sessions never seen are NOT stalled
    (no false positive on cold sessions)."""
    if not proj_sid:
        return False
    entry = session_state.get(proj_sid, _NS, _K_LAST_EVENT)
    if entry is None:
        return False
    return (time.monotonic() - entry["ts"]) > threshold_s


def seconds_since_event(proj_sid: str) -> float | None:
    """Elapsed seconds since the last event for this session; None if
    no event recorded. For dashboard display."""
    entry = session_state.get(proj_sid, _NS, _K_LAST_EVENT)
    if entry is None:
        return None
    return time.monotonic() - entry["ts"]


def get_conv_sid(proj_sid: str) -> str | None:
    """Return the conv_sid associated with `proj_sid`'s last event,
    or None if either no event was recorded or the caller never supplied
    a conv_sid. Used by the stall callback to scope escalation."""
    entry = session_state.get(proj_sid, _NS, _K_LAST_EVENT)
    if entry is None:
        return None
    return entry.get("conv_sid")


def clear(proj_sid: str) -> None:
    """Drop tracking for `proj_sid`. Called when the stream legitimately
    ends (success or expected error path) so the next poll doesn't see
    a stale entry and fire a spurious stall."""
    if not proj_sid:
        return
    session_state.clear(proj_sid, _NS, _K_LAST_EVENT)


# ─── active-task registry (cancel-and-retry on stall) ──────────────────────


def register_task(proj_sid: str, task: "asyncio.Task") -> None:
    """Register the in-flight relay producer task for `proj_sid` so the
    stall callback can cancel it when silence exceeds threshold. The
    proxy's `_stream_proxy` calls this at producer-task creation and
    pairs it with `unregister_task` in `finally`. A subsequent
    `register_task` for the same `proj_sid` replaces the previous
    handle without raising — last-writer-wins matches the proxy's
    one-active-stream-per-session invariant.

    No-op when proj_sid is empty or task is None — callers can
    fire-and-forget without guarding."""
    if not proj_sid or task is None:
        return
    _ACTIVE_TASKS[proj_sid] = task


def unregister_task(proj_sid: str, task: "asyncio.Task | None" = None) -> None:
    """Drop the registered task for `proj_sid`. If `task` is supplied,
    only unregister when it matches the current registration — prevents
    a late-finishing producer from clobbering a freshly-registered new
    stream's handle. With `task=None`, unconditionally drop."""
    if not proj_sid:
        return
    if task is None:
        _ACTIVE_TASKS.pop(proj_sid, None)
        return
    cur = _ACTIVE_TASKS.get(proj_sid)
    if cur is task:
        _ACTIVE_TASKS.pop(proj_sid, None)


def get_active_task(proj_sid: str) -> "asyncio.Task | None":
    """Return the registered relay task for `proj_sid`, or None if no
    task is currently registered. For stall callbacks + tests."""
    if not proj_sid:
        return None
    return _ACTIVE_TASKS.get(proj_sid)


def cancel_active_task(proj_sid: str) -> bool:
    """Cancel the registered relay task for `proj_sid` if one is live
    (not already done). Returns True when a cancel was issued, False
    when no live task was registered. Safe to call from any coroutine —
    `task.cancel()` is just a flag flip on the task object; the
    CancelledError fires at the task's next await point.

    The caller does NOT remove the entry from `_ACTIVE_TASKS` — that
    happens via the producer task's own `unregister_task` call in
    `finally`. Removing here would race with a fresh `register_task`."""
    task = _ACTIVE_TASKS.get(proj_sid)
    if task is None or task.done():
        return False
    task.cancel()
    return True


# ─── background loop ──────────────────────────────────────────────────────


# Two callback shapes are accepted; start_watchdog picks the right call form
# via inspect.signature once at startup. The 2-arg form gets conv_sid so
# stall escalation can be scoped to a single conversation; the 1-arg form
# is kept for legacy callers that haven't migrated.
_StallCallback2 = Callable[[str, Union[str, None]], Union[None, Awaitable[None]]]
_StallCallback1 = Callable[[str], Union[None, Awaitable[None]]]
StallCallback = Union[_StallCallback2, _StallCallback1]


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
                        entry = session_state.get(sid, _NS, _K_LAST_EVENT) or {}
                        conv_sid = entry.get("conv_sid")
                        session_state.clear(sid, _NS, _K_LAST_EVENT)
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
    out: dict[str, dict[str, Any]] = {}
    snap = session_state.snapshot()
    for sid, by_ns in snap.items():
        entry = by_ns.get(_NS, {}).get(_K_LAST_EVENT)
        if entry is None:
            continue
        out[sid] = {"seconds_since_event": now - entry["ts"],
                    "conv_sid": entry.get("conv_sid")}
    return out


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that one. Also drops any registered active task handles
    for the affected scope — tests that build a fresh `_ACTIVE_TASKS`
    don't leak stale handles into the next test."""
    if proj_sid is None:
        for sid in list(_LAST_EVENT.keys()):
            session_state.clear(sid, _NS, _K_LAST_EVENT)
        _ACTIVE_TASKS.clear()
        return
    session_state.clear(proj_sid, _NS, _K_LAST_EVENT)
    _ACTIVE_TASKS.pop(proj_sid, None)
