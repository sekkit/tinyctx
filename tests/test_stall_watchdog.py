"""Unit tests for tinyctx.stall_watchdog."""
from __future__ import annotations

import asyncio
import time

import pytest

from tinyctx import stall_watchdog as sw


@pytest.fixture(autouse=True)
def _reset_state():
    sw.reset_state()
    yield
    sw.reset_state()


def test_mark_then_short_threshold_is_stalled():
    sw.mark_event("sidA")
    time.sleep(0.005)
    assert sw.check_stalled("sidA", threshold_s=0.001) is True


def test_mark_then_large_threshold_is_not_stalled():
    sw.mark_event("sidA")
    assert sw.check_stalled("sidA", threshold_s=10.0) is False


def test_unknown_sid_is_not_stalled():
    # No mark_event call ever — must NOT report stalled (no false positives).
    assert sw.check_stalled("never-seen", threshold_s=0.0001) is False
    assert sw.seconds_since_event("never-seen") is None


def test_clear_removes_state():
    sw.mark_event("sidA")
    assert sw.seconds_since_event("sidA") is not None
    sw.clear("sidA")
    assert sw.seconds_since_event("sidA") is None
    assert sw.check_stalled("sidA", threshold_s=0.0001) is False


def test_state_snapshot_includes_seconds():
    sw.mark_event("sidA")
    snap = sw.state_snapshot()
    assert "sidA" in snap
    assert "seconds_since_event" in snap["sidA"]
    assert snap["sidA"]["seconds_since_event"] >= 0


def test_empty_proj_sid_is_noop():
    sw.mark_event("")
    assert sw.check_stalled("", threshold_s=0.0001) is False
    sw.clear("")  # must not raise


@pytest.mark.asyncio
async def test_background_task_fires_only_for_stalled_session():
    fired: list[str] = []

    def _on_stall(sid: str) -> None:
        fired.append(sid)

    sw.mark_event("stale-sid")
    sw.mark_event("fresh-sid")
    # Backdate stale-sid so it's older than the threshold even before
    # the watchdog wakes up. fresh-sid keeps current monotonic.
    sw._LAST_EVENT["stale-sid"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.02,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        # Re-mark fresh-sid each iter so it never goes stale.
        for _ in range(8):
            await asyncio.sleep(0.02)
            sw.mark_event("fresh-sid")
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert "stale-sid" in fired
    assert "fresh-sid" not in fired
    # After firing, the sid should be removed (no re-fire on next iter).
    assert fired.count("stale-sid") == 1
    assert sw.seconds_since_event("stale-sid") is None


@pytest.mark.asyncio
async def test_async_callback_is_awaited():
    awaited: list[str] = []

    async def _on_stall(sid: str) -> None:
        await asyncio.sleep(0.005)
        awaited.append(sid)

    sw.mark_event("sidA")
    sw._LAST_EVENT["sidA"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.01,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert awaited == ["sidA"]


@pytest.mark.asyncio
async def test_callback_exception_does_not_crash_loop():
    fired: list[str] = []

    def _on_stall(sid: str) -> None:
        fired.append(sid)
        if sid == "boom":
            raise RuntimeError("intentional")

    sw.mark_event("boom")
    sw._LAST_EVENT["boom"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.01,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.1)
        # Loop should still be alive — register another stalled session.
        sw.mark_event("ok")
        sw._LAST_EVENT["ok"]["ts"] = time.monotonic() - 1.0
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert "boom" in fired
    assert "ok" in fired


@pytest.mark.asyncio
async def test_monotonic_used_via_monkeypatch(monkeypatch):
    """Sanity check: monkeypatching time.monotonic shifts perceived elapsed
    time. Confirms the module reads from time.monotonic, not time.time."""
    fake = [1000.0]

    def _now():
        return fake[0]

    monkeypatch.setattr(sw.time, "monotonic", _now)
    sw.mark_event("sidA")
    assert sw.check_stalled("sidA", threshold_s=10.0) is False
    fake[0] += 60.0
    assert sw.check_stalled("sidA", threshold_s=10.0) is True
    elapsed = sw.seconds_since_event("sidA")
    assert elapsed is not None and elapsed >= 60.0


# ─── Bug 6: conv_sid plumbing for per-conversation escalation ─────────────


def test_mark_event_stores_conv_sid_alongside_ts():
    """When caller supplies conv_sid, both timestamp and conv_sid are
    stored so the stall callback can scope force-frontier to just that
    conversation rather than the whole project."""
    sw.mark_event("projA", conv_sid="projA:conv-1")
    assert sw.get_conv_sid("projA") == "projA:conv-1"


def test_mark_event_without_conv_sid_stores_none():
    """Back-compat: callers that don't supply conv_sid still work; the
    stall callback falls back to the iterated proj_sid."""
    sw.mark_event("projA")
    assert sw.get_conv_sid("projA") is None


def test_mark_event_preserves_previously_stored_conv_sid():
    """If conv_sid was set on an earlier event and a later event omits
    it (e.g. one mark_event call site forgot to pass it), don't clobber
    the previously-known conv_sid back to None."""
    sw.mark_event("projA", conv_sid="projA:conv-1")
    sw.mark_event("projA")  # no conv_sid this time
    assert sw.get_conv_sid("projA") == "projA:conv-1"


def test_mark_event_updates_conv_sid_when_explicit():
    """Explicit conv_sid overrides a previous value — useful if the
    same proj_sid is reused across multiple conversations sequentially."""
    sw.mark_event("projA", conv_sid="projA:conv-1")
    sw.mark_event("projA", conv_sid="projA:conv-2")
    assert sw.get_conv_sid("projA") == "projA:conv-2"


def test_state_snapshot_exposes_conv_sid():
    """Dashboard / inspector helper surfaces conv_sid alongside elapsed."""
    sw.mark_event("projA", conv_sid="projA:conv-x")
    snap = sw.state_snapshot()
    assert snap["projA"]["conv_sid"] == "projA:conv-x"
    assert "seconds_since_event" in snap["projA"]


def test_clear_drops_conv_sid():
    """Clearing removes the entry entirely so neither ts nor conv_sid
    leaks into a fresh future cycle for the same proj_sid."""
    sw.mark_event("projA", conv_sid="projA:conv-1")
    sw.clear("projA")
    assert sw.get_conv_sid("projA") is None
    assert sw.seconds_since_event("projA") is None


@pytest.mark.asyncio
async def test_on_stall_receives_conv_sid_when_known():
    """The watchdog callback gets conv_sid as the 2nd arg when caller
    supplied one — used by proxy.py to scope force-frontier escalation."""
    received: list[tuple[str, str | None]] = []

    def _on_stall(sid: str, conv_sid: str | None = None) -> None:
        received.append((sid, conv_sid))

    sw.mark_event("projA", conv_sid="projA:conv-1")
    sw._LAST_EVENT["projA"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.02,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert ("projA", "projA:conv-1") in received


@pytest.mark.asyncio
async def test_on_stall_falls_back_to_none_when_conv_sid_absent():
    """Callers that never supplied conv_sid (chat-completions path,
    legacy code) still work — callback receives None and falls back to
    proj_sid scoping."""
    received: list[tuple[str, str | None]] = []

    def _on_stall(sid: str, conv_sid: str | None = None) -> None:
        received.append((sid, conv_sid))

    sw.mark_event("projB")  # no conv_sid
    sw._LAST_EVENT["projB"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.02,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert ("projB", None) in received


@pytest.mark.asyncio
async def test_legacy_single_arg_callback_still_works():
    """Back-compat: pre-fix callbacks that only accept sid still get
    called. The watchdog retries with one arg when the two-arg form
    raises TypeError."""
    received: list[str] = []

    def _legacy_on_stall(sid: str) -> None:
        received.append(sid)

    sw.mark_event("projC", conv_sid="projC:conv-z")
    sw._LAST_EVENT["projC"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.02,
                             threshold_s=0.1,
                             on_stall=_legacy_on_stall)
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert received == ["projC"]


@pytest.mark.asyncio
async def test_internal_typeerror_from_two_arg_callback_propagates(caplog):
    """A TypeError raised from DEEP inside a 2-arg callback (e.g. an
    `int(None)` bug in a log helper) must NOT be silently retried as
    the 1-arg form — that masked real bugs AND ran side effects twice.
    Instead it should reach the outer callback-error logger."""
    import logging
    fired: list[tuple[str, str | None]] = []

    def _on_stall(sid: str, conv_sid: str | None = None) -> None:
        fired.append((sid, conv_sid))
        raise TypeError("deep bug — not an arity mismatch")

    sw.mark_event("projX", conv_sid="projX:conv-1")
    sw._LAST_EVENT["projX"]["ts"] = time.monotonic() - 1.0

    with caplog.at_level(logging.WARNING, logger="tinyctx.stall_watchdog"):
        task = sw.start_watchdog(check_interval_s=0.02,
                                 threshold_s=0.1,
                                 on_stall=_on_stall)
        try:
            await asyncio.sleep(0.1)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # Callback fired EXACTLY ONCE — no silent retry.
    assert len(fired) == 1
    assert fired[0] == ("projX", "projX:conv-1")
    # The TypeError reached the outer callback-error logger, not a
    # silent swallow.
    assert any("stall_watchdog_callback_error" in r.message
               and "deep bug" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_two_arg_callback_fires_exactly_once_when_stalled():
    """Regression for the old TypeError fallback firing twice: the
    callback (and any side effects it performs) must run a single time
    per stall event, even when it sets module-level state."""
    fire_count = {"n": 0}

    def _on_stall(sid: str, conv_sid: str | None = None) -> None:
        fire_count["n"] += 1

    sw.mark_event("projY", conv_sid="projY:conv-1")
    sw._LAST_EVENT["projY"]["ts"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.02,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.15)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert fire_count["n"] == 1


def test_stall_escalation_isolates_conversations():
    """Integration scenario: a stall in conv A's stream sets the
    force-frontier flag scoped to conv A, NOT conv B. Conv B's next
    request must not be auto-escalated by conv A's stall."""
    from tinyctx import empty_response_guard as erg
    erg.reset_state()
    sw.mark_event("projA", conv_sid="projA:conv-a")
    sw.mark_event("projA", conv_sid="projA:conv-b")  # last writer wins
    # Simulate _on_stall with the conv-b conv_sid (it would be the
    # currently-active one captured at mark_event time).
    stalled_conv = sw.get_conv_sid("projA")
    assert stalled_conv == "projA:conv-b"
    # The escalation routine in proxy.py uses conv_sid when known.
    erg.force_next_to_frontier(stalled_conv, "mid_stream_stall")
    # Conv-a is NOT escalated
    assert erg.consume_force_frontier("projA:conv-a") is None
    # Conv-b IS escalated (one-shot)
    info = erg.consume_force_frontier("projA:conv-b")
    assert info is not None
    assert "mid_stream_stall" in info["reason"]
