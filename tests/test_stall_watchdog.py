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
    sw._LAST_EVENT_TS["stale-sid"] = time.monotonic() - 1.0

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
    sw._LAST_EVENT_TS["sidA"] = time.monotonic() - 1.0

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
    sw._LAST_EVENT_TS["boom"] = time.monotonic() - 1.0

    task = sw.start_watchdog(check_interval_s=0.01,
                             threshold_s=0.1,
                             on_stall=_on_stall)
    try:
        await asyncio.sleep(0.1)
        # Loop should still be alive — register another stalled session.
        sw.mark_event("ok")
        sw._LAST_EVENT_TS["ok"] = time.monotonic() - 1.0
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
