"""Regression + behavior tests for tinyctx.project_store.

The headline test is `test_record_new_cwd_does_not_deadlock`: record() acquires
the per-cwd lock and then, on a cache miss, calls _init_project() which
re-acquires the SAME lock. With a non-reentrant threading.Lock that
self-deadlocked — and because record() runs synchronously on the proxy's
event-loop thread (proxy._record_token_tracker at stream end), a single
first-request-for-a-new-project froze the entire proxy. The lock is now an
RLock; these tests lock that in.
"""
from __future__ import annotations

import threading

import tinyctx.project_store as ps


def _fresh_state(tmp_path):
    """Point the store at a temp dir and clear module caches."""
    ps.STATE_DIR = tmp_path / "projects"
    ps._cache.clear()
    ps._locks.clear()


def test_record_new_cwd_does_not_deadlock(tmp_path):
    _fresh_state(tmp_path)
    done = threading.Event()

    def go():
        # cache miss -> record() -> _init_project() re-enters the per-cwd lock
        ps.record(cwd="/tmp/proj-A", est_input_tokens=100,
                  forwarded_tokens=40, route="local")
        done.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), "record() deadlocked on a new cwd (RLock regression)"


def test_record_accumulates_stats(tmp_path):
    _fresh_state(tmp_path)
    ps.record(cwd="/tmp/proj-B", est_input_tokens=100, forwarded_tokens=30, route="local")
    ps.record(cwd="/tmp/proj-B", est_input_tokens=50, forwarded_tokens=50, route="frontier")
    data = ps._cache[ps._cwd_hash("/tmp/proj-B")]
    tok = data["token"]
    assert tok["requests"] == 2
    assert tok["est_input_tokens"] == 150
    assert tok["forwarded_tokens"] == 80
    assert tok["saved_tokens"] == 70  # (100-30) + max(0, 50-50)
    assert data["by_route"] == {"local": 1, "frontier": 1}


def test_concurrent_record_same_cwd_no_deadlock(tmp_path):
    _fresh_state(tmp_path)
    errors: list[BaseException] = []

    def go():
        try:
            for _ in range(25):
                ps.record(cwd="/tmp/proj-C", est_input_tokens=10,
                          forwarded_tokens=5, route="local")
        except BaseException as e:  # noqa: BLE001 — surface to assertion
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "concurrent record() deadlocked"
    assert not errors, f"record() raised under concurrency: {errors}"
    assert ps._cache[ps._cwd_hash("/tmp/proj-C")]["token"]["requests"] == 8 * 25
