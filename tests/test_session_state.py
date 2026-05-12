"""Unified per-conversation state container.

P1 of the SessionState refactor. The codebase currently has 15+ ad-hoc
`_X_PER_SESSION: dict[str, ...]` dicts scattered across many modules.
`tinyctx.session_state` consolidates them behind a single API keyed by
`conv_sid` (falls back to `proj_sid` when no conv_sid is available).

Test surface mirrors the public API documented in the P1 brief.
"""
from __future__ import annotations

import time

import pytest


# ─── counters ────────────────────────────────────────────────────────────

def test_counter_default_is_zero():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.get("c1", "ns", "k", 0) == 0


def test_counter_increment_returns_new_value():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.increment("c1", "ns", "k") == 1
    assert ss.increment("c1", "ns", "k") == 2
    assert ss.increment("c1", "ns", "k", by=3) == 5


def test_counter_persists_across_get_calls():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.increment("c1", "ns", "k")
    ss.increment("c1", "ns", "k")
    assert ss.get("c1", "ns", "k", 0) == 2
    assert ss.get("c1", "ns", "k", 0) == 2  # idempotent read


def test_counter_isolated_per_conv_sid():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.increment("convA", "ns", "k")
    ss.increment("convA", "ns", "k")
    ss.increment("convB", "ns", "k")
    assert ss.get("convA", "ns", "k", 0) == 2
    assert ss.get("convB", "ns", "k", 0) == 1


# ─── flag (set / peek / consume) ─────────────────────────────────────────

def test_flag_set_then_peek_returns_same_value():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns", "fired", True)
    assert ss.get("c1", "ns", "fired") is True


def test_flag_supports_dict_payload():
    from tinyctx import session_state as ss
    ss.reset_all()
    payload = {"set_at": 12.0, "reason": "test"}
    ss.set("c1", "ns", "force", payload)
    assert ss.get("c1", "ns", "force") == payload


def test_flag_consume_returns_then_clears():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns", "flag", "value")
    assert ss.consume("c1", "ns", "flag") == "value"
    assert ss.get("c1", "ns", "flag") is None


def test_flag_consume_missing_returns_none():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.consume("c1", "ns", "missing") is None


def test_get_default_for_missing():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.get("c1", "ns", "missing") is None
    assert ss.get("c1", "ns", "missing", "fallback") == "fallback"


# ─── timestamp ───────────────────────────────────────────────────────────

def test_mark_timestamp_then_seconds_since_is_small():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.mark_timestamp("c1", "ns", "evt")
    elapsed = ss.seconds_since("c1", "ns", "evt")
    assert elapsed is not None
    assert 0.0 <= elapsed < 1.0


def test_seconds_since_unset_returns_none():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.seconds_since("c1", "ns", "never_set") is None


def test_mark_timestamp_advances_with_time():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.mark_timestamp("c1", "ns", "evt")
    time.sleep(0.02)
    elapsed = ss.seconds_since("c1", "ns", "evt")
    assert elapsed >= 0.02


# ─── bounded history ─────────────────────────────────────────────────────

def test_append_bounded_preserves_order_within_maxlen():
    from tinyctx import session_state as ss
    ss.reset_all()
    for v in [1, 2, 3]:
        ss.append_bounded("c1", "ns", "hist", v, maxlen=5)
    assert ss.get_history("c1", "ns", "hist") == [1, 2, 3]


def test_append_bounded_enforces_maxlen():
    from tinyctx import session_state as ss
    ss.reset_all()
    for v in range(10):
        ss.append_bounded("c1", "ns", "hist", v, maxlen=3)
    assert ss.get_history("c1", "ns", "hist") == [7, 8, 9]


def test_get_history_default_when_empty():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.get_history("c1", "ns", "never") == []


# ─── namespace isolation ─────────────────────────────────────────────────

def test_same_key_under_different_namespaces_do_not_collide():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns_a", "k", "from_a")
    ss.set("c1", "ns_b", "k", "from_b")
    assert ss.get("c1", "ns_a", "k") == "from_a"
    assert ss.get("c1", "ns_b", "k") == "from_b"


def test_increment_namespace_isolated():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.increment("c1", "ns_a", "k")
    ss.increment("c1", "ns_a", "k")
    ss.increment("c1", "ns_b", "k")
    assert ss.get("c1", "ns_a", "k", 0) == 2
    assert ss.get("c1", "ns_b", "k", 0) == 1


# ─── reset hooks ─────────────────────────────────────────────────────────

def test_register_compaction_reset_and_reset_compaction_clears_only_listed():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.register_compaction_reset("mymod", ["scoped_a", "scoped_b"])
    ss.set("c1", "mymod", "scoped_a", 5)
    ss.set("c1", "mymod", "scoped_b", "x")
    ss.set("c1", "mymod", "persistent", "keep_me")
    ss.reset_compaction("c1")
    assert ss.get("c1", "mymod", "scoped_a") is None
    assert ss.get("c1", "mymod", "scoped_b") is None
    assert ss.get("c1", "mymod", "persistent") == "keep_me"


def test_reset_compaction_isolated_per_conv_sid():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.register_compaction_reset("mymod", ["scoped"])
    ss.set("convA", "mymod", "scoped", 1)
    ss.set("convB", "mymod", "scoped", 2)
    ss.reset_compaction("convA")
    assert ss.get("convA", "mymod", "scoped") is None
    assert ss.get("convB", "mymod", "scoped") == 2


def test_reset_compaction_with_falsy_conv_sid_is_noop():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.reset_compaction(None)
    ss.reset_compaction("")
    # No exception


def test_reset_all_full_wipe():
    from tinyctx import session_state as ss
    ss.set("c1", "ns", "k", 1)
    ss.set("c2", "ns", "k", 2)
    ss.reset_all()
    assert ss.get("c1", "ns", "k") is None
    assert ss.get("c2", "ns", "k") is None


def test_reset_all_for_single_conv_sid():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns", "k", 1)
    ss.set("c2", "ns", "k", 2)
    ss.reset_all("c1")
    assert ss.get("c1", "ns", "k") is None
    assert ss.get("c2", "ns", "k") == 2


# ─── snapshot ────────────────────────────────────────────────────────────

def test_snapshot_single_conv_returns_namespaced_values():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns_a", "k1", 10)
    ss.set("c1", "ns_a", "k2", "x")
    ss.set("c1", "ns_b", "k3", True)
    snap = ss.snapshot("c1")
    # Snapshot shape: {namespace: {key: value}}
    assert snap["ns_a"] == {"k1": 10, "k2": "x"}
    assert snap["ns_b"] == {"k3": True}


def test_snapshot_all_returns_per_conv():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("c1", "ns", "k", 1)
    ss.set("c2", "ns", "k", 2)
    snap = ss.snapshot()
    assert snap["c1"]["ns"] == {"k": 1}
    assert snap["c2"]["ns"] == {"k": 2}


def test_snapshot_missing_conv_returns_empty_dict():
    from tinyctx import session_state as ss
    ss.reset_all()
    assert ss.snapshot("never_set") == {}


# ─── edge: falsy conv_sid ────────────────────────────────────────────────

def test_falsy_conv_sid_get_returns_default():
    """get/set under empty conv_sid should not crash and should not
    silently pollute global state."""
    from tinyctx import session_state as ss
    ss.reset_all()
    # No write happened — empty conv_sid should read default
    assert ss.get("", "ns", "k", "default") == "default"
    assert ss.get(None, "ns", "k", "default") == "default"  # type: ignore[arg-type]


def test_falsy_conv_sid_set_is_noop():
    from tinyctx import session_state as ss
    ss.reset_all()
    ss.set("", "ns", "k", "v")
    ss.set(None, "ns", "k", "v")  # type: ignore[arg-type]
    # No state recorded under any conv_sid
    assert ss.snapshot() == {}
