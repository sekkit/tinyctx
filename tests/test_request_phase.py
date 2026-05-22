"""Per-request lifecycle phase tracking (P3). Covers set/get/snapshot/
reset, multi-session isolation, transition timing, and the str-Enum
JSON-serializability contract."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tinyctx import request_phase as _rp
    _rp.reset_state()
    yield
    _rp.reset_state()


def test_set_and_get_phase_round_trip():
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase

    set_phase("sid-1", RequestPhase.received, request_id="rq_abc")
    info = get_phase("sid-1")
    assert info is not None
    assert info["phase"] == "received"
    assert info["request_id"] == "rq_abc"
    assert isinstance(info["since_ts"], float)
    assert info["since_ts"] <= time.time() + 0.001


def test_get_phase_unknown_session_returns_none():
    from tinyctx.request_phase import get_phase
    assert get_phase("never-set") is None


def test_set_phase_with_string_value_also_works():
    """set_phase tolerates plain str (not just the enum) since proxy
    callers might pass a value directly in error paths."""
    from tinyctx.request_phase import set_phase, get_phase
    set_phase("sid-2", "stalled")
    assert get_phase("sid-2")["phase"] == "stalled"


def test_set_phase_empty_proj_sid_is_noop():
    from tinyctx.request_phase import RequestPhase, set_phase, state_snapshot
    set_phase("", RequestPhase.received)
    assert state_snapshot() == {}


def test_phase_transition_replaces_previous_entry():
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase

    set_phase("sid-3", RequestPhase.received, "rq_1")
    first_ts = get_phase("sid-3")["since_ts"]
    time.sleep(0.01)
    set_phase("sid-3", RequestPhase.routing, "rq_1")
    second = get_phase("sid-3")
    assert second["phase"] == "routing"
    assert second["since_ts"] > first_ts


def test_multi_session_isolation():
    """Two distinct proj_sids each carry their own current phase."""
    from tinyctx.request_phase import RequestPhase, set_phase, state_snapshot

    set_phase("sid-A", RequestPhase.received, "rq_A")
    set_phase("sid-B", RequestPhase.backend_streaming, "rq_B")

    snap = state_snapshot()
    assert set(snap.keys()) == {"sid-A", "sid-B"}
    assert snap["sid-A"]["phase"] == "received"
    assert snap["sid-B"]["phase"] == "backend_streaming"
    assert snap["sid-A"]["request_id"] == "rq_A"
    assert snap["sid-B"]["request_id"] == "rq_B"


def test_state_snapshot_returns_independent_copies():
    """Mutating snapshot dicts must not corrupt internal state."""
    from tinyctx.request_phase import RequestPhase, set_phase, state_snapshot, get_phase

    set_phase("sid-snap", RequestPhase.classifying, "rq_x")
    snap = state_snapshot()
    snap["sid-snap"]["phase"] = "tampered"
    snap["sid-snap"]["request_id"] = "tampered"

    fresh = get_phase("sid-snap")
    assert fresh["phase"] == "classifying"
    assert fresh["request_id"] == "rq_x"


def test_reset_state_per_session_only_clears_target():
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase, reset_state

    set_phase("sid-x", RequestPhase.received)
    set_phase("sid-y", RequestPhase.routing)
    reset_state("sid-x")
    assert get_phase("sid-x") is None
    assert get_phase("sid-y") is not None


def test_reset_state_global_clears_all():
    from tinyctx.request_phase import RequestPhase, set_phase, state_snapshot, reset_state

    set_phase("sid-1", RequestPhase.received)
    set_phase("sid-2", RequestPhase.routing)
    reset_state()
    assert state_snapshot() == {}


def test_request_phase_enum_values_match_spec():
    """The phase set is part of the dashboard's public contract — locking
    the value strings prevents silent breakage when external monitors
    parse /api/v1/state."""
    from tinyctx.request_phase import RequestPhase
    expected = {
        "received", "classifying", "routing", "backend_streaming",
        "post_stream_classifying", "injecting", "done", "stalled",
        "retrying", "escalated_to_frontier", "empty_guarded",
        "compacting",
    }
    assert {p.value for p in RequestPhase} == expected


def test_request_phase_is_json_serializable():
    """str-Enum subclass means json.dumps works without a custom encoder."""
    from tinyctx.request_phase import RequestPhase
    out = json.dumps({"phase": RequestPhase.received})
    assert json.loads(out) == {"phase": "received"}


def test_transition_timing_records_since_ts_per_transition():
    """Each set_phase resets since_ts — operators read this to compute
    the age of the current phase (e.g. backend_streaming for >60s)."""
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase

    t0 = time.time()
    set_phase("sid-time", RequestPhase.received)
    info = get_phase("sid-time")
    assert info["since_ts"] >= t0 - 0.01
    assert info["since_ts"] <= time.time() + 0.01

    time.sleep(0.05)
    set_phase("sid-time", RequestPhase.backend_streaming)
    info2 = get_phase("sid-time")
    assert info2["since_ts"] - info["since_ts"] >= 0.04


# ─── P3 SessionState integration ──────────────────────────────────────────


def test_session_state_stores_phase_under_request_phase_namespace():
    """P3: phase entry lives in SessionState ns=request_phase, key=current.
    set_phase / get_phase are thin wrappers over `session_state.set/get`."""
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase
    from tinyctx import session_state as ss

    set_phase("sid-ns", RequestPhase.backend_streaming, "rq_ns")
    raw = ss.get("sid-ns", "request_phase", "current")
    assert raw is not None
    assert raw["phase"] == "backend_streaming"
    assert raw["request_id"] == "rq_ns"
    # Public getter mirrors what's in SessionState.
    assert get_phase("sid-ns")["phase"] == raw["phase"]


def test_phase_cleared_on_compaction():
    """P3: `current` is registered for compaction reset — a post-compaction
    request flow is logically fresh and the stale phase badge would
    mislead the dashboard."""
    from tinyctx.request_phase import RequestPhase, set_phase, get_phase
    from tinyctx import session_state as ss

    set_phase("sid-cp", RequestPhase.backend_streaming, "rq_cp")
    assert get_phase("sid-cp") is not None
    ss.reset_compaction("sid-cp")
    assert get_phase("sid-cp") is None
