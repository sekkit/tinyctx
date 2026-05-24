"""TDD tests for graduated escalation ladder.

The ladder prevents the binary "local fails once → jump to dead frontier"
stall seen in the Battle City session. Each level changes strategy before
escalating further.
"""
from __future__ import annotations

from tinyctx.escalation import (
    EscalationLevel,
    EscalationResult,
    evaluate_escalation,
    record_outcome,
    reset_session,
)


def test_normal_below_threshold():
    """0-2 failures → L0 normal, no reminder, no force_route."""
    result = evaluate_escalation(consecutive_failures=0, pivot_count=0)
    assert result.level == EscalationLevel.NORMAL
    assert result.force_route is None
    assert result.reminder is None


def test_two_failures_still_normal():
    result = evaluate_escalation(consecutive_failures=2, pivot_count=0)
    assert result.level == EscalationLevel.NORMAL


def test_three_failures_triggers_refine():
    """3 consecutive failures → L1 REFINE: strategy-change reminder, no force."""
    result = evaluate_escalation(consecutive_failures=3, pivot_count=0)
    assert result.level == EscalationLevel.REFINE
    assert result.force_route is None
    assert result.reminder is not None
    assert "strategy" in result.reminder.lower() or "策略" in result.reminder


def test_five_failures_triggers_pivot():
    """5 consecutive failures → L2 PIVOT: force frontier + pivot reminder."""
    result = evaluate_escalation(consecutive_failures=5, pivot_count=0)
    assert result.level == EscalationLevel.PIVOT
    assert result.force_route == "frontier"
    assert result.reminder is not None


def test_seven_failures_still_pivot():
    """7 failures with 0 pivots → still PIVOT (same level, pivots don't
    increment until the pivot action itself fires)."""
    result = evaluate_escalation(consecutive_failures=7, pivot_count=0)
    assert result.level == EscalationLevel.PIVOT


def test_two_pivots_triggers_search():
    """2 PIVOT escalations without a keep → L3 SEARCH."""
    result = evaluate_escalation(consecutive_failures=10, pivot_count=2)
    assert result.level == EscalationLevel.SEARCH
    assert result.force_route == "frontier"
    assert result.reminder is not None
    assert "search" in result.reminder.lower() or "搜索" in result.reminder


def test_three_pivots_triggers_blocker():
    """3 PIVOT escalations without a keep → L4 BLOCKER: handoff to human."""
    result = evaluate_escalation(consecutive_failures=15, pivot_count=3)
    assert result.level == EscalationLevel.BLOCKER
    assert result.force_route == "frontier"
    assert result.reminder is not None


def test_keep_resets_everything():
    """A successful turn resets consecutive_failures and pivot_count to 0."""
    # Simulate: 4 failures → still at REFINE
    r1 = evaluate_escalation(consecutive_failures=4, pivot_count=0)
    assert r1.level == EscalationLevel.REFINE
    # Then a keep happens: counters reset to 0
    r2 = evaluate_escalation(consecutive_failures=0, pivot_count=0)
    assert r2.level == EscalationLevel.NORMAL


def test_level_does_not_regress_within_same_cycle():
    """Once we hit PIVOT, more failures at same pivot_count stay at PIVOT,
    not back to REFINE."""
    r = evaluate_escalation(consecutive_failures=8, pivot_count=1)
    assert r.level == EscalationLevel.PIVOT


def test_refine_reminder_is_chinese():
    """The REFINE reminder should be in Chinese (matching tinyctx's locale)."""
    result = evaluate_escalation(consecutive_failures=3, pivot_count=0)
    assert len(result.reminder or "") > 20  # substantive, not empty


def test_failure_count_boundary():
    """Exactly at threshold triggers; one below does not."""
    assert evaluate_escalation(2, 0).level == EscalationLevel.NORMAL
    assert evaluate_escalation(3, 0).level == EscalationLevel.REFINE
    assert evaluate_escalation(4, 0).level == EscalationLevel.REFINE
    assert evaluate_escalation(5, 0).level == EscalationLevel.PIVOT


# ─── session_state integration ─────────────────────────────────────────


def test_record_outcome_increments_failures():
    """record_outcome with ok=False increments consecutive_failures."""
    sid = "test_session_1"
    reset_session(sid)
    # 2 failures
    record_outcome(sid, ok=False)
    record_outcome(sid, ok=False)
    from tinyctx import session_state
    assert session_state.get(sid, "escalation", "consecutive_failures", 0) == 2
    assert session_state.get(sid, "escalation", "pivot_count", 0) == 0
    reset_session(sid)


def test_record_outcome_resets_on_keep():
    """A keep resets consecutive_failures and pivot_count."""
    sid = "test_session_2"
    reset_session(sid)
    for _ in range(4):
        record_outcome(sid, ok=False)
    from tinyctx import session_state
    assert session_state.get(sid, "escalation", "consecutive_failures", 0) == 4
    # Now a keep
    record_outcome(sid, ok=True)
    assert session_state.get(sid, "escalation", "consecutive_failures", 0) == 0
    assert session_state.get(sid, "escalation", "pivot_count", 0) == 0
    reset_session(sid)


def test_record_outcome_increments_pivot_at_boundary():
    """When the level reaches PIVOT, pivot_count increments."""
    sid = "test_session_3"
    reset_session(sid)
    # 5 failures → PIVOT
    for _ in range(5):
        record_outcome(sid, ok=False)
    from tinyctx import session_state
    assert session_state.get(sid, "escalation", "consecutive_failures", 0) == 5
    assert session_state.get(sid, "escalation", "pivot_count", 0) == 1
    reset_session(sid)


def test_record_outcome_ok_true_clears_level():
    """After a keep, last_level resets to normal."""
    sid = "test_session_4"
    reset_session(sid)
    for _ in range(3):
        record_outcome(sid, ok=False)
    from tinyctx import session_state
    assert session_state.get(sid, "escalation", "last_level", "") == "refine"
    record_outcome(sid, ok=True)
    assert session_state.get(sid, "escalation", "last_level", "") == "normal"
    reset_session(sid)


def test_evaluate_escalation_result_is_immutable():
    """EscalationResult should be frozen/hashable."""
    r1 = evaluate_escalation(3, 0)
    r2 = evaluate_escalation(3, 0)
    assert r1 == r2
    assert hash(r1) == hash(r2)
