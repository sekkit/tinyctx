"""GuardPipeline + 5 wrapper guards (P4 refactor).

Pre-flight guard logic in `tinyctx/proxy.py` is currently a sequence of
inline calls into `empty_response_guard`, `stuck_loop`,
`synthetic_continue`, `soft_completion`, `plan_persistence`. P4 extracts
a priority-ordered `GuardPipeline` so:

- Guard ordering is explicit (priority on each guard class).
- Each guard's effect is captured in a `GuardResult` (uniform shape).
- Adding a new guard doesn't touch the central handler.
- One log emit per pipeline.run() carries the ordered effect list.

The 5 wrappers preserve EXACT existing behavior — they only reshape
where the call happens. Behavior tests live in each underlying module's
suite. These tests cover the pipeline mechanics + the wrapper contracts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ─── pipeline mechanics ──────────────────────────────────────────────────


def test_empty_pipeline_returns_empty_result_list():
    from tinyctx.guards import GuardPipeline, GuardContext
    pipeline = GuardPipeline([])
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    assert pipeline.run(ctx) == []


def test_single_guard_fires_result_captured():
    from tinyctx.guards import GuardPipeline, GuardContext, GuardResult

    class FakeGuard:
        name = "fake"
        priority = 100

        def apply(self, ctx):  # noqa: ANN001
            return GuardResult(guard_name=self.name, fired=True, reason="ok")

    pipeline = GuardPipeline([FakeGuard()])
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    results = pipeline.run(ctx)
    assert len(results) == 1
    assert results[0].guard_name == "fake"
    assert results[0].fired is True
    assert results[0].reason == "ok"


def test_multiple_guards_run_in_priority_order():
    from tinyctx.guards import GuardPipeline, GuardContext, GuardResult

    seen: list[str] = []

    class G:
        def __init__(self, name: str, priority: int):
            self.name = name
            self.priority = priority

        def apply(self, ctx):  # noqa: ANN001
            seen.append(self.name)
            return GuardResult(guard_name=self.name, fired=False)

    # Pass in reverse-priority order to ensure the pipeline re-sorts.
    pipeline = GuardPipeline([G("c", 30), G("a", 10), G("b", 20)])
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    pipeline.run(ctx)
    assert seen == ["a", "b", "c"]


def test_higher_priority_guard_mutations_visible_to_later_guards():
    """A guard sets ctx.force_route; the next-priority guard sees it."""
    from tinyctx.guards import GuardPipeline, GuardContext, GuardResult

    seen_force_route: list[str | None] = []

    class SetForceFrontier:
        name = "set"
        priority = 10

        def apply(self, ctx):  # noqa: ANN001
            ctx.force_route = "frontier"
            return GuardResult(guard_name=self.name, fired=True,
                               force_route="frontier")

    class Observer:
        name = "observer"
        priority = 20

        def apply(self, ctx):  # noqa: ANN001
            seen_force_route.append(ctx.force_route)
            return GuardResult(guard_name=self.name, fired=True)

    pipeline = GuardPipeline([Observer(), SetForceFrontier()])
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    pipeline.run(ctx)
    assert seen_force_route == ["frontier"]


def test_guard_exception_does_not_break_pipeline():
    from tinyctx.guards import GuardPipeline, GuardContext, GuardResult

    class Boom:
        name = "boom"
        priority = 10

        def apply(self, ctx):  # noqa: ANN001
            raise RuntimeError("kaboom")

    class Good:
        name = "good"
        priority = 20

        def apply(self, ctx):  # noqa: ANN001
            return GuardResult(guard_name=self.name, fired=True, reason="ok")

    pipeline = GuardPipeline([Boom(), Good()])
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    results = pipeline.run(ctx)
    assert len(results) == 2
    # Boom result captured with exception info; fired=False.
    assert results[0].guard_name == "boom"
    assert results[0].fired is False
    assert "kaboom" in results[0].reason
    assert results[0].additional_log.get("exception_type") == "RuntimeError"
    # Good still ran.
    assert results[1].guard_name == "good"
    assert results[1].fired is True


def test_guard_result_default_shape():
    """GuardResult has sensible defaults."""
    from tinyctx.guards import GuardResult
    r = GuardResult(guard_name="x", fired=False)
    assert r.reason == ""
    assert r.body_mutated is False
    assert r.force_route is None
    assert r.additional_log == {}


# ─── ForceFrontierGuard ──────────────────────────────────────────────────


def test_force_frontier_guard_fires_when_flag_set():
    from tinyctx import empty_response_guard as _erg
    from tinyctx.guards import (ForceFrontierGuard, GuardContext)
    _erg.reset_state()
    _erg.force_next_to_frontier("c1", reason="empty-stream")
    g = ForceFrontierGuard()
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=3)
    r = g.apply(ctx)
    assert r.fired is True
    assert r.force_route == "frontier"
    assert ctx.force_route == "frontier"
    assert "empty-stream" in r.reason
    # Flag is consumed.
    assert _erg.peek_force_frontier("c1") is None


def test_force_frontier_guard_falls_back_to_proj_sid():
    """When conv_sid has no flag but proj_sid does (set by mid-stream
    stall escalation), the proj-scoped flag should still trigger."""
    from tinyctx import empty_response_guard as _erg
    from tinyctx.guards import (ForceFrontierGuard, GuardContext)
    _erg.reset_state()
    _erg.force_next_to_frontier("p1", reason="upstream-error")
    g = ForceFrontierGuard()
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    r = g.apply(ctx)
    assert r.fired is True
    assert ctx.force_route == "frontier"


def test_force_frontier_guard_no_op_when_flag_unset():
    from tinyctx import empty_response_guard as _erg
    from tinyctx.guards import (ForceFrontierGuard, GuardContext)
    _erg.reset_state()
    g = ForceFrontierGuard()
    ctx = GuardContext(body={}, proj_sid="p1", conv_sid="c1", turn_count=0)
    r = g.apply(ctx)
    assert r.fired is False
    assert ctx.force_route is None


# ─── BudgetReminderGuard ─────────────────────────────────────────────────


def test_budget_reminder_guard_fires_when_over_budget():
    from tinyctx import synthetic_continue as _syn
    from tinyctx.guards import BudgetReminderGuard, GuardContext
    _syn.reset_state()
    # Bump injection_count above the budget.
    for _ in range(6):
        _syn.build_continue_injection("c1", max_injections=5)
    body = {"input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}]}
    g = BudgetReminderGuard(max_injections=5)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is True
    assert r.body_mutated is True
    # Body in ctx was replaced with reminder-appended copy.
    assert len(ctx.body["input"]) == 2


def test_budget_reminder_guard_skipped_when_compaction():
    from tinyctx import synthetic_continue as _syn
    from tinyctx.guards import BudgetReminderGuard, GuardContext
    _syn.reset_state()
    for _ in range(6):
        _syn.build_continue_injection("c1", max_injections=5)
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = BudgetReminderGuard(max_injections=5)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=True, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is False


def test_budget_reminder_guard_skipped_when_forced_by_client_model():
    from tinyctx import synthetic_continue as _syn
    from tinyctx.guards import BudgetReminderGuard, GuardContext
    _syn.reset_state()
    for _ in range(6):
        _syn.build_continue_injection("c1", max_injections=5)
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = BudgetReminderGuard(max_injections=5)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=True)
    r = g.apply(ctx)
    assert r.fired is False


def test_budget_reminder_guard_no_op_when_under_budget():
    from tinyctx import synthetic_continue as _syn
    from tinyctx.guards import BudgetReminderGuard, GuardContext
    _syn.reset_state()
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = BudgetReminderGuard(max_injections=5)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is False
    assert r.body_mutated is False


# ─── StuckLoopGuard ──────────────────────────────────────────────────────


def test_stuck_loop_guard_fires_at_trigger():
    from tinyctx import stuck_loop
    from tinyctx.guards import StuckLoopGuard, GuardContext
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "go"}]}
    g = StuckLoopGuard(turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=80,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is True
    assert r.body_mutated is True
    assert len(ctx.body["input"]) == 2


def test_stuck_loop_guard_skipped_when_compaction():
    from tinyctx import stuck_loop
    from tinyctx.guards import StuckLoopGuard, GuardContext
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "go"}]}
    g = StuckLoopGuard(turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=100,
                       is_compaction=True, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is False


def test_stuck_loop_guard_skipped_when_forced_by_client_model():
    from tinyctx import stuck_loop
    from tinyctx.guards import StuckLoopGuard, GuardContext
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "go"}]}
    g = StuckLoopGuard(turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=100,
                       is_compaction=False, forced_by_client_model=True)
    r = g.apply(ctx)
    assert r.fired is False


def test_stuck_loop_guard_uses_proj_sid_for_advisor_scope():
    """Verify the guard passes proj_sid as advisor_scope_sid so advisor
    activity in any sub-thread quiets nudges across the project."""
    from tinyctx import stuck_loop
    from tinyctx.guards import StuckLoopGuard, GuardContext
    stuck_loop.reset_state()
    # Advisor was just called for proj_sid.
    stuck_loop.mark_advisor_call("p1")
    body = {"input": [{"role": "user", "content": "go"}]}
    g = StuckLoopGuard(turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=100,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    # Grace window suppresses reminder even though turn_count >= trigger.
    assert r.fired is False


# ─── SoftCompletionGate ──────────────────────────────────────────────────


def test_soft_completion_gate_fires_when_flag_set():
    from tinyctx import soft_completion
    from tinyctx.guards import SoftCompletionGate, GuardContext
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="question_mark_punt")
    body = {"input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}]}
    g = SoftCompletionGate()
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is True
    assert r.body_mutated is True
    assert len(ctx.body["input"]) == 2
    assert r.additional_log.get("pattern")


def test_soft_completion_gate_skipped_when_compaction():
    from tinyctx import soft_completion
    from tinyctx.guards import SoftCompletionGate, GuardContext
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="x")
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = SoftCompletionGate()
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=True, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is False


def test_soft_completion_gate_skipped_when_forced_by_client_model():
    from tinyctx import soft_completion
    from tinyctx.guards import SoftCompletionGate, GuardContext
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="x")
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = SoftCompletionGate()
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=True)
    r = g.apply(ctx)
    assert r.fired is False


def test_soft_completion_gate_no_op_when_flag_unset():
    from tinyctx import soft_completion
    from tinyctx.guards import SoftCompletionGate, GuardContext
    soft_completion.reset_state()
    body = {"input": [{"role": "user", "content": "hi"}]}
    g = SoftCompletionGate()
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=2,
                       is_compaction=False, forced_by_client_model=False)
    r = g.apply(ctx)
    assert r.fired is False


# ─── PlanPersistenceInjector ─────────────────────────────────────────────


def test_plan_persistence_injector_saves_when_plan_in_body(tmp_path: Path):
    from tinyctx.guards import PlanPersistenceInjector, GuardContext
    # Body includes an update_plan call so extract_plan_text returns text.
    body = {
        "input": [
            {"type": "function_call", "name": "update_plan",
             "call_id": "c1",
             "arguments": '{"plan": [{"step": "do thing", "status": "pending"}]}'},
        ],
    }
    g = PlanPersistenceInjector(state_dir=tmp_path,
                                 cwd="/some/cwd", plan_ttl_s=3600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=5)
    r = g.apply(ctx)
    # Saved. Either fired=True if injection also happened on a fresh
    # thread, or fired=False but the on-disk file exists.
    saved_path = tmp_path / "plans"
    assert saved_path.exists()


def test_plan_persistence_injector_injects_on_fresh_thread(tmp_path: Path):
    from tinyctx import plan_persistence
    from tinyctx.guards import PlanPersistenceInjector, GuardContext
    # Seed a saved plan on disk for the cwd.
    plan_persistence.save_plan(tmp_path, "/repo/x",
                                "1. step a (pending)\n2. step b (pending)",
                                session_id="prev", turn_count=42)
    body = {"input": [], "instructions": "system prompt"}
    g = PlanPersistenceInjector(state_dir=tmp_path,
                                 cwd="/repo/x", plan_ttl_s=3600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=0)
    r = g.apply(ctx)
    assert r.fired is True
    assert r.body_mutated is True
    assert "<persisted-plan" in ctx.body["instructions"]


def test_plan_persistence_injector_no_op_when_not_fresh_thread(
        tmp_path: Path):
    from tinyctx import plan_persistence
    from tinyctx.guards import PlanPersistenceInjector, GuardContext
    plan_persistence.save_plan(tmp_path, "/repo/x",
                                "1. step a (pending)",
                                session_id="prev", turn_count=42)
    body = {"input": [], "instructions": "system prompt"}
    g = PlanPersistenceInjector(state_dir=tmp_path,
                                 cwd="/repo/x", plan_ttl_s=3600.0)
    ctx = GuardContext(body=body, proj_sid="p1", conv_sid="c1", turn_count=5)
    r = g.apply(ctx)
    # turn_count != 0 → no inject (the save path may still run, that's fine).
    assert r.body_mutated is False
    assert "<persisted-plan" not in (ctx.body.get("instructions") or "")
