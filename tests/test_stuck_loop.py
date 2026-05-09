"""Stuck-loop watchdog: triggers, throttle, advisor grace, isolation.

Live trace 2026-05-10 showed a single codex session running 1323 turns
without convergence. The watchdog injects a `<system-reminder>` into
`body.input` after the agent crosses the turn-count trigger, telling
it to either consult advisor or surface its blocker to the user.
"""
from __future__ import annotations

import time

import pytest


# ─── trigger threshold ──────────────────────────────────────────────────────


def test_no_inject_below_trigger():
    """Below `turn_trigger`, the body comes back untouched."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "hi"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, proj_sid="p1", turn_count=10, turn_trigger=80)
    assert injected is False
    assert out is body  # same object — no copy when no-op
    assert len(out["input"]) == 1


def test_injects_at_trigger():
    """At `turn_trigger`, fires once; body grows by exactly one item."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "go"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, proj_sid="p1", turn_count=80, turn_trigger=80)
    assert injected is True
    assert len(out["input"]) == 2
    # Last item is the reminder, role=user, system-reminder block inside
    last = out["input"][-1]
    assert last["role"] == "user"
    text = last["content"][0]["text"]
    assert "<system-reminder>" in text
    assert "stuck-loop watchdog" in text
    assert "80" in text  # turn_count is interpolated
    assert "spawn_agent(role=\"advisor\"" in text


# ─── throttle / gap ─────────────────────────────────────────────────────────


def test_throttle_between_consecutive_reminders():
    """Two reminders within `turn_gap` of each other → only the first
    fires. The second call sees the throttle and skips."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}

    out1, inj1 = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=100, turn_trigger=80, turn_gap=50)
    assert inj1 is True

    # 30 turns later — still within gap → skip
    out2, inj2 = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=130, turn_trigger=80, turn_gap=50)
    assert inj2 is False
    assert out2 is body


def test_re_injects_after_gap():
    """Past the gap → fires again. Long-stuck sessions get periodic nudges."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}

    _, inj1 = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=100, turn_trigger=80, turn_gap=50)
    assert inj1 is True

    # 50 turns later — past gap → re-fires
    _, inj2 = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=150, turn_trigger=80, turn_gap=50)
    assert inj2 is True


# ─── advisor grace window ──────────────────────────────────────────────────


def test_advisor_call_grace_skips_reminder():
    """If the agent already called advisor recently, watchdog stays
    quiet — the agent is doing the right thing on its own."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    # Mark advisor just now
    stuck_loop.mark_advisor_call("p1")
    body = {"input": [{"role": "user", "content": "x"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    assert injected is False
    assert out is body


def test_advisor_grace_expires():
    """After the grace window, the watchdog can fire again."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    # Mark advisor 700s ago (past 600s grace)
    stuck_loop._LAST_ADVISOR_TS["p1"] = time.time() - 700.0
    body = {"input": [{"role": "user", "content": "x"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    assert injected is True


# ─── per-project isolation ─────────────────────────────────────────────────


def test_state_isolated_by_proj_sid():
    """Two projects with same session id must NOT share state.
    Reminder fired in project A → still fires for project B."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}

    _, injA = stuck_loop.maybe_inject_stuck_reminder(
        body, "projA:global", turn_count=100, turn_trigger=80, turn_gap=50)
    assert injA is True

    # Same turn, different proj_sid → fires independently
    _, injB = stuck_loop.maybe_inject_stuck_reminder(
        body, "projB:global", turn_count=100, turn_trigger=80, turn_gap=50)
    assert injB is True


def test_advisor_grace_isolated_by_proj_sid():
    """Advisor call in project A doesn't suppress reminder in project B."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    stuck_loop.mark_advisor_call("projA:global")
    body = {"input": [{"role": "user", "content": "x"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, "projB:global", turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    assert injected is True


# ─── malformed / edge cases ────────────────────────────────────────────────


def test_no_input_array_skips_silently():
    """Body without an input array → no-op, no exception."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"messages": [{"role": "user", "content": "hi"}]}  # chat-style
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=200, turn_trigger=80)
    assert injected is False


def test_does_not_mutate_input_body():
    """Original body must remain untouched (callers may keep a reference
    to the pre-injection body for trace records / replay)."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    original = {"input": [{"role": "user", "content": "x"}]}
    items_id = id(original["input"])
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        original, "p1", turn_count=200, turn_trigger=80)
    assert injected is True
    # original input list untouched
    assert len(original["input"]) == 1
    assert id(original["input"]) == items_id
    # new body has fresh list
    assert len(out["input"]) == 2
    assert id(out["input"]) != items_id


# ─── default config ────────────────────────────────────────────────────────


def test_config_defaults_sane():
    """Watchdog defaults: enabled, trigger ≥ 50 (don't false-fire),
    grace at least 5 minutes (gives the agent a real chance)."""
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.stuck_loop_watchdog_enabled is True
    assert 50 <= cfg.stuck_loop_turn_trigger <= 200
    assert 20 <= cfg.stuck_loop_turn_gap <= 200
    assert 300 <= cfg.stuck_loop_advisor_grace_s <= 3600


# ─── trace fields ──────────────────────────────────────────────────────────


def test_trace_fields_default_off():
    """Trace defaults: stuck flag false, count zero. Only flips on
    actual injection in proxy.py."""
    from tinyctx.trace import RequestTrace
    t = RequestTrace()
    assert t.stuck_reminder_injected is False
    assert t.stuck_turn_count_at_inject == 0
