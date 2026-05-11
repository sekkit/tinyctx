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


# ─── multi-conversation isolation (conv_sid scoping) ─────────────────────


def test_new_conversation_not_blocked_by_old_reminder_turn():
    """Critical regression: previous conversation marked
    `_LAST_REMINDER_TURN=175`; a new conversation starting at turn 0
    must NOT be blocked by `0 - 175 = -175 < gap`. Caller scopes the
    gate by per-conversation key so each conversation has its own
    independent counter."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}
    old_conv = "proj:old-uuid"
    new_conv = "proj:new-uuid"
    # Old conversation fired a reminder at turn 175
    _, inj_old = stuck_loop.maybe_inject_stuck_reminder(
        body, old_conv, turn_count=175, turn_trigger=80, turn_gap=50)
    assert inj_old is True
    # New conversation at turn 80 (its own fresh count) → fires
    _, inj_new = stuck_loop.maybe_inject_stuck_reminder(
        body, new_conv, turn_count=80, turn_trigger=80, turn_gap=50)
    assert inj_new is True


def test_per_conversation_gate_isolation():
    """Two conversations in same project: reminder fired in conv A at
    turn 100, conv B can still fire at its own turn 100."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}
    proj = "p"
    conv_a = f"{proj}:aaa"
    conv_b = f"{proj}:bbb"
    _, inj_a = stuck_loop.maybe_inject_stuck_reminder(
        body, conv_a, turn_count=100, turn_trigger=80, turn_gap=50)
    assert inj_a is True
    _, inj_b = stuck_loop.maybe_inject_stuck_reminder(
        body, conv_b, turn_count=100, turn_trigger=80, turn_gap=50)
    assert inj_b is True


def test_advisor_scope_decouples_from_reminder_gate():
    """When `advisor_scope_sid` differs from `proj_sid`, advisor grace
    looks up under the broader project-level key while the reminder
    gate stays per-conversation. Advisor activity in any sub-thread
    quiets nudges across all conversations in the project."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    proj = "p"
    conv = f"{proj}:aaa"
    body = {"input": [{"role": "user", "content": "x"}]}
    # Mark advisor under project-level key (where proxy.py marks it)
    stuck_loop.mark_advisor_call(proj)
    # Conversation-scoped reminder check defers to project-level advisor
    _, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, conv, turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0,
        advisor_scope_sid=proj)
    assert injected is False  # advisor recent → grace skips
    # Without override: conv-key has no advisor TS → fires
    _, injected2 = stuck_loop.maybe_inject_stuck_reminder(
        body, conv, turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    # Note: same conv_sid means the gate sees its own previous turn
    # mark? No — first call didn't inject (advisor blocked), so the
    # gate is clean. This call fires.
    assert injected2 is True


def test_back_compat_when_no_advisor_scope_passed():
    """Old call signature (no advisor_scope_sid) must keep working
    — advisor TS looks up under the same key as the reminder gate."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    stuck_loop.mark_advisor_call("p1")
    body = {"input": [{"role": "user", "content": "x"}]}
    out, injected = stuck_loop.maybe_inject_stuck_reminder(
        body, "p1", turn_count=200,
        turn_trigger=80, turn_gap=50, advisor_grace_s=600.0)
    assert injected is False
    assert out is body


# ─── Bug 4: compaction boundary clears reminder-turn baseline ─────────────


def test_reset_compaction_state_clears_last_reminder_turn():
    """After codex compacts, the post-compaction conversation should be
    able to fire a fresh stuck-loop reminder without waiting another
    `turn_gap` turns past the pre-compaction reminder turn."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}
    conv = "p:conv-a"
    _, inj1 = stuck_loop.maybe_inject_stuck_reminder(
        body, conv, turn_count=100, turn_trigger=80, turn_gap=50)
    assert inj1 is True
    # Without reset, a follow-up at turn 110 would be throttled.
    _, inj_pre = stuck_loop.maybe_inject_stuck_reminder(
        body, conv, turn_count=110, turn_trigger=80, turn_gap=50)
    assert inj_pre is False
    # Compaction boundary: clear the per-conv baseline.
    stuck_loop.reset_compaction_state(conv)
    # Now the same turn fires fresh (because the gate's last-reminder
    # baseline was zeroed and 110 >= turn_trigger).
    _, inj_post = stuck_loop.maybe_inject_stuck_reminder(
        body, conv, turn_count=110, turn_trigger=80, turn_gap=50)
    assert inj_post is True


def test_reset_compaction_state_preserves_advisor_grace():
    """Advisor activity remains relevant across compaction — don't lose
    the timestamp, otherwise the post-compaction conversation would
    re-nudge immediately even though the agent JUST consulted advisor."""
    import time
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    conv = "p:conv-a"
    stuck_loop.mark_advisor_call(conv)
    grace_ts_before = stuck_loop._LAST_ADVISOR_TS[conv]
    assert grace_ts_before > 0
    stuck_loop.reset_compaction_state(conv)
    grace_ts_after = stuck_loop._LAST_ADVISOR_TS.get(conv)
    assert grace_ts_after == grace_ts_before


def test_reset_compaction_state_isolated_per_conversation():
    """Compaction in conv A must NOT touch conv B's reminder gate."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    body = {"input": [{"role": "user", "content": "x"}]}
    proj = "p"
    conv_a = f"{proj}:aaa"
    conv_b = f"{proj}:bbb"
    stuck_loop.maybe_inject_stuck_reminder(
        body, conv_a, turn_count=100, turn_trigger=80, turn_gap=50)
    stuck_loop.maybe_inject_stuck_reminder(
        body, conv_b, turn_count=100, turn_trigger=80, turn_gap=50)
    stuck_loop.reset_compaction_state(conv_a)
    assert stuck_loop._LAST_REMINDER_TURN.get(conv_a, 0) == 0
    assert stuck_loop._LAST_REMINDER_TURN.get(conv_b, 0) == 100


def test_reset_compaction_state_handles_none_gracefully():
    """conv_sid may be None/empty — call must be a no-op without raising."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    stuck_loop.reset_compaction_state(None)
    stuck_loop.reset_compaction_state("")


def test_reset_compaction_state_prefix_sweeps_when_proj_sid_supplied():
    """Codex's compaction-handoff request can omit `prompt_cache_key`,
    degrading conv_sid to proj_sid. Normal-turn keys for the same
    project look like `f"{proj_sid}:{cache_key}"` so a bare
    `pop(proj_sid)` would miss them. When caller supplies `proj_sid`,
    sweep every matching key; other projects untouched. Advisor grace
    timestamps stay intact regardless."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    stuck_loop._LAST_REMINDER_TURN["global:abc"] = 50
    stuck_loop._LAST_REMINDER_TURN["global:def"] = 80
    stuck_loop._LAST_REMINDER_TURN["other:xyz"] = 30
    stuck_loop._LAST_ADVISOR_TS["global:abc"] = 12345.0
    stuck_loop.reset_compaction_state("global", proj_sid="global")
    assert "global:abc" not in stuck_loop._LAST_REMINDER_TURN
    assert "global:def" not in stuck_loop._LAST_REMINDER_TURN
    assert stuck_loop._LAST_REMINDER_TURN.get("other:xyz") == 30
    # Advisor grace preserved.
    assert stuck_loop._LAST_ADVISOR_TS.get("global:abc") == 12345.0


def test_reset_compaction_state_back_compat_single_arg():
    """Old callers passing only conv_sid get exact-key pop, no sweep."""
    from tinyctx import stuck_loop
    stuck_loop.reset_state()
    stuck_loop._LAST_REMINDER_TURN["global:abc"] = 50
    stuck_loop._LAST_REMINDER_TURN["global:def"] = 80
    stuck_loop.reset_compaction_state("global:abc")
    assert "global:abc" not in stuck_loop._LAST_REMINDER_TURN
    assert stuck_loop._LAST_REMINDER_TURN.get("global:def") == 80
