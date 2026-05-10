"""Multi-strategy synthetic continue: rotate through codex builtin
tool calls (shell / local_shell / update_plan) to break the
finish_reason=stop pause loop."""
from __future__ import annotations

import json

import pytest


def test_strategies_have_required_fields():
    from tinyctx.synthetic_continue import STRATEGIES
    assert len(STRATEGIES) >= 3
    for s in STRATEGIES:
        assert "label" in s
        assert "tool_name" in s
        assert "args" in s
        assert isinstance(s["args"], dict)


def test_first_strategy_is_shell_noop():
    """shell/true is the safest non-destructive default."""
    from tinyctx.synthetic_continue import STRATEGIES
    first = STRATEGIES[0]
    assert first["tool_name"] in ("shell", "local_shell")
    cmd = first["args"].get("command", [])
    assert cmd == ["true"]  # POSIX no-op


def test_pick_next_rotates_through_strategies():
    from tinyctx import synthetic_continue
    synthetic_continue.reset_state()
    seen = []
    for _ in range(len(synthetic_continue.STRATEGIES) + 2):
        s = synthetic_continue.pick_next_strategy("p1")
        seen.append(s["label"])
    # First N picks should be the strategies in order
    assert seen[:3] == [s["label"] for s in synthetic_continue.STRATEGIES[:3]]
    # After exhausting, rotates back to first
    assert seen[len(synthetic_continue.STRATEGIES)] == synthetic_continue.STRATEGIES[0]["label"]


def test_per_session_strategy_isolation():
    from tinyctx import synthetic_continue
    synthetic_continue.reset_state()
    a1 = synthetic_continue.pick_next_strategy("projA")
    b1 = synthetic_continue.pick_next_strategy("projB")
    a2 = synthetic_continue.pick_next_strategy("projA")
    # projA's index is independent from projB's
    assert a1["label"] == b1["label"]  # both at index 0
    assert a2["label"] == synthetic_continue.STRATEGIES[1]["label"]


def test_synthetic_tool_call_events_count_and_order():
    from tinyctx.synthetic_continue import synthetic_tool_call_events
    events = synthetic_tool_call_events(
        "shell", {"command": ["true"]})
    assert len(events) == 4
    blob = b"".join(events).decode("utf-8")
    pos1 = blob.find("response.output_item.added")
    pos2 = blob.find("response.function_call_arguments.delta")
    pos3 = blob.find("response.function_call_arguments.done")
    pos4 = blob.find("response.output_item.done")
    assert pos1 < pos2 < pos3 < pos4


def test_synthetic_events_carry_args_and_share_id():
    from tinyctx.synthetic_continue import synthetic_tool_call_events
    events = synthetic_tool_call_events(
        "update_plan",
        {"explanation": "test", "plan": []})
    blob = b"".join(events).decode("utf-8")
    assert "update_plan" in blob
    assert "test" in blob
    # All 4 events should reference the same fc_tinyctx_<hex> id
    import re
    ids = re.findall(r"fc_tinyctx_[0-9a-f]+", blob)
    assert len(set(ids)) == 1, "all 4 events should share one item_id"


def test_build_continue_injection_returns_events_and_strategy():
    from tinyctx import synthetic_continue
    synthetic_continue.reset_state()
    events, strategy = synthetic_continue.build_continue_injection("p1")
    assert len(events) == 4
    assert strategy["label"] == synthetic_continue.STRATEGIES[0]["label"]
    # Args of first strategy embedded in the output
    blob = b"".join(events).decode("utf-8")
    assert strategy["tool_name"] in blob


def test_reset_strategy_index_returns_to_zero():
    from tinyctx import synthetic_continue
    synthetic_continue.reset_state()
    synthetic_continue.pick_next_strategy("p1")
    synthetic_continue.pick_next_strategy("p1")
    snap = synthetic_continue.state_snapshot("p1")
    assert snap["next_strategy_idx"] == 2
    synthetic_continue.reset_strategy_index("p1")
    snap = synthetic_continue.state_snapshot("p1")
    assert snap["next_strategy_idx"] == 0


def test_args_are_well_formed_json():
    """Each strategy's args must serialize cleanly so codex's JSON
    parser doesn't choke."""
    from tinyctx.synthetic_continue import STRATEGIES
    for s in STRATEGIES:
        json.dumps(s["args"])  # raises if not serializable


def test_state_snapshot_lists_strategies():
    from tinyctx.synthetic_continue import state_snapshot, STRATEGIES
    snap = state_snapshot("any")
    assert "available_strategies" in snap
    assert len(snap["available_strategies"]) == len(STRATEGIES)


# ─── P2: per-session injection budget cap ─────────────────────────────────


def test_injection_count_increments_on_build_continue():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    assert _syn.injection_count("p1") == 0
    _syn.build_continue_injection("p1")
    assert _syn.injection_count("p1") == 1
    _syn.build_continue_injection("p1")
    assert _syn.injection_count("p1") == 2


def test_is_over_budget_threshold():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    assert _syn.is_over_budget("p1", max_injections=3) is False
    for _ in range(3):
        _syn.build_continue_injection("p1", max_injections=3)
    assert _syn.is_over_budget("p1", max_injections=3) is True
    assert _syn.injection_count("p1") == 3


def test_build_continue_returns_budget_exhausted_when_over():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    for _ in range(2):
        events, strategy = _syn.build_continue_injection(
            "p1", max_injections=2)
        assert strategy["label"] != "budget_exhausted"
        assert len(events) == 4
    events, strategy = _syn.build_continue_injection(
        "p1", max_injections=2)
    assert strategy["label"] == "budget_exhausted"
    assert events == []
    # Counter should NOT increment past the cap
    assert _syn.injection_count("p1") == 2


def test_build_continue_default_max_is_back_compat():
    """Calling build_continue_injection(proj_sid) with no max_injections
    must still work (default=20)."""
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    events, strategy = _syn.build_continue_injection("p1")
    assert len(events) == 4
    assert strategy["label"] != "budget_exhausted"


def test_reset_state_clears_injection_counter():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    _syn.build_continue_injection("p1", max_injections=10)
    _syn.build_continue_injection("p1", max_injections=10)
    assert _syn.injection_count("p1") == 2
    _syn.reset_state("p1")
    assert _syn.injection_count("p1") == 0


def test_state_snapshot_exposes_injection_count_and_over_budget():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    _syn.build_continue_injection("p1", max_injections=2)
    snap = _syn.state_snapshot("p1", max_injections=2)
    assert snap["injection_count"] == 1
    assert snap["over_budget"] is False
    _syn.build_continue_injection("p1", max_injections=2)
    snap = _syn.state_snapshot("p1", max_injections=2)
    assert snap["injection_count"] == 2
    assert snap["over_budget"] is True


def test_build_budget_exhausted_reminder_is_non_empty_and_mentions_count():
    from tinyctx import synthetic_continue as _syn
    text = _syn.build_budget_exhausted_reminder("p1", count=12)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "12" in text


def test_maybe_inject_budget_reminder_appends_once():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    body = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hello"}]}
        ]
    }
    new_body, fired = _syn.maybe_inject_budget_reminder(
        body, "p1", count=20)
    assert fired is True
    assert len(new_body["input"]) == 2
    last = new_body["input"][-1]
    assert last["role"] == "user"
    assert "tinyctx" in last["content"][0]["text"].lower()
    # Original body untouched
    assert len(body["input"]) == 1
    # Second call no-ops
    new_body2, fired2 = _syn.maybe_inject_budget_reminder(
        new_body, "p1", count=20)
    assert fired2 is False
    assert new_body2 is new_body or new_body2 == new_body


def test_maybe_inject_budget_reminder_handles_missing_input():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    new_body, fired = _syn.maybe_inject_budget_reminder(
        {"no": "input"}, "p1", count=10)
    assert fired is False


def test_reset_state_clears_budget_reminder_flag():
    from tinyctx import synthetic_continue as _syn
    _syn.reset_state()
    body = {"input": [{"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "x"}]}]}
    _, fired = _syn.maybe_inject_budget_reminder(body, "p1", count=5)
    assert fired is True
    _syn.reset_state("p1")
    _, fired2 = _syn.maybe_inject_budget_reminder(body, "p1", count=5)
    assert fired2 is True


def test_config_default_max_continue_injections():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.max_continue_injections_per_session > 0
    assert cfg.max_continue_injections_per_session <= 100
