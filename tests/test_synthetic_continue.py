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
