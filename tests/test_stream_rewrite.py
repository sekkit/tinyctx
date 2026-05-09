"""Stream-rewrite SSE-event synthesis: detect response.completed,
build task body, generate synthetic function_call events."""
from __future__ import annotations

import json

import pytest


# ─── response.completed detection ──────────────────────────────────────────


def test_looks_like_response_completed_positive():
    from tinyctx.stream_rewrite import looks_like_response_completed
    chunk = b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"
    assert looks_like_response_completed(chunk) is True


def test_looks_like_response_completed_negative():
    from tinyctx.stream_rewrite import looks_like_response_completed
    chunk = b"event: response.output_text.delta\ndata: {\"delta\":\"x\"}\n\n"
    assert looks_like_response_completed(chunk) is False
    assert looks_like_response_completed(b"") is False


def test_split_at_completed_with_marker_in_middle():
    from tinyctx.stream_rewrite import split_at_completed
    chunk = (b"event: response.output_text.delta\ndata: {\"delta\":\"x\"}\n\n"
             b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n")
    pre, completed = split_at_completed(chunk)
    assert pre == b"event: response.output_text.delta\ndata: {\"delta\":\"x\"}\n\n"
    assert completed.startswith(b"event: response.completed")


def test_split_at_completed_no_marker_returns_chunk_as_completed():
    """When the marker isn't in the chunk, fallback returns (b'', chunk)
    so the caller's logic (gated on `looks_like_response_completed`)
    doesn't accidentally yield the chunk twice."""
    from tinyctx.stream_rewrite import split_at_completed
    chunk = b"event: foo\ndata: bar\n\n"
    pre, completed = split_at_completed(chunk)
    assert pre == b""
    assert completed == chunk


# ─── synthetic event builder ──────────────────────────────────────────────


def test_synthetic_advisor_events_count_and_order():
    from tinyctx.stream_rewrite import synthetic_advisor_call_events
    events = synthetic_advisor_call_events(task="do the work")
    # 4 SSE events: added, args.delta, args.done, item.done
    assert len(events) == 4
    text = b"".join(events).decode("utf-8")
    # Order matters — codex parses sequentially
    pos1 = text.find("response.output_item.added")
    pos2 = text.find("response.function_call_arguments.delta")
    pos3 = text.find("response.function_call_arguments.done")
    pos4 = text.find("response.output_item.done")
    assert pos1 < pos2 < pos3 < pos4
    # All four are present and ordered


def test_synthetic_events_share_item_id():
    """All four events for one function_call must share the same item id —
    codex correlates them by id."""
    from tinyctx.stream_rewrite import synthetic_advisor_call_events
    events = synthetic_advisor_call_events(task="x")
    ids = []
    for evt in events:
        body = evt.decode("utf-8")
        # find data: line
        for line in body.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                # added/done items have id under "item.id"; delta/args.done have
                # item_id at top level
                if "item" in payload:
                    ids.append(payload["item"].get("id"))
                elif "item_id" in payload:
                    ids.append(payload["item_id"])
    assert len(ids) == 4
    assert len(set(ids)) == 1, f"all 4 events must share one id; got {set(ids)}"


def test_synthetic_events_carry_task_argument():
    from tinyctx.stream_rewrite import synthetic_advisor_call_events
    events = synthetic_advisor_call_events(task="my specific task body")
    blob = b"".join(events).decode("utf-8")
    assert "my specific task body" in blob


def test_synthetic_events_use_configured_tool_name():
    """tool_name parameter overrides the default — important for codex
    versions where mcp__advisor__ask_advisor has the dispatcher bug."""
    from tinyctx.stream_rewrite import synthetic_advisor_call_events
    events = synthetic_advisor_call_events(task="x",
                                            tool_name="custom__advisor")
    blob = b"".join(events).decode("utf-8")
    assert "custom__advisor" in blob
    assert "mcp__advisor__ask_advisor" not in blob


def test_synthetic_events_output_index_avoids_collision():
    """output_index defaults to 99 to avoid colliding with upstream's
    own indexed items (which start at 0). The first three indexes
    (0, 1, 2) are commonly used by codex for reasoning + text + tool
    items; 99 keeps us safely past the cluster."""
    from tinyctx.stream_rewrite import synthetic_advisor_call_events
    events = synthetic_advisor_call_events(task="x")
    blob = b"".join(events).decode("utf-8")
    assert '"output_index": 99' in blob


# ─── task body builder ────────────────────────────────────────────────────


def test_build_task_body_includes_classifier_signals():
    from tinyctx.stream_rewrite import build_task_body
    task = build_task_body(
        text_excerpt="Here is my plan: 1. do A. 2. do B. 3. do C.",
        classifier_reason="plan without tool call",
        classifier_p=0.92,
    )
    assert "plan without tool call" in task
    assert "0.92" in task
    assert "do A" in task and "do C" in task
    # Decision shape — advisor must reply ask: or work:
    assert "work:" in task
    assert "ask:" in task


def test_build_task_body_caps_excerpt_at_1500_chars():
    """Long agent outputs get truncated so advisor input stays bounded."""
    from tinyctx.stream_rewrite import build_task_body
    long = "A" * 5000
    task = build_task_body(long, "x", 0.9)
    # The excerpt portion should be at most 1500 chars (last 1500)
    # Find the marker between metadata and excerpt
    excerpt_start = task.find("---") + 3
    excerpt_end = task.rfind("---")
    excerpt = task[excerpt_start:excerpt_end]
    assert len(excerpt.strip()) <= 1500


# ─── default config + flag wiring ─────────────────────────────────────────


def test_config_default_disabled():
    """Stream rewrite is OFF by default — opt-in via config.toml."""
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.soft_completion_stream_rewrite_enabled is False
    # Threshold sane (between 0.7 gate threshold and 1.0)
    assert 0.7 < cfg.soft_completion_stream_rewrite_threshold <= 1.0
    # Tool name default is the standard MCP advisor binding
    assert "advisor" in cfg.soft_completion_stream_rewrite_tool_name
