"""Tests for tinyctx.stats: synthesize JSONL, verify aggregation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyctx.stats import (
    _iter_events,
    aggregate,
    aggregate_quality,
    render,
    render_quality,
)


def _w(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_aggregate_counts_by_route_and_reason():
    events = [
        {"event": "route", "decision": "local", "reason": "small/short -> cheap path",
         "est_tokens": 100, "is_compaction": False},
        {"event": "route", "decision": "local", "reason": "compaction handoff -> cheap path",
         "est_tokens": 200, "is_compaction": True},
        {"event": "route", "decision": "frontier", "reason": "est_tokens=80000 >= 60000",
         "est_tokens": 80000, "is_compaction": False},
        {"event": "stream_done", "route": "local", "elapsed_s": 0.5, "bytes": 1000},
        {"event": "stream_done", "route": "frontier", "elapsed_s": 2.5, "bytes": 50000},
        {"event": "upstream_error"},
    ]
    s = aggregate(iter(events))
    assert s["total_requests"] == 3
    assert s["by_route"] == {"local": 2, "frontier": 1}
    assert s["compaction_redirects"] == 1
    assert s["est_input_tokens_routed"]["local"] == 300
    assert s["est_input_tokens_routed"]["frontier"] == 80000
    assert s["upstream_errors"] == 1
    assert "local" in s["avg_stream_seconds"]
    assert s["avg_stream_seconds"]["local"] == 0.5


def test_render_human_readable():
    s = aggregate(iter([
        {"event": "route", "decision": "local", "reason": "x", "est_tokens": 10},
    ]))
    out = render(s)
    assert "requests: 1" in out
    assert "local=1" in out


def test_iter_events_handles_missing_dir():
    out = list(_iter_events(Path("/nonexistent/path"), since=None))
    assert out == []


def test_iter_events_filters_by_since():
    with TemporaryDirectory() as td:
        d = Path(td)
        # one file dated 2026-01-01, one 2026-05-06
        old = d / "tinyctx-20260101.jsonl"
        new = d / "tinyctx-20260506.jsonl"
        _w(old, [{"event": "route", "decision": "local"}])
        _w(new, [{"event": "route", "decision": "frontier"}])
        all_events = list(_iter_events(d, since=None))
        assert len(all_events) == 2
        recent = list(_iter_events(d, since="2026-03-01"))
        # only the May file remains
        assert len(recent) == 1
        assert recent[0]["decision"] == "frontier"


def test_render_no_data():
    s = aggregate(iter([]))
    assert "no requests" in render(s)


# ─────────────────────────── quality report ───────────────────────────


def _trace(**fields):
    base = {
        "event": "request_trace", "session_id": "s",
        "route": "local", "is_compaction": False,
        "est_input_tokens": 1000, "forwarded_tokens_est": 800,
        "status": 200, "tools_trimmed_applied": False,
        "proactive_compact_applied": False,
        "read_delta_candidates": 0, "read_delta_replacements": 0,
        "read_delta_bytes_saved": 0,
    }
    base.update(fields)
    return base


def test_quality_returns_zero_when_no_traces():
    q = aggregate_quality(iter([]))
    assert q["total_traces"] == 0
    out = render_quality(q)
    assert "no request_trace events" in out


def test_quality_skips_non_trace_events():
    events = [
        {"event": "route", "decision": "local"},     # ignored
        {"event": "stream_done", "route": "local"},  # ignored
    ]
    q = aggregate_quality(iter(events))
    assert q["total_traces"] == 0


def test_quality_modest_compression_grades_A():
    """All local, modest 30% compression (forwarded=70% of input). The
    formula is honest: 30% shed → 30/100 on the compression dimension.
    Other dims are perfect, so the weighted score lands in A territory,
    not S — S is reserved for genuinely strong compression."""
    events = [_trace(route="local", est_input_tokens=10_000,
                     forwarded_tokens_est=7_000) for _ in range(20)]
    q = aggregate_quality(iter(events))
    assert q["total_traces"] == 20
    assert q["local_share"] == 1.0
    assert q["error_rate"] == 0.0
    # Compression dim: 30/100. Routing/compaction/reliability all 100.
    # Active weight sum: 0.75. Score ≈ (25+20+6+10)/0.75 ≈ 81 → A.
    assert q["grade"] == "A"
    assert 75 <= q["score"] <= 90


def test_quality_strong_compression_grades_S():
    """Heavy compression (forwarded=20% of input) plus perfect routing,
    no errors → S grade."""
    events = [_trace(route="local", est_input_tokens=10_000,
                     forwarded_tokens_est=2_000) for _ in range(20)]
    q = aggregate_quality(iter(events))
    assert q["grade"] == "S"
    assert q["score"] >= 90.0


def test_quality_all_frontier_no_compression_grades_low():
    """All frontier, no token shrink, no read-delta candidates, no
    trim — minimum useful proxy work. Should land below B."""
    events = [_trace(route="frontier", est_input_tokens=10_000,
                     forwarded_tokens_est=10_000,
                     tools_trimmed_applied=False) for _ in range(20)]
    q = aggregate_quality(iter(events))
    assert q["local_share"] == 0.0
    assert q["frontier_share"] == 1.0
    assert q["score"] < 70.0
    assert q["grade"] in ("C", "D", "F")


def test_quality_errors_drag_score_down():
    events = ([_trace(route="local", status=500) for _ in range(10)]
              + [_trace(route="local", status=200) for _ in range(10)])
    q = aggregate_quality(iter(events))
    assert q["error_rate"] == 0.5
    # reliability dim is 50/100 → drags overall down
    assert q["components"]["reliability"] == 50.0


def test_quality_read_delta_score_skipped_when_no_candidates():
    """If no read_delta candidates ever appeared, that dimension should
    be skipped (None) — not penalized to 0."""
    events = [_trace(read_delta_candidates=0) for _ in range(10)]
    q = aggregate_quality(iter(events))
    assert q["components"]["read_delta_savings"] is None


def test_quality_read_delta_score_reflects_replacement_ratio():
    events = [
        _trace(read_delta_candidates=10, read_delta_replacements=8),
        _trace(read_delta_candidates=10, read_delta_replacements=2),
    ]
    q = aggregate_quality(iter(events))
    # 10/20 replaced = 50%
    assert q["components"]["read_delta_savings"] == 50.0


def test_quality_compaction_discipline_drops_when_proactive_compact_fires():
    events = [_trace(proactive_compact_applied=True) for _ in range(5)] + \
             [_trace(proactive_compact_applied=False) for _ in range(5)]
    q = aggregate_quality(iter(events))
    # 5/10 = 50% compaction share -> discipline = 50.0
    assert abs(q["components"]["compaction_discipline"] - 50.0) < 0.1


def test_quality_token_compression_skipped_when_no_signal():
    """Older traces don't have forwarded_tokens_est. That dimension
    should be None and weight redistributed to others."""
    events = [_trace(forwarded_tokens_est=0) for _ in range(10)]
    q = aggregate_quality(iter(events))
    assert q["components"]["token_compression"] is None


def test_quality_tool_trim_skipped_with_no_frontier_traffic():
    events = [_trace(route="local") for _ in range(10)]
    q = aggregate_quality(iter(events))
    assert q["components"]["tool_trim_savings"] is None


def test_quality_render_lists_dimensions_and_grade():
    events = [_trace(route="local") for _ in range(10)]
    q = aggregate_quality(iter(events))
    out = render_quality(q)
    assert "quality grade:" in out
    assert "routing efficiency" in out
    assert "reliability" in out


def test_quality_skipped_dimension_does_not_zero_score():
    """Critical: when a dimension is skipped (None) we redistribute its
    weight; we don't silently treat None as 0 and tank the score."""
    events = [_trace(route="local", forwarded_tokens_est=0,
                     tools_trimmed_applied=False) for _ in range(20)]
    # Two dimensions have signal: routing_efficiency (100), reliability
    # (100), compaction_discipline (100). Skipped: token_compression
    # (no fwd), read_delta_savings (no candidates), tool_trim_savings
    # (no frontier). Score should still be S despite 3 skips.
    q = aggregate_quality(iter(events))
    assert q["score"] >= 90.0
    assert q["grade"] == "S"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(failed)
