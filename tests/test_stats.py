"""Tests for tinyctx.stats: synthesize JSONL, verify aggregation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyctx.stats import _iter_events, aggregate, render


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
