"""Tests for tinyctx.trace: dataclass round-trip, reader, filters,
rendering."""
from __future__ import annotations

import io
import contextlib
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import trace
from tinyctx.trace import (
    RequestTrace, filter_traces, iter_traces,
    render_compact_row, render_verbose,
)


def _make(**kw):
    base = dict(
        request_id="rq_test1", session_id="sess1",
        started_at=time.time(),
        route="local", route_reason="small/short -> cheap path",
        est_input_tokens=11865,
        tools_before=19, tools_after=13,
        tool_types_dropped=["image_generation", "namespace", "web_search"],
        fields_stripped=["client_metadata", "prompt_cache_key"],
        fields_injected=["text.format.type"],
        target_url="http://127.0.0.1:1234/v1/responses",
        target_wire_api="responses",
        target_model="qwen3.6-27b-crack",
        status=200, bytes_out=131245,
        translated=True, translated_calls=1,
        elapsed_s=5.2,
    )
    base.update(kw)
    return RequestTrace(**base)


def test_emit_writes_jsonl_event():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        t = _make()
        t.emit(log_dir)
        files = list(log_dir.glob("tinyctx-*.jsonl"))
        assert len(files) == 1
        line = files[0].read_text().strip()
        d = json.loads(line)
        assert d["event"] == "request_trace"
        assert d["request_id"] == "rq_test1"
        assert d["tool_types_dropped"] == ["image_generation", "namespace", "web_search"]
        assert d["status"] == 200


def test_iter_traces_round_trip():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        a = _make(request_id="rq_a")
        b = _make(request_id="rq_b", session_id="sess2")
        a.emit(log_dir)
        b.emit(log_dir)
        # mix in an unrelated event line so the filter must skip it
        unrelated = log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"
        with unrelated.open("a") as fh:
            fh.write(json.dumps({"event": "route", "decision": "local"}) + "\n")
        loaded = list(iter_traces(log_dir))
        ids = {t.request_id for t in loaded}
        assert ids == {"rq_a", "rq_b"}


def test_filter_by_session_id():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        _make(request_id="rq_a", session_id="alpha").emit(log_dir)
        _make(request_id="rq_b", session_id="bravo").emit(log_dir)
        _make(request_id="rq_c", session_id="alpha").emit(log_dir)
        out = list(filter_traces(iter_traces(log_dir), session_id="alpha"))
        assert {t.request_id for t in out} == {"rq_a", "rq_c"}


def test_filter_by_request_id():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        _make(request_id="rq_x").emit(log_dir)
        _make(request_id="rq_y").emit(log_dir)
        out = list(filter_traces(iter_traces(log_dir), request_id="rq_y"))
        assert len(out) == 1
        assert out[0].request_id == "rq_y"


def test_render_compact_row_includes_diffs():
    t = _make()
    row = render_compact_row(t)
    assert "local" in row
    assert "11865" in row    # token count
    assert "19→13" in row    # tool diff
    assert "5.2s" in row     # elapsed (1dp formatting)
    assert "trans=" in row   # translation indicator


def test_render_compact_row_for_compactor_path():
    t = _make(compactor_used=True, compactor_outcome="judged",
              route_reason="compaction handoff -> cheap path")
    row = render_compact_row(t)
    assert "COMPACT/judged" in row


def test_render_verbose_contains_all_sections():
    t = _make(
        encrypted_content_stripped=2,
        mutation_wanted=True, mutation_fired=True,
        mutation_gate_reason="context_usage=72% >= 65%",
        deduped_calls=3, purged_inputs=1, historian_substituted=True,
    )
    out = render_verbose(t)
    for marker in ("ROUTE", "TOKENS", "TRANSFORMS", "FORWARD",
                   "RESPONSE", "ELAPSED",
                   "encrypted_content_stripped:  2",
                   "tools:                       19 → 13",
                   "image_generation",
                   "client_metadata",
                   "text.format.type",
                   "context_usage=72%",
                   "dedup=3"):
        assert marker in out, f"missing {marker!r}"


def test_cli_default_table_renders_recent_traces():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        for i in range(3):
            _make(request_id=f"rq_{i}", est_input_tokens=10000 + i * 100).emit(log_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = trace.main(["--log-dir", str(log_dir), "--last", "5"])
        out = buf.getvalue()
        assert rc == 0
        # header present
        assert "ROUTE" in out
        # all three rows rendered
        for tok in ("10000", "10100", "10200"):
            assert tok in out


def test_cli_request_filter_renders_verbose():
    with TemporaryDirectory() as td:
        log_dir = Path(td)
        _make(request_id="rq_x", session_id="sx").emit(log_dir)
        _make(request_id="rq_y", session_id="sy").emit(log_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = trace.main(["--log-dir", str(log_dir), "--request", "rq_y"])
        out = buf.getvalue()
        assert rc == 0
        assert "rq_y" in out
        assert "rq_x" not in out
        # verbose multi-line shape
        assert "ROUTE" in out and "FORWARD" in out


def test_cli_returns_1_when_no_traces():
    with TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = trace.main(["--log-dir", str(td), "--last", "5"])
        assert rc == 1
        assert "no matching" in buf.getvalue()


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
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
