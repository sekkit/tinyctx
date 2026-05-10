"""Forensics module: capture full request + response for failure post-mortem."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def test_capture_request_snapshot_stored_in_ring():
    from tinyctx import forensics
    forensics.reset_state()
    forensics.capture_request_snapshot(
        proj_sid="p1",
        request_id="rq_abc",
        url="https://api.deepseek.com/v1/chat/completions",
        body={"model": "deepseek-v4-flash", "messages": [{"role": "user",
                                                           "content": "go"}]},
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-secret"},
        request_started_at=time.time(),
    )
    snap = forensics.get_recent_request("p1")
    assert snap is not None
    assert snap["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert snap["request_id"] == "rq_abc"
    # Auth header should be redacted
    assert "<redacted" in snap["headers"]["Authorization"]
    # Content-Type kept
    assert snap["headers"]["Content-Type"] == "application/json"


def test_summarize_body_truncates_long_strings():
    from tinyctx.forensics import _summarize_body
    body = {"prompt": "x" * 10000, "model": "test"}
    out = _summarize_body(body)
    assert out["model"] == "test"
    assert len(out["prompt"]) < 10000
    assert "truncated" in out["prompt"]


def test_summarize_body_keeps_first_last_of_lists():
    """For long input arrays, keep head + tail so we see context-window
    contents but don't blow up disk."""
    from tinyctx.forensics import _summarize_body
    long_input = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    body = {"input": long_input}
    out = _summarize_body(body)
    summary = out["input"]
    assert summary["_total_items"] == 20
    assert len(summary["first_3"]) == 3
    assert len(summary["last_3"]) == 3
    assert summary["middle_omitted"] == 14
    # First message preserved
    assert summary["first_3"][0]["content"] == "msg 0"


def test_write_dump_creates_file_with_request_response_pair(tmp_path: Path):
    from tinyctx import forensics
    forensics.reset_state()
    forensics.capture_request_snapshot(
        proj_sid="p1",
        request_id="rq_test",
        url="https://api.deepseek.com/v1/chat/completions",
        body={"model": "x", "messages": [{"role": "user", "content": "go"}]},
        headers={"Content-Type": "application/json"},
        request_started_at=time.time(),
    )
    path = forensics.write_forensics_dump(
        forensics_dir=tmp_path,
        proj_sid="p1",
        trigger="empty_response",
        response_buffer='data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n',
        timing={"elapsed_s": 12.3},
        extra={"completion_tokens": 1, "finish_reason": "stop"},
    )
    assert path is not None
    assert path.exists()
    dump = json.loads(path.read_text(encoding="utf-8"))
    assert dump["trigger"] == "empty_response"
    assert dump["proj_sid"] == "p1"
    assert dump["request"]["request_id"] == "rq_test"
    assert dump["response"]["buffer_chars"] > 0
    assert dump["timing"]["elapsed_s"] == 12.3
    assert dump["extra"]["completion_tokens"] == 1


def test_write_dump_handles_missing_request(tmp_path: Path):
    """When session has no captured request, dump still writes with a
    note (not None). Useful for sessions where request capture failed
    silently."""
    from tinyctx import forensics
    forensics.reset_state()
    path = forensics.write_forensics_dump(
        forensics_dir=tmp_path,
        proj_sid="never_seen",
        trigger="test",
        response_buffer="data: ...",
    )
    assert path is not None
    dump = json.loads(path.read_text(encoding="utf-8"))
    assert dump["request"]["_note"]


def test_max_dumps_rolls_oldest(tmp_path: Path):
    from tinyctx import forensics
    forensics.reset_state()
    # Write 5 dumps with max=3 → only 3 should remain
    for i in range(5):
        forensics.write_forensics_dump(
            forensics_dir=tmp_path,
            proj_sid="p1",
            trigger=f"test_{i}",
            response_buffer=f"chunk {i}",
            max_dumps=3,
        )
        time.sleep(0.01)  # ensure unique timestamps
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 3
    # Last 3 by name (timestamp prefix) — should be the latest writes
    triggers = [f.name.split("-")[2] for f in files]
    assert all(t.startswith("test_") for t in triggers)


def test_list_dumps_returns_metadata(tmp_path: Path):
    from tinyctx import forensics
    forensics.reset_state()
    forensics.write_forensics_dump(
        forensics_dir=tmp_path, proj_sid="p1",
        trigger="empty_response", response_buffer="x")
    forensics.write_forensics_dump(
        forensics_dir=tmp_path, proj_sid="p1",
        trigger="punt_high_p", response_buffer="y")
    listing = forensics.list_dumps(tmp_path)
    assert len(listing) == 2
    triggers = {d["trigger"] for d in listing}
    assert triggers == {"empty_response", "punt_high_p"}


def test_per_session_request_isolation():
    """projA's request shouldn't show up under projB."""
    from tinyctx import forensics
    forensics.reset_state()
    forensics.capture_request_snapshot(
        proj_sid="projA", request_id="rqA", url="urlA",
        body={"model": "A"}, headers={}, request_started_at=time.time())
    forensics.capture_request_snapshot(
        proj_sid="projB", request_id="rqB", url="urlB",
        body={"model": "B"}, headers={}, request_started_at=time.time())
    a = forensics.get_recent_request("projA")
    b = forensics.get_recent_request("projB")
    assert a["request_id"] == "rqA"
    assert b["request_id"] == "rqB"
    assert forensics.get_recent_request("nope") is None


def test_default_config_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.forensics_enabled is True
    assert cfg.forensics_capture_punts is False  # off by default (disk spam)
    assert cfg.forensics_max_dumps >= 30
