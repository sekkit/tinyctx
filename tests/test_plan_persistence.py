"""Cross-thread plan persistence: save update_plan calls per-cwd,
inject on fresh codex threads to bridge the context-loss gap."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def test_cwd_hash_stable():
    from tinyctx.plan_persistence import _cwd_hash
    h1 = _cwd_hash("/Users/sekkit/dev/tinyxr")
    h2 = _cwd_hash("/Users/sekkit/dev/tinyxr")
    assert h1 == h2
    h3 = _cwd_hash("/Users/sekkit/dev/other")
    assert h1 != h3
    # Empty cwd → "default" sentinel
    assert _cwd_hash("") == "default"


def test_save_then_load_roundtrip(tmp_path: Path):
    from tinyctx import plan_persistence as _pp
    plan_text = "  1. [completed] X\n  2. [pending] Y"
    saved = _pp.save_plan(tmp_path, "/repo/a", plan_text,
                           session_id="s1", turn_count=42)
    assert saved is True
    data = _pp.load_plan(tmp_path, "/repo/a")
    assert data is not None
    assert data["plan_text"] == plan_text
    assert data["session_id_at_save"] == "s1"
    assert data["turn_count_at_save"] == 42
    assert data["cwd"] == "/repo/a"


def test_save_dedups_identical_content(tmp_path: Path):
    """Writing the same plan twice should be a no-op on the second call."""
    from tinyctx import plan_persistence as _pp
    plan_text = "x"
    saved1 = _pp.save_plan(tmp_path, "/repo", plan_text)
    saved2 = _pp.save_plan(tmp_path, "/repo", plan_text)
    assert saved1 is True
    assert saved2 is False  # no-op


def test_load_returns_none_when_no_file(tmp_path: Path):
    from tinyctx.plan_persistence import load_plan
    assert load_plan(tmp_path, "/repo/never-saved") is None


def test_load_respects_ttl(tmp_path: Path):
    """Stale plans (older than TTL) shouldn't be auto-injected."""
    from tinyctx import plan_persistence as _pp
    _pp.save_plan(tmp_path, "/repo", "old plan")
    # Manually backdate the file's updated_at
    p = _pp.plan_path(tmp_path, "/repo")
    data = json.loads(p.read_text())
    data["updated_at"] = time.time() - 10 * 24 * 3600  # 10 days ago
    p.write_text(json.dumps(data))
    # Default TTL = 7 days → expired
    assert _pp.load_plan(tmp_path, "/repo", ttl_s=7 * 24 * 3600) is None
    # Loose TTL → still loadable
    assert _pp.load_plan(tmp_path, "/repo", ttl_s=30 * 24 * 3600) is not None


def test_save_skips_empty_plan_text(tmp_path: Path):
    from tinyctx.plan_persistence import save_plan
    assert save_plan(tmp_path, "/repo", "") is False
    assert save_plan(tmp_path, "/repo", "   ") is False


def test_inject_prepends_block_to_instructions(tmp_path: Path):
    from tinyctx.plan_persistence import inject_plan
    body = {"instructions": "you are codex.", "input": []}
    plan_data = {
        "cwd": "/repo",
        "plan_text": "  1. [pending] foo",
        "updated_at_iso": "2026-05-10T12:00",
        "turn_count_at_save": 100,
    }
    out, was_inj = inject_plan(body, plan_data)
    assert was_inj is True
    assert out["instructions"].startswith("<persisted-plan")
    assert "[pending] foo" in out["instructions"]
    # Original instructions still there at the end
    assert "you are codex" in out["instructions"]


def test_inject_no_op_when_plan_empty():
    from tinyctx.plan_persistence import inject_plan
    body = {"instructions": "x"}
    out, was = inject_plan(body, {"plan_text": ""})
    assert was is False


def test_inject_creates_instructions_when_missing():
    """Body without an `instructions` field gets one created on inject —
    we'd rather have the plan visible to the agent than skip silently."""
    from tinyctx.plan_persistence import inject_plan
    body = {"input": []}
    out, was = inject_plan(body, {
        "plan_text": "  1. [pending] foo",
        "updated_at_iso": "2026-05-10",
        "cwd": "/r",
        "turn_count_at_save": 1,
    })
    assert was is True
    assert "<persisted-plan" in out["instructions"]
    assert "[pending] foo" in out["instructions"]


def test_extract_plan_text_from_body():
    """body.input has an update_plan call; extract should return its
    rendered plan text."""
    from tinyctx import plan_persistence as _pp
    args = json.dumps({"plan": [
        {"step": "do A", "status": "completed"},
        {"step": "do B", "status": "in_progress"},
    ]})
    body = {
        "instructions": "x",
        "input": [
            {"role": "user", "content": "go"},
            {"type": "function_call", "name": "update_plan",
             "arguments": args, "call_id": "c1"},
        ],
    }
    text = _pp.extract_plan_text(body)
    assert "do A" in text
    assert "completed" in text
    assert "do B" in text


def test_extract_plan_text_empty_when_no_tracker():
    from tinyctx.plan_persistence import extract_plan_text
    body = {"input": [{"role": "user", "content": "hi"}]}
    assert extract_plan_text(body) == ""


def test_list_plans_returns_metadata(tmp_path: Path):
    from tinyctx import plan_persistence as _pp
    _pp.save_plan(tmp_path, "/repo/a", "plan A", turn_count=1)
    _pp.save_plan(tmp_path, "/repo/b", "plan B", turn_count=99)
    listing = _pp.list_plans(tmp_path)
    assert len(listing) == 2
    cwds = {p["cwd"] for p in listing}
    assert cwds == {"/repo/a", "/repo/b"}


def test_clear_plan_removes_file(tmp_path: Path):
    from tinyctx import plan_persistence as _pp
    _pp.save_plan(tmp_path, "/repo", "x")
    assert _pp.plan_path(tmp_path, "/repo").exists()
    assert _pp.clear_plan(tmp_path, "/repo") is True
    assert not _pp.plan_path(tmp_path, "/repo").exists()
    # Idempotent: clear again returns False
    assert _pp.clear_plan(tmp_path, "/repo") is False


def test_atomic_write_uses_tmp_then_rename(tmp_path: Path):
    """Verify there's no .tmp file left after a successful save."""
    from tinyctx import plan_persistence as _pp
    _pp.save_plan(tmp_path, "/repo", "x")
    plans_dir = tmp_path / "plans"
    files = list(plans_dir.glob("*"))
    # Should be one .json file, no .tmp leftovers
    assert all(not f.name.endswith(".tmp") for f in files)


def test_default_config_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.plan_persistence_enabled is True
    # 7 days default
    assert cfg.plan_persistence_ttl_s == 7 * 24 * 3600
