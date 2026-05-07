"""Tests for tinyctx.keypin: rollout-frequency-based key-file pinning."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import keypin


def _write_rollout(rollout_dir: Path, sid: str, events: list[dict]) -> None:
    p = rollout_dir / f"rollout-{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_scan_rollouts_counts_read_calls():
    with TemporaryDirectory() as td:
        rd = Path(td) / "rollouts"
        _write_rollout(rd, "s1", [
            {"tool_name": "Read", "tool_input": {"file_path": "src/auth.py"}},
            {"tool_name": "Read", "tool_input": {"file_path": "src/auth.py"}},
            {"tool_name": "Read", "tool_input": {"file_path": "src/db.py"}},
        ])
        _write_rollout(rd, "s2", [
            {"tool_name": "read_file", "tool_input": {"path": "src/auth.py"}},
        ])
        counts = keypin.scan_rollouts(rd, days=365)
        assert counts["src/auth.py"] == 3
        assert counts["src/db.py"] == 1


def test_scan_rollouts_skips_non_read_tools():
    with TemporaryDirectory() as td:
        rd = Path(td) / "rollouts"
        _write_rollout(rd, "s1", [
            {"tool_name": "Write", "tool_input": {"file_path": "/x.py"}},
            {"tool_name": "Bash",  "tool_input": {"command": "ls"}},
            {"tool_name": "Read",  "tool_input": {"file_path": "src/a.py"}},
        ])
        counts = keypin.scan_rollouts(rd, days=365)
        assert counts == Counter({"src/a.py": 1})


def test_scan_rollouts_handles_mcp_routed_reads():
    with TemporaryDirectory() as td:
        rd = Path(td) / "rollouts"
        _write_rollout(rd, "s1", [
            {"tool_name": "mcp__plugin_context-mode_context-mode__view_file",
             "tool_input": {"path": "src/util.py"}},
            {"tool_name": "mcp__server__read_thing",
             "tool_input": {"file_path": "src/foo.py"}},
            {"tool_name": "mcp__server__write_thing",
             "tool_input": {"file_path": "src/bar.py"}},  # not a read
        ])
        counts = keypin.scan_rollouts(rd, days=365)
        assert counts["src/util.py"] == 1
        assert counts["src/foo.py"] == 1
        assert "src/bar.py" not in counts


def test_filter_to_project_normalizes_relative_and_absolute():
    with TemporaryDirectory() as td:
        proj = Path(td) / "proj"
        proj.mkdir()
        (proj / "src").mkdir()
        # global counts include both abs and rel paths
        global_counts = Counter({
            str(proj / "src" / "auth.py"): 5,    # absolute under project
            "src/db.py":                  3,     # relative under project
            "/etc/passwd":                10,    # outside project
            "../other/x.py":              2,     # outside project
        })
        proj_counts = keypin.filter_to_project(global_counts, proj)
        assert proj_counts.get("src/auth.py") == 5
        assert proj_counts.get("src/db.py") == 3
        # outside-project paths dropped
        for k in proj_counts:
            assert "etc" not in k and "other" not in k


def test_write_keyfiles_orders_by_count_then_name():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            counts = Counter({"src/b.py": 5, "src/a.py": 5, "src/c.py": 3})
            path = keypin.write_keyfiles(counts, proj, top_n=3)
            text = path.read_text()
            # both 5-count entries come before the 3-count one,
            # and within the 5-count tie src/a.py comes before src/b.py.
            ai = text.index("src/a.py")
            bi = text.index("src/b.py")
            ci = text.index("src/c.py")
            assert ai < bi < ci


def test_write_keyfiles_handles_empty():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            path = keypin.write_keyfiles(Counter(), proj)
            text = path.read_text()
            assert "No Read-tool calls" in text


def test_scan_handles_missing_rollout_dir():
    counts = keypin.scan_rollouts(Path("/nonexistent/rollouts"), days=30)
    assert counts == Counter()


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
