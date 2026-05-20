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
        norm = {k.replace("\\", "/"): v for k, v in proj_counts.items()}
        assert norm.get("src/auth.py") == 5
        assert norm.get("src/db.py") == 3
        # outside-project paths dropped
        for k in proj_counts:
            assert "etc" not in k and "other" not in k


def test_write_keyfiles_orders_by_count_then_name():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
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
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            path = keypin.write_keyfiles(Counter(), proj)
            text = path.read_text()
            assert "No Read-tool calls" in text


def test_scan_handles_missing_rollout_dir():
    counts = keypin.scan_rollouts(Path("/nonexistent/rollouts"), days=30)
    assert counts == Counter()


def test_scan_skips_malformed_json_lines(tmp_path):
    """Bad JSON lines, blank lines, and missing tool_name don't blow up the scan."""
    rd = tmp_path / "rollouts"
    rd.mkdir()
    p = rd / "rollout-mixed.jsonl"
    p.write_text(
        "this is not json\n"
        "\n"
        "{\"unrelated\": true}\n"
        + json.dumps({"tool_name": "Read", "tool_input": {"file_path": "src/keep.py"}})
        + "\n"
        "{broken json"
    )
    counts = keypin.scan_rollouts(rd, days=365)
    assert counts == Counter({"src/keep.py": 1})


def test_scan_respects_days_cutoff(tmp_path):
    """A rollout file older than the cutoff is skipped entirely."""
    import os as _os
    rd = tmp_path / "rollouts"
    rd.mkdir()
    fresh = rd / "rollout-fresh.jsonl"
    fresh.write_text(json.dumps(
        {"tool_name": "Read", "tool_input": {"file_path": "fresh.py"}}) + "\n")
    stale = rd / "rollout-stale.jsonl"
    stale.write_text(json.dumps(
        {"tool_name": "Read", "tool_input": {"file_path": "stale.py"}}) + "\n")
    # Backdate the stale file to 100 days ago.
    old_ts = stale.stat().st_mtime - 100 * 86_400
    _os.utime(stale, (old_ts, old_ts))
    counts = keypin.scan_rollouts(rd, days=30)
    assert counts == Counter({"fresh.py": 1})


def test_extract_path_priority_file_path_wins_over_path():
    """First key in _PATH_KEYS that's a non-empty string wins; file_path > path."""
    val = keypin._extract_path({"file_path": "winner.py", "path": "loser.py"})
    assert val == "winner.py"


def test_extract_path_falls_through_to_secondary_keys():
    """When file_path is missing/empty, falls through to path/target_file/etc."""
    assert keypin._extract_path({"path": "second.py"}) == "second.py"
    assert keypin._extract_path({"target_file": "third.py"}) == "third.py"
    assert keypin._extract_path({"filename": "fourth.py"}) == "fourth.py"
    assert keypin._extract_path({"filepath": "fifth.py"}) == "fifth.py"


def test_extract_path_handles_stringified_json_input():
    """tool_input arriving as a JSON string is parsed before key lookup."""
    s = json.dumps({"file_path": "via_string.py"})
    assert keypin._extract_path(s) == "via_string.py"


def test_extract_path_rejects_non_string_or_empty_values():
    """Empty string and non-string types must NOT be returned."""
    assert keypin._extract_path({"file_path": ""}) is None
    assert keypin._extract_path({"file_path": 42}) is None
    assert keypin._extract_path({"file_path": None}) is None
    assert keypin._extract_path({}) is None
    assert keypin._extract_path("not even json") is None
    assert keypin._extract_path(None) is None


def test_looks_like_read_mcp_routing():
    """MCP-routed tools count only when name contains read/view/get."""
    assert keypin._looks_like_read("mcp__server__read_file")
    assert keypin._looks_like_read("mcp__plugin__view_file")
    assert keypin._looks_like_read("mcp__plugin__get_thing")
    assert keypin._looks_like_read("Read")
    assert keypin._looks_like_read("read_file")
    # Negative cases:
    assert not keypin._looks_like_read("mcp__plugin__write_file")
    assert not keypin._looks_like_read("mcp__plugin__delete_thing")
    assert not keypin._looks_like_read("Write")
    assert not keypin._looks_like_read("Bash")


def test_scan_uses_payload_tool_name_fallback(tmp_path):
    """Some rollout schemas wrap tool_name/tool_input under a 'payload' field."""
    rd = tmp_path / "rollouts"
    rd.mkdir()
    p = rd / "rollout-payload.jsonl"
    p.write_text(json.dumps({
        "payload": {
            "tool_name": "Read",
            "tool_input": {"file_path": "via_payload.py"},
        }
    }) + "\n")
    counts = keypin.scan_rollouts(rd, days=365)
    assert counts == Counter({"via_payload.py": 1})


def test_filter_to_project_drops_paths_outside_root(tmp_path):
    """Paths that resolve outside the project root must not leak into output."""
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    counts = Counter({
        "/etc/passwd": 99,
        str(tmp_path / "elsewhere" / "x.py"): 7,
        "src/safe.py": 4,
    })
    out = keypin.filter_to_project(counts, proj)
    norm = {k.replace("\\", "/"): v for k, v in out.items()}
    assert norm == {"src/safe.py": 4}


def test_write_keyfiles_top_n_caps_results(tmp_path):
    """Only the top-N entries land in the rendered table."""
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        counts = Counter({f"f{i}.py": 100 - i for i in range(10)})
        path = keypin.write_keyfiles(counts, proj, top_n=3)
        text = path.read_text()
        # f0.py..f2.py should be present; f3.py..f9.py should not.
        for i in range(3):
            assert f"f{i}.py" in text
        for i in range(3, 10):
            assert f"f{i}.py" not in text


def test_write_keyfiles_byte_stable_across_runs(tmp_path):
    """Same input -> identical bytes (sorted by count desc, name asc)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        counts = Counter({"src/a.py": 3, "src/b.py": 5, "src/c.py": 5})
        p1 = keypin.write_keyfiles(counts, proj, top_n=10)
        bytes1 = p1.read_bytes()
        p2 = keypin.write_keyfiles(counts, proj, top_n=10)
        bytes2 = p2.read_bytes()
        assert bytes1 == bytes2


def test_main_show_missing_file_returns_1(tmp_path, capsys):
    """`show` with no prior scan emits a hint to stderr and exits 1."""
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        rc = keypin.main(["show", "--root", str(proj)])
        err = capsys.readouterr().err
        assert rc == 1
        assert "no keyfiles.md" in err


def test_main_scan_then_show_roundtrip(tmp_path, capsys):
    """End-to-end: scan a synthetic rollout dir, then show the rendered file."""
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    _write_rollout(rollouts, "s1", [
        {"tool_name": "Read", "tool_input": {"file_path": str(proj / "src" / "x.py")}},
        {"tool_name": "Read", "tool_input": {"file_path": str(proj / "src" / "x.py")}},
    ])
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        # patch the registry import so scan doesn't try to register globally
        with mock.patch("tinyctx.registry.register", lambda *_a, **_kw: None):
            rc_scan = keypin.main([
                "scan",
                "--root", str(proj),
                "--rollout-dir", str(rollouts),
                "--days", "365",
                "--top-n", "5",
            ])
        scan_out = capsys.readouterr()
        assert rc_scan == 0
        # path printed to stdout
        assert "keyfiles.md" in scan_out.out
        # show should now succeed
        rc_show = keypin.main(["show", "--root", str(proj)])
        show_out = capsys.readouterr().out
        assert rc_show == 0
        assert "src/x.py" in show_out or "src\\x.py" in show_out
        assert "| 2 |" in show_out


def test_scan_handles_unreadable_rollout_file(tmp_path):
    """OSError while reading a rollout file is caught; other files still scanned."""
    rd = tmp_path / "rollouts"
    rd.mkdir()
    good = rd / "rollout-good.jsonl"
    good.write_text(json.dumps(
        {"tool_name": "Read", "tool_input": {"file_path": "ok.py"}}) + "\n")
    bad = rd / "rollout-bad.jsonl"
    bad.write_text("placeholder\n")

    real_read_text = Path.read_text
    def flaky(self, *args, **kwargs):
        if self.name == "rollout-bad.jsonl":
            raise OSError("permission denied")
        return real_read_text(self, *args, **kwargs)

    with mock.patch("pathlib.Path.read_text", flaky):
        counts = keypin.scan_rollouts(rd, days=365)
    assert counts == Counter({"ok.py": 1})


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
