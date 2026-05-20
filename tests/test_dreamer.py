"""Tests for tinyctx.dreamer. We don't run real sub-commands — we patch
subprocess.run to capture invocations and return controlled exit codes."""
from __future__ import annotations

import io
import os
import contextlib
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import dreamer, registry


def _fake_run_factory(invocations):
    def _fake_run(argv, capture_output=True, text=True, timeout=300):
        invocations.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")
    return _fake_run


def test_run_calls_keypin_and_scout_for_each_project():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        # two projects, only one has a graph
        a = td_p / "alpha"; a.mkdir()
        (a / "tinyctx-graph.json").write_text("{}")
        b = td_p / "beta"; b.mkdir()
        invocations = []
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(a)
            registry.register(b)
            with mock.patch.object(subprocess, "run",
                                   side_effect=_fake_run_factory(invocations)):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = dreamer.main(["run"])
                assert rc == 0
        # alpha got both scout refresh + keypin scan; beta got only keypin.
        scout_calls = [c for c in invocations if "tinyctx-scout" in c[0]]
        keypin_calls = [c for c in invocations if "tinyctx-keypin" in c[0]]
        assert len(scout_calls) == 1, scout_calls
        assert len(keypin_calls) == 2, keypin_calls
        assert any(str(a) in " ".join(c) for c in scout_calls)
        assert any(str(b) in " ".join(c) for c in keypin_calls)


def test_run_with_ingest_mem_calls_mem():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        a = td_p / "alpha"; a.mkdir()
        invocations = []
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(a)
            with mock.patch.object(subprocess, "run",
                                   side_effect=_fake_run_factory(invocations)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = dreamer.main(["run", "--ingest-mem"])
                assert rc == 0
        mem_calls = [c for c in invocations if "tinyctx-mem" in c[0]]
        assert len(mem_calls) == 1


def test_run_returns_1_when_no_projects():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dreamer.main(["run"])
            assert rc == 1
            assert "no projects registered" in buf.getvalue()


def test_gc_removes_old_session_dirs():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        cache_root = td_p / ".tinyctx" / "cache"
        old = cache_root / "abc123" / "sessions" / "old-session"
        new = cache_root / "abc123" / "sessions" / "new-session"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        # write an old file (mtime in the past) and a fresh file
        old_md = old / "compaction-1.md"; old_md.write_text("x")
        new_md = new / "compaction-1.md"; new_md.write_text("x")
        old_mtime = time.time() - 60 * 86_400
        os.utime(old_md, (old_mtime, old_mtime))
        env_patch = {"HOME": str(td_p), "USERPROFILE": str(td_p)}
        with mock.patch.dict(os.environ, env_patch):
            deleted = dreamer._gc_old_sessions(retention_days=30)
            assert deleted == 1
            assert not old.exists()
            assert new.exists()


def test_register_unregister_via_cli():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "p"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = dreamer.main(["register", "--root", str(proj)])
            assert rc == 0
            assert registry.is_registered(proj)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = dreamer.main(["unregister", "--root", str(proj)])
            assert rc == 0
            assert not registry.is_registered(proj)


def test_install_cron_prints_a_crontab_line():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dreamer.main(["install-cron"])
    assert rc == 0
    out = buf.getvalue()
    assert "0 3 * * *" in out
    assert "tinyctx-dreamer" in out


def test_install_launchd_writes_plist_when_exe_on_path():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        # Pretend tinyctx-dreamer is on PATH.
        with mock.patch.object(shutil, "which", return_value="/usr/local/bin/tinyctx-dreamer"):
            with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = dreamer.main(["install-launchd"])
                assert rc == 0
                plist = (td_p / "Library" / "LaunchAgents"
                         / "com.tinyctx.dreamer.plist")
                assert plist.is_file()
                assert "com.tinyctx.dreamer" in plist.read_text()


def test_install_launchd_fails_clean_when_exe_missing():
    with mock.patch.object(shutil, "which", return_value=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dreamer.main(["install-launchd"])
        assert rc == 1


# ---------------------------------------------------------------------------
# Additional coverage: _run error paths, _gc_old_sessions edge cases,
# graphify-out detection, sub-failure return codes, and CLI list behaviour.
# ---------------------------------------------------------------------------


def test_run_helper_returns_1_on_filenotfound():
    """If the sub-command binary doesn't exist, _run must catch the
    FileNotFoundError, log a one-line note and return 1."""
    def _missing(*a, **kw):
        raise FileNotFoundError(2, "no such file")
    buf = io.StringIO()
    with mock.patch.object(subprocess, "run", side_effect=_missing):
        with contextlib.redirect_stdout(buf):
            rc = dreamer._run("scout refresh", ["does-not-exist"])
    assert rc == 1
    assert "FileNotFoundError" in buf.getvalue()


def test_run_helper_returns_1_on_timeout():
    """Timeouts in the sub-command must surface as rc=1, not propagate."""
    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=300)
    buf = io.StringIO()
    with mock.patch.object(subprocess, "run", side_effect=_timeout):
        with contextlib.redirect_stdout(buf):
            rc = dreamer._run("keypin scan", ["tinyctx-keypin", "scan"])
    assert rc == 1
    assert "TimeoutExpired" in buf.getvalue()


def test_run_helper_truncates_long_note_to_160_chars():
    """The streamed one-line note must be truncated at 160 chars to keep
    log lines readable."""
    long_line = "x" * 500
    fake = subprocess.CompletedProcess(["echo"], 0, stdout=long_line + "\n", stderr="")
    buf = io.StringIO()
    with mock.patch.object(subprocess, "run", return_value=fake):
        with contextlib.redirect_stdout(buf):
            rc = dreamer._run("label", ["echo"])
    assert rc == 0
    # The printed note should contain at most 160 'x's.
    out = buf.getvalue()
    assert "x" * 160 in out
    assert "x" * 161 not in out


def test_run_helper_falls_back_to_stderr_when_stdout_empty():
    """When stdout is empty but stderr has content, the note must come from
    stderr — useful for surfacing failure reasons."""
    fake = subprocess.CompletedProcess(["x"], 1, stdout="", stderr="boom-err\n")
    buf = io.StringIO()
    with mock.patch.object(subprocess, "run", return_value=fake):
        with contextlib.redirect_stdout(buf):
            rc = dreamer._run("label", ["x"])
    assert rc == 1
    assert "boom-err" in buf.getvalue()


def test_cmd_run_detects_graphify_out_graph_json():
    """A project that has graphify-out/graph.json (instead of the legacy
    tinyctx-graph.json) must still trigger scout refresh."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "with-graphify"; proj.mkdir()
        (proj / "graphify-out").mkdir()
        (proj / "graphify-out" / "graph.json").write_text("{}")
        invocations = []
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(proj)
            with mock.patch.object(subprocess, "run",
                                   side_effect=_fake_run_factory(invocations)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = dreamer.main(["run"])
                assert rc == 0
        scout_calls = [c for c in invocations if "tinyctx-scout" in c[0]]
        assert len(scout_calls) == 1


def test_cmd_run_skips_scout_when_no_graph_json_present():
    """No graph file => skip scout refresh silently (printed note), keypin
    scan still runs."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "no-graph"; proj.mkdir()
        invocations = []
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(proj)
            with mock.patch.object(subprocess, "run",
                                   side_effect=_fake_run_factory(invocations)):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = dreamer.main(["run"])
                assert rc == 0
        scout_calls = [c for c in invocations if "tinyctx-scout" in c[0]]
        assert len(scout_calls) == 0
        assert "scout refresh: skip" in buf.getvalue()


def test_cmd_run_returns_1_when_subcommand_fails():
    """If any sub-command exits non-zero, cmd_run must return 1 even if
    others succeeded."""
    def _fail_keypin(argv, capture_output=True, text=True, timeout=300):
        rc = 0 if "tinyctx-keypin" not in argv[0] else 2
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="bad\n")
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "p"; proj.mkdir()
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(proj)
            with mock.patch.object(subprocess, "run", side_effect=_fail_keypin):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = dreamer.main(["run"])
                assert rc == 1


def test_gc_old_sessions_returns_zero_when_cache_root_missing():
    """No cache directory at all -> _gc_old_sessions must return 0
    without raising."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            assert dreamer._gc_old_sessions(retention_days=30) == 0


def test_gc_old_sessions_skips_dirs_without_md_files():
    """A session dir with no .md files must be skipped (no removal,
    no exception)."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        cache = td_p / ".tinyctx" / "cache" / "repo" / "sessions" / "empty"
        cache.mkdir(parents=True)
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            assert dreamer._gc_old_sessions(retention_days=30) == 0
        assert cache.is_dir()


def test_gc_old_sessions_keeps_recent_sessions():
    """Sessions whose newest .md file is fresher than the cutoff are kept
    intact."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        sess = td_p / ".tinyctx" / "cache" / "r" / "sessions" / "fresh"
        sess.mkdir(parents=True)
        (sess / "compaction-1.md").write_text("y")
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            deleted = dreamer._gc_old_sessions(retention_days=30)
        assert deleted == 0
        assert sess.is_dir()


def test_cmd_list_prints_none_when_empty_and_returns_1():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dreamer.main(["list"])
            assert rc == 1
            assert "(none)" in buf.getvalue()


def test_cmd_list_prints_each_registered_project():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        a = td_p / "a"; a.mkdir()
        b = td_p / "b"; b.mkdir()
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            registry.register(a)
            registry.register(b)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dreamer.main(["list"])
            assert rc == 0
            out = buf.getvalue()
            assert str(a) in out
            assert str(b) in out


def test_cmd_register_idempotent_says_already_registered():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "p"; proj.mkdir()
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = dreamer.main(["register", "--root", str(proj)])
            assert rc == 0
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dreamer.main(["register", "--root", str(proj)])
            assert rc == 0
            assert "already registered" in buf.getvalue()


def test_cmd_unregister_unknown_project_says_not_registered():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "missing"; proj.mkdir()
        with mock.patch.dict(os.environ,
                             {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dreamer.main(["unregister", "--root", str(proj)])
            assert rc == 0
            assert "not registered" in buf.getvalue()


def test_install_launchd_plist_schedules_03_00_daily():
    """The generated launchd plist must schedule the run at 03:00 daily."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        with mock.patch.object(shutil, "which",
                               return_value="/opt/bin/tinyctx-dreamer"):
            with mock.patch.dict(os.environ,
                                 {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = dreamer.main(["install-launchd"])
                assert rc == 0
                plist = (td_p / "Library" / "LaunchAgents"
                         / "com.tinyctx.dreamer.plist")
                txt = plist.read_text()
        assert "<key>Hour</key><integer>3</integer>" in txt
        assert "<key>Minute</key><integer>0</integer>" in txt
        assert "/opt/bin/tinyctx-dreamer" in txt


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
