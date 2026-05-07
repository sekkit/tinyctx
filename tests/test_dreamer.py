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
