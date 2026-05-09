"""Tests for tinyctx.scout_hook_bootstrap."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import scout_hook_bootstrap as shb


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    hooks = tmp_path / ".codex" / "hooks.json"
    monkeypatch.setattr(shb, "CODEX_HOOKS_PATH", hooks)
    monkeypatch.setattr(shb, "TINYCTX_HOME", tmp_path)
    monkeypatch.setattr(shb, "LOG_FILE", tmp_path / "logs" / "boot.log")
    yield hooks, tmp_path


def _fake_script(tmp_path: Path, monkeypatch) -> Path:
    """Create a fake scout-session-start.sh and force the bootstrap to
    use it."""
    script = tmp_path / "scout-session-start.sh"
    script.write_text("#!/bin/bash\necho '{}'\n")
    script.chmod(0o755)
    monkeypatch.setenv("TINYCTX_SCOUT_HOOK_SCRIPT", str(script))
    return script


def test_state_no_hooks_file(isolated):
    hooks, _ = isolated
    s = shb.detect_state(hooks)
    assert s.hooks_file_exists is False
    assert s.hook_already_registered is False


def test_state_script_missing(isolated, monkeypatch):
    hooks, tmp = isolated
    monkeypatch.setenv("TINYCTX_SCOUT_HOOK_SCRIPT", "/nonexistent/abs/path")
    s = shb.detect_state(hooks)
    assert s.script_exists is False


def test_register_creates_new_hooks_file(isolated, monkeypatch):
    hooks, tmp = isolated
    _fake_script(tmp, monkeypatch)
    ok, msg = shb.register(hooks)
    assert ok
    assert hooks.is_file()
    data = json.loads(hooks.read_text())
    se = data["hooks"]["SessionStart"]
    assert len(se) == 1
    assert "scout-session-start.sh" in se[0]["hooks"][0]["command"]


def test_register_appends_to_existing_session_start(isolated, monkeypatch):
    """The existing entry (e.g. cm-hook-shim) must be preserved verbatim."""
    hooks, tmp = isolated
    _fake_script(tmp, monkeypatch)
    hooks.parent.mkdir(parents=True, exist_ok=True)
    hooks.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [{"hooks": [
                {"type": "command", "command": "cm-hook-shim pretooluse"}
            ]}],
            "SessionStart": [{"hooks": [
                {"type": "command", "command": "cm-hook-shim sessionstart"}
            ]}],
        }
    }))
    ok, _ = shb.register(hooks)
    assert ok
    data = json.loads(hooks.read_text())
    # PreToolUse untouched
    assert "cm-hook-shim pretooluse" in \
        data["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    # SessionStart now has 2 groups
    assert len(data["hooks"]["SessionStart"]) == 2
    # Existing cm-hook-shim entry preserved
    assert "cm-hook-shim sessionstart" in \
        data["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_register_idempotent(isolated, monkeypatch):
    hooks, tmp = isolated
    _fake_script(tmp, monkeypatch)
    shb.register(hooks)
    snap = hooks.read_text()
    ok, msg = shb.register(hooks)
    assert ok
    assert "already registered" in msg
    assert hooks.read_text() == snap


def test_register_dry_run_does_not_write(isolated, monkeypatch):
    hooks, tmp = isolated
    _fake_script(tmp, monkeypatch)
    ok, msg = shb.register(hooks, dry_run=True)
    assert ok
    assert "DRY-RUN" in msg
    assert not hooks.exists()


def test_register_fails_when_script_missing(isolated, monkeypatch):
    hooks, _ = isolated
    monkeypatch.setenv("TINYCTX_SCOUT_HOOK_SCRIPT", "/no/such/file")
    ok, msg = shb.register(hooks)
    assert ok is False
    assert "missing" in msg


def test_unregister_removes_only_scout_entry(isolated, monkeypatch):
    hooks, tmp = isolated
    _fake_script(tmp, monkeypatch)
    hooks.parent.mkdir(parents=True, exist_ok=True)
    hooks.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command",
                             "command": "cm-hook-shim sessionstart"}]},
                {"hooks": [{"type": "command",
                             "command": str(tmp / "scout-session-start.sh"),
                             "_added_by": "tinyctx.scout_hook_bootstrap"}]},
            ],
        }
    }))
    ok, msg = shb.unregister(hooks)
    assert ok
    assert "removed" in msg
    data = json.loads(hooks.read_text())
    se = data["hooks"]["SessionStart"]
    # Only one entry remaining: the cm-hook-shim
    assert len(se) == 1
    assert "cm-hook-shim sessionstart" in se[0]["hooks"][0]["command"]


def test_unregister_no_op_when_absent(isolated, monkeypatch):
    hooks, tmp = isolated
    hooks.parent.mkdir(parents=True, exist_ok=True)
    hooks.write_text(json.dumps({
        "hooks": {"SessionStart": [
            {"hooks": [{"type": "command",
                         "command": "cm-hook-shim sessionstart"}]},
        ]}
    }))
    snap = hooks.read_text()
    ok, msg = shb.unregister(hooks)
    assert ok
    assert "no scout" in msg
    assert hooks.read_text() == snap


def test_disabled_env_short_circuits_main(isolated, monkeypatch, capsys):
    hooks, _ = isolated
    monkeypatch.setenv("TINYCTX_SCOUT_HOOK_DISABLE", "1")
    rc = shb.main(["install", "--hooks", str(hooks)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "TINYCTX_SCOUT_HOOK_DISABLE" in err
    assert not hooks.exists()


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
