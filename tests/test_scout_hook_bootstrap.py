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


def test_register_replaces_stale_scout_hook(isolated, monkeypatch):
    hooks, tmp = isolated
    old_script = tmp / "scout-session-start.sh"
    new_script = tmp / "scout-session-start.bat"
    old_script.write_text("#!/bin/bash\n")
    new_script.write_text("@echo off\n")
    monkeypatch.setenv("TINYCTX_SCOUT_HOOK_SCRIPT", str(new_script))
    hooks.parent.mkdir(parents=True, exist_ok=True)
    hooks.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command",
                            "command": "cm-hook-shim sessionstart"}]},
                {"hooks": [{"type": "command",
                            "command": str(old_script),
                            "_added_by": "tinyctx.scout_hook_bootstrap"}]},
            ],
        }
    }))

    ok, msg = shb.register(hooks)

    assert ok
    assert "updated" in msg
    data = json.loads(hooks.read_text())
    commands = [
        h["command"]
        for group in data["hooks"]["SessionStart"]
        for h in group["hooks"]
    ]
    assert "cm-hook-shim sessionstart" in commands
    assert str(new_script) in commands
    assert str(old_script) not in commands


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


def test_resolve_main_repo_strips_worktree_path():
    """Agent harnesses run inside throwaway worktrees at
    `<main>/.claude/worktrees/<branch>/`. The bootstrap must NOT pin
    hook commands to a worktree path — those become dangling refs the
    moment the worktree is deleted at conversation end. Resolve back
    to the stable main checkout."""
    worktree = Path("/Users/x/dev/tinyctx/.claude/worktrees/keen-gates-ec705d")
    assert shb._resolve_main_repo(worktree) == Path("/Users/x/dev/tinyctx")


def test_resolve_main_repo_preserves_normal_path():
    """For a non-worktree path, just return it unchanged."""
    main = Path("/Users/x/dev/tinyctx")
    assert shb._resolve_main_repo(main) == main


def test_resolve_main_repo_ignores_unrelated_worktrees_dir():
    """A directory called 'worktrees' that's NOT under '.claude/' shouldn't
    be stripped (false positive)."""
    p = Path("/Users/x/projects/worktrees/myrepo")
    assert shb._resolve_main_repo(p) == p


def test_default_script_path_prefers_main_checkout_over_worktree(monkeypatch,
                                                                   tmp_path):
    """When bootstrap is imported from a worktree, _default_script_path
    must return the main-checkout copy of the platform scout script, not
    the worktree-local one."""
    script_name = shb._default_script_name()
    main_repo = tmp_path / "tinyctx-main"
    worktree = main_repo / ".claude" / "worktrees" / "branch-name"
    main_scripts = main_repo / "scripts"
    worktree_scripts = worktree / "scripts"
    main_scripts.mkdir(parents=True)
    worktree_scripts.mkdir(parents=True)
    (main_scripts / script_name).write_text("#!/bin/bash\n")
    (worktree_scripts / script_name).write_text("#!/bin/bash\n")

    # Pretend the bootstrap module lives inside the worktree
    fake_file = worktree / "tinyctx" / "scout_hook_bootstrap.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()
    monkeypatch.setattr(shb, "__file__", str(fake_file))
    monkeypatch.delenv("TINYCTX_SCOUT_HOOK_SCRIPT", raising=False)

    out = shb._default_script_path()
    # Must point to main checkout, NOT to the worktree
    assert str(main_scripts / script_name) == out
    assert ".claude/worktrees" not in out


def test_default_script_path_uses_bat_on_windows(monkeypatch, tmp_path):
    main_repo = tmp_path / "tinyctx-main"
    scripts = main_repo / "scripts"
    scripts.mkdir(parents=True)
    bat = scripts / "scout-session-start.bat"
    bat.write_text("@echo off\n")

    fake_file = main_repo / "tinyctx" / "scout_hook_bootstrap.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()
    monkeypatch.setattr(shb, "__file__", str(fake_file))
    monkeypatch.setattr(shb.os, "name", "nt")
    monkeypatch.delenv("TINYCTX_SCOUT_HOOK_SCRIPT", raising=False)

    out = shb._default_script_path()
    assert str(bat) == out


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
