"""Tests for tinyctx.caveman_bootstrap."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import caveman_bootstrap as cb


@pytest.fixture
def isolated(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-caveman-") as td:
        home = Path(td)
        vendor = home / "vendor" / "caveman"
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(cb, "TINYCTX_HOME", home)
        monkeypatch.setattr(cb, "DEFAULT_VENDOR", vendor)
        monkeypatch.setattr(cb, "LOG_FILE", home / "logs" / "boot.log")
        monkeypatch.setattr(cb, "CODEX_CONFIG_DEFAULT", codex)
        yield home, vendor, codex


def test_state_no_vendor(monkeypatch, isolated):
    _, vendor, codex = isolated
    monkeypatch.setattr(cb, "_which", lambda c: "")
    s = cb.detect_state(vendor=vendor, codex_config=codex)
    assert s.vendor_present is False
    assert s.entry_present is False


def test_state_with_vendor_and_entry(monkeypatch, isolated):
    _, vendor, codex = isolated
    # Create fake vendor + .git + entry script
    (vendor / ".git").mkdir(parents=True, exist_ok=True)
    entry = vendor / cb.SHRINK_REL_PATH
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("// fake\n")
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("git", "node") else "")
    s = cb.detect_state(vendor=vendor, codex_config=codex)
    assert s.vendor_present is True
    assert s.entry_present is True
    assert s.git_path == "/u/b/git"
    assert s.node_path == "/u/b/node"


def test_disabled_short_circuits(monkeypatch, isolated):
    _, vendor, codex = isolated
    monkeypatch.setenv("TINYCTX_CAVEMAN_DISABLE", "1")
    monkeypatch.setattr(cb, "_which", lambda c: "")
    report = cb.bootstrap(vendor_dir=vendor, codex_config=codex)
    assert "TINYCTX_CAVEMAN_DISABLE=1" in report.skipped


def test_vendor_no_op_when_present(monkeypatch, isolated):
    _, vendor, codex = isolated
    (vendor / ".git").mkdir(parents=True, exist_ok=True)
    state = cb.State(vendor_present=True, git_path="/u/b/git",
                     vendor_dir=str(vendor))
    captured = []
    monkeypatch.setattr(cb, "_run",
                        lambda c, *, timeout=300, cwd=None:
                        captured.append(c) or (True, "ok"))
    ok, msg = cb.vendor(state, vendor_dir=vendor)
    assert ok
    assert "up-to-date" in msg or "skip pull" in msg
    # ran git pull, not git clone
    if captured:
        assert any("pull" in tok for tok in captured[0])


def test_vendor_clones_when_absent(monkeypatch, isolated):
    _, vendor, codex = isolated
    state = cb.State(vendor_present=False, git_path="/u/b/git",
                     vendor_dir=str(vendor))
    captured = []
    monkeypatch.setattr(cb, "_run",
                        lambda c, *, timeout=300, cwd=None:
                        captured.append(c) or (True, "ok"))
    ok, msg = cb.vendor(state, vendor_dir=vendor)
    assert ok
    assert any("clone" in tok for tok in captured[0])
    assert any("--depth" in tok for tok in captured[0])


def test_vendor_dry_run_does_not_run(monkeypatch, isolated):
    _, vendor, codex = isolated
    state = cb.State(vendor_present=False, git_path="/u/b/git",
                     vendor_dir=str(vendor))
    called = []
    monkeypatch.setattr(cb, "_run",
                        lambda c, *, timeout=300, cwd=None:
                        called.append(c) or (False, "should not run"))
    ok, msg = cb.vendor(state, vendor_dir=vendor, dry_run=True)
    assert ok
    assert "DRY-RUN" in msg
    assert called == []


def test_vendor_fails_when_no_git(isolated):
    _, vendor, codex = isolated
    state = cb.State(vendor_present=False, git_path="")
    ok, msg = cb.vendor(state, vendor_dir=vendor)
    assert ok is False
    assert "git not on PATH" in msg


def test_patch_codex_config_requires_entry(isolated):
    _, vendor, codex = isolated
    state = cb.State(entry_present=False, node_path="/u/b/node")
    ok, msg = cb.patch_codex_config(state, config_path=codex)
    assert ok is False
    assert "missing" in msg


def test_patch_codex_config_requires_node(isolated):
    _, vendor, codex = isolated
    state = cb.State(entry_present=True, node_path="",
                     entry_path="/path/to/index.js")
    ok, msg = cb.patch_codex_config(state, config_path=codex)
    assert ok is False
    assert "node" in msg


def test_patch_codex_config_creates(isolated):
    _, vendor, codex = isolated
    state = cb.State(entry_present=True, node_path="/u/b/node",
                     entry_path="/path/to/caveman/mcp-servers/caveman-shrink/index.js")
    ok, _ = cb.patch_codex_config(state, config_path=codex)
    assert ok
    txt = codex.read_text()
    assert "[mcp_servers.caveman-shrink]" in txt
    assert '"/u/b/node"' in txt


def test_default_install_does_not_register_standalone_mcp_block(monkeypatch,
                                                                  isolated):
    """caveman-shrink is middleware that wraps another MCP server, NOT a
    standalone server. Auto-registering [mcp_servers.caveman-shrink] with
    no upstream args produces a server that crashes immediately on spawn
    ("missing upstream command"). Default install must NOT write that
    block."""
    _, vendor, codex = isolated
    (vendor / ".git").mkdir(parents=True, exist_ok=True)
    entry = vendor / cb.SHRINK_REL_PATH
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("// fake")
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("git", "node", "npm") else "")
    monkeypatch.setattr(cb, "_run", lambda *a, **kw: (True, "ok"))
    cb.bootstrap(vendor_dir=vendor, codex_config=codex)
    if codex.is_file():
        assert "[mcp_servers.caveman-shrink]" not in codex.read_text()


def test_default_install_strips_existing_broken_block(monkeypatch, isolated):
    """If a previous version of this bootstrap (or anything else) wrote
    a standalone [mcp_servers.caveman-shrink] block that codex can't
    actually start, the new default install must remove it on next run."""
    _, vendor, codex = isolated
    (vendor / ".git").mkdir(parents=True, exist_ok=True)
    entry = vendor / cb.SHRINK_REL_PATH
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("// fake")
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text(
        "[mcp_servers.advisor]\n"
        'cmd = "x"\n\n'
        "# Added by tinyctx (caveman_bootstrap). [legacy broken]\n"
        "[mcp_servers.caveman-shrink]\n"
        'type = "stdio"\n'
        'command = "/u/b/node"\n'
        'args = ["/path/to/index.js"]\n'
    )
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("git", "node", "npm") else "")
    monkeypatch.setattr(cb, "_run", lambda *a, **kw: (True, "ok"))
    report = cb.bootstrap(vendor_dir=vendor, codex_config=codex)
    final = codex.read_text()
    assert "[mcp_servers.caveman-shrink]" not in final
    assert "[mcp_servers.advisor]" in final
    assert any("stripped broken" in a for a in report.actions)


def test_npm_install_invoked_during_vendor(monkeypatch, isolated):
    """After git clone the bootstrap must run `npm install` in
    mcp-servers/caveman-shrink/ so caveman-shrink's runtime deps are
    present. Without this the server crashes with 'Cannot find module'."""
    _, vendor, codex = isolated
    (vendor / "mcp-servers" / "caveman-shrink").mkdir(parents=True)
    captured: list = []

    def _fake_run(cmd, *, timeout=300, cwd=None):
        captured.append((cmd, cwd))
        return True, "ok"

    monkeypatch.setattr(cb, "_run", _fake_run)
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("git", "node", "npm") else "")
    state = cb.detect_state(vendor=vendor, codex_config=codex)
    cb.npm_install_shrink(state, vendor_dir=vendor)
    npm_calls = [c for c in captured
                 if c[0] and c[0][0].endswith("npm")]
    assert npm_calls
    cmd, cwd = npm_calls[0]
    assert "install" in cmd
    assert cwd and cwd.endswith("caveman-shrink")


def test_patch_codex_config_idempotent(isolated):
    _, vendor, codex = isolated
    state = cb.State(entry_present=True, node_path="/u/b/node",
                     entry_path="/x.js")
    cb.patch_codex_config(state, config_path=codex)
    snap = codex.read_text()
    state2 = cb.detect_state(vendor=vendor, codex_config=codex)
    assert state2.codex_config_has_caveman is True
    cb.patch_codex_config(state2, config_path=codex)
    assert codex.read_text() == snap


def test_full_bootstrap_dry_run(monkeypatch, isolated):
    _, vendor, codex = isolated
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("git", "node") else "")
    report = cb.bootstrap(vendor_dir=vendor, codex_config=codex, dry_run=True)
    # vendor dry-run + config skipped (entry not present yet)
    assert any("DRY-RUN" in a for a in report.actions)
    assert not codex.exists(), "dry-run must not write codex config"


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
