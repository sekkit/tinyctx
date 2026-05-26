"""Tests for tinyctx.caveman_bootstrap — npm-install based."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import caveman_bootstrap as cb


@pytest.fixture
def isolated(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-caveman-") as td:
        home = Path(td)
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(cb, "TINYCTX_HOME", home)
        monkeypatch.setattr(cb, "LOG_FILE", home / "logs" / "boot.log")
        monkeypatch.setattr(cb, "CODEX_CONFIG_DEFAULT", codex)
        yield home, codex


def test_state_no_binary(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(cb, "_which", lambda c: "")
    s = cb.detect_state(codex_config=codex)
    assert s.caveman_shrink_present is False


def test_state_with_binary(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(cb, "_which",
                        lambda c: "/u/b/node" if c == "node" else
                                  "/u/b/npm" if c == "npm" else
                                  "/u/b/caveman-shrink" if c == "caveman-shrink" else "")
    s = cb.detect_state(codex_config=codex)
    assert s.caveman_shrink_present is True
    assert s.caveman_shrink_path == "/u/b/caveman-shrink"
    assert s.node_path == "/u/b/node"
    assert s.npm_present is True


def test_disabled_short_circuits(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setenv("TINYCTX_CAVEMAN_DISABLE", "1")
    monkeypatch.setattr(cb, "_which", lambda c: "")
    report = cb.bootstrap(codex_config=codex)
    assert "TINYCTX_CAVEMAN_DISABLE=1" in report.skipped


def test_install_via_npm_runs(monkeypatch):
    captured: list = []

    def _fake_run(cmd, **kwargs):
        captured.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(cb.subprocess, "run", _fake_run)
    monkeypatch.setattr(cb, "_which",
                        lambda c: "/u/b/node" if c == "node" else
                                  "/u/b/npm" if c == "npm" else "")
    ok, msg = cb.install_via_npm()
    assert ok, msg
    assert captured
    assert any("install" in tok for tok in captured[0])
    assert any("caveman-shrink" in t for t in captured[0])


def test_install_via_npm_dry_run():
    ok, msg = cb.install_via_npm(dry_run=True)
    assert ok
    assert "DRY-RUN" in msg


def test_install_via_npm_no_npm(monkeypatch):
    monkeypatch.setattr(cb, "_which", lambda c: "")
    ok, msg = cb.install_via_npm()
    assert ok is False
    assert "npm not on PATH" in msg


def test_patch_codex_config_requires_binary(isolated):
    _, codex = isolated
    state = cb.State(caveman_shrink_present=False)
    ok, msg = cb.patch_codex_config(state, config_path=codex)
    assert ok is False


def test_patch_codex_config_creates(isolated):
    _, codex = isolated
    state = cb.State(caveman_shrink_present=True, node_path="/u/b/node",
                     caveman_shrink_path="/u/b/caveman-shrink")
    ok, _ = cb.patch_codex_config(state, config_path=codex)
    assert ok
    txt = codex.read_text()
    assert "[mcp_servers.caveman-shrink]" in txt
    assert '"/u/b/node"' in txt
    assert '"/u/b/caveman-shrink"' in txt


def test_default_install_skips_config_block(monkeypatch, isolated):
    """Default install must NOT register [mcp_servers.caveman-shrink] —
    caveman-shrink is middleware that needs an upstream."""
    _, codex = isolated
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("node", "npm") else "")
    monkeypatch.setattr(cb.subprocess, "run",
                        lambda *a, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    cb.bootstrap(codex_config=codex)
    if codex.is_file():
        assert "[mcp_servers.caveman-shrink]" not in codex.read_text()


def test_default_install_strips_broken_block(monkeypatch, isolated):
    _, codex = isolated
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
                        lambda c: f"/u/b/{c}" if c in ("node", "npm") else "")
    monkeypatch.setattr(cb.subprocess, "run",
                        lambda *a, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    report = cb.bootstrap(codex_config=codex)
    final = codex.read_text()
    assert "[mcp_servers.caveman-shrink]" not in final
    assert "[mcp_servers.advisor]" in final
    assert any("stripped broken" in a for a in report.actions)


def test_patch_codex_config_idempotent(isolated):
    _, codex = isolated
    state = cb.State(caveman_shrink_present=True, node_path="/u/b/node",
                     caveman_shrink_path="/x.js")
    cb.patch_codex_config(state, config_path=codex)
    snap = codex.read_text()
    state2 = cb.detect_state(codex_config=codex)
    assert state2.codex_config_has_caveman is True
    cb.patch_codex_config(state2, config_path=codex)
    assert codex.read_text() == snap


def test_full_bootstrap_dry_run(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(cb, "_which",
                        lambda c: f"/u/b/{c}" if c in ("node", "npm") else "")
    report = cb.bootstrap(codex_config=codex, dry_run=True)
    assert any("DRY-RUN" in a for a in report.actions)
    assert not codex.exists(), "dry-run must not write codex config"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
