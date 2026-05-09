"""Tests for tinyctx.gitnexus_bootstrap.

Covers:
  - state detection (no node / old node / fresh node / no npm / binary present)
  - codex config patching (empty file / existing file / already-patched)
  - block-stripping (uninstall path)
  - license-ack idempotency
  - dry-run never writes
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from tinyctx import gitnexus_bootstrap as gb


# ─────────────────────── helpers ────────────────────────────────


@pytest.fixture
def isolated_home(monkeypatch):
    """Redirect TINYCTX_HOME and codex config to a tmp dir so tests don't
    touch the user's real environment."""
    with TemporaryDirectory(prefix="tinyctx-test-home-") as td:
        home = Path(td)
        monkeypatch.setattr(gb, "TINYCTX_HOME", home)
        monkeypatch.setattr(gb, "LOG_FILE", home / "logs" / "boot.log")
        monkeypatch.setattr(gb, "LICENSE_ACK_FILE", home / ".acked")
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(gb, "CODEX_CONFIG_DEFAULT", codex)
        yield home, codex


# ─────────────────────── node detection ─────────────────────────


def test_node_major_parses_v_prefix():
    assert gb._node_major("v22.5.1") == 22
    assert gb._node_major("22.5.1") == 22
    assert gb._node_major("v0.10.0") == 0
    assert gb._node_major("garbage") == 0
    assert gb._node_major("") == 0


def test_detect_state_no_node(isolated_home, monkeypatch):
    _, codex = isolated_home
    # No `node` on PATH
    monkeypatch.setattr(gb, "_which", lambda c: "")
    s = gb.detect_state(codex)
    assert s.node_present is False
    assert s.node_meets_min is False
    assert s.npm_present is False
    assert s.gitnexus_present is False


def test_detect_state_node_too_old(isolated_home, monkeypatch):
    _, codex = isolated_home

    def _w(c):
        return {"node": "/usr/bin/node", "npm": "/usr/bin/npm"}.get(c, "")
    monkeypatch.setattr(gb, "_which", _w)

    class _R:
        stdout = "v18.20.0\n"
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: _R())
    s = gb.detect_state(codex)
    assert s.node_present is True
    assert s.node_version == "v18.20.0"
    assert s.node_meets_min is False


def test_detect_state_node_meets_min(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")

    class _R:
        stdout = "v22.5.1\n"
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: _R())
    s = gb.detect_state(codex)
    assert s.node_meets_min is True
    assert s.gitnexus_present is True
    assert s.gitnexus_path == "/usr/bin/gitnexus"


def test_disabled_env_short_circuits(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_GITNEXUS_DISABLE", "1")
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    report = gb.bootstrap(codex_config=codex)
    assert report.success
    assert "TINYCTX_GITNEXUS_DISABLE=1" in report.skipped
    assert not codex.exists(), "must not write config when disabled"


# ─────────────────────── codex config patching ──────────────────


def test_patch_codex_config_creates_new_file(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    state = gb.detect_state(codex)
    ok, msg = gb.patch_codex_config(state, config_path=codex)
    assert ok
    assert codex.is_file()
    text = codex.read_text()
    assert gb._GITNEXUS_CONFIG_MARKER in text
    assert 'command = "/usr/bin/gitnexus"' in text


def test_patch_codex_config_appends_to_existing(isolated_home, monkeypatch):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        '[mcp_servers.advisor]\n'
        'type = "stdio"\n'
        'command = "/path/to/advisor"\n'
    )
    codex.write_text(existing)
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    state = gb.detect_state(codex)
    ok, _ = gb.patch_codex_config(state, config_path=codex)
    assert ok
    text = codex.read_text()
    assert "[mcp_servers.advisor]" in text
    assert gb._GITNEXUS_CONFIG_MARKER in text
    # advisor block preserved verbatim
    assert text.startswith("[mcp_servers.advisor]")


def test_patch_codex_config_idempotent(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    state = gb.detect_state(codex)
    gb.patch_codex_config(state, config_path=codex)
    text1 = codex.read_text()
    state2 = gb.detect_state(codex)
    assert state2.codex_config_has_gitnexus is True
    ok, msg = gb.patch_codex_config(state2, config_path=codex)
    assert ok
    assert msg == "already configured"
    text2 = codex.read_text()
    assert text1 == text2, "second patch must not change file"


def test_patch_codex_config_dry_run_does_not_write(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    state = gb.detect_state(codex)
    ok, msg = gb.patch_codex_config(state, config_path=codex, dry_run=True)
    assert ok
    assert "DRY-RUN" in msg
    assert not codex.exists()


def test_resolve_command_prefers_absolute_path(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/usr/bin/{c}")
    state = gb.detect_state(codex)
    cmd = gb._resolve_gitnexus_command(state)
    assert cmd == "/usr/bin/gitnexus"


def test_resolve_command_falls_back_to_bare(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: "")
    state = gb.detect_state(codex)
    state.gitnexus_path = ""
    cmd = gb._resolve_gitnexus_command(state)
    assert cmd == "gitnexus"


# ─────────────────────── strip block (uninstall) ────────────────


def test_strip_block_removes_only_gitnexus_section():
    text = (
        "[mcp_servers.advisor]\n"
        'type = "stdio"\n'
        'command = "x"\n'
        "\n"
        "# Added by tinyctx (gitnexus_bootstrap). Safe to delete or edit.\n"
        "# GitNexus is a tree-sitter codebase knowledge-graph MCP server.\n"
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/u/b/gitnexus"\n'
        'args = ["mcp"]\n'
        "\n"
        "[mcp_servers.serena]\n"
        'type = "stdio"\n'
    )
    out = gb._strip_block(text)
    assert "[mcp_servers.gitnexus]" not in out
    assert "[mcp_servers.advisor]" in out
    assert "[mcp_servers.serena]" in out
    assert "GitNexus is a tree-sitter" not in out, \
        "tinyctx-written comments should also be removed"


def test_strip_block_at_eof():
    text = (
        "[mcp_servers.advisor]\n"
        "command = \"x\"\n"
        "\n"
        "# Added by tinyctx (gitnexus_bootstrap)\n"
        "[mcp_servers.gitnexus]\n"
        'command = "g"\n'
    )
    out = gb._strip_block(text)
    assert "[mcp_servers.gitnexus]" not in out
    assert "[mcp_servers.advisor]" in out


def test_strip_block_no_op_if_absent():
    text = '[mcp_servers.advisor]\ncommand = "x"\n'
    out = gb._strip_block(text)
    assert out == text


# ─────────────────────── license ack ─────────────────────────────


def test_print_license_once_writes_marker(isolated_home, capsys):
    home, _ = isolated_home
    fired = gb.print_license_once()
    assert fired
    assert (home / ".acked").is_file()
    err = capsys.readouterr().err
    assert "PolyForm-Noncommercial" in err


def test_print_license_no_op_after_ack(isolated_home, capsys):
    home, _ = isolated_home
    gb.print_license_once()
    capsys.readouterr()  # flush
    fired = gb.print_license_once()
    assert fired is False
    err = capsys.readouterr().err
    assert "PolyForm-Noncommercial" not in err


# ─────────────────────── full bootstrap ──────────────────────────


def test_bootstrap_dry_run_with_node_no_binary(isolated_home, monkeypatch):
    _, codex = isolated_home

    def _w(c):
        return {"node": "/u/b/node", "npm": "/u/b/npm"}.get(c, "")
    monkeypatch.setattr(gb, "_which", _w)

    class _NodeR:
        stdout = "v22.0.0\n"
    monkeypatch.setattr(gb.subprocess, "run",
                        lambda *a, **kw: _NodeR())

    report = gb.bootstrap(codex_config=codex, dry_run=True)
    assert report.success
    assert any("DRY-RUN" in a for a in report.actions)
    assert not codex.exists(), "dry-run must not write codex config"


def test_bootstrap_skips_when_node_missing(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: "")
    report = gb.bootstrap(codex_config=codex)
    assert any("node not on PATH" in s for s in report.skipped)
    assert not codex.exists()


def test_bootstrap_configures_when_binary_already_present(isolated_home,
                                                          monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/u/b/{c}")

    class _NodeR:
        stdout = "v22.0.0\n"
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: _NodeR())

    report = gb.bootstrap(codex_config=codex)
    assert report.success
    assert codex.is_file()
    assert gb._GITNEXUS_CONFIG_MARKER in codex.read_text()
    # license fires on first activation
    assert any("license" in a for a in report.actions)


def test_bootstrap_idempotent_second_run(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setattr(gb, "_which", lambda c: f"/u/b/{c}")

    class _NodeR:
        stdout = "v22.0.0\n"
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: _NodeR())

    gb.bootstrap(codex_config=codex)
    snapshot = codex.read_text()
    report2 = gb.bootstrap(codex_config=codex)
    assert report2.success
    assert codex.read_text() == snapshot
    # license already acked → not in actions on second run
    assert not any("license" in a for a in report2.actions)


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
