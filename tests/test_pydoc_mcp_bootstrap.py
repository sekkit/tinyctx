"""Tests for tinyctx.pydoc_mcp_bootstrap — codex config registration."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import pydoc_mcp_bootstrap as pmb


@pytest.fixture
def isolated(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-pydoc-mcp-") as td:
        home = Path(td)
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(pmb, "TINYCTX_HOME", home)
        monkeypatch.setattr(pmb, "LOG_FILE", home / "logs" / "boot.log")
        monkeypatch.setattr(pmb, "CODEX_CONFIG_DEFAULT", codex)
        monkeypatch.delenv("TINYCTX_PYDOC_MCP_DISABLE", raising=False)
        monkeypatch.delenv("TINYCTX_PYDOC_MCP_PYTHON", raising=False)
        yield home, codex


def test_state_default(isolated):
    _, codex = isolated
    s = pmb.detect_state(codex)
    assert s.disabled is False
    # The running interpreter must have tinyctx — otherwise this test
    # couldn't even import the module.
    assert s.python_path  # non-empty
    assert s.tinyctx_importable is True
    assert s.codex_config_exists is False
    assert s.codex_config_has_pydoc is False


def test_state_env_python_override(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setenv("TINYCTX_PYDOC_MCP_PYTHON", "/custom/python")
    s = pmb.detect_state(codex)
    assert s.python_path == "/custom/python"
    # Override != sys.executable, so tinyctx_importable defaults to False
    # unless we actually probe (which we don't on the hot path).
    assert s.tinyctx_importable is False


def test_disabled_short_circuits(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setenv("TINYCTX_PYDOC_MCP_DISABLE", "1")
    report = pmb.bootstrap(codex_config=codex)
    assert "TINYCTX_PYDOC_MCP_DISABLE=1" in report.skipped
    assert not codex.exists()
    assert report.success is True


def test_patch_codex_config_creates(isolated):
    _, codex = isolated
    ok, _ = pmb.patch_codex_config(config_path=codex)
    assert ok
    txt = codex.read_text()
    assert "[mcp_servers.pydoc]" in txt
    assert 'type = "stdio"' in txt
    assert '"-m", "tinyctx.pydoc_mcp"' in txt
    assert "tinyctx-pydoc-mcp-block-version" in txt


def test_patch_codex_config_idempotent(isolated):
    _, codex = isolated
    pmb.patch_codex_config(config_path=codex)
    first = codex.read_text()
    pmb.patch_codex_config(config_path=codex)
    assert codex.read_text() == first


def test_patch_codex_config_uses_env_python(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setenv("TINYCTX_PYDOC_MCP_PYTHON", "/opt/venv/bin/python")
    ok, _ = pmb.patch_codex_config(config_path=codex)
    assert ok
    assert '"/opt/venv/bin/python"' in codex.read_text()


def test_bootstrap_writes_then_detects(isolated):
    _, codex = isolated
    report = pmb.bootstrap(codex_config=codex)
    assert report.success is True
    assert any("codex config" in a for a in report.actions)
    state_after = pmb.detect_state(codex)
    assert state_after.codex_config_has_pydoc is True


def test_main_status_runs(capsys, isolated):
    _, codex = isolated
    rc = pmb.main(["status", "--codex-config", str(codex)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pydoc-mcp state:" in out


def test_main_uninstall_strips_block(isolated):
    _, codex = isolated
    pmb.patch_codex_config(config_path=codex)
    assert "[mcp_servers.pydoc]" in codex.read_text()
    rc = pmb.main(["uninstall", "--codex-config", str(codex)])
    assert rc == 0
    assert "[mcp_servers.pydoc]" not in codex.read_text()


def test_main_uninstall_no_config_is_noop(isolated, capsys):
    _, codex = isolated
    rc = pmb.main(["uninstall", "--codex-config", str(codex)])
    assert rc == 0
    assert "no codex config" in capsys.readouterr().out


def test_installer_pipeline_includes_pydoc_mcp():
    """Regression guard: pydoc-mcp must stay in the unified installer
    pipeline so `tinyctx install` and the proxy startup reach it."""
    from tinyctx import installer
    names = [fn.__name__ for fn in installer._INSTALLERS]
    assert "_register_pydoc_mcp" in names


def test_installer_status_reports_pydoc_mcp():
    """status_all() must surface pydoc-mcp so the dashboard reflects it."""
    from tinyctx import installer
    result = installer.status_all()
    assert "pydoc-mcp" in result
    assert "installed" in result["pydoc-mcp"]
    assert "python" in result["pydoc-mcp"]
