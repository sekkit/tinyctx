"""Tests for tinyctx.advisor_bootstrap.

Covers:
  - state detection (no config / config without block / config with block)
  - patch writes a [mcp_servers.advisor] block with sys.executable as command
  - patch is idempotent (second call leaves file untouched)
  - dry-run never writes
  - TINYCTX_ADVISOR_DISABLE=1 short-circuits and writes nothing
  - TINYCTX_ADVISOR_PYTHON env override is honored
  - uninstall strips the block back out
  - the registered command is an absolute path (no /Users/<name> hardcode
    surviving a per-machine install)
"""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import advisor_bootstrap as ab


@pytest.fixture
def isolated_home(monkeypatch):
    """Redirect TINYCTX_HOME and codex config to a tmp dir."""
    with TemporaryDirectory(prefix="tinyctx-test-advisor-") as td:
        home = Path(td)
        monkeypatch.setattr(ab, "TINYCTX_HOME", home)
        monkeypatch.setattr(ab, "LOG_FILE", home / "logs" / "boot.log")
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(ab, "CODEX_CONFIG_DEFAULT", codex)
        yield home, codex


def test_resolve_python_uses_sys_executable(monkeypatch):
    monkeypatch.delenv("TINYCTX_ADVISOR_PYTHON", raising=False)
    py = ab._resolve_python()
    # Use .absolute() (not .resolve()) so venv symlinks to system Python
    # are NOT followed — codex must spawn the venv binary so site-packages
    # is the venv's, not the framework Python's.
    assert py == str(Path(sys.executable).absolute())
    assert Path(py).is_absolute()


def test_resolve_python_does_not_follow_venv_symlink(tmp_path, monkeypatch):
    """On macOS, .venv/bin/python is a symlink to /Library/.../python3.9.
    Following the symlink loses the venv — _resolve_python must NOT do that.
    """
    real_python = tmp_path / "system-python"
    real_python.write_text("#!/bin/sh\nexit 0\n")
    real_python.chmod(0o755)

    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    try:
        venv_python.symlink_to(real_python)
    except OSError as e:
        if getattr(e, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise

    monkeypatch.setenv("TINYCTX_ADVISOR_PYTHON", str(venv_python))
    resolved = ab._resolve_python()
    # The unresolved venv path must survive — that's what activates the
    # venv when spawned (Python detects venv via pyvenv.cfg next to argv0).
    assert resolved == str(venv_python)
    assert "system-python" not in resolved


def test_resolve_python_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("TINYCTX_ADVISOR_PYTHON", str(fake))
    assert ab._resolve_python() == str(fake)


def test_detect_state_no_config(isolated_home):
    _, codex = isolated_home
    s = ab.detect_state(codex)
    assert s.codex_config_exists is False
    assert s.codex_config_has_advisor is False
    assert s.python_path == str(Path(sys.executable).absolute())
    assert s.python_exists is True


def test_detect_state_config_without_advisor(isolated_home):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text("[profiles.tinyctx]\nmodel = \"x\"\n")
    s = ab.detect_state(codex)
    assert s.codex_config_exists is True
    assert s.codex_config_has_advisor is False


def test_bootstrap_writes_block_with_absolute_python(isolated_home):
    _, codex = isolated_home
    report = ab.bootstrap(codex_config=codex)
    assert report.success
    assert codex.is_file()
    content = codex.read_text(encoding="utf-8")
    assert "[mcp_servers.advisor]" in content
    assert 'args = ["-m", "tinyctx.advisor"]' in content
    # The registered command must be the actual interpreter currently
    # running this bootstrap (i.e. resolved at install time per machine),
    # not a hardcoded literal that only works on one developer's box.
    expected = str(Path(sys.executable).absolute())
    assert f'command = "{expected}"' in content


def test_example_agent_config_has_codex_agent_name():
    root = Path(__file__).resolve().parents[1]
    agent_text = (root / "examples" / "agent-advisor.toml").read_text(encoding="utf-8")
    assert 'name = "advisor"' in agent_text
    assert "100-200 words" in agent_text


def test_bootstrap_command_uses_per_machine_path(isolated_home, tmp_path,
                                                    monkeypatch):
    """Override TINYCTX_ADVISOR_PYTHON to a foreign path and confirm the
    written block contains *that* path verbatim — proves the value is
    runtime-resolved, not baked into the source."""
    _, codex = isolated_home
    foreign = tmp_path / "another-user-venv" / "bin" / "python"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\nexit 0\n")
    foreign.chmod(0o755)
    monkeypatch.setenv("TINYCTX_ADVISOR_PYTHON", str(foreign))
    ab.bootstrap(codex_config=codex)
    content = codex.read_text(encoding="utf-8")
    assert f'command = "{foreign}"' in content


def test_bootstrap_idempotent(isolated_home):
    _, codex = isolated_home
    ab.bootstrap(codex_config=codex)
    first = codex.read_text(encoding="utf-8")
    report2 = ab.bootstrap(codex_config=codex)
    assert report2.success
    assert codex.read_text(encoding="utf-8") == first, \
        "second bootstrap must not duplicate the block"
    # exactly one [mcp_servers.advisor] header
    assert codex.read_text(encoding="utf-8").count("[mcp_servers.advisor]") == 1


def test_bootstrap_dry_run_writes_nothing(isolated_home):
    _, codex = isolated_home
    report = ab.bootstrap(codex_config=codex, dry_run=True)
    assert report.success
    assert not codex.exists()


def test_disabled_env_short_circuits(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_ADVISOR_DISABLE", "1")
    report = ab.bootstrap(codex_config=codex)
    assert report.success
    assert "TINYCTX_ADVISOR_DISABLE=1" in report.skipped
    assert not codex.exists(), "must not write config when disabled"


def test_bootstrap_preserves_existing_blocks(isolated_home):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        "[profiles.tinyctx]\n"
        'model = "tinyctx-auto"\n\n'
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/usr/local/bin/gitnexus"\n'
    )
    codex.write_text(existing)
    report = ab.bootstrap(codex_config=codex)
    assert report.success
    new = codex.read_text(encoding="utf-8")
    assert "[profiles.tinyctx]" in new
    assert "[mcp_servers.gitnexus]" in new
    assert "[mcp_servers.advisor]" in new


def test_uninstall_strips_block(isolated_home):
    _, codex = isolated_home
    ab.bootstrap(codex_config=codex)
    assert "[mcp_servers.advisor]" in codex.read_text(encoding="utf-8")
    rc = ab._cmd_uninstall(codex, dry_run=False, quiet=True)
    assert rc == 0
    assert "[mcp_servers.advisor]" not in codex.read_text(encoding="utf-8")


def test_uninstall_when_block_absent_is_noop(isolated_home):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text("[profiles.tinyctx]\nmodel = \"x\"\n")
    rc = ab._cmd_uninstall(codex, dry_run=False, quiet=True)
    assert rc == 0
    assert codex.read_text(encoding="utf-8") == "[profiles.tinyctx]\nmodel = \"x\"\n"


def test_status_reports_no_advisor_initially(isolated_home, capsys):
    _, codex = isolated_home
    s = ab.detect_state(codex)
    ab._print_state_human(s)
    out = capsys.readouterr().out
    assert "advisor state" in out
    assert "[mcp_servers.advisor]" in out


def test_main_install_then_status(isolated_home, monkeypatch, capsys):
    _, codex = isolated_home
    rc = ab.main(["install", "--quiet", "--codex-config", str(codex)])
    assert rc == 0
    rc2 = ab.main(["status", "--codex-config", str(codex)])
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "yes" in out  # config now has advisor block


def test_main_disabled_env_returns_zero(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_ADVISOR_DISABLE", "1")
    rc = ab.main(["install", "--quiet", "--codex-config", str(codex)])
    assert rc == 0
    assert not codex.exists()
