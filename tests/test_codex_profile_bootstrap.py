"""Tests for tinyctx.codex_profile_bootstrap.

Covers the install-time auto-write of:
  [model_providers.tinyctx]
  [profiles.tinyctx]
  [profiles.tinyctx-goal]

The base provider/profile are required for `codex --profile tinyctx` to
find the proxy after `./scripts/install.sh` completes; `tinyctx-goal`
adds the long-running `/goal` shape without changing the default profile.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import codex_profile_bootstrap as cpb


@pytest.fixture
def isolated_home(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-test-cpb-") as td:
        home = Path(td)
        monkeypatch.setattr(cpb, "TINYCTX_HOME", home)
        monkeypatch.setattr(cpb, "LOG_FILE", home / "logs" / "boot.log")
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(cpb, "CODEX_CONFIG_DEFAULT", codex)
        yield home, codex


def test_detect_state_no_config(isolated_home):
    _, codex = isolated_home
    s = cpb.detect_state(codex)
    assert s.codex_config_exists is False
    assert s.has_provider_block is False
    assert s.has_profile_block is False
    assert s.has_goal_profile_block is False


def test_bootstrap_writes_both_blocks(isolated_home):
    _, codex = isolated_home
    report = cpb.bootstrap(codex_config=codex)
    assert report.success
    content = codex.read_text()
    assert "[model_providers.tinyctx]" in content
    assert "[profiles.tinyctx]" in content
    assert "[profiles.tinyctx-goal]" in content
    assert 'base_url = "http://127.0.0.1:4141/v1"' in content
    assert 'model_provider = "tinyctx"' in content
    assert "model_context_window = 400000" in content
    assert "model_context_window = 1050000" in content
    assert "features = { goals = true }" in content


def test_bootstrap_is_idempotent(isolated_home):
    _, codex = isolated_home
    cpb.bootstrap(codex_config=codex)
    first = codex.read_text()
    cpb.bootstrap(codex_config=codex)
    assert codex.read_text() == first
    text = codex.read_text()
    assert text.count("[model_providers.tinyctx]") == 1
    assert text.count("[profiles.tinyctx]") == 1
    assert text.count("[profiles.tinyctx-goal]") == 1


def test_bootstrap_dry_run_writes_nothing(isolated_home):
    _, codex = isolated_home
    report = cpb.bootstrap(codex_config=codex, dry_run=True)
    assert report.success
    assert not codex.exists()


def test_disabled_env_short_circuits(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_CODEX_PROFILE_DISABLE", "1")
    report = cpb.bootstrap(codex_config=codex)
    assert report.success
    assert "TINYCTX_CODEX_PROFILE_DISABLE=1" in report.skipped
    assert not codex.exists()


def test_proxy_url_env_override(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_PROXY_URL", "http://10.0.0.5:9999/v1")
    cpb.bootstrap(codex_config=codex)
    assert 'base_url = "http://10.0.0.5:9999/v1"' in codex.read_text()


def test_profile_model_env_override(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_PROFILE_MODEL", "tinyctx-frontier")
    cpb.bootstrap(codex_config=codex)
    assert 'model = "tinyctx-frontier"' in codex.read_text()


def test_profile_context_env_override(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_PROFILE_CONTEXT", "200000")
    monkeypatch.setenv("TINYCTX_PROFILE_AUTO_COMPACT", "32000")
    cpb.bootstrap(codex_config=codex)
    text = codex.read_text()
    assert "model_context_window = 200000" in text
    assert "model_auto_compact_token_limit = 32000" in text


def test_bootstrap_preserves_existing_blocks(isolated_home):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text(
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/usr/local/bin/gitnexus"\n'
    )
    cpb.bootstrap(codex_config=codex)
    text = codex.read_text()
    assert "[mcp_servers.gitnexus]" in text
    assert "[model_providers.tinyctx]" in text
    assert "[profiles.tinyctx]" in text


def test_uninstall_strips_both_blocks(isolated_home):
    _, codex = isolated_home
    cpb.bootstrap(codex_config=codex)
    assert "[model_providers.tinyctx]" in codex.read_text()
    rc = cpb._cmd_uninstall(codex, dry_run=False, quiet=True)
    assert rc == 0
    text = codex.read_text()
    assert "[model_providers.tinyctx]" not in text
    assert "[profiles.tinyctx]" not in text
    assert "[profiles.tinyctx-goal]" not in text


def test_uninstall_preserves_other_blocks(isolated_home):
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text(
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/usr/local/bin/gitnexus"\n'
    )
    cpb.bootstrap(codex_config=codex)
    cpb._cmd_uninstall(codex, dry_run=False, quiet=True)
    text = codex.read_text()
    assert "[mcp_servers.gitnexus]" in text


def test_partial_existing_only_provider(isolated_home):
    """If the user already has [model_providers.tinyctx] (e.g. they once
    pasted manually) but is missing [profiles.tinyctx], the bootstrap
    should add only the missing one."""
    _, codex = isolated_home
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text(
        "[model_providers.tinyctx]\n"
        'name = "manually pasted"\n'
        'base_url = "http://127.0.0.1:4141/v1"\n'
    )
    cpb.bootstrap(codex_config=codex)
    text = codex.read_text()
    assert text.count("[model_providers.tinyctx]") == 1
    assert text.count("[profiles.tinyctx]") == 1
    # Existing block is preserved verbatim
    assert 'name = "manually pasted"' in text


def test_main_install_then_status(isolated_home, capsys):
    _, codex = isolated_home
    rc = cpb.main(["install", "--quiet", "--codex-config", str(codex)])
    assert rc == 0
    rc2 = cpb.main(["status", "--codex-config", str(codex)])
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "[model_providers.tinyctx]" in out
    assert "[profiles.tinyctx]" in out


def test_main_disabled_returns_zero(isolated_home, monkeypatch):
    _, codex = isolated_home
    monkeypatch.setenv("TINYCTX_CODEX_PROFILE_DISABLE", "1")
    rc = cpb.main(["install", "--quiet", "--codex-config", str(codex)])
    assert rc == 0
    assert not codex.exists()


def test_install_then_codex_can_resolve_profile(isolated_home):
    """End-to-end semantic check: after bootstrap, parsing the file as
    TOML should yield a profile that codex can lookup by name."""
    _, codex = isolated_home
    cpb.bootstrap(codex_config=codex)
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]
    parsed = tomllib.loads(codex.read_text())
    assert "profiles" in parsed
    assert "tinyctx" in parsed["profiles"]
    assert "tinyctx-goal" in parsed["profiles"]
    assert parsed["profiles"]["tinyctx"]["model_provider"] == "tinyctx"
    goal_profile = parsed["profiles"]["tinyctx-goal"]
    assert goal_profile["model_provider"] == "tinyctx"
    assert goal_profile["features"]["goals"] is True
    assert parsed["model_providers"]["tinyctx"]["wire_api"] == "responses"


def test_default_profile_ships_with_l1_features(isolated_home):
    """The default [profiles.tinyctx] block ships with the same L1 fields
    as tinyctx-goal: features.goals=true + approval_policy="never" +
    sandbox_mode="danger-full-access".

    Why: tinyctx is operated as an autonomous local-first agent runner.
    Approval prompts and sandbox restrictions would break the
    self-classify / advisor escalation flow. Users who need guards
    should override these fields in a derived profile.
    """
    _, codex = isolated_home
    cpb.bootstrap(codex_config=codex)
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]
    parsed = tomllib.loads(codex.read_text())
    default = parsed["profiles"]["tinyctx"]
    assert default["features"]["goals"] is True
    assert default["approval_policy"] == "never"
    assert default["sandbox_mode"] == "danger-full-access"
