"""Tests for tinyctx.serena_bootstrap."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyctx import serena_bootstrap as sb


@pytest.fixture
def isolated(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-serena-test-") as td:
        home = Path(td)
        monkeypatch.setattr(sb, "TINYCTX_HOME", home)
        monkeypatch.setattr(sb, "LOG_FILE", home / "logs" / "boot.log")
        codex = home / ".codex" / "config.toml"
        monkeypatch.setattr(sb, "CODEX_CONFIG_DEFAULT", codex)
        yield home, codex


def test_state_no_serena(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(sb, "_which", lambda c: "")
    s = sb.detect_state(codex)
    assert s.serena_present is False


def test_state_finds_serena_mcp_server_first(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(sb, "_which",
                        lambda c: "/u/b/serena-mcp-server"
                        if c == "serena-mcp-server" else "")
    s = sb.detect_state(codex)
    assert s.serena_present is True
    assert s.serena_path == "/u/b/serena-mcp-server"


def test_state_falls_back_to_serena_alias(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(sb, "_which",
                        lambda c: "/u/b/serena-mcp" if c == "serena-mcp"
                        else "")
    s = sb.detect_state(codex)
    assert s.serena_path == "/u/b/serena-mcp"


def test_disabled_env_short_circuits(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setenv("TINYCTX_SERENA_DISABLE", "1")
    monkeypatch.setattr(sb, "_which", lambda c: "")
    report = sb.bootstrap(codex_config=codex)
    assert "TINYCTX_SERENA_DISABLE=1" in report.skipped
    assert not codex.exists()


def test_install_chain_uv_first(monkeypatch, isolated):
    state = sb.State(uv_path="/u/b/uv", pipx_path="/u/b/pipx")
    calls = []
    monkeypatch.setattr(sb, "install_via_uv",
                        lambda s, *, dry_run=False:
                        calls.append("uv") or (True, "ok"))
    monkeypatch.setattr(sb, "install_via_pipx",
                        lambda s, *, dry_run=False:
                        calls.append("pipx") or (True, "ok"))
    ok, _, used = sb.install_globally(state)
    assert ok and used == "uv"
    assert calls == ["uv"]


def test_install_chain_falls_back_to_pipx(monkeypatch, isolated):
    state = sb.State(uv_path="/u/b/uv", pipx_path="/u/b/pipx")
    monkeypatch.setattr(sb, "install_via_uv",
                        lambda s, *, dry_run=False: (False, "broke"))
    monkeypatch.setattr(sb, "install_via_pipx",
                        lambda s, *, dry_run=False: (True, "ok"))
    ok, _, used = sb.install_globally(state)
    assert ok and used == "pipx"


def test_install_chain_bootstraps_uv_when_nothing(monkeypatch, isolated):
    state = sb.State()
    bootstrap_called = []
    monkeypatch.setattr(sb, "bootstrap_uv_via_curl",
                        lambda *, dry_run=False:
                        bootstrap_called.append(True) or (True, "ok"))
    monkeypatch.setattr(sb, "detect_state",
                        lambda *a, **kw: sb.State(uv_path="/u/b/uv"))
    monkeypatch.setattr(sb, "install_via_uv",
                        lambda s, *, dry_run=False: (True, "ok"))
    ok, msg, used = sb.install_globally(state)
    assert ok
    assert used == "uv-bootstrapped"
    assert bootstrap_called == [True]


def test_install_chain_dry_run(monkeypatch, isolated):
    state = sb.State(uv_path="/u/b/uv")
    called = []
    monkeypatch.setattr(sb, "_run",
                        lambda c, *, timeout=600:
                        called.append(c) or (False, "should not run"))
    ok, msg, used = sb.install_globally(state, dry_run=True)
    assert ok and "DRY-RUN" in msg
    assert called == []


def test_patch_codex_config_creates(isolated):
    _, codex = isolated
    state = sb.State(serena_path="/u/b/serena-mcp-server")
    ok, _ = sb.patch_codex_config(state, config_path=codex)
    assert ok
    txt = codex.read_text()
    assert sb._SERENA_CONFIG_MARKER in txt
    assert '"/u/b/serena-mcp-server"' in txt


def test_patch_codex_config_idempotent(isolated):
    _, codex = isolated
    state = sb.State(serena_path="/u/b/serena-mcp-server")
    sb.patch_codex_config(state, config_path=codex)
    snapshot = codex.read_text()
    state2 = sb.detect_state(codex)
    assert state2.codex_config_has_serena is True
    sb.patch_codex_config(state2, config_path=codex)
    assert codex.read_text() == snapshot


def test_strip_block_keeps_neighbors(isolated):
    text = (
        "[mcp_servers.advisor]\n"
        'cmd = "x"\n'
        "\n"
        "# Added by tinyctx (serena_bootstrap)\n"
        "[mcp_servers.serena]\n"
        'type = "stdio"\n'
        '\n'
        "[mcp_servers.gitnexus]\n"
        'cmd = "y"\n'
    )
    out = sb.strip_mcp_block(text, sb._SERENA_CONFIG_MARKER)
    assert "[mcp_servers.serena]" not in out
    assert "[mcp_servers.advisor]" in out
    assert "[mcp_servers.gitnexus]" in out


def test_bootstrap_already_present_only_patches_config(monkeypatch, isolated):
    _, codex = isolated
    monkeypatch.setattr(sb, "_which",
                        lambda c: "/u/b/serena-mcp-server"
                        if c == "serena-mcp-server" else "")
    install_called = []
    monkeypatch.setattr(sb, "install_globally",
                        lambda s, *, dry_run=False:
                        install_called.append(True) or (True, "no", "x"))
    report = sb.bootstrap(codex_config=codex)
    assert install_called == []
    assert codex.is_file()
    assert sb._SERENA_CONFIG_MARKER in codex.read_text()


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
