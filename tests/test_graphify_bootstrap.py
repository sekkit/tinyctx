"""Tests for tinyctx.graphify_bootstrap."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from tinyctx import graphify_bootstrap as gb


@pytest.fixture
def isolated_home(monkeypatch):
    with TemporaryDirectory(prefix="tinyctx-graphify-test-") as td:
        home = Path(td)
        monkeypatch.setattr(gb, "TINYCTX_HOME", home)
        monkeypatch.setattr(gb, "LOG_FILE", home / "logs" / "boot.log")
        yield home


# ──────────────────── detect_state ────────────────────


def test_state_all_missing(monkeypatch, isolated_home):
    monkeypatch.setattr(gb, "_which", lambda c: "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    s = gb.detect_state()
    assert s.graphify_present is False
    assert s.uv_path == ""
    assert s.pipx_path == ""
    assert s.python_310_plus_path == ""


def test_state_uv_only(monkeypatch, isolated_home):
    monkeypatch.setattr(gb, "_which",
                        lambda c: "/u/b/uv" if c == "uv" else "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    s = gb.detect_state()
    assert s.uv_path == "/u/b/uv"
    assert s.graphify_present is False
    assert s.pipx_path == ""


def test_state_graphify_already_installed(monkeypatch, isolated_home):
    monkeypatch.setattr(gb, "_which",
                        lambda c: "/u/b/graphify" if c == "graphify" else "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    s = gb.detect_state()
    assert s.graphify_present is True
    assert s.graphify_path == "/u/b/graphify"


def test_disabled_env_short_circuits(monkeypatch, isolated_home):
    monkeypatch.setenv("TINYCTX_GRAPHIFY_DISABLE", "1")
    monkeypatch.setattr(gb, "_which", lambda c: "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    report = gb.bootstrap()
    assert "TINYCTX_GRAPHIFY_DISABLE=1" in report.skipped
    # skipped doesn't fail; final success still True
    assert report.success is True


# ──────────────────── installer chain ────────────────────


def test_install_chain_prefers_uv(monkeypatch, isolated_home):
    state = gb.State(uv_path="/u/b/uv", pipx_path="/u/b/pipx",
                     python_310_plus_path="/u/b/python3.11")
    calls: list[str] = []

    def _fake_uv(s, *, dry_run=False):
        calls.append("uv"); return True, "ok"

    def _fake_pipx(s, *, dry_run=False):
        calls.append("pipx"); return True, "ok"

    monkeypatch.setattr(gb, "install_via_uv", _fake_uv)
    monkeypatch.setattr(gb, "install_via_pipx", _fake_pipx)
    ok, _, used = gb.install_globally(state)
    assert ok and used == "uv"
    assert calls == ["uv"]   # pipx never called


def test_install_chain_falls_through_to_pipx(monkeypatch, isolated_home):
    state = gb.State(uv_path="/u/b/uv", pipx_path="/u/b/pipx")

    monkeypatch.setattr(gb, "install_via_uv",
                        lambda s, *, dry_run=False: (False, "uv broke"))
    monkeypatch.setattr(gb, "install_via_pipx",
                        lambda s, *, dry_run=False: (True, "ok"))
    ok, _, used = gb.install_globally(state)
    assert ok and used == "pipx"


def test_install_chain_pip_user_when_only_python(monkeypatch, isolated_home):
    state = gb.State(python_310_plus_path="/u/b/python3.11")
    monkeypatch.setattr(gb, "install_via_pip_user",
                        lambda s, *, dry_run=False: (True, "ok"))
    ok, _, used = gb.install_globally(state)
    assert ok and used == "pip-user"


def test_install_chain_bootstraps_uv_when_nothing_else(monkeypatch,
                                                       isolated_home):
    """When uv/pipx/python3.10+ all missing, fall back to curl-installing
    uv if auto_uv_allowed=True. Verify the chain calls bootstrap_uv_via_curl
    and then re-attempts uv install."""
    state = gb.State(auto_uv_allowed=True)

    bootstrap_called = []
    monkeypatch.setattr(gb, "bootstrap_uv_via_curl",
                        lambda *, dry_run=False: (
                            bootstrap_called.append(True) or (True, "ok")))
    # After bootstrap, detect_state will re-fire and find uv:
    monkeypatch.setattr(gb, "detect_state",
                        lambda: gb.State(uv_path="/u/b/uv",
                                          auto_uv_allowed=True))
    monkeypatch.setattr(gb, "install_via_uv",
                        lambda s, *, dry_run=False: (True, "ok"))
    ok, msg, used = gb.install_globally(state)
    assert ok
    assert used == "uv-bootstrapped"
    assert bootstrap_called == [True]


def test_install_chain_skips_curl_when_auto_uv_disabled(monkeypatch,
                                                        isolated_home):
    state = gb.State(auto_uv_allowed=False)
    called = []
    monkeypatch.setattr(gb, "bootstrap_uv_via_curl",
                        lambda *, dry_run=False:
                        (called.append(True) or (True, "ok")))
    ok, msg, used = gb.install_globally(state)
    assert ok is False
    assert called == [], "must NOT bootstrap uv when auto disabled"
    assert "no installer available" in msg


def test_resolve_install_source_pypi_default(monkeypatch, isolated_home):
    """Default: install from PyPI, not from git. PyPI keeps up with git
    tags within ~1 minute, so git+ adds no value for typical users."""
    monkeypatch.setattr(gb, "GRAPHIFY_GIT_REF", "")
    args, label = gb._resolve_install_source()
    assert label == "pypi"
    assert args == [gb.GRAPHIFY_PYPI_PKG]


def test_resolve_install_source_git_ref_overrides_to_main(monkeypatch,
                                                           isolated_home):
    """When TINYCTX_GRAPHIFY_GIT_REF is set, install from git@ref using
    `--from git+url@ref pkg-name`. Useful when chasing main HEAD between
    PyPI publishes."""
    monkeypatch.setattr(gb, "GRAPHIFY_GIT_REF", "main")
    args, label = gb._resolve_install_source()
    assert label == "git@main"
    assert args == ["--from",
                    f"git+{gb.GRAPHIFY_GIT_URL}@main",
                    gb.GRAPHIFY_PYPI_PKG]


def test_pip_target_arg_pypi_vs_git(monkeypatch, isolated_home):
    monkeypatch.setattr(gb, "GRAPHIFY_GIT_REF", "")
    assert gb._pip_target_arg() == gb.GRAPHIFY_PYPI_PKG
    monkeypatch.setattr(gb, "GRAPHIFY_GIT_REF", "abc123")
    assert gb._pip_target_arg() == (
        f"{gb.GRAPHIFY_PYPI_PKG} @ git+{gb.GRAPHIFY_GIT_URL}@abc123")


def test_install_via_uv_uses_git_when_ref_set(monkeypatch, isolated_home):
    """install_via_uv must construct the right `uv tool install --from`
    command when GRAPHIFY_GIT_REF is set, not the bare-pkg form."""
    monkeypatch.setattr(gb, "GRAPHIFY_GIT_REF", "main")
    state = gb.State(uv_path="/u/b/uv")
    captured: list[list[str]] = []

    def _fake_run(cmd, *, timeout):
        captured.append(cmd)
        return True, "ok"
    monkeypatch.setattr(gb, "_run", _fake_run)
    ok, msg = gb.install_via_uv(state)
    assert ok
    assert captured == [["/u/b/uv", "tool", "install",
                         "--from",
                         f"git+{gb.GRAPHIFY_GIT_URL}@main",
                         gb.GRAPHIFY_PYPI_PKG]]
    assert "git@main" in msg


def test_install_chain_dry_run_does_not_run(monkeypatch, isolated_home):
    state = gb.State(uv_path="/u/b/uv")
    called = []

    def _fake(cmd, *, timeout=600):
        called.append(cmd)
        return False, "should not be called"

    monkeypatch.setattr(gb, "_run", _fake)
    ok, msg, used = gb.install_globally(state, dry_run=True)
    assert ok and "DRY-RUN" in msg
    assert called == []


# ──────────────────── per-project install ────────────────────


def test_install_into_project_no_binary(isolated_home):
    state = gb.State(graphify_path="")
    with TemporaryDirectory() as td:
        ok, msg = gb.install_into_project(Path(td), state)
    assert ok is False
    assert "binary not available" in msg


def test_install_into_project_missing_dir(isolated_home):
    state = gb.State(graphify_path="/u/b/graphify")
    ok, msg = gb.install_into_project(Path("/nonexistent/abs/path"), state)
    assert ok is False
    assert "missing" in msg


def test_install_into_project_dry_run(isolated_home):
    state = gb.State(graphify_path="/u/b/graphify")
    with TemporaryDirectory() as td:
        ok, msg = gb.install_into_project(Path(td), state, dry_run=True)
    assert ok is True
    assert "DRY-RUN" in msg
    assert "graphify codex install" in msg


def test_install_into_project_idempotent_already(monkeypatch, isolated_home):
    """When graphify says 'already configured', return ok with that
    message rather than treating it as a problem."""
    state = gb.State(graphify_path="/u/b/graphify")

    class _R:
        returncode = 0
        stdout = "graphify already configured in AGENTS.md\n"
        stderr = ""
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: _R())
    with TemporaryDirectory() as td:
        ok, msg = gb.install_into_project(Path(td), state)
    assert ok is True
    assert "already" in msg


# ──────────────────── full bootstrap ────────────────────


def test_bootstrap_no_install_when_missing_no_installer(monkeypatch,
                                                        isolated_home):
    monkeypatch.setattr(gb, "_which", lambda c: "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    monkeypatch.setattr(gb, "bootstrap_uv_via_curl",
                        lambda *, dry_run=False: (False, "no curl"))
    report = gb.bootstrap()
    assert report.success is True  # missing installer is not a hard fail
    assert any("no installer" in a or "all installers failed" in a
               for a in report.actions)


def test_bootstrap_already_present_only_does_per_project(monkeypatch,
                                                         isolated_home):
    monkeypatch.setattr(gb, "_which",
                        lambda c: "/u/b/graphify" if c == "graphify" else "")
    monkeypatch.setattr(gb, "_find_python_310_plus", lambda: "")
    install_globally_called = []
    monkeypatch.setattr(gb, "install_globally",
                        lambda s, *, dry_run=False:
                        install_globally_called.append(True) or
                        (True, "should not be called", "x"))
    with TemporaryDirectory() as td:
        report = gb.bootstrap(project_roots=[Path(td)], dry_run=True)
    assert install_globally_called == [], (
        "install_globally must not run when graphify already on PATH")
    # the dry-run project install path was hit
    assert any("DRY-RUN" in a for a in report.actions)


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
