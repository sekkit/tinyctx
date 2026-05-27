"""Tests for `mm_bootstrap` — detection logic only.

Auto-install runs an external `curl | sh`, which we don't exercise in
unit tests. The detection paths around PATH lookup, env overrides, and
the disabled flag are what matter for correctness.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tinyctx import mm_bootstrap as mmb


def test_detect_when_disabled(monkeypatch):
    monkeypatch.setenv("TINYCTX_MM_DISABLE", "1")
    state = mmb.detect_state()
    assert state.disabled is True


def test_detect_not_present(monkeypatch):
    monkeypatch.delenv("TINYCTX_MM_DISABLE", raising=False)
    monkeypatch.delenv("TINYCTX_MM_BIN", raising=False)
    monkeypatch.setattr(mmb, "_which", lambda *_: "")
    state = mmb.detect_state()
    assert state.mm_present is False
    assert state.mm_path == ""
    assert state.disabled is False


def test_detect_with_forced_bin(monkeypatch, tmp_path):
    fake_bin = tmp_path / "mm"
    fake_bin.write_text("#!/bin/sh\necho mm 0.1.0\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("TINYCTX_MM_BIN", str(fake_bin))
    state = mmb.detect_state()
    assert state.mm_present is True
    assert state.mm_path == str(fake_bin)
    # Version may or may not parse cleanly depending on shell — just check
    # the detector didn't crash trying to read it.


def test_install_skipped_when_present(monkeypatch, tmp_path):
    fake_bin = tmp_path / "mm"
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(mmb, "_which", lambda *_: str(fake_bin))
    monkeypatch.delenv("TINYCTX_MM_DISABLE", raising=False)
    result = mmb.install()
    assert result.get("skipped") is True
    assert result.get("installed") is False


def test_install_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("TINYCTX_MM_DISABLE", "1")
    result = mmb.install()
    assert result == {"installed": False, "skipped": True, "reason": "disabled"}


def test_installer_registry_includes_mm():
    """mm should be in the unified installer's _INSTALLERS list AND
    show up in status_all()."""
    from tinyctx import installer
    install_names = [fn.__name__ for fn in installer._INSTALLERS]
    assert "_install_mm" in install_names

    status = installer.status_all()
    assert "mm" in status
    assert "installed" in status["mm"]


def test_install_mm_returns_fast_when_missing(monkeypatch):
    """_install_mm must not block startup when mm is missing.

    Live trace 2026-05-26: synchronous curl|sh install took ~10min
    while uv downloaded mm-ctx, blocking the proxy startup hook.
    Default behavior must spawn the install in a daemon thread.
    """
    import time as _t
    from tinyctx import installer, mm_bootstrap

    # Pretend mm is missing.
    monkeypatch.setattr(mm_bootstrap, "detect_state", lambda: mm_bootstrap.MmState(
        mm_present=False, mm_path="", version="", disabled=False))

    # If background spawn fired, `install` is invoked from a thread —
    # confirm it WAS scheduled but the call returned in <1s. We patch
    # the actual install to a slow stub that we never wait for.
    install_called = {"n": 0}
    def _slow_install():
        install_called["n"] += 1
        _t.sleep(10)  # would block startup if called inline
        return {"installed": True}
    monkeypatch.setattr(mm_bootstrap, "install", _slow_install)
    monkeypatch.delenv("TINYCTX_MM_SYNC_INSTALL", raising=False)

    started = _t.time()
    r = installer._install_mm()
    elapsed = _t.time() - started
    assert elapsed < 1.5, (
        f"_install_mm blocked for {elapsed:.1f}s — startup hook would hang")
    assert r.was_missing is True
    assert any("background" in a for a in r.actions)
    # Give the daemon a moment to schedule (not to finish).
    _t.sleep(0.1)
    assert install_called["n"] == 1


def test_install_mm_sync_path_via_env(monkeypatch):
    """When TINYCTX_MM_SYNC_INSTALL=1, _install_mm blocks until done.
    Used by diagnostics."""
    from tinyctx import installer, mm_bootstrap

    monkeypatch.setattr(mm_bootstrap, "detect_state", lambda: mm_bootstrap.MmState(
        mm_present=False, mm_path="", version="", disabled=False))
    monkeypatch.setattr(mm_bootstrap, "install",
                        lambda: {"installed": True, "path": "/fake/mm"})
    monkeypatch.setenv("TINYCTX_MM_SYNC_INSTALL", "1")
    r = installer._install_mm()
    assert r.was_missing is True
    assert r.installed is True
    assert any("sync" in a for a in r.actions)
