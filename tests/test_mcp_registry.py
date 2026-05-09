"""Auto-MCP registration: detect external code-graph tools and idempotently
write their codex-config blocks. Tests guard the contract:

  - detection returns None when binary isn't on PATH (no false positives)
  - graphify is detected but NOT registered as MCP (single-graph limitation)
  - gitnexus is detected AND registered as MCP
  - the managed block is bracketed by stable markers
  - re-running without changes is a no-op (no churn)
  - re-running adds/removes tools without touching the rest of the file
  - missing config file → graceful no-op
  - backup is made before the first write of the day
  - coexists with explicit per-section bootstraps (gitnexus_bootstrap, etc.):
    if a `[mcp_servers.<name>]` already exists OUTSIDE the BEGIN/END managed
    block, mcp_registry skips it instead of producing a duplicate key
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest


# ─── detection ─────────────────────────────────────────────────────────────


def test_detect_gitnexus_returns_none_when_missing(monkeypatch):
    from tinyctx import mcp_registry
    monkeypatch.setattr("shutil.which", lambda c: None)
    assert mcp_registry.detect_gitnexus() is None


def test_detect_gitnexus_returns_tool_with_section_when_present(monkeypatch):
    from tinyctx import mcp_registry
    monkeypatch.setattr("shutil.which",
                        lambda c: "/usr/local/bin/gitnexus" if c == "gitnexus" else None)
    t = mcp_registry.detect_gitnexus()
    assert t is not None
    assert t.name == "gitnexus"
    assert t.binary_path == "/usr/local/bin/gitnexus"
    assert "[mcp_servers.gitnexus]" in t.section_toml
    assert '"mcp"' in t.section_toml  # the args field
    # license caveat must be surfaced for downstream logging
    assert "PolyForm" in t.license_note
    assert "Noncommercial" in t.license_note


def test_detect_graphify_returns_tool_with_empty_section(monkeypatch):
    """graphify is detected but intentionally NOT registered (single-graph
    MCP, codex MCPs are persistent → would lock to one project)."""
    from tinyctx import mcp_registry
    monkeypatch.setattr("shutil.which",
                        lambda c: "/usr/local/bin/graphify" if c == "graphify" else None)
    t = mcp_registry.detect_graphify()
    assert t is not None
    assert t.name == "graphify"
    # intentionally empty: don't register graphify as MCP
    assert t.section_toml == ""
    assert "MIT" in t.license_note


def test_detect_all_combines_present_tools(monkeypatch):
    from tinyctx import mcp_registry
    fake = {"graphify": "/usr/g", "gitnexus": "/usr/n"}
    monkeypatch.setattr("shutil.which", lambda c: fake.get(c))
    out = mcp_registry.detect_all()
    names = sorted(t.name for t in out)
    assert names == ["gitnexus", "graphify"]


# ─── config writer ──────────────────────────────────────────────────────────


def _fake_tool(name: str = "gitnexus", path: str = "/usr/g") -> "DetectedTool":
    from tinyctx.mcp_registry import DetectedTool
    return DetectedTool(
        name=name,
        binary_path=path,
        section_toml=f'[mcp_servers.{name}]\ntype = "stdio"\ncommand = "{path}"\nargs = ["mcp"]\n',
        license_note="test license",
    )


def test_register_in_codex_config_inserts_managed_block(tmp_path):
    """First-time write: original config gets the managed block appended
    between BEGIN/END markers, original content untouched."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'model_provider = "tinyctx"\n'
        'model = "gpt-5.5"\n'
        '\n'
        '[mcp_servers.context-mode]\ncommand = "context-mode"\n',
        encoding="utf-8",
    )
    changed, msg = mcp_registry.register_in_codex_config(
        [_fake_tool()], config_path=cfg)
    assert changed is True
    final = cfg.read_text()
    # Original survived
    assert 'model_provider = "tinyctx"' in final
    assert '[mcp_servers.context-mode]' in final
    # Managed block added with markers
    assert mcp_registry.MANAGED_BEGIN in final
    assert mcp_registry.MANAGED_END in final
    assert '[mcp_servers.gitnexus]' in final


def test_register_is_idempotent_byte_stable(tmp_path):
    """Calling register twice with identical input must not change the
    file the second time (no churn → cache-friendly + safe to call on
    every proxy startup)."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text("# user config\nmodel = 'x'\n", encoding="utf-8")
    changed1, _ = mcp_registry.register_in_codex_config(
        [_fake_tool()], config_path=cfg)
    assert changed1 is True
    snapshot = cfg.read_text()
    changed2, msg = mcp_registry.register_in_codex_config(
        [_fake_tool()], config_path=cfg)
    assert changed2 is False
    assert "in sync" in msg
    assert cfg.read_text() == snapshot


def test_register_replaces_old_managed_block_in_place(tmp_path):
    """When the set of detected tools changes, register must REPLACE the
    old managed block (not duplicate it)."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text("[other]\nfoo = 'bar'\n", encoding="utf-8")
    # Round 1 — gitnexus only
    mcp_registry.register_in_codex_config([_fake_tool("gitnexus", "/g")],
                                          config_path=cfg)
    text1 = cfg.read_text()
    assert text1.count(mcp_registry.MANAGED_BEGIN) == 1
    assert "[mcp_servers.gitnexus]" in text1
    # Round 2 — different tool (simulating a new addition / config change)
    mcp_registry.register_in_codex_config([_fake_tool("othersrv", "/o")],
                                          config_path=cfg)
    text2 = cfg.read_text()
    # Still exactly ONE managed block (not two)
    assert text2.count(mcp_registry.MANAGED_BEGIN) == 1
    assert text2.count(mcp_registry.MANAGED_END) == 1
    # Old tool gone, new tool present
    assert "[mcp_servers.gitnexus]" not in text2
    assert "[mcp_servers.othersrv]" in text2
    # Non-managed content survives both rounds
    assert "[other]" in text2
    assert "foo = 'bar'" in text2


def test_register_no_op_when_no_registerable_tools(tmp_path):
    """graphify (empty section_toml) alone → no-op."""
    from tinyctx import mcp_registry
    from tinyctx.mcp_registry import DetectedTool
    cfg = tmp_path / "config.toml"
    cfg.write_text("a = 1\n", encoding="utf-8")
    snapshot = cfg.read_text()
    changed, msg = mcp_registry.register_in_codex_config(
        [DetectedTool(name="graphify", binary_path="/g", section_toml="",
                       license_note="MIT")],
        config_path=cfg,
    )
    assert changed is False
    assert "no MCP-registerable" in msg
    assert cfg.read_text() == snapshot


def test_register_no_op_when_config_missing(tmp_path):
    """Don't create codex's config out of thin air; codex owns that file.
    If it's missing, just log and skip."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "does-not-exist.toml"
    changed, msg = mcp_registry.register_in_codex_config(
        [_fake_tool()], config_path=cfg)
    assert changed is False
    assert "not found" in msg
    assert not cfg.exists()


def test_register_makes_backup_before_first_write_of_day(tmp_path):
    """First write of the day → backup file appears with today's date."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text("# original\n", encoding="utf-8")
    mcp_registry.register_in_codex_config([_fake_tool()], config_path=cfg)
    backups = list(cfg.parent.glob(cfg.name + ".tinyctx-bak.*"))
    assert len(backups) == 1
    # Backup retains the ORIGINAL content (not the new content)
    assert backups[0].read_text() == "# original\n"


# ─── unregister ─────────────────────────────────────────────────────────────


def test_unregister_removes_managed_block_only(tmp_path):
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text("[user]\nx = 1\n", encoding="utf-8")
    mcp_registry.register_in_codex_config([_fake_tool()], config_path=cfg)
    assert "[mcp_servers.gitnexus]" in cfg.read_text()

    changed, msg = mcp_registry.unregister_from_codex_config(config_path=cfg)
    assert changed is True
    final = cfg.read_text()
    assert mcp_registry.MANAGED_BEGIN not in final
    assert mcp_registry.MANAGED_END not in final
    assert "[mcp_servers.gitnexus]" not in final
    # User config preserved
    assert "[user]" in final
    assert "x = 1" in final


def test_unregister_no_op_when_no_managed_block(tmp_path):
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text("[user]\nx = 1\n", encoding="utf-8")
    snap = cfg.read_text()
    changed, msg = mcp_registry.unregister_from_codex_config(config_path=cfg)
    assert changed is False
    assert "nothing to remove" in msg
    assert cfg.read_text() == snap


# ─── bootstrap orchestration ────────────────────────────────────────────────


def test_bootstrap_logs_when_no_tools_present(monkeypatch, tmp_path):
    from tinyctx import mcp_registry
    monkeypatch.setattr("shutil.which", lambda c: None)
    events: list = []
    def lf(ev, **fields): events.append((ev, fields))
    out = mcp_registry.bootstrap(config_path=tmp_path / "x.toml", log_fn=lf)
    assert out["detected"] == []
    assert out["changed"] is False
    assert any(ev == "mcp_registry_no_tools" for ev, _ in events)


def test_bootstrap_registers_when_tools_present(monkeypatch, tmp_path):
    from tinyctx import mcp_registry
    fake = {"gitnexus": "/usr/g", "graphify": "/usr/p"}
    monkeypatch.setattr("shutil.which", lambda c: fake.get(c))
    cfg = tmp_path / "config.toml"
    cfg.write_text("# initial\n", encoding="utf-8")
    events: list = []
    def lf(ev, **fields): events.append((ev, fields))
    out = mcp_registry.bootstrap(config_path=cfg, log_fn=lf)
    assert sorted(out["detected"]) == ["gitnexus", "graphify"]
    assert out["changed"] is True
    final = cfg.read_text()
    # gitnexus IS registered, graphify is NOT
    assert "[mcp_servers.gitnexus]" in final
    assert "[mcp_servers.graphify]" not in final
    # license-warning event for gitnexus was logged
    detected_evs = [f for ev, f in events if ev == "mcp_registry_detected"]
    licenses = {f["tool"]: f["license_note"] for f in detected_evs}
    assert "PolyForm" in licenses["gitnexus"]
    assert "MIT" in licenses["graphify"]
    # restart-required notice
    assert any(ev == "mcp_registry_codex_restart_required" for ev, _ in events)


def test_bootstrap_idempotent_across_restarts(monkeypatch, tmp_path):
    """Realistic scenario: proxy crashes, launchd respawns, bootstrap
    runs again — config must NOT be touched the second time."""
    from tinyctx import mcp_registry
    monkeypatch.setattr("shutil.which",
                        lambda c: "/usr/g" if c == "gitnexus" else None)
    cfg = tmp_path / "config.toml"
    cfg.write_text("# user\n", encoding="utf-8")
    out1 = mcp_registry.bootstrap(config_path=cfg, log_fn=lambda *a, **k: None)
    snap = cfg.read_text()
    out2 = mcp_registry.bootstrap(config_path=cfg, log_fn=lambda *a, **k: None)
    assert out1["changed"] is True
    assert out2["changed"] is False
    assert cfg.read_text() == snap


# ─── default config flag ────────────────────────────────────────────────────


def test_auto_register_mcp_defaults_to_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.auto_register_mcp_servers is True


# ─── coexistence with explicit per-section bootstraps ─────────────────────


def test_register_skips_tool_when_already_in_outside_section(tmp_path):
    """When tinyctx-gitnexus install (or some other tool) has already
    written `[mcp_servers.gitnexus]` outside the BEGIN/END managed block,
    mcp_registry must DETECT and SKIP it. Otherwise codex's TOML parser
    rejects the file with `duplicate key`. This is the exact bug we hit
    in production on 2026-05-10."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "model = 'x'\n\n"
        "# Added by tinyctx (gitnexus_bootstrap)\n"
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/u/g"\n',
        encoding="utf-8",
    )
    snap_before = cfg.read_text()
    changed, msg = mcp_registry.register_in_codex_config(
        [_fake_tool("gitnexus", "/u/g")], config_path=cfg)
    assert changed is False
    assert "already registered outside the managed block" in msg
    # NO managed block introduced (would have caused duplicate)
    assert mcp_registry.MANAGED_BEGIN not in cfg.read_text()
    assert cfg.read_text() == snap_before
    # Exactly one [mcp_servers.gitnexus] in the file
    assert sum(1 for ln in cfg.read_text().splitlines()
               if ln.strip() == "[mcp_servers.gitnexus]") == 1


def test_register_partial_skip_keeps_other_tools(tmp_path):
    """If we have 2 detected tools and one is already pre-registered
    outside the block, the other one still goes into the managed block."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "/u/g"\n',
        encoding="utf-8",
    )
    changed, msg = mcp_registry.register_in_codex_config(
        [_fake_tool("gitnexus", "/u/g"),
         _fake_tool("othersrv", "/u/o")],
        config_path=cfg,
    )
    assert changed is True
    final = cfg.read_text()
    assert "skipped 1" in msg
    assert "gitnexus" in msg
    # Only one [mcp_servers.gitnexus] (preserved from outside)
    assert sum(1 for ln in final.splitlines()
               if ln.strip() == "[mcp_servers.gitnexus]") == 1
    # The new tool DID land in managed block
    assert "[mcp_servers.othersrv]" in final
    assert mcp_registry.MANAGED_BEGIN in final


def test_register_does_not_match_marker_inside_comment(tmp_path):
    """A comment that mentions `[mcp_servers.gitnexus]` (e.g. user
    documentation) must NOT cause mcp_registry to skip writing the
    real section. Line-exact check, not substring."""
    from tinyctx import mcp_registry
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "# documentation: see [mcp_servers.gitnexus] format docs at ...\n"
        "model = 'x'\n",
        encoding="utf-8",
    )
    changed, _ = mcp_registry.register_in_codex_config(
        [_fake_tool("gitnexus", "/u/g")], config_path=cfg)
    assert changed is True
    final = cfg.read_text()
    # The real section was added (inside managed block)
    real_lines = [ln for ln in final.splitlines()
                  if ln.strip() == "[mcp_servers.gitnexus]"]
    assert len(real_lines) == 1
    # Doc comment preserved
    assert "see [mcp_servers.gitnexus] format docs" in final
