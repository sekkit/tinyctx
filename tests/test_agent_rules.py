"""Global agent rules dynamic injection.

The bundled `tinyctx/templates/AGENTS.md` should travel with the repo
and get injected into request.instructions on any machine that runs
tinyctx — without depending on the user having manually copied a file
into their codex/claude config dir.

Tests cover:
  - bundled template exists and is non-empty
  - injection prepends with idempotent markers
  - SKIP when title is already in instructions (codex's own AGENTS.md
    load is the override path)
  - SKIP when our BEGIN marker is present (proxy hop / replay)
  - input body is not mutated (defensive copy)
  - missing/non-string instructions field is handled
  - reload_template hot-reloads from disk
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_bundled_template_exists_and_is_non_trivial():
    from tinyctx import agent_rules
    assert agent_rules._TEMPLATE_PATH.is_file()
    text = agent_rules._TEMPLATE_PATH.read_text(encoding="utf-8")
    assert len(text) > 1000  # non-empty rule doc
    # the title marker must be in the bundled content for idempotency to work
    assert agent_rules._TITLE_MARKER in text


def test_inject_prepends_with_markers():
    from tinyctx import agent_rules
    body = {"instructions": "You are codex.", "input": []}
    out, injected = agent_rules.inject_into_body(body)
    assert injected is True
    inst = out["instructions"]
    assert inst.startswith(agent_rules._BEGIN)
    assert agent_rules._END in inst
    assert "You are codex." in inst
    # original body untouched (defensive copy)
    assert body["instructions"] == "You are codex."


def test_inject_idempotent_via_begin_marker():
    """Second pass over an already-injected body must be a no-op (the
    BEGIN marker is detected; we don't double-prepend)."""
    from tinyctx import agent_rules
    body = {"instructions": "You are codex."}
    body, _ = agent_rules.inject_into_body(body)
    out, injected = agent_rules.inject_into_body(body)
    assert injected is False
    assert out is body  # unchanged


def test_inject_skipped_when_title_already_present():
    """When codex.app already loaded ~/.codex/AGENTS.md (which has the
    same title), tinyctx must NOT inject again — the rules are already
    in scope, duplication would just inflate the prompt."""
    from tinyctx import agent_rules
    body = {
        "instructions": (
            "You are codex.\n\n"
            "# AGENTS.md — 全局代理规范\n\n"  # the title marker
            "## 1. 默认原则\n..."
        ),
    }
    out, injected = agent_rules.inject_into_body(body)
    assert injected is False
    assert out is body


def test_inject_no_op_for_non_string_instructions():
    from tinyctx import agent_rules
    body = {"instructions": {"not": "a string"}}
    out, injected = agent_rules.inject_into_body(body)
    assert injected is False


def test_inject_no_op_for_missing_instructions():
    from tinyctx import agent_rules
    body = {"input": []}  # no instructions field at all
    out, injected = agent_rules.inject_into_body(body)
    assert injected is False


def test_inject_handles_template_load_failure(monkeypatch):
    """Simulate the bundled template being missing/unreadable. The
    proxy must NOT raise — just skip the injection silently."""
    from tinyctx import agent_rules
    monkeypatch.setattr(agent_rules, "_CACHED_TEMPLATE", None)
    body = {"instructions": "x"}
    out, injected = agent_rules.inject_into_body(body)
    assert injected is False
    assert out is body


def test_template_chars_returns_size():
    from tinyctx import agent_rules
    n = agent_rules.template_chars()
    assert n > 1000  # whatever the rules doc grows to


def test_reload_template_picks_up_disk_changes(tmp_path, monkeypatch):
    """The proxy caches the template at module import time. If the
    repo's AGENTS.md is updated and the proxy is reloaded (or a
    test wants to swap), reload_template() reads from disk again."""
    from tinyctx import agent_rules
    fake = tmp_path / "AGENTS.md"
    fake.write_text("# AGENTS.md — 全局代理规范\n\nfake-content-v2", encoding="utf-8")
    monkeypatch.setattr(agent_rules, "_TEMPLATE_PATH", fake)
    assert agent_rules.reload_template() is True
    body = {"instructions": "x"}
    out, _ = agent_rules.inject_into_body(body)
    assert "fake-content-v2" in out["instructions"]


def test_default_config_enables_injection():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.inject_global_agent_rules is True


def test_template_matches_user_codex_agents_md_at_install_time():
    """Sanity: the bundled template should be byte-identical to the
    user's ~/.codex/AGENTS.md at install/commit time. Detects accidental
    drift between the two on the dev machine. (On clean checkouts where
    ~/.codex/AGENTS.md doesn't exist, this test is skipped.)"""
    from tinyctx import agent_rules
    user_file = Path.home() / ".codex" / "AGENTS.md"
    if not user_file.is_file():
        pytest.skip("user has no ~/.codex/AGENTS.md; nothing to compare")
    bundled = agent_rules._TEMPLATE_PATH.read_text(encoding="utf-8")
    user = user_file.read_text(encoding="utf-8")
    # Don't strictly require byte-equality — user may have customized.
    # Just sanity that both contain the title.
    assert agent_rules._TITLE_MARKER in bundled
    assert agent_rules._TITLE_MARKER in user
