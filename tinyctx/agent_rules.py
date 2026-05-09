"""Inject the global agent-rules document into every request's
`body.instructions` so the rules apply on any machine that runs tinyctx,
without depending on codex.app/Claude Code having loaded
`~/.codex/AGENTS.md` or `~/.claude/CLAUDE.md` (those user files don't
ship with the repo and don't exist on a fresh clone).

The bundled rules live at `tinyctx/templates/AGENTS.md`, version-
controlled with the rest of the codebase. `git log` against that file
shows when rules changed and why.

Resolution + injection contract:

  1. If `body.instructions` already contains the rules title
     (`AGENTS.md — 全局代理规范`), skip injection. This handles two
     legitimate cases without duplicating content:
       - User has `~/.codex/AGENTS.md` and codex.app loaded it into
         instructions before we saw the request.
       - A previous tinyctx pass already injected (proxy hop / replay).
  2. Else, prepend the bundled template content with idempotent
     BEGIN/END markers. Cache-friendly: the prefix is byte-stable
     across requests, so prompt-cache hits in the upstream model.

Override paths:
  - `~/.codex/AGENTS.md` exists → codex.app loads it, our injection is
    skipped (user's customization wins).
  - User can disable entirely via `cfg.inject_global_agent_rules=False`.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "AGENTS.md"

# Two markers: a generic title we use for "is the content already there"
# detection (so codex's own AGENTS.md load short-circuits us), plus
# tinyctx-specific BEGIN/END so we can recognize OUR injections vs
# codex's, for hop/replay idempotency and future cleanup.
_TITLE_MARKER = "AGENTS.md — 全局代理规范"
_BEGIN = "<!-- tinyctx global-agent-rules BEGIN -->"
_END = "<!-- tinyctx global-agent-rules END -->"


def _load_template() -> str | None:
    """Read the bundled AGENTS.md template. Returns None on any failure
    (the proxy then skips injection silently — never blocks)."""
    try:
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return None


# Cache the template at import time so repeated calls don't hit disk.
# It's a few KB, no point re-reading per request. Clear via the
# explicit reload helper below for hot-reload during dev.
_CACHED_TEMPLATE: str | None = _load_template()


def reload_template() -> bool:
    """Refresh the cached template from disk (test/dev hot-reload helper)."""
    global _CACHED_TEMPLATE
    _CACHED_TEMPLATE = _load_template()
    return _CACHED_TEMPLATE is not None


def inject_into_body(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Prepend the bundled global agent rules to `body.instructions`.

    Returns (new_body, was_injected). Idempotent — second pass is no-op.
    Skips silently when:
      - the body's instructions field is missing or non-string;
      - the title marker is already present (rules already in scope);
      - the bundled template couldn't be loaded.
    """
    if _CACHED_TEMPLATE is None:
        return body, False
    if not isinstance(body, dict):
        return body, False
    inst = body.get("instructions")
    if not isinstance(inst, str):
        return body, False
    # already there from any source (codex's own load, or our previous pass)
    if _TITLE_MARKER in inst or _BEGIN in inst:
        return body, False
    block = (
        f"{_BEGIN}\n"
        f"{_CACHED_TEMPLATE.strip()}\n"
        f"{_END}\n\n"
    )
    out = dict(body)
    out["instructions"] = block + inst
    return out, True


def template_chars() -> int:
    """Cached template size in chars. Useful for trace metrics."""
    return len(_CACHED_TEMPLATE) if _CACHED_TEMPLATE else 0
