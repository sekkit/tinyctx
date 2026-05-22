"""Auto-register external MCP servers in codex's config on proxy startup.

What this does (and what it doesn't):

  ✓ Detect graphify (safishamsi/graphify) and gitnexus
    (abhigyanpatwari/GitNexus) installations on the user's PATH.
  ✓ Idempotently insert their MCP server entries into
    ~/.codex/config.toml between explicit BEGIN/END markers so we can
    update or remove them on subsequent runs without leaving stale
    sections.
  ✓ Back up the config file before any write (one backup per day,
    `<file>.tinyctx-bak.<YYYYMMDD>`).
  ✓ Log a license warning when gitnexus is detected (PolyForm
    Noncommercial 1.0.0 — fine for personal use, restricted for
    commercial).
  ✓ No-op when neither tool is installed; no-op when the managed
    block is already in sync with what we'd write.

  ✗ Does NOT install the tools (no auto pipx / npm install — too
    intrusive). User installs manually:
        pipx install graphifyy        # MIT
        npm install -g gitnexus       # PolyForm Noncommercial
  ✗ Does NOT register graphify's MCP server. Reason: graphify's
    `python -m graphify.serve` binds to ONE graph.json per instance,
    and codex MCP servers are persistent (one process for all
    projects). Registering would lock graphify's MCP to a single
    project. Users instead get graphify's static project graph via
    tinyctx auto_scout (proxy.py + scout.py), which works per-project.
  ✗ Does NOT modify any non-mcp_servers section of codex config —
    only inserts/replaces a single bracketed block.
  ✗ Does NOT restart codex.app — the user must restart codex for new
    MCP servers to be loaded. We log a one-time "please restart codex"
    notice when we touch the config.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Markers bracket the tinyctx-managed block in codex's config so we can
# safely re-insert/update without touching anything else. Keep these
# strings stable forever — changing them turns the block into orphan
# user-edited config from tinyctx's POV and we'll silently leave it
# alone (idempotent-by-content).
MANAGED_BEGIN = "# --- tinyctx-managed MCP servers (auto, do not hand-edit) ---"
MANAGED_END = "# --- end tinyctx-managed MCP servers ---"

DEFAULT_CODEX_CONFIG = Path.home() / ".codex" / "config.toml"

log = logging.getLogger("tinyctx.mcp_registry")


def _is_executable_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if os.name != "nt":
        return os.access(path, os.X_OK)
    suffix = Path(path).suffix.lower()
    pathext = {
        ext.lower()
        for ext in os.environ.get(
            "PATHEXT", ".COM;.EXE;.BAT;.CMD;.PS1").split(";")
        if ext
    }
    if suffix in pathext:
        return True
    try:
        if Path(path).read_text(encoding="utf-8", errors="ignore")[:2] == "#!":
            return True
    except OSError:
        return False
    try:
        return bool(os.stat(path).st_mode & (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except OSError:
        return False


@dataclass
class DetectedTool:
    """One MCP tool discovered on the user's PATH."""
    name: str
    binary_path: str
    section_toml: str  # full [mcp_servers.<name>] block to insert
    license_note: str = ""  # human-readable license caveat (logged on detect)


# ────────────────────────────── detection ──────────────────────────────


# Common user-install locations that don't always make it into the
# launchd / systemd PATH the proxy inherits. Fixes false-negative
# 2026-05-10 where mcp_registry logged "no graphify or gitnexus on PATH"
# even though both were installed — proxy was launched by launchd
# whose default PATH excludes ~/.local/bin and ~/.local/node/bin.
#
# Order: prefer node/bin first because gitnexus + many JS-installed CLIs
# live there; user-local then system bin then homebrew.
_FALLBACK_BIN_DIRS: tuple[str, ...] = (
    "~/.local/node/bin",
    "~/.local/bin",
    "~/.cargo/bin",
    "~/.bun/bin",
    "/opt/homebrew/bin",   # Apple Silicon brew
    "/usr/local/bin",      # Intel brew + many GUI app installs
)


def _which_with_fallbacks(name: str) -> str | None:
    """`shutil.which` first; on miss, scan a small allowlist of standard
    user-install directories for an executable file with that name.
    Returns absolute path or None.

    Why: when the proxy is started by launchd / systemd / a stripped
    shell, $PATH often omits ~/.local/bin etc., causing detect_*
    to miss legitimately installed tools. The fallback scan only adds
    a handful of `os.path.isfile` checks — cheap and bounded."""
    p = shutil.which(name)
    if p:
        return p
    for d in _FALLBACK_BIN_DIRS:
        cand = os.path.join(os.path.expanduser(d), name)
        if _is_executable_file(cand):
            return cand
    return None


def detect_gitnexus() -> DetectedTool | None:
    """Locate gitnexus binary; return None if not on PATH (or fallback dirs)."""
    p = _which_with_fallbacks("gitnexus")
    if not p:
        return None
    return DetectedTool(
        name="gitnexus",
        binary_path=p,
        section_toml=(
            f"[mcp_servers.gitnexus]\n"
            f"# Static code-graph + impact analysis (Tree-sitter AST, not LLM).\n"
            f"# Multi-repo native: gitnexus has a global registry at\n"
            f"# ~/.gitnexus/registry.json. Run `gitnexus analyze <repo>` once\n"
            f"# per project before its tools become useful for that project.\n"
            f'type = "stdio"\n'
            f'command = "{p}"\n'
            f'args = ["mcp"]\n'
        ),
        license_note=(
            "GitNexus is licensed under PolyForm Noncommercial 1.0.0. "
            "Personal/internal/non-commercial use is fine; commercial use "
            "requires a separate license from akonlabs.com. Verify your "
            "use case before relying on it in commercial work."
        ),
    )


def detect_graphify() -> DetectedTool | None:
    """Locate graphify binary; return DetectedTool with NO section_toml.

    We don't register graphify as a codex MCP server (its MCP serve mode
    is single-graph per process; codex MCPs are persistent → would lock
    one project). Detection still happens so the proxy can use graphify
    in auto_scout (offline graph build for static project context).
    """
    p = _which_with_fallbacks("graphify")
    if not p:
        return None
    return DetectedTool(
        name="graphify",
        binary_path=p,
        section_toml="",  # intentionally empty — not registered as MCP
        license_note="MIT (no restrictions).",
    )


def detect_all() -> list[DetectedTool]:
    """Return every tool we know about that's actually on PATH."""
    out: list[DetectedTool] = []
    for fn in (detect_graphify, detect_gitnexus):
        t = fn()
        if t is not None:
            out.append(t)
    return out


# ────────────────────────────── config writer ──────────────────────────────


def _build_managed_block(tools: Iterable[DetectedTool]) -> str:
    """Compose the full block to be inserted between the markers.
    Stable byte-for-byte for the same input so we don't churn the file."""
    parts: list[str] = [MANAGED_BEGIN, ""]
    for t in tools:
        if not t.section_toml.strip():
            continue
        parts.append(t.section_toml.rstrip())
        parts.append("")
    parts.append(MANAGED_END)
    return "\n".join(parts) + "\n"


def _strip_existing_block(text: str) -> str:
    """Remove a previous tinyctx-managed block (between markers) if present."""
    begin_idx = text.find(MANAGED_BEGIN)
    if begin_idx < 0:
        return text
    end_idx = text.find(MANAGED_END, begin_idx)
    if end_idx < 0:
        # Half-broken block; leave alone to avoid eating user config
        return text
    end_line_break = text.find("\n", end_idx)
    if end_line_break < 0:
        end_line_break = len(text)
    # also gobble one leading \n before BEGIN if present, so we don't
    # accumulate blank lines on repeated runs
    drop_start = begin_idx
    if drop_start > 0 and text[drop_start - 1] == "\n":
        drop_start -= 1
    return text[:drop_start] + text[end_line_break + 1:]


def _backup_once_per_day(path: Path) -> Path | None:
    """Make a `<path>.tinyctx-bak.<YYYYMMDD>` copy if today's backup
    doesn't exist yet. Returns the backup path on success."""
    if not path.is_file():
        return None
    bak = path.with_name(path.name + f".tinyctx-bak.{time.strftime('%Y%m%d')}")
    if bak.exists():
        return bak
    try:
        shutil.copy2(path, bak)
        return bak
    except OSError:
        # Why: daily-backup is best-effort safety net; if copy fails
        # (no space, permissions), proceed without a backup rather
        # than abort registration. Returning None signals the caller
        # to surface "no backup" in the user message.
        return None


def register_in_codex_config(
    tools: Iterable[DetectedTool],
    *,
    config_path: Path = DEFAULT_CODEX_CONFIG,
) -> tuple[bool, str]:
    """Idempotently write the managed MCP block into the codex config.

    Returns (changed, summary). `changed` is True iff the file was actually
    rewritten (i.e. the new block differs from the existing one byte-for-
    byte). `summary` is a one-line human-readable status for logging.

    Never raises. If the config file is missing, no-op. If the only
    detected tool has empty section_toml (graphify), no-op.
    """
    tools_list = [t for t in tools if t.section_toml.strip()]
    if not tools_list:
        return False, "no MCP-registerable tools detected"

    if not config_path.is_file():
        return False, f"codex config not found at {config_path} — skipping"

    try:
        original = config_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"could not read {config_path}: {e}"

    # Coexist with explicit per-section bootstrap modules
    # (gitnexus_bootstrap / serena_bootstrap / etc.) which write
    # `[mcp_servers.<name>]` outside our BEGIN/END managed block. Both
    # paths target the same TOML keys, and codex rejects duplicates.
    # We use _codex_toml's line-exact match (NOT substring — comments
    # mentioning the marker must not trigger false positives) on the
    # text WITH our managed block stripped, so a section we previously
    # wrote ourselves doesn't false-positive.
    from . import _codex_toml as _ct
    text_outside_block = _strip_existing_block(original)
    skipped_already: list[str] = []
    final_tools: list[DetectedTool] = []
    for t in tools_list:
        marker = f"[mcp_servers.{t.name}]"
        if _ct._marker_present(text_outside_block, marker):
            skipped_already.append(t.name)
            continue
        final_tools.append(t)
    if not final_tools:
        return False, (
            f"all detected tools already registered outside the managed "
            f"block ({','.join(skipped_already)}); leaving alone"
        )

    cleaned = _strip_existing_block(original)
    new_block = _build_managed_block(final_tools)
    # Place block at end of file, with one separating blank line
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    if cleaned and not cleaned.endswith("\n\n"):
        cleaned += "\n"
    new_text = cleaned + new_block

    if new_text == original:
        return False, (
            f"managed MCP block already in sync ({len(final_tools)} tool(s))"
        )

    # Backup before write
    bak = _backup_once_per_day(config_path)
    try:
        config_path.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return False, f"could not write {config_path}: {e}"

    skipped_note = (f" (skipped {len(skipped_already)} already-registered: "
                    f"{','.join(skipped_already)})") if skipped_already else ""
    return True, (
        f"updated {config_path} with {len(final_tools)} MCP server(s){skipped_note}; "
        f"backup at {bak}"
        if bak else
        f"updated {config_path} with {len(final_tools)} MCP server(s){skipped_note} (no backup)"
    )


def unregister_from_codex_config(
    *,
    config_path: Path = DEFAULT_CODEX_CONFIG,
) -> tuple[bool, str]:
    """Remove the tinyctx-managed block. For revert / disable cases.
    Returns (changed, summary). Never raises."""
    if not config_path.is_file():
        return False, f"codex config not found at {config_path}"
    try:
        original = config_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"could not read {config_path}: {e}"
    cleaned = _strip_existing_block(original)
    if cleaned == original:
        return False, "no tinyctx-managed block present; nothing to remove"
    bak = _backup_once_per_day(config_path)
    try:
        config_path.write_text(cleaned, encoding="utf-8")
    except OSError as e:
        return False, f"could not write {config_path}: {e}"
    return True, f"removed tinyctx-managed block (backup: {bak})"


# ────────────────────────────── orchestration ──────────────────────────────


def bootstrap(
    *,
    config_path: Path = DEFAULT_CODEX_CONFIG,
    log_fn=None,
) -> dict:
    """One-shot detect-then-register entry point. Designed to be called
    from the proxy's startup hook. Always safe; never blocks.

    `log_fn(event_name, **fields)` is called for every meaningful event
    so the proxy can wire it to its JSONL log. Falls back to module-level
    logger if not supplied.
    """
    if log_fn is None:
        def log_fn(event, **fields):
            log.info("%s %s", event, fields)

    detected = detect_all()
    if not detected:
        log_fn(
            "mcp_registry_no_tools",
            note=(
                "no graphify or gitnexus on PATH; tinyctx auto-MCP-register "
                "skipped. To enable: `pipx install graphifyy` (MIT) "
                "and/or `npm install -g gitnexus` (PolyForm Noncommercial)."
            ),
        )
        return {"detected": [], "changed": False}

    summary = {"detected": [t.name for t in detected]}
    for t in detected:
        log_fn(
            "mcp_registry_detected",
            tool=t.name,
            binary_path=t.binary_path,
            registers_mcp=bool(t.section_toml.strip()),
            license_note=t.license_note,
        )

    changed, msg = register_in_codex_config(detected, config_path=config_path)
    summary["changed"] = changed
    summary["status"] = msg
    log_fn("mcp_registry_register", changed=changed, status=msg)
    if changed:
        log_fn(
            "mcp_registry_codex_restart_required",
            note=(
                "Codex.app must be quit and restarted for the new MCP "
                "servers to load. tinyctx cannot do this for you (codex.app "
                "is a GUI process). After restart, the new tools become "
                "available to the model."
            ),
        )
    return summary
