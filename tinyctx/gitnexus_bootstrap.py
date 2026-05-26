"""GitNexus auto-install + auto-wire bootstrap.

GitNexus (abhigyanpatwari/GitNexus, npm `gitnexus`, PolyForm-Noncommercial-1.0.0)
is a tree-sitter knowledge-graph MCP server. tinyctx auto-enables it so the
user gets full-codebase structural awareness without manual setup, exactly
like the Advisor MCP.

This module does four idempotent things on demand:

  1. Detect Node ≥ 20 (gitnexus npm engines requirement).
  2. Install `npm i -g gitnexus@<pin>` if the binary is absent.
  3. Patch ~/.codex/config.toml to add `[mcp_servers.gitnexus]` if missing.
  4. Print a ONE-TIME PolyForm Noncommercial license notice (legal sanity).

Failure handling: every step logs to ~/.tinyctx/logs/gitnexus-bootstrap.log
and never raises out. Bootstrap can fail silently (no node, no npm, sandboxed
machine) without breaking proxy startup.

CLI:
    python -m tinyctx.gitnexus_bootstrap              # install if needed
    python -m tinyctx.gitnexus_bootstrap status       # report state, no changes
    python -m tinyctx.gitnexus_bootstrap install      # explicit install
    python -m tinyctx.gitnexus_bootstrap config-only  # only patch codex toml
    python -m tinyctx.gitnexus_bootstrap --dry-run    # show plan, change nothing

Env vars:
    TINYCTX_GITNEXUS_DISABLE=1     bypass everything
    TINYCTX_GITNEXUS_VERSION=...   pin a different version (default: 1.6.3)
    TINYCTX_GITNEXUS_NPM=...       override npm binary path
    TINYCTX_CODEX_CONFIG=PATH      override ~/.codex/config.toml path
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


GITNEXUS_VERSION = os.environ.get("TINYCTX_GITNEXUS_VERSION", "1.6.3")
GITNEXUS_BIN = "gitnexus"
GITNEXUS_NPM_PKG = "gitnexus"

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)
TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "gitnexus-bootstrap.log"
LICENSE_ACK_FILE = TINYCTX_HOME / ".gitnexus-acked"

MIN_NODE_MAJOR = 20

# Marker the codex config patcher uses to detect prior install.
_GITNEXUS_CONFIG_MARKER = "[mcp_servers.gitnexus]"

_GITNEXUS_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (gitnexus_bootstrap). Safe to delete or edit.
# GitNexus is a tree-sitter codebase knowledge-graph MCP server.
# Source: https://github.com/abhigyanpatwari/GitNexus
{marker}
type = "stdio"
command = "{cmd}"
args = ["mcp"]
startup_timeout_sec = 30.0
"""

_LICENSE_NOTICE = """
[tinyctx] enabled GitNexus MCP server.
  Source: https://github.com/abhigyanpatwari/GitNexus
  License: PolyForm-Noncommercial-1.0.0 (free for non-commercial use only).
  For commercial use you need a commercial license from the upstream author.
  Disable any time:  TINYCTX_GITNEXUS_DISABLE=1
                     or remove [mcp_servers.gitnexus] from ~/.codex/config.toml
"""


# ─────────────────────────── logging ─────────────────────────────


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Why: _log itself must never raise — callers use it on hot
        # paths and a logging failure should not abort bootstrap.
        pass


# ─────────────────────────── detection ───────────────────────────


@dataclass
class State:
    disabled: bool = False
    node_present: bool = False
    node_version: str = ""
    node_meets_min: bool = False
    npm_present: bool = False
    gitnexus_present: bool = False
    gitnexus_path: str = ""
    codex_config_exists: bool = False
    codex_config_has_gitnexus: bool = False
    license_acked: bool = False


def _which(cmd: str) -> str:
    from .mcp_registry import _which_with_fallbacks as _wf
    return _wf(cmd) or ""


def _node_major(version_str: str) -> int:
    m = re.match(r"v?(\d+)\.", version_str)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_GITNEXUS_DISABLE", "0") == "1"

    node = _which("node")
    if node:
        s.node_present = True
        try:
            r = subprocess.run([node, "--version"],
                               capture_output=True, text=True, timeout=5)
            s.node_version = (r.stdout or "").strip()
            s.node_meets_min = _node_major(s.node_version) >= MIN_NODE_MAJOR
        except (subprocess.SubprocessError, OSError) as e:
            # Detection probe: leave node_version empty so bootstrap
            # downgrades to "node not usable" rather than crashing.
            _log(f"node probe failed: {type(e).__name__}: {e}")

    npm_path = os.environ.get("TINYCTX_GITNEXUS_NPM") or _which("npm")
    if npm_path:
        s.npm_present = True

    gn = _which(GITNEXUS_BIN)
    if gn:
        s.gitnexus_present = True
        s.gitnexus_path = gn

    if codex_config.is_file():
        s.codex_config_exists = True
        try:
            text = codex_config.read_text(encoding="utf-8", errors="replace")
            s.codex_config_has_gitnexus = _GITNEXUS_CONFIG_MARKER in text
        except OSError as e:
            # Detection probe: codex config exists but unreadable.
            # Leave codex_config_has_gitnexus=False so bootstrap retries
            # patching (idempotent — re-patch is safe).
            _log(f"codex config read failed: {type(e).__name__}: {e}")

    s.license_acked = LICENSE_ACK_FILE.is_file()
    return s


# ─────────────────────────── install ────────────────────────────


def install_via_npm(*, dry_run: bool = False) -> tuple[bool, str]:
    """Run `npm i -g gitnexus@<version>`. Returns (ok, message)."""
    npm = os.environ.get("TINYCTX_GITNEXUS_NPM") or _which("npm")
    if not npm:
        return False, "npm not on PATH"
    cmd = [npm, "install", "-g", f"{GITNEXUS_NPM_PKG}@{GITNEXUS_VERSION}"]
    if dry_run:
        return True, "DRY-RUN: " + " ".join(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "npm install timed out after 5min"
    except OSError as e:
        return False, f"npm install failed: {e}"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-500:]
        return False, f"npm install rc={r.returncode}: {tail}"
    return True, "npm install ok"


# ───────────────────────── codex config ─────────────────────────


def _resolve_gitnexus_command(state: State) -> str:
    """Best command for codex to spawn. Prefer absolute path so codex
    PATH issues don't break startup. Fall back to bare `gitnexus`."""
    if state.gitnexus_path:
        return state.gitnexus_path
    return GITNEXUS_BIN


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    """Idempotent: appends `[mcp_servers.gitnexus]` block if absent.
    Delegates to _codex_toml.append_mcp_block."""
    from ._codex_toml import append_mcp_block
    cmd = _resolve_gitnexus_command(state)
    block = _GITNEXUS_CONFIG_BLOCK_TEMPLATE.format(
        marker=_GITNEXUS_CONFIG_MARKER, cmd=cmd)
    return append_mcp_block(config_path, _GITNEXUS_CONFIG_MARKER,
                             block, dry_run=dry_run)


# ────────────────────────── license ack ─────────────────────────


def print_license_once() -> bool:
    """Print PolyForm Noncommercial notice on first activation. Return True
    if printed (i.e. the user has not seen it before)."""
    if LICENSE_ACK_FILE.is_file():
        return False
    try:
        LICENSE_ACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LICENSE_ACK_FILE.write_text(
            f"acked: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"version: {GITNEXUS_VERSION}\n",
            encoding="utf-8",
        )
    except OSError:
        return False
    sys.stderr.write(_LICENSE_NOTICE)
    return True


# ───────────────────────── orchestration ────────────────────────


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list[str]
    final_state: dict
    skipped: list[str]
    success: bool


def bootstrap(*, install: bool = True, config: bool = True,
              dry_run: bool = False,
              codex_config: Path = CODEX_CONFIG_DEFAULT) -> BootstrapReport:
    """Run full bootstrap. Returns a structured report."""
    actions: list[str] = []
    skipped: list[str] = []

    state = detect_state(codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_GITNEXUS_DISABLE=1")
        _log("disabled by env; abort")
        final = detect_state(codex_config)
        return BootstrapReport(state_before, actions, asdict(final),
                               skipped, success=True)

    if install and not state.gitnexus_present:
        if not state.node_present:
            skipped.append("node not on PATH; install Node ≥20 manually")
            _log("skip install: no node")
        elif not state.node_meets_min:
            skipped.append(
                f"node {state.node_version} < required v{MIN_NODE_MAJOR}; "
                f"upgrade node")
            _log(f"skip install: node {state.node_version} too old")
        elif not state.npm_present:
            skipped.append("npm not on PATH (despite node)")
            _log("skip install: no npm")
        else:
            ok, msg = install_via_npm(dry_run=dry_run)
            actions.append(f"install: {msg}")
            _log(f"install rc: {msg}")
            if ok and not dry_run:
                # re-detect so subsequent steps see the new binary
                state = detect_state(codex_config)
    elif state.gitnexus_present:
        actions.append(f"install: already present at {state.gitnexus_path}")

    if config and state.gitnexus_present:
        ok, msg = patch_codex_config(state, config_path=codex_config,
                                      dry_run=dry_run)
        actions.append(f"codex config: {msg}")
        _log(f"config: {msg}")
        if not dry_run and ok and not state.license_acked:
            print_license_once()
            actions.append("license: PolyForm-Noncommercial notice shown")
    elif config and not state.gitnexus_present:
        skipped.append("config: gitnexus binary still absent, "
                       "won't configure codex")

    final = detect_state(codex_config)
    success = all("rc=" not in a or "rc=0" in a for a in actions)
    _log(f"final state={asdict(final)} success={success}")
    return BootstrapReport(state_before, actions,
                           asdict(final), skipped, success)


# ─────────────────────────────── CLI ────────────────────────────


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("node", state.node_version or "missing"),
        ("node ≥20", "yes" if state.node_meets_min else "no"),
        ("npm", "yes" if state.npm_present else "missing"),
        ("gitnexus binary", state.gitnexus_path or "missing"),
        ("codex config exists", "yes" if state.codex_config_exists else "no"),
        ("[mcp_servers.gitnexus]", "yes" if state.codex_config_has_gitnexus
                                   else "no"),
        ("license ack file", "yes" if state.license_acked else "no"),
    ]
    print("gitnexus state:")
    for k, v in rows:
        print(f"  {k:<26}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.gitnexus_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "config-only", "uninstall"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)

    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        state = detect_state(codex_path)
        _print_state_human(state)
        return 0

    if args.cmd == "uninstall":
        return _cmd_uninstall(codex_path, dry_run=args.dry_run,
                              quiet=args.quiet)

    do_install = args.cmd == "install"
    do_config = args.cmd in ("install", "config-only")

    report = bootstrap(install=do_install, config=do_config,
                       dry_run=args.dry_run, codex_config=codex_path)

    if not args.quiet:
        if args.cmd == "install":
            print(f"[gitnexus-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  ✓ {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


def _cmd_uninstall(codex_path: Path, *,
                   dry_run: bool, quiet: bool) -> int:
    """Remove the [mcp_servers.gitnexus] block and the license ack file.
    Does NOT npm uninstall the binary (user may want it for other tools)."""
    actions: list[str] = []
    if codex_path.is_file():
        try:
            text = codex_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  ✗ read {codex_path}: {e}", file=sys.stderr)
            return 1
        if _GITNEXUS_CONFIG_MARKER in text:
            new_text = _strip_block(text)
            if dry_run:
                actions.append(f"DRY-RUN strip block from {codex_path}")
            else:
                try:
                    codex_path.write_text(new_text, encoding="utf-8")
                    actions.append(f"stripped block from {codex_path}")
                except OSError as e:
                    print(f"  ✗ write: {e}", file=sys.stderr)
                    return 1
        else:
            actions.append("no gitnexus block in codex config")
    if LICENSE_ACK_FILE.is_file() and not dry_run:
        try:
            LICENSE_ACK_FILE.unlink()
            actions.append("removed license ack file")
        except OSError as e:
            # Why: best-effort cleanup on uninstall; missing/locked file
            # should not abort the uninstall flow.
            _log(f"license ack unlink failed: {type(e).__name__}: {e}")
    if not quiet:
        for a in actions:
            print(f"  ✓ {a}")
    return 0


def _strip_block(text: str) -> str:
    """Thin wrapper for backward-compat with existing tests."""
    from ._codex_toml import strip_mcp_block
    return strip_mcp_block(text, _GITNEXUS_CONFIG_MARKER)


if __name__ == "__main__":
    raise SystemExit(main())
