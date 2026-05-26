"""Caveman (caveman-shrink MCP) auto-install bootstrap.

Caveman (JuliusBrussee/caveman, MIT) ships `caveman-shrink` as a standalone
npm package — an MCP **middleware**: it wraps an upstream MCP server and
compresses prose fields (tool descriptions, etc.) on the wire. It is NOT a
standalone MCP server — running it without an upstream command crashes
immediately with "missing upstream command" (verified live 2026-05-10).

Install: `npm i -g caveman-shrink` — the binary lands at the npm global
bin dir (typically ~/.local/node/bin/caveman-shrink).

Bootstrap does TWO things and deliberately skips the third:
  1. npm i -g caveman-shrink (so the binary is on PATH)
  2. idempotent re-run safe — skips if already installed
  3. NO auto codex config block — caveman-shrink is middleware, cannot
     run without an upstream server; the user composes their own wrapper

To USE caveman-shrink after install, the user manually wraps a target
MCP server in their codex config:

    [mcp_servers.gitnexus-shrunk]
    type = "stdio"
    command = "caveman-shrink"
    args = ["/path/to/gitnexus", "mcp"]

Disable: TINYCTX_CAVEMAN_DISABLE=1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ._codex_toml import append_mcp_block, has_mcp_block, strip_mcp_block
from .mcp_registry import _which_with_fallbacks


CAVEMAN_SHRINK_PKG = os.environ.get(
    "TINYCTX_CAVEMAN_SHRINK_PKG", "caveman-shrink")
CAVEMAN_SHRINK_BIN = os.environ.get(
    "TINYCTX_CAVEMAN_SHRINK_BIN", "caveman-shrink")

TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "caveman-bootstrap.log"

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)

_CONFIG_MARKER = "[mcp_servers.caveman-shrink]"

_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (caveman_bootstrap). Safe to delete or edit.
# Caveman-shrink: tool-description / output token compressor MCP middleware.
# Source: https://github.com/JuliusBrussee/caveman
{marker}
type = "stdio"
command = "{node}"
args = ["{entry}"]
startup_timeout_sec = 30.0
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def _which(cmd: str) -> str:
    return _which_with_fallbacks(cmd) or ""


def _find_caveman_shrink() -> str:
    """Find caveman-shrink binary. Checks env override first, then PATH
    (with fallback dirs for launchd/systemd PATH stripping)."""
    forced = os.environ.get("TINYCTX_CAVEMAN_SHRINK_BIN")
    if forced:
        return forced
    return _which(CAVEMAN_SHRINK_BIN)


def _npm() -> str | None:
    npm = os.environ.get("TINYCTX_NPM") or _which("npm")
    return npm or None


# ──────────────────────────── state ───────────────────────────────


@dataclass
class State:
    disabled: bool = False
    caveman_shrink_path: str = ""
    caveman_shrink_present: bool = False
    node_path: str = ""
    npm_present: bool = False
    codex_config_has_caveman: bool = False


def detect_state(*, codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_CAVEMAN_DISABLE", "0") == "1"
    s.node_path = _which("node")
    s.npm_present = bool(_npm())
    cs = _find_caveman_shrink()
    if cs:
        s.caveman_shrink_path = cs
        s.caveman_shrink_present = True
    s.codex_config_has_caveman = has_mcp_block(codex_config, _CONFIG_MARKER)
    return s


# ─────────────────────────── install ──────────────────────────────


def install_via_npm(*, dry_run: bool = False) -> tuple[bool, str]:
    """Run `npm i -g caveman-shrink@latest`. Returns (ok, message)."""
    npm = _npm()
    if not npm:
        return False, "npm not on PATH"
    cmd = [npm, "install", "-g", f"{CAVEMAN_SHRINK_PKG}@latest"]
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


# ─────────────────────── codex config ─────────────────────────────


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    if not state.caveman_shrink_present:
        return False, f"caveman-shrink binary missing"
    block = _CONFIG_BLOCK_TEMPLATE.format(
        marker=_CONFIG_MARKER, node=state.node_path or "node",
        entry=state.caveman_shrink_path)
    return append_mcp_block(config_path, _CONFIG_MARKER, block,
                             dry_run=dry_run)


# ────────────────────────── bootstrap ─────────────────────────────


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list
    final_state: dict
    skipped: list
    success: bool


def bootstrap(*, install: bool = True, do_config: bool = False,
              dry_run: bool = False,
              codex_config: Path = CODEX_CONFIG_DEFAULT
              ) -> BootstrapReport:
    """Default: npm i -g caveman-shrink. NO codex config write — caveman-shrink
    is middleware that wraps another MCP server, not a standalone server.

    Pass `do_config=True` only if you know what you're doing — it writes
    a block that needs manual upstream configuration before codex can start it.
    """
    actions: list[str] = []
    skipped: list[str] = []
    state = detect_state(codex_config=codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_CAVEMAN_DISABLE=1")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config=codex_config)),
                               skipped, success=True)

    if install and not state.caveman_shrink_present:
        ok, msg = install_via_npm(dry_run=dry_run)
        actions.append(f"npm-install: {msg}")
        if ok and not dry_run:
            state = detect_state(codex_config=codex_config)
    elif install:
        skipped.append("caveman-shrink already installed")

    # Default: do_config=False because caveman-shrink is middleware, not
    # a standalone server. Strip any previously written standalone block.
    if not do_config and not dry_run:
        if codex_config.is_file():
            try:
                text = codex_config.read_text(encoding="utf-8", errors="replace")
                if _CONFIG_MARKER in text:
                    cleaned = strip_mcp_block(text, _CONFIG_MARKER)
                    if cleaned != text:
                        codex_config.write_text(cleaned, encoding="utf-8")
                        actions.append(
                            "codex config: stripped broken standalone "
                            f"{_CONFIG_MARKER} block (caveman-shrink is "
                            "middleware, not standalone)"
                        )
            except OSError as e:
                _log(f"strip-broken-block failed: {e}")

    if do_config and state.caveman_shrink_present:
        ok, msg = patch_codex_config(state, config_path=codex_config,
                                      dry_run=dry_run)
        actions.append(f"codex config: {msg}")
    elif do_config and not state.caveman_shrink_present:
        skipped.append("config: caveman-shrink binary not found; install first")

    final = detect_state(codex_config=codex_config)
    success = all(
        ("rc=" not in a or "rc=0" in a) and "npm not on PATH" not in a
        for a in actions
    )
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success)


# ──────────────────────────── CLI ─────────────────────────────────


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("node", state.node_path or "missing"),
        ("npm", "yes" if state.npm_present else "no"),
        ("caveman-shrink path", state.caveman_shrink_path or "missing"),
        ("caveman-shrink present", "yes" if state.caveman_shrink_present else "no"),
        ("[mcp_servers.caveman-shrink]",
         "yes" if state.codex_config_has_caveman else "no"),
    ]
    print("caveman state:")
    for k, v in rows:
        print(f"  {k:<32}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.caveman_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall", "config-only"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)
    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        _print_state_human(detect_state(codex_config=codex_path))
        return 0

    if args.cmd == "uninstall":
        if codex_path.is_file():
            text = codex_path.read_text(encoding="utf-8", errors="replace")
            if _CONFIG_MARKER in text:
                if args.dry_run:
                    print(f"DRY-RUN strip {_CONFIG_MARKER}")
                else:
                    codex_path.write_text(
                        strip_mcp_block(text, _CONFIG_MARKER),
                        encoding="utf-8")
                    print("removed caveman block from codex config")
            else:
                print("no caveman block to remove")
        return 0

    do_install = args.cmd == "install"
    do_c = args.cmd == "config-only"
    report = bootstrap(install=do_install, do_config=do_c,
                       dry_run=args.dry_run, codex_config=codex_path)

    if not args.quiet:
        print(f"[caveman-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
