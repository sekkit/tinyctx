"""Caveman (caveman-shrink MCP) auto-vendor bootstrap.

Caveman (JuliusBrussee/caveman, MIT) ships a `caveman-shrink` stdio
**middleware**: it wraps an upstream MCP server and compresses prose
fields (tool descriptions, etc.) on the wire. It is NOT a standalone
MCP server — running it without an upstream command crashes immediately
with "missing upstream command" (verified live 2026-05-10).

So this bootstrap does TWO of the three obvious things and skips the
third:
  1. ✓ git-clone caveman to ~/.tinyctx/vendor/caveman
  2. ✓ npm install the caveman-shrink subdir so its deps are present
  3. ✗ Auto-register a `[mcp_servers.caveman-shrink]` block —
     deliberately omitted. There's no sensible default upstream to wrap;
     wrapping requires per-project decision (e.g. wrap gitnexus, or
     wrap a tools-heavy MCP that has bloated descriptions).

To USE caveman-shrink after the vendor step, the user manually wraps
a target MCP server in their codex config:

    [mcp_servers.gitnexus-shrunk]
    type = "stdio"
    command = "/path/to/node"
    args = [
        "/Users/x/.tinyctx/vendor/caveman/mcp-servers/caveman-shrink/index.js",
        "/path/to/gitnexus", "mcp",
    ]

Disable: TINYCTX_CAVEMAN_DISABLE=1
Override clone target: TINYCTX_CAVEMAN_VENDOR=...
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


CAVEMAN_REPO = os.environ.get(
    "TINYCTX_CAVEMAN_REPO",
    "https://github.com/JuliusBrussee/caveman.git")

TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
DEFAULT_VENDOR = Path(
    os.environ.get("TINYCTX_CAVEMAN_VENDOR", str(TINYCTX_HOME / "vendor" / "caveman"))
)
LOG_FILE = TINYCTX_HOME / "logs" / "caveman-bootstrap.log"

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)

# Caveman ships multiple MCP servers; this is the most useful one.
SHRINK_REL_PATH = "mcp-servers/caveman-shrink/index.js"

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


def _which(c: str) -> str:
    return shutil.which(c) or ""


@dataclass
class State:
    disabled: bool = False
    vendor_dir: str = ""
    vendor_present: bool = False
    entry_path: str = ""
    entry_present: bool = False
    git_path: str = ""
    node_path: str = ""
    codex_config_has_caveman: bool = False


def detect_state(*, vendor: Path = DEFAULT_VENDOR,
                 codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_CAVEMAN_DISABLE", "0") == "1"
    s.vendor_dir = str(vendor)
    s.vendor_present = vendor.is_dir() and (vendor / ".git").is_dir()
    s.entry_path = str(vendor / SHRINK_REL_PATH)
    s.entry_present = (vendor / SHRINK_REL_PATH).is_file()
    s.git_path = _which("git")
    s.node_path = _which("node")
    s.codex_config_has_caveman = has_mcp_block(codex_config, _CONFIG_MARKER)
    return s


def _run(cmd: list[str], *, timeout: int = 300, cwd: str | None = None
         ) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except OSError as e:
        return False, f"OSError: {e}"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-300:].strip()
        return False, f"rc={r.returncode}: {tail}"
    return True, "ok"


def vendor(state: State, *, vendor_dir: Path = DEFAULT_VENDOR,
           dry_run: bool = False) -> tuple[bool, str]:
    if state.vendor_present:
        # update with `git pull` to keep close to head; non-fatal if it fails.
        if dry_run:
            return True, f"DRY-RUN would `git -C {vendor_dir} pull`"
        if not state.git_path:
            return True, "vendor present (git not found, skip pull)"
        ok, msg = _run([state.git_path, "-C", str(vendor_dir), "pull",
                        "--ff-only", "--quiet"], timeout=60)
        if not ok:
            _log(f"git pull non-fatal: {msg}")
        return True, "vendor up-to-date"
    if not state.git_path:
        return False, "git not on PATH"
    if dry_run:
        return True, (f"DRY-RUN: git clone --depth 1 "
                      f"{CAVEMAN_REPO} {vendor_dir}")
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    return _run([state.git_path, "clone", "--depth", "1", CAVEMAN_REPO,
                 str(vendor_dir)], timeout=300)


def npm_install_shrink(state: State, *, vendor_dir: Path = DEFAULT_VENDOR,
                       dry_run: bool = False) -> tuple[bool, str]:
    """Run `npm install` in mcp-servers/caveman-shrink/ so caveman-shrink's
    runtime deps are present. Without this, spawning the server fails with
    `Cannot find module ...`. Idempotent — npm itself short-circuits when
    deps are already up to date.
    """
    npm = _which("npm")
    if not npm:
        return False, "npm not on PATH (caveman-shrink deps will be missing)"
    shrink_dir = vendor_dir / "mcp-servers" / "caveman-shrink"
    if not shrink_dir.is_dir():
        return False, f"shrink subdir missing: {shrink_dir}"
    if dry_run:
        return True, f"DRY-RUN would `npm install` in {shrink_dir}"
    return _run([npm, "install", "--silent", "--no-audit",
                 "--no-fund", "--omit=dev"],
                timeout=300, cwd=str(shrink_dir))


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    if not state.entry_present:
        return False, f"caveman entry missing: {state.entry_path}"
    if not state.node_path:
        return False, "node not on PATH"
    block = _CONFIG_BLOCK_TEMPLATE.format(
        marker=_CONFIG_MARKER, node=state.node_path,
        entry=state.entry_path)
    return append_mcp_block(config_path, _CONFIG_MARKER, block,
                             dry_run=dry_run)


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list
    final_state: dict
    skipped: list
    success: bool


def bootstrap(*, do_vendor: bool = True, do_config: bool = False,
              dry_run: bool = False,
              vendor_dir: Path = DEFAULT_VENDOR,
              codex_config: Path = CODEX_CONFIG_DEFAULT
              ) -> BootstrapReport:
    """Default: clone caveman + npm-install caveman-shrink deps. NO codex
    config write — caveman-shrink is middleware that wraps another MCP
    server, not a standalone server, so a default `[mcp_servers.caveman-shrink]`
    block would just crash at startup. The user composes their own
    wrapper entry; see module docstring for the example.

    Pass `do_config=True` only if you know what you're doing — it writes
    a broken standalone block that codex cannot start.
    """
    actions: list[str] = []
    skipped: list[str] = []
    state = detect_state(vendor=vendor_dir, codex_config=codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_CAVEMAN_DISABLE=1")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(vendor=vendor_dir,
                                                    codex_config=codex_config)),
                               skipped, success=True)

    if do_vendor:
        ok, msg = vendor(state, vendor_dir=vendor_dir, dry_run=dry_run)
        actions.append(f"vendor: {msg}")
        if ok and not dry_run:
            state = detect_state(vendor=vendor_dir, codex_config=codex_config)
        # npm install caveman-shrink deps; non-fatal if it fails
        ok2, msg2 = npm_install_shrink(state, vendor_dir=vendor_dir,
                                        dry_run=dry_run)
        actions.append(f"npm-install-shrink: {msg2}")

    # Default: do_config=False because caveman-shrink is middleware, not
    # a standalone server. Whatever was previously written by an older
    # version of this bootstrap is now broken; remove it on every run.
    if not do_config and not dry_run:
        from ._codex_toml import strip_mcp_block
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

    if do_config and state.entry_present:
        ok, msg = patch_codex_config(state, config_path=codex_config,
                                      dry_run=dry_run)
        actions.append(f"codex config: {msg}")
    elif do_config and not state.entry_present:
        skipped.append("config: caveman entry not found "
                       f"({state.entry_path}); vendor first")

    final = detect_state(vendor=vendor_dir, codex_config=codex_config)
    success = all(("rc=" not in a or "rc=0" in a) and "git not on PATH" not in a
                  for a in actions)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success)


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("git", state.git_path or "missing"),
        ("node", state.node_path or "missing"),
        ("vendor dir", state.vendor_dir),
        ("vendor cloned", "yes" if state.vendor_present else "no"),
        ("entry script", state.entry_path),
        ("entry present", "yes" if state.entry_present else "no"),
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
    p.add_argument("--vendor-dir", default=str(DEFAULT_VENDOR))
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)
    vendor_path = Path(args.vendor_dir).expanduser()
    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        _print_state_human(detect_state(vendor=vendor_path,
                                         codex_config=codex_path))
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

    do_v = args.cmd == "install"
    # `config-only` is now an explicit "I know what I'm doing — write the
    # standalone block anyway" escape hatch; default `install` does NOT
    # write the standalone block (caveman-shrink is middleware).
    do_c = args.cmd == "config-only"
    report = bootstrap(do_vendor=do_v, do_config=do_c,
                       dry_run=args.dry_run, vendor_dir=vendor_path,
                       codex_config=codex_path)

    if not args.quiet:
        print(f"[caveman-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
