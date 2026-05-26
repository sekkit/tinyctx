"""Serena auto-install + auto-wire bootstrap.

Serena (oraios/serena, MIT) is a LSP-backed symbolic code-operations MCP
server: find_symbol, find_referencing_symbols, replace_body, etc. tinyctx
auto-enables it for codex sessions, mirroring gitnexus_bootstrap.

Install path priority (only first available is used):
  a. Already on PATH       → skip
  b. uv tool install       → preferred (Astral, manages its own Python)
  c. pipx install          → second choice
  d. uv via curl           → bootstrap uv first, then case (b)

After install: write `[mcp_servers.serena]` to ~/.codex/config.toml so
codex spawns it on session start.

Disable by env: TINYCTX_SERENA_DISABLE=1
Override binary: TINYCTX_SERENA_BIN=...
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


SERENA_PYPI_PKG = os.environ.get("TINYCTX_SERENA_PKG", "serena-agent")
# serena-agent ships a `serena-mcp-server` script (preferred) and `serena`.
SERENA_BIN_CANDIDATES = ("serena-mcp-server", "serena-mcp", "serena")

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)
TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "serena-bootstrap.log"

UV_INSTALL_URL = "https://astral.sh/uv/install.sh"

_SERENA_CONFIG_MARKER = "[mcp_servers.serena]"

_SERENA_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (serena_bootstrap). Safe to delete or edit.
# Serena: LSP-backed symbolic code-operations MCP server.
# Source: https://github.com/oraios/serena
#
# Spawn flags (verified live 2026-05-10):
#   --context codex          codex-optimized prompt mode (built-in;
#                            without it, serena defaults to "desktop-app"
#                            which has too-verbose tool descriptions for
#                            codex's prompt budget)
#   --project-from-cwd       auto-detect project from codex session's cwd
#                            (Path-search: .serena/project.yml -> .git -> cwd
#                            fallback). Without this, serena boots into
#                            "no active project" state, the model would
#                            need to call `activate_project` first, and
#                            the dashboard at :24282 stays empty.
#   --open-web-dashboard false
#                            Suppress browser-popup-on-launch. The dashboard
#                            stays running on :24282 - you can still open it
#                            manually if you want. Without this flag, every
#                            codex session spawns a serena that pops a new
#                            browser tab (codex spawns one MCP per session,
#                            so this can flood with 3-5 tabs in a busy day).
#
# The bare `serena` CLI also has `--mode codex` - that's a DIFFERENT
# option than `--context codex` and the bootstrap MUST NOT use it
# (verified: rejected as "No such option").
{marker}
type = "stdio"
command = "{cmd}"
args = ["start-mcp-server", "--context", "codex", "--project-from-cwd",
        "--open-web-dashboard", "false"]
startup_timeout_sec = 30.0
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        # Why: _log itself must never raise. Logging failure (disk full,
        # read-only fs) is acceptable; bootstrap continues silently.
        pass


def _which(cmd: str) -> str:
    from .mcp_registry import _which_with_fallbacks as _wf
    return _wf(cmd) or ""


def _find_serena() -> str:
    """First serena-flavoured binary on PATH, or ''."""
    forced = os.environ.get("TINYCTX_SERENA_BIN")
    if forced and Path(forced).is_file():
        return forced
    for name in SERENA_BIN_CANDIDATES:
        p = _which(name)
        if p:
            return p
    return ""


@dataclass
class State:
    disabled: bool = False
    serena_path: str = ""
    serena_present: bool = False
    uv_path: str = ""
    pipx_path: str = ""
    codex_config_has_serena: bool = False


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_SERENA_DISABLE", "0") == "1"
    s.serena_path = _find_serena()
    s.serena_present = bool(s.serena_path)
    s.uv_path = _which("uv")
    s.pipx_path = _which("pipx")
    s.codex_config_has_serena = has_mcp_block(codex_config,
                                              _SERENA_CONFIG_MARKER)
    return s


def _run(cmd: list[str], *, timeout: int = 600) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except OSError as e:
        return False, f"OSError: {e}"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-300:].strip()
        return False, f"rc={r.returncode}: {tail}"
    return True, "ok"


def install_via_uv(state: State, *, dry_run: bool = False
                   ) -> tuple[bool, str]:
    if not state.uv_path:
        return False, "uv not on PATH"
    cmd = [state.uv_path, "tool", "install", SERENA_PYPI_PKG]
    if dry_run:
        return True, "DRY-RUN: " + " ".join(cmd)
    return _run(cmd, timeout=600)


def install_via_pipx(state: State, *, dry_run: bool = False
                     ) -> tuple[bool, str]:
    if not state.pipx_path:
        return False, "pipx not on PATH"
    cmd = [state.pipx_path, "install", SERENA_PYPI_PKG]
    if dry_run:
        return True, "DRY-RUN: " + " ".join(cmd)
    return _run(cmd, timeout=600)


def bootstrap_uv_via_curl(*, dry_run: bool = False) -> tuple[bool, str]:
    if dry_run:
        return True, f"DRY-RUN: curl -LsSf {UV_INSTALL_URL} | sh"
    if not _which("curl") or not _which("sh"):
        return False, "curl or sh missing"
    tmp = Path(f"/tmp/tinyctx-serena-uv-install-{os.getpid()}.sh")
    try:
        ok, msg = _run(["curl", "-LsSf", UV_INSTALL_URL, "-o", str(tmp)],
                       timeout=120)
        if not ok:
            return False, f"download failed: {msg}"
        ok, msg = _run(["sh", str(tmp)], timeout=180)
        if not ok:
            return False, f"installer failed: {msg}"
        return True, "uv bootstrapped"
    finally:
        try:
            tmp.unlink()
        except OSError as e:
            # Why: temp installer script cleanup is best-effort; install
            # already finished (success or failure), so swallow but log.
            _log(f"tmp installer unlink failed: {type(e).__name__}: {e}")


def install_globally(state: State, *, dry_run: bool = False
                     ) -> tuple[bool, str, str]:
    if state.uv_path:
        ok, msg = install_via_uv(state, dry_run=dry_run)
        if ok:
            return True, msg, "uv"
        _log(f"uv install failed: {msg}")
    if state.pipx_path:
        ok, msg = install_via_pipx(state, dry_run=dry_run)
        if ok:
            return True, msg, "pipx"
        _log(f"pipx install failed: {msg}")
    if not state.uv_path:
        ok, msg = bootstrap_uv_via_curl(dry_run=dry_run)
        if not ok:
            return False, f"all installers failed (last: {msg})", ""
        if dry_run:
            return True, msg, "uv-bootstrapped (dry-run)"
        new_state = detect_state()
        if not new_state.uv_path:
            return False, "uv bootstrap reported success but uv not found", ""
        ok, msg = install_via_uv(new_state, dry_run=dry_run)
        return ok, f"uv bootstrap + install: {msg}", "uv-bootstrapped"
    return False, "no installer available", ""


def _resolve_command(state: State) -> str:
    """Best command for codex's stdio spawn: absolute path preferred."""
    return state.serena_path or "serena-mcp-server"


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    cmd = _resolve_command(state)
    block = _SERENA_CONFIG_BLOCK_TEMPLATE.format(
        marker=_SERENA_CONFIG_MARKER, cmd=cmd)
    return append_mcp_block(config_path, _SERENA_CONFIG_MARKER,
                             block, dry_run=dry_run)


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list
    final_state: dict
    skipped: list
    success: bool


def bootstrap(*, install: bool = True, config: bool = True,
              dry_run: bool = False,
              codex_config: Path = CODEX_CONFIG_DEFAULT) -> BootstrapReport:
    actions: list[str] = []
    skipped: list[str] = []
    state = detect_state(codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_SERENA_DISABLE=1")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config)),
                               skipped, success=True)

    if install and not state.serena_present:
        ok, msg, used = install_globally(state, dry_run=dry_run)
        actions.append(f"global install via {used or 'none'}: {msg}")
        if ok and not dry_run:
            state = detect_state(codex_config)
    elif state.serena_present:
        actions.append(f"global install: already at {state.serena_path}")

    if config and state.serena_present:
        ok, msg = patch_codex_config(state, config_path=codex_config,
                                      dry_run=dry_run)
        actions.append(f"codex config: {msg}")
    elif config:
        skipped.append("config: serena binary still missing")

    final = detect_state(codex_config)
    success = all("rc=" not in a or "rc=0" in a for a in actions)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success)


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("serena binary", state.serena_path or "missing"),
        ("uv", state.uv_path or "missing"),
        ("pipx", state.pipx_path or "missing"),
        ("[mcp_servers.serena]", "yes" if state.codex_config_has_serena
                                 else "no"),
    ]
    print("serena state:")
    for k, v in rows:
        print(f"  {k:<22}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.serena_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall", "config-only"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)
    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        _print_state_human(detect_state(codex_path))
        return 0

    if args.cmd == "uninstall":
        if codex_path.is_file():
            text = codex_path.read_text(encoding="utf-8", errors="replace")
            if _SERENA_CONFIG_MARKER in text:
                if args.dry_run:
                    print(f"DRY-RUN strip {_SERENA_CONFIG_MARKER}")
                else:
                    codex_path.write_text(
                        strip_mcp_block(text, _SERENA_CONFIG_MARKER),
                        encoding="utf-8")
                    print("removed serena block from codex config")
            else:
                print("no serena block in codex config")
        return 0

    do_install = args.cmd == "install"
    do_config = args.cmd in ("install", "config-only")
    report = bootstrap(install=do_install, config=do_config,
                       dry_run=args.dry_run, codex_config=codex_path)

    if not args.quiet:
        print(f"[serena-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
