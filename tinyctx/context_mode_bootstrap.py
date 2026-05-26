"""context-mode MCP server auto-registration bootstrap.

context-mode (npm `context-mode`, MIT) is an MCP plugin that saves context
window via sandboxed code execution and FTS5 knowledge base. The binary
is installed by the unified installer (`npm i -g context-mode@latest`).

This bootstrap ONLY handles codex config registration — writing a
`[mcp_servers.context-mode]` stdio block to ~/.codex/config.toml so
codex spawns it on session start. It does NOT install the binary.

CLI:
    tinyctx-context-mode install     # register in codex config
    tinyctx-context-mode status      # show detection state
    tinyctx-context-mode uninstall   # remove from codex config

Disable: TINYCTX_CONTEXT_MODE_DISABLE=1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ._codex_toml import append_mcp_block, has_mcp_block, strip_mcp_block
from .mcp_registry import _which_with_fallbacks


TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "context-mode-bootstrap.log"

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)

_CONFIG_MARKER = "[mcp_servers.context-mode]"

_CONFIG_BLOCK_TEMPLATE = """\
# Added by tinyctx (context_mode_bootstrap). Safe to delete or edit.
# context-mode: sandboxed code execution + FTS5 knowledge base MCP server.
# Source: https://npm.im/context-mode
{marker}
type = "stdio"
command = "{cmd}"
startup_timeout_sec = 30.0
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        pass


@dataclass
class State:
    disabled: bool = False
    context_mode_path: str = ""
    context_mode_present: bool = False
    codex_config_exists: bool = False
    codex_config_has_context_mode: bool = False


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_CONTEXT_MODE_DISABLE", "0") == "1"
    cm = _which_with_fallbacks("context-mode")
    if cm:
        s.context_mode_path = cm
        s.context_mode_present = True
    s.codex_config_exists = codex_config.is_file()
    if s.codex_config_exists:
        s.codex_config_has_context_mode = has_mcp_block(
            codex_config, _CONFIG_MARKER)
    return s


def patch_codex_config(*, config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    state = detect_state(config_path)
    if not state.context_mode_present:
        return False, "context-mode binary not on PATH"
    block = _CONFIG_BLOCK_TEMPLATE.format(
        marker=_CONFIG_MARKER, cmd=state.context_mode_path)
    return append_mcp_block(config_path, _CONFIG_MARKER, block,
                             dry_run=dry_run)


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list[str]
    final_state: dict
    skipped: list[str]
    success: bool


def bootstrap(*, dry_run: bool = False,
              codex_config: Path = CODEX_CONFIG_DEFAULT) -> BootstrapReport:
    actions: list[str] = []
    skipped: list[str] = []

    state = detect_state(codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_CONTEXT_MODE_DISABLE=1")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config)),
                               skipped, success=True)

    if not state.context_mode_present:
        skipped.append("context-mode binary not on PATH")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config)),
                               skipped, success=True)

    if state.codex_config_has_context_mode:
        skipped.append("already registered in codex config")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config)),
                               skipped, success=True)

    ok, msg = patch_codex_config(config_path=codex_config, dry_run=dry_run)
    actions.append(f"codex config: {msg}")

    final = detect_state(codex_config)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success=ok)


# ──────────────────────────── CLI ─────────────────────────────────


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("context-mode path", state.context_mode_path or "missing"),
        ("context-mode present", "yes" if state.context_mode_present else "no"),
        ("codex config exists", "yes" if state.codex_config_exists else "no"),
        ("[mcp_servers.context-mode]",
         "yes" if state.codex_config_has_context_mode else "no"),
    ]
    print("context-mode state:")
    for k, v in rows:
        print(f"  {k:<32}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.context_mode_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)
    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        _print_state_human(detect_state(codex_path))
        return 0

    if args.cmd == "uninstall":
        if not codex_path.is_file():
            print("no codex config to clean")
            return 0
        text = codex_path.read_text(encoding="utf-8", errors="replace")
        if _CONFIG_MARKER not in text:
            print("no context-mode block to remove")
            return 0
        if args.dry_run:
            print(f"DRY-RUN strip {_CONFIG_MARKER}")
            return 0
        codex_path.write_text(
            strip_mcp_block(text, _CONFIG_MARKER), encoding="utf-8")
        print("removed context-mode block from codex config")
        return 0

    report = bootstrap(dry_run=args.dry_run, codex_config=codex_path)
    if not args.quiet:
        print(f"[context-mode-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
