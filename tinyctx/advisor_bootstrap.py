"""Advisor MCP auto-wire bootstrap.

The advisor (`tinyctx.advisor`) is a stdio MCP server that lives inside this
very project — installed by `pip install -e .` into the tinyctx venv. Unlike
gitnexus / serena (external binaries), there's no install step; we only
need to register `[mcp_servers.advisor]` in `~/.codex/config.toml` so
codex spawns it.

Why a bootstrap module instead of static config: the advisor's `command`
must be the absolute path to the venv's Python, which differs per machine.
We resolve it at install time from `sys.executable` so the registered
block is correct on every developer's box.

Idempotent: detects existing `[mcp_servers.advisor]` via section-header
match (see `_codex_toml.append_mcp_block`).

Env vars:
    TINYCTX_ADVISOR_DISABLE=1     bypass everything
    TINYCTX_ADVISOR_PYTHON=PATH   override venv-python detection
    TINYCTX_CODEX_CONFIG=PATH     override ~/.codex/config.toml path

CLI:
    python -m tinyctx.advisor_bootstrap            # install (config-only)
    python -m tinyctx.advisor_bootstrap status     # report state
    python -m tinyctx.advisor_bootstrap uninstall  # strip block
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)
TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "advisor-bootstrap.log"

_ADVISOR_CONFIG_MARKER = "[mcp_servers.advisor]"

_ADVISOR_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (advisor_bootstrap). Safe to delete or edit.
# Advisor MCP — frontier consultation tool (Anthropic Advisor Strategy).
# The executor (DeepSeek) calls ask_advisor when it's stuck on a hard
# decision; the call routes through the running tinyctx proxy with
# model=tinyctx-frontier so all auth/logging is shared.
{marker}
type = "stdio"
command = "{python}"
args = ["-m", "tinyctx.advisor"]

[mcp_servers.advisor.env]
TINYCTX_PROXY_URL = "http://127.0.0.1:4141/v1"
TINYCTX_ADVISOR_MODEL = "tinyctx-frontier"
TINYCTX_ADVISOR_TIMEOUT_S = "180"
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


@dataclass
class State:
    disabled: bool = False
    python_path: str = ""
    python_exists: bool = False
    codex_config_exists: bool = False
    codex_config_has_advisor: bool = False


def _resolve_python() -> str:
    """Best Python interpreter to register as the advisor command.

    Prefer an explicit override, else the interpreter currently running
    this bootstrap (which, when invoked by install.sh, is the project
    venv's python). Returns an absolute path.

    We deliberately use `.absolute()` and NOT `.resolve()`: on macOS,
    a venv's `bin/python` is typically a symlink back to the system
    framework Python; `.resolve()` follows it, which strips the venv
    site-packages off the spawned interpreter and breaks `import tinyctx`
    inside the advisor. The unresolved venv path is what makes Python
    detect the venv via `pyvenv.cfg`.
    """
    forced = os.environ.get("TINYCTX_ADVISOR_PYTHON")
    if forced:
        return str(Path(forced).expanduser().absolute())
    return str(Path(sys.executable).absolute())


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_ADVISOR_DISABLE", "0") == "1"
    s.python_path = _resolve_python()
    s.python_exists = Path(s.python_path).is_file()
    if codex_config.is_file():
        s.codex_config_exists = True
        try:
            from ._codex_toml import has_mcp_block
            s.codex_config_has_advisor = has_mcp_block(
                codex_config, _ADVISOR_CONFIG_MARKER)
        except OSError:
            pass
    return s


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    """Idempotent: appends `[mcp_servers.advisor]` block if absent."""
    from ._codex_toml import append_mcp_block
    block = _ADVISOR_CONFIG_BLOCK_TEMPLATE.format(
        marker=_ADVISOR_CONFIG_MARKER, python=state.python_path)
    return append_mcp_block(config_path, _ADVISOR_CONFIG_MARKER,
                            block, dry_run=dry_run)


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
        skipped.append("TINYCTX_ADVISOR_DISABLE=1")
        _log("disabled by env; abort")
        final = detect_state(codex_config)
        return BootstrapReport(state_before, actions, asdict(final),
                               skipped, success=True)

    if not state.python_exists:
        skipped.append(f"python interpreter missing: {state.python_path}")
        _log(f"skip: python missing at {state.python_path}")
        final = detect_state(codex_config)
        return BootstrapReport(state_before, actions, asdict(final),
                               skipped, success=False)

    ok, msg = patch_codex_config(state, config_path=codex_config,
                                 dry_run=dry_run)
    actions.append(f"codex config: {msg}")
    _log(f"config: {msg}")

    final = detect_state(codex_config)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success=ok)


def _strip_block(text: str) -> str:
    from ._codex_toml import strip_mcp_block
    return strip_mcp_block(text, _ADVISOR_CONFIG_MARKER)


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("python", state.python_path),
        ("python exists", "yes" if state.python_exists else "no"),
        ("codex config exists", "yes" if state.codex_config_exists else "no"),
        ("[mcp_servers.advisor]",
         "yes" if state.codex_config_has_advisor else "no"),
    ]
    print("advisor state:")
    for k, v in rows:
        print(f"  {k:<24}  {v}")


def _cmd_uninstall(codex_path: Path, *, dry_run: bool, quiet: bool) -> int:
    if not codex_path.is_file():
        if not quiet:
            print("  ✓ no codex config to clean")
        return 0
    try:
        text = codex_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  ✗ read {codex_path}: {e}", file=sys.stderr)
        return 1
    if _ADVISOR_CONFIG_MARKER not in text:
        if not quiet:
            print("  ✓ no advisor block in codex config")
        return 0
    new_text = _strip_block(text)
    if dry_run:
        if not quiet:
            print(f"  ✓ DRY-RUN strip block from {codex_path}")
        return 0
    try:
        codex_path.write_text(new_text, encoding="utf-8")
    except OSError as e:
        print(f"  ✗ write: {e}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"  ✓ stripped block from {codex_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.advisor_bootstrap")
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
        return _cmd_uninstall(codex_path, dry_run=args.dry_run,
                              quiet=args.quiet)

    report = bootstrap(dry_run=args.dry_run, codex_config=codex_path)
    if not args.quiet:
        for a in report.actions:
            print(f"  ✓ {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
