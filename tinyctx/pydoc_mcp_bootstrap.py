"""pydoc-mcp MCP server auto-registration bootstrap.

pydoc-mcp (tinyctx.pydoc_mcp) is an in-tree, zero-network MCP server that
exposes Python's installed-package introspection (importlib.metadata +
pydoc + inspect) to codex. It's the offline replacement for context7 /
DeepWiki-style hosted doc lookups — the data source is whatever's
already installed in the interpreter, so no API key, no network, no
third-party rate limits.

This bootstrap ONLY writes a `[mcp_servers.pydoc]` stdio block to
~/.codex/config.toml. The server itself ships with tinyctx and runs as
`<python> -m tinyctx.pydoc_mcp` — no separate install step.

CLI:
    tinyctx-pydoc-mcp install     # register in codex config
    tinyctx-pydoc-mcp status      # show detection state
    tinyctx-pydoc-mcp uninstall   # remove from codex config

Disable: TINYCTX_PYDOC_MCP_DISABLE=1
Override python: TINYCTX_PYDOC_MCP_PYTHON=/path/to/python
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ._codex_toml import append_mcp_block, has_mcp_block, strip_mcp_block


TINYCTX_HOME = Path(os.environ.get(
    "TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "pydoc-mcp-bootstrap.log"

CODEX_CONFIG_DEFAULT = (
    Path(os.environ.get("TINYCTX_CODEX_CONFIG", ""))
    if os.environ.get("TINYCTX_CODEX_CONFIG")
    else Path.home() / ".codex" / "config.toml"
)

_CONFIG_MARKER = "[mcp_servers.pydoc]"

# Bump when the block template changes — _codex_toml.append_mcp_block
# uses this to detect stale blocks on already-installed machines and
# rewrite. Keep this in sync with the template body below.
_BLOCK_VERSION = "tinyctx-pydoc-mcp-block-version: 1"

_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (pydoc_mcp_bootstrap). Safe to delete or edit.
# {version}
# pydoc-mcp: zero-network local Python docs (importlib.metadata + pydoc).
# Offline alternative to hosted doc services like context7 / DeepWiki.
# Source: in-tree (tinyctx/pydoc_mcp.py) — no separate install needed.
{marker}
type = "stdio"
command = "{python}"
args = ["-m", "tinyctx.pydoc_mcp"]
startup_timeout_sec = 15.0
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def _resolve_python() -> str:
    """Pick the interpreter that has tinyctx importable.

    The codex MCP child needs a Python that can `import tinyctx`. The
    safest answer is the interpreter that's running this bootstrap —
    if tinyctx is installed in its site-packages, the child will see it
    too. Override with TINYCTX_PYDOC_MCP_PYTHON when bootstrapping from
    one venv but wanting codex to spawn against another."""
    forced = os.environ.get("TINYCTX_PYDOC_MCP_PYTHON")
    if forced:
        return forced
    return sys.executable


@dataclass
class State:
    disabled: bool = False
    python_path: str = ""
    tinyctx_importable: bool = False
    codex_config_exists: bool = False
    codex_config_has_pydoc: bool = False


def _tinyctx_importable_by(python_path: str) -> bool:
    """Cheap check: does `<python> -c 'import tinyctx.pydoc_mcp'` succeed?

    Used at status time, not on every detect_state — keep this off the
    hot path by only calling when the caller actually needs it.
    """
    import subprocess
    try:
        r = subprocess.run(
            [python_path, "-c", "import tinyctx.pydoc_mcp"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT, *,
                 check_importable: bool = False) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_PYDOC_MCP_DISABLE", "0") == "1"
    s.python_path = _resolve_python()
    if check_importable:
        s.tinyctx_importable = _tinyctx_importable_by(s.python_path)
    else:
        # Skip the subprocess check by default — assume the running
        # interpreter has tinyctx (it must, since this module is in it).
        s.tinyctx_importable = (s.python_path == sys.executable)
    s.codex_config_exists = codex_config.is_file()
    if s.codex_config_exists:
        s.codex_config_has_pydoc = has_mcp_block(
            codex_config, _CONFIG_MARKER)
    return s


def patch_codex_config(*, config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    python = _resolve_python()
    block = _CONFIG_BLOCK_TEMPLATE.format(
        marker=_CONFIG_MARKER, python=python, version=_BLOCK_VERSION)
    return append_mcp_block(
        config_path, _CONFIG_MARKER, block,
        dry_run=dry_run, version_tag=_BLOCK_VERSION)


@dataclass
class BootstrapReport:
    state_before: dict
    actions: list
    final_state: dict
    skipped: list
    success: bool


def bootstrap(*, dry_run: bool = False,
              codex_config: Path = CODEX_CONFIG_DEFAULT) -> BootstrapReport:
    actions: list[str] = []
    skipped: list[str] = []
    state = detect_state(codex_config)
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run}")

    if state.disabled:
        skipped.append("TINYCTX_PYDOC_MCP_DISABLE=1")
        return BootstrapReport(state_before, actions,
                               asdict(detect_state(codex_config)),
                               skipped, success=True)

    if state.codex_config_has_pydoc:
        # version_tag below handles template revisions — if same version,
        # this branch is the idempotent fast path.
        pass

    ok, msg = patch_codex_config(config_path=codex_config, dry_run=dry_run)
    actions.append(f"codex config: {msg}")

    final = detect_state(codex_config)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success=ok)


# ──────────────────────────── CLI ─────────────────────────────────


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("python interpreter", state.python_path),
        ("tinyctx importable", "yes" if state.tinyctx_importable else "no"),
        ("codex config exists",
         "yes" if state.codex_config_exists else "no"),
        ("[mcp_servers.pydoc]",
         "yes" if state.codex_config_has_pydoc else "no"),
    ]
    print("pydoc-mcp state:")
    for k, v in rows:
        print(f"  {k:<25}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.pydoc_mcp_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--codex-config", default=str(CODEX_CONFIG_DEFAULT))
    args = p.parse_args(argv)
    codex_path = Path(args.codex_config).expanduser()

    if args.cmd == "status":
        _print_state_human(detect_state(codex_path, check_importable=True))
        return 0

    if args.cmd == "uninstall":
        if not codex_path.is_file():
            print("no codex config to clean")
            return 0
        text = codex_path.read_text(encoding="utf-8", errors="replace")
        if _CONFIG_MARKER not in text:
            print("no pydoc-mcp block to remove")
            return 0
        if args.dry_run:
            print(f"DRY-RUN strip {_CONFIG_MARKER}")
            return 0
        codex_path.write_text(
            strip_mcp_block(text, _CONFIG_MARKER), encoding="utf-8")
        print("removed pydoc-mcp block from codex config")
        return 0

    report = bootstrap(dry_run=args.dry_run, codex_config=codex_path)
    if not args.quiet:
        print(f"[pydoc-mcp-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
