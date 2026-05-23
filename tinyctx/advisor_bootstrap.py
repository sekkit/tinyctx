"""Advisor auto-wire bootstrap.

For modern Codex (0.128+), the supported advisor route is the native
sub-agent protocol:

    spawn_agent(role="advisor", task="...")

That requires TWO pieces of config:

  1. `~/.codex/config.toml` must contain `[agents.advisor]`
  2. `~/.codex/agents/advisor.toml` must exist

We also keep the legacy MCP registration:

  3. `~/.codex/config.toml` contains `[mcp_servers.advisor]`

The MCP server remains useful as a fallback path and for internal tinyctx
flows, but it is *not* sufficient to make `spawn_agent(role="advisor")`
work. The bug behind "advisor never triggers" in live setups was exactly
that: only the MCP block existed, while the agent registration/file were
missing.

Why a bootstrap module instead of static config: the advisor's `command`
must be the absolute path to the venv's Python, which differs per machine.
We resolve it at install time from `sys.executable` so the registered
block is correct on every developer's box, and we write the agent file from
the project-shipped template so the instructions stay in sync with tinyctx.

Idempotent: detects existing config blocks via section-header match (see
`_codex_toml.append_mcp_block`) and only writes the agent file when absent.

Env vars:
    TINYCTX_ADVISOR_DISABLE=1     bypass everything
    TINYCTX_ADVISOR_PYTHON=PATH   override venv-python detection
    TINYCTX_CODEX_CONFIG=PATH     override ~/.codex/config.toml path

CLI:
    python -m tinyctx.advisor_bootstrap            # install / repair config
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
_ADVISOR_AGENT_MARKER = "[agents.advisor]"
_ADVISOR_AGENT_RELATIVE_PATH = Path("agents") / "advisor.toml"

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

_ADVISOR_AGENT_CONFIG_BLOCK_TEMPLATE = """
# Added by tinyctx (advisor_bootstrap). Safe to delete or edit.
# Codex native advisor route. `spawn_agent(role="advisor", task=...)`
# resolves through this block to ~/.codex/agents/advisor.toml.
{marker}
description = "Consult a more capable advisor model (gpt-5.5 / Opus-class) for HARD decisions when stuck. Use when: (1) torn between architectural choices with real consequences, (2) tried 2+ failed approaches and need a fresh perspective, (3) about to make a non-trivial security/correctness decision, (4) user intent ambiguous and the wrong interpretation will waste significant work."
config_file = "agents/advisor.toml"
"""

_ADVISOR_AGENT_FILE_TEMPLATE = '''# Advisor agent — codex-native implementation of Anthropic's Advisor Strategy
# (claude.com/blog/the-advisor-strategy).
#
# Why this exists vs. the tinyctx/advisor.py MCP server:
# Codex 0.128.0-alpha.1's namespace MCP dispatcher returns "unsupported call"
# for `mcp__advisor__ask_advisor` even though the executor sees and invokes
# the tool. Reverse-engineering codex.app revealed that codex's internal
# `spawn_agent` system (multi_agent feature, stable+true) is the supported
# path: the executor calls `spawn_agent(role="advisor", task=...)`, codex
# starts a sub-thread bound to this config, the sub-thread runs against
# tinyctx's frontier route, and the result is awaited via wait_agent.
#
# Routing: model="tinyctx-frontier" is a tinyctx-recognised id that
# bypasses the cheap-or-frontier router entirely and goes straight to the
# configured frontier backend (gpt-5.5 by default). Every advisor call
# shows up in tinyctx-trace with `forced_by_client_model=true`.
#
# Cost shape: each spawn should be a short consultation, not an
# executor loop. Budget ~1-3 calls/task.

name = "advisor"
model = "tinyctx-frontier"
model_reasoning_effort = "high"
sandbox_mode = "read-only"
web_search = "disabled"

developer_instructions = """
You are an expert advisor for a coding agent. The agent has hit a decision it can't reasonably make on its own and is consulting you. Your job:

1. Give CONCISE, ACTIONABLE guidance — 100-200 words is the target.
2. If the question is well-posed: answer directly with the best approach + 1-2 sentences on WHY.
3. If the question is ambiguous or under-specified: state your assumption clearly, then advise based on it.
4. If you'd need information the agent didn't provide: name what specific information you'd need (don't ask follow-ups, just enumerate).
5. Include risks / sharp edges the executor should watch for.
6. Do NOT recap the question. Do NOT add filler. Do NOT apologise.

Output is going back to a coding agent that will execute on your advice. Optimise for the agent reading and acting on it, not for human prose.

## Use this advisor for
- Multiple architectural choices with real consequences (data model, API shape, retry semantics, concurrency)
- Stuck after 2+ failed approaches at the same problem
- Non-trivial security or correctness decisions (auth flow, schema migration)
- User intent ambiguous and a wrong interpretation will waste significant work

## Don't use this advisor for
- Routine code edits, file reads, simple refactors
- Looking up syntax / API references
- Padding the response with extra opinion

## Output shape
Default to this structure unless the executor's request asks for a different shape:

**Recommendation**
[Single best approach in 1-2 sentences]

**Why**
[2-4 bullet points or a short paragraph — the load-bearing reasons, no recap]

**Sharp edges**
[Bulleted risks the executor should watch for, including failure modes that aren't obvious from the question]

**Open questions** (only if applicable)
[List specific information you'd need to give a stronger answer; don't ask the executor to clarify, just enumerate]

The executor reads this verbatim. Don't pad. Don't apologize. Don't echo the question back.
"""
'''


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Why: _log itself must never raise — bootstrap log is advisory.
        pass


@dataclass
class State:
    disabled: bool = False
    python_path: str = ""
    python_exists: bool = False
    codex_config_exists: bool = False
    codex_config_has_advisor: bool = False
    codex_config_has_advisor_agent: bool = False
    advisor_agent_path: str = ""
    advisor_agent_file_exists: bool = False


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


def _agent_path_for_config(codex_config: Path) -> Path:
    return codex_config.expanduser().parent / _ADVISOR_AGENT_RELATIVE_PATH


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_ADVISOR_DISABLE", "0") == "1"
    s.python_path = _resolve_python()
    s.python_exists = Path(s.python_path).is_file()
    agent_path = _agent_path_for_config(codex_config)
    s.advisor_agent_path = str(agent_path)
    s.advisor_agent_file_exists = agent_path.is_file()
    if codex_config.is_file():
        s.codex_config_exists = True
        try:
            from ._codex_toml import has_mcp_block
            s.codex_config_has_advisor = has_mcp_block(
                codex_config, _ADVISOR_CONFIG_MARKER)
            s.codex_config_has_advisor_agent = has_mcp_block(
                codex_config, _ADVISOR_AGENT_MARKER)
        except OSError as e:
            # Detection probe: config exists but unreadable. Leave flag
            # False so bootstrap re-patches (idempotent — safe).
            _log(f"codex config probe failed: {type(e).__name__}: {e}")
    return s


def patch_codex_config(state: State, *,
                       config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, str]:
    """Idempotent: appends advisor MCP + agent blocks when absent."""
    from ._codex_toml import append_mcp_block
    mcp_block = _ADVISOR_CONFIG_BLOCK_TEMPLATE.format(
        marker=_ADVISOR_CONFIG_MARKER, python=state.python_path)
    agent_block = _ADVISOR_AGENT_CONFIG_BLOCK_TEMPLATE.format(
        marker=_ADVISOR_AGENT_MARKER)
    ok1, msg1 = append_mcp_block(
        config_path, _ADVISOR_CONFIG_MARKER, mcp_block, dry_run=dry_run)
    ok2, msg2 = append_mcp_block(
        config_path, _ADVISOR_AGENT_MARKER, agent_block, dry_run=dry_run)
    return ok1 and ok2, f"{msg1}; {msg2}"


def write_agent_file(*, config_path: Path = CODEX_CONFIG_DEFAULT,
                     dry_run: bool = False) -> tuple[bool, str]:
    path = _agent_path_for_config(config_path)
    if path.is_file():
        return True, "advisor agent file already present"
    if dry_run:
        return True, f"DRY-RUN would write advisor agent file to {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_ADVISOR_AGENT_FILE_TEMPLATE, encoding="utf-8")
    except OSError as e:
        return False, f"write advisor agent file failed: {e}"
    return True, f"wrote advisor agent file to {path}"


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
    ok_file, file_msg = write_agent_file(config_path=codex_config,
                                         dry_run=dry_run)
    actions.append(f"advisor agent file: {file_msg}")
    _log(f"agent-file: {file_msg}")

    final = detect_state(codex_config)
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success=ok and ok_file)


def _strip_block(text: str) -> str:
    from ._codex_toml import strip_mcp_block
    out = strip_mcp_block(text, _ADVISOR_CONFIG_MARKER)
    return strip_mcp_block(out, _ADVISOR_AGENT_MARKER)


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("python", state.python_path),
        ("python exists", "yes" if state.python_exists else "no"),
        ("codex config exists", "yes" if state.codex_config_exists else "no"),
        ("[mcp_servers.advisor]",
         "yes" if state.codex_config_has_advisor else "no"),
        ("[agents.advisor]",
         "yes" if state.codex_config_has_advisor_agent else "no"),
        ("agents/advisor.toml",
         "yes" if state.advisor_agent_file_exists else "no"),
    ]
    print("advisor state:")
    for k, v in rows:
        print(f"  {k:<24}  {v}")


def _cmd_uninstall(codex_path: Path, *, dry_run: bool, quiet: bool) -> int:
    agent_path = _agent_path_for_config(codex_path)
    had_agent_file = agent_path.is_file()
    if not codex_path.is_file():
        if not had_agent_file:
            if not quiet:
                print("  ✓ no codex config to clean")
            return 0
    try:
        text = (codex_path.read_text(encoding="utf-8", errors="replace")
                if codex_path.is_file() else "")
    except OSError as e:
        print(f"  ✗ read {codex_path}: {e}", file=sys.stderr)
        return 1
    has_any_block = (_ADVISOR_CONFIG_MARKER in text
                     or _ADVISOR_AGENT_MARKER in text)
    if dry_run:
        if not quiet:
            if has_any_block:
                print(f"  ✓ DRY-RUN strip advisor blocks from {codex_path}")
            if had_agent_file:
                print(f"  ✓ DRY-RUN remove {agent_path}")
            if not has_any_block and not had_agent_file:
                print("  ✓ no advisor config to clean")
        return 0
    if has_any_block:
        new_text = _strip_block(text)
        try:
            codex_path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            print(f"  ✗ write: {e}", file=sys.stderr)
            return 1
    if had_agent_file:
        try:
            agent_path.unlink()
        except OSError as e:
            print(f"  ✗ remove {agent_path}: {e}", file=sys.stderr)
            return 1
    if not quiet:
        if has_any_block:
            print(f"  ✓ stripped advisor blocks from {codex_path}")
        else:
            print("  ✓ no advisor blocks in codex config")
        if had_agent_file:
            print(f"  ✓ removed {agent_path}")
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
