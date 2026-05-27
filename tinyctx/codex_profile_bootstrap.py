"""Auto-write tinyctx Codex provider/profile blocks to ~/.codex/config.toml.

Without the provider/profile blocks, `codex --profile tinyctx` fails with
"profile not found" — meaning the proxy is up but codex never routes
through it. The previous install.sh PRINTED the blocks at the end and
asked the user to copy-paste them into the codex config; for anyone who
followed the README's "Quick start" line `./scripts/install.sh &&
codex --profile tinyctx` that step was easy to miss, leaving the install
silently broken.

This module makes registration the install script's responsibility, with
the same idempotent append_mcp_block helper the per-server bootstraps
use. Each marker gets an append call, lock-protected and safe to re-run.

Env vars:
    TINYCTX_CODEX_PROFILE_DISABLE=1   bypass entirely
    TINYCTX_PROXY_URL=...             override base_url (default 127.0.0.1:4141)
    TINYCTX_PROFILE_MODEL=...         override profile model (default tinyctx-auto)
    TINYCTX_PROFILE_CONTEXT=...       override model_context_window
    TINYCTX_PROFILE_AUTO_COMPACT=...  override model_auto_compact_token_limit
    TINYCTX_CODEX_CONFIG=PATH         override ~/.codex/config.toml path

When env vars are unset, context_window and auto_compact_token_limit
are derived from the [local] section of ~/.tinyctx/config.toml so
switching the local model (deepseek ↔ qwen) adjusts the profile
automatically.

CLI:
    python -m tinyctx.codex_profile_bootstrap            # install blocks
    python -m tinyctx.codex_profile_bootstrap status     # report state
    python -m tinyctx.codex_profile_bootstrap uninstall  # strip both blocks
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
LOG_FILE = TINYCTX_HOME / "logs" / "codex-profile-bootstrap.log"

_PROVIDER_MARKER = "[model_providers.tinyctx]"
_PROFILE_MARKER = "[profiles.tinyctx]"
_GOAL_PROFILE_MARKER = "[profiles.tinyctx-goal]"

# Bump the version tag in a template's comment whenever the block content
# changes so append_mcp_block can force-update existing installs.
_PROVIDER_BLOCK_VERSION = "tinyctx-block-version: 1"
_PROFILE_BLOCK_VERSION = "tinyctx-block-version: 4"
_GOAL_PROFILE_BLOCK_VERSION = "tinyctx-block-version: 2"

_PROVIDER_BLOCK_TEMPLATE = """
# Added by tinyctx (codex_profile_bootstrap). Safe to delete or edit.
# tinyctx-block-version: 1
# Tells codex how to reach the local-first routing proxy. The proxy
# itself is started by `scripts/start.sh` (or `tinyctx-up start`).
{marker}
name = "tinyctx local-first router"
base_url = "{base_url}"
wire_api = "responses"
request_max_retries = 4
stream_max_retries = 10
stream_idle_timeout_ms = 300000
# No env_key -> codex's existing Authorization header is forwarded.
"""

_PROFILE_BLOCK_TEMPLATE = """
# Added by tinyctx (codex_profile_bootstrap). Safe to delete or edit.
# tinyctx-block-version: 4
# Activate with `codex --profile tinyctx`.
{marker}
model_provider = "tinyctx"
model = "{model}"
model_context_window = {ctx}
model_auto_compact_token_limit = {auto_compact}
model_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"
features = {{ goals = true }}
"""

_GOAL_PROFILE_BLOCK_TEMPLATE = """
# Added by tinyctx (codex_profile_bootstrap). Safe to delete or edit.
# tinyctx-block-version: 2
# Long-running Codex goal profile. Activate with `codex --profile tinyctx-goal`.
# This profile enables goal mode and bypasses approvals/sandboxing; use only
# in project paths you explicitly trust.
{marker}
model_provider = "tinyctx"
model = "{model}"
model_context_window = {ctx}
model_auto_compact_token_limit = {auto_compact}
model_reasoning_effort = "high"
plan_mode_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"
features = {{ goals = true }}
"""


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Why: _log itself must never raise — bootstrap log is advisory.
        pass


# Hardcoded fallbacks used when the tinyctx config file is absent / unreadable.
_FALLBACK_CTX = 400000
_FALLBACK_AUTO_COMPACT = 64000
_FALLBACK_GOAL_AUTO_COMPACT = 997500


def _local_model_config() -> tuple[int, float] | None:
    """Return (context_window, context_safe_fraction) from [local] config,
    or None when the config is unavailable or unconfigured."""
    try:
        from .config import load_config
        cfg = load_config()
        if cfg.local.context_window > 0:
            return (cfg.local.context_window, cfg.local.context_safe_fraction)
    except Exception:
        pass
    return None


def _proxy_url() -> str:
    return os.environ.get("TINYCTX_PROXY_URL", "http://127.0.0.1:4141/v1")


def _profile_model() -> str:
    return os.environ.get("TINYCTX_PROFILE_MODEL", "tinyctx-auto")


def _profile_context() -> int:
    raw = os.environ.get("TINYCTX_PROFILE_CONTEXT", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    lc = _local_model_config()
    if lc is not None:
        return lc[0]
    return _FALLBACK_CTX


def _profile_auto_compact() -> int:
    raw = os.environ.get("TINYCTX_PROFILE_AUTO_COMPACT", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    lc = _local_model_config()
    if lc is not None:
        return int(lc[0] * 0.5)
    return _FALLBACK_AUTO_COMPACT


def _goal_auto_compact() -> int:
    lc = _local_model_config()
    if lc is not None:
        return int(lc[0] * lc[1])
    return _FALLBACK_GOAL_AUTO_COMPACT


@dataclass
class State:
    disabled: bool = False
    codex_config_exists: bool = False
    has_provider_block: bool = False
    has_profile_block: bool = False
    has_goal_profile_block: bool = False


def detect_state(codex_config: Path = CODEX_CONFIG_DEFAULT) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_CODEX_PROFILE_DISABLE", "0") == "1"
    if codex_config.is_file():
        s.codex_config_exists = True
        try:
            from ._codex_toml import has_mcp_block
            s.has_provider_block = has_mcp_block(codex_config, _PROVIDER_MARKER)
            s.has_profile_block = has_mcp_block(codex_config, _PROFILE_MARKER)
            s.has_goal_profile_block = has_mcp_block(
                codex_config, _GOAL_PROFILE_MARKER)
        except OSError as e:
            # Detection probe: config exists but unreadable. Leave both
            # flags False so bootstrap re-patches (idempotent — safe).
            _log(f"codex config probe failed: {type(e).__name__}: {e}")
    return s


def _build_provider_block() -> str:
    return _PROVIDER_BLOCK_TEMPLATE.format(
        marker=_PROVIDER_MARKER, base_url=_proxy_url())


def _build_profile_block() -> str:
    return _PROFILE_BLOCK_TEMPLATE.format(
        marker=_PROFILE_MARKER,
        model=_profile_model(),
        ctx=_profile_context(),
        auto_compact=_profile_auto_compact(),
    )


def _build_goal_profile_block() -> str:
    return _GOAL_PROFILE_BLOCK_TEMPLATE.format(
        marker=_GOAL_PROFILE_MARKER,
        model=_profile_model(),
        ctx=_profile_context(),
        auto_compact=_goal_auto_compact(),
    )


def patch_codex_config(*, config_path: Path = CODEX_CONFIG_DEFAULT,
                       dry_run: bool = False) -> tuple[bool, list[str]]:
    """Append both blocks if missing. Returns (overall_ok, [messages])."""
    from ._codex_toml import append_mcp_block
    msgs: list[str] = []
    overall_ok = True

    ok1, msg1 = append_mcp_block(
        config_path, _PROVIDER_MARKER,
        _build_provider_block(), dry_run=dry_run,
        version_tag=_PROVIDER_BLOCK_VERSION)
    msgs.append(f"provider: {msg1}")
    if not ok1:
        overall_ok = False

    ok2, msg2 = append_mcp_block(
        config_path, _PROFILE_MARKER,
        _build_profile_block(), dry_run=dry_run,
        version_tag=_PROFILE_BLOCK_VERSION)
    msgs.append(f"profile: {msg2}")
    if not ok2:
        overall_ok = False

    ok3, msg3 = append_mcp_block(
        config_path, _GOAL_PROFILE_MARKER,
        _build_goal_profile_block(), dry_run=dry_run,
        version_tag=_GOAL_PROFILE_BLOCK_VERSION)
    msgs.append(f"goal profile: {msg3}")
    if not ok3:
        overall_ok = False

    return overall_ok, msgs


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
        skipped.append("TINYCTX_CODEX_PROFILE_DISABLE=1")
        _log("disabled by env; abort")
        final = detect_state(codex_config)
        return BootstrapReport(state_before, actions, asdict(final),
                               skipped, success=True)

    ok, msgs = patch_codex_config(config_path=codex_config, dry_run=dry_run)
    actions.extend(msgs)
    for m in msgs:
        _log(m)

    final = detect_state(codex_config)
    return BootstrapReport(state_before, actions,
                           asdict(final), skipped, success=ok)


def _strip_blocks(text: str) -> str:
    from ._codex_toml import strip_mcp_block
    out = strip_mcp_block(text, _GOAL_PROFILE_MARKER)
    out = strip_mcp_block(out, _PROFILE_MARKER)
    out = strip_mcp_block(out, _PROVIDER_MARKER)
    return out


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("codex config exists", "yes" if state.codex_config_exists else "no"),
        ("[model_providers.tinyctx]",
         "yes" if state.has_provider_block else "no"),
        ("[profiles.tinyctx]",
         "yes" if state.has_profile_block else "no"),
        ("[profiles.tinyctx-goal]",
         "yes" if state.has_goal_profile_block else "no"),
    ]
    print("codex-profile state:")
    for k, v in rows:
        print(f"  {k:<28}  {v}")


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
    if (_PROFILE_MARKER not in text) and (_PROVIDER_MARKER not in text):
        if not quiet:
            print("  ✓ no tinyctx profile/provider blocks in codex config")
        return 0
    new_text = _strip_blocks(text)
    if dry_run:
        if not quiet:
            print(f"  ✓ DRY-RUN strip blocks from {codex_path}")
        return 0
    try:
        codex_path.write_text(new_text, encoding="utf-8")
    except OSError as e:
        print(f"  ✗ write: {e}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"  ✓ stripped tinyctx profile/provider blocks from {codex_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.codex_profile_bootstrap")
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
