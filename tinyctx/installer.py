"""Unified installer: detect + install all missing tinyctx components.

One entry point (`install_all_missing`) runs every component's bootstrap
and returns structured results. Also serves as the `tinyctx` CLI entry.

CLI:
    tinyctx install          # install all missing components
    tinyctx status           # show install status for all components
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────── result type ───────────────────────────


@dataclass
class ComponentResult:
    component: str
    was_missing: bool = False
    installed: bool = False
    error: str = ""
    actions: list[str] = field(default_factory=list)


# ─────────────────────── context-mode (npm) ────────────────────────


def _which(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    for d in (
        os.path.expanduser("~/.local/node/bin"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        os.path.expanduser("~/.bun/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ):
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _install_context_mode() -> ComponentResult:
    r = ComponentResult(component="context-mode")
    if _which("context-mode"):
        return r  # already installed

    r.was_missing = True
    node = _which("node")
    if not node:
        r.error = "node not on PATH"
        return r

    npm = os.environ.get("TINYCTX_NPM") or _which("npm")
    if not npm:
        r.error = "npm not on PATH"
        return r

    try:
        proc = subprocess.run(
            [npm, "install", "-g", "context-mode@latest"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        r.error = "npm install timed out after 5min"
        return r
    except OSError as e:
        r.error = f"npm install failed: {e}"
        return r

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        r.error = f"npm install rc={proc.returncode}: {tail}"
        return r

    r.installed = True
    r.actions.append("npm i -g context-mode@latest")
    return r


# ─────────────────────── gitnexus (npm) ────────────────────────────


def _install_gitnexus() -> ComponentResult:
    r = ComponentResult(component="gitnexus")
    from . import gitnexus_bootstrap as gb

    state = gb.detect_state()
    if state.gitnexus_present and state.codex_config_has_gitnexus:
        return r  # already installed + registered

    r.was_missing = True
    report = gb.bootstrap(install=True, config=True, dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    if not report.success:
        r.error = "install failed (see bootstrap log for details)"
    return r


# ─────────────────────── graphify (uv/pipx) ────────────────────────


def _install_graphify() -> ComponentResult:
    r = ComponentResult(component="graphify")
    from . import graphify_bootstrap as gfb

    state = gfb.detect_state()
    cwd = Path.cwd().resolve()
    project_agents = cwd / "AGENTS.md"
    project_hooks = cwd / ".codex" / "hooks.json"
    project_wired = (
        gfb.AGENTS_MARKER in (
            project_agents.read_text(encoding="utf-8", errors="replace")
            if project_agents.is_file() else ""
        )
        and "graphify" in (
            project_hooks.read_text(encoding="utf-8", errors="replace")
            if project_hooks.is_file() else ""
        )
    )

    if state.graphify_present and project_wired:
        return r

    r.was_missing = True
    report = gfb.bootstrap(install=True, project_roots=[cwd], dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    if not report.success:
        r.error = "install failed (see bootstrap log for details)"
    return r


# ─────────────────────── serena (uv/pipx) ──────────────────────────


def _install_serena() -> ComponentResult:
    r = ComponentResult(component="serena")
    from . import serena_bootstrap as sb

    state = sb.detect_state()
    if state.serena_present and state.codex_config_has_serena:
        return r

    r.was_missing = True
    report = sb.bootstrap(install=True, config=True, dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    if not report.success:
        r.error = "install failed (see bootstrap log for details)"
    return r


# ─────────────────────── caveman (git clone) ───────────────────────


def _install_caveman() -> ComponentResult:
    r = ComponentResult(component="caveman")
    from . import caveman_bootstrap as cb

    state = cb.detect_state()
    if state.caveman_shrink_present:
        return r

    r.was_missing = True
    report = cb.bootstrap(install=True, do_config=False, dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    if not report.success:
        r.error = "install failed (see bootstrap log for details)"
    return r


# ──────────────────── config-only bootstraps ───────────────────────


def _register_context_mode_config() -> ComponentResult:
    """Write [mcp_servers.context-mode] to codex config. Idempotent."""
    r = ComponentResult(component="context-mode-config")
    from . import context_mode_bootstrap as cmb

    state = cmb.detect_state()
    if not state.context_mode_present:
        r.error = "context-mode binary not on PATH"
        return r
    if state.codex_config_has_context_mode:
        return r

    r.was_missing = True
    report = cmb.bootstrap(dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    return r


def _register_codex_profile() -> ComponentResult:
    """Write [model_providers.tinyctx] + profiles to codex config.
    Idempotent — skips if already present."""
    r = ComponentResult(component="codex-profile")
    from . import codex_profile_bootstrap as cpb

    state = cpb.detect_state()
    if (state.has_provider_block and state.has_profile_block
            and state.has_goal_profile_block):
        return r

    r.was_missing = True
    report = cpb.bootstrap(dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    return r


def _register_advisor() -> ComponentResult:
    """Write advisor MCP + agent config to codex config and agent file.
    Idempotent — skips if already present."""
    r = ComponentResult(component="advisor")
    from . import advisor_bootstrap as ab

    state = ab.detect_state()
    if (state.codex_config_has_advisor
            and state.codex_config_has_advisor_agent
            and state.advisor_agent_file_exists):
        return r

    r.was_missing = True
    report = ab.bootstrap(dry_run=False)
    if report.success:
        r.installed = True
    if report.actions:
        r.actions = list(report.actions)
    return r


def _register_scout_hook() -> ComponentResult:
    """Write SessionStart scout hook to ~/.codex/hooks.json.
    Idempotent — skips if already registered."""
    r = ComponentResult(component="scout-hook")
    from . import scout_hook_bootstrap as shb

    state = shb.detect_state()
    if state.hook_already_registered and state.script_exists:
        return r
    if not state.script_exists:
        r.error = f"scout script missing: {state.script_path}"
        return r

    r.was_missing = True
    ok, msg = shb.register(dry_run=False)
    if ok:
        r.installed = True
    r.actions.append(msg)
    return r


# ─────────────────────── unified entry ─────────────────────────────


_INSTALLERS = [
    _install_context_mode,
    _register_context_mode_config,
    _register_codex_profile,
    _register_advisor,
    _register_scout_hook,
    _install_gitnexus,
    _install_graphify,
    _install_serena,
    _install_caveman,
]


def install_all_missing() -> dict[str, Any]:
    """Run every component's install if missing, then cross-component
    wiring. Never raises."""
    results: dict[str, dict[str, Any]] = {}
    for fn in _INSTALLERS:
        try:
            r = fn()
            results[r.component] = {
                "was_missing": r.was_missing,
                "installed": r.installed,
                "error": r.error,
                "actions": r.actions,
            }
        except Exception as exc:  # noqa: BLE001
            name = fn.__name__.replace("_install_", "")
            results[name] = {
                "was_missing": True,
                "installed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "actions": [],
            }
    return results


def status_all() -> dict[str, dict[str, Any]]:
    """Detect state for every component without installing."""
    results: dict[str, dict[str, Any]] = {}

    # context-mode
    cm = _which("context-mode")
    results["context-mode"] = {"installed": bool(cm), "path": cm or ""}

    # gitnexus
    from . import gitnexus_bootstrap as gb
    gs = gb.detect_state()
    results["gitnexus"] = {
        "installed": gs.gitnexus_present,
        "path": gs.gitnexus_path,
        "registered": gs.codex_config_has_gitnexus,
    }

    # graphify
    from . import graphify_bootstrap as gfb
    gfs = gfb.detect_state()
    results["graphify"] = {
        "installed": gfs.graphify_present,
        "path": gfs.graphify_path,
    }

    # serena
    from . import serena_bootstrap as sb
    ss = sb.detect_state()
    results["serena"] = {
        "installed": ss.serena_present,
        "path": ss.serena_path,
        "registered": ss.codex_config_has_serena,
    }

    # caveman
    from . import caveman_bootstrap as cb
    cs = cb.detect_state()
    results["caveman"] = {
        "installed": cs.caveman_shrink_present,
        "path": cs.caveman_shrink_path,
    }

    # codex profile (model_provider + profiles)
    from . import codex_profile_bootstrap as cpb
    cps = cpb.detect_state()
    results["codex-profile"] = {
        "installed": (cps.has_provider_block and cps.has_profile_block
                      and cps.has_goal_profile_block),
        "provider": cps.has_provider_block,
        "default_profile": cps.has_profile_block,
        "goal_profile": cps.has_goal_profile_block,
    }

    # advisor (MCP + agent config)
    from . import advisor_bootstrap as ab
    ads = ab.detect_state()
    results["advisor"] = {
        "installed": (ads.codex_config_has_advisor
                      and ads.codex_config_has_advisor_agent
                      and ads.advisor_agent_file_exists),
        "python": ads.python_path,
    }

    # scout hook
    from . import scout_hook_bootstrap as shb
    shs = shb.detect_state()
    results["scout-hook"] = {
        "installed": shs.hook_already_registered and shs.script_exists,
        "script": shs.script_path,
    }

    # context-mode config
    from . import context_mode_bootstrap as cmb
    cms = cmb.detect_state()
    results["context-mode-config"] = {
        "installed": cms.codex_config_has_context_mode,
    }

    return results


# ─────────────────────────── CLI ───────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx")
    p.add_argument(
        "cmd", nargs="?", default="install",
        choices=["install", "status"],
    )
    p.add_argument("--json", action="store_true", help="Output as JSON")
    args = p.parse_args(argv)

    if args.cmd == "status":
        result = status_all()
    else:
        result = install_all_missing()

    if args.json:
        import json as _json
        print(_json.dumps(result, indent=2, default=str))
    else:
        for name, info in result.items():
            if args.cmd == "status":
                status = "installed" if info.get("installed") else "missing"
            else:
                if info.get("error"):
                    status = f"ERROR: {info['error']}"
                elif info.get("installed"):
                    status = "installed (was missing)"
                elif info.get("was_missing"):
                    status = "FAILED"
                else:
                    status = "already installed"
            actions = info.get("actions", [])
            action_str = f"  actions: {actions}" if actions else ""
            print(f"{name}: {status}{action_str}")

    # Return non-zero if any component failed
    for info in result.values():
        if info.get("error"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
