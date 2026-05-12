"""Auto-register the scout SessionStart hook in ~/.codex/hooks.json.

The scout subagent generates a per-repo `~/.tinyctx/cache/<hash>/scout.md`
summary; codex's SessionStart hook is what actually injects it as
`additionalContext` for turn 1. Without that hook, the scout cache exists
but never reaches a session — exactly the gap observed in production.

This module merges a single SessionStart entry pointing at
`scripts/scout-session-start.sh` into the codex hooks file. Idempotent:
detected via the hook's command path; if it's already present we leave
the file untouched. Co-exists with the existing `cm-hook-shim
sessionstart` entry — codex runs every entry under SessionStart in order.

Disable: TINYCTX_SCOUT_HOOK_DISABLE=1
Override script path: TINYCTX_SCOUT_HOOK_SCRIPT=/path/to/scout-session-start.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


CODEX_HOOKS_PATH = Path(
    os.environ.get("TINYCTX_CODEX_HOOKS",
                   str(Path.home() / ".codex" / "hooks.json"))
)
TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME",
                                    str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "scout-hook-bootstrap.log"


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        # Why: _log itself must never raise — hook bootstrap is advisory.
        pass


def _resolve_main_repo(repo_root: Path) -> Path:
    """Return the stable main-checkout root if `repo_root` is a worktree.

    Agent harnesses (claude-code, etc.) commonly run inside throwaway
    worktrees at `<main>/.claude/worktrees/<branch>/`. Pinning hook
    commands to a worktree path turns into a dangling reference the
    moment the worktree is deleted (which happens at conversation end).

    Strip the `.claude/worktrees/<name>/` segment so we always return a
    path that survives worktree cleanup.

    Example:
        /Users/x/dev/tinyctx/.claude/worktrees/keen-gates/  →
        /Users/x/dev/tinyctx/
    """
    parts = repo_root.parts
    try:
        idx = parts.index("worktrees")
    except ValueError:
        return repo_root
    if idx >= 1 and parts[idx - 1] == ".claude":
        return Path(*parts[: idx - 1])
    return repo_root


def _default_script_path() -> str:
    """Locate scripts/scout-session-start.sh shipped with this install,
    preferring the stable main checkout over any worktree the bootstrap
    happens to be running from."""
    forced = os.environ.get("TINYCTX_SCOUT_HOOK_SCRIPT")
    if forced:
        return forced
    # tinyctx/scout_hook_bootstrap.py -> repo (or worktree) root
    raw_root = Path(__file__).resolve().parent.parent
    here = _resolve_main_repo(raw_root)
    cand = here / "scripts" / "scout-session-start.sh"
    if cand.is_file():
        return str(cand)
    # Fall back to ~/dev/tinyctx (typical user layout)
    cand2 = (Path.home() / "dev" / "tinyctx"
             / "scripts" / "scout-session-start.sh")
    if cand2.is_file():
        return str(cand2)
    # Last resort: worktree-local. Caller will check existence.
    return str(raw_root / "scripts" / "scout-session-start.sh")


@dataclass
class State:
    disabled: bool = False
    hooks_file_exists: bool = False
    script_path: str = ""
    script_exists: bool = False
    hook_already_registered: bool = False


def detect_state(hooks_path: Path = CODEX_HOOKS_PATH) -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_SCOUT_HOOK_DISABLE", "0") == "1"
    s.script_path = _default_script_path()
    s.script_exists = Path(s.script_path).is_file()
    s.hooks_file_exists = hooks_path.is_file()
    if s.hooks_file_exists:
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
            session_start = (data.get("hooks") or {}).get("SessionStart") or []
            for group in session_start:
                if not isinstance(group, dict):
                    continue
                for h in (group.get("hooks") or []):
                    if not isinstance(h, dict):
                        continue
                    cmd = h.get("command") or ""
                    if "scout-session-start" in cmd:
                        s.hook_already_registered = True
        except (OSError, json.JSONDecodeError) as e:
            _log(f"hooks.json read failed: {e}")
    return s


def register(hooks_path: Path = CODEX_HOOKS_PATH, *,
             dry_run: bool = False) -> tuple[bool, str]:
    """Idempotently add a SessionStart entry running scout-session-start.sh.
    Co-exists with any pre-existing SessionStart entries.
    """
    state = detect_state(hooks_path)
    if state.hook_already_registered:
        return True, "already registered"
    if not state.script_exists:
        return False, f"hook script missing: {state.script_path}"

    # Read or initialize hooks JSON.
    if state.hooks_file_exists:
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return False, f"hooks.json unreadable: {e}"
    else:
        data = {"hooks": {}}
    if not isinstance(data, dict):
        data = {"hooks": {}}
    if not isinstance(data.get("hooks"), dict):
        data["hooks"] = {}
    if not isinstance(data["hooks"].get("SessionStart"), list):
        data["hooks"]["SessionStart"] = []

    # Append our entry. Mirror codex's documented schema:
    #   { "hooks": [ {"type":"command","command":"<sh>"} ] }
    data["hooks"]["SessionStart"].append({
        "hooks": [
            {"type": "command", "command": state.script_path,
             "_added_by": "tinyctx.scout_hook_bootstrap"},
        ]
    })

    if dry_run:
        return True, f"DRY-RUN would write {hooks_path}"

    try:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(json.dumps(data, indent=2) + "\n",
                              encoding="utf-8")
    except OSError as e:
        return False, f"write failed: {e}"
    return True, f"registered SessionStart hook at {hooks_path}"


def unregister(hooks_path: Path = CODEX_HOOKS_PATH, *,
               dry_run: bool = False) -> tuple[bool, str]:
    """Remove our SessionStart hook entry. Leaves other entries (e.g.
    cm-hook-shim) untouched."""
    if not hooks_path.is_file():
        return True, "no hooks.json to clean"
    try:
        data = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, f"hooks.json unreadable: {e}"

    session_start = (data.get("hooks") or {}).get("SessionStart") or []
    new_groups: list = []
    removed = 0
    for group in session_start:
        if not isinstance(group, dict):
            new_groups.append(group)
            continue
        kept_hooks = []
        for h in group.get("hooks") or []:
            if isinstance(h, dict):
                cmd = h.get("command") or ""
                if "scout-session-start" in cmd or \
                   h.get("_added_by") == "tinyctx.scout_hook_bootstrap":
                    removed += 1
                    continue
            kept_hooks.append(h)
        if kept_hooks:
            new_group = dict(group)
            new_group["hooks"] = kept_hooks
            new_groups.append(new_group)
    if removed == 0:
        return True, "no scout hook entry found"

    if "hooks" not in data or not isinstance(data["hooks"], dict):
        data["hooks"] = {}
    data["hooks"]["SessionStart"] = new_groups
    if dry_run:
        return True, f"DRY-RUN would remove {removed} entry/entries"
    try:
        hooks_path.write_text(json.dumps(data, indent=2) + "\n",
                              encoding="utf-8")
    except OSError as e:
        return False, f"write failed: {e}"
    return True, f"removed {removed} scout entry/entries from {hooks_path}"


def _print_state(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("script path", state.script_path),
        ("script exists", "yes" if state.script_exists else "no"),
        ("hooks file exists", "yes" if state.hooks_file_exists else "no"),
        ("scout hook registered",
         "yes" if state.hook_already_registered else "no"),
    ]
    print("scout-hook state:")
    for k, v in rows:
        print(f"  {k:<28}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.scout_hook_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--hooks", default=str(CODEX_HOOKS_PATH))
    args = p.parse_args(argv)
    hooks_path = Path(args.hooks).expanduser()

    if args.cmd == "status":
        _print_state(detect_state(hooks_path))
        return 0

    state = detect_state(hooks_path)
    if state.disabled:
        if not args.quiet:
            print("⏭ TINYCTX_SCOUT_HOOK_DISABLE=1 — skipping",
                  file=sys.stderr)
        return 0

    if args.cmd == "uninstall":
        ok, msg = unregister(hooks_path, dry_run=args.dry_run)
    else:
        ok, msg = register(hooks_path, dry_run=args.dry_run)

    if not args.quiet:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
