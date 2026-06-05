"""Trigger-file mechanism for keeping scout fresh between codex sessions.

Borrowed-as-idea (not code) from zilliztech/claude-context, where every
``Edit|Write`` PostToolUse hook ``touch``-es ``~/.context/.sync-trigger``
and an ``fs.watch`` listener debounces those events into one incremental
sync per 2-second window
(``packages/mcp/src/sync.ts:317-388``).

tinyctx uses a simpler shape that requires no daemon:

  1. Every codex Edit/Write fires a PostToolUse hook that runs
     :func:`touch_trigger` for the project root, recording an mtime on
     ``~/.tinyctx/cache/<repo_hash>/.scout-trigger``.

  2. The next SessionStart hook treats "trigger newer than scout.md" as
     ``stale`` (see :func:`scout.is_stale`), kicking off the existing
     background-refresh path. Between sessions, any number of edits
     collapse into one mtime bump.

For users who want sub-session latency, ``tinyctx-trigger watch`` runs a
polling loop that refreshes scout in the background as soon as the
trigger fires. Optional, opt-in.

CLI::

    tinyctx-trigger touch                  # touch trigger for cwd
    tinyctx-trigger touch --root /path/to/project
    tinyctx-trigger status                 # show trigger / scout mtimes
    tinyctx-trigger watch [--interval 5]   # poll & refresh in background
    tinyctx-trigger install                # register PostToolUse codex hook
    tinyctx-trigger uninstall
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import scout as scout_mod


CODEX_HOOKS_PATH = Path(
    os.environ.get("TINYCTX_CODEX_HOOKS",
                   str(Path.home() / ".codex" / "hooks.json"))
)
TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME",
                                    str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "scout-trigger.log"

TRIGGER_BASENAME = ".scout-trigger"
"""Filename inside ``cache/<repo_hash>/``. Mirrors claude-context's
``.sync-trigger`` so users porting hooks between the two tools see
the same shape."""


# ---------------------------------------------------------- pure helpers

def trigger_path(project_root: Path) -> Path:
    """Path to the trigger sentinel for this repo. Sibling of scout.md."""
    return scout_mod.cache_dir(project_root) / TRIGGER_BASENAME


def touch_trigger(project_root: Path) -> Path:
    """Bump the trigger mtime; create the file if missing. Returns the path.

    Idempotent and fast (single ``utime`` syscall in the steady state) so
    it's safe to call from a hook on every Edit/Write.
    """
    p = trigger_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        os.utime(p, None)
    else:
        p.write_bytes(b"")
    return p


def is_trigger_newer_than_scout(project_root: Path) -> bool:
    """True iff the trigger was touched after scout.md was last written.

    Returns False when either file is missing — staleness in those cases
    is detected by the existing manifest / file-hash walk.
    """
    trig = trigger_path(project_root)
    sm = scout_mod.scout_path(project_root)
    if not trig.is_file() or not sm.is_file():
        return False
    try:
        return trig.stat().st_mtime > sm.stat().st_mtime
    except OSError:
        return False


# ---------------------------------------------------------- watcher loop

def _refresh_in_background(project_root: Path, venv_py: str) -> None:
    """Spawn ``tinyctx-scout refresh`` detached so the watcher can keep
    polling. Output goes to the standard scout-refresh log."""
    log_path = TINYCTX_HOME / "logs" / "scout-refresh.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.Popen(
            [venv_py, "-m", "tinyctx.scout", "refresh", "--root", str(project_root)],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def watch(project_roots: list[Path], *, interval_s: float = 5.0,
          venv_py: str | None = None) -> None:
    """Poll trigger mtimes; on change, run scout refresh in the background.

    Designed to be supervised by launchd / systemd / a tmux pane. Loops
    forever; ``KeyboardInterrupt`` exits cleanly.
    """
    venv_py = venv_py or sys.executable
    last_seen: dict[str, float] = {}
    print(f"[trigger-watch] watching {len(project_roots)} repos "
          f"every {interval_s:g}s", file=sys.stderr)
    try:
        while True:
            for root in project_roots:
                trig = trigger_path(root)
                if not trig.is_file():
                    continue
                try:
                    m = trig.stat().st_mtime
                except OSError:
                    continue
                key = str(root)
                if last_seen.get(key, 0.0) < m:
                    last_seen[key] = m
                    print(f"[trigger-watch] {root} → refresh", file=sys.stderr)
                    _refresh_in_background(root, venv_py)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("[trigger-watch] stopped", file=sys.stderr)


# ---------------------------------------------------------- bootstrap

@dataclass
class HookState:
    disabled: bool = False
    hooks_file_exists: bool = False
    script_path: str = ""
    script_exists: bool = False
    hook_already_registered: bool = False


def _resolve_main_repo(repo_root: Path) -> Path:
    """Same worktree-stripping logic as scout_hook_bootstrap so hook
    paths survive worktree cleanup."""
    parts = repo_root.parts
    try:
        idx = parts.index("worktrees")
    except ValueError:
        return repo_root
    if idx >= 1 and parts[idx - 1] == ".claude":
        return Path(*parts[: idx - 1])
    return repo_root


def _default_script_path() -> str:
    forced = os.environ.get("TINYCTX_TRIGGER_HOOK_SCRIPT")
    if forced:
        return forced
    raw_root = Path(__file__).resolve().parent.parent
    here = _resolve_main_repo(raw_root)
    cand = here / "scripts" / "scout-posttool.sh"
    if cand.is_file():
        return str(cand)
    cand2 = (Path.home() / "dev" / "tinyctx"
             / "scripts" / "scout-posttool.sh")
    if cand2.is_file():
        return str(cand2)
    return str(raw_root / "scripts" / "scout-posttool.sh")


def detect_hook_state(hooks_path: Path = CODEX_HOOKS_PATH) -> HookState:
    s = HookState()
    s.disabled = os.environ.get("TINYCTX_TRIGGER_HOOK_DISABLE", "0") == "1"
    s.script_path = _default_script_path()
    s.script_exists = Path(s.script_path).is_file()
    s.hooks_file_exists = hooks_path.is_file()
    if s.hooks_file_exists:
        try:
            data = json.loads(hooks_path.read_text(encoding="utf-8"))
            ptu = (data.get("hooks") or {}).get("PostToolUse") or []
            for group in ptu:
                if not isinstance(group, dict):
                    continue
                for h in (group.get("hooks") or []):
                    if isinstance(h, dict) and "scout-posttool" in (h.get("command") or ""):
                        s.hook_already_registered = True
        except (OSError, json.JSONDecodeError):
            pass
    return s


def register_hook(hooks_path: Path = CODEX_HOOKS_PATH, *,
                  dry_run: bool = False) -> tuple[bool, str]:
    """Idempotently add a PostToolUse entry running scout-posttool.sh.
    Co-exists with any pre-existing PostToolUse entries (e.g. cm-hook-shim)."""
    state = detect_hook_state(hooks_path)
    if state.hook_already_registered:
        return True, "already registered"
    if not state.script_exists:
        return False, f"hook script missing: {state.script_path}"

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
    if not isinstance(data["hooks"].get("PostToolUse"), list):
        data["hooks"]["PostToolUse"] = []

    data["hooks"]["PostToolUse"].append({
        "matcher": "Edit|Write",
        "hooks": [
            {"type": "command", "command": state.script_path,
             "_added_by": "tinyctx.scout_trigger"},
        ],
    })

    if dry_run:
        return True, f"DRY-RUN would write {hooks_path}"
    try:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(json.dumps(data, indent=2) + "\n",
                              encoding="utf-8")
    except OSError as e:
        return False, f"write failed: {e}"
    return True, f"registered PostToolUse hook at {hooks_path}"


def unregister_hook(hooks_path: Path = CODEX_HOOKS_PATH, *,
                    dry_run: bool = False) -> tuple[bool, str]:
    if not hooks_path.is_file():
        return True, "no hooks.json to clean"
    try:
        data = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, f"hooks.json unreadable: {e}"

    ptu = (data.get("hooks") or {}).get("PostToolUse") or []
    new_groups: list = []
    removed = 0
    for group in ptu:
        if not isinstance(group, dict):
            new_groups.append(group)
            continue
        kept = []
        for h in (group.get("hooks") or []):
            if isinstance(h, dict):
                cmd = h.get("command") or ""
                added_by = h.get("_added_by") or ""
                if "scout-posttool" in cmd or added_by == "tinyctx.scout_trigger":
                    removed += 1
                    continue
            kept.append(h)
        if kept:
            ng = dict(group)
            ng["hooks"] = kept
            new_groups.append(ng)
    if removed == 0:
        return True, "no scout-trigger entry found"

    if "hooks" not in data or not isinstance(data["hooks"], dict):
        data["hooks"] = {}
    data["hooks"]["PostToolUse"] = new_groups
    if dry_run:
        return True, f"DRY-RUN would remove {removed} entry/entries"
    try:
        hooks_path.write_text(json.dumps(data, indent=2) + "\n",
                              encoding="utf-8")
    except OSError as e:
        return False, f"write failed: {e}"
    return True, f"removed {removed} scout-trigger entry/entries"


# ---------------------------------------------------------------------- CLI

def _cmd_touch(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    p = touch_trigger(root)
    if not args.quiet:
        print(p)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    trig = trigger_path(root)
    sm = scout_mod.scout_path(root)
    out = {
        "project_root": str(root),
        "trigger_path": str(trig),
        "trigger_exists": trig.is_file(),
        "trigger_mtime": trig.stat().st_mtime if trig.is_file() else None,
        "scout_path": str(sm),
        "scout_exists": sm.is_file(),
        "scout_mtime": sm.stat().st_mtime if sm.is_file() else None,
        "trigger_newer_than_scout": is_trigger_newer_than_scout(root),
    }
    hook = detect_hook_state()
    out["hook_registered"] = hook.hook_already_registered
    out["hook_script"] = hook.script_path
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"  {k:<32}  {v}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    if args.root != []:
        roots = [Path(r).resolve() for r in args.root]
    else:
        # Watch every registered project (mirrors dreamer behaviour).
        try:
            from . import registry
            roots = [Path(r).resolve() for r in registry.list_projects()]
        except Exception:  # noqa: BLE001
            roots = [Path.cwd().resolve()]
    if not roots:
        print("no project roots to watch (none registered, none given)",
              file=sys.stderr)
        return 1
    watch(roots, interval_s=args.interval)
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    state = detect_hook_state()
    if state.disabled:
        if not args.quiet:
            print("⏭ TINYCTX_TRIGGER_HOOK_DISABLE=1 — skipping",
                  file=sys.stderr)
        return 0
    ok, msg = register_hook(dry_run=args.dry_run)
    if not args.quiet:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {msg}")
    return 0 if ok else 1


def _cmd_uninstall(args: argparse.Namespace) -> int:
    ok, msg = unregister_hook(dry_run=args.dry_run)
    if not args.quiet:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {msg}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx-trigger")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("touch", help="bump trigger mtime for a repo")
    pt.add_argument("--root", default=".")
    pt.add_argument("--quiet", action="store_true")
    pt.set_defaults(func=_cmd_touch)

    ps = sub.add_parser("status", help="show trigger + scout mtimes + hook state")
    ps.add_argument("--root", default=".")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=_cmd_status)

    pw = sub.add_parser("watch",
                        help="poll registered repos and refresh scout when trigger bumps")
    pw.add_argument("--root", action="append", default=[],
                    help="repeatable; defaults to all registered projects")
    pw.add_argument("--interval", type=float, default=5.0)
    pw.set_defaults(func=_cmd_watch)

    pi = sub.add_parser("install",
                        help="register the PostToolUse codex hook")
    pi.add_argument("--dry-run", action="store_true")
    pi.add_argument("--quiet", action="store_true")
    pi.set_defaults(func=_cmd_install)

    pu = sub.add_parser("uninstall",
                        help="remove the PostToolUse codex hook")
    pu.add_argument("--dry-run", action="store_true")
    pu.add_argument("--quiet", action="store_true")
    pu.set_defaults(func=_cmd_uninstall)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
