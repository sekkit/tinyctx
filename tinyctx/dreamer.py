"""Periodic maintenance for tinyctx state.

Inspired by cortexkit/magic-context's Dreamer (overnight consolidation),
but stripped to the bones: a single CLI you wire into cron / launchd /
systemd. For each registered project (see registry.py) it runs:

    1. tinyctx-scout refresh           rebuild scout.md if any tracked file changed
    2. tinyctx-keypin scan             refresh keyfiles.md from recent rollouts
    3. tinyctx-mem ingest-compaction   (if --ingest-mem) push facts into mem0
    4. (--gc) garbage-collect old session caches

All sub-commands are spawned as separate processes so a failure in one
project doesn't kill the run for others. Output is streamed line-by-line
to stdout (and to ~/.tinyctx/logs/dreamer.log when run via the bundled
launchd plist).

CLI:
    tinyctx-dreamer run [--ingest-mem] [--gc] [--retention-days 30]
    tinyctx-dreamer list
    tinyctx-dreamer register   [--root .]
    tinyctx-dreamer unregister [--root .]
    tinyctx-dreamer install-launchd     # macOS daily 03:00
    tinyctx-dreamer install-cron        # print a crontab line for Linux
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import registry


# ------------------------------------------------------------- run


def _run(label: str, argv: list[str]) -> int:
    """Run a sub-command and stream a one-line summary."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"  {label}: error ({e.__class__.__name__})")
        return 1
    out = (r.stdout or "").strip().splitlines()
    err = (r.stderr or "").strip().splitlines()
    note = (out[-1] if out else (err[-1] if err else "")).strip()[:160]
    print(f"  {label}: rc={r.returncode}  {note}")
    return r.returncode


def cmd_run(args) -> int:
    projects = registry.all_projects()
    if not projects:
        print(
            "(no projects registered; run scout init / keypin scan in a "
            "repo first, or `tinyctx-dreamer register --root .`)"
        )
        return 1

    started = time.time()
    fails = 0
    for proj in projects:
        print(f"=== {proj}")
        graph_paths = (
            list(proj.glob("tinyctx-graph.json")) +
            list((proj / "graphify-out").glob("graph.json"))
        )
        if graph_paths:
            if _run("scout refresh",
                    ["tinyctx-scout", "refresh", "--root", str(proj)]) != 0:
                fails += 1
        else:
            print("  scout refresh: skip (no graph.json)")
        if _run("keypin scan",
                ["tinyctx-keypin", "scan", "--root", str(proj)]) != 0:
            fails += 1
        if args.ingest_mem:
            if _run("mem ingest",
                    ["tinyctx-mem", "ingest-compaction", "--root", str(proj)]) != 0:
                fails += 1

    if args.gc:
        deleted = _gc_old_sessions(args.retention_days)
        print(f"=== gc: removed {deleted} session dirs older than "
              f"{args.retention_days}d")

    print(f"=== done in {time.time() - started:.1f}s "
          f"({len(projects)} projects, {fails} sub-failures)")
    return 0 if fails == 0 else 1


def _gc_old_sessions(retention_days: int) -> int:
    cutoff = time.time() - retention_days * 86_400
    cache_root = Path.home() / ".tinyctx" / "cache"
    if not cache_root.is_dir():
        return 0
    deleted = 0
    for repo_cache in cache_root.iterdir():
        sessions = repo_cache / "sessions"
        if not sessions.is_dir():
            continue
        for sdir in sessions.iterdir():
            if not sdir.is_dir():
                continue
            try:
                files = list(sdir.glob("*.md"))
                if not files:
                    continue
                latest_mtime = max(f.stat().st_mtime for f in files)
                if latest_mtime < cutoff:
                    shutil.rmtree(sdir)
                    deleted += 1
            except OSError:
                continue
    return deleted


# --------------------------------------------------------- registration


def cmd_list(args) -> int:
    projects = registry.all_projects()
    if not projects:
        print("(none)")
        return 1
    for p in projects:
        print(p)
    return 0


def cmd_register(args) -> int:
    added = registry.register(Path(args.root).resolve())
    print(f"{'registered' if added else 'already registered'}: "
          f"{Path(args.root).resolve()}")
    return 0


def cmd_unregister(args) -> int:
    removed = registry.unregister(Path(args.root).resolve())
    print(f"{'unregistered' if removed else 'not registered'}: "
          f"{Path(args.root).resolve()}")
    return 0


# ----------------------------------------------------------- launchd / cron


_LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.tinyctx.dreamer</string>
  <key>ProgramArguments</key>
  <array>
    <string>{exe}</string>
    <string>run</string>
    <string>--gc</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>{path}</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>{log}</string>
  <key>StandardErrorPath</key>
  <string>{err}</string>
</dict>
</plist>
"""


def cmd_install_launchd(args) -> int:
    exe = shutil.which("tinyctx-dreamer")
    if not exe:
        print("tinyctx-dreamer not on PATH; install tinyctx with `pip install -e .`")
        return 1
    plist_path = (
        Path.home() / "Library" / "LaunchAgents" / "com.tinyctx.dreamer.plist"
    )
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path.home() / ".tinyctx" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(_LAUNCHD_PLIST_TEMPLATE.format(
        exe=exe,
        path="/usr/local/bin:/usr/bin:/bin",
        log=str(log_dir / "dreamer.log"),
        err=str(log_dir / "dreamer.err"),
    ))
    print(f"wrote {plist_path}")
    print(f"to enable: launchctl load -w {plist_path}")
    print(f"to disable: launchctl unload {plist_path}")
    return 0


def cmd_install_cron(args) -> int:
    exe = shutil.which("tinyctx-dreamer") or "tinyctx-dreamer"
    log = Path.home() / ".tinyctx" / "logs" / "dreamer.log"
    line = f"0 3 * * *  {exe} run --gc >> {log} 2>&1"
    print("# Add this to your crontab (`crontab -e`):")
    print(line)
    return 0


# ----------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.dreamer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run all maintenance for every registered project")
    pr.add_argument("--ingest-mem", action="store_true",
                    help="also push compaction facts into mem0")
    pr.add_argument("--gc", action="store_true",
                    help="garbage-collect session dirs older than --retention-days")
    pr.add_argument("--retention-days", type=int, default=30)
    pr.set_defaults(_fn=cmd_run)

    sub.add_parser("list", help="print registered projects").set_defaults(_fn=cmd_list)

    pre = sub.add_parser("register"); pre.add_argument("--root", default="."); pre.set_defaults(_fn=cmd_register)
    pun = sub.add_parser("unregister"); pun.add_argument("--root", default="."); pun.set_defaults(_fn=cmd_unregister)

    sub.add_parser("install-launchd",
                   help="install macOS launchd plist (daily 03:00)").set_defaults(_fn=cmd_install_launchd)
    sub.add_parser("install-cron",
                   help="print a crontab line for Linux").set_defaults(_fn=cmd_install_cron)

    args = p.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
