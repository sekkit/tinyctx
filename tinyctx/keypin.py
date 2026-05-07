"""Pinned key files: scan codex's rollout JSONL for Read-tool frequency, then
emit a `keyfiles.md` ranked by how often each file actually got read across
recent sessions.

Inspired by cortexkit/magic-context's `dreamer.pin_key_files`. Static graph
ranking (tinyctx.interest, the compression-biased PageRank) tells us what
SHOULD be load-bearing in theory; this module tells us what HAS been
load-bearing in practice. Both signals are useful; they're complementary.

Codex stores every session under
    ~/.codex/sessions/YYYY/MM/DD/rollout-<session-id>.jsonl

Each line is one event. Read-tool calls show up as
    {"tool_name": "Read", "tool_input": {"file_path": "..."}, ...}
or as MCP-routed tool calls. We collect every file-path-shaped argument
(`file_path` / `path` / `target_file` etc.) across all recent rollouts and
filter to the project root the user invoked us in.

The output `keyfiles.md` is byte-stable across rebuilds (sorted by count
desc, then path asc) so it's safe to drop into a prompt-cached preamble.

CLI:
    tinyctx-keypin scan [--root .] [--top-n 20] [--days 30]
    tinyctx-keypin show [--root .]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from .scout import cache_dir   # reuse the per-repo hashing convention


_DEFAULT_ROLLOUT_DIR = Path.home() / ".codex" / "sessions"
_DEFAULT_DAYS = 60       # only consider rollouts modified in the last N days
_DEFAULT_TOP_N = 20

# Tool-input keys we treat as file-path arguments. Order matters: first hit wins.
_PATH_KEYS = ("file_path", "path", "target_file", "filename", "filepath")

# Tool names whose calls represent reading or listing files. We deliberately
# include MCP-routed read tools (mcp__plugin_*__) by prefix match.
_READ_TOOL_NAMES = {
    "Read", "read", "read_file", "Cat", "cat",
    "view_file", "view", "Get", "fs_read",
}


def _looks_like_read(tool_name: str) -> bool:
    if tool_name in _READ_TOOL_NAMES:
        return True
    low = tool_name.lower()
    if low.startswith("mcp__"):
        return ("read" in low) or ("view" in low) or ("get" in low)
    return False


def _extract_path(tool_input: object) -> str | None:
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_input, dict):
        return None
    for key in _PATH_KEYS:
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _iter_rollout_files(rollout_dir: Path, *, days: int) -> Iterable[Path]:
    if not rollout_dir.is_dir():
        return
    cutoff = time.time() - days * 86_400
    for p in rollout_dir.rglob("rollout-*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        yield p


def scan_rollouts(rollout_dir: Path = _DEFAULT_ROLLOUT_DIR,
                  *, days: int = _DEFAULT_DAYS) -> Counter[str]:
    """Scan all rollouts modified within `days` and count file-path
    references in Read-style tool calls."""
    counts: Counter[str] = Counter()
    for f in _iter_rollout_files(rollout_dir, days=days):
        try:
            for line in f.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tool_name = ev.get("tool_name") or (
                    ev.get("payload") or {}).get("tool_name") or ""
                if not _looks_like_read(tool_name):
                    continue
                tool_input = ev.get("tool_input") or (
                    ev.get("payload") or {}).get("tool_input")
                path = _extract_path(tool_input)
                if path:
                    counts[path] += 1
        except OSError:
            continue
    return counts


def filter_to_project(counts: Counter[str], project_root: Path) -> Counter[str]:
    """Keep only file paths that resolve inside `project_root`. Returned
    counter uses paths relative to the project root."""
    proj = project_root.resolve()
    out: Counter[str] = Counter()
    for path, n in counts.items():
        try:
            p = Path(path)
            resolved = p.resolve() if p.is_absolute() else (project_root / p).resolve()
            try:
                rel = resolved.relative_to(proj)
            except ValueError:
                continue
            out[str(rel)] += n
        except (OSError, RuntimeError):
            continue
    return out


def keyfiles_path(project_root: Path) -> Path:
    return cache_dir(project_root) / "keyfiles.md"


def write_keyfiles(counts: Counter[str], project_root: Path,
                   *, top_n: int = _DEFAULT_TOP_N) -> Path:
    """Render the top-N files into a byte-stable markdown digest."""
    cdir = cache_dir(project_root)
    cdir.mkdir(parents=True, exist_ok=True)
    p = keyfiles_path(project_root)

    items = counts.most_common(top_n)
    # secondary sort: lex asc within same count for byte-stability.
    items.sort(key=lambda kv: (-kv[1], kv[0]))

    if not items:
        body = (
            "# Frequently-accessed files\n\n"
            "No Read-tool calls observed in recent codex rollouts for this repo.\n"
            "Run a few sessions and re-run `tinyctx-keypin scan`.\n"
        )
    else:
        lines = [
            "# Frequently-accessed files",
            "",
            "Ranked by Read-tool frequency in recent codex sessions. "
            "Complementary to `scout.md`'s structural ranking.",
            "",
            "| Reads | File |",
            "|------:|:-----|",
        ]
        for path, n in items:
            lines.append(f"| {n} | `{path}` |")
        body = "\n".join(lines) + "\n"
    p.write_text(body)
    return p


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.keypin")
    sub = p.add_subparsers(dest="cmd", required=True)

    pscan = sub.add_parser("scan", help="rebuild keyfiles.md from codex rollouts")
    pscan.add_argument("--root", default=".")
    pscan.add_argument("--rollout-dir", default=str(_DEFAULT_ROLLOUT_DIR))
    pscan.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    pscan.add_argument("--days", type=int, default=_DEFAULT_DAYS)

    pshow = sub.add_parser("show", help="print keyfiles.md")
    pshow.add_argument("--root", default=".")

    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    if args.cmd == "scan":
        all_counts = scan_rollouts(Path(args.rollout_dir).expanduser(),
                                   days=args.days)
        if not all_counts:
            sys.stderr.write(f"(no Read-tool calls in {args.rollout_dir} "
                             f"within {args.days}d)\n")
        proj_counts = filter_to_project(all_counts, root)
        path = write_keyfiles(proj_counts, root, top_n=args.top_n)
        # Auto-register so `tinyctx-dreamer run` picks this repo up.
        try:
            from . import registry
            registry.register(root)
        except Exception:  # noqa: BLE001
            pass
        print(path)
        return 0

    if args.cmd == "show":
        path = keyfiles_path(root)
        if not path.is_file():
            sys.stderr.write(f"(no keyfiles.md at {path}; run scan first)\n")
            return 1
        sys.stdout.write(path.read_text())
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
