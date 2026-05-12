"""Session continuity: persist compaction summaries so a new session can
recall them after the previous one ran out of context.

Layout (per repo, derived from cwd at session time):

    ~/.tinyctx/cache/<repo-hash>/sessions/
        <session-id>/
            compaction-1.md
            compaction-2.md
            ...
        latest.md           # symlink (or copy on platforms w/o symlink) to the
                            # most recent compaction.md across all sessions

Why per-repo not per-session: when codex hits its limit and you `/clear`, the
new session has a fresh session_id but is still working on the same repo. The
useful continuity unit is the repo. We keep per-session sub-dirs only to
make timelines reproducible.

This module is intentionally tiny — persistence + recall + a CLI. It does
NOT try to auto-inject into the next session (that would conflict with
codex's own resume behaviour). The user runs `tinyctx-recall` and pastes the
summary into the new session, or wires it into AGENTS.md if they want.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .scout import cache_dir, repo_hash  # reuse the same per-repo hashing


def sessions_dir(project_root: Path) -> Path:
    return cache_dir(project_root) / "sessions"


def _next_compaction_index(session_root: Path) -> int:
    if not session_root.is_dir():
        return 1
    n = 0
    for f in session_root.glob("compaction-*.md"):
        try:
            n = max(n, int(f.stem.split("-", 1)[1]))
        except (IndexError, ValueError):
            # Why: malformed stem (e.g. compaction-foo.md from manual
            # editing) — skip it and keep scanning. Caller derives next
            # index from the well-formed files.
            continue
    return n + 1


def save_compaction(project_root: Path, session_id: str, summary: str,
                    *, telemetry: dict | None = None,
                    structured: dict[str, Any] | None = None) -> Path:
    """Persist a compaction summary. Returns the path of compaction-N.md.

    When `structured` is provided (compartments / facts / open_questions),
    a sibling `compaction-N.json` is also written so `tinyctx-recall` can
    surface a subset (just facts, just one compartment, etc.) without
    forcing the full markdown back into the next session's context.

    Updates `latest.md` for cheap recall.
    """
    sid = session_id or "_unknown"
    sroot = sessions_dir(project_root) / sid
    sroot.mkdir(parents=True, exist_ok=True)
    n = _next_compaction_index(sroot)
    p = sroot / f"compaction-{n}.md"

    header = (
        f"<!-- tinyctx compaction\n"
        f"     session: {sid}\n"
        f"     when:    {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"-->\n\n"
    )
    if telemetry:
        header += f"<!-- telemetry: {telemetry} -->\n\n"
    p.write_text(header + summary)

    if structured is not None:
        json_path = sroot / f"compaction-{n}.json"
        payload = {
            "session_id": sid,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "compartments": structured.get("compartments") or [],
            "facts": structured.get("facts") or [],
            "open_questions": structured.get("open_questions") or [],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Update latest.md (try symlink, fall back to copy).
    latest = sessions_dir(project_root) / "latest.md"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(p, latest)
    except (OSError, NotImplementedError):
        # Symlink unsupported (Windows, restricted fs) — fall back to copy.
        try:
            latest.write_text(p.read_text())
        except OSError:
            # Why: latest.md is a convenience pointer for recall; if
            # both symlink and copy fail, the compaction itself is
            # already persisted at `p` so recall can still find it.
            pass
    return p


def latest_structured(project_root: Path) -> dict[str, Any] | None:
    """Load the latest compaction's .json sidecar if present."""
    paths = recall(project_root, all_sessions=False, limit=1)
    if not paths:
        return None
    json_path = paths[0].with_suffix(".json")
    if not json_path.is_file():
        return None
    try:
        return json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        # Why: corrupted or unreadable sidecar — caller treats None as
        # "no structured recall available" and falls back to markdown.
        return None


def recall(project_root: Path, *, all_sessions: bool = False,
           limit: int = 1) -> list[Path]:
    """Return up to `limit` most recent compaction paths. With all_sessions
    False, considers only the latest session (by mtime); with True, considers
    every session under this repo."""
    root = sessions_dir(project_root)
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    if all_sessions:
        for sd in root.iterdir():
            if sd.is_dir():
                candidates.extend(sd.glob("compaction-*.md"))
    else:
        # most-recently-touched session dir
        sdirs = [d for d in root.iterdir() if d.is_dir()]
        if not sdirs:
            return []
        latest_sdir = max(sdirs, key=lambda d: d.stat().st_mtime)
        candidates = list(latest_sdir.glob("compaction-*.md"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def list_sessions(project_root: Path) -> list[tuple[str, int, float]]:
    """Return (session_id, compaction_count, last_mtime) per session."""
    root = sessions_dir(project_root)
    if not root.is_dir():
        return []
    out = []
    for sd in root.iterdir():
        if not sd.is_dir():
            continue
        files = list(sd.glob("compaction-*.md"))
        if not files:
            continue
        last = max(f.stat().st_mtime for f in files)
        out.append((sd.name, len(files), last))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.recall")
    p.add_argument("--root", default=".")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--all-sessions", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="list per-session compaction counts instead of printing content")
    p.add_argument("--facts-only", action="store_true",
                   help="print only the structured facts list from the latest compaction")
    p.add_argument("--compartment", default=None,
                   help="print only the named compartment's summary from the latest compaction")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    if args.list:
        sessions = list_sessions(root)
        if not sessions:
            sys.stderr.write("(no compactions stored for this repo)\n")
            return 1
        for sid, count, mtime in sessions:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            print(f"{ts}  {count} compactions  {sid}")
        return 0

    if args.facts_only or args.compartment:
        data = latest_structured(root)
        if not data:
            sys.stderr.write(
                "(no structured compaction found for this repo; "
                "this requires compactor_debate output)\n"
            )
            return 1
        if args.facts_only:
            facts = data.get("facts") or []
            if not facts:
                sys.stderr.write("(no facts recorded)\n")
                return 1
            for f in facts:
                claim = f.get("claim", "")
                ev = f.get("evidence", "")
                line = f"- {claim}"
                if ev:
                    line += f"  _(evidence: {ev})_"
                print(line)
            return 0
        if args.compartment:
            comps = data.get("compartments") or []
            match = next(
                (c for c in comps if c.get("name") == args.compartment
                 or c.get("topic") == args.compartment),
                None,
            )
            if not match:
                sys.stderr.write(
                    f"(no compartment named {args.compartment!r}; available: "
                    f"{[c.get('name') for c in comps]})\n"
                )
                return 1
            print(f"# {match.get('name')} — {match.get('topic', '')}")
            print(match.get("summary", ""))
            files = match.get("files") or []
            if files:
                print("\n## Files\n" + "\n".join(f"- `{f}`" for f in files))
            return 0

    paths = recall(root, all_sessions=args.all_sessions, limit=args.limit)
    if not paths:
        sys.stderr.write("(no compactions stored for this repo)\n")
        return 1
    for i, path in enumerate(paths):
        if i > 0:
            print("\n---\n")
        print(f"# {path}")
        print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
