"""tinyctx-stats — read JSONL logs and summarize routing decisions.

Usage:
    python -m tinyctx.stats                       # all logs
    python -m tinyctx.stats --since 2026-05-01    # date range
    python -m tinyctx.stats --json                # machine-readable

Reads files matching `~/.tinyctx/logs/tinyctx-*.jsonl` (each line is one
event emitted by `proxy._log`). Aggregates:

  - total request count
  - split by route (local / frontier)
  - split by reason (compaction redirect / size threshold / streak / forced)
  - estimated input tokens routed to each backend
  - upstream + stream errors
  - average elapsed time per route

Output is two lines per dimension max; designed to fit a terminal at a
glance, not to be a dashboard.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _iter_events(log_dir: Path, since: str | None = None) -> Iterable[dict[str, Any]]:
    if not log_dir.is_dir():
        return
    files = sorted(log_dir.glob("tinyctx-*.jsonl"))
    for f in files:
        if since:
            stem = f.stem.replace("tinyctx-", "")
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
                target = datetime.fromisoformat(since).date()
                if d < target:
                    continue
            except ValueError:
                pass
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue


def aggregate(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    route_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    tokens_by_route: defaultdict[str, int] = defaultdict(int)
    elapsed_by_route: defaultdict[str, list[float]] = defaultdict(list)
    bytes_by_route: defaultdict[str, int] = defaultdict(int)
    compaction_redirected = 0
    upstream_errors = 0
    stream_errors = 0

    for ev in events:
        et = ev.get("event")
        if et == "route" or et == "route_chat":
            total += 1
            route = ev.get("decision") or "unknown"
            route_counts[route] += 1
            reason = ev.get("reason") or "?"
            reason_counts[reason] += 1
            tokens_by_route[route] += int(ev.get("est_tokens") or 0)
            if ev.get("is_compaction"):
                compaction_redirected += 1
        elif et == "stream_done":
            r = ev.get("route") or "unknown"
            elapsed_by_route[r].append(float(ev.get("elapsed_s") or 0.0))
            bytes_by_route[r] += int(ev.get("bytes") or 0)
        elif et == "upstream_error":
            upstream_errors += 1
        elif et == "stream_error":
            stream_errors += 1

    avg_elapsed = {r: (sum(v) / max(1, len(v))) for r, v in elapsed_by_route.items()}
    return {
        "total_requests": total,
        "by_route": dict(route_counts),
        "by_reason": dict(reason_counts.most_common(10)),
        "est_input_tokens_routed": dict(tokens_by_route),
        "stream_bytes_returned": dict(bytes_by_route),
        "avg_stream_seconds": avg_elapsed,
        "compaction_redirects": compaction_redirected,
        "upstream_errors": upstream_errors,
        "stream_errors": stream_errors,
    }


def render(stats: dict[str, Any]) -> str:
    if stats["total_requests"] == 0:
        return "no requests in log range"
    lines = []
    total = stats["total_requests"]
    lines.append(f"requests: {total}")
    by_route = stats["by_route"]
    parts = [f"{r}={n} ({100 * n / total:.1f}%)" for r, n in by_route.items()]
    lines.append("  route mix:    " + ", ".join(parts))

    tokens = stats["est_input_tokens_routed"]
    parts = [f"{r}={tok}" for r, tok in tokens.items()]
    lines.append("  est tokens:   " + ", ".join(parts) if parts else "  est tokens:  0")

    avg = stats["avg_stream_seconds"]
    parts = [f"{r}={s:.2f}s" for r, s in avg.items()]
    if parts:
        lines.append("  avg latency:  " + ", ".join(parts))

    lines.append(f"  compaction redirects: {stats['compaction_redirects']}")
    if stats["upstream_errors"] or stats["stream_errors"]:
        lines.append(
            f"  errors: upstream={stats['upstream_errors']} "
            f"stream={stats['stream_errors']}"
        )
    if stats["by_reason"]:
        lines.append("  top reasons:")
        for reason, n in list(stats["by_reason"].items())[:5]:
            lines.append(f"    {n:>5}  {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.stats")
    p.add_argument("--log-dir", default=str(Path.home() / ".tinyctx" / "logs"))
    p.add_argument("--since", default=None, help="YYYY-MM-DD, inclusive")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    events = _iter_events(Path(args.log_dir), since=args.since)
    stats = aggregate(events)
    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
