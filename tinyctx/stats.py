"""tinyctx-stats — read JSONL logs and summarize proxy behavior.

Usage:
    python -m tinyctx.stats                       # all logs, default report
    python -m tinyctx.stats --since 2026-05-01    # date range
    python -m tinyctx.stats --json                # machine-readable
    python -m tinyctx.stats --quality             # quality score report (S-F)

Reads files matching `~/.tinyctx/logs/tinyctx-*.jsonl` (each line is one
event emitted by `proxy._log` or `RequestTrace.emit`).

Default report aggregates:
  - total request count
  - split by route (local / frontier)
  - split by reason (compaction redirect / size threshold / streak / forced)
  - estimated input tokens routed to each backend
  - upstream + stream errors
  - average elapsed time per route

Quality report (--quality) consumes `request_trace` events and grades
proxy efficiency on six dimensions, then maps to S/A/B/C/D/F. The
dimensions and weights are inspired by token-optimizer's quality score
but reframed for what tinyctx (a wire-level proxy) can actually
observe and control.
"""
from __future__ import annotations

import argparse
import json
import os
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
                # Why: non-date stem (e.g. "tinyctx-archive.jsonl"). Skip
                # the date filter and read the file like any other.
                pass
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Why: malformed JSONL line (partial write); skip.
                    continue
        except OSError:
            # Why: file unreadable (locked, rotated); skip and move
            # on to the next file in the list.
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


###############################################################################
#  Quality report (--quality)
###############################################################################
#
# Six dimensions, each scored 0–100; weighted sum mapped to a letter grade.
# Inspired by alexgreensh/token-optimizer's quality score (PolyForm
# Noncommercial — idea borrowed, not code), reframed for what tinyctx can
# observe at the wire layer.
#
# Inputs:
#   - `request_trace` events (per-request, emitted by RequestTrace.emit)
#
# Dimensions:
#
#   1. Routing efficiency (25%)
#        % of requests served by the local backend. Higher is better — the
#        whole point of tinyctx is to keep the cheap path hot. 100% local =
#        100; 0% local = 0; linear in between. Compaction redirects count
#        as local (they're the highest-leverage savings).
#
#   2. Compaction discipline (20%)
#        How often we hit proactive_compact (history truncated mid-flight)
#        and codex's own compaction handoff. LOW usage is good — it means
#        history was being kept slim by other means (mutations + read_delta)
#        before reaching the limit. High usage = poor discipline.
#        score = 100 × (1 − share_compact)
#
#   3. Token compression (20%)
#        avg(forwarded_tokens / est_input_tokens). Less than 1.0 means our
#        sanitize/scrub/inject pipeline is shedding bytes before they
#        reach the upstream. Score = 100 × (1 − ratio), clamped to [0, 100].
#        Skipped on requests where forwarded_tokens_est == 0 (older logs).
#
#   4. Read-delta savings (15%)
#        Of the candidate read-tool results we saw, what fraction did we
#        successfully delta-collapse? High = we caught waste; low = either
#        no waste was there OR our heuristics didn't recognize it. Skipped
#        when there were no candidates at all (no signal).
#
#   5. Tool-trim savings (10%)
#        Frontier requests where we trimmed the tool list. Reflects how
#        well frontier_trim_tools is paying off. Score = 100 × applied_share.
#
#   6. Reliability (10%)
#        1 − error_rate. error = `status >= 400` or `status == 0` (network
#        failure). Hits the hardest below 95% — sustained errors mean
#        upstream config trouble.
#
# Each dimension contributes (weight × score). Total weight = 100.
# Final letter grade:
#   S ≥ 90, A ≥ 80, B ≥ 70, C ≥ 60, D ≥ 50, F < 50

_QUALITY_WEIGHTS = {
    "routing_efficiency":  0.25,
    "compaction_discipline": 0.20,
    "token_compression":   0.20,
    "read_delta_savings":  0.15,
    "tool_trim_savings":   0.10,
    "reliability":         0.10,
}


def _grade(score: float) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def aggregate_quality(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Read JSONL events, return a quality-report dict ready for render.

    Only `request_trace` events are consumed. Older logs without these
    events return `total_traces=0` and the renderer says "no traces".
    """
    total = 0
    local_count = 0
    frontier_count = 0
    compaction_count = 0          # codex's own compaction redirected
    proactive_compact_count = 0   # tinyctx proactive_compact applied
    error_count = 0

    # token compression: sum (forwarded / est) across requests with both > 0
    compression_sum = 0.0
    compression_n = 0

    # read_delta candidates / replacements (across all traces)
    rd_candidates = 0
    rd_replacements = 0
    rd_bytes_saved = 0

    # frontier trim
    frontier_seen = 0
    frontier_trim_applied = 0

    for ev in events:
        if ev.get("event") != "request_trace":
            continue
        total += 1
        route = ev.get("route") or ""
        if route == "local":
            local_count += 1
        elif route == "frontier":
            frontier_count += 1
            frontier_seen += 1
            if ev.get("tools_trimmed_applied"):
                frontier_trim_applied += 1
        if ev.get("is_compaction"):
            compaction_count += 1
        if ev.get("proactive_compact_applied"):
            proactive_compact_count += 1
        status = int(ev.get("status") or 0)
        if status == 0 or status >= 400:
            error_count += 1
        est = int(ev.get("est_input_tokens") or 0)
        fwd = int(ev.get("forwarded_tokens_est") or 0)
        if est > 0 and fwd > 0:
            compression_sum += fwd / est
            compression_n += 1
        rd_candidates += int(ev.get("read_delta_candidates") or 0)
        rd_replacements += int(ev.get("read_delta_replacements") or 0)
        rd_bytes_saved += int(ev.get("read_delta_bytes_saved") or 0)

    if total == 0:
        return {"total_traces": 0}

    routing_eff = 100.0 * (local_count / total)
    share_compact = (compaction_count + proactive_compact_count) / total
    compaction_disc = max(0.0, 100.0 * (1 - share_compact))

    if compression_n > 0:
        avg_ratio = compression_sum / compression_n
        token_compression = max(0.0, min(100.0, 100.0 * (1.0 - avg_ratio)))
    else:
        avg_ratio = None
        token_compression = None  # no signal

    if rd_candidates > 0:
        read_delta_score = 100.0 * (rd_replacements / rd_candidates)
    else:
        read_delta_score = None  # no signal — don't penalize

    if frontier_seen > 0:
        tool_trim_score = 100.0 * (frontier_trim_applied / frontier_seen)
    else:
        tool_trim_score = None  # no frontier traffic in window

    reliability = 100.0 * (1 - (error_count / total))

    # Compute weighted score, redistributing weight from any dimension
    # that was skipped (None) so users with light/specialized usage
    # still get a fair grade.
    components = {
        "routing_efficiency": routing_eff,
        "compaction_discipline": compaction_disc,
        "token_compression": token_compression,
        "read_delta_savings": read_delta_score,
        "tool_trim_savings": tool_trim_score,
        "reliability": reliability,
    }
    active = {k: v for k, v in components.items() if v is not None}
    if not active:
        weighted = 0.0
    else:
        active_weight_sum = sum(_QUALITY_WEIGHTS[k] for k in active)
        weighted = sum(_QUALITY_WEIGHTS[k] * v for k, v in active.items()) \
            / active_weight_sum

    return {
        "total_traces": total,
        "local_share": local_count / total,
        "frontier_share": frontier_count / total,
        "error_rate": error_count / total,
        "compaction_share": share_compact,
        "avg_token_compression_ratio": avg_ratio,
        "read_delta_candidates": rd_candidates,
        "read_delta_replacements": rd_replacements,
        "read_delta_bytes_saved": rd_bytes_saved,
        "frontier_trim_applied_share": (
            frontier_trim_applied / frontier_seen if frontier_seen else None
        ),
        "components": components,
        "weights": _QUALITY_WEIGHTS,
        "score": round(weighted, 1),
        "grade": _grade(weighted),
    }


def render_quality(q: dict[str, Any]) -> str:
    if q.get("total_traces", 0) == 0:
        return ("no request_trace events found — make sure traces are being "
                "emitted (RequestTrace.emit) and the log dir is correct.")
    lines: list[str] = []
    lines.append(f"quality grade: {q['grade']} ({q['score']:.1f}/100)  "
                 f"over {q['total_traces']} traces")
    lines.append("")
    lines.append("dimension                    score   weight  contrib")
    fmt = "  {name:<26}  {score:>5}   {w:>4}%   {contrib:>5}"
    contributions: list[tuple[str, float]] = []
    for name, weight in _QUALITY_WEIGHTS.items():
        v = q["components"].get(name)
        if v is None:
            score_str = "  —  "
            contrib_str = "skip"
        else:
            score_str = f"{v:>4.1f}"
            contrib_str = f"{weight * v:>4.1f}"
            contributions.append((name, weight * v))
        lines.append(fmt.format(
            name=name.replace("_", " "),
            score=score_str,
            w=int(weight * 100),
            contrib=contrib_str,
        ))
    lines.append("")
    lines.append("highlights:")
    lines.append(f"  routing:       local={q['local_share']:.1%}  "
                 f"frontier={q['frontier_share']:.1%}")
    lines.append(f"  errors:        {q['error_rate']:.1%}")
    lines.append(f"  compaction:    {q['compaction_share']:.1%}")
    if q.get("avg_token_compression_ratio") is not None:
        lines.append(f"  token ratio:   "
                     f"forwarded/input = {q['avg_token_compression_ratio']:.2f}")
    if q.get("read_delta_candidates"):
        lines.append(
            f"  read-delta:    {q['read_delta_replacements']}/"
            f"{q['read_delta_candidates']} candidates rewritten, "
            f"{q['read_delta_bytes_saved']} bytes saved"
        )
    trim = q.get("frontier_trim_applied_share")
    if trim is not None:
        lines.append(f"  tool-trim:     applied on {trim:.1%} of frontier reqs")
    return "\n".join(lines)


###############################################################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.stats")
    # Honor TINYCTX_LOG_DIR (the proxy writes there via load_config) so the CLI
    # reads the same logs the proxy produced; --log-dir still overrides.
    p.add_argument("--log-dir",
                   default=os.environ.get("TINYCTX_LOG_DIR")
                   or str(Path.home() / ".tinyctx" / "logs"))
    p.add_argument("--since", default=None, help="YYYY-MM-DD, inclusive")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quality", action="store_true",
                   help="produce a quality-grade report (S/A/B/C/D/F) from "
                        "request_trace events instead of the default routing "
                        "summary")
    args = p.parse_args(argv)

    events = list(_iter_events(Path(args.log_dir), since=args.since))
    if args.quality:
        q = aggregate_quality(events)
        if args.json:
            print(json.dumps(q, indent=2, sort_keys=True, default=str))
        else:
            print(render_quality(q))
        return 0

    stats = aggregate(events)
    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
