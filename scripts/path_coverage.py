"""Path-coverage report for tinyctx.

Reads ~/.tinyctx/logs/tinyctx-*.jsonl, counts how many times each known
code path was exercised in production, and highlights paths that have
never fired (= not validated in real traffic).

Usage:
    python scripts/path_coverage.py [--since YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


# Each path entry: (label, predicate(event_dict) -> bool, source_module)
# Predicates run on a single JSONL event, return True iff the path fired.
PATHS: list[tuple[str, str, callable, str]] = [
    # ── routing ────────────────────────────────────────────────
    ("router.decide → local",
     "request_trace",
     lambda e: e.get("route") == "local",
     "router.py"),
    ("router.decide → frontier",
     "request_trace",
     lambda e: e.get("route") == "frontier",
     "router.py"),
    ("router.is_compaction_request hit",
     "request_trace",
     lambda e: bool(e.get("is_compaction")),
     "router.py"),
    ("client-forced model (tinyctx-local/-frontier)",
     "request_trace",
     lambda e: bool(e.get("forced_by_client_model")),
     "proxy.py:240"),
    ("error_streak escalation",
     "request_trace",
     lambda e: int(e.get("error_streak") or 0) > 0
               and "error_streak" in (e.get("route_reason") or ""),
     "router.py:129"),

    # ── sanitize ───────────────────────────────────────────────
    ("encrypted_content scrub",
     "request_trace",
     lambda e: int(e.get("encrypted_content_stripped") or 0) > 0,
     "sanitize.strip_encrypted_content"),
    ("tools scrubbed (type filter)",
     "request_trace",
     lambda e: bool(e.get("tool_types_dropped")),
     "sanitize.scrub_unsupported_tools"),
    ("MCP namespace dropped by scrub (= expand did NOT cover)",
     "request_trace",
     lambda e: "namespace" in (e.get("tool_types_dropped") or []),
     "sanitize.expand_mcp_namespaces (negative signal)"),
    ("fields stripped",
     "request_trace",
     lambda e: bool(e.get("fields_stripped")),
     "sanitize.strip_unsupported_responses_fields"),
    ("fields injected (defaults)",
     "request_trace",
     lambda e: bool(e.get("fields_injected")),
     "sanitize.inject_responses_defaults"),
    ("fields capped (max_output_tokens)",
     "request_trace",
     lambda e: bool(e.get("fields_capped")),
     "sanitize.cap_responses_fields"),
    ("advisor hint skipped on frontier",
     "request_trace",
     lambda e: bool(e.get("advisor_hint_skipped")),
     "proxy.py:396"),

    # ── mutation gate ──────────────────────────────────────────
    ("CacheAwareMutator wanted",
     "request_trace",
     lambda e: bool(e.get("mutation_wanted")),
     "sanitize.CacheAwareMutator"),
    ("CacheAwareMutator fired",
     "request_trace",
     lambda e: bool(e.get("mutation_fired")),
     "sanitize.CacheAwareMutator"),
    ("dedup_tool_calls applied",
     "request_trace",
     lambda e: int(e.get("deduped_calls") or 0) > 0,
     "sanitize.dedup_tool_calls"),
    ("purge_failed_tool_inputs applied",
     "request_trace",
     lambda e: int(e.get("purged_inputs") or 0) > 0,
     "sanitize.purge_failed_tool_inputs"),
    ("historian.apply_to_body substituted",
     "request_trace",
     lambda e: bool(e.get("historian_substituted")),
     "historian.apply_to_body"),
    ("read_delta candidates seen",
     "request_trace",
     lambda e: int(e.get("read_delta_candidates") or 0) > 0,
     "read_delta.collapse_repeated_reads"),
    ("read_delta replacements made",
     "request_trace",
     lambda e: int(e.get("read_delta_replacements") or 0) > 0,
     "read_delta.collapse_repeated_reads"),

    # ── proactive_compact ──────────────────────────────────────
    ("proactive_compact threshold gate evaluated",
     "request_trace",
     lambda e: int(e.get("proactive_compact_threshold_used") or 0) > 0,
     "sanitize.proactive_compact"),
    ("proactive_compact applied",
     "request_trace",
     lambda e: bool(e.get("proactive_compact_applied")),
     "sanitize.proactive_compact"),
    ("proactive_compact synthetic call stub",
     "request_trace",
     lambda e: int(e.get("proactive_compact_synthetic_calls") or 0) > 0,
     "sanitize.proactive_compact"),

    # ── tools_trim ─────────────────────────────────────────────
    ("trim_tools_for_frontier applied",
     "request_trace",
     lambda e: bool(e.get("tools_trimmed_applied")),
     "sanitize.trim_tools_for_frontier"),

    # ── compactor ──────────────────────────────────────────────
    ("compactor (3-role debate) used",
     "request_trace",
     lambda e: bool(e.get("compactor_used")),
     "compactor.compact_with_debate"),
    ("compactor outcome=judged",
     "request_trace",
     lambda e: e.get("compactor_outcome") == "judged",
     "compactor.compact_with_debate"),
    ("compactor outcome=fallback/error",
     "request_trace",
     lambda e: e.get("compactor_outcome") in (
         "judge_failed_concat", "single_draft_polished",
         "single_draft_raw", "all_failed", "failed_fallback",
     ),
     "compactor.compact_with_debate"),

    # ── translation ────────────────────────────────────────────
    ("tool_call XML translation",
     "request_trace",
     lambda e: bool(e.get("translated"))
               and int(e.get("translated_calls") or 0) > 0,
     "tool_call_translator"),
    ("chat→responses translation (no calls)",
     "request_trace",
     lambda e: bool(e.get("translated"))
               and int(e.get("translated_calls") or 0) == 0
               and (e.get("target_wire_api") or "") != "responses",
     "tool_call_translator.ChatToResponsesTranslator"),

    # ── streaming ──────────────────────────────────────────────
    ("streaming response",
     "request_trace",
     lambda e: bool(e.get("is_stream")),
     "proxy._stream_proxy"),
    ("non-streaming response",
     "request_trace",
     lambda e: not e.get("is_stream") and int(e.get("status") or 0) > 0,
     "proxy._forward"),
    ("upstream HTTP error (status>=400)",
     "request_trace",
     lambda e: int(e.get("status") or 0) >= 400,
     "proxy._forward"),
    ("upstream connection failure (status=0 trace)",
     "request_trace",
     lambda e: e.get("status") == 0 and bool(e.get("started_at"))
               and not e.get("compactor_used"),
     "proxy._stream_proxy / _forward"),

    # ── event-level (not request_trace) ────────────────────────
    ("route event emitted",
     "route",
     lambda e: True,
     "proxy._log"),
    ("mutation_gate event emitted",
     "mutation_gate",
     lambda e: True,
     "proxy._log"),
    ("read_delta event emitted",
     "read_delta",
     lambda e: True,
     "proxy._log"),
    ("compactor_done event",
     "compactor_done",
     lambda e: True,
     "proxy._log"),
    ("compactor_saved event",
     "compactor_saved",
     lambda e: True,
     "proxy._log"),
    ("cap_fields event",
     "cap_fields",
     lambda e: True,
     "proxy._log"),
    ("frontier_trim_tools event",
     "frontier_trim_tools",
     lambda e: True,
     "proxy._log"),
    ("proactive_compact event",
     "proactive_compact",
     lambda e: True,
     "proxy._log"),
    ("upstream_error event",
     "upstream_error",
     lambda e: True,
     "proxy._log"),
    ("stream_error event",
     "stream_error",
     lambda e: True,
     "proxy._log"),
    ("stream_done event",
     "stream_done",
     lambda e: True,
     "proxy._log"),
]


def iter_events(log_dir: Path, *, since: str | None = None):
    if not log_dir.is_dir():
        return
    files = sorted(log_dir.glob("tinyctx-*.jsonl"))
    if since:
        try:
            target = datetime.fromisoformat(since).date()
            kept = []
            for f in files:
                try:
                    d = datetime.strptime(
                        f.stem.replace("tinyctx-", ""), "%Y%m%d").date()
                    if d >= target:
                        kept.append(f)
                except ValueError:
                    kept.append(f)
            files = kept
        except ValueError:
            pass
    for f in files:
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir",
                   default=str(Path.home() / ".tinyctx" / "logs"))
    p.add_argument("--since", default=None,
                   help="YYYY-MM-DD inclusive")
    args = p.parse_args(argv)

    log_dir = Path(args.log_dir)
    counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    total_traces = 0

    for ev in iter_events(log_dir, since=args.since):
        et = ev.get("event")
        if et:
            event_counts[et] += 1
        if et == "request_trace":
            total_traces += 1
        for label, expect_event, predicate, _src in PATHS:
            if expect_event != et:
                continue
            try:
                if predicate(ev):
                    counts[label] += 1
            except Exception:  # noqa: BLE001
                continue

    print(f"=== tinyctx path coverage ===")
    print(f"log dir:        {log_dir}")
    print(f"since:          {args.since or '(all)'}")
    print(f"request_traces: {total_traces}")
    print(f"event mix:      "
          + ", ".join(f"{k}={v}" for k, v in event_counts.most_common(8)))
    print()
    print(f"{'path':<54}  {'hits':>8}  source")
    print("-" * 96)

    fired: list[tuple[str, int, str]] = []
    cold: list[tuple[str, str]] = []
    for label, _et, _pred, src in PATHS:
        n = counts[label]
        if n > 0:
            fired.append((label, n, src))
        else:
            cold.append((label, src))

    fired.sort(key=lambda r: -r[1])
    for label, n, src in fired:
        print(f"{label:<54}  {n:>8}  {src}")

    print()
    print(f"=== never fired ({len(cold)}/{len(PATHS)}) ===")
    for label, src in cold:
        print(f"  {label:<52}  {src}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
