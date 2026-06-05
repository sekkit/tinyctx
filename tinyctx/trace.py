"""Per-request RequestTrace + CLI viewer.

The proxy already emits granular `route` / `mutation_gate` / `stream_done`
events to JSONL — fine for `tinyctx-stats` aggregation but you can't see
"this one request" as a coherent story. This module adds a single
`request_trace` event per request that joins everything: routing decision,
sanitize/scrub/inject diffs, mutation-gate verdict, forward target, response
status / bytes / latency, translator activity.

CLI:
    tinyctx-trace                       # last 10 requests, compact table
    tinyctx-trace -v                    # verbose, multi-line per request
    tinyctx-trace --last 50             # last 50
    tinyctx-trace --watch               # tail-follow
    tinyctx-trace --request <rid>       # one specific request, full detail
    tinyctx-trace --session <sid>       # filter by session id
    tinyctx-trace --since 2026-05-06    # date filter
    tinyctx-trace --json                # raw JSONL passthrough
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4


@dataclass
class RequestTrace:
    """Consolidated per-request record. Every field is optional / safe-default
    so partial population works for any code path (compaction handler,
    stream errors, etc.)."""
    request_id: str = field(default_factory=lambda: "rq_" + uuid4().hex[:20])
    session_id: str = ""
    # Composite key (cwd_hash + session_id) used for per-conversation
    # state so multi-project users don't suffer cross-project pollution
    # in proactive_compact cache, error_streak escalation, and
    # mutation-gate timing. See proxy._project_session_key.
    project_session_key: str = ""
    cwd: str = ""  # from x-codex-cwd header, for per-project stats
    started_at: float = field(default_factory=time.time)

    # routing decision
    route: str = ""                       # "local" | "frontier"
    route_reason: str = ""
    is_compaction: bool = False
    est_input_tokens: int = 0
    turn_count: int = 0
    error_streak: int = 0
    requested_model: str = ""
    forced_by_client_model: bool = False
    adaptive_model_calls: int = 0
    adaptive_model_failures: int = 0
    adaptive_model_failure_rate: float = 0.0
    adaptive_model_triggered: bool = False

    # sanitize / scrub / inject — counts and diffs
    encrypted_content_stripped: int = 0
    tools_before: int = 0
    tools_after: int = 0
    tool_types_dropped: list[str] = field(default_factory=list)
    fields_stripped: list[str] = field(default_factory=list)
    fields_injected: list[str] = field(default_factory=list)
    # Fields whose inbound value exceeded a configured cap and were
    # lowered before forwarding (e.g. max_output_tokens 128000 → 16000
    # to prevent DeepSeek runaway thinking loops).
    fields_capped: list[str] = field(default_factory=list)

    # mutation gate (cache-aware deferral for dedup/purge/historian-substitute)
    mutation_wanted: bool = False
    mutation_fired: bool = False
    mutation_gate_reason: str = ""
    deduped_calls: int = 0
    purged_inputs: int = 0
    historian_substituted: bool = False

    # forward target
    target_url: str = ""
    target_wire_api: str = ""
    target_model: str = ""

    # response
    status: int = 0
    is_stream: bool = True
    bytes_out: int = 0
    translated: bool = False
    translated_calls: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    prompt_cache_hit_ratio: float = 0.0

    # timing
    elapsed_s: float = 0.0

    # the special path (compactor debate ran instead of forward)
    compactor_used: bool = False
    compactor_outcome: str = ""

    # proactive_compact (proxy-side history truncation when est_tokens
    # exceeds CFG.proactive_compact_threshold, defending against codex
    # 0.128's "ran out of room" with auto-compact disabled at the model
    # level). Fires before forward; codex's local history is unchanged.
    proactive_compact_applied: bool = False
    proactive_compact_reason: str = ""
    proactive_compact_items_before: int = 0
    proactive_compact_items_after: int = 0
    proactive_compact_middle_compacted: int = 0
    # Synthetic function_call stubs we inserted into tail to repair
    # orphan function_call_outputs left after middle-truncation. >0 here
    # means the slice boundary cut a real (call, output) pair and we
    # repaired it; 0 means tail was already structurally consistent.
    proactive_compact_synthetic_calls: int = 0
    # Effective threshold actually used for the gate decision. When 0,
    # the gate was skipped entirely. When > 0 and not equal to the
    # config's static `proactive_compact_threshold`, that's the dynamic
    # value derived from frontier.context_window × safe_fraction.
    proactive_compact_threshold_used: int = 0

    # What we forwarded UPSTREAM (after every transform). Use these to
    # calculate "how many tokens the user actually paid for" vs. what
    # codex.app sent in (est_input_tokens). The diff measures tinyctx's
    # win.
    forwarded_bytes: int = 0          # serialized JSON length of forward_body
    forwarded_tokens_est: int = 0     # cheap char/3.6 estimate
    forwarded_breakdown: dict = field(default_factory=dict)
    # forwarded_breakdown keys (all int token estimates):
    #   instructions, tools, input, other

    # Frontier-only optimizations
    advisor_hint_skipped: bool = False
    tools_trimmed_applied: bool = False
    tools_trimmed_before: int = 0
    tools_trimmed_after: int = 0
    tools_trimmed_dropped: list = field(default_factory=list)

    # Auto-scout: project-context summary auto-injected into instructions.
    # `scout_injected` is True when the cached scout.md was read AND
    # prepended to body.instructions on this request. `scout_chars` is
    # the cached file's size for cost accounting.
    scout_injected: bool = False
    scout_chars: int = 0
    ctx_pack_injected: bool = False
    ctx_pack_chars: int = 0
    snapshot_injected: bool = False
    snapshot_chars: int = 0

    # Global agent rules (tinyctx/templates/AGENTS.md) injection. True
    # when the proxy prepended the bundled rules block this request.
    # False when codex.app's own AGENTS.md load already had it in the
    # instructions (idempotent skip), the feature is disabled, or the
    # template couldn't be loaded.
    global_agent_rules_injected: bool = False
    global_agent_rules_chars: int = 0

    # Platform-specific tool/shell rules (tinyctx/platform_rules.py)
    # injection. True when the proxy prepended the host's block this
    # request; False when the host is unrecognised, the block was
    # already present (hop/replay/user customization), or the feature
    # is disabled via CFG.inject_platform_rules.
    platform_rules_injected: bool = False
    platform_rules_chars: int = 0

    # Model-driven self-classification (Anthropic Advisor Strategy
    # aligned): when the proxy asked the local model "should this turn
    # escalate?", these fields capture its answer. p=0.0 (default)
    # means classifier didn't run or returned None (skipped — body
    # wasn't a fresh user query, classifier unreachable, etc.).
    # `self_classify_overrode` flips the route from local→frontier
    # based on the classifier's verdict.
    self_classify_p: float = 0.0
    self_classify_reason: str = ""
    self_classify_cached: bool = False
    self_classify_overrode: bool = False
    self_consistency_agreed: bool | None = None
    self_consistency_reason: str = ""
    self_consistency_samples: int = 0

    # Stuck-loop watchdog. True on the turn we injected the reminder.
    # `stuck_turn_count_at_inject` records the turn_count at injection
    # so we can correlate effectiveness in post-hoc trace analysis (did
    # the next 50 turns produce a different tool-call signature, or did
    # the model just keep looping?). See stuck_loop.py.
    stuck_reminder_injected: bool = False
    stuck_turn_count_at_inject: int = 0

    # Soft-completion gate. `*_detected` flips True when the streaming
    # response from upstream matches a "soft punt to user" pattern (per
    # soft_completion.py). `*_gate_injected` flips True on the *next*
    # request to the same session, when we inject the advisor-vet
    # reminder into body.input. Each session can show one detection
    # per stream and one injection per stream consumed.
    soft_completion_detected: bool = False
    soft_completion_pattern: str = ""
    soft_completion_gate_injected: bool = False
    soft_completion_gate_pattern: str = ""

    # Number of SSE keepalive `: tinyctx keepalive\n\n` comment lines
    # emitted during this stream while waiting for upstream chunks. >0
    # means the upstream went silent for at least
    # `stream_keepalive_interval_s` seconds at least once. Each
    # keepalive is ~24 bytes — negligible cost, prevents idle timeouts
    # on slow model thinking turns.
    keepalives_emitted: int = 0

    # Repeat-Read delta collapse (read_delta.collapse_repeated_reads).
    # Fires alongside dedup/purge under the cache-aware gate. When it
    # applied, reads ≥2 of the same path were rewritten as unified
    # diffs vs. their first occurrence in body.input.
    read_delta_applied: bool = False
    read_delta_candidates: int = 0
    read_delta_replacements: int = 0
    read_delta_bytes_saved: int = 0
    read_delta_paths: list = field(default_factory=list)

    # LLMLingua-2 pre-escalation compression (frontier-only, opt-in).
    # Compresses bulky tool-result payloads before forwarding to frontier
    # while preserving cache-critical fields (instructions/tools/messages).
    lingua_applied: bool = False
    lingua_items_compressed: int = 0
    lingua_chars_before: int = 0
    lingua_chars_after: int = 0

    # Headroom content-type-aware tool output compression. Applies
    # SmartCrusher (JSON arrays), CodeCompressor, SearchCompressor,
    # LogCompressor to function_call_output / tool_result items.
    # Silent no-op when headroom-ai is not installed.
    headroom_compress_applied: bool = False
    headroom_compress_chars_before: int = 0
    headroom_compress_chars_after: int = 0

    # Forge-inspired guardrail observability. `guard_results` records the
    # existing preflight GuardPipeline wrappers. `guardrail_decisions`
    # records protocol-neutral policy decisions such as failure-signal
    # escalation; final Responses/SSE wire emission remains elsewhere.
    guard_results: list = field(default_factory=list)
    guardrail_decisions: list = field(default_factory=list)
    failure_signal_score: int = 0
    failure_signals: list = field(default_factory=list)

    # Task Orchestrator / Symphony-inspired supervisor. Records the
    # current-turn Skill/MCP guidance tinyctx injected before routing.
    orchestrator_injected: bool = False
    orchestrator_task_type: str = ""
    orchestrator_confidence: float = 0.0
    orchestrator_skills: list[str] = field(default_factory=list)
    orchestrator_mcp: list[str] = field(default_factory=list)
    orchestrator_dynamic_skill_hash: str = ""
    orchestrator_rationale: str = ""
    orchestrator_execution_mode: str = "serial"
    orchestrator_execution_reason: str = ""
    orchestrator_parallel_subtasks: list[dict[str, str]] = field(default_factory=list)
    task_id: str = ""
    task_title: str = ""
    task_state: str = ""

    def emit(self, log_dir: Path) -> None:
        """Write one `request_trace` JSONL event to today's log file."""
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"
        d = asdict(self)
        d["t"] = self.started_at
        d["event"] = "request_trace"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(d, default=str, ensure_ascii=False) + "\n")
        except OSError:
            # Why: trace emission is best-effort telemetry; a log-write
            # failure (disk full, fs read-only) must not fail the
            # request that's emitting the trace.
            pass

        # Record outcome for graduated escalation ladder.
        # ok=True only for clean 200 with substantive response and no
        # soft-completion punt. Everything else counts as a failure.
        if self.session_id:
            try:
                from . import escalation
                ok = (
                    self.status == 200
                    and self.bytes_out > 0
                    and not self.soft_completion_detected
                )
                escalation.record_outcome(
                    self.project_session_key or self.session_id, ok=ok)
            except Exception:
                pass  # escalation is best-effort advisory


# ───────────────────────────── reader ─────────────────────────────


def iter_traces(log_dir: Path, *, since: str | None = None
                ) -> Iterator[RequestTrace]:
    if not log_dir.is_dir():
        return
    files = sorted(log_dir.glob("tinyctx-*.jsonl"))
    if since:
        try:
            target = datetime.fromisoformat(since).date()
            files = [f for f in files
                     if _date_from_stem(f.stem) >= target]
        except ValueError:
            # Why: caller passed an unparseable `since` value — fall
            # through to "no filter" so traces are still returned.
            pass
    for f in files:
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line or '"request_trace"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # Why: malformed JSONL line; skip.
                    continue
                d.pop("event", None)
                d.pop("t", None)
                # keep only fields RequestTrace knows about
                allowed = set(RequestTrace.__dataclass_fields__.keys())
                kw = {k: v for k, v in d.items() if k in allowed}
                yield RequestTrace(**kw)
        except OSError:
            # Why: log file rotated or unreadable; skip to next file.
            continue


def _date_from_stem(stem: str):
    try:
        return datetime.strptime(stem.replace("tinyctx-", ""), "%Y%m%d").date()
    except ValueError:
        return datetime.min.date()


# ──────────────────────────── rendering ────────────────────────────


def render_compact_row(t: RequestTrace) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(t.started_at))
    route = t.route or "?"
    reason = (t.route_reason or "")[:34]
    if t.compactor_used:
        reason = f"COMPACT/{t.compactor_outcome}"[:34]
    transforms_bits: list[str] = []
    if t.tools_before != t.tools_after:
        transforms_bits.append(f"tools {t.tools_before}→{t.tools_after}")
    if t.fields_stripped:
        transforms_bits.append(f"-{len(t.fields_stripped)}f")
    if t.fields_injected:
        transforms_bits.append(f"+{len(t.fields_injected)}f")
    if t.mutation_fired:
        bits = []
        if t.deduped_calls:
            bits.append(f"dedup{t.deduped_calls}")
        if t.purged_inputs:
            bits.append(f"purge{t.purged_inputs}")
        if t.historian_substituted:
            bits.append("hist-sub")
        if bits:
            transforms_bits.append("/".join(bits))
    elif t.mutation_wanted:
        transforms_bits.append("mut-defer")
    if t.proactive_compact_applied:
        transforms_bits.append(
            f"pcompact-{t.proactive_compact_middle_compacted}")
    transforms = " ".join(transforms_bits) or "—"
    trans = f"x{t.translated_calls}" if t.translated_calls else ("✓" if t.translated else "—")
    return (
        f"{ts}  {route:<8}  {reason:<34}  "
        f"tok={t.est_input_tokens:>6}  "
        f"{transforms:<22}  "
        f"{t.elapsed_s:>5.1f}s  trans={trans}"
    )


def render_verbose(t: RequestTrace) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t.started_at))
    lines = [
        "─" * 86,
        f"[{t.request_id} · sess={t.session_id} · {ts}]",
        f"  ROUTE        {t.route:<10}  reason: {t.route_reason}",
        f"  TOKENS       est={t.est_input_tokens}  turns={t.turn_count}  "
        f"error_streak={t.error_streak}",
        f"  ADAPTIVE     calls={t.adaptive_model_calls}  "
        f"failures={t.adaptive_model_failures}  "
        f"failure_rate={t.adaptive_model_failure_rate:.0%}  "
        f"triggered={t.adaptive_model_triggered}",
        f"  COMPACTION   {'yes' if t.is_compaction else 'no'}",
        f"  REQUESTED    model={t.requested_model!r}  "
        f"forced_by_client={t.forced_by_client_model}",
        "",
        "  TRANSFORMS",
        f"    encrypted_content_stripped:  {t.encrypted_content_stripped}",
        f"    tools:                       {t.tools_before} → {t.tools_after}"
        + (f"   dropped: {', '.join(t.tool_types_dropped)}"
           if t.tool_types_dropped else ""),
        f"    fields stripped:             {', '.join(t.fields_stripped) or '—'}",
        f"    fields injected:             {', '.join(t.fields_injected) or '—'}",
        f"    mutation gate:               wanted={t.mutation_wanted}  "
        f"fired={t.mutation_fired}",
    ]
    if t.mutation_gate_reason:
        lines.append(f"        reason:                  {t.mutation_gate_reason}")
    if t.mutation_fired:
        lines.append(
            f"    dedup={t.deduped_calls}  purge={t.purged_inputs}  "
            f"historian_subst={t.historian_substituted}"
        )
    lines.extend([
        "",
        f"  FORWARD      url={t.target_url}",
        f"               wire={t.target_wire_api}  model={t.target_model}",
        "",
        f"  RESPONSE     status={t.status}  bytes={t.bytes_out}  "
        f"is_stream={t.is_stream}",
        f"               translated={t.translated}  "
        f"translated_calls={t.translated_calls}",
        f"  ELAPSED      {t.elapsed_s:.2f}s",
    ])
    if t.compactor_used:
        lines.append(f"  COMPACTOR    used  outcome={t.compactor_outcome}")
    if t.proactive_compact_applied or t.proactive_compact_reason:
        lines.append(
            f"  PROACTIVE_C  applied={t.proactive_compact_applied}  "
            f"reason={t.proactive_compact_reason}"
        )
        if t.proactive_compact_applied:
            lines.append(
                f"               items {t.proactive_compact_items_before}"
                f" → {t.proactive_compact_items_after} "
                f"(middle compacted: {t.proactive_compact_middle_compacted})"
            )
    return "\n".join(lines)


# ───────────────────────────── filtering ─────────────────────────────


def filter_traces(traces: Iterable[RequestTrace], *, request_id: str | None = None,
                  session_id: str | None = None
                  ) -> Iterator[RequestTrace]:
    for t in traces:
        if request_id and t.request_id != request_id:
            continue
        if session_id and t.session_id != session_id:
            continue
        yield t


# ───────────────────────────── watch mode ─────────────────────────────


def watch(log_dir: Path, *, verbose: bool = False) -> None:
    """tail-follow today's JSONL and print each new request_trace event."""
    path = log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"
    if not path.exists():
        # Wait for it to exist
        sys.stderr.write(f"(waiting for {path} to appear...)\n")
        while not path.exists():
            time.sleep(0.5)
    pos = path.stat().st_size
    while True:
        new_size = path.stat().st_size
        if new_size > pos:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                for line in fh:
                    if '"request_trace"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        # Why: incomplete line at the tail of live JSONL
                        # — skip; the next poll iteration will see it.
                        continue
                    allowed = set(RequestTrace.__dataclass_fields__.keys())
                    kw = {k: v for k, v in d.items() if k in allowed}
                    t = RequestTrace(**kw)
                    if verbose:
                        print(render_verbose(t))
                    else:
                        print(render_compact_row(t))
            pos = new_size
        elif new_size < pos:
            # Log was rotated/truncated; re-seek to start.
            pos = 0
        time.sleep(0.5)


# ───────────────────────────── CLI entry ─────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.trace")
    # Honor TINYCTX_LOG_DIR (where the proxy writes) so the viewer reads the
    # same logs the proxy produced; --log-dir still overrides.
    p.add_argument("--log-dir",
                   default=os.environ.get("TINYCTX_LOG_DIR")
                   or str(Path.home() / ".tinyctx" / "logs"))
    p.add_argument("--last", type=int, default=10,
                   help="show the last N traces (default: 10)")
    p.add_argument("--all", action="store_true", help="show all traces")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="multi-line per-request rendering")
    p.add_argument("--watch", action="store_true",
                   help="follow today's log; print each new trace as it lands")
    p.add_argument("--request", default=None,
                   help="filter to one request_id (works with --verbose)")
    p.add_argument("--session", default=None,
                   help="filter to one session_id")
    p.add_argument("--since", default=None,
                   help="YYYY-MM-DD; only files dated >= since")
    p.add_argument("--json", action="store_true",
                   help="raw JSONL passthrough (matching filters)")
    args = p.parse_args(argv)
    log_dir = Path(args.log_dir)

    if args.watch:
        watch(log_dir, verbose=args.verbose)
        return 0

    traces = list(filter_traces(
        iter_traces(log_dir, since=args.since),
        request_id=args.request,
        session_id=args.session,
    ))
    if not traces:
        sys.stderr.write("(no matching request_trace events)\n")
        return 1

    if not args.all:
        traces = traces[-args.last:]

    if args.json:
        for t in traces:
            d = asdict(t)
            d["event"] = "request_trace"
            print(json.dumps(d, default=str, ensure_ascii=False))
        return 0

    if args.verbose or args.request:
        for t in traces:
            print(render_verbose(t))
        return 0

    # default: compact table
    print(f"{'TIME':<8}  {'ROUTE':<8}  {'REASON':<34}  "
          f"{'TOK':>9}  {'TRANSFORMS':<22}  {'TIME':>5}     {'TRANS'}")
    for t in traces:
        print(render_compact_row(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
