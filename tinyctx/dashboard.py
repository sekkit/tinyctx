"""tinyctx web dashboard.

Single-page real-time view at `http://127.0.0.1:4141/dashboard`. Surfaces
everything the Monitor in chat surfaced, plus aggregates and per-session
state, in a browser instead of stderr.

Endpoints (mounted by proxy.py):
  GET /dashboard               — HTML page
  GET /dashboard/stream        — Server-Sent Events: tail of today's JSONL,
                                 filtered to interesting events
  GET /dashboard/state         — JSON snapshot of in-memory per-session state
  GET /dashboard/aggregates    — JSON rollup of last N minutes (configurable
                                 via `?since_s=900`)

Zero new dependencies — uses FastAPI/Starlette which the proxy already
imports. The HTML page is a single string with inline CSS + vanilla JS;
no build step.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config_io import (
    effective_config,
    env_overrides,
    merge_sections_into_toml,
    read_config_text,
    save_config_text,
)
from .config_schema import config_presets, config_schema, validate_sections


_START_TS = time.time()


def _today_log(log_dir: Path) -> Path:
    return log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"


# ─── interestingness filter ────────────────────────────────────────────────


_INTERESTING_EVENTS = {
    "request_trace",
    "stuck_reminder_injected",
    "soft_completion_classified",
    "soft_completion_gate_injected",
    "soft_completion_classify_skipped",
    "soft_completion_classify_backend_error",
    "soft_completion_classify_parse_failed",
    "stream_error",
    "upstream_error",
    "self_classify_error",
    "stuck_loop_error",
    "exec_resume_poke",
    "exec_resume_poke_error",
}


def _format_event_for_dashboard(e: dict[str, Any]) -> dict[str, Any] | None:
    """Project a JSONL event into the compact shape the dashboard renders.
    Returns None to filter out (uninteresting events, test traffic)."""
    ev = e.get("event", "")
    if ev not in _INTERESTING_EVENTS:
        return None

    base = {"t": e.get("t", 0), "event": ev}

    if ev == "request_trace":
        if e.get("forced_by_client_model") and e.get("requested_model") == "tinyctx-local":
            return None  # pytest traffic
        base.update({
            "kind": ("advisor" if (e.get("forced_by_client_model")
                                    and e.get("requested_model") == "tinyctx-frontier")
                     else "main"),
            "turn_count": e.get("turn_count", 0),
            "elapsed_s": e.get("elapsed_s", 0),
            "bytes_in": e.get("forwarded_bytes", 0),
            "bytes_out": e.get("bytes_out", 0),
            "status": e.get("status", 0),
            "route": e.get("route", ""),
            "route_reason": (e.get("route_reason", "") or "")[:80],
            "keepalives": e.get("keepalives_emitted", 0),
            "error_streak": e.get("error_streak", 0),
            "stuck": bool(e.get("stuck_reminder_injected")),
            "soft_punt_gate": bool(e.get("soft_completion_gate_injected")),
            "self_classify_p": e.get("self_classify_p", 0.0),
            "self_classify_reason": (e.get("self_classify_reason", "") or "")[:80],
            "self_classify_overrode": bool(e.get("self_classify_overrode")),
            "orchestrator_injected": bool(e.get("orchestrator_injected")),
            "orchestrator_task_type": (e.get("orchestrator_task_type", "") or "")[:40],
            "orchestrator_confidence": float(e.get("orchestrator_confidence", 0.0) or 0.0),
            "orchestrator_skills": list((e.get("orchestrator_skills") or []))[:3],
            "orchestrator_mcp": list((e.get("orchestrator_mcp") or []))[:3],
            "task_state": (e.get("task_state", "") or "")[:20],
            "session_id": (e.get("session_id", "") or "")[:24],
        })
    elif ev == "soft_completion_classified":
        base.update({
            "soft_punt": bool(e.get("soft_punt")),
            "p": e.get("p", 0.0),
            "reason": (e.get("reason", "") or "")[:80],
        })
    elif ev == "soft_completion_classify_skipped":
        base.update({
            "reason": e.get("reason", ""),
            "finish_reason": e.get("finish_reason", ""),
            "extracted_text_chars": e.get("extracted_text_chars", 0),
            "raw_buffer_chars": e.get("raw_buffer_chars", 0),
        })
    elif ev == "soft_completion_gate_injected":
        base.update({"pattern": (e.get("pattern", "") or "")[:80]})
    elif ev == "stuck_reminder_injected":
        base.update({"turn_count": e.get("turn_count", 0),
                     "proj_sid": (e.get("proj_sid", "") or "")[:24]})
    elif ev in ("soft_completion_classify_backend_error",
                "soft_completion_classify_parse_failed",
                "stream_error", "upstream_error",
                "self_classify_error", "stuck_loop_error",
                "exec_resume_poke_error"):
        base.update({
            "error": (e.get("error", "") or e.get("body", "") or "")[:200],
            "status": e.get("status", 0),
        })
    elif ev == "exec_resume_poke":
        base.update({
            "status_label": (e.get("status", "") or "")[:20],
            "reason": (e.get("reason", "") or "")[:80],
            "pid": e.get("pid", 0),
            "p": e.get("p", 0.0),
            "resolved_session_id": (e.get("resolved_session_id", "") or "")[:24],
        })
    return base


# ─── SSE event stream ──────────────────────────────────────────────────────


async def stream_events(log_dir: Path,
                         poll_interval_s: float = 0.5
                         ) -> AsyncIterator[bytes]:
    """Tail today's JSONL forever, yielding SSE-formatted events for new
    interesting lines. Handles midnight log rotation by re-resolving the
    `today` filename each iteration."""
    pos = 0
    current_path = _today_log(log_dir)
    if current_path.exists():
        pos = current_path.stat().st_size

    yield b": tinyctx dashboard ready\n\n"
    last_keepalive = time.time()
    while True:
        await asyncio.sleep(poll_interval_s)
        # midnight rollover: re-resolve path and reset pos if new
        new_path = _today_log(log_dir)
        if new_path != current_path:
            current_path = new_path
            pos = 0
        if not current_path.exists():
            # send keepalive comment so EventSource doesn't time out
            if time.time() - last_keepalive > 15:
                yield b": tinyctx-tick\n\n"
                last_keepalive = time.time()
            continue
        try:
            size = current_path.stat().st_size
        except OSError:
            # Why: file may have been rotated/unlinked between checks
            # in this tail loop. Skip this poll and retry next tick.
            continue
        if size < pos:
            # log rotated / truncated
            pos = 0
        if size == pos:
            if time.time() - last_keepalive > 15:
                yield b": tinyctx-tick\n\n"
                last_keepalive = time.time()
            continue
        try:
            with current_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            # Why: log file may have been rotated mid-read. Skip this
            # poll; the watcher loop retries with the rotated file
            # next tick (size-shrink check above handles the rewind).
            continue
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                # Why: JSONL line corrupted (partial write during rotate
                # or upstream truncation). Skip the line; the dashboard
                # is best-effort visibility, not audit-grade.
                continue
            formatted = _format_event_for_dashboard(evt)
            if formatted is None:
                continue
            payload = json.dumps(formatted, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            last_keepalive = time.time()


# ─── state snapshot ────────────────────────────────────────────────────────


def state_snapshot() -> dict[str, Any]:
    """Read all per-session state from in-memory module dicts. Safe for
    concurrent access (Python dict.copy() is atomic; values are tuples
    or simple types)."""
    out: dict[str, Any] = {
        "uptime_s": round(time.time() - _START_TS, 1),
        "proxy_pid": os.getpid(),
        "now_ts": time.time(),
    }
    # stuck_loop state
    try:
        from . import stuck_loop
        out["stuck_loop"] = {
            "last_reminder_turn": dict(stuck_loop._LAST_REMINDER_TURN),
            "last_advisor_ts": {k: round(v, 1)
                                 for k, v in stuck_loop._LAST_ADVISOR_TS.items()},
        }
    except Exception as e:  # noqa: BLE001
        out["stuck_loop"] = {"error": str(e)}
    # soft_completion state
    try:
        from . import soft_completion
        flags = {}
        for sid, f in soft_completion._SOFT_COMPLETION_FLAG.items():
            if f and f.get("active"):
                flags[sid] = {k: v for k, v in f.items() if k != "active"}
        out["soft_completion"] = {
            "active_flags": flags,
            "buffer_sizes": {sid: len(b) for sid, b
                              in soft_completion._OUTPUT_BUFFER.items()},
        }
    except Exception as e:  # noqa: BLE001
        out["soft_completion"] = {"error": str(e)}
    # session error streaks (from proxy module)
    try:
        from . import proxy as _proxy
        out["session_error_streaks"] = {sid: v for sid, v
                                          in _proxy._SESSION_ERROR_STREAK.items()
                                          if v > 0}
    except Exception as e:  # noqa: BLE001
        out["session_error_streaks"] = {"error": str(e)}
    # self_classify cache size
    try:
        from . import self_classify
        out["self_classify_cache_entries"] = len(self_classify._CACHE)
    except Exception as e:  # noqa: BLE001
        out["self_classify_cache_entries"] = -1
    # empty_response_guard pending flags
    try:
        from . import empty_response_guard
        out["empty_response_guard_flags"] = (
            empty_response_guard.state_snapshot())
    except Exception as e:  # noqa: BLE001
        out["empty_response_guard_flags"] = {"error": str(e)}
    # Per-session request lifecycle phase (P3)
    try:
        from . import request_phase as _rp
        snap = _rp.state_snapshot()
        now = time.time()
        out["request_phase"] = {
            sid: {**info,
                  "age_s": round(now - info.get("since_ts", now), 2)}
            for sid, info in snap.items()
        }
    except Exception as e:  # noqa: BLE001
        out["request_phase"] = {"error": str(e)}
    # forensics dump count
    try:
        from pathlib import Path as _P
        # Best-effort: dashboard doesn't have CFG so derive from default
        import os as _os
        forensics_dir = _P(_os.path.expanduser("~/.tinyctx/forensics"))
        if forensics_dir.exists():
            out["forensics_dumps_count"] = len(list(forensics_dir.glob("*.json")))
        else:
            out["forensics_dumps_count"] = 0
    except Exception:  # noqa: BLE001
        out["forensics_dumps_count"] = -1
    # C-4 exec_resume poke summary
    try:
        from . import exec_resume as _xr
        out["exec_resume"] = _xr.state_snapshot()
    except Exception as e:  # noqa: BLE001
        out["exec_resume"] = {"error": str(e)}
    # Tool-call frequency by namespace (live)
    try:
        from . import tool_metrics as _tm
        snap = _tm.snapshot()
        # Compact form for state endpoint — full detail at
        # /dashboard/tool-metrics
        out["tool_metrics"] = {
            "total_calls": snap["total_calls"],
            "distinct_tools": snap["distinct_tools"],
            "by_namespace": snap["by_namespace"],
        }
    except Exception as e:  # noqa: BLE001
        out["tool_metrics"] = {"error": str(e)}
    return out


# ─── rolling aggregates ────────────────────────────────────────────────────


_AGG_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_AGG_TTL_S = 5.0


def reset_aggregates_cache() -> None:
    """Test/dev helper to clear the rolling-aggregates cache."""
    _AGG_CACHE.clear()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def aggregates(log_dir: Path, since_s: int = 900) -> dict[str, Any]:
    """Roll up the last `since_s` seconds of today's JSONL into stats.
    Cached `_AGG_TTL_S` to avoid re-parsing the file on every dashboard
    poll."""
    now = time.time()
    # Cache key includes log_dir so concurrent test runs / multiple log
    # roots don't share state. Also includes the today-file's mtime so
    # the cache invalidates as new lines arrive faster than the TTL.
    path_for_key = _today_log(log_dir)
    try:
        mtime = path_for_key.stat().st_mtime if path_for_key.exists() else 0.0
    except OSError:
        mtime = 0.0
    cache_key = f"{log_dir}:{since_s}:{mtime}"
    cached = _AGG_CACHE.get(cache_key)
    if cached and now - cached[0] < _AGG_TTL_S:
        return cached[1]

    cutoff = now - since_s
    path = _today_log(log_dir)
    if not path.exists():
        result = {"since_s": since_s, "turns_real": 0, "advisor_calls": 0,
                  "by_route": {}, "by_status": {},
                  "stuck_reminders": 0, "soft_punt_classified": 0,
                  "soft_punt_gates": 0, "anomalies": 0,
                  "keepalive_saves": 0,
                  "p50_elapsed_s": 0, "p99_elapsed_s": 0,
                  "median_bytes_in": 0, "median_bytes_out": 0}
        _AGG_CACHE[cache_key] = (now, result)
        return result

    by_route: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    elapsed: list[float] = []
    bytes_in: list[int] = []
    bytes_out: list[int] = []
    turns_real = 0
    turns_advisor = 0
    stuck_n = 0
    sc_classified = 0
    sc_gates = 0
    anomalies = 0
    keepalive_saves = 0

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    # Why: partial-write line in JSONL — skip it. The
                    # aggregator is statistical, single skipped lines
                    # are noise-floor.
                    continue
                if e.get("t", 0) < cutoff:
                    continue
                ev = e.get("event", "")
                if ev == "stuck_reminder_injected":
                    stuck_n += 1
                elif ev == "soft_completion_classified":
                    sc_classified += 1
                elif ev == "soft_completion_gate_injected":
                    sc_gates += 1
                elif ev == "request_trace":
                    if e.get("forced_by_client_model"):
                        if e.get("requested_model") == "tinyctx-frontier":
                            turns_advisor += 1
                        else:
                            continue  # test
                    else:
                        turns_real += 1
                        by_route[e.get("route", "?")] += 1
                        by_status[str(e.get("status", 0))] += 1
                        elapsed.append(float(e.get("elapsed_s", 0) or 0))
                        bytes_in.append(int(e.get("forwarded_bytes", 0) or 0))
                        bytes_out.append(int(e.get("bytes_out", 0) or 0))
                        if (e.get("status", 0) not in (200,)
                                or e.get("error_streak", 0) > 0):
                            anomalies += 1
                        if e.get("keepalives_emitted", 0) > 0:
                            keepalive_saves += 1
    except OSError:
        # Why: log file unreadable or rotated mid-scan. Return whatever
        # we aggregated so far; dashboard shows partial counts rather
        # than 500-erroring the endpoint.
        pass

    result = {
        "since_s": since_s,
        "turns_real": turns_real,
        "advisor_calls": turns_advisor,
        "by_route": dict(by_route),
        "by_status": dict(by_status),
        "stuck_reminders": stuck_n,
        "soft_punt_classified": sc_classified,
        "soft_punt_gates": sc_gates,
        "anomalies": anomalies,
        "keepalive_saves": keepalive_saves,
        "p50_elapsed_s": round(_percentile(elapsed, 50), 2),
        "p99_elapsed_s": round(_percentile(elapsed, 99), 2),
        "median_bytes_in": int(statistics.median(bytes_in)) if bytes_in else 0,
        "median_bytes_out": int(statistics.median(bytes_out)) if bytes_out else 0,
        "turns_per_min": round(turns_real / max(since_s / 60.0, 1.0), 2),
    }
    _AGG_CACHE[cache_key] = (now, result)
    return result


# ─── recent events list (for initial page load) ────────────────────────────


def recent_events(log_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    """Return the last `limit` interesting events from today's JSONL —
    used to populate the live feed on initial page load before SSE kicks in."""
    path = _today_log(log_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        # cheap tail: read last 256KB only, parse line-by-line backward
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, size - 262144))
            chunk = f.read()
    except OSError:
        # Why: today's log was rotated/unlinked between exists() and
        # open. Return empty so the dashboard renders without crashing.
        return []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            # Why: partial-write line at the tail of a live JSONL.
            # Skip the malformed line and keep parsing the rest.
            continue
        formatted = _format_event_for_dashboard(e)
        if formatted is not None:
            out.append(formatted)
    return out[-limit:]


# ─── HTML page ─────────────────────────────────────────────────────────────


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinyctx dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif; font-size: 13px; line-height: 1.45; color: #1f2937; background: #f5f6f8; }
  h1 { font-size: 16px; margin: 0 0 12px 0; font-weight: 600; }
  h2 { font-size: 13px; margin: 0 0 8px 0; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  .card.full { grid-column: 1 / -1; }
  .stat-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 10px; }
  .stat { background: #f9fafb; padding: 6px 10px; border-radius: 6px; }
  .stat-label { font-size: 10px; text-transform: uppercase; color: #6b7280; letter-spacing: 0.5px; }
  .stat-value { font-size: 16px; font-weight: 600; color: #111827; font-variant-numeric: tabular-nums; }
  .feed { max-height: 360px; overflow-y: auto; font-family: "SF Mono", "Menlo", monospace; font-size: 11.5px; line-height: 1.5; }
  .feed-row { padding: 3px 6px; border-radius: 4px; margin-bottom: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .feed-row .age { color: #9ca3af; margin-right: 6px; font-variant-numeric: tabular-nums; display: inline-block; min-width: 50px; }
  .feed-row .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-right: 6px; vertical-align: 1px; }
  .b-anomaly { background: #fee2e2; color: #991b1b; }
  .b-advisor { background: #dbeafe; color: #1e40af; }
  .b-stuck   { background: #fef3c7; color: #92400e; }
  .b-gate    { background: #ede9fe; color: #5b21b6; }
  .b-punt    { background: #fce7f3; color: #9d174d; }
  .b-okp     { background: #d1fae5; color: #065f46; }
  .b-route   { background: #e0f2fe; color: #075985; }
  .b-skip    { background: #f3f4f6; color: #6b7280; }
  .b-error   { background: #fef2f2; color: #b91c1c; }
  table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #f3f4f6; font-variant-numeric: tabular-nums; }
  th { color: #6b7280; font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  td.num { text-align: right; }
  pre { background: #f3f4f6; padding: 8px; border-radius: 4px; font-size: 10.5px; overflow-x: auto; max-height: 240px; }
  .footer { margin-top: 20px; text-align: center; color: #9ca3af; font-size: 10.5px; }
  .conn { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
  .conn-on { background: #10b981; box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2); }
  .conn-off { background: #ef4444; }
  details { margin-top: 8px; }
  details summary { cursor: pointer; color: #6b7280; font-size: 11px; }
</style>
</head>
<body>
  <h1>tinyctx dashboard <span id="conn-indicator"><span class="conn conn-off"></span><span id="conn-text">connecting…</span></span></h1>

  <div class="grid">
    <div class="card full">
      <h2>aggregates · last 15 min</h2>
      <div class="stat-row" id="agg-stats">…</div>
      <details><summary>raw aggregate JSON</summary><pre id="agg-raw">…</pre></details>
    </div>

    <div class="card">
      <h2>live feed</h2>
      <div class="feed" id="feed"></div>
    </div>

    <div class="card">
      <h2>per-session state</h2>
      <table id="state-table">
        <thead><tr><th>session</th><th class="num">last reminder turn</th><th class="num">advisor age</th><th>soft-punt flag</th><th class="num">err streak</th></tr></thead>
        <tbody></tbody>
      </table>
      <details><summary>raw state JSON</summary><pre id="state-raw">…</pre></details>
    </div>
  </div>

  <div class="footer">
    proxy uptime <span id="uptime">…</span> · pid <span id="pid">…</span> · self_classify cache <span id="cache">…</span> entries · refresh state every 3s
  </div>

<script>
(function () {
  const feed = document.getElementById("feed");
  const connIndicator = document.getElementById("conn-indicator");
  const FEED_MAX_ROWS = 80;

  function addFeed(evt) {
    const row = document.createElement("div");
    row.className = "feed-row";
    const age = formatAge(Date.now() / 1000 - (evt.t || 0));
    row.innerHTML = `<span class="age">${age}</span>${renderEvent(evt)}`;
    feed.insertBefore(row, feed.firstChild);
    while (feed.children.length > FEED_MAX_ROWS) {
      feed.removeChild(feed.lastChild);
    }
  }

  function formatAge(s) {
    if (s < 0) s = 0;
    if (s < 60) return `t-${s.toFixed(0)}s`;
    if (s < 3600) return `t-${(s / 60).toFixed(0)}m`;
    return `t-${(s / 3600).toFixed(1)}h`;
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
  }

  function renderEvent(e) {
    const ev = e.event;
    if (ev === "request_trace") {
      const isAdvisor = e.kind === "advisor";
      const isAnomaly = (e.status !== 200 && e.status !== 0)
        || (e.status === 0 && e.bytes_out === 0)
        || e.error_streak > 0
        || e.keepalives > 0;
      let badge = "", info = "";
      if (isAdvisor) {
        badge = `<span class="badge b-advisor">ADVISOR</span>`;
        info = `${e.elapsed_s.toFixed(1)}s in=${kb(e.bytes_in)} out=${kb(e.bytes_out)} st=${e.status}`;
      } else if (isAnomaly) {
        let kind = e.status === 0 ? "DISCONNECT" : `st=${e.status}`;
        if (e.keepalives > 0) kind += ` keep=${e.keepalives}`;
        if (e.error_streak > 0) kind += ` err=${e.error_streak}`;
        badge = `<span class="badge b-anomaly">${kind}</span>`;
        info = `route=${e.route} ${e.elapsed_s.toFixed(1)}s out=${kb(e.bytes_out)}`;
      } else if (e.stuck) {
        badge = `<span class="badge b-stuck">STUCK</span>`;
        info = `turn=${e.turn_count} ${e.elapsed_s.toFixed(1)}s out=${kb(e.bytes_out)}`;
      } else if (e.soft_punt_gate) {
        badge = `<span class="badge b-gate">GATE</span>`;
        info = `${e.elapsed_s.toFixed(1)}s out=${kb(e.bytes_out)}`;
      } else if (e.self_classify_reason) {
        const route = e.self_classify_overrode ? "ESC→FRONTIER" : "no-esc";
        badge = `<span class="badge b-route">${route}</span>`;
        info = `p=${e.self_classify_p.toFixed(2)} ${escapeHTML(e.self_classify_reason)}`;
      } else {
        // routine turn
        badge = `<span class="badge b-skip">turn</span>`;
        info = `t=${e.turn_count} ${e.elapsed_s.toFixed(1)}s ${kb(e.bytes_in)}→${kb(e.bytes_out)} ${e.route}`;
      }
      const orch = e.orchestrator_injected
        ? ` · orch=${escapeHTML(e.orchestrator_task_type || "unknown")}(${Number(e.orchestrator_confidence || 0).toFixed(2)})`
          + (Array.isArray(e.orchestrator_skills) && e.orchestrator_skills.length
            ? ` skills=${escapeHTML(e.orchestrator_skills.join(","))}` : "")
          + (e.task_state ? ` state=${escapeHTML(e.task_state)}` : "")
        : "";
      return badge + info + orch;
    }
    if (ev === "soft_completion_classified") {
      const cls = e.soft_punt && e.p >= 0.7 ? "b-punt" : "b-okp";
      const verdict = e.soft_punt ? "PUNT" : "OK";
      return `<span class="badge ${cls}">${verdict} p=${e.p.toFixed(2)}</span>${escapeHTML(e.reason)}`;
    }
    if (ev === "soft_completion_gate_injected") {
      return `<span class="badge b-gate">GATE FIRED</span>${escapeHTML(e.pattern || "")}`;
    }
    if (ev === "soft_completion_classify_skipped") {
      return `<span class="badge b-skip">skip ${escapeHTML(e.reason || "?")}</span>finish=${escapeHTML(e.finish_reason || "?")} text=${e.extracted_text_chars} raw=${e.raw_buffer_chars}`;
    }
    if (ev === "soft_completion_classify_backend_error" || ev === "soft_completion_classify_parse_failed"
        || ev === "stream_error" || ev === "upstream_error" || ev === "self_classify_error" || ev === "stuck_loop_error") {
      return `<span class="badge b-error">${escapeHTML(ev)}</span>${escapeHTML(e.error || "")}`;
    }
    if (ev === "stuck_reminder_injected") {
      return `<span class="badge b-stuck">STUCK INJECT</span>turn=${e.turn_count}`;
    }
    return `<span class="badge b-skip">${escapeHTML(ev)}</span>`;
  }

  function kb(n) {
    if (!n) return "0";
    if (n < 1024) return n + "B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + "K";
    return (n / 1024 / 1024).toFixed(1) + "M";
  }

  // initial fetch of recent events
  fetch("/dashboard/recent").then(r => r.json()).then(events => {
    events.forEach(addFeed);
  }).catch(() => {});

  // SSE subscribe
  function connect() {
    const es = new EventSource("/dashboard/stream");
    es.onopen = () => {
      connIndicator.innerHTML = '<span class="conn conn-on"></span><span id="conn-text">live</span>';
    };
    es.onerror = () => {
      connIndicator.innerHTML = '<span class="conn conn-off"></span><span id="conn-text">reconnecting…</span>';
      setTimeout(connect, 2000);
      es.close();
    };
    es.onmessage = (e) => {
      try { addFeed(JSON.parse(e.data)); } catch {}
    };
  }
  connect();

  // poll state + aggregates
  function pollState() {
    fetch("/dashboard/state").then(r => r.json()).then(s => {
      document.getElementById("uptime").textContent = formatUptime(s.uptime_s);
      document.getElementById("pid").textContent = s.proxy_pid;
      document.getElementById("cache").textContent = s.self_classify_cache_entries ?? "?";
      document.getElementById("state-raw").textContent = JSON.stringify(s, null, 2);

      const tbody = document.querySelector("#state-table tbody");
      tbody.innerHTML = "";
      const sids = new Set([
        ...Object.keys(s.stuck_loop?.last_reminder_turn || {}),
        ...Object.keys(s.stuck_loop?.last_advisor_ts || {}),
        ...Object.keys(s.soft_completion?.active_flags || {}),
        ...Object.keys(s.session_error_streaks || {}),
      ]);
      const rows = [];
      sids.forEach(sid => {
        const lastReminder = s.stuck_loop?.last_reminder_turn?.[sid] || 0;
        const advisorTs = s.stuck_loop?.last_advisor_ts?.[sid];
        const advisorAge = advisorTs ? formatAge(s.now_ts - advisorTs) : "—";
        const flag = s.soft_completion?.active_flags?.[sid];
        const errStreak = s.session_error_streaks?.[sid] || 0;
        rows.push({sid, lastReminder, advisorAge, flag, errStreak});
      });
      rows.sort((a, b) => b.lastReminder - a.lastReminder);
      rows.slice(0, 12).forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHTML(r.sid)}</td><td class="num">${r.lastReminder}</td><td class="num">${r.advisorAge}</td><td>${r.flag ? `<span class="badge b-punt">flag</span> ${escapeHTML(r.flag.matched_pattern || "")}` : "—"}</td><td class="num">${r.errStreak}</td>`;
        tbody.appendChild(tr);
      });
    }).catch(() => {});
  }

  function pollAgg() {
    fetch("/dashboard/aggregates?since_s=900").then(r => r.json()).then(a => {
      const html = [
        ["turns",        a.turns_real],
        ["advisor",      a.advisor_calls],
        ["stuck",        a.stuck_reminders],
        ["soft-punt",    `${a.soft_punt_classified} (${a.soft_punt_gates} gates)`],
        ["anomalies",    a.anomalies],
        ["keepalive saves", a.keepalive_saves],
        ["p50 elapsed",  `${a.p50_elapsed_s}s`],
        ["p99 elapsed",  `${a.p99_elapsed_s}s`],
        ["med bytes-in", kb(a.median_bytes_in)],
        ["med bytes-out", kb(a.median_bytes_out)],
        ["turns/min",    a.turns_per_min],
      ].map(([k, v]) => `<div class="stat"><div class="stat-label">${k}</div><div class="stat-value">${escapeHTML(String(v))}</div></div>`).join("");
      document.getElementById("agg-stats").innerHTML = html;
      document.getElementById("agg-raw").textContent = JSON.stringify(a, null, 2);
    }).catch(() => {});
  }

  function formatUptime(s) {
    if (s < 60) return `${s.toFixed(0)}s`;
    if (s < 3600) return `${(s / 60).toFixed(0)}m`;
    return `${(s / 3600).toFixed(1)}h`;
  }

  pollState(); pollAgg();
  setInterval(pollState, 3000);
  setInterval(pollAgg, 5000);
})();
</script>
</body>
</html>
"""


_CONFIG_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tinyctx Config Center</title>
<style>
:root { color-scheme: light dark; --bg:#0f172a; --panel:#111827; --muted:#94a3b8; --line:#334155; --text:#e5e7eb; --accent:#38bdf8; --ok:#22c55e; --bad:#ef4444; }
body { margin:0; font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Arial; background:linear-gradient(135deg,#020617,#111827); color:var(--text); }
main { max-width:1180px; margin:0 auto; padding:28px; }
h1 { margin:0 0 4px; font-size:28px; }
p { color:var(--muted); }
.grid { display:grid; grid-template-columns:260px 1fr 320px; gap:16px; align-items:start; }
.card { background:rgba(15,23,42,.88); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 20px 60px rgba(0,0,0,.25); }
.preset { width:100%; text-align:left; margin:8px 0; padding:10px; border:1px solid var(--line); border-radius:12px; background:#0b1220; color:var(--text); cursor:pointer; }
.preset:hover { border-color:var(--accent); }
fieldset { border:1px solid var(--line); border-radius:14px; margin:0 0 14px; padding:12px; }
legend { color:#bae6fd; padding:0 8px; }
label { display:grid; grid-template-columns:190px 1fr; gap:10px; align-items:center; margin:8px 0; }
input, select, textarea { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:10px; padding:8px 10px; background:#020617; color:var(--text); }
textarea { min-height:70px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
.row { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
button { border:0; border-radius:999px; padding:9px 14px; background:var(--accent); color:#00111f; font-weight:700; cursor:pointer; }
button.secondary { background:#1e293b; color:var(--text); border:1px solid var(--line); }
pre { white-space:pre-wrap; overflow:auto; max-height:340px; background:#020617; border:1px solid var(--line); border-radius:12px; padding:12px; }
.ok { color:var(--ok); } .bad { color:var(--bad); } .muted { color:var(--muted); }
</style>
</head>
<body>
<main>
  <h1>tinyctx Config Center</h1>
  <p>Visual editor for <code>~/.tinyctx/config.toml</code>. Presets get you close; validation and test calls catch the sharp edges.</p>
  <div class="grid">
    <section class="card">
      <h2>Presets</h2>
      <div id="presets"></div>
    </section>
    <section class="card">
      <h2>Core Config</h2>
      <form id="config-form"></form>
      <div class="row">
        <button type="button" id="validate">Validate</button>
        <button type="button" id="save">Save</button>
        <button type="button" id="test-local" class="secondary">Test Local</button>
        <button type="button" id="test-frontier" class="secondary">Test Frontier</button>
      </div>
    </section>
    <aside class="card">
      <h2>Status</h2>
      <p id="path" class="muted"></p>
      <h3>Result</h3>
      <pre id="result">Loading…</pre>
      <h3>Env Overrides</h3>
      <pre id="env"></pre>
    </aside>
  </div>
</main>
<script>
const core = {
  server: ["host", "port", "verbose"],
  routing: ["force_route", "redirect_compaction_to_local", "sanitize_encrypted_content", "self_classify_escalates_to_frontier", "escalate_input_tokens", "escalate_turn_count", "escalate_on_error_streak"],
  local: ["base_url", "wire_api", "model", "context_window", "timeout_s", "strip_tools", "api_key_env", "headers"],
  frontier: ["base_url", "wire_api", "model", "timeout_s", "api_key_env"]
};
let state = {};
let schema = {};
let presets = {};

function $(id) { return document.getElementById(id); }
function show(obj) { $("result").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2); }
function fieldId(section, key) { return `${section}__${key}`; }

async function load() {
  const res = await fetch("/api/v1/config");
  state = await res.json();
  schema = state.schema.sections;
  presets = state.presets;
  $("path").textContent = state.path;
  $("env").textContent = JSON.stringify(state.env_overrides, null, 2);
  renderPresets();
  renderForm(state.effective);
  show({loaded: true, needs_restart_after_save: true});
}

function renderPresets() {
  $("presets").innerHTML = "";
  Object.entries(presets).forEach(([name, preset]) => {
    const btn = document.createElement("button");
    btn.className = "preset";
    btn.type = "button";
    btn.innerHTML = `<strong>${preset.label}</strong><br><span class="muted">${preset.description}</span>`;
    btn.onclick = () => {
      applySections(preset.sections);
      show({preset: name, applied: preset.sections});
    };
    $("presets").appendChild(btn);
  });
}

function renderForm(effective) {
  const form = $("config-form");
  form.innerHTML = "";
  for (const [section, keys] of Object.entries(core)) {
    const fs = document.createElement("fieldset");
    fs.innerHTML = `<legend>[${section}]</legend>`;
    keys.forEach(key => {
      const value = (effective[section] || {})[key];
      const meta = (schema[section] || []).find(f => f.name === key) || {};
      const row = document.createElement("label");
      row.innerHTML = `<span>${key}</span>`;
      let input;
      if (meta.type === "boolean") {
        input = document.createElement("select");
        input.innerHTML = '<option value=""></option><option value="true">true</option><option value="false">false</option>';
        input.value = value === true ? "true" : value === false ? "false" : "";
      } else if (meta.type === "enum") {
        input = document.createElement("select");
        input.innerHTML = '<option value=""></option>' + (meta.options || []).map(v => `<option value="${v}">${v}</option>`).join("");
        input.value = value ?? "";
      } else if (meta.type === "object") {
        input = document.createElement("textarea");
        input.value = value && Object.keys(value).length ? JSON.stringify(value, null, 2) : "";
      } else {
        input = document.createElement("input");
        input.value = value ?? "";
      }
      input.id = fieldId(section, key);
      input.dataset.type = meta.type || "string";
      row.appendChild(input);
      fs.appendChild(row);
    });
    form.appendChild(fs);
  }
}

function collectSections() {
  const sections = {};
  for (const [section, keys] of Object.entries(core)) {
    sections[section] = {};
    keys.forEach(key => {
      const input = $(fieldId(section, key));
      if (!input) return;
      const raw = input.value.trim();
      if (raw === "") return;
      if (input.dataset.type === "boolean") sections[section][key] = raw === "true";
      else if (input.dataset.type === "integer") sections[section][key] = Number.parseInt(raw, 10);
      else if (input.dataset.type === "object") {
        try { sections[section][key] = JSON.parse(raw); }
        catch (e) { sections[section][key] = raw; }
      } else sections[section][key] = raw;
    });
    if (!Object.keys(sections[section]).length) delete sections[section];
  }
  return sections;
}

function applySections(sections) {
  for (const [section, values] of Object.entries(sections || {})) {
    for (const [key, value] of Object.entries(values || {})) {
      const input = $(fieldId(section, key));
      if (!input) continue;
      input.value = typeof value === "object" && value !== null ? JSON.stringify(value, null, 2) : String(value);
    }
  }
}

async function post(url) {
  const res = await fetch(url, {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({sections: collectSections()})});
  const data = await res.json();
  show(data);
}
$("validate").onclick = () => post("/api/v1/config/validate");
$("save").onclick = () => post("/api/v1/config/save");
$("test-local").onclick = () => post("/api/v1/config/test-local");
$("test-frontier").onclick = () => post("/api/v1/config/test-frontier");
load().catch(err => show({error: String(err)}));
</script>
</body>
</html>
"""


def _payload_sections(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("sections"), dict):
        return payload["sections"]
    return {}


def _write_allowed(request: Request) -> bool:
    if os.environ.get("TINYCTX_DASHBOARD_WRITE") == "1":
        return True
    host = (request.client.host if request.client else "") or ""
    return host in {"127.0.0.1", "::1", "localhost"}


def _merged_effective_sections(sections: dict[str, Any]) -> dict[str, Any]:
    merged = effective_config()
    try:
        parsed = read_config_text().get("parsed") or {}
    except Exception:
        parsed = {}
    for section, values in parsed.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
    for section, values in sections.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
    return merged


def _backend_headers(backend: dict[str, Any]) -> dict[str, str]:
    headers = dict(backend.get("headers") or {})
    api_key_env = backend.get("api_key_env")
    if api_key_env and "Authorization" not in headers:
        api_key = os.environ.get(str(api_key_env))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _json_error(message: str, status_code: int = 400, **extra: Any) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message, **extra}, status_code=status_code)


def _safe_preview(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return json.dumps(data["data"][:3], ensure_ascii=False)[:500]
            if "choices" in data:
                return json.dumps(data["choices"][:1], ensure_ascii=False)[:500]
            if "output" in data:
                return json.dumps(data["output"][:1], ensure_ascii=False)[:500]
            if "error" in data:
                return json.dumps(data["error"], ensure_ascii=False)[:500]
        return json.dumps(data, ensure_ascii=False)[:500]
    except Exception:
        return (getattr(response, "text", "") or "")[:500]


# ─── FastAPI route registration ────────────────────────────────────────────


def register(app: Any, log_dir: Path) -> None:
    """Mount dashboard routes on `app`. Call once at proxy startup."""

    @app.get("/dashboard")
    def _dashboard_html() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/dashboard/config")
    def _dashboard_config_html() -> HTMLResponse:
        return HTMLResponse(_CONFIG_HTML)

    @app.get("/api/v1/config")
    def _api_v1_config() -> JSONResponse:
        try:
            current = read_config_text()
            return JSONResponse({
                **current,
                "schema": config_schema(),
                "presets": config_presets(),
                "effective": effective_config(),
                "env_overrides": env_overrides(),
            })
        except Exception as e:  # noqa: BLE001 - config page should surface parse errors
            return _json_error(str(e), status_code=400)

    @app.post("/api/v1/config/validate")
    async def _api_v1_config_validate(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return _json_error("invalid JSON body")
        sections = _payload_sections(payload)
        result = validate_sections(sections)
        try:
            current = read_config_text()
            result["rendered"] = merge_sections_into_toml(current["raw"], sections)
        except Exception as e:  # noqa: BLE001
            result["render_error"] = str(e)
        return JSONResponse(result)

    @app.post("/api/v1/config/save")
    async def _api_v1_config_save(request: Request) -> JSONResponse:
        if not _write_allowed(request):
            return _json_error(
                "config writes require localhost or TINYCTX_DASHBOARD_WRITE=1",
                status_code=403,
            )
        try:
            payload = await request.json()
        except Exception:
            return _json_error("invalid JSON body")
        sections = _payload_sections(payload)
        result = validate_sections(sections)
        if not result["ok"]:
            return JSONResponse(result, status_code=400)
        try:
            current = read_config_text()
            rendered = merge_sections_into_toml(current["raw"], sections)
            saved = save_config_text(rendered)
            return JSONResponse({
                "ok": True,
                **saved,
                "needs_restart": True,
                "warnings": result["warnings"],
            })
        except Exception as e:  # noqa: BLE001
            return _json_error(str(e), status_code=400)

    @app.post("/api/v1/config/test-local")
    async def _api_v1_config_test_local(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return _json_error("invalid JSON body")
        sections = _payload_sections(payload)
        result = validate_sections(sections)
        if not result["ok"]:
            return JSONResponse(result, status_code=400)
        backend = _merged_effective_sections(sections).get("local", {})
        base_url = str(backend.get("base_url") or "").rstrip("/")
        model = str(backend.get("model") or "local")
        wire_api = str(backend.get("wire_api") or "chat")
        if not base_url:
            return _json_error("local.base_url is required")
        headers = _backend_headers(backend)
        try:
            with httpx.Client(timeout=10) as client:
                models = client.get(f"{base_url}/models", headers=headers)
                if wire_api == "responses":
                    body = {
                        "model": model,
                        "input": "Reply with pong.",
                        "stream": False,
                        "max_output_tokens": 16,
                    }
                    completion = client.post(f"{base_url}/responses", json=body, headers=headers)
                else:
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with pong."}],
                        "stream": False,
                        "max_tokens": 16,
                    }
                    completion = client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            return JSONResponse({
                "ok": models.status_code < 400 and completion.status_code < 400,
                "models_status": models.status_code,
                "completion_status": completion.status_code,
                "models_preview": _safe_preview(models),
                "completion_preview": _safe_preview(completion),
            })
        except Exception as e:  # noqa: BLE001
            return _json_error(str(e), status_code=502)

    @app.post("/api/v1/config/test-frontier")
    async def _api_v1_config_test_frontier(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return _json_error("invalid JSON body")
        sections = _payload_sections(payload)
        result = validate_sections(sections)
        if not result["ok"]:
            return JSONResponse(result, status_code=400)
        backend = _merged_effective_sections(sections).get("frontier", {})
        base_url = str(backend.get("base_url") or "")
        is_codex_official = "chatgpt.com/backend-api/codex" in base_url
        return JSONResponse({
            "ok": bool(base_url and backend.get("wire_api") and backend.get("model")),
            "base_url": base_url,
            "wire_api": backend.get("wire_api"),
            "model": backend.get("model"),
            "official_codex_backend": is_codex_official,
            "requires_openai_api_key": False if is_codex_official else bool(backend.get("api_key_env")),
            "message": (
                "Codex official backend uses Codex/ChatGPT auth; OPENAI_API_KEY is not required."
                if is_codex_official else
                "Non-official frontier should set api_key_env when the provider requires a key."
            ),
        })

    @app.get("/dashboard/stream")
    async def _dashboard_stream(request: Request) -> StreamingResponse:
        async def _gen() -> AsyncIterator[bytes]:
            async for chunk in stream_events(log_dir):
                if await request.is_disconnected():
                    break
                yield chunk
        return StreamingResponse(_gen(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                            "X-Accel-Buffering": "no"})

    @app.get("/dashboard/state")
    def _dashboard_state() -> JSONResponse:
        return JSONResponse(state_snapshot())

    @app.get("/dashboard/aggregates")
    def _dashboard_aggregates(since_s: int = 900) -> JSONResponse:
        if since_s < 60:
            since_s = 60
        if since_s > 86400:
            since_s = 86400
        return JSONResponse(aggregates(log_dir, since_s=since_s))

    @app.get("/dashboard/recent")
    def _dashboard_recent(limit: int = 30) -> JSONResponse:
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200
        return JSONResponse(recent_events(log_dir, limit=limit))

    @app.get("/dashboard/plans")
    def _dashboard_plans() -> JSONResponse:
        """List persisted plans (per-cwd) with metadata. Lets you see
        which repos have a saved plan and when it was last updated."""
        try:
            from . import plan_persistence as _pp
            state_dir = log_dir.parent / "state"
            return JSONResponse({
                "dir": str(state_dir / "plans"),
                "plans": _pp.list_plans(state_dir),
            })
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/dashboard/plans")
    def _dashboard_plans_clear(cwd: str = "") -> JSONResponse:
        """Delete the persisted plan for `cwd`. Use when you want to
        start a genuinely fresh task and don't want stale context."""
        try:
            from . import plan_persistence as _pp
            state_dir = log_dir.parent / "state"
            removed = _pp.clear_plan(state_dir, cwd)
            return JSONResponse({"removed": removed, "cwd": cwd})
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/forensics")
    def _dashboard_forensics_list(limit: int = 30) -> JSONResponse:
        """List recent forensics dumps. Each dump pairs a problematic
        request with its response + timing for post-mortem analysis."""
        try:
            from . import forensics as _fx
            forensics_dir = log_dir.parent / "forensics"
            return JSONResponse({
                "dir": str(forensics_dir),
                "dumps": _fx.list_dumps(forensics_dir, limit=limit),
            })
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/forensics/{name}")
    def _dashboard_forensics_get(name: str) -> JSONResponse:
        """Read a specific forensics dump by filename."""
        try:
            forensics_dir = log_dir.parent / "forensics"
            # Sanitize: only allow alphanumeric + dash + dot, no path traversal
            if not all(c.isalnum() or c in "-_." for c in name):
                return JSONResponse({"error": "invalid name"}, status_code=400)
            path = forensics_dir / name
            if not path.exists() or path.parent != forensics_dir:
                return JSONResponse({"error": "not found"}, status_code=404)
            return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/tool-metrics")
    def _dashboard_tool_metrics() -> JSONResponse:
        """Per-(namespace, tool) call counts — surfaces which MCP
        servers + built-in tools the agent actually uses. Helps catch
        dead-tool issues like the 2026-05-10 advisor-trim-bug:
        spawn_agent had 0 calls because frontier_trim_tools was
        dropping it (essentials list missing the agent protocol)."""
        try:
            from . import tool_metrics as _tm
            return JSONResponse(_tm.snapshot())
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/exec-resume")
    def _dashboard_exec_resume(limit: int = 50) -> JSONResponse:
        """Recent `codex exec resume` poke attempts. C-4 hybrid module
        fires these when soft_completion classifier returns a high-
        confidence PUNT — turning the passive force_frontier flag into
        an active side-process that doesn't wait on user input."""
        try:
            from . import exec_resume as _xr
            return JSONResponse({
                "history": _xr.history_snapshot(limit=limit),
                "state": _xr.state_snapshot(),
            })
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/dashboard/force-frontier")
    def _dashboard_force_frontier(
            proj_sid: str = "global",
            reason: str = "manual",
    ) -> JSONResponse:
        """Manually set the empty-response-guard flag for a session so
        the NEXT request from codex auto-routes to frontier. Used to
        recover sessions that hit an empty-response BEFORE the
        empty_response_guard code was deployed."""
        try:
            from . import empty_response_guard as _erg
            _erg.force_next_to_frontier(proj_sid, reason)
            return JSONResponse({
                "set": True,
                "proj_sid": proj_sid,
                "reason": reason,
                "current_state": _erg.peek_force_frontier(proj_sid),
            })
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    # ─── P4: structured machine-readable JSON API ──────────────────────
    # /api/v1/state aggregates all per-session module snapshots into a
    # single response so external monitors don't have to scrape multiple
    # /dashboard/* endpoints. /api/v1/escalate is the operator's
    # one-button "force this session to frontier next turn" hook.

    @app.get("/api/v1/state")
    def _api_v1_state() -> JSONResponse:
        """Snapshot of all per-session in-memory state. Pulls from each
        module's `state_snapshot()`/`history_snapshot()` so it stays in
        sync without dashboard knowing internals."""
        from datetime import datetime, timezone

        active: list[dict[str, Any]] = []
        force_frontier_flags: dict[str, Any] = {}
        stuck_loop_state: dict[str, Any] = {}
        synthetic_continue_state: dict[str, Any] = {}
        exec_resume_history: list[dict[str, Any]] = []
        exec_resume_state: dict[str, Any] = {}
        request_phase_snap: dict[str, Any] = {}

        try:
            from . import request_phase as _rp
            request_phase_snap = _rp.state_snapshot()
            now = time.time()
            for sid, info in request_phase_snap.items():
                active.append({
                    "proj_sid": sid,
                    "phase": info.get("phase", ""),
                    "since_ts": info.get("since_ts", 0.0),
                    "age_s": round(now - info.get("since_ts", now), 2),
                    "request_id": info.get("request_id", ""),
                })
        except Exception:  # noqa: BLE001
            request_phase_snap = {}
        try:
            from . import empty_response_guard as _erg
            force_frontier_flags = _erg.state_snapshot()
        except Exception:  # noqa: BLE001
            force_frontier_flags = {}
        # stuck_loop.state_snapshot is per-sid; iterate known sids
        try:
            from . import stuck_loop as _sl
            for sid in request_phase_snap.keys():
                stuck_loop_state[sid] = _sl.state_snapshot(sid)
        except Exception:  # noqa: BLE001
            stuck_loop_state = {}
        try:
            from . import synthetic_continue as _syn
            for sid in request_phase_snap.keys():
                synthetic_continue_state[sid] = _syn.state_snapshot(sid)
        except Exception:  # noqa: BLE001
            synthetic_continue_state = {}
        try:
            from . import exec_resume as _xr
            exec_resume_history = _xr.history_snapshot(20)
            exec_resume_state = _xr.state_snapshot()
        except Exception:  # noqa: BLE001
            exec_resume_history = []
            exec_resume_state = {}

        return JSONResponse({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "active_sessions": len(active),
                "force_frontier_flagged": len(force_frontier_flags),
            },
            "active": active,
            "force_frontier_flags": force_frontier_flags,
            "stuck_loop_state": stuck_loop_state,
            "exec_resume_history": exec_resume_history,
            "synthetic_continue_state": synthetic_continue_state,
            "exec_resume_state": exec_resume_state,
        })

    @app.post("/api/v1/escalate")
    async def _api_v1_escalate(request: Request) -> JSONResponse:
        """Body: {"proj_sid": "..."}. Sets empty-response-guard flag
        so the NEXT request to that session forces the frontier route.
        Returns 202 on success, 400 when proj_sid is missing."""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"error": "missing or invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": "body must be a JSON object"}, status_code=400)
        proj_sid = payload.get("proj_sid") or ""
        if not proj_sid:
            return JSONResponse(
                {"error": "proj_sid is required"}, status_code=400)
        try:
            from . import empty_response_guard as _erg
            _erg.force_next_to_frontier(proj_sid, "manual_api")
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse({"escalated": proj_sid}, status_code=202)
