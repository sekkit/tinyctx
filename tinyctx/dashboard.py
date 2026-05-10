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

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


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
                "self_classify_error", "stuck_loop_error"):
        base.update({
            "error": (e.get("error", "") or e.get("body", "") or "")[:200],
            "status": e.get("status", 0),
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
            continue
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
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
        return []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
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
      return badge + info;
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


# ─── FastAPI route registration ────────────────────────────────────────────


def register(app: Any, log_dir: Path) -> None:
    """Mount dashboard routes on `app`. Call once at proxy startup."""

    @app.get("/dashboard")
    def _dashboard_html() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

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
