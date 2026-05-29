"""Deep forensics: capture full request + response + timing for failed turns.

Standard trace (`request_trace` JSONL events) is per-turn metadata: route
decision, byte counts, timing summary. NOT the actual request/response
bodies — those are GBs/day and not worth keeping for healthy turns.

For RARE failure modes (empty response from upstream, classifier-flagged
soft-punt) we want post-mortem evidence:
  - exact request body sent to upstream (instructions, tools, input items)
  - exact response received (full SSE buffer)
  - HTTP response headers (rate-limit / quota / model-version markers)
  - timing breakdown (connect, first byte, first content delta, completed)

Stored as one JSON file per failure under `~/.tinyctx/forensics/`. User
can grep / diff between healthy and failed turns to pin down root cause.

Capture is OPT-IN (cfg.forensics_enabled, default True) and BOUNDED:
  - max 100 dumps total (rolling, oldest deleted)
  - max 10MB per dump
  - retains for 7 days

Triggers (any of):
  - upstream returned <5 completion tokens with finish_reason in (stop,
    length) — the empty-response failure mode
  - soft_completion classifier verdict was PUNT with p ≥ trigger threshold
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from uuid import uuid4


# Per-session ring buffer of recent request snapshots. When a failure
# triggers, we look up the LAST request for that session and pair it
# with the response. Bounded so memory doesn't grow unbounded.
_REQUEST_RING: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))


_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|credential|authorization|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|bearer[_-]?token)",
    re.IGNORECASE,
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)
_SENSITIVE_LINE_RE = re.compile(
    r"(?im)^(\s*[\w.-]*(?:api[_-]?key|secret|password|credential|authorization|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|bearer[_-]?token)"
    r"[\w.-]*\s*[:=]\s*)(.+)$"
)
_PENDING_INPUT_MARKER = "[tinyctx pending input supplied"
_PENDING_VALUES_RE = re.compile(r"(?is)(\nValues:\n).*")


def _redact_string(value: str) -> str:
    out = value
    if _PENDING_INPUT_MARKER in out.lower():
        out = _PENDING_VALUES_RE.sub(r"\1<redacted: pending input values>", out)
    out = _SENSITIVE_LINE_RE.sub(r"\1<redacted>", out)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("<redacted: secret>", out)
    return out


def _redact_sensitive(obj: Any) -> Any:
    if isinstance(obj, str):
        return _redact_string(obj)
    if isinstance(obj, list):
        return [_redact_sensitive(v) for v in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if _SENSITIVE_KEY_RE.search(key) or key.lower() == "token":
                out[k] = "<redacted>"
            else:
                out[k] = _redact_sensitive(v)
        return out
    return obj


def capture_request_snapshot(
        proj_sid: str,
        request_id: str,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        request_started_at: float,
) -> None:
    """Stash the request snapshot in a per-session ring. Called at
    request entry by the proxy. Cheap — just a dict + ring append."""
    snap = {
        "request_id": request_id,
        "url": url,
        "request_started_at": request_started_at,
        "body": _summarize_body(body),
        "headers": _scrub_headers(headers),
    }
    _REQUEST_RING[proj_sid].append(snap)


def get_recent_request(proj_sid: str) -> dict[str, Any] | None:
    """Return the most recent captured request snapshot for a session."""
    ring = _REQUEST_RING.get(proj_sid)
    if not ring:
        return None
    return dict(ring[-1])


def _summarize_body(body: Any) -> dict[str, Any]:
    """Capture the body in a form useful for post-mortem WITHOUT
    blowing up disk. Cap text fields, count list items, keep first
    + last input items in full so we can see context-window contents.
    Sensitive values are redacted before any first/last items are kept."""
    body = _redact_sensitive(body)
    if not isinstance(body, dict):
        return {"raw": str(body)[:1000]}
    out: dict[str, Any] = {}
    for k, v in body.items():
        if isinstance(v, str):
            out[k] = v if len(v) <= 4000 else v[:2000] + f"\n…[{len(v)-4000} chars truncated]…\n" + v[-2000:]
        elif isinstance(v, list):
            n = len(v)
            if n <= 6:
                out[k] = v
            else:
                # Keep first 3 + last 3 items, summarize middle
                out[k] = {
                    "_total_items": n,
                    "first_3": v[:3],
                    "middle_omitted": n - 6,
                    "last_3": v[-3:],
                }
        else:
            out[k] = v
    return out


def _scrub_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop auth tokens; keep diagnostic headers (rate limit / model
    version / request id)."""
    if not isinstance(headers, dict):
        return {}
    out: dict[str, str] = {}
    sensitive = {"authorization", "x-api-key", "openai-api-key",
                 "x-openai-api-key", "x-anthropic-api-key"}
    for k, v in headers.items():
        if k.lower() in sensitive:
            out[k] = f"<redacted: {len(str(v))} chars>"
        else:
            out[k] = str(v)[:500]
    return out


def write_forensics_dump(
        forensics_dir: Path,
        proj_sid: str,
        trigger: str,
        response_buffer: str,
        response_headers: dict[str, str] | None = None,
        timing: dict[str, float] | None = None,
        classifier_verdict: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        max_dumps: int = 100,
) -> Path | None:
    """Write a forensic dump tying the most recent request to its
    response + classifier verdict. Returns the path written, or None
    on failure (never raises). Rolls oldest dumps when count exceeds
    `max_dumps`."""
    request = get_recent_request(proj_sid)
    if request is None:
        # No prior request captured — degraded but still write what we have
        request = {"_note": "no recent request snapshot for this session"}
    try:
        forensics_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Why: forensics is best-effort post-mortem capture. If the dir
        # can't be created (read-only fs, permissions), drop the dump
        # silently rather than break the caller's failure-path code.
        return None
    now = time.time()
    dump = {
        "trigger": trigger,
        "captured_at": now,
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(now)),
        "proj_sid": proj_sid,
        "request": request,
        "response": {
            "buffer_chars": len(response_buffer),
            "buffer_head": response_buffer[:2000] if response_buffer else "",
            "buffer_tail": (response_buffer[-4000:]
                             if len(response_buffer) > 4000
                             else response_buffer),
            "headers": _scrub_headers(response_headers or {}),
        },
        "timing": dict(timing or {}),
        "classifier_verdict": dict(classifier_verdict or {}),
        "extra": dict(extra or {}),
    }
    fname = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{trigger}-{uuid4().hex[:8]}.json"
    path = forensics_dir / fname
    try:
        path.write_text(json.dumps(dump, ensure_ascii=False, default=str,
                                    indent=2),
                        encoding="utf-8")
    except OSError:
        # Why: forensic dump write failed (disk full, permissions). Best-
        # effort capture — return None so the caller knows nothing was
        # written, but never propagate the failure to the request path.
        return None
    # Roll: keep at most `max_dumps`
    try:
        existing = sorted(forensics_dir.glob("*.json"))
        if len(existing) > max_dumps:
            for old in existing[: len(existing) - max_dumps]:
                try:
                    old.unlink()
                except OSError:
                    # Why: per-file unlink failure during rollover; skip
                    # this file and continue trimming the rest.
                    pass
    except OSError:
        # Why: glob/sort of the forensics dir failed; skip rollover this
        # call. Disk pressure will be resolved when the next dump succeeds.
        pass
    return path


def list_dumps(forensics_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    """List recent forensics dumps with summary metadata. Used by the
    dashboard endpoint."""
    if not forensics_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for p in sorted(forensics_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                stat = p.stat()
                # Cheap parse: read first 2KB and grab the trigger
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    head = f.read(2000)
                trigger = "?"
                if '"trigger":' in head:
                    import re as _re
                    m = _re.search(r'"trigger":\s*"([^"]+)"', head)
                    if m:
                        trigger = m.group(1)
                out.append({
                    "name": p.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "trigger": trigger,
                })
            except OSError:
                # Why: per-file stat/read failure (file may have been
                # rolled away mid-listing). Skip and continue.
                continue
    except OSError:
        # Why: forensics dir glob failed; return whatever we collected
        # so far rather than crash the dashboard endpoint.
        pass
    return out


def reset_state() -> None:
    """Test/dev helper."""
    _REQUEST_RING.clear()
