"""Per-request lifecycle phase tracking.

Inspired by openai/symphony SPEC §7.2's run-attempt phase enum: explicit
lifecycle states so operators can answer "what is request X stuck on
right now?" without grepping `tinyctx-YYYYMMDD.jsonl`.

Each `proj_sid` (composite project + session key) carries at most ONE
active phase entry — the latest one transitioned through the request
handler. Long-running phases (`backend_streaming`, `injecting`,
`post_stream_classifying`) are the most actionable: a session sitting in
`backend_streaming` for >60s likely indicates a stalled upstream.

Mechanism
─────────
proxy.py calls `set_phase(proj_sid, RequestPhase.X, request_id=...)`
at ~10 natural lifecycle points. The dashboard reads the dict via
`state_snapshot()` and renders the current phase + age per session.

Storage is in-memory module-level (single-process proxy); test harness
clears it via `reset_state()`.
"""
from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum
from typing import Any


class RequestPhase(str, Enum):
    """Lifecycle stages a /v1/responses request transitions through.
    Inheriting from `str` makes the enum JSON-serializable (FastAPI's
    JSONResponse encodes str subclasses natively)."""

    received = "received"
    classifying = "classifying"
    routing = "routing"
    backend_streaming = "backend_streaming"
    post_stream_classifying = "post_stream_classifying"
    injecting = "injecting"
    done = "done"
    stalled = "stalled"
    retrying = "retrying"
    escalated_to_frontier = "escalated_to_frontier"
    empty_guarded = "empty_guarded"


# ─── per-session state ─────────────────────────────────────────────────────
# Keyed by proj_sid. Each entry: {"phase": str, "since_ts": float,
# "request_id": str}. Single-entry-per-session: a new transition replaces
# the previous one. State is per-process (proxy is single-process).

_PHASE: dict[str, dict[str, Any]] = defaultdict(dict)


def set_phase(proj_sid: str,
              phase: RequestPhase | str,
              request_id: str = "") -> None:
    """Record `proj_sid` is now in `phase`. `request_id` is opaque —
    operators correlate it with `request_trace` JSONL entries.

    Always succeeds (no-op if proj_sid is empty). Never raises so
    proxy.py call sites can fire-and-forget."""
    if not proj_sid:
        return
    value = phase.value if isinstance(phase, RequestPhase) else str(phase)
    _PHASE[proj_sid] = {
        "phase": value,
        "since_ts": time.time(),
        "request_id": request_id or "",
    }


def get_phase(proj_sid: str) -> dict[str, Any] | None:
    """Return current phase entry for `proj_sid`, or None when no
    transition has been recorded for this session."""
    info = _PHASE.get(proj_sid)
    return dict(info) if info else None


def state_snapshot() -> dict[str, dict[str, Any]]:
    """All active phase entries, copied. For dashboard rendering."""
    return {sid: dict(info)
            for sid, info in _PHASE.items()
            if info}


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that one."""
    if proj_sid is None:
        _PHASE.clear()
        return
    _PHASE.pop(proj_sid, None)
