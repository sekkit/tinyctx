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

State storage
─────────────
P3 migration: state lives in `tinyctx.session_state` under namespace
`request_phase` key `current` (single dict value per proj_sid). The key
is registered for compaction reset because a compaction boundary marks
the start of a logically fresh request flow — surfacing the pre-
compaction phase on the dashboard would be misleading.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

from . import session_state


class RequestPhase(str, Enum):
    """Lifecycle stages a /v1/responses request transitions through.
    Inheriting from `str` makes the enum JSON-serializable (FastAPI's
    JSONResponse encodes str subclasses natively)."""

    received = "received"
    classifying = "classifying"
    routing = "routing"
    compacting = "compacting"
    backend_streaming = "backend_streaming"
    post_stream_classifying = "post_stream_classifying"
    injecting = "injecting"
    done = "done"
    stalled = "stalled"
    retrying = "retrying"
    escalated_to_frontier = "escalated_to_frontier"
    empty_guarded = "empty_guarded"


# ─── SessionState namespace + compaction reset policy ─────────────────────

_NS = "request_phase"
_K_CURRENT = "current"

# Post-compaction is a logically fresh request flow — drop the stale
# phase entry so the dashboard doesn't render an outdated "stuck in
# backend_streaming" badge for a session whose stream actually ended
# at the compaction boundary.
session_state.register_compaction_reset(_NS, [_K_CURRENT])


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
    session_state.set(proj_sid, _NS, _K_CURRENT, {
        "phase": value,
        "since_ts": time.time(),
        "request_id": request_id or "",
    })


def get_phase(proj_sid: str) -> dict[str, Any] | None:
    """Return current phase entry for `proj_sid`, or None when no
    transition has been recorded for this session."""
    info = session_state.get(proj_sid, _NS, _K_CURRENT)
    return dict(info) if info else None


def state_snapshot() -> dict[str, dict[str, Any]]:
    """All active phase entries, copied. For dashboard rendering."""
    snap = session_state.snapshot()
    out: dict[str, dict[str, Any]] = {}
    for sid, by_ns in snap.items():
        info = by_ns.get(_NS, {}).get(_K_CURRENT)
        if info:
            out[sid] = dict(info)
    return out


def reset_state(proj_sid: str | None = None) -> None:
    """Test/dev helper. With no arg, clear all sessions; with a key,
    clear just that one."""
    if proj_sid is None:
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if _K_CURRENT in by_ns.get(_NS, {}):
                session_state.clear(sid, _NS, _K_CURRENT)
        return
    session_state.clear(proj_sid, _NS, _K_CURRENT)
