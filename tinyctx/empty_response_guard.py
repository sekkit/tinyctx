"""Detect upstream empty responses and force the next turn to frontier.

Live trace 2026-05-10 turn 1780 (~05:07): DeepSeek-v4-flash with 724K
input context returned `completion_tokens=1, content="", finish_reason=stop`
— effectively an empty response. Codex displayed the empty content and
paused for user input. Session sat dormant for 3.5 hours until the
laptop ran out of battery. The user came back to a "stuck" session.

Root cause: the local backend silently degraded under long-context
pressure (or hit some internal limit). tinyctx forwarded the empty
response cleanly, but neither tinyctx nor codex had any "this looks
broken" detection.

Mechanism
─────────
1. After each stream ends, parse the SSE buffer's tail for the upstream's
   `usage` block. Extract `completion_tokens` (and `finish_reason`).
2. If `completion_tokens` is very small (< threshold, default 5) AND
   `finish_reason` indicates a normal stop (`stop` or `length`), set
   the per-session `_FORCE_NEXT_TO_FRONTIER` flag.
3. On the next `/v1/responses` request from codex for the same session,
   the proxy reads the flag, forces `route=frontier` (bypassing the
   self-classifier and route heuristic), and clears the flag.

This gives the user one automatic retry on the frontier model
(gpt-5.5 via chatgpt.com codex backend), which is far less likely to
silently truncate. If the frontier ALSO returns empty, the flag won't
re-fire next turn (it's one-shot per detection) and we degrade
gracefully — at least the user gets a real response one way or another.

Manual trigger
──────────────
For testing or manual recovery (e.g. an empty response that happened
before this code was deployed), call `force_next_to_frontier(proj_sid)`
directly. The next request to that session will be routed to frontier
regardless of body shape.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any


# ─── per-session state ─────────────────────────────────────────────────────

_FORCE_NEXT_TO_FRONTIER: dict[str, dict[str, Any]] = defaultdict(dict)
# Each entry: {"set_at": ts, "reason": str, "completion_tokens": int}


# ─── usage / finish_reason extraction ─────────────────────────────────────
# Both Responses-API and Chat-Completions emit a `usage` object in the
# final event of a successful stream. For Chat-Completions:
#   {"usage":{"prompt_tokens":N,"completion_tokens":M,...}}
# For Responses-API:
#   {"type":"response.completed","response":{"usage":{"input_tokens":N,
#    "output_tokens":M,...}}}
# We scan the buffer tail for either shape.

_COMPLETION_TOKENS_RE = re.compile(
    r'"completion_tokens"\s*:\s*(\d+)')
# Responses-API uses `output_tokens` instead
_OUTPUT_TOKENS_RE = re.compile(
    r'"output_tokens"\s*:\s*(\d+)')
_FINISH_REASON_RE = re.compile(
    r'"finish_reason"\s*:\s*"([^"]+)"')
# Responses-API status: stop / completed / incomplete
_STATUS_RE = re.compile(
    r'"status"\s*:\s*"(stop|completed|incomplete|length)"')


def _extract_tail_usage(raw_buffer: str
                         ) -> tuple[int | None, str]:
    """Pull `(completion_tokens, finish_reason)` from the buffer's tail.
    Returns (None, "") when neither field is found. `completion_tokens`
    falls back to `output_tokens` for Responses-API. `finish_reason`
    falls back to the Responses-API `status` field."""
    if not raw_buffer:
        return None, ""
    # Search the LAST 2KB of the buffer — usage is always in the last event
    tail = raw_buffer[-2000:] if len(raw_buffer) > 2000 else raw_buffer
    # completion_tokens (chat) > output_tokens (responses-api)
    completion: int | None = None
    m = _COMPLETION_TOKENS_RE.search(tail)
    if m:
        try:
            completion = int(m.group(1))
        except (ValueError, TypeError):
            pass
    if completion is None:
        m = _OUTPUT_TOKENS_RE.search(tail)
        if m:
            try:
                completion = int(m.group(1))
            except (ValueError, TypeError):
                pass
    # finish_reason from chat / status from responses-api
    finish = ""
    fm = _FINISH_REASON_RE.search(tail)
    if fm:
        finish = fm.group(1)
    if not finish:
        sm = _STATUS_RE.search(tail)
        if sm:
            finish = sm.group(1)
    return completion, finish


# ─── detection + flag setting ─────────────────────────────────────────────


def maybe_flag_empty_response(
        proj_sid: str,
        raw_buffer: str,
        *,
        min_completion_tokens: int = 5,
        only_normal_finish: bool = True,
) -> dict[str, Any] | None:
    """Inspect a finished stream's buffer. If the upstream returned an
    "effectively empty" response (very few completion tokens AND
    finish_reason is a normal stop/length), set the force-frontier
    flag for this session and return the diagnostic dict. None when
    response looked normal.

    `only_normal_finish=True` (default) ignores cases where finish was
    `tool_calls` (agent IS doing something) or empty (incomplete event).
    Only `stop` or `length` count as "the model intended to be done"."""
    completion, finish = _extract_tail_usage(raw_buffer)
    if completion is None:
        return None  # couldn't read usage — don't act
    if completion >= min_completion_tokens:
        return None  # response had real content
    if only_normal_finish and finish not in ("stop", "length",
                                              "completed", "incomplete"):
        return None  # tool_calls etc. — agent is acting, not empty
    # Empty response detected
    info = {
        "set_at": time.time(),
        "reason": (f"empty_response: completion_tokens={completion} "
                    f"finish_reason={finish or 'unknown'}"),
        "completion_tokens": completion,
        "finish_reason": finish,
    }
    _FORCE_NEXT_TO_FRONTIER[proj_sid] = info
    return info


# ─── flag check + consume ─────────────────────────────────────────────────


def consume_force_frontier(proj_sid: str) -> dict[str, Any] | None:
    """Atomically check + clear the flag. If set, returns the info dict
    (caller forces frontier route this turn); else None (normal routing)."""
    info = _FORCE_NEXT_TO_FRONTIER.pop(proj_sid, None)
    return info if info else None


def peek_force_frontier(proj_sid: str) -> dict[str, Any] | None:
    """Look at the flag without consuming it. For dashboard display."""
    info = _FORCE_NEXT_TO_FRONTIER.get(proj_sid)
    return dict(info) if info else None


def force_next_to_frontier(proj_sid: str, reason: str = "manual") -> None:
    """Manually set the flag for testing or recovery. Used when an
    empty response was observed BEFORE this code was deployed and
    you want the next turn forced to frontier without waiting for
    a fresh empty response."""
    _FORCE_NEXT_TO_FRONTIER[proj_sid] = {
        "set_at": time.time(),
        "reason": f"manual: {reason}",
        "completion_tokens": -1,
        "finish_reason": "manual",
    }


# ─── test/dev helpers ─────────────────────────────────────────────────────


def reset_state(proj_sid: str | None = None) -> None:
    if proj_sid is None:
        _FORCE_NEXT_TO_FRONTIER.clear()
        return
    _FORCE_NEXT_TO_FRONTIER.pop(proj_sid, None)


def state_snapshot() -> dict[str, dict[str, Any]]:
    """Inspect all flagged sessions. For dashboard."""
    return {sid: dict(info)
            for sid, info in _FORCE_NEXT_TO_FRONTIER.items()
            if info}
