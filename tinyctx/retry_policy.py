"""Unified upstream-retry classification policy for tinyctx proxy.

User directive: "凡是中断了都要加重试" — every interruption should trigger
retry, with escalation to a different backend if same-backend retry
exhausts. Bounded by an attempt counter to avoid infinite loops.

This module is a PURE classifier — it has no I/O, no httpx calls, no
state. The proxy's `_forward` / `_stream_proxy` wrappers consult
`classify_failure(...)` to decide the next action, then act on the
returned `RetryAction`.

Policy matrix (see Config.retry_*):

  Local 4xx (400/422 — schema mismatch / body shape) → retry_escalate
  Local 4xx (401/403/404)                            → propagate (permanent)
  Local 429 (rate limit)                             → retry_same (with backoff)
  Local 5xx                                          → retry_same → retry_escalate
  Local connection drop / read timeout               → retry_same → retry_escalate
  Frontier 4xx                                       → propagate
                                                        (flag force_next_to_frontier
                                                         so codex's own retry doesn't
                                                         reuse stale body on local)
  Frontier 5xx                                       → retry_same → propagate
  Frontier connection drop                           → retry_same → propagate
  Compaction request (is_compaction=True)            → propagate (codex self-retries)
  All retries exhausted (>= max_total_retries)       → propagate

The classifier is consulted exclusively BEFORE any client bytes have
been yielded. Once we yield a single byte to codex, partial-content
risk forbids a retry (codex's SSE parser would see duplicate events).
The proxy enforces this with a `bytes_yielded` guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ─── Public types ─────────────────────────────────────────────────────────


RetryDecision = Literal["retry_same", "retry_escalate", "propagate"]


@dataclass(frozen=True)
class RetryAction:
    """The classifier's verdict.

    decision:
      retry_same     — retry the same backend (same url, same body).
                       Increments `attempts_remaining_same`.
      retry_escalate — switch to frontier on the next attempt.
                       Should also set force_next_to_frontier so codex's
                       subsequent retries land on frontier too.
      propagate      — give up; surface the error to codex as-is.

    reason: short human-readable string, included in retry_attempted log
            event and forensics dump.

    backoff_s: optional sleep before retry. Used for 429 with a non-zero
               value; defaults to 0.0 for everything else.

    escalate_flag_reason: when decision == retry_escalate, the string the
                          proxy passes to force_next_to_frontier(...) so
                          the NEXT turn from codex (not just this one)
                          also routes frontier-side.
    """
    decision: RetryDecision
    reason: str
    backoff_s: float = 0.0
    escalate_flag_reason: str = ""


# ─── Classifier ───────────────────────────────────────────────────────────


# Permanent 4xx codes — retry / escalation can't fix these.
_PERMANENT_LOCAL_4XX = frozenset({401, 403, 404})
_PERMANENT_FRONTIER_4XX = frozenset({401, 403, 404})

# 4xx codes a different backend MIGHT accept. 400/422 are body-schema
# rejections (LMStudio "Unexpected message role.", strict OpenAI-compat
# servers rejecting `developer` role, missing `text.format`, etc.).
# Frontier (chatgpt.com) accepts every codex-emitted field+role, so
# escalating these is generally productive.
_ESCALATABLE_LOCAL_4XX = frozenset({400, 422})


def classify_failure(
    *,
    route: str,
    status: int | None,
    is_connection_error: bool,
    is_compaction: bool,
    attempts_used: int,
    max_total_retries: int,
    upstream_retry_count: int,
    retry_on_local_4xx_escalate_frontier: bool = True,
    retry_on_frontier_4xx: bool = False,
    retry_after_s: float = 0.0,
) -> RetryAction:
    """Pure classifier — returns the next action for this failure.

    Arguments:
      route                : current decision route ("local" or "frontier")
      status               : HTTP status code, or None if connection error
      is_connection_error  : True for httpx.ConnectError / ReadTimeout /
                             RemoteProtocolError / ReadError / WriteError.
                             If True, `status` is ignored.
      is_compaction        : codex sent its own compaction-summary request.
                             We never retry these; codex self-retries.
      attempts_used        : how many attempts have ALREADY completed
                             (1 = initial attempt done, currently in failure).
      max_total_retries    : safety bound across all retry kinds for this
                             request (CFG.max_total_retries_per_request).
      upstream_retry_count : per-backend same-backend retry budget
                             (CFG.upstream_retry_count, currently 1).
      retry_on_local_4xx_escalate_frontier:
                             escalate to frontier on local 400/422.
      retry_on_frontier_4xx:
                             retry same-backend on frontier 4xx (off by
                             default — chatgpt.com is strict, retrying
                             with same body is unlikely to help).
      retry_after_s        : value the upstream gave us in Retry-After
                             header (for 429); 0 means no header.
    """
    # Compaction requests: never retry. Codex's compactor re-runs with a
    # different shape if it fails — proxy-side retry just doubles cost.
    if is_compaction:
        return RetryAction("propagate", "compaction_request_no_retry")

    # Safety cap — hard stop regardless of error type.
    if attempts_used >= max_total_retries:
        return RetryAction("propagate",
                           f"max_total_retries({max_total_retries})_exhausted")

    # Connection-level failure → transient. Retry same backend first;
    # escalate if same-backend budget exhausted.
    if is_connection_error:
        if route == "local":
            if attempts_used <= upstream_retry_count:
                return RetryAction("retry_same",
                                   "local_connection_error_retry_same")
            return RetryAction("retry_escalate",
                               "local_connection_error_escalate_frontier",
                               escalate_flag_reason="retry_escalate_connection_error")
        # frontier
        if attempts_used <= upstream_retry_count:
            return RetryAction("retry_same",
                               "frontier_connection_error_retry_same")
        return RetryAction("propagate",
                           "frontier_connection_error_no_further_escalation")

    # HTTP status known.
    s = int(status or 0)

    if route == "local":
        if s in _PERMANENT_LOCAL_4XX:
            return RetryAction("propagate",
                               f"local_{s}_permanent_no_retry")
        if s == 429:
            # Rate-limited — bounded retry same backend with backoff.
            # Cap the backoff so we don't block the request forever.
            backoff = min(max(retry_after_s, 1.0), 5.0) if retry_after_s else 1.0
            if attempts_used <= upstream_retry_count:
                return RetryAction("retry_same",
                                   "local_429_retry_after",
                                   backoff_s=backoff)
            return RetryAction("retry_escalate",
                               "local_429_escalate_frontier",
                               escalate_flag_reason="retry_escalate_429")
        if s in _ESCALATABLE_LOCAL_4XX:
            if retry_on_local_4xx_escalate_frontier:
                return RetryAction("retry_escalate",
                                   f"local_{s}_escalate_frontier",
                                   escalate_flag_reason=f"retry_escalate_4xx_{s}")
            return RetryAction("propagate", f"local_{s}_escalation_disabled")
        if 500 <= s < 600:
            if attempts_used <= upstream_retry_count:
                return RetryAction("retry_same",
                                   f"local_{s}_retry_same")
            return RetryAction("retry_escalate",
                               f"local_{s}_escalate_frontier",
                               escalate_flag_reason=f"retry_escalate_5xx_{s}")
        if 400 <= s < 500:
            # 4xx not covered above (e.g. 408, 409, 410, 413) — try once
            # more on the same backend, then propagate. These are often
            # request-shape issues that retry alone won't fix, but a
            # single retry costs little and catches transient races.
            if attempts_used <= upstream_retry_count:
                return RetryAction("retry_same",
                                   f"local_{s}_retry_same_uncategorized_4xx")
            return RetryAction("propagate",
                               f"local_{s}_propagate_uncategorized_4xx")
        # 1xx/3xx/etc — shouldn't happen on POST, but propagate cleanly.
        return RetryAction("propagate", f"local_status_{s}_unexpected")

    # route == "frontier"
    if s in _PERMANENT_FRONTIER_4XX:
        return RetryAction("propagate", f"frontier_{s}_permanent_no_retry")
    if s == 429:
        backoff = min(max(retry_after_s, 1.0), 5.0) if retry_after_s else 1.0
        if attempts_used <= upstream_retry_count:
            return RetryAction("retry_same",
                               "frontier_429_retry_after",
                               backoff_s=backoff)
        return RetryAction("propagate",
                           "frontier_429_no_further_retry")
    if 400 <= s < 500:
        # Frontier 4xx: codex's chatgpt.com endpoint is strict — retry
        # with the same body is unlikely to help. Propagate AND mark
        # the session so codex's NEXT request doesn't reuse a stale body
        # on local. The mark fires via escalate_flag_reason; the proxy
        # decides whether to actually re-route based on whether
        # retry_on_frontier_4xx is enabled.
        if retry_on_frontier_4xx and attempts_used <= upstream_retry_count:
            return RetryAction("retry_same",
                               f"frontier_{s}_retry_opt_in")
        # Still set the flag so future turns from this session don't
        # immediately fall back to local with the same shape; future
        # turn body may differ but if it's same conv it'll re-hit
        # frontier anyway (force_next_to_frontier covers that case).
        return RetryAction(
            "propagate",
            f"frontier_{s}_propagate",
            escalate_flag_reason=f"retry_force_next_frontier_4xx_{s}",
        )
    if 500 <= s < 600:
        if attempts_used <= upstream_retry_count:
            return RetryAction("retry_same",
                               f"frontier_{s}_retry_same")
        return RetryAction("propagate",
                           f"frontier_{s}_no_further_retry")
    return RetryAction("propagate", f"frontier_status_{s}_unexpected")


# ─── Inline counter for request-scoped attempt tracking ───────────────────


@dataclass
class RequestRetryState:
    """Per-request retry counter the proxy maintains across attempts.

    The proxy creates one of these on request entry and passes it through
    the `_forward` / `_stream_proxy` retry loop. Reset is implicit (a new
    request gets a new state instance) — no global table to manage.
    """
    attempts_used: int = 0
    escalations_used: int = 0
    last_action: RetryAction | None = None

    def record_attempt(self) -> None:
        self.attempts_used += 1

    def record_escalation(self) -> None:
        self.escalations_used += 1
