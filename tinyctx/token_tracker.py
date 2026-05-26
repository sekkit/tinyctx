"""Real-time token usage tracker — in-memory, thread-safe.

Tracks per-request token estimates for dashboard display. Resets on
proxy restart.

Dimensions:
  - est_input: codex's original request (before proxy processing)
  - injections: proxy-added context (scout, ctx-pack, rules, orchestrator)
  - forwarded: total sent to LLM (est_input + injections ± compression)
  - per-route breakdown (local vs frontier)
  - advisor call overhead
"""

from __future__ import annotations

import threading

_lock = threading.Lock()

_total_requests = 0
_total_est_input = 0
_total_injections = 0
_total_forwarded = 0
_total_bytes_out = 0
_advisor_requests = 0
_advisor_est_input = 0
_advisor_forwarded = 0

_local_requests = 0
_local_est_input = 0
_local_forwarded = 0
_frontier_requests = 0
_frontier_est_input = 0
_frontier_forwarded = 0


def record(
    *,
    est_input_tokens: int = 0,
    injection_tokens: int = 0,
    forwarded_tokens: int = 0,
    bytes_out: int = 0,
    route: str = "",
    is_advisor: bool = False,
) -> None:
    global _total_requests, _total_est_input, _total_injections, _total_forwarded, _total_bytes_out
    global _advisor_requests, _advisor_est_input, _advisor_forwarded
    global _local_requests, _local_est_input, _local_forwarded
    global _frontier_requests, _frontier_est_input, _frontier_forwarded

    with _lock:
        _total_requests += 1
        _total_est_input += est_input_tokens
        _total_injections += injection_tokens
        _total_forwarded += forwarded_tokens
        _total_bytes_out += bytes_out

        if is_advisor:
            _advisor_requests += 1
            _advisor_est_input += est_input_tokens
            _advisor_forwarded += forwarded_tokens

        if route == "local":
            _local_requests += 1
            _local_est_input += est_input_tokens
            _local_forwarded += forwarded_tokens
        elif route == "frontier":
            _frontier_requests += 1
            _frontier_est_input += est_input_tokens
            _frontier_forwarded += forwarded_tokens


def snapshot() -> dict:
    with _lock:
        # net = forwarded minus injections ≈ what the original input became after transforms
        net = _total_forwarded - _total_injections
        delta = _total_est_input - net  # positive = net compression, negative = net expansion
        return {
            "requests": _total_requests,
            "est_input_tokens": _total_est_input,
            "injection_tokens": _total_injections,
            "forwarded_tokens": _total_forwarded,
            "net_tokens": net,
            "delta": delta,
            "bytes_out": _total_bytes_out,
            "advisor": {
                "requests": _advisor_requests,
                "est_input_tokens": _advisor_est_input,
                "forwarded_tokens": _advisor_forwarded,
            },
            "by_route": {
                "local": {
                    "requests": _local_requests,
                    "est_input_tokens": _local_est_input,
                    "forwarded_tokens": _local_forwarded,
                },
                "frontier": {
                    "requests": _frontier_requests,
                    "est_input_tokens": _frontier_est_input,
                    "forwarded_tokens": _frontier_forwarded,
                },
            },
        }
