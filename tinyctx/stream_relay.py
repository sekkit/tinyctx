"""Stream relay: producer/consumer plumbing for the SSE upstream → codex path.

Extracted from `proxy._stream_proxy` in P6. The extraction preserves
behavior byte-for-byte; the goal is structural — turn one 1000-LOC async
generator into a small set of focused components with explicit
responsibilities so each can be reasoned about and tested in isolation.

Architecture
────────────

  ┌───────────────────┐    chunk_q     ┌────────────────────┐
  │ StreamProducer    │ ──────────────▶│ StreamConsumer     │
  │ (httpx + retry)   │  (tag,payload) │ (keepalive+yield)  │
  └───────────────────┘                └────────────────────┘
            ▲                                   │
            │ register/cancel                   │ yield bytes
            │                                   ▼
  ┌───────────────────┐                ┌────────────────────┐
  │ StallSupervisor   │                │ codex SSE parser   │
  │ (watchdog wiring) │                └────────────────────┘
  └───────────────────┘

`relay_stream(...)` is the top-level orchestrator that wires them
together. The post-stream concerns (soft-completion classify,
stream-rewrite synthesis, empty-response guard, terminator emission,
forensics) currently live in the proxy's `_stream_proxy` wrapper around
`relay_stream`; they are flagged for migration to a dedicated
PostStream component in P7.

Queue sentinels
───────────────

The producer pushes tagged tuples onto `chunk_q`:

  (None,       chunk_bytes)            — a body chunk to forward
  (_STATUS,    (status_code, err_body)) — upstream response head (200 or 4xx/5xx)
  (_SENTINEL,  None)                    — clean end of producer
  (_ERR,       exc)                     — terminal exception (incl. StallCancelledError)

These sentinels are module-private; only StreamProducer pushes them and
only StreamConsumer reads them.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING
from uuid import uuid4

import httpx

from . import retry_policy
from . import stall_watchdog as _stall
from .request_phase import RequestPhase, set_phase as _phase_set
from .router import Decision
from .tool_call_translator import ChatToResponsesTranslator, StreamTranslator

if TYPE_CHECKING:
    from .config import Config
    from .trace import RequestTrace


# Queue sentinels — module-private. Producer and consumer must agree.
_STATUS = object()
_SENTINEL = object()
_ERR = object()


# ─── StallSupervisor ──────────────────────────────────────────────────────


class StallSupervisor:
    """Thin wrapper over stall_watchdog register/unregister so the
    relay can opt in or out via one flag and the producer/consumer
    don't import stall_watchdog directly.

    Cancellation itself is driven by stall_watchdog's background loop
    via the on-stall callback registered at app startup — we just
    plug the producer task into its registry."""

    def __init__(self, proj_sid: str, *, enabled: bool):
        self.proj_sid = proj_sid
        self.enabled = enabled
        self._task: asyncio.Task | None = None

    def register(self, task: asyncio.Task) -> None:
        if not self.enabled:
            return
        self._task = task
        try:
            _stall.register_task(self.proj_sid, task)
        except Exception:  # noqa: BLE001
            pass

    def unregister(self) -> None:
        if not self.enabled or self._task is None:
            return
        try:
            _stall.unregister_task(self.proj_sid, self._task)
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        if not self.enabled:
            return
        try:
            _stall.clear(self.proj_sid)
        except Exception:  # noqa: BLE001
            pass


# ─── StreamProducer ───────────────────────────────────────────────────────


class StreamProducer:
    """Wraps the httpx upstream stream with a retry loop that consults
    retry_policy.classify_failure on every failure.

    Pushes tagged items onto a shared queue; the consumer is oblivious
    to retry semantics — it sees only the FINAL outcome. Once a body
    chunk has been pushed, retry is no longer permitted (partial
    content in flight).

    On asyncio.CancelledError (typically from stall_watchdog), pushes a
    synthetic StallCancelledError onto the queue so the consumer can
    emit a clean terminator rather than the generator dying mid-yield.

    The Authorization header is REBUILT for the frontier backend on
    retry_escalate (preserves f8c2489: a local-backend bearer must
    never leak to chatgpt.com or it returns 401 at bytes_out=0)."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        proj_sid: str,
        conv_sid: str | None,
        decision: Decision,
        timeout: httpx.Timeout,
        transport: httpx.AsyncBaseTransport | None,
        erg_key: str,
        request_id: str,
        cfg: "Config",
        log: Callable[..., None],
        build_frontier_retry_target: Callable[..., Any],
        resolve_api_key: Callable[..., str | None],
    ):
        self._url = url
        self._headers = headers
        self._body = body
        self._proj_sid = proj_sid
        self._conv_sid = conv_sid
        self._decision = decision
        self._timeout = timeout
        self._transport = transport
        self._erg_key = erg_key
        self._request_id = request_id
        self._cfg = cfg
        self._log = log
        self._build_frontier_retry_target = build_frontier_retry_target
        self._resolve_api_key = resolve_api_key
        # Mutable per-attempt state.
        self._retry_state = retry_policy.RequestRetryState()
        self._attempt_url = url
        self._attempt_headers = headers
        self._attempt_body = body
        self._attempt_decision = decision

    async def run(self, chunk_q: asyncio.Queue) -> None:
        """Drive the upstream request, retrying per policy. Always
        pushes a SENTINEL or ERR before returning — the consumer's
        receive loop relies on it."""
        produced_chunk = False
        try:
            while True:
                self._retry_state.record_attempt()
                attempt_started = time.time()
                http_status: int | None = None
                conn_error = False
                exc_caught: Exception | None = None
                err_body_for_retry = ""
                retry_after_s = 0.0
                try:
                    async with httpx.AsyncClient(
                            timeout=self._timeout, transport=self._transport) as client:
                        async with client.stream(
                                "POST", self._attempt_url,
                                headers=self._attempt_headers,
                                json=self._attempt_body) as r:
                            http_status = r.status_code
                            if r.status_code >= 400:
                                err_body_for_retry = (
                                    await r.aread()).decode("utf-8", "replace")
                                ra = (r.headers.get("retry-after")
                                      or r.headers.get("Retry-After"))
                                try:
                                    retry_after_s = float(ra) if ra else 0.0
                                except (TypeError, ValueError):
                                    retry_after_s = 0.0
                                # Fall through to the policy check below.
                            else:
                                # Success — push STATUS then body chunks.
                                # From here, retry is impossible.
                                await chunk_q.put(
                                    (_STATUS, (r.status_code, None)))
                                async for chunk in r.aiter_raw():
                                    produced_chunk = True
                                    await chunk_q.put((None, chunk))
                                await chunk_q.put((_SENTINEL, None))
                                return
                except Exception as e:  # noqa: BLE001
                    conn_error = True
                    exc_caught = e
                # Failure path.
                if produced_chunk:
                    if exc_caught is not None:
                        raise exc_caught
                    await chunk_q.put(
                        (_STATUS, (http_status or 0, err_body_for_retry)))
                    await chunk_q.put((_SENTINEL, None))
                    return
                action = retry_policy.classify_failure(
                    route=self._attempt_decision.route,
                    status=http_status,
                    is_connection_error=conn_error,
                    is_compaction=self._attempt_decision.is_compaction,
                    attempts_used=self._retry_state.attempts_used,
                    max_total_retries=self._cfg.max_total_retries_per_request,
                    upstream_retry_count=self._cfg.upstream_retry_count,
                    retry_on_local_4xx_escalate_frontier=(
                        self._cfg.retry_on_local_4xx_escalate_frontier),
                    retry_on_frontier_4xx=self._cfg.retry_on_frontier_4xx,
                    retry_after_s=retry_after_s,
                )
                self._retry_state.last_action = action
                if action.decision == "propagate":
                    if action.escalate_flag_reason:
                        self._flag_force_frontier(action.escalate_flag_reason)
                    if exc_caught is not None:
                        raise exc_caught
                    await chunk_q.put(
                        (_STATUS, (http_status or 0, err_body_for_retry)))
                    await chunk_q.put((_SENTINEL, None))
                    return
                # retry_same / retry_escalate
                self._apply_retry_action(action, http_status, attempt_started,
                                         conn_error)
                if action.backoff_s > 0:
                    try:
                        await asyncio.sleep(action.backoff_s)
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            # Stall watchdog (or shutdown) cancelled us — surface as
            # a synthetic exception via the queue so the consumer can
            # emit a clean terminator. We MUST NOT re-raise here: the
            # producer is fire-and-forget; the consumer is the one with
            # the live SSE socket and must own termination.
            try:
                elapsed = _stall.seconds_since_event(self._proj_sid)
            except Exception:  # noqa: BLE001
                elapsed = None
            synthetic = _stall.StallCancelledError(
                "stall_watchdog_cancelled_relay",
                proj_sid=self._proj_sid,
                conv_sid=self._conv_sid,
                elapsed_silent_s=elapsed,
            )
            try:
                await chunk_q.put((_ERR, synthetic))
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            await chunk_q.put((_ERR, exc))
        else:
            await chunk_q.put((_SENTINEL, None))

    def _flag_force_frontier(self, reason: str) -> None:
        try:
            from . import empty_response_guard as _erg
            _erg.force_next_to_frontier(self._erg_key, reason)
        except Exception:  # noqa: BLE001
            pass

    def _apply_retry_action(
        self,
        action: "retry_policy.RetryAction",
        http_status: int | None,
        attempt_started: float,
        conn_error: bool,
    ) -> None:
        """Mutate per-attempt state for the next loop iteration, log the
        retry, and reset the stall timer."""
        new_url = self._attempt_url
        new_headers = self._attempt_headers
        new_decision = self._attempt_decision
        if action.decision == "retry_escalate":
            esc_url, _esc_headers_proto, _b, esc_decision, _bk = (
                self._build_frontier_retry_target(
                    None, self._attempt_body, action.reason))
            # Rebuild Authorization for the frontier backend so a
            # local-backend bearer (e.g. LMStudio's sk-*) doesn't leak
            # to chatgpt.com and trigger a 401 at bytes_out=0.
            # Preserves codex routing headers (openai-beta, x-codex-*).
            merged = dict(self._attempt_headers)
            merged["Content-Type"] = "application/json"
            merged.setdefault("Accept", "text/event-stream")
            fb_key = self._resolve_api_key(self._cfg.frontier, None)
            if fb_key:
                merged["Authorization"] = (
                    fb_key if fb_key.lower().startswith(("bearer ", "basic "))
                    else f"Bearer {fb_key}")
            else:
                merged.pop("Authorization", None)
            new_url = esc_url
            new_headers = merged
            new_decision = esc_decision
            if action.escalate_flag_reason:
                self._flag_force_frontier(action.escalate_flag_reason)
            self._retry_state.record_escalation()
        self._log("retry_attempted",
                  session=self._proj_sid,
                  attempt_number=self._retry_state.attempts_used,
                  original_status=http_status,
                  retry_target=action.decision,
                  original_url=self._attempt_url,
                  new_url=new_url,
                  reason=action.reason,
                  request_id=self._request_id,
                  conn_error=conn_error,
                  elapsed_s=round(time.time() - attempt_started, 3))
        # Reset stall watchdog's last-event timestamp on the retry
        # boundary — without this, the silent retry can run out the
        # threshold window before the watchdog ever fires.
        try:
            _stall.mark_event(self._proj_sid, conv_sid=self._conv_sid)
        except Exception:  # noqa: BLE001
            pass
        self._attempt_url = new_url
        self._attempt_headers = new_headers
        self._attempt_decision = new_decision


# ─── StreamConsumer ───────────────────────────────────────────────────────


class StreamConsumer:
    """Reads tagged items from chunk_q and yields SSE bytes to codex.

    Responsibilities:
      - Idle keepalive: emit `: tinyctx keepalive\\n\\n` every
        `keepalive_interval` seconds when the queue is silent.
      - Translator dispatch: feed each chunk through the optional
        StreamTranslator / ChatToResponsesTranslator.
      - mark_event on each chunk (and on STATUS) so the stall watchdog
        knows the stream is live.
      - Soft-completion accumulator: forward each chunk into the
        soft_completion buffer for the post-stream classifier to read.
      - stream-rewrite intercept: hold back the response.completed
        marker so the outer orchestrator can decide whether to inject
        a synthetic continuation event before flushing.

    Errors propagate via the queue's _ERR tag — the consumer raises
    them, and the outer orchestrator's try/except emits the terminator."""

    def __init__(
        self,
        *,
        chunk_q: asyncio.Queue,
        translator: StreamTranslator | ChatToResponsesTranslator | None,
        proj_sid: str,
        conv_sid: str | None,
        keepalive_interval: float,
        capture_outgoing: Callable[[bytes], bytes],
        intercept_completed: Callable[[bytes], bytes],
        cfg: "Config",
        log: Callable[..., None],
        url: str,
        on_status_error: Callable[[int, str], None],
    ):
        self._q = chunk_q
        self._translator = translator
        self._proj_sid = proj_sid
        self._conv_sid = conv_sid
        self._keepalive_interval = keepalive_interval
        self._capture_outgoing = capture_outgoing
        self._intercept_completed = intercept_completed
        self._cfg = cfg
        self._log = log
        self._url = url
        self._on_status_error = on_status_error
        # Output stats — relay_stream reads these after the consumer ends.
        self.bytes_out = 0
        self.status = 200
        self.keepalives_emitted = 0
        self.upstream_failed = False
        self.upstream_failure_msg = ""

    async def yield_to_client(self) -> AsyncIterator[bytes]:
        """Yield SSE bytes to codex, with idle keepalives. Initial
        keepalive is the orchestrator's responsibility (so it can fire
        BEFORE the producer task is even scheduled)."""
        while True:
            try:
                tag, payload = await asyncio.wait_for(
                    self._q.get(), timeout=self._keepalive_interval)
            except asyncio.TimeoutError:
                yield b": tinyctx keepalive\n\n"
                self.keepalives_emitted += 1
                continue
            if tag is _SENTINEL:
                break
            if tag is _ERR:
                raise payload  # type: ignore[misc]
            if tag is _STATUS:
                status_code, err_body = payload
                self.status = status_code
                if self._cfg.stall_watchdog_enabled:
                    try:
                        _stall.mark_event(self._proj_sid, conv_sid=self._conv_sid)
                    except Exception:  # noqa: BLE001
                        pass
                if err_body is not None:
                    self._on_status_error(status_code, err_body)
                    self.upstream_failed = True
                    self.upstream_failure_msg = (
                        f"upstream {status_code}: {err_body[:200]}")
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'status': status_code, 'body': err_body[:2000]})}"
                        f"\n\n").encode()
                continue
            # tag is None → real response-body chunk
            for out in self._handle_chunk(payload):
                yield out
        # Drain translator flush — but only on clean success path.
        if self._translator is not None and not self.upstream_failed:
            for out in self._translator.flush():
                out_bytes = self._intercept_completed(out)
                if out_bytes:
                    yield self._capture_outgoing(out_bytes)

    def _handle_chunk(self, payload: bytes) -> "list[bytes]":
        self.bytes_out += len(payload)
        if self._cfg.stall_watchdog_enabled:
            try:
                _stall.mark_event(self._proj_sid, conv_sid=self._conv_sid)
            except Exception:  # noqa: BLE001
                pass
        # Soft-completion buffer: accumulate raw upstream bytes. The
        # behavioral classifier runs ONCE at stream end so the hot path
        # stays cheap; never decide mid-stream.
        if self._cfg.soft_completion_gate_enabled:
            try:
                from . import soft_completion as _sc
                _sc.accumulate_chunk(self._proj_sid, payload)
            except Exception:  # noqa: BLE001
                pass
        out_list: list[bytes] = []
        if self._translator is None:
            out_bytes = self._intercept_completed(payload)
            if out_bytes:
                out_list.append(self._capture_outgoing(out_bytes))
        else:
            for out in self._translator.feed(payload):
                out_bytes = self._intercept_completed(out)
                if out_bytes:
                    out_list.append(self._capture_outgoing(out_bytes))
        return out_list


# ─── Public orchestrator: relay_stream ────────────────────────────────────


# These are pulled in lazily inside relay_stream so the module-level
# import graph stays clean and unit tests can patch the proxy bindings
# without dragging the proxy import.


async def relay_stream(
    *,
    chunk_q: asyncio.Queue,
    producer: StreamProducer,
    consumer: StreamConsumer,
    supervisor: StallSupervisor,
    keepalive_interval: float,
) -> AsyncIterator[bytes]:
    """Wire producer + consumer + supervisor together and yield bytes
    to codex. The caller (proxy._stream_proxy) owns the surrounding
    forensics/terminator/post-stream logic; this function is the
    minimum kernel that needs all three components live at once.

    Contract:
      - Emits exactly one initial keepalive comment frame BEFORE
        scheduling the producer task — codex.app's parser disconnects
        after ~60s of zero bytes, and the producer's first byte can
        easily exceed that on a cold-start.
      - On exit (success OR exception), the producer task is awaited
        or cancelled and the supervisor is unregistered.
      - Exceptions from the consumer (incl. StallCancelledError raised
        from the queue) propagate to the caller unchanged.
    """
    # Initial keepalive: ensure codex sees activity within the first
    # event loop turn, independent of upstream latency.
    yield b": tinyctx keepalive\n\n"
    consumer.keepalives_emitted += 1

    producer_task = asyncio.create_task(producer.run(chunk_q))
    supervisor.register(producer_task)
    try:
        async for out in consumer.yield_to_client():
            yield out
    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        supervisor.unregister()


# ─── Terminator helper ────────────────────────────────────────────────────


def build_terminator_event(message: str, model: str | None,
                           *, status_label: str = "incomplete") -> bytes:
    """Build a synthetic response.completed SSE event so codex.app's
    parser doesn't raise "stream closed before response.completed"
    when we close after an `event: error`. Status=incomplete keeps the
    failure surfaced correctly."""
    rid = "resp_" + uuid4().hex[:24]
    payload = {
        "type": "response.completed",
        "response": {
            "id": rid,
            "object": "response",
            "model": model or "tinyctx",
            "status": status_label,
            "incomplete_details": {"reason": "tinyctx_proxy_terminator",
                                   "message": message[:500]},
            "output": [],
        },
    }
    return f"event: response.completed\ndata: {json.dumps(payload)}\n\n".encode()
