"""Tests for the unified retry policy (retry_policy.py) and its wiring
into the proxy's _forward / _stream_proxy hot paths.

Run:
  uv run pytest tests/test_proxy_retry.py -x -v
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from tinyctx import retry_policy
from tinyctx.retry_policy import (
    RequestRetryState,
    RetryAction,
    classify_failure,
)


# ─── classify_failure pure unit tests ─────────────────────────────────────


class TestClassifyFailure:
    """Pure classifier tests — no I/O, no proxy state."""

    def _base_kwargs(self, **overrides):
        d = dict(
            route="local",
            status=200,
            is_connection_error=False,
            is_compaction=False,
            attempts_used=1,
            max_total_retries=3,
            upstream_retry_count=1,
        )
        d.update(overrides)
        return d

    def test_local_400_escalates_to_frontier(self):
        a = classify_failure(**self._base_kwargs(status=400))
        assert a.decision == "retry_escalate"
        assert "400" in a.reason
        assert a.escalate_flag_reason.startswith("retry_escalate_4xx_400")

    def test_local_422_escalates_to_frontier(self):
        a = classify_failure(**self._base_kwargs(status=422))
        assert a.decision == "retry_escalate"

    def test_local_401_propagates_permanent(self):
        a = classify_failure(**self._base_kwargs(status=401))
        assert a.decision == "propagate"
        assert "permanent" in a.reason

    def test_local_403_propagates_permanent(self):
        a = classify_failure(**self._base_kwargs(status=403))
        assert a.decision == "propagate"

    def test_local_404_propagates_permanent(self):
        a = classify_failure(**self._base_kwargs(status=404))
        assert a.decision == "propagate"

    def test_local_500_retries_same_then_escalates(self):
        # first failure → retry_same
        a1 = classify_failure(**self._base_kwargs(status=500, attempts_used=1))
        assert a1.decision == "retry_same"
        # second failure → retry_escalate
        a2 = classify_failure(**self._base_kwargs(status=500, attempts_used=2))
        assert a2.decision == "retry_escalate"

    def test_local_429_retries_same_with_backoff(self):
        a = classify_failure(**self._base_kwargs(
            status=429, attempts_used=1, retry_after_s=2.0))
        assert a.decision == "retry_same"
        assert 0 < a.backoff_s <= 5

    def test_local_429_caps_backoff_at_5s(self):
        a = classify_failure(**self._base_kwargs(
            status=429, attempts_used=1, retry_after_s=600.0))
        assert a.backoff_s <= 5.0

    def test_local_connection_error_retries_same_then_escalates(self):
        a1 = classify_failure(**self._base_kwargs(
            status=None, is_connection_error=True, attempts_used=1))
        assert a1.decision == "retry_same"
        a2 = classify_failure(**self._base_kwargs(
            status=None, is_connection_error=True, attempts_used=2))
        assert a2.decision == "retry_escalate"

    def test_frontier_4xx_propagates_by_default(self):
        a = classify_failure(**self._base_kwargs(
            route="frontier", status=400, attempts_used=1))
        assert a.decision == "propagate"
        # Should still mark session for next-turn frontier routing
        assert a.escalate_flag_reason

    def test_frontier_4xx_retries_when_opt_in(self):
        a = classify_failure(**self._base_kwargs(
            route="frontier", status=400, attempts_used=1,
        ), retry_on_frontier_4xx=True)
        assert a.decision == "retry_same"

    def test_frontier_5xx_retries_same_then_propagates(self):
        a1 = classify_failure(**self._base_kwargs(
            route="frontier", status=500, attempts_used=1))
        assert a1.decision == "retry_same"
        a2 = classify_failure(**self._base_kwargs(
            route="frontier", status=500, attempts_used=2))
        assert a2.decision == "propagate"

    def test_frontier_connection_error_retries_same_then_propagates(self):
        a1 = classify_failure(**self._base_kwargs(
            route="frontier", status=None, is_connection_error=True,
            attempts_used=1))
        assert a1.decision == "retry_same"
        a2 = classify_failure(**self._base_kwargs(
            route="frontier", status=None, is_connection_error=True,
            attempts_used=2))
        assert a2.decision == "propagate"

    def test_max_total_retries_hard_cap(self):
        # attempts_used >= cap → propagate regardless of error type
        a = classify_failure(**self._base_kwargs(
            status=500, attempts_used=3, max_total_retries=3))
        assert a.decision == "propagate"
        assert "max_total_retries" in a.reason

    def test_compaction_never_retried(self):
        a = classify_failure(**self._base_kwargs(
            status=500, is_compaction=True))
        assert a.decision == "propagate"
        assert "compaction" in a.reason

    def test_compaction_never_retried_even_on_connection_error(self):
        a = classify_failure(**self._base_kwargs(
            status=None, is_connection_error=True, is_compaction=True))
        assert a.decision == "propagate"


# ─── proxy _forward non-stream retry wiring ──────────────────────────────


@pytest.fixture
def proxy_module(monkeypatch, tmp_path):
    """Import proxy fresh with a tmp log dir and no config file."""
    monkeypatch.setenv("TINYCTX_VERBOSE", "0")
    monkeypatch.setenv("TINYCTX_CONFIG", "/dev/null")
    monkeypatch.setenv("TINYCTX_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TINYCTX_LOCAL_BASE_URL", "http://local.test/v1")
    monkeypatch.setenv("TINYCTX_LOCAL_WIRE_API", "responses")
    monkeypatch.setenv("TINYCTX_LOCAL_MODEL", "qwen-test")
    monkeypatch.setenv("TINYCTX_FRONTIER_BASE_URL", "http://frontier.test/v1")
    monkeypatch.setenv("TINYCTX_FRONTIER_MODEL", "gpt-test")
    import sys
    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod
    return proxy_mod


def _make_mock_post(responses: list):
    """Given a list of (status_code, body_dict) tuples (or Exception
    instances), return an async function suitable for monkeypatching
    httpx.AsyncClient.post that consumes them in order, recording the
    target URL of each attempt."""
    state = {"calls": [], "idx": 0}

    async def _post(self, url, *args, **kwargs):
        state["calls"].append(url)
        if state["idx"] >= len(responses):
            raise RuntimeError(f"unexpected extra call to {url}")
        r = responses[state["idx"]]
        state["idx"] += 1
        if isinstance(r, Exception):
            raise r
        status_code, body = r
        return httpx.Response(
            status_code=status_code,
            json=body,
            request=httpx.Request("POST", url),
        )

    return _post, state


class TestForwardRetryNonStream:
    """End-to-end retry tests against _forward's non-stream branch."""

    @pytest.mark.asyncio
    async def test_local_400_escalates_to_frontier(self, proxy_module):
        responses = [
            (400, {"error": {"message": "schema mismatch"}}),  # local 400
            (200, {"id": "resp", "output": [], "tag": "frontier_ok"}),  # frontier 200
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-a")
            result = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json", "Authorization": "Bearer x"},
                {"model": "qwen-test", "input": [{"role": "user", "content": "hi"}]},
                is_stream=False,
                sid="sess-a",
                decision=decision,
                trace=trace,
                conv_sid="conv-a",
            )
        assert len(state["calls"]) == 2
        assert "local.test" in state["calls"][0]
        assert "frontier.test" in state["calls"][1]
        # success → 200
        assert result.status_code == 200
        # force_next_to_frontier flag should be set on conv-a
        from tinyctx import empty_response_guard as _erg
        info = _erg.consume_force_frontier("conv-a")
        assert info is not None
        assert "4xx" in info["reason"] or "escalate" in info["reason"]

    @pytest.mark.asyncio
    async def test_local_500_retries_same_then_escalates(self, proxy_module):
        responses = [
            (500, {"error": "internal"}),  # local 500 #1
            (500, {"error": "internal"}),  # local 500 #2 (retry_same)
            (200, {"id": "resp", "output": [], "tag": "frontier_ok"}),  # frontier escalate
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-b")
            result = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": [{"role": "user", "content": "hi"}]},
                is_stream=False,
                sid="sess-b",
                decision=decision,
                trace=trace,
                conv_sid="conv-b",
            )
        # 3 attempts: local local frontier
        assert len(state["calls"]) == 3
        assert "local.test" in state["calls"][0]
        assert "local.test" in state["calls"][1]
        assert "frontier.test" in state["calls"][2]
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_local_connection_error_retries_then_escalates(self, proxy_module):
        responses = [
            httpx.ConnectError("connection refused"),  # local connect err
            httpx.ConnectError("connection refused"),  # local connect err
            (200, {"id": "resp", "tag": "frontier_ok"}),  # frontier escalate
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-c")
            result = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="sess-c",
                decision=decision,
                trace=trace,
                conv_sid="conv-c",
            )
        assert len(state["calls"]) == 3
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_frontier_4xx_no_auto_retry(self, proxy_module):
        responses = [
            (400, {"error": "bad request"}),
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("frontier", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-d")
            result = await proxy_module._forward(
                "http://frontier.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "gpt-test", "input": []},
                is_stream=False,
                sid="sess-d",
                decision=decision,
                trace=trace,
                conv_sid="conv-d",
            )
        assert len(state["calls"]) == 1, \
            "frontier 4xx must NOT auto-retry by default"
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_frontier_5xx_retries_same(self, proxy_module):
        responses = [
            (500, {"error": "internal"}),  # frontier 500
            (200, {"id": "resp", "tag": "frontier_ok"}),  # success on retry
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("frontier", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-e")
            result = await proxy_module._forward(
                "http://frontier.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "gpt-test", "input": []},
                is_stream=False,
                sid="sess-e",
                decision=decision,
                trace=trace,
                conv_sid="conv-e",
            )
        assert len(state["calls"]) == 2
        assert "frontier.test" in state["calls"][0]
        assert "frontier.test" in state["calls"][1]
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_retry_count_cap_honored(self, proxy_module, monkeypatch):
        # Patch the cap on the LIVE module config so policy fires earlier
        monkeypatch.setattr(proxy_module.CFG, "max_total_retries_per_request", 2)
        responses = [
            (500, {"error": "1"}),  # local 500
            (500, {"error": "2"}),  # local 500 — but cap=2 means this is last attempt
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-cap")
            result = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="sess-cap",
                decision=decision,
                trace=trace,
                conv_sid="conv-cap",
            )
        # With cap=2 we make 2 attempts then propagate the last 500
        assert len(state["calls"]) == 2
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_compaction_never_retried(self, proxy_module):
        responses = [
            (500, {"error": "internal"}),  # compaction handoff fails
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "compaction", is_compaction=True)
            trace = RequestTrace(session_id="sess-comp")
            result = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="sess-comp",
                decision=decision,
                trace=trace,
                conv_sid="conv-comp",
            )
        assert len(state["calls"]) == 1, \
            "compaction requests must NEVER retry"
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_retry_attempted_event_logged(self, proxy_module, tmp_path,
                                                 monkeypatch):
        # Force verbose so the log file gets written
        monkeypatch.setattr(proxy_module.CFG, "verbose", True)
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(proxy_module.CFG, "log_dir", log_dir)
        responses = [
            (400, {"error": "bad"}),
            (200, {"id": "resp", "output": []}),
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="sess-log")
            await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="sess-log",
                decision=decision,
                trace=trace,
                conv_sid="conv-log",
            )
        # Read today's log file and look for retry_attempted event
        import time as _t
        log_file = log_dir / f"tinyctx-{_t.strftime('%Y%m%d')}.jsonl"
        text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        assert "retry_attempted" in text, \
            f"expected retry_attempted in log, got:\n{text[:500]}"
        # Parse line for fields
        retry_lines = [json.loads(l) for l in text.splitlines()
                       if "retry_attempted" in l]
        assert retry_lines
        ev = retry_lines[0]
        assert ev["event"] == "retry_attempted"
        assert ev["attempt_number"] == 1
        assert ev["retry_target"] == "retry_escalate"
        assert "local" in ev["original_url"]
        assert "frontier" in ev["new_url"]

    @pytest.mark.asyncio
    async def test_force_next_to_frontier_set_on_escalation(self, proxy_module):
        responses = [
            (400, {"error": "schema mismatch"}),  # local 400 → escalate
            (200, {"id": "ok"}),
        ]
        post_fn, _state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="sess-flag",
                decision=decision,
                trace=RequestTrace(session_id="sess-flag"),
                conv_sid="conv-flag",
            )
        from tinyctx import empty_response_guard as _erg
        info = _erg.consume_force_frontier("conv-flag")
        assert info is not None
        # cleanup other sessions
        _erg.reset_state()


# ─── proxy _stream_proxy retry wiring ────────────────────────────────────


class _MockStreamCtx:
    """Async context manager mimicking httpx response stream returned
    by `AsyncClient.stream("POST", ...)`. Yields configured chunks on
    aiter_raw() and reports the configured status_code."""

    def __init__(self, status_code: int, *, err_body: bytes = b"",
                 chunks: list[bytes] | None = None,
                 headers: dict | None = None):
        self.status_code = status_code
        self._err_body = err_body
        self._chunks = chunks or []
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return self._err_body

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


def _make_mock_stream(scripts: list):
    """scripts: list where each entry is either
      ("ok", [b"chunk1", b"chunk2"])  → 200 stream
      ("err", status, body_bytes)     → error status with body
      ("exc", Exception)              → raise inside the context

    Returns (stream_fn, state) where stream_fn replaces
    httpx.AsyncClient.stream.
    """
    state = {"calls": [], "idx": 0}

    def stream_fn(self, method, url, **kwargs):
        state["calls"].append(url)
        if state["idx"] >= len(scripts):
            raise RuntimeError(f"unexpected extra stream call to {url}")
        sc = scripts[state["idx"]]
        state["idx"] += 1
        kind = sc[0]
        if kind == "ok":
            chunks = sc[1]
            return _MockStreamCtx(200, chunks=chunks)
        if kind == "err":
            status, body = sc[1], sc[2]
            return _MockStreamCtx(status, err_body=body)
        if kind == "exc":
            exc = sc[1]
            class _Raising:
                async def __aenter__(s):
                    raise exc
                async def __aexit__(s, *a):
                    return False
            return _Raising()
        raise AssertionError(f"unknown script kind {kind}")

    return stream_fn, state


async def _drain_stream(streaming_response):
    """Walk a StreamingResponse to collect all yielded bytes."""
    chunks = []
    async for chunk in streaming_response.body_iterator:
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(bytes(chunk))
        else:
            chunks.append(str(chunk).encode())
    return b"".join(chunks)


class TestStreamProxyRetry:

    @pytest.mark.asyncio
    async def test_stream_local_400_escalates_to_frontier(self, proxy_module,
                                                            monkeypatch):
        monkeypatch.setattr(proxy_module.CFG, "stream_keepalive_interval_s", 1.0)
        scripts = [
            ("err", 400, b'{"error":"schema mismatch"}'),
            ("ok", [b'event: response.completed\ndata: {"type":"response.completed"}\n\n']),
        ]
        stream_fn, state = _make_mock_stream(scripts)
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-stream-a")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-stream-a",
                decision=decision,
                trace=trace,
                conv_sid="conv-stream-a",
            )
            body_bytes = await _drain_stream(sr)
        # Two attempts: local then frontier
        assert len(state["calls"]) == 2
        assert "local.test" in state["calls"][0]
        assert "frontier.test" in state["calls"][1]
        # success body forwarded to client (not the 400 error)
        assert b"response.completed" in body_bytes
        # No `event: error` from the 400 should have been streamed to
        # client (the retry swallowed it before producing any bytes)
        assert b'"status": 400' not in body_bytes

    @pytest.mark.asyncio
    async def test_stream_compaction_never_retried(self, proxy_module,
                                                     monkeypatch):
        monkeypatch.setattr(proxy_module.CFG, "stream_keepalive_interval_s", 1.0)
        scripts = [
            ("err", 500, b'{"error":"internal"}'),
        ]
        stream_fn, state = _make_mock_stream(scripts)
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "compaction", is_compaction=True)
            trace = RequestTrace(session_id="s-comp")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-comp",
                decision=decision,
                trace=trace,
                conv_sid="conv-stream-comp",
            )
            await _drain_stream(sr)
        assert len(state["calls"]) == 1, \
            "compaction stream must NEVER retry"


# ─── stall-cancel: watchdog-triggered relay cancellation ─────────────────


class _HangingStreamCtx:
    """Mock httpx stream that opens with 200 and then HANGS forever on
    aiter_raw — simulates the upstream-silence stall scenario the
    watchdog is designed to break."""

    def __init__(self, started_event: asyncio.Event):
        self.status_code = 200
        self.headers = {}
        self._started = started_event

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return b""

    async def aiter_raw(self):
        # Signal that we've opened — test can then trigger the cancel.
        self._started.set()
        # Sleep forever; respects cancellation via the next await.
        await asyncio.sleep(3600)
        if False:
            yield b""


class TestStreamProxyStallCancel:
    """End-to-end: stall watchdog cancels the in-flight relay producer,
    consumer emits a clean SSE terminator with status=incomplete, and
    force_next_to_frontier is set on the conv key so codex's follow-up
    turn routes to frontier."""

    @pytest.mark.asyncio
    async def test_stall_cancel_emits_terminator_and_sets_force_frontier(
            self, proxy_module, monkeypatch):
        from tinyctx import stall_watchdog as _sw
        from tinyctx import empty_response_guard as _erg
        # Short keepalive so the consumer wakes quickly enough.
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 0.05)
        _sw.reset_state()
        _erg.reset_state()

        started = asyncio.Event()

        def stream_fn(self, method, url, **kwargs):
            return _HangingStreamCtx(started)

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-stall")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-stall",
                decision=decision,
                trace=trace,
                conv_sid="conv-stall",
            )

            collected = bytearray()

            async def _drain():
                async for chunk in sr.body_iterator:
                    if isinstance(chunk, (bytes, bytearray)):
                        collected.extend(chunk)
                    else:
                        collected.extend(str(chunk).encode())

            drain_task = asyncio.create_task(_drain())

            # Wait for the producer to be live and registered.
            await asyncio.wait_for(started.wait(), timeout=2.0)
            # Give it a beat so register_task ran on the producer-create
            # line right after asyncio.create_task.
            await asyncio.sleep(0.05)

            # Simulate the watchdog firing: cancel the registered task.
            assert _sw.get_active_task("s-stall") is not None
            cancelled = _sw.cancel_active_task("s-stall")
            assert cancelled is True
            # And set the force_next flag the real on_stall callback
            # would set (test the proxy path, not the watchdog wiring).
            _erg.force_next_to_frontier("conv-stall", "mid_stream_stall")

            try:
                await asyncio.wait_for(drain_task, timeout=3.0)
            except asyncio.CancelledError:
                pass

        body_bytes = bytes(collected)
        # Structurally valid SSE close — codex's parser needs
        # response.completed to accept the stream.
        assert b"response.completed" in body_bytes
        # Status=incomplete signals "this attempt failed, please retry"
        # to codex.
        assert b"incomplete" in body_bytes
        # The synthetic stall_cancelled marker should appear in the
        # error event.
        assert b"stall_cancelled" in body_bytes
        # force_next_to_frontier flag is consumable on the conv key.
        info = _erg.consume_force_frontier("conv-stall")
        assert info is not None
        assert "mid_stream_stall" in info["reason"]

    @pytest.mark.asyncio
    async def test_register_unregister_in_stream_lifecycle(
            self, proxy_module, monkeypatch):
        """After a normal-success stream completes, the registered
        producer task is unregistered — leaks would let stale handles
        confuse the next stream's cancel call."""
        from tinyctx import stall_watchdog as _sw
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 0.5)
        _sw.reset_state()

        scripts = [
            ("ok", [b'event: response.completed\ndata: {"type":"response.completed"}\n\n']),
        ]
        stream_fn, _state = _make_mock_stream(scripts)
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-lifecycle")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-lifecycle",
                decision=decision,
                trace=trace,
                conv_sid="conv-lifecycle",
            )
            await _drain_stream(sr)
        # After the stream finished, the registry must be empty.
        assert _sw.get_active_task("s-lifecycle") is None


# ─── retry policy state machine ──────────────────────────────────────────


class TestRequestRetryState:

    def test_record_attempt_increments(self):
        s = RequestRetryState()
        assert s.attempts_used == 0
        s.record_attempt()
        assert s.attempts_used == 1
        s.record_attempt()
        assert s.attempts_used == 2

    def test_record_escalation_independent(self):
        s = RequestRetryState()
        s.record_attempt()
        s.record_escalation()
        assert s.attempts_used == 1
        assert s.escalations_used == 1


# ─── stall-timer reset on retry boundary ─────────────────────────────────
#
# Diagnosed 2026-05-12: a request was wedged with phase=stalled for ~80s
# with no events. Sequence: LMStudio 400 at T+14.6s → retry_escalate to
# frontier → frontier silently awaits first byte. The LAST mark_event was
# at T+14.6s (the 400 arrival), so stall_watchdog wouldn't fire until
# T+14.6+180 = T+194.6s. Codex's own client-side idle timeout fires first
# (~60-120s) and the user sees "task interrupted".
#
# Fix: every time the retry layer escalates or retries, call mark_event
# to reset the 180s stall window for the NEW upstream attempt.


class TestRetryResetsStallTimer:
    """Each retry attempt (same backend or escalate) must reset the stall
    watchdog's last-event timestamp via mark_event(proj_sid, conv_sid).
    Otherwise the watchdog won't fire on the new attempt's silence until
    `(time-of-previous-failure + 180s)` — far too late."""

    @pytest.mark.asyncio
    async def test_forward_nonstream_retry_calls_mark_event(self, proxy_module):
        from tinyctx import stall_watchdog as _sw
        _sw.reset_state()
        # Spy on mark_event.
        calls: list[tuple[str, str | None]] = []
        real_mark = _sw.mark_event

        def spy_mark(proj_sid, conv_sid=None):
            calls.append((proj_sid, conv_sid))
            return real_mark(proj_sid, conv_sid=conv_sid)

        responses = [
            (400, {"error": {"message": "schema mismatch"}}),  # local 400
            (200, {"id": "resp", "output": [], "tag": "frontier_ok"}),
        ]
        post_fn, state = _make_mock_post(responses)
        with patch.object(httpx.AsyncClient, "post", post_fn), \
             patch.object(proxy_module._stall, "mark_event", spy_mark):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-retry-mark")
            await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=False,
                sid="s-retry-mark",
                decision=decision,
                trace=trace,
                conv_sid="conv-retry-mark",
            )
        # Two upstream attempts (local 400 → frontier escalate).
        assert len(state["calls"]) == 2
        # mark_event must have been called for proj_sid=s-retry-mark
        # at least once on the retry boundary, with conv_sid=conv-retry-mark.
        mark_calls = [(p, c) for (p, c) in calls if p == "s-retry-mark"]
        assert mark_calls, (
            "expected mark_event(proj_sid='s-retry-mark', ...) to be called "
            "on the retry boundary so the stall watchdog's 180s window "
            f"resets for the new attempt; got calls={calls}"
        )
        # And the conv_sid must be plumbed through.
        assert any(c == "conv-retry-mark" for (_, c) in mark_calls), (
            f"expected at least one mark_event call with "
            f"conv_sid='conv-retry-mark'; got mark_calls={mark_calls}"
        )

    @pytest.mark.asyncio
    async def test_stream_retry_calls_mark_event(
            self, proxy_module, monkeypatch):
        """The CRITICAL ordering: mark_event must be called BEFORE the
        next upstream stream attempt opens. Without this, the watchdog
        won't fire on the new attempt's silence until
        `(time-of-previous-failure + 180s)` — leaving codex to time out
        first."""
        from tinyctx import stall_watchdog as _sw
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 1.0)
        _sw.reset_state()
        # Shared event log: ("mark", proj, conv) or ("stream", url).
        events: list[tuple] = []
        real_mark = _sw.mark_event

        def spy_mark(proj_sid, conv_sid=None):
            events.append(("mark", proj_sid, conv_sid))
            return real_mark(proj_sid, conv_sid=conv_sid)

        scripts = [
            ("err", 400, b'{"error":"schema mismatch"}'),  # local 400
            ("ok", [b'event: response.completed\ndata: {"type":"response.completed"}\n\n']),
        ]
        stream_fn_inner, state = _make_mock_stream(scripts)

        def spy_stream(self, method, url, **kwargs):
            events.append(("stream", url))
            return stream_fn_inner(self, method, url, **kwargs)

        with patch.object(httpx.AsyncClient, "stream", spy_stream), \
             patch.object(proxy_module._stall, "mark_event", spy_mark):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-stream-retry-mark")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-stream-retry-mark",
                decision=decision,
                trace=trace,
                conv_sid="conv-stream-retry-mark",
            )
            await _drain_stream(sr)
        assert len(state["calls"]) == 2, (
            f"expected 2 upstream stream attempts; got {state['calls']}"
        )
        # Find the indices of the two stream calls and any mark_event
        # call for this proj_sid that falls BETWEEN them.
        stream_idxs = [i for i, e in enumerate(events) if e[0] == "stream"]
        assert len(stream_idxs) == 2
        between = [
            e for e in events[stream_idxs[0] + 1: stream_idxs[1]]
            if e[0] == "mark" and e[1] == "s-stream-retry-mark"
        ]
        assert between, (
            "expected mark_event(proj_sid='s-stream-retry-mark', ...) to "
            "be called AFTER the first (failed) upstream attempt and "
            "BEFORE the retry attempt opens, so the stall watchdog's "
            "180s window resets for the new attempt. "
            f"Got events between attempts: "
            f"{events[stream_idxs[0] + 1: stream_idxs[1]]}"
        )
        # And conv_sid must be plumbed through.
        assert any(c == "conv-stream-retry-mark"
                   for (_, _, c) in between), (
            f"expected the between-attempts mark_event to carry "
            f"conv_sid='conv-stream-retry-mark'; got {between}"
        )


# ─── early keepalive on stream open (pre-first-upstream-byte) ────────────
#
# Diagnosed 2026-05-12 alongside the stall-timer reset bug: even with
# the producer-side keepalive (which fires every keepalive_interval s
# while the queue is idle), at the default interval of 15s codex's
# client-side idle timeout can fire BEFORE the first keepalive ever
# emits. Fix: emit one keepalive frame IMMEDIATELY when the SSE
# response generator opens, so codex sees activity within < 1s of
# response open — independent of how long the upstream takes to send
# its first byte.


class TestEarlyKeepaliveOnStreamOpen:
    """Within the first ~1s of response.open the consumer must observe
    at least one byte (a keepalive comment frame), even if upstream takes
    much longer than keepalive_interval to produce its first byte."""

    @pytest.mark.asyncio
    async def test_initial_keepalive_emitted_before_first_upstream_byte(
            self, proxy_module, monkeypatch):
        from tinyctx import stall_watchdog as _sw
        # Use a LARGE keepalive_interval so the existing
        # "idle-loop keepalive" cannot rescue the test — only an
        # explicit pre-loop initial keepalive can produce bytes within
        # the 1s assertion window.
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 30.0)
        _sw.reset_state()

        started = asyncio.Event()

        class _HangingThenComplete:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, started_event: asyncio.Event):
                self._started = started_event

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                self._started.set()
                # Block for longer than the assertion window.
                await asyncio.sleep(5.0)
                yield (b'event: response.completed\n'
                       b'data: {"type":"response.completed"}\n\n')

        def stream_fn(self, method, url, **kwargs):
            return _HangingThenComplete(started)

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "test", is_compaction=False)
            trace = RequestTrace(session_id="s-early-ka")
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid="s-early-ka",
                decision=decision,
                trace=trace,
                conv_sid="conv-early-ka",
            )

            t_start = time.monotonic()
            first_byte_t: float | None = None
            collected = bytearray()

            async def _drain():
                nonlocal first_byte_t
                async for chunk in sr.body_iterator:
                    b = (chunk if isinstance(chunk, (bytes, bytearray))
                         else str(chunk).encode())
                    if first_byte_t is None and b:
                        first_byte_t = time.monotonic() - t_start
                    collected.extend(b)

            drain_task = asyncio.create_task(_drain())
            try:
                await asyncio.wait_for(drain_task, timeout=8.0)
            except asyncio.CancelledError:
                pass

        assert first_byte_t is not None, (
            "no bytes ever observed by the consumer — drain hung"
        )
        # With KA=30s and upstream-first-byte-delay=5s, an idle-loop
        # keepalive cannot fire within 1s. Only an explicit pre-loop
        # initial keepalive can produce bytes that early.
        assert first_byte_t < 1.0, (
            f"first byte observed at t={first_byte_t:.2f}s — expected "
            f"< 1.0s if initial keepalive fires immediately on stream "
            f"open. With keepalive_interval=30s, this proves no early "
            f"keepalive is being emitted."
        )
        # And the first bytes should be a structurally-valid SSE
        # keepalive comment (lines starting with `:` are SSE comments
        # ignored by spec-compliant parsers).
        first_chunk = bytes(collected).split(b"\n\n", 1)[0]
        assert first_chunk.startswith(b":"), (
            f"first bytes must be an SSE comment frame (starting with "
            f"':'); got: {first_chunk[:80]!r}"
        )
