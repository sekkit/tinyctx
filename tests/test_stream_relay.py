"""Unit tests for tinyctx.stream_relay components.

These cover the StreamProducer / StreamConsumer / StallSupervisor /
relay_stream layer extracted from proxy._stream_proxy in P6. The goal
is to lock in the contract of each component so future refactors
(P7 PostStream extraction, P8+ translator surgery) can move safely.

Integration coverage already exists in test_proxy_retry.py /
test_keepalive.py / test_integration_workflow.py — those drive the
full proxy path. The tests here pin behaviors that are easier to
exercise on the components directly:

  • StreamProducer's retry loop calls retry_policy.classify_failure
  • StreamProducer rebuilds Authorization on retry_escalate
  • StreamProducer pushes a synthetic StallCancelledError on cancel
  • StreamProducer resets the stall timer (mark_event) on every retry
  • StreamConsumer emits idle keepalives when the queue is silent
  • StreamConsumer dispatches chunks through the translator
  • StreamConsumer raises errors that were pushed via _ERR
  • StallSupervisor wires register/unregister correctly
  • relay_stream emits an initial keepalive before any upstream byte
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from tinyctx import stream_relay as sr
from tinyctx import soft_completion as _sc
from tinyctx import stall_watchdog as _sw


# ─── Test doubles ─────────────────────────────────────────────────────────


class _MockStreamCtx:
    """Mimics httpx response stream returned by AsyncClient.stream(POST, ...).
    Same shape as the one in test_proxy_retry.py — duplicated here to
    keep this test file standalone."""

    def __init__(self, status_code: int, *, err_body: bytes = b"",
                 chunks: "list[bytes] | None" = None,
                 headers: "dict[str, str] | None" = None):
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


class _FakeBackend:
    base_url = "http://fake.test/v1"
    model = "fake"
    api_key_env = ""


class _FakeCfg:
    """Minimum config surface StreamProducer/StreamConsumer reads."""
    max_total_retries_per_request = 3
    upstream_retry_count = 1
    retry_on_local_4xx_escalate_frontier = True
    retry_on_frontier_4xx = False
    stall_watchdog_enabled = False
    soft_completion_gate_enabled = False
    frontier = _FakeBackend()


class _RecordingLog:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))


def _make_decision(route: str = "local", reason: str = "test",
                   is_compaction: bool = False):
    from tinyctx.router import Decision
    return Decision(route, reason, is_compaction=is_compaction)


def _make_scripted_stream(scripts: list):
    """scripts: list where each entry is
      ("ok", [b"chunk1", ...])    → 200 stream
      ("err", status, body_bytes) → error response with body
      ("exc", exception)          → raise on aenter
    Returns (stream_fn, state)."""
    state = {"calls": [], "headers": [], "json": [], "idx": 0}

    def stream_fn(self, method, url, **kwargs):
        state["calls"].append(url)
        state["headers"].append(dict(kwargs.get("headers") or {}))
        state["json"].append(kwargs.get("json"))
        sc = scripts[state["idx"]]
        state["idx"] += 1
        kind = sc[0]
        if kind == "ok":
            return _MockStreamCtx(200, chunks=sc[1])
        if kind == "err":
            return _MockStreamCtx(sc[1], err_body=sc[2])
        if kind == "exc":
            exc = sc[1]

            class _Raising:
                async def __aenter__(self):
                    raise exc

                async def __aexit__(self, *a):
                    return False

            return _Raising()
        raise AssertionError(f"unknown script kind {kind}")

    return stream_fn, state


def _make_producer(*, url="http://local.test/v1/responses",
                   decision=None, cfg=None,
                   build_frontier_retry_target=None,
                   resolve_api_key=None) -> sr.StreamProducer:
    cfg = cfg or _FakeCfg()
    decision = decision or _make_decision("local")
    if build_frontier_retry_target is None:
        def build_frontier_retry_target(headers, body, reason):
            return (
                "http://frontier.test/v1/responses",
                {"X-Frontier": "1"},
                body,
                _make_decision("frontier", reason),
                cfg.frontier,
            )
    if resolve_api_key is None:
        def resolve_api_key(_backend, _codex_auth):
            return "frontier-rebuilt-key"
    return sr.StreamProducer(
        url=url,
        headers={"Authorization": "Bearer local-leaky-key",
                 "X-Codex-Session-Id": "s1"},
        body={"model": "x", "input": []},
        proj_sid="proj-1",
        conv_sid="conv-1",
        decision=decision,
        timeout=httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0),
        transport=None,
        erg_key="conv-1",
        request_id="rid-test",
        cfg=cfg,
        log=_RecordingLog(),
        build_frontier_retry_target=build_frontier_retry_target,
        resolve_api_key=resolve_api_key,
    )


async def _drain_queue(chunk_q: asyncio.Queue) -> list:
    out: list = []
    while True:
        item = await chunk_q.get()
        out.append(item)
        if item[0] is sr._SENTINEL or item[0] is sr._ERR:
            break
    return out


# ─── StreamProducer tests ─────────────────────────────────────────────────


class TestStreamProducer:

    @pytest.mark.asyncio
    async def test_happy_path_streams_chunks_then_sentinel(self):
        scripts = [("ok", [b"chunk-a", b"chunk-b"])]
        stream_fn, state = _make_scripted_stream(scripts)
        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        items = []
        while not chunk_q.empty():
            items.append(chunk_q.get_nowait())
        # status first, then chunks (each tagged None), then sentinel
        assert items[0][0] is sr._STATUS
        assert items[0][1] == (200, None)
        chunk_items = [it for it in items if it[0] is None]
        assert [it[1] for it in chunk_items] == [b"chunk-a", b"chunk-b"]
        assert items[-1][0] is sr._SENTINEL

    @pytest.mark.asyncio
    async def test_local_400_retry_escalates_to_frontier(self):
        scripts = [
            ("err", 400, b'{"error":"schema mismatch"}'),
            ("ok", [b'event: response.completed\ndata: {}\n\n']),
        ]
        stream_fn, state = _make_scripted_stream(scripts)
        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        assert len(state["calls"]) == 2
        assert "local.test" in state["calls"][0]
        assert "frontier.test" in state["calls"][1]
        # No STATUS with err_body was pushed (the 400 was swallowed by retry)
        status_items = [it for it in list(chunk_q._queue)
                        if it[0] is sr._STATUS]
        # Only the SUCCESS status (200, None) should be visible to consumer.
        assert all(it[1] == (200, None) for it in status_items)

    @pytest.mark.asyncio
    async def test_retry_escalate_rebuilds_authorization(self):
        scripts = [
            ("err", 400, b'{"error":"schema mismatch"}'),
            ("ok", [b"ok\n\n"]),
        ]
        stream_fn, state = _make_scripted_stream(scripts)
        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        # First call carries the (leaky) local bearer; second call must
        # carry the rebuilt frontier key, NOT the local one.
        assert state["headers"][0]["Authorization"] == "Bearer local-leaky-key"
        new_auth = state["headers"][1].get("Authorization", "")
        assert "local-leaky-key" not in new_auth
        assert new_auth == "Bearer frontier-rebuilt-key"

    @pytest.mark.asyncio
    async def test_retry_escalate_rewrites_body_model(self):
        scripts = [
            ("err", 400, b'{"error":"schema mismatch"}'),
            ("ok", [b"ok\n\n"]),
        ]
        stream_fn, state = _make_scripted_stream(scripts)

        class _FrontierBackend:
            base_url = "http://frontier.test/v1"
            model = "gpt-5.5"
            api_key_env = ""

        class _Cfg(_FakeCfg):
            frontier = _FrontierBackend()

        def build_frontier_retry_target(headers, body, reason):
            retry_body = dict(body)
            retry_body["model"] = "gpt-5.5"
            return (
                "http://frontier.test/v1/responses",
                {"X-Frontier": "1"},
                retry_body,
                _make_decision("frontier", reason),
                _Cfg.frontier,
            )

        producer = _make_producer(
            cfg=_Cfg(),
            build_frontier_retry_target=build_frontier_retry_target,
        )
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        assert len(state["json"]) == 2
        assert state["json"][0]["model"] == "x"
        assert state["json"][1]["model"] == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_retry_escalate_drops_auth_when_no_frontier_key(self):
        scripts = [
            ("err", 400, b"bad"),
            ("ok", [b"ok"]),
        ]
        stream_fn, state = _make_scripted_stream(scripts)

        def no_key(_backend, _codex_auth):
            return None

        producer = _make_producer(resolve_api_key=no_key)
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        # When no frontier key is resolvable, Authorization must be
        # ABSENT (anything but the leaky local bearer).
        assert "Authorization" not in state["headers"][1]

    @pytest.mark.asyncio
    async def test_compaction_never_retries(self):
        scripts = [("err", 500, b"boom")]
        stream_fn, state = _make_scripted_stream(scripts)
        producer = _make_producer(
            decision=_make_decision("local", "compaction", is_compaction=True))
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        assert len(state["calls"]) == 1
        # The 500 must surface to consumer (no swallow).
        items = list(chunk_q._queue)
        assert any(it[0] is sr._STATUS and it[1][0] == 500 for it in items)

    @pytest.mark.asyncio
    async def test_retry_resets_stall_timer_via_mark_event(self):
        """6b6024d: every retry boundary must call _stall.mark_event so
        the watchdog's countdown references the NEW attempt's open time,
        not the failed attempt's last byte."""
        scripts = [
            ("err", 400, b"bad"),
            ("ok", [b"ok"]),
        ]
        stream_fn, _ = _make_scripted_stream(scripts)
        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(_sw, "mark_event") as mark_event:
            with patch.object(httpx.AsyncClient, "stream", stream_fn):
                await producer.run(chunk_q)
        # At least one mark_event call on the retry boundary.
        assert mark_event.call_count >= 1
        # The proj_sid passed in must match.
        proj_sids_seen = {c.args[0] for c in mark_event.call_args_list}
        assert "proj-1" in proj_sids_seen

    @pytest.mark.asyncio
    async def test_cancellation_pushes_synthetic_stall_error(self):
        """When stall_watchdog cancels the producer task, it MUST push
        a StallCancelledError onto the queue instead of letting the
        CancelledError kill the consumer's generator."""

        started_evt = asyncio.Event()

        class _Hanging:
            status_code = 200
            headers: dict = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                started_evt.set()
                await asyncio.sleep(60)
                if False:
                    yield b""

        def stream_fn(self, method, url, **kwargs):
            return _Hanging()

        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            task = asyncio.create_task(producer.run(chunk_q))
            await asyncio.wait_for(started_evt.wait(), timeout=1.0)
            task.cancel()
            await asyncio.wait_for(task, timeout=1.0)
        # Drain queue — expect a STATUS(200, None) then _ERR with
        # StallCancelledError.
        items = []
        while not chunk_q.empty():
            items.append(chunk_q.get_nowait())
        err_items = [it for it in items if it[0] is sr._ERR]
        assert len(err_items) == 1
        err = err_items[0][1]
        assert isinstance(err, _sw.StallCancelledError)
        assert err.proj_sid == "proj-1"
        assert err.conv_sid == "conv-1"

    @pytest.mark.asyncio
    async def test_stream_relay_survives_cancel_during_synthetic_err_put(self):
        """If task.cancel() fires DURING the synthetic-error put on the
        chunk queue (e.g. queue full + we are awaiting), the consumer
        must still receive _ERR — not hang forever.

        This is the race that explains the production stall_kill (not
        stall_cancelled) observation: the producer enters the cancel
        handler, builds the synthetic StallCancelledError, then a second
        CancelledError lands during `await chunk_q.put(...)` and the
        synthetic never makes it to the consumer. The producer task
        ends in `done()` state, stall_watchdog's `cancel_active_task`
        returns False on next sweep, and `stall_kill` is emitted while
        the consumer hangs on `await chunk_q.get()` indefinitely.

        We trigger the race deterministically by monkeypatching
        `chunk_q.put` so the first call (inside the cancel handler)
        raises `CancelledError`, simulating a cancel that lands during
        the awaitable put.
        """
        started_evt = asyncio.Event()

        class _Hanging:
            status_code = 200
            headers: dict = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                started_evt.set()
                await asyncio.sleep(60)
                if False:
                    yield b""

        def stream_fn(self, method, url, **kwargs):
            return _Hanging()

        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()

        # Wrap chunk_q.put so we can simulate a cancel-during-put race
        # ONLY when the producer is in its cancel handler (after the
        # test pulls `race_armed`). Earlier puts (e.g. STATUS) must
        # behave normally so the producer can reach the hang point.
        real_put = chunk_q.put
        race_armed = {"on": False}

        async def racing_put(item):
            if race_armed["on"]:
                # Race: a cancel lands DURING this await — the synthetic
                # never reaches the queue.
                raise asyncio.CancelledError()
            return await real_put(item)

        chunk_q.put = racing_put  # type: ignore[assignment]

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            task = asyncio.create_task(producer.run(chunk_q))
            await asyncio.wait_for(started_evt.wait(), timeout=1.0)
            # Arm the race: the next `await chunk_q.put(...)` raises
            # CancelledError. This is the put inside the cancel handler
            # (the synthetic StallCancelledError push). The current code
            # uses `await chunk_q.put(...)` which is itself cancellable.
            race_armed["on"] = True
            task.cancel()
            # The producer MUST complete (not hang) within a bounded
            # window even when the synthetic-err put races with cancel.
            await asyncio.wait_for(task, timeout=2.0)

        # CONTRACT: a terminating item (_ERR or _SENTINEL) MUST eventually
        # appear on the queue within a bounded window — otherwise the
        # consumer's `await chunk_q.get()` loop hangs forever. Drain up
        # to a few items, but the cumulative wait must be bounded.
        async def _drain_until_terminator(deadline: float):
            seen: list = []
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                item = await asyncio.wait_for(chunk_q.get(), timeout=remaining)
                seen.append(item)
                if item[0] in (sr._ERR, sr._SENTINEL):
                    return seen

        deadline = asyncio.get_event_loop().time() + 1.0
        try:
            items = await _drain_until_terminator(deadline)
        except asyncio.TimeoutError:
            pytest.fail(
                "consumer would hang forever: producer was cancelled "
                "during synthetic-err put and pushed no terminator "
                "(_ERR/_SENTINEL) onto the queue — this is the "
                "stall_kill production bug.")

        terminator = items[-1]
        # Prefer the synthetic _ERR for clean terminator emission; a
        # _SENTINEL also satisfies the no-hang contract.
        assert terminator[0] in (sr._ERR, sr._SENTINEL), (
            f"unexpected terminator after cancel-during-put: {terminator!r}")
        if terminator[0] is sr._ERR:
            assert isinstance(terminator[1], _sw.StallCancelledError)
            assert terminator[1].proj_sid == "proj-1"

    @pytest.mark.asyncio
    async def test_connection_error_propagates_via_err_tag(self):
        """The producer never raises to its caller — terminal failures
        surface as an _ERR-tagged queue item so the consumer can emit
        a clean terminator."""
        scripts = [
            ("exc", httpx.ConnectError("dns failed")),
            ("exc", httpx.ConnectError("dns failed")),
            ("exc", httpx.ConnectError("dns failed")),
        ]
        stream_fn, _ = _make_scripted_stream(scripts)
        producer = _make_producer()
        chunk_q: asyncio.Queue = asyncio.Queue()
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            await producer.run(chunk_q)
        items = []
        while not chunk_q.empty():
            items.append(chunk_q.get_nowait())
        err_items = [it for it in items if it[0] is sr._ERR]
        assert len(err_items) == 1
        assert isinstance(err_items[0][1], httpx.ConnectError)


# ─── StreamConsumer tests ────────────────────────────────────────────────


def _make_consumer(*, chunk_q=None, translator=None, keepalive_interval=0.05,
                   cfg=None) -> sr.StreamConsumer:
    chunk_q = chunk_q if chunk_q is not None else asyncio.Queue()
    cfg = cfg or _FakeCfg()
    return sr.StreamConsumer(
        chunk_q=chunk_q,
        translator=translator,
        proj_sid="proj-1",
        conv_sid="conv-1",
        keepalive_interval=keepalive_interval,
        capture_outgoing=lambda b: b,
        intercept_completed=lambda b: b,
        cfg=cfg,
        log=_RecordingLog(),
        url="http://x/v1",
        on_status_error=lambda status, body: None,
    )


class TestStreamConsumer:

    @pytest.mark.asyncio
    async def test_forwards_status_and_chunks_then_stops_on_sentinel(self):
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (200, None)))
        await q.put((None, b"hello "))
        await q.put((None, b"world"))
        await q.put((sr._SENTINEL, None))
        consumer = _make_consumer(chunk_q=q, keepalive_interval=10.0)
        out = b""
        async for b in consumer.yield_to_client():
            out += b
        assert b"hello world" in out
        assert consumer.bytes_out == len(b"hello ") + len(b"world")
        assert consumer.status == 200
        assert not consumer.upstream_failed

    @pytest.mark.asyncio
    async def test_keeps_raw_buffer_snapshot_for_post_stream(self):
        class _SoftCfg(_FakeCfg):
            soft_completion_gate_enabled = True

        _sc.reset_state()
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (200, None)))
        await q.put((None, b"data: one"))
        await q.put((None, b" two"))
        await q.put((sr._SENTINEL, None))
        consumer = _make_consumer(
            chunk_q=q, keepalive_interval=10.0, cfg=_SoftCfg())

        async for _ in consumer.yield_to_client():
            pass

        assert consumer.raw_buffer_snapshot == "data: one two"

    @pytest.mark.asyncio
    async def test_idle_keepalive_fires_when_queue_silent(self):
        q: asyncio.Queue = asyncio.Queue()
        consumer = _make_consumer(chunk_q=q, keepalive_interval=0.02)

        async def _producer_side():
            # Stay silent for 80ms so the consumer must emit keepalives.
            await asyncio.sleep(0.08)
            await q.put((sr._SENTINEL, None))

        prod = asyncio.create_task(_producer_side())
        chunks = []
        async for b in consumer.yield_to_client():
            chunks.append(b)
        await prod
        assert consumer.keepalives_emitted >= 1
        assert all(c.startswith(b":") for c in chunks)

    @pytest.mark.asyncio
    async def test_err_tag_raises_to_caller(self):
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (200, None)))
        await q.put((sr._ERR, RuntimeError("boom")))
        consumer = _make_consumer(chunk_q=q, keepalive_interval=10.0)
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in consumer.yield_to_client():
                pass

    @pytest.mark.asyncio
    async def test_status_error_emits_error_event_and_flags_failed(self):
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (500, "internal error")))
        await q.put((sr._SENTINEL, None))
        consumer = _make_consumer(chunk_q=q, keepalive_interval=10.0)
        out = b""
        async for b in consumer.yield_to_client():
            out += b
        assert b"event: error" in out
        assert consumer.upstream_failed
        assert "upstream 500" in consumer.upstream_failure_msg

    @pytest.mark.asyncio
    async def test_translator_feed_path(self):
        """Each chunk passes through translator.feed before yield."""

        class _RecordTranslator:
            def __init__(self):
                self.fed: list[bytes] = []
                self.flushed = False

            def feed(self, chunk: bytes):
                self.fed.append(chunk)
                # Emit a transformed marker so we can verify routing.
                yield b"X:" + chunk

            def flush(self):
                self.flushed = True
                yield b"FLUSH"

        translator = _RecordTranslator()
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (200, None)))
        await q.put((None, b"a"))
        await q.put((None, b"b"))
        await q.put((sr._SENTINEL, None))
        consumer = _make_consumer(chunk_q=q, translator=translator,
                                  keepalive_interval=10.0)
        out = b""
        async for b in consumer.yield_to_client():
            out += b
        assert b"X:a" in out and b"X:b" in out
        assert translator.fed == [b"a", b"b"]
        assert translator.flushed
        assert b"FLUSH" in out

    @pytest.mark.asyncio
    async def test_intercept_completed_can_hold_bytes(self):
        """When intercept_completed returns empty bytes, the consumer
        must NOT yield anything for that chunk — the stream-rewrite
        path uses this to hold response.completed back."""
        q: asyncio.Queue = asyncio.Queue()
        await q.put((sr._STATUS, (200, None)))
        await q.put((None, b"held"))
        await q.put((None, b"out"))
        await q.put((sr._SENTINEL, None))

        def held_intercept(b: bytes) -> bytes:
            return b"" if b == b"held" else b

        consumer = sr.StreamConsumer(
            chunk_q=q, translator=None, proj_sid="p", conv_sid="c",
            keepalive_interval=10.0,
            capture_outgoing=lambda b: b,
            intercept_completed=held_intercept,
            cfg=_FakeCfg(), log=_RecordingLog(), url="http://x",
            on_status_error=lambda s, b: None,
        )
        out = b""
        async for b in consumer.yield_to_client():
            out += b
        assert b"held" not in out
        assert b"out" in out


# ─── StallSupervisor tests ───────────────────────────────────────────────


class TestStallSupervisor:

    @pytest.mark.asyncio
    async def test_register_and_unregister_round_trip(self):
        _sw.reset_state()
        sup = sr.StallSupervisor("proj-x", enabled=True)

        async def _noop():
            await asyncio.sleep(0)

        task = asyncio.create_task(_noop())
        sup.register(task)
        assert _sw._ACTIVE_TASKS.get("proj-x") is task
        await task
        sup.unregister()
        assert "proj-x" not in _sw._ACTIVE_TASKS

    def test_disabled_supervisor_is_inert(self):
        sup = sr.StallSupervisor("proj-y", enabled=False)
        # All ops must be no-ops; constructing a dummy task fails outside
        # an event loop, so just call unregister/clear.
        sup.unregister()
        sup.clear()  # no crash


# ─── relay_stream integration test ───────────────────────────────────────


class TestRelayStream:
    """Validates that the orchestrator wires the components correctly:
    initial keepalive fires before any producer byte; sentinel ends the
    loop cleanly; producer task is awaited on exit."""

    @pytest.mark.asyncio
    async def test_initial_keepalive_fires_before_producer_byte(self):
        scripts = [("ok", [b"event: response.completed\n"
                           b"data: {\"type\":\"response.completed\"}\n\n"])]
        stream_fn, _ = _make_scripted_stream(scripts)
        producer = _make_producer()
        q: asyncio.Queue = asyncio.Queue()
        consumer = _make_consumer(chunk_q=q, keepalive_interval=10.0)
        sup = sr.StallSupervisor("proj-1", enabled=False)
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            chunks: list[bytes] = []
            async for b in sr.relay_stream(
                    chunk_q=q, producer=producer, consumer=consumer,
                    supervisor=sup, keepalive_interval=10.0):
                chunks.append(b)
        # First yielded bytes must be the keepalive comment.
        assert chunks, "relay yielded no bytes"
        assert chunks[0] == b": tinyctx keepalive\n\n"
        # And the upstream body MUST still be delivered.
        joined = b"".join(chunks)
        assert b"response.completed" in joined

    @pytest.mark.asyncio
    async def test_relay_registers_producer_with_stall_watchdog(self):
        """Locks in the contract: while relay_stream is actively
        consuming, the producer task MUST be registered in
        stall_watchdog._ACTIVE_TASKS under proj_sid so that
        cancel_active_task can find and cancel it. Without this,
        the on_stall callback logs `stall_kill` (no task to cancel)
        and the request hangs until codex's own idle timeout.

        Reproduces the rq_X production scenario where post-P6
        stream_relay wiring was suspected of breaking the
        registration contract from ad7b2f1."""
        _sw.reset_state()
        proj_sid = "proj-register-test"

        started_evt = asyncio.Event()
        release_evt = asyncio.Event()

        class _Hanging:
            status_code = 200
            headers: dict = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                # Push one byte so STATUS path fires, then hang until
                # released — mimics the production trace (upstream
                # opened, then went silent mid-body).
                yield b"chunk-1"
                started_evt.set()
                await release_evt.wait()

        def stream_fn(self, method, url, **kwargs):
            return _Hanging()

        # Construct a producer that uses proj_sid.
        def build_frontier_retry_target(headers, body, reason):
            return (
                "http://frontier.test/v1/responses",
                {"X-Frontier": "1"},
                body,
                _make_decision("frontier", reason),
                _FakeCfg().frontier,
            )

        class _CfgStallOn(_FakeCfg):
            stall_watchdog_enabled = True

        cfg = _CfgStallOn()
        producer = sr.StreamProducer(
            url="http://local.test/v1/responses",
            headers={"Authorization": "Bearer x"},
            body={"model": "x", "input": []},
            proj_sid=proj_sid,
            conv_sid="conv-r",
            decision=_make_decision("local"),
            timeout=httpx.Timeout(connect=10.0, read=10.0, write=10.0,
                                  pool=10.0),
            transport=None,
            erg_key="conv-r",
            request_id="rid-r",
            cfg=cfg,
            log=_RecordingLog(),
            build_frontier_retry_target=build_frontier_retry_target,
            resolve_api_key=lambda b, c: "fk",
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = sr.StreamConsumer(
            chunk_q=q, translator=None,
            proj_sid=proj_sid, conv_sid="conv-r",
            keepalive_interval=10.0,
            capture_outgoing=lambda b: b,
            intercept_completed=lambda b: b,
            cfg=cfg, log=_RecordingLog(), url="http://local.test/v1",
            on_status_error=lambda s, b: None,
        )
        sup = sr.StallSupervisor(proj_sid, enabled=True)

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            gen = sr.relay_stream(
                chunk_q=q, producer=producer, consumer=consumer,
                supervisor=sup, keepalive_interval=10.0)

            # Pull bytes from relay_stream until the producer is alive
            # (started_evt set after upstream's first chunk).
            collected: list[bytes] = []
            try:
                # First __anext__: initial keepalive frame.
                collected.append(await asyncio.wait_for(
                    gen.__anext__(), timeout=1.0))
                # Second __anext__: producer task scheduled + supervisor
                # registered, consumer drains STATUS+first chunk.
                # Wait for upstream first-chunk signal first to ensure
                # the producer is mid-flight before we assert.
                async def _drain_one():
                    return await gen.__anext__()

                drain_task = asyncio.create_task(_drain_one())
                await asyncio.wait_for(started_evt.wait(), timeout=1.0)
                # Now the producer task is hanging at `await release_evt`,
                # which is EXACTLY the production scenario at stall fire.

                # CONTRACT: the producer task must be registered.
                active = _sw.get_active_task(proj_sid)
                assert active is not None, (
                    "stream_relay did not register the producer task "
                    "with stall_watchdog — cancel_active_task will return "
                    "False at stall fire and the request will hang.")
                assert not active.done(), (
                    "producer task already done — registration was stale")
                # Cancel to clean up the test.
                release_evt.set()
                try:
                    collected.append(await asyncio.wait_for(
                        drain_task, timeout=1.0))
                except (StopAsyncIteration, asyncio.CancelledError):
                    pass
            finally:
                release_evt.set()
                await gen.aclose()
                _sw.reset_state()

    @pytest.mark.asyncio
    async def test_relay_emits_stall_cancelled_not_stall_kill(self):
        """End-to-end: simulate a wedged upstream, fire the watchdog,
        and verify the path produces a StallCancelledError (cancellation
        worked) rather than leaving the producer hung (stall_kill
        fallback)."""
        _sw.reset_state()
        proj_sid = "proj-cancel-test"

        started_evt = asyncio.Event()

        class _Hanging:
            status_code = 200
            headers: dict = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                yield b"first-chunk"
                started_evt.set()
                # Hang forever — caller MUST cancel to unblock.
                await asyncio.sleep(60)

        def stream_fn(self, method, url, **kwargs):
            return _Hanging()

        class _CfgStallOn(_FakeCfg):
            stall_watchdog_enabled = True

        cfg = _CfgStallOn()

        def build_frontier_retry_target(headers, body, reason):
            return (
                "http://frontier.test/v1/responses",
                {"X-Frontier": "1"},
                body,
                _make_decision("frontier", reason),
                cfg.frontier,
            )

        producer = sr.StreamProducer(
            url="http://local.test/v1/responses",
            headers={"Authorization": "Bearer x"},
            body={"model": "x", "input": []},
            proj_sid=proj_sid,
            conv_sid="conv-c",
            decision=_make_decision("local"),
            timeout=httpx.Timeout(connect=10.0, read=10.0, write=10.0,
                                  pool=10.0),
            transport=None,
            erg_key="conv-c",
            request_id="rid-c",
            cfg=cfg,
            log=_RecordingLog(),
            build_frontier_retry_target=build_frontier_retry_target,
            resolve_api_key=lambda b, c: "fk",
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = sr.StreamConsumer(
            chunk_q=q, translator=None,
            proj_sid=proj_sid, conv_sid="conv-c",
            keepalive_interval=10.0,
            capture_outgoing=lambda b: b,
            intercept_completed=lambda b: b,
            cfg=cfg, log=_RecordingLog(), url="http://local.test/v1",
            on_status_error=lambda s, b: None,
        )
        sup = sr.StallSupervisor(proj_sid, enabled=True)

        # Race: relay_stream consumes; once the producer hangs, we
        # invoke cancel_active_task (simulating the watchdog), and
        # the consumer must surface StallCancelledError (NOT just
        # silence and a hung queue.get).
        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            collected: list[bytes] = []
            raised: list[Exception] = []

            async def _consume():
                try:
                    async for b in sr.relay_stream(
                            chunk_q=q, producer=producer,
                            consumer=consumer, supervisor=sup,
                            keepalive_interval=10.0):
                        collected.append(b)
                except _sw.StallCancelledError as e:
                    raised.append(e)
                except Exception as e:  # noqa: BLE001
                    raised.append(e)

            consume_task = asyncio.create_task(_consume())
            # Wait until the producer is hanging mid-aiter_raw.
            await asyncio.wait_for(started_evt.wait(), timeout=1.0)
            # Simulate the watchdog firing cancel_active_task.
            ok = _sw.cancel_active_task(proj_sid)
            assert ok, (
                "cancel_active_task returned False — producer task "
                "was not registered (this is the stall_kill bug).")
            # Wait for the consumer to surface the cancellation.
            await asyncio.wait_for(consume_task, timeout=2.0)
            assert raised, "no exception surfaced after cancel"
            assert isinstance(raised[0], _sw.StallCancelledError), (
                f"expected StallCancelledError, got {type(raised[0])}")
        _sw.reset_state()


# ─── Terminator helper ───────────────────────────────────────────────────


class TestBuildTerminator:

    def test_terminator_is_valid_sse_with_incomplete_status(self):
        evt = sr.build_terminator_event("upstream 500: oops",
                                        model="qwen-test")
        assert evt.startswith(b"event: response.completed\ndata: ")
        assert b'"status": "incomplete"' in evt
        assert b"tinyctx_proxy_terminator" in evt
