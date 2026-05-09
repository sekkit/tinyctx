"""SSE keepalive injection during long upstream streams.

When the upstream (DeepSeek / chatgpt.com) is silently thinking and
no SSE bytes flow for a long time, codex.app's stream parser, TCP
middleboxes, and firewalls can declare the connection dead and abort.
The proxy injects `: tinyctx keepalive\\n\\n` SSE comment lines (which
spec-compliant clients ignore) every `stream_keepalive_interval_s`
seconds during idle to keep everything alive.

These tests use a fake httpx response with a controllable yield delay
to verify keepalives fire at the right rate without buffering chunks.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─── default config ────────────────────────────────────────────────────────


def test_default_keepalive_interval_set_and_reasonable():
    """Default must be > 0 (feature on) and < codex.app's default
    stream_idle_timeout (300s) with safety margin."""
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.stream_keepalive_interval_s > 0, "feature should be on by default"
    assert cfg.stream_keepalive_interval_s <= 60, (
        "should be well below codex's stream_idle_timeout=300s with margin"
    )


# ─── trace field ────────────────────────────────────────────────────────────


def test_trace_field_exists_and_defaults_zero():
    from tinyctx.trace import RequestTrace
    t = RequestTrace()
    assert t.keepalives_emitted == 0


# ─── unit: keepalive logic in isolation ─────────────────────────────────────


class _SlowChunkResponse:
    """Mimic httpx response.aiter_raw() — yields chunks with controllable
    delay between them. Used to verify keepalive timing."""

    def __init__(self, chunks: list[bytes], delay_before_each: float):
        self._chunks = chunks
        self._delay = delay_before_each
        self.status_code = 200

    async def aread(self) -> bytes:
        return b""

    def aiter_raw(self):
        return self._aiter()

    async def _aiter(self):
        for c in self._chunks:
            await asyncio.sleep(self._delay)
            yield c


@pytest.mark.asyncio
async def test_keepalive_fires_during_idle_chunks():
    """When upstream takes >keepalive_interval to produce the next chunk,
    the consumer must see a keepalive comment line BEFORE the chunk."""
    # Replicate the queue/wait_for pattern in proxy._stream_proxy directly.
    KA = 0.3  # keepalive_interval seconds

    chunks_yielded: list[bytes] = []
    keepalives = 0

    chunk_q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def producer():
        # Mimic upstream that takes 0.7s between chunks (> KA threshold)
        for c in [b"data: A\n\n", b"data: B\n\n", b"data: C\n\n"]:
            await asyncio.sleep(0.7)
            await chunk_q.put((None, c))
        await chunk_q.put((_SENTINEL, None))

    p = asyncio.create_task(producer())
    try:
        while True:
            try:
                tag, payload = await asyncio.wait_for(
                    chunk_q.get(), timeout=KA)
            except asyncio.TimeoutError:
                chunks_yielded.append(b": keepalive\n\n")
                keepalives += 1
                continue
            if tag is _SENTINEL:
                break
            chunks_yielded.append(payload)
    finally:
        if not p.done():
            p.cancel()

    # 3 chunks at 0.7s intervals → between each the consumer waits 0.7s.
    # KA=0.3 → roughly 2 keepalives between each chunk pair (0.7 / 0.3 ≈ 2).
    assert keepalives >= 4, (
        f"expected at least 4 keepalives across 3 slow chunks, got {keepalives}"
    )
    # The 3 real chunks must all have arrived
    real_chunks = [c for c in chunks_yielded if not c.startswith(b": keepalive")]
    assert real_chunks == [b"data: A\n\n", b"data: B\n\n", b"data: C\n\n"]


@pytest.mark.asyncio
async def test_keepalive_skipped_when_chunks_arrive_quickly():
    """When upstream is fast (chunks arrive faster than keepalive
    interval), no keepalives should be emitted — the proxy stays out
    of the way."""
    KA = 1.0
    keepalives = 0
    real_chunks: list[bytes] = []
    chunk_q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def producer():
        for c in [b"a", b"b", b"c"]:
            await asyncio.sleep(0.05)  # << KA
            await chunk_q.put((None, c))
        await chunk_q.put((_SENTINEL, None))

    p = asyncio.create_task(producer())
    try:
        while True:
            try:
                tag, payload = await asyncio.wait_for(
                    chunk_q.get(), timeout=KA)
            except asyncio.TimeoutError:
                keepalives += 1
                continue
            if tag is _SENTINEL:
                break
            real_chunks.append(payload)
    finally:
        if not p.done():
            p.cancel()

    assert keepalives == 0, (
        f"no chunk arrived after KA={KA}s of idle; got {keepalives} keepalives"
    )
    assert real_chunks == [b"a", b"b", b"c"]


# ─── integration: through the actual _stream_proxy + StreamingResponse ─────


def test_stream_proxy_emits_keepalive_with_slow_backend():
    """End-to-end: spin up a fake SSE backend that delays between
    chunks; verify that the client receives keepalive comment lines
    interleaved with the data."""
    import json as _json

    class _SlowSSEBackend(BaseHTTPRequestHandler):
        # delay_per_chunk on the server side, set by class attr
        delay_per_chunk = 0.6

        def log_message(self, *a, **k): pass

        def do_POST(self):
            # discard input
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                time.sleep(self.delay_per_chunk)
                msg = f"data: chunk-{i}\n\n".encode()
                self.wfile.write(f"{len(msg):x}\r\n".encode())
                self.wfile.write(msg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _SlowSSEBackend)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # Configure tinyctx to point at our fake backend with a tight keepalive
        import os, sys
        os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
        os.environ["TINYCTX_LOCAL_MODEL"] = "fake"
        os.environ["TINYCTX_VERBOSE"] = "0"
        os.environ["TINYCTX_CONFIG"] = "/dev/null"
        os.environ.pop("TINYCTX_FORCE_ROUTE", None)

        for m in list(sys.modules):
            if m.startswith("tinyctx"):
                del sys.modules[m]
        import tinyctx.proxy as proxy_mod

        # Override config for tighter keepalive
        proxy_mod.CFG.stream_keepalive_interval_s = 0.2

        import uvicorn
        proxy_port = _free_port()
        cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1",
                             port=proxy_port, log_level="error")
        server = uvicorn.Server(cfg)

        def _run(): asyncio.run(server.serve())
        threading.Thread(target=_run, daemon=True).start()

        # wait ready
        from urllib.request import Request, urlopen
        for _ in range(50):
            try:
                urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)

        # Stream a request — backend yields 1 chunk per 0.6s, keepalive
        # interval is 0.2s → expect ~2 keepalives between each chunk.
        body = {
            "model": "tinyctx-local",
            "instructions": "test",
            "input": [{"type":"message","role":"user",
                       "content":[{"type":"input_text","text":"hi"}]}],
            "stream": True,
        }
        req = Request(
            f"http://127.0.0.1:{proxy_port}/v1/responses",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            method="POST",
        )
        with urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", "replace")

        # Real chunks present
        assert "chunk-0" in raw and "chunk-1" in raw and "chunk-2" in raw
        # Keepalive comment lines present
        assert ": tinyctx keepalive" in raw, (
            f"expected at least one keepalive line; got:\n{raw!r}"
        )
        # And there should be MULTIPLE (server delay 0.6s, KA 0.2s,
        # 3 chunks → at least 4-6 keepalives expected)
        ka_count = raw.count(": tinyctx keepalive")
        assert ka_count >= 3, f"expected >= 3 keepalives, got {ka_count}"
    finally:
        httpd.shutdown()


def test_stream_proxy_emits_keepalive_during_header_wait():
    """Regression: codex.app's stream parser disconnects after ~60s of
    zero bytes from the proxy. With a slow upstream (e.g. DeepSeek
    loading a 500K-token context) the upstream may take many seconds
    just to send response HEADERS — with NO chunks yet. The proxy must
    emit keepalives during this header-wait phase, not only during the
    body-streaming phase.

    Backend below sleeps before sending status/headers (simulating slow
    inference cold-start) and verifies keepalive bytes flow to the
    client during that gap."""
    import json as _json

    class _SlowStartBackend(BaseHTTPRequestHandler):
        # delay BEFORE sending response status/headers
        header_delay = 1.5

        def log_message(self, *a, **k): pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            # Sleep BEFORE sending any response bytes
            time.sleep(self.header_delay)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Then immediately send a single chunk
            msg = b"data: ok\n\n"
            self.wfile.write(f"{len(msg):x}\r\n".encode())
            self.wfile.write(msg)
            self.wfile.write(b"\r\n0\r\n\r\n")

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _SlowStartBackend)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        import os, sys
        os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
        os.environ["TINYCTX_LOCAL_MODEL"] = "fake"
        os.environ["TINYCTX_VERBOSE"] = "0"
        os.environ["TINYCTX_CONFIG"] = "/dev/null"
        os.environ.pop("TINYCTX_FORCE_ROUTE", None)

        for m in list(sys.modules):
            if m.startswith("tinyctx"):
                del sys.modules[m]
        import tinyctx.proxy as proxy_mod

        # Tight keepalive (0.4s) vs server header_delay (1.5s):
        # we should see at least 2-3 keepalives BEFORE the chunk arrives.
        proxy_mod.CFG.stream_keepalive_interval_s = 0.4

        import uvicorn
        proxy_port = _free_port()
        cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1",
                             port=proxy_port, log_level="error")
        server = uvicorn.Server(cfg)
        threading.Thread(target=lambda: asyncio.run(server.serve()),
                         daemon=True).start()

        from urllib.request import Request, urlopen
        for _ in range(50):
            try:
                urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)

        body = {"model": "tinyctx-local", "instructions": "x",
                "input": [{"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True}
        req = Request(
            f"http://127.0.0.1:{proxy_port}/v1/responses",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            method="POST",
        )
        # Stream and observe arrival timing of bytes
        first_byte_t: float | None = None
        keepalive_bytes_t: list[float] = []
        chunk_arrived_t: float | None = None
        start = time.time()
        with urlopen(req, timeout=10) as r:
            buf = b""
            while True:
                ch = r.read(64)
                if not ch:
                    break
                if first_byte_t is None:
                    first_byte_t = time.time() - start
                buf += ch
                # Track keepalive bytes
                if b": tinyctx keepalive" in buf and not keepalive_bytes_t:
                    keepalive_bytes_t.append(time.time() - start)
                if b"data: ok" in buf:
                    chunk_arrived_t = time.time() - start
                    # keep reading to drain
        raw = buf.decode("utf-8", "replace")

        # Real chunk eventually arrived
        assert "data: ok" in raw, f"expected response chunk; got:\n{raw!r}"
        # Keepalives MUST appear in the stream
        assert ": tinyctx keepalive" in raw, (
            "expected keepalive(s) during header-wait phase; got:\n" + raw
        )
        # And the first keepalive must arrive BEFORE the chunk
        assert keepalive_bytes_t, "keepalive bytes never observed"
        assert chunk_arrived_t is not None, "chunk never arrived"
        assert keepalive_bytes_t[0] < chunk_arrived_t, (
            f"keepalive arrived at t={keepalive_bytes_t[0]:.2f}s but chunk "
            f"arrived earlier at t={chunk_arrived_t:.2f}s — phase-1 "
            f"keepalive is not firing"
        )
    finally:
        httpd.shutdown()


def test_stream_proxy_zero_interval_disables_keepalive():
    """With stream_keepalive_interval_s = 0, no keepalive lines should
    be injected even on a slow backend. Lets users opt out."""
    import json as _json

    class _SlowSSEBackend(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            time.sleep(0.5)
            msg = b"data: hi\n\n"
            self.wfile.write(f"{len(msg):x}\r\n".encode())
            self.wfile.write(msg)
            self.wfile.write(b"\r\n0\r\n\r\n")

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _SlowSSEBackend)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        import os, sys
        os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
        os.environ["TINYCTX_LOCAL_MODEL"] = "fake"
        os.environ["TINYCTX_VERBOSE"] = "0"
        os.environ["TINYCTX_CONFIG"] = "/dev/null"
        os.environ.pop("TINYCTX_FORCE_ROUTE", None)
        for m in list(sys.modules):
            if m.startswith("tinyctx"):
                del sys.modules[m]
        import tinyctx.proxy as proxy_mod
        proxy_mod.CFG.stream_keepalive_interval_s = 0.0  # disabled

        import uvicorn
        proxy_port = _free_port()
        cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1",
                             port=proxy_port, log_level="error")
        server = uvicorn.Server(cfg)
        threading.Thread(target=lambda: asyncio.run(server.serve()),
                         daemon=True).start()
        from urllib.request import Request, urlopen
        for _ in range(50):
            try:
                urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        body = {"model":"tinyctx-local","instructions":"x",
                "input":[{"type":"message","role":"user",
                          "content":[{"type":"input_text","text":"hi"}]}],
                "stream": True}
        req = Request(f"http://127.0.0.1:{proxy_port}/v1/responses",
            data=_json.dumps(body).encode(),
            headers={"Content-Type":"application/json"}, method="POST")
        with urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", "replace")
        assert "data: hi" in raw
        assert ": tinyctx keepalive" not in raw, (
            f"keepalive should be DISABLED; got:\n{raw!r}"
        )
    finally:
        httpd.shutdown()
