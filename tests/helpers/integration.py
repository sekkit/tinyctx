"""Shared helpers for integration-saga tests.

These helpers mirror the patterns used in tests/test_proxy_retry.py and
tests/test_keepalive.py but extract the small pieces every saga needs:
mock SSE stream context, mock script driver, and a drain helper. Keep this
module thin — anything saga-specific stays in the saga's test file.
"""
from __future__ import annotations

import asyncio
from typing import Any


# ─── mock SSE stream context ─────────────────────────────────────────────


class MockStreamCtx:
    """Async context manager mimicking the object returned by
    `httpx.AsyncClient.stream("POST", ...)`. Yields configured chunks on
    `aiter_raw()` and reports the configured status_code. For error
    responses, returns `err_body` on `aread()`."""

    def __init__(self, status_code: int, *, err_body: bytes = b"",
                 chunks: list[bytes] | None = None,
                 headers: dict | None = None) -> None:
        self.status_code = status_code
        self._err_body = err_body
        self._chunks = chunks or []
        self.headers = headers or {}

    async def __aenter__(self) -> "MockStreamCtx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def aread(self) -> bytes:
        return self._err_body

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


def make_mock_stream(scripts: list):
    """Build a stand-in for `httpx.AsyncClient.stream` from a script list.

    Each script entry is one of:
        ("ok", [chunks_bytes])  -> 200 stream with these chunks
        ("err", status, body)   -> error status with this body
        ("exc", exc)            -> raise inside the context entry

    Returns (stream_fn, state) where state["calls"] accumulates the URL
    of each attempt and state["idx"] tracks consumption.
    """
    state: dict[str, Any] = {"calls": [], "idx": 0}

    def stream_fn(self, method, url, **kwargs):
        state["calls"].append(url)
        if state["idx"] >= len(scripts):
            raise RuntimeError(f"unexpected extra stream call to {url}")
        sc = scripts[state["idx"]]
        state["idx"] += 1
        kind = sc[0]
        if kind == "ok":
            chunks = sc[1]
            return MockStreamCtx(200, chunks=chunks)
        if kind == "err":
            status, body = sc[1], sc[2]
            return MockStreamCtx(status, err_body=body)
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


# ─── mock for non-stream POST ────────────────────────────────────────────


def make_mock_post(responses: list):
    """Mock `httpx.AsyncClient.post`. Each entry is either an Exception
    (raised) or a (status_code, body_dict) tuple."""
    import httpx
    state: dict[str, Any] = {"calls": [], "idx": 0}

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


# ─── stream draining ─────────────────────────────────────────────────────


async def drain_stream(streaming_response) -> bytes:
    """Walk a Starlette StreamingResponse's body_iterator to bytes."""
    chunks: list[bytes] = []
    async for chunk in streaming_response.body_iterator:
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(bytes(chunk))
        else:
            chunks.append(str(chunk).encode())
    return b"".join(chunks)


# ─── SSE chunks for healthy turns ────────────────────────────────────────


def healthy_local_chunks(call_name: str = "exec_command") -> list[bytes]:
    """A minimal LMStudio-shaped Responses-API SSE stream with one
    function_call item and a response.completed terminator."""
    return [
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"id":"resp_local_1"}}\n\n',
        (b'event: response.output_item.added\n'
         b'data: {"type":"response.output_item.added",'
         b'"item":{"type":"function_call","name":"' + call_name.encode()
         + b'","arguments":"{}","call_id":"call_1"}}\n\n'),
        b'event: response.completed\n'
        b'data: {"type":"response.completed","response":{"id":"resp_local_1","status":"completed"}}\n\n',
    ]


def healthy_frontier_chunks() -> list[bytes]:
    """chatgpt.com-style frontier SSE with reasoning + tool_call + done."""
    return [
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"id":"resp_frontier_1"}}\n\n',
        b'event: response.reasoning_summary_text.delta\n'
        b'data: {"type":"response.reasoning_summary_text.delta","delta":"planning"}\n\n',
        (b'event: response.output_item.added\n'
         b'data: {"type":"response.output_item.added",'
         b'"item":{"type":"function_call","name":"shell",'
         b'"arguments":"{}","call_id":"call_f1"}}\n\n'),
        b'event: response.completed\n'
        b'data: {"type":"response.completed","response":{"id":"resp_frontier_1","status":"completed"}}\n\n',
    ]
