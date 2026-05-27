"""Integration tests for choice arbiter: judge + advisor + store + guard
inject. Uses mock HTTP servers to simulate local (judge) and frontier
(advisor) backends."""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import pytest

from tinyctx.choice_arbiter import (
    JudgeResult,
    Verdict,
    advisor_decide,
    consume_verdict,
    inject_verdict_into_body,
    intercept,
    judge_and_extract,
    reset_state,
    store_verdict,
)
from tinyctx.guards import ChoiceArbiterGuard, GuardContext


# ─── mock server helpers ────────────────────────────────────────────────

class _MockChatHandler(BaseHTTPRequestHandler):
    response_json: dict[str, Any] = {}
    response_status: int = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        self.server.last_request_body = body  # type: ignore[attr-defined]
        resp = json.dumps(self.response_json).encode()
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass


class _MockSSEHandler(BaseHTTPRequestHandler):
    sse_events: list[str] = []
    response_status: int = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        self.server.last_request_body = body  # type: ignore[attr-defined]
        self.send_response(self.response_status)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for evt in self.sse_events:
            self.wfile.write(evt.encode())
            self.wfile.flush()

    def log_message(self, format, *args):
        pass


def _start_server(handler_class, port: int = 0) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), handler_class)
    srv.last_request_body = b""  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _server_url(srv: HTTPServer) -> str:
    host, port = srv.server_address
    return f"http://{host}:{port}/v1"


# ─── judge integration ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_judge_choice_ask_yes():
    handler = type("H", (_MockChatHandler,), {
        "response_json": {
            "choices": [{"message": {"content": (
                '{"is_choice_ask": true, "question": "A or B?", '
                '"options": ["A", "B"], "context_summary": "picking"}'  )}}]
        }
    })
    srv = _start_server(handler)
    try:
        result = await judge_and_extract(
            "Should I use A or B?",
            local_base_url=_server_url(srv),
            local_model="test",
            timeout_s=5.0,
        )
        assert result is not None
        assert result.is_choice_ask is True
        assert result.question == "A or B?"
        assert result.options == ["A", "B"]
    finally:
        srv.shutdown()


@pytest.mark.asyncio
async def test_judge_choice_ask_no():
    handler = type("H", (_MockChatHandler,), {
        "response_json": {
            "choices": [{"message": {"content": (
                '{"is_choice_ask": false, "question": "", '
                '"options": [], "context_summary": ""}'  )}}]
        }
    })
    srv = _start_server(handler)
    try:
        result = await judge_and_extract(
            "Here is the code.",
            local_base_url=_server_url(srv),
            local_model="test",
            timeout_s=5.0,
        )
        assert result is not None
        assert result.is_choice_ask is False
    finally:
        srv.shutdown()


@pytest.mark.asyncio
async def test_judge_empty_text_returns_none():
    result = await judge_and_extract(
        "",
        local_base_url="http://127.0.0.1:1/v1",
        local_model="test",
        timeout_s=0.5,
    )
    assert result is None


# ─── advisor integration ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_advisor_picks_option():
    handler = type("H", (_MockSSEHandler,), {
        "sse_events": [
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"CHOSEN: generators\\nREASON: memory efficiency\\n"}\n\n',
            'event: response.completed\ndata: {"type":"response.completed"}\n\n',
        ]
    })
    srv = _start_server(handler)
    try:
        result = await advisor_decide(
            JudgeResult(
                is_choice_ask=True,
                question="Generators or pandas?",
                options=["generators", "pandas"],
                context_summary="building data pipeline",
            ),
            frontier_base_url=_server_url(srv),
            frontier_model="test",
            timeout_s=5.0,
        )
        assert result is not None
        assert "generators" in result
    finally:
        srv.shutdown()


@pytest.mark.asyncio
async def test_advisor_no_options_returns_none():
    result = await advisor_decide(
        JudgeResult(is_choice_ask=True, question="Q", options=[], context_summary=""),
        frontier_base_url="http://127.0.0.1:1/v1",
        frontier_model="test",
        timeout_s=0.5,
    )
    assert result is None


# ─── full intercept pipeline ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intercept_full_pipeline():
    # judge server
    jh = type("JH", (_MockChatHandler,), {
        "response_json": {
            "choices": [{"message": {"content": (
                '{"is_choice_ask": true, "question": "httpx or aiohttp?", '
                '"options": ["httpx", "aiohttp", "requests"], '
                '"context_summary": "HTTP client selection"}'  )}}]
        }
    })
    jsrv = _start_server(jh)
    # advisor server
    ah = type("AH", (_MockSSEHandler,), {
        "sse_events": [
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"CHOSEN: httpx\\nREASON: async support\\nFALLBACK: aiohttp"}\n\n',
            'event: response.completed\ndata: {"type":"response.completed"}\n\n',
        ]
    })
    asrv = _start_server(ah)

    reset_state("test_intercept_pipeline")
    try:
        verdict = await intercept(
            "Should I use httpx or aiohttp?",
            conv_sid="test_intercept_pipeline",
            local_base_url=_server_url(jsrv),
            local_model="test",
            frontier_base_url=_server_url(asrv),
            frontier_model="test",
            judge_timeout_s=5.0,
            advisor_timeout_s=5.0,
        )
        assert verdict is not None
        assert "httpx" in verdict.advisor_choice
        assert verdict.question == "httpx or aiohttp?"
        assert len(verdict.options) == 3

        stored = consume_verdict("test_intercept_pipeline")
        assert stored is not None
        assert stored.advisor_choice == verdict.advisor_choice
    finally:
        jsrv.shutdown()
        asrv.shutdown()


@pytest.mark.asyncio
async def test_intercept_not_choice_ask():
    handler = type("H", (_MockChatHandler,), {
        "response_json": {
            "choices": [{"message": {"content": (
                '{"is_choice_ask": false, "question": "", '
                '"options": [], "context_summary": ""}'  )}}]
        }
    })
    srv = _start_server(handler)
    reset_state("test_not_choice")
    try:
        verdict = await intercept(
            "Here is the completed code.",
            conv_sid="test_not_choice",
            local_base_url=_server_url(srv),
            local_model="test",
            frontier_base_url="http://127.0.0.1:1/v1",
            frontier_model="test",
            advisor_timeout_s=0.5,
        )
        assert verdict is None
        assert consume_verdict("test_not_choice") is None
    finally:
        srv.shutdown()


# ─── guard injection ────────────────────────────────────────────────────

def test_guard_fires_with_stored_verdict():
    reset_state("test_guard_integration")
    store_verdict("test_guard_integration", Verdict(
        advisor_choice="CHOSEN: option B\nREASON: safer approach",
        question="A or B?",
        options=["A", "B"],
        ts=1234.0,
    ))
    ctx = GuardContext(
        body={
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
        },
        proj_sid="test_guard_integration",
        conv_sid="test_guard_integration",
        turn_count=1,
    )
    g = ChoiceArbiterGuard()
    result = g.apply(ctx)

    assert result.fired is True
    assert result.body_mutated is True
    assert "option B" in result.reason

    items = ctx.body["input"]
    assert len(items) == 2
    last = items[-1]
    assert last["role"] == "user"
    assert "option B" in last["content"][0]["text"]

    assert consume_verdict("test_guard_integration") is None
