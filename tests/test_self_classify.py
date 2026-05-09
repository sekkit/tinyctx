"""Model-driven escalation classifier (self_classify).

The proxy asks the local model itself whether to escalate the current
turn — aligned with Anthropic's Advisor Strategy where the executor
decides escalation, not infrastructure-by-bytes.

These tests stand up a fake OpenAI-compat /chat/completions backend
that returns scripted JSON responses, and verify the parser, cache,
and skip-cases.
"""
from __future__ import annotations

import asyncio
import json as _json
import socket
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


# ─── _looks_like_user_query ─────────────────────────────────────────────────


def test_looks_like_user_query_true_for_user_message():
    from tinyctx.self_classify import looks_like_user_query
    body = {"input": [
        {"role": "user", "content": "do thing"},
    ]}
    assert looks_like_user_query(body) is True


def test_looks_like_user_query_true_for_typed_message_user():
    from tinyctx.self_classify import looks_like_user_query
    body = {"input": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "do thing"}]}
    ]}
    assert looks_like_user_query(body) is True


def test_looks_like_user_query_false_for_tool_result_tail():
    """A turn whose last item is a function_call_output is a tool-
    roundtrip, not a fresh user query. Skip self-classify — the model
    already knows what to do based on what was just read."""
    from tinyctx.self_classify import looks_like_user_query
    body = {"input": [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "x", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
    ]}
    assert looks_like_user_query(body) is False


def test_looks_like_user_query_false_for_empty():
    from tinyctx.self_classify import looks_like_user_query
    assert looks_like_user_query({}) is False
    assert looks_like_user_query({"input": []}) is False


# ─── _parse_response ────────────────────────────────────────────────────────


def test_parse_response_extracts_clean_json():
    from tinyctx.self_classify import _parse_response
    text = '{"escalate": true, "p": 0.85, "reason": "architectural choice"}'
    r = _parse_response(text)
    assert r is not None
    assert r.escalate is True
    assert abs(r.p - 0.85) < 1e-6
    assert r.reason == "architectural choice"


def test_parse_response_extracts_from_markdown_fenced():
    """Some models wrap their JSON in markdown despite instructions.
    The parser must tolerate it."""
    from tinyctx.self_classify import _parse_response
    text = (
        "Here's my analysis:\n"
        "```json\n"
        '{"escalate": false, "p": 0.2, "reason": "simple rename"}\n'
        "```\n"
    )
    r = _parse_response(text)
    assert r is not None
    assert r.escalate is False
    assert r.p == 0.2


def test_parse_response_clamps_p_to_unit_range():
    from tinyctx.self_classify import _parse_response
    r = _parse_response('{"escalate": true, "p": 5.0, "reason": "x"}')
    assert r is not None
    assert r.p == 1.0
    r = _parse_response('{"escalate": false, "p": -1, "reason": "x"}')
    assert r is not None
    assert r.p == 0.0


def test_parse_response_returns_none_for_garbage():
    from tinyctx.self_classify import _parse_response
    assert _parse_response("") is None
    assert _parse_response(None) is None  # type: ignore[arg-type]
    assert _parse_response("just prose, no json") is None
    assert _parse_response("{not valid json}") is None


def test_parse_response_handles_missing_optional_fields():
    """If reason / p are missing, defaults are used and result is still
    valid (don't drop the whole response over a missing optional)."""
    from tinyctx.self_classify import _parse_response
    r = _parse_response('{"escalate": true}')
    assert r is not None
    assert r.escalate is True
    assert r.reason == ""


# ─── _cache_key ─────────────────────────────────────────────────────────────


def test_cache_key_same_for_same_body_and_scope():
    from tinyctx.self_classify import _cache_key
    body = {"instructions": "x", "input": [
        {"role": "user", "content": "do thing"}]}
    a = _cache_key(body, "scopeA")
    b = _cache_key(body, "scopeA")
    assert a == b


def test_cache_key_different_for_different_scope():
    """Multi-project isolation: same body in project A and project B
    must NOT share a cache entry."""
    from tinyctx.self_classify import _cache_key
    body = {"instructions": "x", "input": [
        {"role": "user", "content": "do thing"}]}
    a = _cache_key(body, "projA:global")
    b = _cache_key(body, "projB:global")
    assert a != b


def test_cache_key_different_for_different_user_message():
    from tinyctx.self_classify import _cache_key
    body_a = {"instructions": "x", "input": [
        {"role": "user", "content": "task A"}]}
    body_b = {"instructions": "x", "input": [
        {"role": "user", "content": "task B"}]}
    assert _cache_key(body_a, "s") != _cache_key(body_b, "s")


# ─── classify (full HTTP flow against fake backend) ─────────────────────────


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _spawn_fake_backend(scripted_response: str) -> tuple[ThreadingHTTPServer, int]:
    """Stand up a /chat/completions backend that returns the scripted
    JSON payload as the model's content."""
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            payload = _json.dumps({
                "choices": [{"message": {"content": scripted_response}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def test_classify_returns_escalate_true_on_clean_yes():
    from tinyctx import self_classify
    self_classify.clear_cache()
    httpd, port = _spawn_fake_backend(
        '{"escalate": true, "p": 0.9, "reason": "auth flow"}')
    try:
        body = {"instructions": "you are codex",
                "input": [{"role": "user",
                           "content": "design jwt rotation strategy"}]}
        r = asyncio.new_event_loop().run_until_complete(
            self_classify.classify(
                body,
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                scope="t1"))
        assert r is not None
        assert r.escalate is True
        assert r.p == 0.9
        assert r.reason == "auth flow"
        assert r.cached is False
    finally:
        httpd.shutdown()


def test_classify_returns_none_on_tool_roundtrip():
    """Body is a tool-result roundtrip — classify must skip without
    even hitting the backend."""
    from tinyctx import self_classify
    self_classify.clear_cache()
    # We don't even need a backend — classify shouldn't reach it
    body = {"input": [
        {"role": "user", "content": "go"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]}
    r = asyncio.new_event_loop().run_until_complete(
        self_classify.classify(
            body,
            local_base_url="http://127.0.0.1:1/v1",  # would fail if reached
            local_model="fake",
            scope="t2"))
    assert r is None


def test_classify_caches_repeat_calls():
    """Same body + same scope within TTL → second call returns cached
    result without hitting backend."""
    from tinyctx import self_classify
    self_classify.clear_cache()
    call_count = {"n": 0}

    class _CountingHandler(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_POST(self):
            call_count["n"] += 1
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            payload = _json.dumps({"choices": [{"message": {
                "content": '{"escalate": false, "p": 0.3, "reason": "routine"}'}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _CountingHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = {"instructions": "x",
                "input": [{"role": "user", "content": "rename foo"}]}
        loop = asyncio.new_event_loop()
        r1 = loop.run_until_complete(self_classify.classify(
            body, local_base_url=f"http://127.0.0.1:{port}/v1",
            local_model="fake", scope="cache-test"))
        r2 = loop.run_until_complete(self_classify.classify(
            body, local_base_url=f"http://127.0.0.1:{port}/v1",
            local_model="fake", scope="cache-test"))
        assert call_count["n"] == 1, (
            f"expected 1 backend call, got {call_count['n']}"
        )
        assert r1.cached is False
        assert r2.cached is True
        assert r2.escalate == r1.escalate
    finally:
        httpd.shutdown()


def test_classify_returns_none_on_backend_error():
    """If local model is unreachable, classify returns None silently —
    the proxy then falls back to the heuristic route decision."""
    from tinyctx import self_classify
    self_classify.clear_cache()
    body = {"instructions": "x",
            "input": [{"role": "user", "content": "task"}]}
    # Point at a port nothing listens on
    r = asyncio.new_event_loop().run_until_complete(
        self_classify.classify(
            body,
            local_base_url="http://127.0.0.1:1/v1",
            local_model="fake",
            timeout_s=0.5,
            scope="t-err"))
    assert r is None


def test_classify_returns_none_on_unparseable_response():
    from tinyctx import self_classify
    self_classify.clear_cache()
    httpd, port = _spawn_fake_backend("just prose, no JSON at all")
    try:
        body = {"instructions": "x",
                "input": [{"role": "user", "content": "task"}]}
        r = asyncio.new_event_loop().run_until_complete(
            self_classify.classify(
                body, local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake", scope="t-bad"))
        assert r is None
    finally:
        httpd.shutdown()


def test_clear_cache_targeted_by_scope():
    from tinyctx import self_classify
    self_classify.clear_cache()
    # Inject some fake entries
    self_classify._CACHE["projA:abcdef0123456789"] = (time.time(), object())  # type: ignore
    self_classify._CACHE["projB:0123456789abcdef"] = (time.time(), object())  # type: ignore
    self_classify.clear_cache("projA:")
    assert not any(k.startswith("projA:") for k in self_classify._CACHE)
    assert any(k.startswith("projB:") for k in self_classify._CACHE)


def test_default_config_enabled_with_sane_threshold():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.self_classify_enabled is True
    assert 0.5 <= cfg.self_classify_threshold <= 0.9
    assert 1.0 <= cfg.self_classify_timeout_s <= 30.0
