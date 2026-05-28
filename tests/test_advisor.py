"""Unit tests for the Advisor Strategy MCP server.

Covers:
  - Tool schema shape (so codex/MCP clients can register it).
  - JSON-RPC handlers: initialize, tools/list, tools/call (happy + bad name).
  - `call_advisor` with httpx Client patched: success / HTTP error / network.

No external network. The httpx.Client is monkey-patched.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

from tinyctx import advisor as adv


# ─────────────────────────── helpers ────────────────────────────


class _FakeStreamResponse:
    """Mimics what httpx.Client.stream() yields as a context manager."""
    def __init__(self, status_code: int, sse_lines: list[str] | None = None,
                 error_body: bytes | None = None):
        self.status_code = status_code
        self._sse_lines = sse_lines or []
        self._error_body = error_body or b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._error_body

    def iter_lines(self):
        for line in self._sse_lines:
            yield line


class _FakeClient:
    """Stand-in for httpx.Client. Captures last call args, returns queued
    stream response (or raises queued exception)."""
    last_url: str = ""
    last_payload: dict | None = None
    last_headers: dict | None = None
    next_response: _FakeStreamResponse | None = None
    next_exception: Exception | None = None

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    next_exception_always: Exception | None = None
    # When set, raise this on EVERY stream() call (for retry tests).

    def stream(self, method, url, json=None, headers=None):  # noqa: A002
        type(self).last_url = url
        type(self).last_payload = json
        type(self).last_headers = headers
        if type(self).next_exception_always is not None:
            raise type(self).next_exception_always
        if type(self).next_exception is not None:
            exc = type(self).next_exception
            type(self).next_exception = None
            raise exc
        resp = type(self).next_response
        if resp is None:
            raise RuntimeError("no fake response queued")
        return resp


@contextmanager
def _patched_httpx():
    import httpx

    original_client = httpx.Client
    httpx.Client = _FakeClient  # type: ignore[assignment]
    try:
        yield _FakeClient
    finally:
        httpx.Client = original_client  # type: ignore[assignment]
        _FakeClient.last_url = ""
        _FakeClient.last_payload = None
        _FakeClient.last_headers = None
        _FakeClient.next_response = None
        _FakeClient.next_exception = None
        _FakeClient.next_exception_always = None


# ─────────────────────────── schema tests ────────────────────────


def test_tool_schema_has_required_shape():
    s = adv._tool_schema()
    assert s["name"] == "ask_advisor"
    assert "description" in s and "Consult" in s["description"]
    inp = s["inputSchema"]
    assert inp["type"] == "object"
    props = inp["properties"]
    assert set(props) == {"question", "context", "previous_attempts"}
    assert inp["required"] == ["question"]
    for k in ("question", "context", "previous_attempts"):
        assert props[k]["type"] == "string"
        assert props[k]["description"]


# ─────────────────────────── call_advisor ────────────────────────


def test_call_advisor_empty_question_returns_error():
    out = adv.call_advisor("   ", context="ctx")
    assert out["error"] == "empty_question"
    assert out["usage"] is None


def _sse(events: list[dict]) -> list[str]:
    out = []
    for e in events:
        out.append(f"event: {e.get('type','')}")
        out.append(f"data: {json.dumps(e)}")
        out.append("")
    return out


def test_call_advisor_success_accumulates_stream_deltas():
    events = [
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_text.delta", "delta": "Use option B "},
        {"type": "response.output_text.delta", "delta": "because it's idempotent."},
        {"type": "response.completed",
         "response": {"id": "r1",
                      "usage": {"input_tokens": 200, "output_tokens": 50}}},
    ]
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(200, sse_lines=_sse(events))
        out = adv.call_advisor(
            question="Pick A or B?",
            context="A is fast but unsafe; B is slow but idempotent.",
            previous_attempts="Tried A; got dup writes.",
        )
        assert out["error"] is None
        assert out["text"] == "Use option B because it's idempotent."
        assert out["usage"]["input_tokens"] == 200

        payload = fc.last_payload
        assert payload["model"] == adv.ADVISOR_MODEL
        assert payload["stream"] is True
        assert payload["store"] is False
        assert fc.last_headers["Accept"] == "text/event-stream"
        first = payload["input"][0]["content"][0]["text"]
        assert "Pick A or B?" in first
        assert "idempotent" in first
        assert "dup writes" in first


def test_call_advisor_output_text_done_overrides_deltas():
    """If the server emits a final `response.output_text.done` with the
    canonical text, that supersedes accumulated deltas."""
    events = [
        {"type": "response.output_text.delta", "delta": "draft "},
        {"type": "response.output_text.delta", "delta": "draft "},
        {"type": "response.output_text.done", "text": "Final canonical text."},
    ]
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(200, sse_lines=_sse(events))
        out = adv.call_advisor("q")
        assert out["error"] is None
        assert out["text"] == "Final canonical text."


def test_call_advisor_http_error_surfaced():
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(500, error_body=b"kaboom")
        out = adv.call_advisor("hi", context="x")
        assert out["error"] == "http_500"
        assert "kaboom" in out["text"]
        assert out["attempts"] == 3  # 5xx is retryable, exhausted all


def test_call_advisor_network_error():
    import httpx

    with _patched_httpx() as fc:
        fc.next_exception_always = httpx.ConnectError("no route")
        out = adv.call_advisor("hi")
        assert out["error"] == "network"
        assert "no route" in out["text"]
        assert out["attempts"] == 3  # 2 retries + 1 initial = 3 total


def test_call_advisor_empty_stream_returns_placeholder():
    """No deltas, no done event → we still return a non-error result with a
    sentinel string (instead of crashing or claiming success silently)."""
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(200, sse_lines=[])
        out = adv.call_advisor("anything")
        assert out["error"] is None
        assert "[advisor returned no output_text]" in out["text"]


def test_call_advisor_proxy_sse_error_bubbles_up():
    """tinyctx proxy turns upstream 4xx into `event: error` SSE; the advisor
    should surface this as a stream_error rather than silently succeeding."""
    events = [
        {"type": "error", "status": 400,
         "body": "{\"detail\":\"Unsupported parameter: max_output_tokens\"}"},
    ]
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(200, sse_lines=_sse(events))
        out = adv.call_advisor("q")
        assert out["error"] == "stream_error"
        assert "Unsupported parameter" in out["text"]
        assert out["attempts"] == 3  # stream error is retryable


def test_call_advisor_payload_omits_max_output_tokens():
    """codex's chatgpt backend rejects max_output_tokens; verify we don't
    send it. The advisor's system prompt already bounds length verbally."""
    with _patched_httpx() as fc:
        fc.next_response = _FakeStreamResponse(200, sse_lines=_sse([
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {}},
        ]))
        adv.call_advisor("q")
        assert "max_output_tokens" not in fc.last_payload


# ─────────────────────────── auth resolution ────────────────────


def test_resolve_auth_token_reads_codex_auth_json(tmp_path=None):
    import json as _json
    import os as _os
    import tempfile

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _json.dump({"tokens": {"access_token": "TOK_FROM_CODEX"}}, f)
    f.close()
    saved_path = adv.CODEX_AUTH_PATH
    saved_key = adv.ADVISOR_API_KEY
    adv.CODEX_AUTH_PATH = f.name
    adv.ADVISOR_API_KEY = ""
    try:
        assert adv._resolve_auth_token() == "TOK_FROM_CODEX"
    finally:
        adv.CODEX_AUTH_PATH = saved_path
        adv.ADVISOR_API_KEY = saved_key
        _os.unlink(f.name)


def test_resolve_auth_token_env_overrides_file():
    saved_key = adv.ADVISOR_API_KEY
    adv.ADVISOR_API_KEY = "OVERRIDE_KEY"
    try:
        assert adv._resolve_auth_token() == "OVERRIDE_KEY"
    finally:
        adv.ADVISOR_API_KEY = saved_key


def test_resolve_auth_token_missing_file_returns_empty():
    saved_path = adv.CODEX_AUTH_PATH
    saved_key = adv.ADVISOR_API_KEY
    adv.CODEX_AUTH_PATH = "/nonexistent/path.json"
    adv.ADVISOR_API_KEY = ""
    try:
        assert adv._resolve_auth_token() == ""
    finally:
        adv.CODEX_AUTH_PATH = saved_path
        adv.ADVISOR_API_KEY = saved_key


# ─────────────────────────── JSON-RPC handlers ───────────────────


def test_handle_initialize():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = adv.handle_message(msg)
    assert resp["id"] == 1
    res = resp["result"]
    assert res["protocolVersion"] == "2024-11-05"
    assert res["serverInfo"]["name"] == adv.SERVER_NAME
    assert "tools" in res["capabilities"]


def test_handle_tools_list_returns_one_tool():
    resp = adv.handle_message({"jsonrpc": "2.0", "id": 2,
                               "method": "tools/list", "params": {}})
    tools = resp["result"]["tools"]
    assert len(tools) == 1 and tools[0]["name"] == "ask_advisor"


def test_handle_tools_call_unknown_name_errors():
    resp = adv.handle_message({"jsonrpc": "2.0", "id": 3,
                               "method": "tools/call",
                               "params": {"name": "do_evil", "arguments": {}}})
    assert "error" in resp
    assert resp["error"]["code"] == -32602


def test_handle_tools_call_routes_to_call_advisor():
    captured = {}

    def fake_call(question, context="", previous_attempts=""):
        captured["question"] = question
        captured["context"] = context
        captured["previous_attempts"] = previous_attempts
        return {"text": "advice content", "usage": None, "error": None}

    original = adv.call_advisor
    adv.call_advisor = fake_call  # type: ignore[assignment]
    try:
        resp = adv.handle_message({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "ask_advisor",
                       "arguments": {"question": "Q", "context": "C",
                                     "previous_attempts": "P"}},
        })
    finally:
        adv.call_advisor = original  # type: ignore[assignment]

    assert captured == {"question": "Q", "context": "C",
                        "previous_attempts": "P"}
    res = resp["result"]
    assert res["isError"] is False
    assert res["content"][0]["text"] == "advice content"


def test_handle_tools_call_error_sets_isError_true():
    adv_orig = adv.call_advisor
    adv.call_advisor = lambda **kw: {  # type: ignore[assignment]
        "text": "[boom]", "usage": None, "error": "http_500"}
    try:
        resp = adv.handle_message({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "ask_advisor", "arguments": {"question": "x"}},
        })
    finally:
        adv.call_advisor = adv_orig  # type: ignore[assignment]
    assert resp["result"]["isError"] is True


def test_notifications_initialized_returns_none():
    assert adv.handle_message({"jsonrpc": "2.0",
                               "method": "notifications/initialized",
                               "params": {}}) is None


def test_unknown_method_with_id_errors():
    resp = adv.handle_message({"jsonrpc": "2.0", "id": 9,
                               "method": "weirdo", "params": {}})
    assert resp["error"]["code"] == -32601


def test_unknown_method_notification_returns_none():
    # No `id` → notification → no response even for unknown methods.
    assert adv.handle_message({"jsonrpc": "2.0",
                               "method": "weirdo", "params": {}}) is None


def test_invalid_json_payload_handled():
    resp = adv.handle_message({"_invalid": True, "_raw": "garbage"})
    assert resp["error"]["code"] == -32700


# ─────────────────────────── runner ──────────────────────────────


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
