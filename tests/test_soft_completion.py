"""Soft-completion gate v2: LLM-based behavioral classifier.

Tests cover:
  - JSON parser (clean / fenced / truncated / garbage / missing fields)
  - delta-text extractor for SSE-wrapped buffers
  - accumulator + buffer cap
  - classifier full HTTP flow (fake backend) — positive verdict, negative,
    short-text skip, backend error
  - gate injection idempotency + body immutability
  - per-session isolation
"""
from __future__ import annotations

import asyncio
import json as _json
import socket
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


# ─── parser ────────────────────────────────────────────────────────────────


def test_parse_response_clean_json():
    from tinyctx.soft_completion import _parse_response
    r = _parse_response('{"soft_punt": true, "p": 0.9, "reason": "asks user"}')
    assert r is not None
    assert r.soft_punt is True
    assert r.p == 0.9
    assert r.reason == "asks user"


def test_parse_response_markdown_fenced():
    from tinyctx.soft_completion import _parse_response
    text = "Sure:\n```json\n{\"soft_punt\": false, \"p\": 0.8, \"reason\": \"tool call\"}\n```"
    r = _parse_response(text)
    assert r is not None
    assert r.soft_punt is False
    assert r.reason == "tool call"


def test_parse_response_strips_thinking():
    """Reasoning-class models leak <think>...</think> into content."""
    from tinyctx.soft_completion import _parse_response
    text = '<think>Reading the response...</think>\n{"soft_punt": true, "p": 0.85}'
    r = _parse_response(text)
    assert r is not None
    assert r.soft_punt is True


def test_parse_response_clamps_p():
    from tinyctx.soft_completion import _parse_response
    r = _parse_response('{"soft_punt": true, "p": 5.0, "reason": "x"}')
    assert r is not None
    assert r.p == 1.0


def test_parse_response_salvages_truncated():
    """Reasoning models occasionally get cut off mid-reason."""
    from tinyctx.soft_completion import _parse_response
    truncated = '{"soft_punt": true, "p": 0.95, "reason": "asks user without veri'
    r = _parse_response(truncated)
    assert r is not None
    assert r.soft_punt is True
    assert r.p == 0.95


def test_parse_response_garbage_returns_none():
    from tinyctx.soft_completion import _parse_response
    assert _parse_response("") is None
    assert _parse_response("just prose, no JSON") is None


# ─── delta-text extractor ─────────────────────────────────────────────────


def test_extract_text_from_responses_api_sse():
    """Responses-API events: {"type":"...delta","delta":"..."}"""
    from tinyctx.soft_completion import _extract_text_from_buffer
    buf = (
        'data: {"type":"response.output_text.delta","delta":"Hello "}\n\n'
        'data: {"type":"response.output_text.delta","delta":"world!"}\n\n'
    )
    text = _extract_text_from_buffer(buf)
    assert "Hello" in text
    assert "world" in text


def test_extract_text_from_chat_completions_sse():
    """Chat-completions: {"choices":[{"delta":{"content":"..."}}]}"""
    from tinyctx.soft_completion import _extract_text_from_buffer
    buf = (
        'data: {"choices":[{"delta":{"content":"What "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"would"}}]}\n\n'
    )
    text = _extract_text_from_buffer(buf)
    assert "What" in text
    assert "would" in text


def test_extract_text_decodes_json_escapes():
    """Newlines / quotes inside delta strings are JSON-escaped on the
    wire; extractor must decode them back to real chars."""
    from tinyctx.soft_completion import _extract_text_from_buffer
    buf = 'data: {"delta":"line1\\nline2 with \\"quote\\""}\n\n'
    text = _extract_text_from_buffer(buf)
    assert "line1\nline2" in text
    assert '"quote"' in text


def test_extract_text_no_match_returns_tail():
    """If buffer doesn't look like SSE-JSON (e.g. raw text), return
    the tail directly so the LLM can still read it."""
    from tinyctx.soft_completion import _extract_text_from_buffer
    raw = "x" * 5000 + "tail content"
    text = _extract_text_from_buffer(raw)
    # tail content is at the very end, within the cap
    assert "tail content" in text
    assert len(text) <= 4000


# ─── accumulator ──────────────────────────────────────────────────────────


def test_accumulate_chunk_buffers_bytes():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("p1", b"chunk1 ")
    soft_completion.accumulate_chunk("p1", b"chunk2")
    snap = soft_completion.state_snapshot("p1")
    assert snap["buffer_chars"] >= len("chunk1 chunk2")


def test_accumulate_chunk_caps_buffer():
    """Buffer must cap at _BUFFER_MAX so memory is bounded even on
    huge streams."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    big = b"x" * 100_000  # > _BUFFER_MAX (64KB)
    soft_completion.accumulate_chunk("p1", big)
    snap = soft_completion.state_snapshot("p1")
    assert snap["buffer_chars"] <= soft_completion._BUFFER_MAX


def test_reset_stream_clears_buffer():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("p1", b"data")
    soft_completion.reset_stream("p1")
    snap = soft_completion.state_snapshot("p1")
    assert snap["buffer_chars"] == 0


# ─── per-session isolation ─────────────────────────────────────────────────


def test_state_isolated_by_proj_sid():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("projA", b"A content")
    soft_completion.accumulate_chunk("projB", b"B content")
    assert "A content" in soft_completion._OUTPUT_BUFFER["projA"]
    assert "B content" in soft_completion._OUTPUT_BUFFER["projB"]
    assert "B" not in soft_completion._OUTPUT_BUFFER["projA"]


def test_force_flag_isolated_by_proj_sid():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("projA", reason="test", p=0.9)
    assert soft_completion.get_flag("projA") is not None
    assert soft_completion.get_flag("projB") is None


# ─── classifier full HTTP flow ────────────────────────────────────────────


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _spawn_fake_backend(scripted: str) -> tuple[ThreadingHTTPServer, int]:
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n: self.rfile.read(n)
            payload = _json.dumps({
                "choices": [{"message": {"content": scripted}}]
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


def test_classify_at_stream_end_positive_sets_flag():
    """LLM verdict soft_punt:true → flag set on session."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    # Buffer some realistic SSE-shaped content
    sse = ('data: {"type":"response.output_text.delta",'
           '"delta":"' + 'x' * 250 + '"}\n\n')
    soft_completion.accumulate_chunk("p1", sse.encode())
    httpd, port = _spawn_fake_backend(
        '{"soft_punt": true, "p": 0.9, "reason": "asks user what next"}')
    try:
        result = asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))
        assert result is not None
        assert result.soft_punt is True
        flag = soft_completion.get_flag("p1")
        assert flag is not None
        assert "asks user" in flag["matched_pattern"]
    finally:
        httpd.shutdown()


def test_classify_at_stream_end_negative_no_flag():
    """LLM verdict soft_punt:false → flag NOT set."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())
    httpd, port = _spawn_fake_backend(
        '{"soft_punt": false, "p": 0.9, "reason": "tool call"}')
    try:
        result = asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))
        assert result is not None
        assert result.soft_punt is False
        assert soft_completion.get_flag("p1") is None
    finally:
        httpd.shutdown()


def test_classify_below_threshold_no_flag():
    """soft_punt:true but p < threshold → flag NOT set (calibration
    safety: only act on high-confidence verdicts)."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())
    httpd, port = _spawn_fake_backend(
        '{"soft_punt": true, "p": 0.5, "reason": "borderline"}')
    try:
        result = asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))
        assert result is not None
        assert result.soft_punt is True
        assert result.p == 0.5
        assert soft_completion.get_flag("p1") is None  # below threshold
    finally:
        httpd.shutdown()


def test_classify_diag_short_text_returns_skipped_reason():
    """Diag path: short text → skipped_reason='short_text', no backend call."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("p1", b"hi")  # < 200 chars
    diag = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end_diag(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",  # would fail if reached
            local_model="fake",
            timeout_s=0.5,
            threshold=0.7))
    assert diag.result is None
    assert diag.skipped_reason == "short_text"
    assert diag.backend_error == ""


def test_classify_diag_no_buffer_returns_no_buffer():
    """Diag path: empty buffer → skipped_reason='no_buffer'."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    diag = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end_diag(
            "never_seen",
            local_base_url="http://127.0.0.1:1/v1",
            local_model="fake",
            timeout_s=0.5))
    assert diag.result is None
    assert diag.skipped_reason == "no_buffer"


def test_classify_diag_backend_error_captured():
    """Diag path: backend unreachable → backend_error populated."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())
    diag = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end_diag(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",
            local_model="fake",
            timeout_s=0.5))
    assert diag.result is None
    assert diag.backend_error  # non-empty
    assert diag.skipped_reason == ""
    # Extracted text was captured even though backend failed
    assert diag.extracted_text_chars > 0


def test_classify_diag_parse_failed_captures_raw():
    """Diag path: backend returns garbage → raw_content_preview populated."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())
    httpd, port = _spawn_fake_backend("just prose, no JSON at all")
    try:
        diag = asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))
        assert diag.result is None
        assert diag.backend_error == ""
        assert diag.skipped_reason == ""
        assert "just prose" in diag.raw_content_preview
        assert diag.backend_status == 200
    finally:
        httpd.shutdown()


def test_classify_skips_short_text():
    """Output too short to be meaningful → skip without backend call."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("p1", b"hi")  # way under 200 chars
    # Backend should not even be reached; pass an unreachable URL
    result = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",  # unreachable
            local_model="fake",
            timeout_s=0.5,
            threshold=0.7))
    assert result is None
    assert soft_completion.get_flag("p1") is None


def test_classify_backend_error_silent_fallback():
    """Backend unreachable → returns None silently, no flag set."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())
    result = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",  # unreachable
            local_model="fake",
            timeout_s=0.5,
            threshold=0.7))
    assert result is None
    assert soft_completion.get_flag("p1") is None


# ─── gate injection ────────────────────────────────────────────────────────


def test_gate_no_inject_when_flag_unset():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    body = {"input": [{"role": "user", "content": "hi"}]}
    out, gated, pat = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is False
    assert pat == ""
    assert out is body


def test_gate_injects_when_flag_set():
    """When LLM classifier set the flag, the next request gets the
    advisor-vet reminder appended."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="asks user what next", p=0.9)

    body = {"input": [{"role": "user", "content": "thanks"}]}
    out, gated, pat = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is True
    assert "asks user what next" in pat
    assert len(out["input"]) == 2
    last = out["input"][-1]
    assert last["role"] == "user"
    text = last["content"][0]["text"]
    assert "<system-reminder>" in text
    assert "soft-completion gate" in text
    assert "spawn_agent(role=\"advisor\"" in text
    assert "ask:" in text and "work:" in text


def test_gate_fires_once_then_clears_flag():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="x", p=1.0)

    body = {"input": [{"role": "user", "content": "x"}]}
    _, g1, _ = soft_completion.maybe_inject_soft_completion_gate(body, "p1")
    assert g1 is True
    body2 = {"input": [{"role": "user", "content": "y"}]}
    _, g2, _ = soft_completion.maybe_inject_soft_completion_gate(body2, "p1")
    assert g2 is False


def test_gate_does_not_mutate_original_body():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="x", p=1.0)
    original = {"input": [{"role": "user", "content": "x"}]}
    items_id = id(original["input"])
    out, gated, _ = soft_completion.maybe_inject_soft_completion_gate(
        original, "p1")
    assert gated is True
    assert len(original["input"]) == 1
    assert id(original["input"]) == items_id
    assert len(out["input"]) == 2


def test_gate_skips_malformed_body():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion._set_flag_for_test("p1", reason="x", p=1.0)
    body = {"messages": [{"role": "user", "content": "hi"}]}  # chat-style
    out, gated, _ = soft_completion.maybe_inject_soft_completion_gate(
        body, "p1")
    assert gated is False


# ─── default config + trace fields ────────────────────────────────────────


def test_config_default_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.soft_completion_gate_enabled is True


def test_trace_fields_default_off():
    from tinyctx.trace import RequestTrace
    t = RequestTrace()
    assert t.soft_completion_detected is False
    assert t.soft_completion_pattern == ""
    assert t.soft_completion_gate_injected is False
    assert t.soft_completion_gate_pattern == ""


# ─── error tolerance ──────────────────────────────────────────────────────


def test_accumulate_tolerates_malformed_utf8():
    from tinyctx import soft_completion
    soft_completion.reset_state()
    soft_completion.accumulate_chunk("p1", b"\xe4 incomplete utf8")
    # Should not raise; chunk is silently coerced via errors='ignore'
