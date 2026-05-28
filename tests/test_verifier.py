"""Tests for tinyctx.verifier — output-quality verifier module."""
from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tinyctx.verifier import (
    VerdictCriteria,
    VerifyDiag,
    VerifyResult,
    _clamp,
    _parse_verdict,
    _set_flag_for_test,
    consume_flag,
    get_flag,
    reset_state,
    state_snapshot,
    verify_at_stream_end,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _spawn_fake_backend(scripted_response: str, *,
                        status: int = 200) -> tuple[ThreadingHTTPServer, int]:
    """Start a fake Chat Completions HTTP server that returns `scripted`."""
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n:
                self.rfile.read(n)
            if status != 200:
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = json.dumps({
                "choices": [{"message": {"content": scripted_response}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _make_sse_buffer(delta_text: str, finish_reason: str = "stop") -> str:
    """Build a minimal SSE buffer with text deltas and a finish_reason."""
    escaped = delta_text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    buf = (
        f'data: {{"type":"response.output_text.delta","delta":"{escaped}"}}\n\n'
        f'data: {{"type":"response.completed","response":{{'
        f'"finish_reason":"{finish_reason}","status":"completed",'
        f'"usage":{{"output_tokens":100}}}}}}\n\n'
    )
    return buf


_LONG_TEXT = (
    "The agent successfully implemented the requested feature. "
    "The code compiles without errors and all tests pass. "
    "The output matches the specification exactly. "
    "Documentation was updated to reflect the changes. "
    "Edge cases are handled correctly and performance is within expected bounds. "
    "The implementation follows the project's coding standards. " * 3
)  # ~400 chars — well above the 100-char short_text floor


def _sse_buf_with_finish(finish_reason: str) -> str:
    """Minimal SSE buffer with a specific finish_reason."""
    return (
        f'data: {{"type":"response.output_text.delta","delta":"Hello world response text here"}}\n\n'
        f'data: {{"type":"response.completed","response":{{'
        f'"finish_reason":"{finish_reason}","status":"completed",'
        f'"usage":{{"output_tokens":10}}}}}}\n\n'
    )


# ── VerdictCriteria tests ────────────────────────────────────────────────────


class TestVerdictCriteria:
    def test_total_sums_three_fields(self):
        c = VerdictCriteria(task_completion=3, output_quality=4, execution_evidence=5)
        assert c.total == 12

    def test_total_zero(self):
        c = VerdictCriteria(task_completion=0, output_quality=0, execution_evidence=0)
        assert c.total == 0


# ── _clamp tests ─────────────────────────────────────────────────────────────


class TestClamp:
    def test_in_range_passes_through(self):
        assert _clamp(3) == 3
        assert _clamp(1) == 1
        assert _clamp(5) == 5

    def test_below_range_clamped(self):
        assert _clamp(0) == 1
        assert _clamp(-5) == 1

    def test_above_range_clamped(self):
        assert _clamp(6) == 5
        assert _clamp(100) == 5


# ── JSON parser tests ────────────────────────────────────────────────────────


class TestParseVerdict:
    def test_parse_clean_json(self):
        resp = '{"task_completion": 4, "output_quality": 5, "execution_evidence": 3, "reason": "all good"}'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 4
        assert c.output_quality == 5
        assert c.execution_evidence == 3

    def test_parse_markdown_fenced(self):
        resp = '```json\n{"task_completion": 2, "output_quality": 1, "execution_evidence": 1, "reason": "bad"}\n```'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 2

    def test_parse_strips_thinking(self):
        resp = '<think>hmm let me evaluate this response carefully</think>\n{"task_completion": 5, "output_quality": 4, "execution_evidence": 4, "reason": "perfect"}'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 5

    def test_parse_clamps_values(self):
        resp = '{"task_completion": 7, "output_quality": 0, "execution_evidence": 3, "reason": "extreme"}'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 5
        assert c.output_quality == 1
        assert c.execution_evidence == 3

    def test_parse_garbage_returns_none(self):
        assert _parse_verdict("") is None
        assert _parse_verdict("just some prose, no JSON here") is None
        assert _parse_verdict(None) is None

    def test_parse_salvages_truncated(self):
        resp = '{"task_completion": 3, "output_quality": 2, "execution_evidence": 1, "reason": "miss'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 3
        assert c.output_quality == 2
        assert c.execution_evidence == 1

    def test_parse_nested_braces(self):
        resp = '{"task_completion": 4, "output_quality": 3, "execution_evidence": 4, "reason": "code block had {some: stuff}"}'
        c = _parse_verdict(resp)
        assert c is not None
        assert c.task_completion == 4

    def test_parse_missing_fields(self):
        resp = '{"task_completion": 2}'
        c = _parse_verdict(resp)
        assert c is None  # need all three: regex requires all fields


# ── VerifyDiag tests ─────────────────────────────────────────────────────────


class TestVerifyDiag:
    def test_defaults(self):
        d = VerifyDiag()
        assert d.result is None
        assert d.skipped_reason == ""
        assert d.backend_error == ""
        assert d.backend_status == 0


# ── flag management tests ────────────────────────────────────────────────────


class TestFlagManagement:
    def teardown_method(self):
        reset_state()

    def test_get_flag_when_unset(self):
        assert get_flag("s1") is None

    def test_set_and_get_flag(self):
        _set_flag_for_test("s1", total=4, reason="bad output")
        f = get_flag("s1")
        assert f is not None
        assert f["total"] == 4
        assert f["active"] is True

    def test_consume_flag_returns_and_clears(self):
        _set_flag_for_test("s1", total=4)
        f = consume_flag("s1")
        assert f is not None
        assert f["total"] == 4
        # Second consume returns None
        assert consume_flag("s1") is None
        assert get_flag("s1") is None

    def test_consume_flag_when_unset(self):
        assert consume_flag("nonexistent") is None

    def test_per_session_isolation(self):
        _set_flag_for_test("p1", total=4)
        _set_flag_for_test("p2", total=12)
        assert get_flag("p1")["total"] == 4
        assert get_flag("p2")["total"] == 12
        consume_flag("p1")
        assert get_flag("p1") is None
        assert get_flag("p2") is not None

    def test_reset_state_clears_all(self):
        _set_flag_for_test("a", total=4)
        _set_flag_for_test("b", total=5)
        reset_state()
        assert get_flag("a") is None
        assert get_flag("b") is None

    def test_reset_state_single_session(self):
        _set_flag_for_test("a", total=4)
        _set_flag_for_test("b", total=5)
        reset_state("a")
        assert get_flag("a") is None
        assert get_flag("b") is not None

    def test_state_snapshot(self):
        _set_flag_for_test("s1", total=6, reason="meh",
                           task_completion=2, output_quality=2,
                           execution_evidence=2)
        snap = state_snapshot("s1")
        assert snap["flag_active"] is True
        assert snap["flag"]["total"] == 6
        assert snap["flag"]["criteria"]["task_completion"] == 2


# ── verify_at_stream_end tests (HTTP) ────────────────────────────────────────


class TestVerifyAtStreamEnd:
    def teardown_method(self):
        reset_state()

    def test_passes_good_output(self):
        resp = '{"task_completion": 5, "output_quality": 5, "execution_evidence": 4, "reason": "excellent work"}'
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer(_LONG_TEXT)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is not None
            assert diag.result.passed is True
            assert diag.result.criteria.total == 14
            assert get_flag("test_proj") is None
        finally:
            httpd.shutdown()

    def test_fails_bad_output(self):
        resp = '{"task_completion": 2, "output_quality": 1, "execution_evidence": 1, "reason": "garbage output"}'
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer(_LONG_TEXT)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is not None
            assert diag.result.passed is False
            assert diag.result.criteria.total == 4
            f = get_flag("test_proj")
            assert f is not None
            assert f["active"] is True
            assert f["total"] == 4
        finally:
            httpd.shutdown()

    def test_fails_borderline(self):
        """Total == threshold should pass (>= threshold)."""
        resp = '{"task_completion": 3, "output_quality": 3, "execution_evidence": 2, "reason": "borderline"}'
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer(_LONG_TEXT)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is not None
            assert diag.result.passed is True  # 8 >= 8
            assert diag.result.criteria.total == 8
        finally:
            httpd.shutdown()

    def test_below_threshold_fails(self):
        """Total == threshold-1 should fail (< threshold)."""
        resp = '{"task_completion": 3, "output_quality": 2, "execution_evidence": 2, "reason": "meh"}'
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer(_LONG_TEXT)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is not None
            assert diag.result.passed is False  # 7 < 8
        finally:
            httpd.shutdown()

    def test_skips_empty_buffer(self):
        """Empty buffer → skipped."""
        diag = _run_verify_sync("", threshold=8)
        assert diag.result is None
        assert diag.skipped_reason == "no_buffer"

    def test_skips_short_text(self):
        """Less than 100 chars extracted → skipped."""
        buf = _sse_buf_with_finish("stop")
        diag = _run_verify_sync(buf, threshold=8)
        assert diag.result is None
        assert diag.skipped_reason == "short_text"

    def test_skips_tool_calls_finish(self):
        buf = _sse_buf_with_finish("tool_calls")
        diag = _run_verify_sync(buf, threshold=8)
        assert diag.result is None
        assert diag.skipped_reason == "tool_calls_finish"

    def test_backend_error_captured(self):
        httpd, port = _spawn_fake_backend("", status=500)
        buf = _make_sse_buffer("some response with enough text to avoid short-circuit " * 5)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is None
            assert diag.backend_error != ""
            assert diag.backend_status == 500
        finally:
            httpd.shutdown()

    def test_backend_unreachable(self):
        buf = _make_sse_buffer("enough text to pass short-circuit check " * 5)
        diag = _run_verify(buf, 19999, threshold=8)  # nothing listening here
        assert diag.result is None
        assert diag.backend_error != ""

    def test_parse_failed_captured(self):
        resp = "not json at all, just prose about scoring"
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer("some text that exceeds the short text floor " * 5)
        try:
            diag = _run_verify(buf, port, threshold=8)
            assert diag.result is None
            assert diag.raw_response_preview != ""
        finally:
            httpd.shutdown()

    def test_respects_custom_threshold(self):
        resp = '{"task_completion": 3, "output_quality": 3, "execution_evidence": 3, "reason": "mid"}'
        httpd, port = _spawn_fake_backend(resp)
        buf = _make_sse_buffer("a moderately ok response text " * 5)
        try:
            diag_high = _run_verify(buf, port, threshold=10)
            assert diag_high.result.passed is False  # 9 < 10

            diag_low = _run_verify(buf, port, threshold=7)
            assert diag_low.result.passed is True  # 9 >= 7
        finally:
            httpd.shutdown()


# ── helpers for async tests ──────────────────────────────────────────────────


def _run_verify(buf: str, port: int, threshold: int = 8) -> VerifyDiag:
    """Synchronous wrapper for the async verify_at_stream_end."""
    import asyncio
    base = f"http://127.0.0.1:{port}/v1"
    return asyncio.run(verify_at_stream_end(
        "test_proj",
        local_base_url=base,
        local_model="fake-model",
        timeout_s=10.0,
        threshold=threshold,
        raw_buffer=buf,
        user_goal="write a function that returns 42",
        tool_summary="total_tool_calls=3; last=['write', 'bash', 'test']",
        conv_sid="test_conv",
    ))


def _run_verify_sync(buf: str, threshold: int = 8) -> VerifyDiag:
    """Run verify_at_stream_end with a bad URL (no backend needed for
    short-circuit path tests)."""
    import asyncio
    return asyncio.run(verify_at_stream_end(
        "test_proj",
        local_base_url="http://127.0.0.1:19999/v1",
        local_model="fake-model",
        timeout_s=2.0,
        threshold=threshold,
        raw_buffer=buf,
        user_goal="test goal",
        tool_summary="no_tool_calls",
    ))
