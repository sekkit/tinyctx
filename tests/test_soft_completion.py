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


# ─── auto-force-frontier on high-confidence PUNT ──────────────────────────


def test_auto_force_frontier_fires_on_high_p_punt():
    """High-confidence PUNT verdict (p ≥ force_frontier_threshold) sets
    the empty_response_guard flag so the next turn auto-routes to
    frontier — the deterministic fix that doesn't depend on agent
    self-discipline or codex parsing synthetic events."""
    import asyncio
    from tinyctx import soft_completion, empty_response_guard
    soft_completion.reset_state()
    empty_response_guard.reset_state()

    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())

    # Backend says PUNT with high p
    httpd, port = _spawn_fake_backend(
        '{"soft_punt": true, "p": 0.95, "reason": "premature done"}')
    try:
        asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7,
                force_frontier_threshold=0.85))
        # gate flag set (existing behavior)
        assert soft_completion.get_flag("p1") is not None
        # NEW: empty_response_guard flag also set → next turn forces frontier
        flag = empty_response_guard.peek_force_frontier("p1")
        assert flag is not None
        assert "soft_punt" in flag["reason"]
        assert "0.95" in flag["reason"]
        assert "premature done" in flag["reason"]
    finally:
        httpd.shutdown()


def test_auto_force_frontier_skips_below_threshold():
    """PUNT with p < force_frontier_threshold → only gate fires, no
    frontier escalation."""
    import asyncio
    from tinyctx import soft_completion, empty_response_guard
    soft_completion.reset_state()
    empty_response_guard.reset_state()

    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())

    # Backend says PUNT but p just barely above the gate threshold (0.7),
    # below the force-frontier threshold (0.85)
    httpd, port = _spawn_fake_backend(
        '{"soft_punt": true, "p": 0.75, "reason": "borderline"}')
    try:
        asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7,
                force_frontier_threshold=0.85))
        # gate fires
        assert soft_completion.get_flag("p1") is not None
        # but force-frontier does NOT fire
        assert empty_response_guard.peek_force_frontier("p1") is None
    finally:
        httpd.shutdown()


def test_auto_force_frontier_skips_on_not_punt():
    """soft_punt:false → no gate, no force-frontier."""
    import asyncio
    from tinyctx import soft_completion, empty_response_guard
    soft_completion.reset_state()
    empty_response_guard.reset_state()

    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())

    httpd, port = _spawn_fake_backend(
        '{"soft_punt": false, "p": 0.95, "reason": "substantive answer"}')
    try:
        asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7,
                force_frontier_threshold=0.85))
        assert soft_completion.get_flag("p1") is None
        assert empty_response_guard.peek_force_frontier("p1") is None
    finally:
        httpd.shutdown()


def test_auto_force_frontier_default_disabled_when_threshold_is_1_01():
    """force_frontier_threshold=1.01 (the default sentinel) means feature
    OFF — even p=1.0 PUNT verdicts won't escalate. Used by callers that
    want classic gate behavior only."""
    import asyncio
    from tinyctx import soft_completion, empty_response_guard
    soft_completion.reset_state()
    empty_response_guard.reset_state()

    sse = 'data: {"delta":"' + 'x' * 250 + '"}\n\n'
    soft_completion.accumulate_chunk("p1", sse.encode())

    httpd, port = _spawn_fake_backend(
        '{"soft_punt": true, "p": 1.0, "reason": "max conf"}')
    try:
        asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))  # force_frontier_threshold defaults to 1.01
        assert soft_completion.get_flag("p1") is not None
        assert empty_response_guard.peek_force_frontier("p1") is None
    finally:
        httpd.shutdown()


def test_default_config_auto_force_frontier_enabled_with_sane_threshold():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.soft_completion_auto_force_frontier_enabled is True
    # Higher than gate's 0.7, lower than 1.0
    assert 0.7 < cfg.soft_completion_auto_force_frontier_threshold < 1.0


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


# ─── context extraction (semantic classifier inputs) ──────────────────────


def test_extract_user_goal_string_content():
    from tinyctx.soft_completion import extract_user_goal
    body_input = [
        {"role": "user", "content": "do task A"},
        {"type": "function_call", "name": "shell", "arguments": "{}"},
        {"role": "user", "content": "actually, do task B instead"},
    ]
    # Returns the most recent user message
    assert extract_user_goal(body_input) == "actually, do task B instead"


def test_extract_user_goal_typed_content():
    from tinyctx.soft_completion import extract_user_goal
    body_input = [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "extract this"},
            {"type": "input_text", "text": "and this"},
        ]},
    ]
    out = extract_user_goal(body_input)
    assert "extract this" in out and "and this" in out


def test_extract_user_goal_empty_when_no_user():
    from tinyctx.soft_completion import extract_user_goal
    assert extract_user_goal([]) == ""
    assert extract_user_goal(None) == ""
    assert extract_user_goal([{"type": "function_call", "name": "x"}]) == ""


def test_extract_user_goal_caps_chars():
    from tinyctx.soft_completion import extract_user_goal
    body_input = [{"role": "user", "content": "x" * 5000}]
    out = extract_user_goal(body_input, max_chars=200)
    assert len(out) <= 200


def test_extract_progress_tracker_from_function_call():
    from tinyctx.soft_completion import extract_progress_tracker
    args = _json.dumps({
        "explanation": "step plan",
        "plan": [
            {"step": "create model", "status": "completed"},
            {"step": "run tests", "status": "in_progress"},
            {"step": "deploy", "status": "pending"},
        ],
    })
    body_input = [
        {"role": "user", "content": "do plan"},
        {"type": "function_call", "name": "update_plan", "arguments": args,
         "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1",
         "output": "Updated plan"},
    ]
    out = extract_progress_tracker(body_input)
    assert "create model" in out
    assert "completed" in out
    assert "in_progress" in out
    assert "deploy" in out


def test_extract_progress_tracker_uses_latest():
    """When multiple update_plan calls exist, return the LATEST state."""
    from tinyctx.soft_completion import extract_progress_tracker
    a1 = _json.dumps({"plan": [{"step": "X", "status": "pending"}]})
    a2 = _json.dumps({"plan": [{"step": "X", "status": "completed"}]})
    body_input = [
        {"type": "function_call", "name": "update_plan", "arguments": a1,
         "call_id": "c1"},
        {"type": "function_call", "name": "update_plan", "arguments": a2,
         "call_id": "c2"},
    ]
    out = extract_progress_tracker(body_input)
    assert "completed" in out  # latest status, not pending


def test_extract_progress_tracker_empty_when_no_plan():
    from tinyctx.soft_completion import extract_progress_tracker
    body_input = [
        {"role": "user", "content": "go"},
        {"type": "function_call", "name": "shell", "arguments": "{}"},
    ]
    assert extract_progress_tracker(body_input) == ""


def test_extract_tool_summary_counts_and_lists_recent():
    from tinyctx.soft_completion import extract_tool_summary
    body_input = [
        {"type": "function_call", "name": "shell", "arguments": "{}"},
        {"type": "function_call", "name": "apply_patch", "arguments": "{}"},
        {"type": "function_call", "name": "shell", "arguments": "{}"},
    ]
    out = extract_tool_summary(body_input)
    assert "total_tool_calls=3" in out
    assert "shell" in out
    assert "apply_patch" in out


def test_extract_tool_summary_empty_when_no_tools():
    from tinyctx.soft_completion import extract_tool_summary
    assert extract_tool_summary([]) == "no_tool_calls"
    assert extract_tool_summary(None) == "no_tool_calls"
    assert extract_tool_summary(
        [{"role": "user", "content": "hi"}]) == "no_tool_calls"


def test_extract_tool_summary_caps_at_last_n():
    from tinyctx.soft_completion import extract_tool_summary
    body_input = [
        {"type": "function_call", "name": f"tool_{i}", "arguments": "{}"}
        for i in range(50)
    ]
    out = extract_tool_summary(body_input, last_n=5)
    assert "total_tool_calls=50" in out
    # Last 5 should be in summary
    assert "tool_45" in out and "tool_49" in out


# ─── finish_reason extraction & short-circuits ─────────────────────────────


def test_extract_finish_reason_tool_calls():
    """The most common case: agent ended with finish_reason=tool_calls."""
    from tinyctx.soft_completion import _extract_finish_reason
    buf = (
        'data: {"choices":[{"delta":{},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[...]}}]}\n\n'
        'data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}\n\n'
        'data: [DONE]\n\n'
    )
    assert _extract_finish_reason(buf) == "tool_calls"


def test_extract_finish_reason_stop():
    from tinyctx.soft_completion import _extract_finish_reason
    buf = (
        'data: {"choices":[{"delta":{"content":"plan"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n'
    )
    assert _extract_finish_reason(buf) == "stop"


def test_extract_finish_reason_length_truncated():
    from tinyctx.soft_completion import _extract_finish_reason
    buf = 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
    assert _extract_finish_reason(buf) == "length"


def test_extract_finish_reason_none_when_only_null():
    """All events show null finish_reason → return None."""
    from tinyctx.soft_completion import _extract_finish_reason
    buf = (
        'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"y"},"finish_reason":null}]}\n\n'
    )
    assert _extract_finish_reason(buf) is None


def test_extract_finish_reason_none_when_no_field():
    from tinyctx.soft_completion import _extract_finish_reason
    assert _extract_finish_reason("") is None
    assert _extract_finish_reason('data: not even json\n\n') is None


def test_classify_short_circuits_tool_calls_without_llm():
    """finish_reason=tool_calls → skip with reason `tool_calls_finish`,
    NEVER call the LLM. Saves cost on every tool-call turn (the
    overwhelming majority of agent turns)."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    sse = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"{\\"file\\":\\"x.py\\"}"}}]},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}\n\n'
    )
    soft_completion.accumulate_chunk("p1", sse.encode())

    # Pass an unreachable URL — if LLM is called, this would raise / time out.
    diag = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end_diag(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",  # unreachable
            local_model="fake",
            timeout_s=0.5,
            threshold=0.7))
    assert diag.result is None
    assert diag.skipped_reason == "tool_calls_finish"
    assert diag.finish_reason == "tool_calls"
    assert diag.backend_error == ""  # didn't reach backend


def test_classify_short_text_threshold_50():
    """Short text + finish=stop → skip without LLM (brief confirmation)."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    # 30 chars of text + finish=stop (< 50 threshold)
    sse = (
        'data: {"choices":[{"delta":{"content":"OK done."},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n'
    )
    soft_completion.accumulate_chunk("p1", sse.encode())
    diag = asyncio.new_event_loop().run_until_complete(
        soft_completion.classify_at_stream_end_diag(
            "p1",
            local_base_url="http://127.0.0.1:1/v1",
            local_model="fake",
            timeout_s=0.5,
            threshold=0.7))
    assert diag.skipped_reason == "short_text"
    assert diag.finish_reason == "stop"
    assert diag.extracted_text_chars < 50


def test_classify_user_content_includes_finish_reason_metadata():
    """Verify the LLM gets finish_reason as part of its input — not just
    text. Without this, the prompt's plan-without-action rule would have
    no signal to fire on (since the text alone is just a plan)."""
    from tinyctx import soft_completion
    soft_completion.reset_state()
    # Long plan text + finish=stop (the case the user hit at turn 1425)
    plan = ("Here's my plan: 1. Re-confirm sample current commit. "
            "2. Trace the XR session blank screencap chain. "
            "3. Locate the minimum fix point. "
            "4. Rebuild APK and verify with logcat + screencap. ") * 5
    sse = (
        f'data: {{"choices":[{{"delta":{{"content":"{plan}"}},"finish_reason":null}}]}}\n\n'
        f'data: {{"choices":[{{"delta":{{"content":""}},"finish_reason":"stop"}}]}}\n\n'
    )
    soft_completion.accumulate_chunk("p1", sse.encode())

    # Capture what payload is sent to the backend
    captured = {}

    class _CapturingHandler(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n) if n else b""
            captured["body"] = body
            payload = _json.dumps({
                "choices": [{"message": {"content":
                    '{"soft_punt": true, "p": 0.9, "reason": "plan without action"}'
                }}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _CapturingHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        diag = asyncio.new_event_loop().run_until_complete(
            soft_completion.classify_at_stream_end_diag(
                "p1",
                local_base_url=f"http://127.0.0.1:{port}/v1",
                local_model="fake",
                threshold=0.7))
        assert diag.result is not None
        assert diag.result.soft_punt is True
        # The LLM saw the structured context in its user message —
        # post-semantic-redesign the format is `finish_reason: stop` /
        # `text_chars: N` / sections for user_goal / tracker / tools
        body_str = captured.get("body", b"").decode("utf-8", "replace")
        assert "finish_reason: stop" in body_str
        assert "text_chars:" in body_str
        # New semantic context sections
        assert "user_goal:" in body_str
        assert "progress_tracker:" in body_str
        assert "tool_summary:" in body_str
        assert "assistant_text:" in body_str
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
    # Three-path gate: A=plan-execute, B=question-via-advisor, C=premature-done
    assert "PATH A" in text and "LISTED STEPS" in text
    assert "PATH B" in text and "spawn_agent(role=\"advisor\"" in text
    assert "PATH C" in text and "tracker" in text.lower()
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
