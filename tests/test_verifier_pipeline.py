"""End-to-end verifier pipeline test without the full proxy stack.

Simulates the exact flow:
  1. Accumulate a "bad" SSE response in soft_completion buffer
  2. Run verify_at_stream_end against a fake local backend
  3. Check the flag is set
  4. Run VerifierGate against a GuardContext
  5. Assert force_route = "frontier"

Run: uv run python tests/test_verifier_pipeline.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── Fake /chat/completions backend ────────────────────────────────────────────


class _FakeChatBackend(BaseHTTPRequestHandler):
    """Returns a configurable JSON verdict for the verifier."""
    verdict_json: str = (
        '{"task_completion": 2, "output_quality": 1, '
        '"execution_evidence": 1, "reason": "garbage output"}'
    )

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        body = json.loads(raw) if raw else {}
        # Store the messages for inspection
        self.__class__.last_messages = body.get("messages", [])
        payload = json.dumps({
            "choices": [{"message": {"content": self.verdict_json}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ── helpers ──────────────────────────────────────────────────────────────────


def _sse_buf(text: str, finish_reason: str = "stop") -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return (
        f'data: {{"type":"response.output_text.delta","delta":"{escaped}"}}\n\n'
        f'data: {{"type":"response.completed","response":{{'
        f'"finish_reason":"{finish_reason}","status":"completed",'
        f'"usage":{{"output_tokens":50}}}}}}\n\n'
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    # 1. Start fake /chat/completions backend
    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeChatBackend)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"

    # 2. Accumulate a "bad" SSE response in the buffer
    bad_text = (
        "The agent claims it completed the task but the output is clearly "
        "wrong. No verification was run. The code has syntax errors. "
        "The file was not actually written to disk. " * 5
    )  # well over 100 chars

    from tinyctx import soft_completion
    proj_sid = "test_proj_verify"
    soft_completion.reset_stream(proj_sid)
    buf = _sse_buf(bad_text, "stop")
    soft_completion.accumulate_chunk(proj_sid, buf.encode())

    # 3. Run the verifier
    from tinyctx import verifier
    import asyncio

    diag = asyncio.run(verifier.verify_at_stream_end(
        proj_sid,
        local_base_url=base,
        local_model="fake-model",
        timeout_s=10.0,
        threshold=8,
        user_goal="Write a Python sorting function with tests",
        tool_summary="total_tool_calls=2; last=['write', 'bash']",
        conv_sid="test_conv",
    ))

    assert diag.result is not None, f"verifier should produce result, got {diag}"
    assert diag.result.passed is False, "bad output should fail verification"
    assert diag.result.criteria.total < 8, \
        f"total {diag.result.criteria.total} should be < 8"

    flag = verifier.get_flag(proj_sid)
    assert flag is not None, "verifier flag should be set"
    assert flag["active"] is True
    print(f"  [OK] Verifier scored total={diag.result.criteria.total}/15: "
          f"{diag.result.reason}")

    # 4. Check verifier prompt content
    msgs = _FakeChatBackend.last_messages
    sys_msg = msgs[0]["content"] if msgs else ""
    assert "output-quality auditor" in sys_msg, \
        f"verifier system prompt mismatch: {sys_msg[:100]}"
    user_msg = msgs[1]["content"] if len(msgs) > 1 else ""
    assert "user_goal:" in user_msg, "user content missing user_goal"
    assert "tool_summary:" in user_msg, "user content missing tool_summary"
    assert "assistant_text:" in user_msg, "user content missing assistant_text"
    print(f"  [OK] Verifier prompt correct ({len(sys_msg)} + {len(user_msg)} chars)")

    # 5. Simulate VerifierGate consuming the flag
    from tinyctx.guards import VerifierGate, GuardContext

    verifier._set_flag_for_test(proj_sid, total=4, reason="test bad output")

    ctx = GuardContext(
        body={"input": [{"role": "user", "content": "continue"}]},
        proj_sid=proj_sid,
        conv_sid="test_conv",
        turn_count=1,
        is_compaction=False,
        forced_by_client_model=False,
    )

    gate = VerifierGate()
    result = gate.apply(ctx)
    assert result.fired is True, f"VerifierGate should fire, got {result}"
    assert ctx.force_route == "frontier", \
        f"force_route should be frontier, got {ctx.force_route}"
    print(f"  [OK] VerifierGate fired: force_route={ctx.force_route}")

    # 6. Guard skip: compaction
    ctx2 = GuardContext(
        body={"input": [{"role": "user", "content": "x"}]},
        proj_sid=proj_sid,
        conv_sid="test_conv",
        turn_count=1,
        is_compaction=True,
        forced_by_client_model=False,
    )
    verifier._set_flag_for_test(proj_sid, total=4)
    r2 = gate.apply(ctx2)
    assert r2.fired is False, f"should skip on compaction, got {r2}"
    print(f"  [OK] VerifierGate skipped on compaction")

    # 7. Guard skip: force_route already set
    ctx3 = GuardContext(
        body={"input": [{"role": "user", "content": "x"}]},
        proj_sid=proj_sid,
        conv_sid="test_conv",
        turn_count=1,
        is_compaction=False,
        forced_by_client_model=False,
    )
    ctx3.force_route = "frontier"  # already set by higher-priority guard
    verifier._set_flag_for_test(proj_sid, total=4)
    r3 = gate.apply(ctx3)
    assert r3.fired is False, f"should skip when force_route already set, got {r3}"
    print(f"  [OK] VerifierGate skipped when force_route already set")

    # Cleanup
    verifier.reset_state()
    soft_completion.reset_stream(proj_sid)
    httpd.shutdown()

    print()
    print("=" * 50)
    print("ALL VERIFIER PIPELINE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
