"""Tests for sanitize.proactive_compact — proxy-side history truncation
that fires when est_tokens crosses a danger threshold.

This is the safety net for codex.app 0.128's "Codex ran out of room" error
when codex's own auto-compact didn't fire (because gpt-5.5 ships with
auto_compact_token_limit=null and the user's profile-scoped override only
applies to one profile, not the default profile being run).
"""
from __future__ import annotations

import json

from tinyctx.sanitize import (
    clear_proactive_cache,
    proactive_compact,
)


def _make_body(n_turns: int, *, with_system_head: bool = False,
               with_codex_compact: bool = False) -> dict:
    """Build a minimal Responses-API body with `n_turns` user/assistant
    pairs in `input`."""
    body: dict = {
        "model": "tinyctx-auto",
        "instructions": (
            "You are performing a CONTEXT CHECKPOINT COMPACTION. "
            "Create a handoff summary for another LLM that will resume the task."
            if with_codex_compact
            else "You are a coding agent."
        ),
        "input": [],
    }
    if with_system_head:
        body["input"].append({
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "system bytes"}],
        })
    for i in range(n_turns):
        body["input"].append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"user turn {i}"}],
        })
        body["input"].append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"assistant reply {i}"}],
        })
    return body


def test_proactive_compact_below_threshold_noop():
    body = _make_body(20)
    out, info = proactive_compact(
        body,
        session_id="s1",
        est_tokens=50_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "below_threshold"
    # body unchanged
    assert out is body or out == body


def test_proactive_compact_skips_codex_compaction_request():
    body = _make_body(50, with_codex_compact=True)
    out, info = proactive_compact(
        body,
        session_id="s2",
        est_tokens=300_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "skip_codex_compaction"


def test_proactive_compact_skips_when_no_input_array():
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    out, info = proactive_compact(
        body,
        session_id="s3",
        est_tokens=300_000,
        threshold_tokens=200_000,
    )
    assert info["applied"] is False
    assert info["reason"] == "no_input_array"


def test_proactive_compact_skips_when_too_few_items():
    body = _make_body(2)  # 4 items < recent_keep(8) + 3
    out, info = proactive_compact(
        body,
        session_id="s4",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is False
    assert "too_few_items" in info["reason"]


def test_proactive_compact_truncates_middle_default_placeholder():
    clear_proactive_cache()
    body = _make_body(30)  # 60 items
    n_before = len(body["input"])
    out, info = proactive_compact(
        body,
        session_id="s5",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is True
    assert info["items_before"] == n_before
    # head=0 (no system items) + 1 summary + 8 recent = 9
    assert info["items_after"] == 9
    assert info["middle_items_compacted"] == n_before - 8
    assert "compacted" in info["reason"]
    assert info["cached"] is False

    # Last 8 items should be the same as the original last 8.
    assert out["input"][-8:] == body["input"][-8:]
    # The summary item is in the middle.
    summary_item = out["input"][-9]
    assert summary_item["role"] == "user"
    assert summary_item["type"] == "message"
    text = summary_item["content"][0]["text"]
    assert "tinyctx auto-compact" in text
    assert "older turns" in text or "omitted to fit context" in text


def test_proactive_compact_keeps_system_head_items():
    clear_proactive_cache()
    body = _make_body(20, with_system_head=True)
    out, info = proactive_compact(
        body,
        session_id="s6",
        est_tokens=300_000,
        threshold_tokens=200_000,
        recent_keep=8,
    )
    assert info["applied"] is True
    # First item should still be the system message we put in
    first = out["input"][0]
    assert first["role"] == "system"
    # Then summary
    assert out["input"][1]["role"] == "user"
    text = out["input"][1]["content"][0]["text"]
    assert "tinyctx auto-compact" in text
    # Then 8 recent
    assert len(out["input"]) == 1 + 1 + 8


def test_proactive_compact_caches_summary_for_same_middle():
    clear_proactive_cache()
    body = _make_body(30)
    out1, info1 = proactive_compact(
        body, session_id="s7", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info1["applied"] is True
    assert info1["cached"] is False

    # Identical body again — should hit cache
    out2, info2 = proactive_compact(
        body, session_id="s7", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info2["applied"] is True
    assert info2["cached"] is True


def test_proactive_compact_uses_summarizer_when_provided():
    clear_proactive_cache()
    body = _make_body(30)
    captured_blob: dict = {}

    def fake_summarizer(blob: str) -> str:
        captured_blob["blob"] = blob
        return "FAKE SUMMARY: 30 turns about adding compact support."

    out, info = proactive_compact(
        body, session_id="s8", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=fake_summarizer,
    )
    assert info["applied"] is True
    summary_text = out["input"][-9]["content"][0]["text"]
    assert "FAKE SUMMARY" in summary_text
    # Summarizer was given a blob with the middle turns
    assert "user turn 0" in captured_blob["blob"]
    assert "assistant reply 0" in captured_blob["blob"]


def test_proactive_compact_summarizer_failure_falls_back_to_placeholder():
    clear_proactive_cache()
    body = _make_body(30)

    def crashing_summarizer(blob: str) -> str:
        raise RuntimeError("backend dead")

    out, info = proactive_compact(
        body, session_id="s9", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=crashing_summarizer,
    )
    # NEVER fail the request — only quality regression
    assert info["applied"] is True
    summary_text = out["input"][-9]["content"][0]["text"]
    assert "backend dead" in summary_text or "summarizer failed" in summary_text


def test_proactive_compact_does_not_mutate_input_body():
    clear_proactive_cache()
    body = _make_body(30)
    snapshot = json.dumps(body, sort_keys=True)
    proactive_compact(
        body, session_id="s10", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    # original body must be untouched (deepcopy semantics)
    assert json.dumps(body, sort_keys=True) == snapshot


def test_proactive_compact_synthesizes_stub_for_orphan_tool_output():
    """When the tail contains a function_call_output whose matching
    function_call lived in the (now compacted) middle, the upstream
    Responses API rejects with 'No tool call found for function call
    output'. We must synthesize a function_call stub immediately before
    the output so the pair is structurally valid.

    This is the exact bug that caused chatgpt.com to 400 today at
    15:44:09 after 5 successful 200s — the 6th request happened to slice
    on a boundary that orphaned a tool result.
    """
    clear_proactive_cache()
    items: list = []
    # 30 plain user/assistant turns (60 items) — these will be "middle"
    for i in range(30):
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": f"u{i}"}]})
        items.append({"type": "message", "role": "assistant",
                      "content": [{"type": "output_text", "text": f"a{i}"}]})
    # Then the last 8 (recent_keep) items include an orphan output:
    # function_call lives at index 60 (will be in middle, dropped),
    # function_call_output lives at index -3 (in tail, surviving).
    items[59] = {  # replace one middle assistant turn with the original call
        "type": "function_call", "call_id": "call_ABC",
        "name": "shell", "arguments": '{"command":["ls"]}',
    }
    items.append({"type": "function_call_output", "call_id": "call_ABC",
                  "output": "file1.txt\n"})
    # Add 7 more recent items to round out tail = 8
    for i in range(7):
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": f"recent_u{i}"}]})

    body = {"model": "tinyctx", "instructions": "x", "input": items}
    out, info = proactive_compact(
        body, session_id="orphan-test", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info["applied"] is True
    assert info["synthetic_call_stubs"] == 1, (
        f"expected 1 synthetic stub, got {info['synthetic_call_stubs']}"
    )

    # Check that the output_call_ABC has its matching synthetic call right before it
    new_input = out["input"]
    found_pair = False
    for i, it in enumerate(new_input):
        if (isinstance(it, dict)
                and it.get("type") == "function_call_output"
                and it.get("call_id") == "call_ABC"):
            # Look at the item just before
            assert i > 0
            prev = new_input[i - 1]
            assert prev.get("type") == "function_call"
            assert prev.get("call_id") == "call_ABC"
            assert prev.get("name") == "tinyctx_compacted_call"
            found_pair = True
            break
    assert found_pair, "synthetic call stub was not placed before the orphan output"


def test_proactive_compact_no_stub_when_tail_pair_intact():
    """If a function_call AND its function_call_output both live in the
    tail (or both in head), no synthetic stub is needed."""
    clear_proactive_cache()
    items: list = []
    for i in range(30):
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": f"u{i}"}]})
        items.append({"type": "message", "role": "assistant",
                      "content": [{"type": "output_text", "text": f"a{i}"}]})
    # Add a clean tool_call/output pair INSIDE the tail (last 8)
    items.append({"type": "function_call", "call_id": "call_XYZ",
                  "name": "shell", "arguments": '{"command":["ls"]}'})
    items.append({"type": "function_call_output", "call_id": "call_XYZ",
                  "output": "ok"})
    for i in range(6):
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": f"r{i}"}]})

    body = {"model": "tinyctx", "instructions": "x", "input": items}
    out, info = proactive_compact(
        body, session_id="pair-intact-test", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
    )
    assert info["applied"] is True
    assert info["synthetic_call_stubs"] == 0


def test_local_summarizer_calls_backend_and_returns_text():
    """proxy._make_local_summarizer should POST to the local backend's
    /chat/completions endpoint and return the assistant message text.

    Fake the upstream with a small HTTPServer in a thread; verify the
    summarizer hits it with the right payload shape and returns the
    reply.
    """
    import json as _json
    import socket
    import threading
    from contextlib import closing
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received: list[dict] = []

    class _FakeChat(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n else b""
            received.append(_json.loads(raw))
            reply = _json.dumps({
                "choices": [{"message": {"role": "assistant",
                                          "content": "FAKE HANDOFF\n## What we were trying to do\nfoo"}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeChat)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        from tinyctx.config import BackendCfg
        from tinyctx.proxy import _make_local_summarizer
        backend = BackendCfg(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key_env="DOES_NOT_EXIST_ENV",
            model="fake-deepseek",
            wire_api="chat",
        )
        summarize = _make_local_summarizer(backend)
        out = summarize("a very large blob of conversation history")

        assert "FAKE HANDOFF" in out
        # check what we sent upstream
        assert len(received) == 1
        sent = received[0]
        assert sent["model"] == "fake-deepseek"
        assert sent["stream"] is False
        msgs = sent["messages"]
        assert msgs[0]["role"] == "system"
        assert "handoff" in msgs[0]["content"].lower()
        assert msgs[1]["role"] == "user"
        assert "very large blob" in msgs[1]["content"]
    finally:
        httpd.shutdown()


def test_proactive_compact_with_summarizer_uses_real_summary():
    """End-to-end: when proactive_compact is given a summarizer, the
    summary item carries the real LLM output, not the placeholder."""
    clear_proactive_cache()
    body = _make_body(30)

    def my_summarizer(blob: str) -> str:
        return ("## What we were trying to do\nUser asked to refactor "
                "the SLAM module.\n## Files & decisions\nMain.kt:42 "
                "removed env-knob.\n## Next step\nRun tests.")

    out, info = proactive_compact(
        body, session_id="real-summary-test", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=my_summarizer,
    )
    assert info["applied"] is True
    summary_text = out["input"][-9]["content"][0]["text"]
    # Real summary should appear in the prepended user item
    assert "SLAM module" in summary_text
    assert "Main.kt:42" in summary_text
    # placeholder marker NOT present in this case (we got a real summary)
    assert "older turns omitted to fit context" not in summary_text


def test_clear_proactive_cache_session_scoped():
    clear_proactive_cache()
    body = _make_body(30)

    def s1_summarizer(blob: str) -> str:
        return "s1-specific summary"

    out1, info1 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info1["cached"] is False
    # cache hit
    out2, info2 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info2["cached"] is True

    # Clear only sA
    clear_proactive_cache("sA")
    out3, info3 = proactive_compact(
        body, session_id="sA", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=s1_summarizer,
    )
    assert info3["cached"] is False
