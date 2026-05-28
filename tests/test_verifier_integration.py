"""End-to-end verifier smoke test: boots the tinyctx proxy against
fake local + frontier backends, sends a streaming request whose output
the verifier should flag as low-quality, and verifies the follow-up
request is force-routed to frontier.

Run:  uv run python tests/test_verifier_integration.py
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ── helpers ──────────────────────────────────────────────────────────────────


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── streaming local backend ──────────────────────────────────────────────────
# Handles TWO endpoints on the same port:
#   POST /v1/responses       → SSE stream with text content
#   POST /v1/chat/completions → verifier callback (returns low-score verdict)

_LOW_SCORE_VERDICT = json.dumps({
    "choices": [{"message": {"content": (
        '{"task_completion": 2, "output_quality": 1, '
        '"execution_evidence": 1, "reason": "garbage output no verification"}'
    )}}],
})

_SSE_BODY = (
    'data: {"type":"response.output_text.delta",'
    '"delta":"Here is some text that is definitely wrong and does not '
    'match what the user asked for at all. The agent claims success '
    'but the output is clearly incorrect and no verification was run. '
    'This is padding to exceed the 100-character short-text floor '
    'in the verifier module so the quality check actually runs."}\\n\\n'
    'data: {"type":"response.completed","response":{'
    '"finish_reason":"stop","status":"completed",'
    '"usage":{"output_tokens":50}}}\\n\\n'
)

_FRONTIER_RESPONSE = json.dumps({
    "id": "frontier_resp", "object": "response", "model": "gpt-test",
    "tag": "frontier",
})


class _LocalBackend(BaseHTTPRequestHandler):
    """Fake local backend: streams SSE on /v1/responses, returns low-score
    verdict on /v1/chat/completions."""

    received_responses: list[dict] = []
    received_chat: list[dict] = []

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        path = self.path.rstrip("/")

        if path.endswith("/chat/completions"):
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                body = {}
            self.__class__.received_chat.append({
                "model": body.get("model"),
                "messages": body.get("messages"),
            })
            payload = _LOW_SCORE_VERDICT.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # /v1/responses or root — SSE stream
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        is_stream = body.get("stream", False)
        self.__class__.received_responses.append({
            "model": body.get("model"),
            "stream": is_stream,
            "input_len": len(str(body.get("input", ""))),
        })

        if is_stream:
            payload = _SSE_BODY.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            payload = json.dumps({
                "id": "local_resp", "object": "response",
                "model": body.get("model", "qwen-test"),
                "tag": "local",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


class _FrontierBackend(BaseHTTPRequestHandler):
    """Simple frontier fake — always returns its tag."""

    received: list[dict] = []

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        self.__class__.received.append({
            "model": body.get("model"),
            "stream": body.get("stream", False),
        })
        payload = _FRONTIER_RESPONSE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _spawn_backend(handler_cls, port: int):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t, httpd


def _start_proxy(local_port: int, frontier_port: int, proxy_port: int):
    os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{local_port}/v1"
    os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
    os.environ["TINYCTX_LOCAL_MODEL"] = "qwen-test"
    os.environ["TINYCTX_FRONTIER_BASE_URL"] = f"http://127.0.0.1:{frontier_port}/v1"
    os.environ["TINYCTX_FRONTIER_MODEL"] = "gpt-test"
    os.environ["TINYCTX_VERBOSE"] = "0"
    os.environ["TINYCTX_CONFIG"] = "/dev/null"
    os.environ.pop("TINYCTX_FORCE_ROUTE", None)

    # nuke cached tinyctx imports so config picks up env
    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]

    import tinyctx.proxy as proxy_mod

    import uvicorn
    cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1", port=proxy_port,
                         log_level="warning")
    server = uvicorn.Server(cfg)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for proxy to be ready
    from urllib.request import urlopen
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    return t


# ── main test ────────────────────────────────────────────────────────────────


def main():
    lp, fp, pp = free_port(), free_port(), free_port()
    _spawn_backend(_LocalBackend, lp)
    _spawn_backend(_FrontierBackend, fp)
    _start_proxy(lp, fp, pp)
    print(f"[setup] local={lp} frontier={fp} proxy={pp}")

    # Reset shared state
    _LocalBackend.received_responses.clear()
    _LocalBackend.received_chat.clear()
    _FrontierBackend.received.clear()

    import httpx

    base = f"http://127.0.0.1:{pp}/v1/responses"

    # ── Step 1: send a STREAMING request → routes to local ───────────────
    print("[test] Step 1: streaming request → should route local")
    body1 = {
        "model": "gpt-5.5",
        "stream": True,
        "input": [
            {"role": "user",
             "content": [{"type": "input_text",
                          "text": "Write a python script to sort a list"}]},
        ],
    }

    streamed_bytes = b""
    with httpx.Client(timeout=httpx.Timeout(10)) as client:
        with client.stream("POST", base, json=body1,
                           headers={"Content-Type": "application/json"}) as r:
            assert r.status_code == 200, f"step1 status {r.status_code}"
            for chunk in r.iter_bytes():
                streamed_bytes += chunk

    streamed_text = streamed_bytes.decode("utf-8", errors="ignore")
    print(f"[test]   streamed {len(streamed_bytes)} bytes, "
          f"first 100: {streamed_text[:100]!r}")

    # Verify local backend received the stream request
    assert len(_LocalBackend.received_responses) >= 1, \
        "local backend should have received the streaming request"
    assert _LocalBackend.received_responses[-1]["stream"] is True, \
        "should have been a streaming request"

    # ── Step 2: wake up event loop so background verifier task runs ──
    print("[test] Step 2: waking event loop for bg verifier...")
    # The verifier is spawned via asyncio.create_task in the proxy's event
    # loop. Async tasks only run when the loop gets control. In production
    # the gap between agent turns (human typing time, 2-30s) gives the
    # loop plenty of idle time. In the test we send a dummy request to
    # force the event loop to drain pending callbacks.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{pp}/dashboard",
                      timeout=httpx.Timeout(2))
        except Exception:
            pass
        time.sleep(0.3)
        if len(_LocalBackend.received_chat) >= 2:
            break

    chat_count = len(_LocalBackend.received_chat)
    print(f"[test]   verifier chat calls received: {chat_count}")
    if chat_count >= 1:
        msg = _LocalBackend.received_chat[0]
        print(f"[test]   verifier prompt begins: "
              f"{str(msg.get('messages', [[],{}])[0].get('content',''))[:120]}...")
    else:
        print("[test]   WARNING: verifier did not call /chat/completions "
              "within 5s — may be a timing issue or config problem")

    # ── Step 3: send a second (non-streaming) request ────────────────────
    # If the verifier set the flag, this should route to frontier.
    print("[test] Step 3: follow-up request → should route frontier "
          "(if verifier fired)")
    body2 = {
        "model": "gpt-5.5",
        "stream": False,
        "input": [
            {"role": "user",
             "content": [{"type": "input_text",
                          "text": "continue"}]},
        ],
    }

    r2 = httpx.post(base, json=body2,
                    headers={"Content-Type": "application/json"},
                    timeout=httpx.Timeout(10))
    assert r2.status_code == 200, f"step3 status {r2.status_code}"
    resp2 = r2.json()
    tag2 = resp2.get("tag", "?")
    print(f"[test]   response tag: {tag2}")

    frontier_hit = len(_FrontierBackend.received) >= 1
    print(f"[test]   frontier backend hit: {frontier_hit}")

    # ── Results ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if frontier_hit:
        print("PASS: Verifier escalated to frontier after low-quality output")
        print(f"  - verifier callback fired: {chat_count >= 1}")
        print(f"  - follow-up routed to frontier: True")
        rc = 0
    elif chat_count >= 1:
        print("PARTIAL: Verifier ran but escalation did not occur")
        print(f"  - verifier callback fired: True")
        print(f"  - follow-up routed to frontier: False")
        print("  - This may indicate the verifier flag was not consumed")
        print("    by VerifierGate, or the guard skipped incorrectly.")
        rc = 1
    else:
        print("FAIL: Verifier never fired — integration broken")
        print(f"  - verifier callback fired: False")
        print(f"  - Check that verifier_enabled=True in config")
        print(f"  - Check that route==local (got streaming request)")
        rc = 1
    print("=" * 60)

    sys.exit(rc)


if __name__ == "__main__":
    main()
