"""Verify that tinyctx's stream proxy ALWAYS ends with a structurally
valid `event: response.completed` SSE terminator, even on upstream errors.

Without this, codex.app raises:
    "stream disconnected before completion: stream closed before
     response.completed"

Live trace today (post-restart 15:48) showed every frontier request
return 200 with ~190KB SSE — but if the upstream EVER errors mid-session,
codex.app would see the `event: error` and immediately complain that the
stream closed without `response.completed`. So we always emit a synthetic
terminator.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _ErrorBackend(BaseHTTPRequestHandler):
    """Fake upstream that always returns HTTP 400 with a JSON error body."""

    def log_message(self, *a, **kw):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n) if n else b""
        body = json.dumps({
            "error": {
                "message": "No tool call found for function call output with call_id call_TEST.",
                "type": "invalid_request_error",
                "param": "input",
            }
        }).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_backend() -> tuple[ThreadingHTTPServer, int, threading.Thread]:
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _ErrorBackend)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port, t


def _start_proxy(local_port: int, frontier_port: int) -> tuple[int, threading.Thread]:
    proxy_port = _free_port()

    os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{local_port}/v1"
    os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
    os.environ["TINYCTX_LOCAL_MODEL"] = "qwen-test"
    os.environ["TINYCTX_FRONTIER_BASE_URL"] = f"http://127.0.0.1:{frontier_port}/v1"
    os.environ["TINYCTX_FRONTIER_MODEL"] = "gpt-test"
    os.environ["TINYCTX_VERBOSE"] = "0"
    os.environ["TINYCTX_CONFIG"] = "/dev/null"
    os.environ.pop("TINYCTX_FORCE_ROUTE", None)

    import sys
    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod
    import uvicorn

    cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1", port=proxy_port,
                         log_level="error")
    server = uvicorn.Server(cfg)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    return proxy_port, t


def test_stream_terminator_emitted_on_upstream_400():
    """When upstream returns 400 to a stream request, tinyctx must emit:
        event: error\\ndata: {...}\\n\\n
        event: response.completed\\ndata: {...}\\n\\n

    so codex.app's SSE parser sees a clean stream close.
    """
    local_httpd, local_port, _ = _start_backend()
    try:
        # Use the same broken backend as both local and frontier — we don't
        # care which the proxy picks, only that the error path emits the
        # terminator.
        proxy_port, _ = _start_proxy(local_port, local_port)

        body = {
            "model": "tinyctx-local",  # force route to local (which 400s)
            "instructions": "test",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "x"}]}],
            "stream": True,
        }
        req = Request(
            f"http://127.0.0.1:{proxy_port}/v1/responses",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", "replace")

        # Must contain BOTH events
        assert "event: error" in raw, f"missing error event:\n{raw}"
        assert "event: response.completed" in raw, (
            f"MISSING response.completed terminator — codex.app would raise "
            f"'stream disconnected before completion'\n{raw}"
        )
        # error must come BEFORE response.completed
        err_idx = raw.index("event: error")
        completed_idx = raw.index("event: response.completed")
        assert err_idx < completed_idx, (
            f"order wrong: error@{err_idx} after completed@{completed_idx}\n{raw}"
        )

        # The synthetic terminator must mark itself incomplete
        # (so codex doesn't think it was a clean success).
        # Find the response.completed data line
        for chunk in raw.split("\n\n"):
            if "event: response.completed" not in chunk:
                continue
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    assert data["response"]["status"] == "incomplete", (
                        f"terminator should mark status=incomplete: {data}"
                    )
                    assert "tinyctx_proxy_terminator" in json.dumps(data), (
                        f"terminator should carry tinyctx marker: {data}"
                    )
                    break

    finally:
        local_httpd.shutdown()
        # let the daemon thread clean itself up; uvicorn lacks a clean
        # shutdown handle here (it's running asyncio in another thread)
        time.sleep(0.05)
