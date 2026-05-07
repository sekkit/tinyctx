"""End-to-end test: real proxy + fake local backend that returns role-
specific replies. Sends a compaction-fingerprint request and verifies the
proxy's SSE response contains the judge-merged summary.
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
from urllib.request import Request as URLRequest, urlopen


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _spawn_role_aware_backend(port: int, received: list[dict]):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n))
            received.append(body)
            sys_msg = ""
            for m in body.get("messages", []):
                if m.get("role") == "system":
                    sys_msg = m.get("content", "").lower()
                    break
            mapping = {
                "you are an archaeologist": "FACTS-from-archaeologist",
                "you are a narrator":       "STORY-from-narrator",
                "you are an enumerator":    "ARTIFACTS-from-enumerator",
                "you are a handoff editor":
                    "## What we are doing and why\nMERGED-BY-JUDGE\n"
                    "## Files & decisions\nx\n## Commands & outcomes\ny\n"
                    "## Open issues / next steps\nz",
            }
            text = next((v for k, v in mapping.items() if k in sys_msg),
                        "(unmatched)")
            payload = json.dumps({
                "choices": [{"message": {"content": text}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _start_proxy(local_port: int, proxy_port: int):
    os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{local_port}/v1"
    os.environ["TINYCTX_LOCAL_WIRE_API"] = "chat"
    os.environ["TINYCTX_LOCAL_MODEL"]    = "qwen-fake"
    os.environ["TINYCTX_FRONTIER_BASE_URL"] = "http://127.0.0.1:1"  # unreachable
    os.environ["TINYCTX_FRONTIER_MODEL"]    = "gpt-fake"
    os.environ["TINYCTX_VERBOSE"] = "0"
    os.environ.pop("TINYCTX_FORCE_ROUTE", None)

    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod
    # ensure debate is on AND threshold is low enough for our short test body.
    proxy_mod.CFG.compactor_debate = True
    proxy_mod.CFG.compactor_min_history_tokens = 10
    proxy_mod.CFG.save_compactions = False  # no persistence side-effect

    import uvicorn
    cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1", port=proxy_port,
                         log_level="error")
    server = uvicorn.Server(cfg)

    def _run():
        asyncio.run(server.serve())

    threading.Thread(target=_run, daemon=True).start()
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("proxy did not start")


def main():
    lp, pp = _free_port(), _free_port()
    received: list[dict] = []
    httpd = _spawn_role_aware_backend(lp, received)
    try:
        _start_proxy(lp, pp)
        body = {
            "model": "gpt-5.5",
            "stream": True,
            "instructions": (
                "Create a handoff summary for another LLM that will resume "
                "the task. Be concise and structured."
            ),
            "input": [
                {"role": "user", "content": "set up auth with jwt"},
                {"role": "assistant", "content": "ran tests, edited src/auth.py"},
                {"role": "user", "content": "now compact"},
            ],
        }
        url = f"http://127.0.0.1:{pp}/v1/responses"
        req = URLRequest(url, data=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=15) as r:
            data = r.read().decode()

        # Verify SSE shape and merged summary
        assert "response.created" in data, "missing response.created event"
        assert "response.output_text.delta" in data, "missing delta"
        assert "response.completed" in data, "missing response.completed"
        assert "MERGED-BY-JUDGE" in data, f"merged summary missing: {data[:500]}"

        # Verify all 4 subagent calls actually fired through the proxy → backend.
        assert len(received) == 4, f"expected 4 backend calls, got {len(received)}"
        sysmsgs = [c["messages"][0]["content"].lower() for c in received]
        for hint in ("you are an archaeologist", "you are a narrator",
                     "you are an enumerator", "you are a handoff editor"):
            assert any(hint in s for s in sysmsgs), f"role missing: {hint}"

        print("PASS test_proxy_compactor_emits_merged_sse")
        print("PASS test_proxy_compactor_calls_all_four_roles_via_backend")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
