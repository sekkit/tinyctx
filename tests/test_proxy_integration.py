"""End-to-end test: boots the tinyctx proxy against fake local + frontier
backends and verifies routing decisions land on the right one.

Run:  python tests/test_proxy_integration.py
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
from urllib.request import Request as URLRequest, urlopen


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _FakeBackend(BaseHTTPRequestHandler):
    """Records every request hit. Returns a static OK JSON.
    Class is parameterized via class-level `tag` so tests can distinguish."""
    tag = "x"
    received: list[dict] = []  # shared

    def log_message(self, *a, **kw):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw)
        except Exception:
            body = {"_raw": raw.decode("utf-8", "replace")}
        self.__class__.received.append({
            "tag": self.tag,
            "path": self.path,
            "model": body.get("model"),
            "stream": body.get("stream", False),
            "instructions": body.get("instructions"),
            "input_first": (body.get("input") or [None])[0],
            "had_encrypted_in_history": _had_encrypted(body),
        })
        payload = json.dumps({"id": "resp_test", "object": "response",
                              "model": body.get("model", "?"),
                              "tag": self.tag}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _had_encrypted(body: dict) -> bool:
    items = body.get("input") or body.get("messages") or []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):
            if "encrypted_content" in it:
                return True
            content = it.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and "encrypted_content" in c:
                        return True
    return False


def _spawn_backend(tag: str, port: int) -> threading.Thread:
    Handler = type(f"Backend_{tag}", (_FakeBackend,), {"tag": tag})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    t._httpd = httpd  # type: ignore[attr-defined]
    return t


def _start_proxy(local_port: int, frontier_port: int, proxy_port: int) -> threading.Thread:
    os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{local_port}/v1"
    os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"  # fake backend speaks responses
    os.environ["TINYCTX_LOCAL_MODEL"] = "qwen-test"
    os.environ["TINYCTX_FRONTIER_BASE_URL"] = f"http://127.0.0.1:{frontier_port}/v1"
    os.environ["TINYCTX_FRONTIER_MODEL"] = "gpt-test"
    os.environ["TINYCTX_VERBOSE"] = "0"
    os.environ.pop("TINYCTX_FORCE_ROUTE", None)

    # Reload module so config picks up env.
    import sys
    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod
    print(f"[debug] proxy CFG.local.base_url={proxy_mod.CFG.local.base_url}")
    print(f"[debug] proxy CFG.frontier.base_url={proxy_mod.CFG.frontier.base_url}")

    import uvicorn
    cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1", port=proxy_port,
                         log_level="error")
    server = uvicorn.Server(cfg)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # wait for ready
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    return t


def _post(url: str, body: dict) -> dict:
    req = URLRequest(url, data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    lp, fp, pp = free_port(), free_port(), free_port()
    _spawn_backend("local", lp)
    _spawn_backend("frontier", fp)
    _start_proxy(lp, fp, pp)
    base = f"http://127.0.0.1:{pp}/v1/responses"
    print(f"[debug] local={lp} frontier={fp} proxy={pp}")

    _FakeBackend.received.clear()

    # 1. Compaction handoff -> must land on local.
    r = _post(base, {
        "model": "gpt-5.5",
        "instructions": "Create a handoff summary for another LLM that will resume the task.",
        "input": [{"role": "user", "content": "..."}],
    })
    assert r["tag"] == "local", f"compaction should route local, got {r}"
    assert _FakeBackend.received[-1]["model"] == "qwen-test"

    # 2. Tiny request -> local.
    r = _post(base, {"model": "gpt-5.5",
                     "input": [{"role": "user",
                                "content": [{"type": "input_text", "text": "rename foo"}]}]})
    assert r["tag"] == "local", f"tiny should route local, got {r}"

    # 3. Big request -> frontier.
    big_text = "x" * 400_000
    r = _post(base, {"model": "gpt-5.5",
                     "input": [{"role": "user",
                                "content": [{"type": "input_text", "text": big_text}]}]})
    assert r["tag"] == "frontier", f"big should route frontier, got {r}"

    print("[debug] tests 1-3 ok, starting test 4")
    # 4. Encrypted reasoning items must be scrubbed before forwarding.
    r = _post(base, {
        "model": "gpt-5.5",
        "input": [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "encrypted_content": "OPAQUE"},
        ],
    })
    rec = _FakeBackend.received[-1]
    assert rec["had_encrypted_in_history"] is False, \
        "sanitizer should strip encrypted_content before forwarding"

    # 5. Client can force a route via model id.
    r = _post(base, {"model": "tinyctx-frontier",
                     "input": [{"role": "user", "content": "tiny"}]})
    assert r["tag"] == "frontier", f"forced frontier failed, got {r}"

    print("PASS test_compaction_routes_local")
    print("PASS test_tiny_routes_local")
    print("PASS test_huge_routes_frontier")
    print("PASS test_sanitizer_strips_encrypted_content_on_wire")
    print("PASS test_client_forced_route_via_model_id")


if __name__ == "__main__":
    main()
