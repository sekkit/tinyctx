"""End-to-end path-coverage harness.

Boots an isolated tinyctx-proxy + fake local + fake frontier backend on
ephemeral ports, sends crafted requests that should each light up specific
code paths, then reads the resulting trace JSONL and asserts every targeted
path actually fired.

Different from `test_proxy_integration.py`:
  - Uses `TINYCTX_LOG_DIR` to isolate trace emission to a temp dir (no
    pollution of the user's production trace log).
  - Targets paths that the production-trace path-coverage report flagged
    as cold: read_delta, MCP namespace expand, CacheAwareMutator fired,
    proactive_compact synthetic stub, advisor hint inject.
  - Reports per-path firing as a table; fails the test only if a target
    path that *should* fire didn't.

Run:  python tests/test_path_coverage_e2e.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request as URLRequest, urlopen


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─────────────────────────── fake backend ────────────────────────────


class _FakeBackend(BaseHTTPRequestHandler):
    """Always-200 OK; records every request body."""
    tag = "x"
    received: list[dict] = []

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            body = {"_raw": raw.decode("utf-8", "replace")}
        self.__class__.received.append({
            "tag": self.tag,
            "path": self.path,
            "model": body.get("model"),
            "n_tools": len(body.get("tools") or []),
            "n_input": len(body.get("input") or body.get("messages") or []),
        })
        payload = json.dumps({
            "id": "resp_test",
            "object": "response",
            "model": body.get("model", "?"),
            "tag": self.tag,
            "output": [{
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _spawn_backend(tag: str, port: int) -> ThreadingHTTPServer:
    Handler = type(f"Backend_{tag}", (_FakeBackend,), {"tag": tag})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ──────────────────────────── proxy boot ─────────────────────────────


def _start_proxy(local_port: int, frontier_port: int,
                 proxy_port: int, log_dir: Path) -> threading.Thread:
    """Launch tinyctx proxy in a thread. Sets every env var the test
    relies on BEFORE module reload, then re-imports tinyctx."""
    os.environ["TINYCTX_LOCAL_BASE_URL"] = f"http://127.0.0.1:{local_port}/v1"
    os.environ["TINYCTX_LOCAL_WIRE_API"] = "responses"
    os.environ["TINYCTX_LOCAL_MODEL"] = "qwen-test"
    os.environ["TINYCTX_FRONTIER_BASE_URL"] = f"http://127.0.0.1:{frontier_port}/v1"
    os.environ["TINYCTX_FRONTIER_MODEL"] = "gpt-test"
    os.environ["TINYCTX_LOG_DIR"] = str(log_dir)
    os.environ["TINYCTX_VERBOSE"] = "1"
    os.environ["TINYCTX_CONFIG"] = "/dev/null"
    os.environ.pop("TINYCTX_FORCE_ROUTE", None)

    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod

    # Force-enable features whose defaults are off OR depend on
    # CacheAwareMutator deferring on first turn — we want to observe
    # them firing in this test.
    proxy_mod.CFG.dedup_tool_calls = True
    proxy_mod.CFG.purge_failed_tool_inputs = True
    proxy_mod.CFG.read_delta_enabled = True
    proxy_mod.CFG.mutation_threshold = 0.0
    # _MUTATOR is constructed at module import from CFG — patch the live
    # instance directly so `usage >= threshold` short-circuits to fire
    # even on the first turn (first_turn_defer is checked AFTER the
    # threshold gate).
    proxy_mod._MUTATOR.threshold = -1.0   # any usage >= -1 → fire
    proxy_mod._MUTATOR.ttl_seconds = 0.0  # also fire on subsequent turns

    import uvicorn
    cfg = uvicorn.Config(proxy_mod.APP, host="127.0.0.1", port=proxy_port,
                         log_level="error")
    server = uvicorn.Server(cfg)
    import asyncio

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{proxy_port}/", timeout=1).read()
            return t
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("proxy did not come up")


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = URLRequest(url, data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


# ─────────────────────────── targeted requests ───────────────────────


# > 400 chars so read_delta doesn't skip
_FILE_CONTENT = "\n".join(f"line {i}: hello world" for i in range(60))
_FILE_CONTENT_DIFF = "\n".join(
    f"line {i}: hello world" if i != 5 else f"line {i}: HELLO WORLD"
    for i in range(60))


def _read_delta_body() -> dict:
    """One body that contains the SAME Read of /tmp/foo.py twice → must
    trigger read_delta replacement on the second occurrence."""
    return {
        "model": "tinyctx-local",
        "input": [
            {"role": "user",
             "content": [{"type": "input_text", "text": "look at /tmp/foo.py"}]},
            {"type": "function_call", "name": "Read",
             "arguments": json.dumps({"path": "/tmp/foo.py"}),
             "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1",
             "output": _FILE_CONTENT},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "ok"}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": "look again"}]},
            {"type": "function_call", "name": "Read",
             "arguments": json.dumps({"path": "/tmp/foo.py"}),
             "call_id": "c2"},
            {"type": "function_call_output", "call_id": "c2",
             "output": _FILE_CONTENT_DIFF},
        ],
    }


def _mcp_namespace_body() -> dict:
    """Body with a codex 0.128-style namespace tool. expand_mcp_namespaces
    should unwrap; the unwrapped function-type tool then survives scrub."""
    return {
        "model": "tinyctx-local",
        "tools": [
            {"type": "function", "name": "shell"},
            {"type": "namespace", "name": "mcp__advisor__",
             "description": "Tools in mcp__advisor__.",
             "tools": [
                 {"type": "function", "name": "ask_advisor",
                  "description": "Consult frontier.",
                  "parameters": {"type": "object",
                                 "properties": {"question": {"type": "string"}},
                                 "required": ["question"]}},
             ]},
            {"type": "web_search"},
        ],
        "input": [{"role": "user",
                   "content": [{"type": "input_text", "text": "hi"}]}],
    }


def _frontier_force_body() -> dict:
    """Force frontier route via client model id; advisor hint should be
    skipped because of frontier_skip_advisor_hint default."""
    return {
        "model": "tinyctx-frontier",
        "instructions": "You are codex. Be helpful.",
        "input": [{"role": "user",
                   "content": [{"type": "input_text", "text": "tiny req"}]}],
    }


# ─────────────────────────── verification ────────────────────────────


# (label, predicate(trace_event) -> bool)
TARGET_PATHS: list[tuple[str, callable]] = [
    ("router → local",
     lambda e: e.get("route") == "local"),
    ("router → frontier",
     lambda e: e.get("route") == "frontier"),
    ("client-forced model",
     lambda e: bool(e.get("forced_by_client_model"))),
    ("encrypted_content scrub (zero-count is also ok)",
     lambda e: "encrypted_content_stripped" in e),
    ("advisor_hint_skipped on frontier",
     lambda e: bool(e.get("advisor_hint_skipped"))),
    ("MCP namespace appeared in dropped types (= expand ran on a real ns)",
     lambda e: "namespace" in (e.get("tool_types_dropped") or [])),
    ("CacheAwareMutator wanted",
     lambda e: bool(e.get("mutation_wanted"))),
    ("CacheAwareMutator fired",
     lambda e: bool(e.get("mutation_fired"))),
    ("read_delta candidates seen",
     lambda e: int(e.get("read_delta_candidates") or 0) >= 2),
    ("read_delta replacements made",
     lambda e: int(e.get("read_delta_replacements") or 0) >= 1),
    ("read_delta bytes saved > 0",
     lambda e: int(e.get("read_delta_bytes_saved") or 0) > 0),
    ("forwarded_breakdown populated",
     lambda e: bool(e.get("forwarded_breakdown"))),
]


def _read_traces(log_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(log_dir.glob("tinyctx-*.jsonl")):
        try:
            for line in f.read_text(errors="replace").splitlines():
                if '"request_trace"' not in line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return out


def main() -> int:
    lp, fp, pp = _free_port(), _free_port(), _free_port()
    _spawn_backend("local", lp)
    _spawn_backend("frontier", fp)

    with tempfile.TemporaryDirectory(prefix="tinyctx-e2e-") as td:
        log_dir = Path(td)
        _start_proxy(lp, fp, pp, log_dir)

        base = f"http://127.0.0.1:{pp}/v1/responses"
        print(f"[e2e] proxy={pp} local={lp} frontier={fp}  log_dir={log_dir}")

        # Drive the targeted scenarios. Each one emits a request_trace
        # event into log_dir which we'll inspect.
        scenarios: list[tuple[str, dict]] = [
            ("client-forced local + read_delta + mutation",
             _read_delta_body()),
            ("client-forced frontier + advisor_hint_skipped",
             _frontier_force_body()),
            ("MCP namespace expansion (local route)",
             _mcp_namespace_body()),
        ]
        for label, body in scenarios:
            status, _ = _post(base, body)
            print(f"  scenario {label!r}: HTTP {status}")

        # Give trace.emit a moment to flush.
        time.sleep(0.5)

        traces = _read_traces(log_dir)
        print(f"[e2e] collected {len(traces)} request_trace events")

        # Per-path coverage
        results: list[tuple[str, int, int]] = []
        for label, pred in TARGET_PATHS:
            n = 0
            for tr in traces:
                try:
                    if pred(tr):
                        n += 1
                except Exception:  # noqa: BLE001
                    continue
            results.append((label, n, len(traces)))

        print()
        print(f"{'path':<55}  {'fired':>5}  {'of':>3}")
        print("-" * 75)
        all_fired = True
        for label, n, total in results:
            mark = "✓" if n > 0 else "✗"
            print(f"{mark} {label:<53}  {n:>5}  {total:>3}")
            if n == 0:
                all_fired = False

        if not all_fired:
            print()
            print("Cold paths above never fired in this e2e run.")
            return 1

        print()
        print("ALL TARGETED PATHS FIRED ✓")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
