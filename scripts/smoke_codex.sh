#!/usr/bin/env bash
# End-to-end smoke test: starts the tinyctx proxy with TINYCTX_FORCE_ROUTE=local
# and a fake local backend, then runs a minimal codex exec against the proxy.
#
# This DOES require codex CLI to be installed and visible on PATH.
# It does NOT require LMStudio or a real frontier provider.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/.smoke.log"
: > "$LOG"

# 1. Start a fake local backend that always returns a Responses-API stream.
FAKE_PY="$(mktemp /tmp/tinyctx-fake.XXXXXX.py)"
cat > "$FAKE_PY" <<'EOF'
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self,*a,**k): pass
    def do_POST(self):
        n=int(self.headers.get("Content-Length","0"))
        body=json.loads(self.rfile.read(n) or b"{}")
        sys.stderr.write(f"[fake-local] saw model={body.get('model')!r} stream={body.get('stream')}\n")
        # Minimal Responses API SSE: one response.created + response.completed
        events=[
            {"type":"response.created","response":{"id":"r1","model":body.get("model"),"object":"response","status":"in_progress"}},
            {"type":"response.output_text.delta","delta":"ok from fake-local","item_id":"i1","output_index":0,"content_index":0},
            {"type":"response.completed","response":{"id":"r1","model":body.get("model"),"object":"response","status":"completed","output":[{"id":"i1","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok from fake-local"}]}]}},
        ]
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()

port=int(sys.argv[1])
ThreadingHTTPServer(("127.0.0.1",port),H).serve_forever()
EOF

LOCAL_PORT=14201
PROXY_PORT=14202

python3 "$FAKE_PY" "$LOCAL_PORT" >> "$LOG" 2>&1 &
FAKE_PID=$!

# 2. Start the proxy pointing at the fake backend.
source "$ROOT/.venv/bin/activate"
TINYCTX_LOCAL_BASE_URL="http://127.0.0.1:$LOCAL_PORT/v1" \
TINYCTX_LOCAL_WIRE_API="responses" \
TINYCTX_LOCAL_MODEL="qwen-fake" \
TINYCTX_FORCE_ROUTE="local" \
TINYCTX_VERBOSE="1" \
python -c "from tinyctx.proxy import APP; import uvicorn; uvicorn.run(APP, host='127.0.0.1', port=$PROXY_PORT, log_level='warning')" >> "$LOG" 2>&1 &
PROXY_PID=$!

cleanup() { kill $FAKE_PID $PROXY_PID 2>/dev/null || true ; rm -f "$FAKE_PY" ; }
trap cleanup EXIT

# 3. Wait for proxy.
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PROXY_PORT/" >/dev/null 2>&1; then break; fi
    sleep 0.2
done

echo "[smoke] proxy up"
curl -s "http://127.0.0.1:$PROXY_PORT/" | python3 -m json.tool

# 4. Hit the proxy directly without codex (request shape from the openai/codex repo).
echo "[smoke] direct request via curl"
curl -sN "http://127.0.0.1:$PROXY_PORT/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gpt-5.5","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}]}' \
    | head -20

# 5. If codex CLI is available, run it through the proxy.
if command -v codex >/dev/null 2>&1; then
    echo "[smoke] running codex exec through proxy (30s budget)"
    if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 30"; else TO=""; fi
    $TO codex exec --skip-git-repo-check \
        -c "model_provider=\"tinyctx-smoke\"" \
        -c "model_providers.tinyctx-smoke.name=\"tinyctx smoke\"" \
        -c "model_providers.tinyctx-smoke.base_url=\"http://127.0.0.1:$PROXY_PORT/v1\"" \
        -c "model_providers.tinyctx-smoke.wire_api=\"responses\"" \
        -c "model=\"gpt-5.5\"" \
        "say hi in 4 words" 2>&1 | tail -30 || echo "[smoke] codex exec finished (may have errored — see $LOG)"
else
    echo "[smoke] codex CLI not on PATH; skipping live codex run"
fi

# 6. Tail the log to show routing decisions.
echo "----- proxy log (tail) -----"
tail -20 "$LOG"
