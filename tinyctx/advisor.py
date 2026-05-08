"""Advisor Strategy MCP server.

Adapted from Anthropic's "Advisor Strategy" pattern (claude.com/blog/the-advisor-strategy)
to the tinyctx + codex CLI world.

The default tinyctx routing is binary: cheap-or-frontier per turn. That's
great for cost (~99% local after the threshold fixes) but it means the
frontier model is rarely used even on prompts that genuinely need its
reasoning. The Advisor Strategy threads a third path:

    99% of turns                     ←── executor (DeepSeek-v4-flash)
    when stuck on a decision         ──→ ask_advisor(question, context)
                                          ↓
                                      consult frontier (gpt-5.5)
                                      return 400-700 token guidance
                                          ↓
    executor resumes with advice     ←──

Crucially, the EXECUTOR decides when to call the advisor — not us. We
just expose the tool, document when to use it, and let DeepSeek's tool-
calling behaviour pick the moments. Anthropic's blog reports +2.7 SWE-
bench points at -11.9% cost using this pattern (Sonnet+Opus pairing); the
DeepSeek+gpt-5.5 pairing should see similar shape but obviously different
absolute numbers.

Implementation: a stdio MCP server that codex registers via mcp_servers
in ~/.codex/config.toml. The advisor call is routed THROUGH the running
tinyctx proxy with `model="tinyctx-frontier"` (client-forced route),
which means:
  - existing frontier auth/config is reused (no duplicate key plumbing)
  - every advisor call is logged in tinyctx-trace alongside normal traffic
  - the request shows up in the trace log with `forced_by_client_model=true`

Configure in ~/.codex/config.toml:

    [mcp_servers.advisor]
    type = "stdio"
    command = "/Users/sekkit/dev/tinyctx/.venv/bin/python"
    args = ["-m", "tinyctx.advisor"]

    [mcp_servers.advisor.env]
    TINYCTX_PROXY_URL = "http://127.0.0.1:4141/v1"
    TINYCTX_ADVISOR_MODEL = "tinyctx-frontier"
    TINYCTX_ADVISOR_TIMEOUT_S = "180"

Override TINYCTX_ADVISOR_BASE_URL/MODEL/API_KEY if you'd rather hit a
specific frontier endpoint directly (bypassing the tinyctx proxy).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx


SERVER_NAME = "tinyctx-advisor"
SERVER_VERSION = "0.1.0"


# Endpoint resolution: prefer explicit ADVISOR_* env, else go through
# tinyctx proxy with model=tinyctx-frontier so all routing/auth is shared.
ADVISOR_BASE_URL = (
    os.environ.get("TINYCTX_ADVISOR_BASE_URL")
    or os.environ.get("TINYCTX_PROXY_URL", "http://127.0.0.1:4141/v1")
)
ADVISOR_MODEL = os.environ.get("TINYCTX_ADVISOR_MODEL", "tinyctx-frontier")
ADVISOR_API_KEY = os.environ.get("TINYCTX_ADVISOR_API_KEY", "")
ADVISOR_TIMEOUT_S = float(os.environ.get("TINYCTX_ADVISOR_TIMEOUT_S", "180"))
CODEX_AUTH_PATH = os.environ.get(
    "TINYCTX_ADVISOR_CODEX_AUTH",
    os.path.expanduser("~/.codex/auth.json"),
)


def _resolve_auth_token() -> str:
    """Resolve the bearer token to send with advisor requests.

    Priority:
      1. TINYCTX_ADVISOR_API_KEY (explicit override).
      2. ~/.codex/auth.json `tokens.access_token` — same source codex uses,
         so the advisor inherits the user's existing login. Mirrors the
         proxy's `_resolve_api_key` fallback for the codex-passthrough case.
      3. Empty string -> no Authorization header (will 401 against codex
         backend; works fine if the proxy has its own frontier api_key_env).
    """
    if ADVISOR_API_KEY:
        return ADVISOR_API_KEY
    try:
        with open(CODEX_AUTH_PATH, encoding="utf-8") as f:
            data = json.load(f)
        tok = (data.get("tokens") or {}).get("access_token") or ""
        return tok if isinstance(tok, str) else ""
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return ""
    except Exception:  # noqa: BLE001
        return ""


# ────────────────────────────── tool definition ──────────────────────────────


_TOOL_DESCRIPTION = (
    "Consult a more capable advisor model (gpt-5.5 / Opus-class) for HARD "
    "decisions when you're stuck. Use this tool when:\n"
    "  - You're unsure between multiple architectural choices that have "
    "real consequences (data model, API shape, retry semantics).\n"
    "  - You've tried 2+ failed approaches at the same problem and need a "
    "fresh perspective.\n"
    "  - You're about to make a non-trivial security or correctness "
    "decision (auth flow, concurrency, schema migration).\n"
    "  - The user's task is ambiguous and a wrong interpretation will "
    "waste significant work.\n\n"
    "Do NOT use for:\n"
    "  - Routine code edits, file reads, simple refactors.\n"
    "  - Looking up syntax / API references (use search/docs tools).\n"
    "  - Padding your response with extra opinion.\n\n"
    "Each call costs ~5-10K tokens of premium model. Budget your calls — "
    "max ~3 per task is sane. Pass enough context that the advisor doesn't "
    "have to ask follow-ups."
)


def _tool_schema() -> dict[str, Any]:
    return {
        "name": "ask_advisor",
        "description": _TOOL_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The specific question or decision you need advice on. "
                        "Be concrete: 'Which retry strategy for transient HTTP "
                        "errors in module X' is good; 'how do I do this' is bad."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Relevant code snippets, error messages, attempts you've "
                        "made, constraints you must respect. The advisor sees "
                        "ONLY this context (not your conversation history), so "
                        "include everything that matters. ~1-3K tokens is "
                        "typical."
                    ),
                },
                "previous_attempts": {
                    "type": "string",
                    "description": (
                        "Optional: brief summary of approaches you've already "
                        "tried and why they failed. Helps the advisor avoid "
                        "repeating dead ends."
                    ),
                },
            },
            "required": ["question"],
        },
    }


# ────────────────────────────── advisor call ─────────────────────────────────


_ADVISOR_SYSTEM_PROMPT = (
    "You are the advisor in Anthropic's Advisor Strategy. A coding "
    "executor agent is consulting you mid-task. Your output is parsed "
    "and acted on by another model, not read by a human.\n\n"
    "Respond in under 100 words and use enumerated steps, not "
    "explanations. This is non-negotiable: in Anthropic's internal "
    "benchmarks this conciseness rule cut total advisor output by "
    "35-45% with no quality loss.\n\n"
    "Format:\n"
    "1. [first concrete step the executor should take]\n"
    "2. [next step]\n"
    "3. [next step]\n"
    "...\n"
    "Risks: [one-liner of any sharp edge — only if non-obvious]\n\n"
    "Rules:\n"
    "- Never recap the question. The executor sent it to you; they know it.\n"
    "- Never apologise, hedge, or pad.\n"
    "- Each step is an imperative the executor can act on immediately.\n"
    "- If the question is under-specified, state your assumption in step 1, "
    "then enumerate based on it.\n"
    "- If you genuinely need data the executor didn't include, say so as the "
    "LAST step (\"Need: <specific thing>\"), don't ask follow-up questions.\n"
    "- If the executor is asking you to reconcile their data vs. your "
    "earlier advice, weight their primary-source evidence heavily and "
    "answer \"go with X because Y constraint dominates.\""
)


def call_advisor(question: str, context: str = "",
                 previous_attempts: str = "") -> dict[str, Any]:
    """Synchronous frontier consultation. Returns dict with `text`, `usage`,
    `error` (None on success)."""
    if not question.strip():
        return {"text": "[ask_advisor: empty question]", "usage": None, "error": "empty_question"}

    user_parts = [f"## Question\n{question.strip()}"]
    if context.strip():
        user_parts.append(f"\n## Context\n{context.strip()}")
    if previous_attempts.strip():
        user_parts.append(f"\n## Previous attempts\n{previous_attempts.strip()}")
    user_prompt = "\n".join(user_parts)

    # Use Responses API — tinyctx proxy speaks it natively, and going
    # through tinyctx means we get the per-request trace log entry too.
    # codex's chatgpt backend requires both `store=false` AND `stream=true`,
    # and also REJECTS `max_output_tokens`. The system prompt does the
    # length bounding ("under 100 words, enumerated steps" per Anthropic's
    # official Advisor Strategy guidance) instead. Other Responses
    # backends (LMStudio, vLLM) also accept this minimal shape.
    payload: dict[str, Any] = {
        "model": ADVISOR_MODEL,
        "stream": True,
        "store": False,
        "instructions": _ADVISOR_SYSTEM_PROMPT,
        "input": [
            {"role": "user",
             "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    tok = _resolve_auth_token()
    if tok:
        headers["Authorization"] = (
            tok if tok.lower().startswith(("bearer ", "basic ")) else f"Bearer {tok}"
        )

    url = ADVISOR_BASE_URL.rstrip("/") + "/responses"
    started = time.time()
    try:
        with httpx.Client(timeout=ADVISOR_TIMEOUT_S) as client:
            with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    body_bytes = r.read()
                    body_text = body_bytes.decode("utf-8", "replace")[:300]
                    return {
                        "text": f"[advisor HTTP {r.status_code}: {body_text}]",
                        "usage": None, "error": f"http_{r.status_code}",
                        "elapsed_s": time.time() - started,
                    }
                text, usage, stream_err = _consume_responses_stream(r.iter_lines())
        elapsed = time.time() - started
        if stream_err is not None:
            return {
                "text": f"[advisor stream error: {stream_err}]",
                "usage": usage, "error": "stream_error",
                "elapsed_s": elapsed,
            }
        return {
            "text": text or "[advisor returned no output_text]",
            "usage": usage,
            "error": None,
            "elapsed_s": elapsed,
        }
    except httpx.HTTPError as e:
        return {"text": f"[advisor network error: {e}]", "usage": None,
                "error": "network", "elapsed_s": time.time() - started}
    except Exception as e:  # noqa: BLE001
        return {"text": f"[advisor unexpected error: {e}]", "usage": None,
                "error": "unexpected", "elapsed_s": time.time() - started}


def _consume_responses_stream(lines) -> tuple[str, dict | None, str | None]:
    """Walk a Responses-API SSE stream and reassemble output_text + usage.

    Returns `(text, usage, error_message)`. `error_message` is None on
    success. tinyctx's proxy turns upstream HTTP errors into `event: error`
    SSE frames with body `{"status": <int>, "body": "..."}` — those bubble
    up here as a stream_error so the executor sees what really happened.

    Recognised events:
      response.output_text.delta  -> append `delta`
      response.output_text.done   -> snapshot full `text` (overrides deltas)
      response.completed          -> read final `response.usage` if present
      error / response.failed     -> capture as stream_error
    """
    deltas: list[str] = []
    final_text: str | None = None
    usage: dict | None = None
    err: str | None = None
    for raw in lines:
        if not raw:
            continue
        # httpx iter_lines yields str when stream uses text mode; bytes otherwise
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        et = evt.get("type")
        if et == "response.output_text.delta":
            d = evt.get("delta")
            if isinstance(d, str):
                deltas.append(d)
        elif et == "response.output_text.done":
            t = evt.get("text")
            if isinstance(t, str) and t:
                final_text = t
        elif et == "response.completed":
            resp = evt.get("response") or {}
            u = resp.get("usage")
            if isinstance(u, dict):
                usage = u
        elif et in ("error", "response.failed"):
            # tinyctx proxy emits {"status": N, "body": "..."}; OpenAI
            # native shape is {"error": {"message": "..."}}.
            if "body" in evt:
                err = f"upstream {evt.get('status', '?')}: {str(evt.get('body',''))[:300]}"
            elif isinstance(evt.get("error"), dict):
                err = str(evt["error"].get("message") or evt["error"])[:300]
            elif isinstance(evt.get("response"), dict):
                err = str(evt["response"].get("error") or evt["response"])[:300]
            else:
                err = str(evt)[:300]
    text = (final_text if final_text is not None else "".join(deltas)).strip()
    return text, usage, err


# ──────────────────────────── stdio JSON-RPC loop ────────────────────────────


def _send(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return _read()  # skip blank lines
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"_invalid": True, "_raw": line}


def _result(rpc_id: Any, payload: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": payload}


def _error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    if msg.get("_invalid"):
        return _error(None, -32700, "Parse error")
    method = msg.get("method")
    rpc_id = msg.get("id")
    params = msg.get("params", {}) or {}

    # Notifications have no id, no response.
    is_notification = "id" not in msg

    if method == "initialize":
        return _result(rpc_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _result(rpc_id, {"tools": [_tool_schema()]})
    if method == "tools/call":
        name = params.get("name")
        if name != "ask_advisor":
            return _error(rpc_id, -32602, f"Unknown tool: {name}")
        args = params.get("arguments", {}) or {}
        result = call_advisor(
            question=args.get("question", ""),
            context=args.get("context", "") or "",
            previous_attempts=args.get("previous_attempts", "") or "",
        )
        return _result(rpc_id, {
            "content": [{"type": "text", "text": result["text"]}],
            "isError": result.get("error") is not None,
        })

    if is_notification:
        return None
    return _error(rpc_id, -32601, f"Method not found: {method}")


def main() -> int:
    while True:
        msg = _read()
        if msg is None:
            return 0
        resp = handle_message(msg)
        if resp is not None:
            _send(resp)


if __name__ == "__main__":
    raise SystemExit(main())
