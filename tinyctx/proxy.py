"""tinyctx proxy: a Codex-CLI-facing Responses-API server that routes between
a local cheap model and a frontier model.

Starts with `python -m tinyctx.proxy` or `tinyctx-proxy`.

What it does on every request:
  1. Decide local vs frontier (router.decide).
  2. Sanitize encrypted_content (sanitize.strip_encrypted_content).
  3. If the local backend speaks `chat` not `responses`, normalize down.
  4. Forward upstream, preserving SSE streaming.
  5. Log a JSONL line with the routing decision and timings.

What it does NOT do (yet):
  - Translate a `chat` SSE response back into Responses-API SSE. For now we
    stream chat back when the client speaks chat; codex will require
    Responses-shaped streaming. So when routing to a chat-only local backend,
    we set wire_api accordingly in config and codex must talk to the local
    backend through a Responses-speaking server (LMStudio's `/v1/responses`
    endpoint, vLLM, SGLang, or your own front-end). The default config keeps
    everything on Responses.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .compactor import (
    build_responses_api_payload,
    build_responses_api_sse,
    compact_with_debate,
)
from .config import BackendCfg, Config, load_config
from .continuity import save_compaction
from . import historian
from .router import Decision, decide
from .sanitize import (
    CacheAwareMutator,
    dedup_tool_calls,
    inject_responses_defaults,
    normalize_for_chat,
    purge_failed_tool_inputs,
    rewrite_model,
    expand_mcp_namespaces,
    inject_advisor_hint,
    scrub_unsupported_tools,
    strip_encrypted_content,
    strip_unsupported_responses_fields,
)
from .tool_call_translator import ChatToResponsesTranslator, StreamTranslator, rebuild_response
from .trace import RequestTrace


CFG: Config = load_config()
APP = FastAPI(title="tinyctx", version="0.1.0")

# Per-session error streak counter for cascade escalation.
_SESSION_ERROR_STREAK: dict[str, int] = defaultdict(int)

# Cache-aware gate for history-mutating transforms (dedup, purge).
_MUTATOR = CacheAwareMutator(
    ttl_seconds=CFG.mutation_ttl_s,
    threshold=CFG.mutation_threshold,
)


def _log(event: str, **fields: Any) -> None:
    if not CFG.verbose:
        return
    line = json.dumps({"t": time.time(), "event": event, **fields},
                      default=str, ensure_ascii=False)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    try:
        log_path = CFG.log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _resolve_api_key(backend: BackendCfg, codex_auth: str | None) -> str | None:
    if backend.api_key_env:
        v = os.environ.get(backend.api_key_env)
        if v:
            return v
    # fall back to codex's own Authorization header (passthrough mode)
    return codex_auth


def _select_backend(decision: Decision) -> BackendCfg:
    return CFG.local if decision.route == "local" else CFG.frontier


def _session_id(request: Request, body: dict[str, Any]) -> str:
    sid = request.headers.get("x-codex-session-id")
    sid = sid or body.get("session_id") or body.get("metadata", {}).get("session_id")
    return sid or "global"


@APP.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "tinyctx",
        "version": "0.1.0",
        "local": {"base_url": CFG.local.base_url, "model": CFG.local.model,
                  "wire_api": CFG.local.wire_api},
        "frontier": {"base_url": CFG.frontier.base_url, "model": CFG.frontier.model,
                     "wire_api": CFG.frontier.wire_api},
        "force_route": CFG.force_route,
    }


@APP.get("/v1/models")
def list_models() -> dict[str, Any]:
    ctx = CFG.default_context_window
    return {
        "object": "list",
        "data": [
            {"id": "tinyctx-auto", "object": "model", "owned_by": "tinyctx",
             "context_window": ctx},
            {"id": "tinyctx-local", "object": "model", "owned_by": "tinyctx",
             "context_window": ctx},
            {"id": "tinyctx-frontier", "object": "model", "owned_by": "tinyctx",
             "context_window": ctx},
        ],
    }


@APP.post("/v1/responses")
async def responses(request: Request) -> Any:
    raw = await request.body()
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid JSON: {e}")

    # Snapshot the pristine body before we mutate (encrypted_content scrub,
    # dedup, purge, historian). The Historian update step gets this pristine
    # copy so it never works from a previously-mutated history (preserving
    # the pristine-recomputation invariant from compactor.py).
    raw_body = json.loads(json.dumps(body))

    sid = _session_id(request, body)
    trace = RequestTrace(session_id=sid)
    streak = _SESSION_ERROR_STREAK[sid]
    decision = decide(body, CFG, error_streak=streak)
    backend = _select_backend(decision)

    # Allow client to force a specific route via the model id sent.
    requested_model = (body.get("model") or "").lower()
    trace.requested_model = requested_model
    if requested_model == "tinyctx-local":
        backend = CFG.local
        decision = Decision("local", "client requested tinyctx-local",
                            est_input_tokens=decision.est_input_tokens,
                            turn_count=decision.turn_count)
        trace.forced_by_client_model = True
    elif requested_model == "tinyctx-frontier":
        backend = CFG.frontier
        decision = Decision("frontier", "client requested tinyctx-frontier",
                            est_input_tokens=decision.est_input_tokens,
                            turn_count=decision.turn_count)
        trace.forced_by_client_model = True

    trace.route = decision.route
    trace.route_reason = decision.reason
    trace.is_compaction = decision.is_compaction
    trace.est_input_tokens = decision.est_input_tokens
    trace.turn_count = decision.turn_count
    trace.error_streak = streak

    # Sanitize before any model swap.
    if CFG.sanitize_encrypted_content:
        trace.encrypted_content_stripped = _count_encrypted(body)
        body = strip_encrypted_content(body)

    # Cache-aware gate for history mutations: only fire dedup/purge/historian
    # substitution when the cache prefix is likely stale anyway (TTL elapsed)
    # or we're heading into a forced compaction (context-usage threshold).
    # Otherwise leave history untouched so prompt-cache reads stay cheap.
    want_mutation = (CFG.dedup_tool_calls or CFG.purge_failed_tool_inputs
                     or CFG.historian_substitute)
    trace.mutation_wanted = want_mutation
    if want_mutation:
        fire, gate_reason = _MUTATOR.should_apply(
            sid,
            est_tokens=decision.est_input_tokens,
            max_tokens=int(body.get("metadata", {}).get("context_window")
                           or CFG.default_context_window),
        )
        trace.mutation_fired = fire
        trace.mutation_gate_reason = gate_reason
        _log("mutation_gate", session=sid, fire=fire, reason=gate_reason)
        if fire:
            if CFG.dedup_tool_calls:
                pre_dedup = json.dumps(body)
                body = dedup_tool_calls(body)
                trace.deduped_calls = _count_dedup_placeholders(body) - \
                                      pre_dedup.count("[tinyctx: identical call deduped")
            if CFG.purge_failed_tool_inputs:
                pre_purge = json.dumps(body)
                body = purge_failed_tool_inputs(
                    body, after_turns=CFG.failed_input_after_turns)
                trace.purged_inputs = _count_purge_placeholders(body) - \
                                      pre_purge.count("[tinyctx: failed input purged")
            if CFG.historian_substitute:
                pre_sub = json.dumps(body)
                body = historian.apply_to_body(
                    body, sid, recent_keep=CFG.historian_recent_keep)
                if "<tinyctx-historian-digest" in json.dumps(body) and \
                   "<tinyctx-historian-digest" not in pre_sub:
                    trace.historian_substituted = True
            _MUTATOR.mark_applied(sid)

    # Async background: run the Historian update if enabled. We hand it the
    # ORIGINAL request body (pre-mutation) so it sees pristine history, in
    # keeping with the pristine-recomputation invariant.
    if CFG.historian_enabled:
        try:
            project_root_hdr = request.headers.get("x-codex-cwd")
            project_root = (Path(project_root_hdr)
                            if project_root_hdr else None)
            historian.spawn_update(
                sid, raw_body, CFG.local,
                min_new_turns=CFG.historian_min_new_turns,
                recent_keep=CFG.historian_recent_keep,
                project_root=project_root,
            )
        except Exception as e:  # noqa: BLE001
            _log("historian_spawn_failed", session=sid, error=str(e))

    # Inject the advisor sub-agent usage hint into instructions BEFORE
    # rewrite_model — the inject function reads body.model to skip the
    # advisor's own sub-thread (model="tinyctx-frontier"). After this
    # call, rewrite_model overwrites body.model with the backend's id.
    body = inject_advisor_hint(body)

    if backend.model:
        body = rewrite_model(body, backend.model)
    trace.target_model = backend.model
    trace.target_wire_api = backend.wire_api

    # Per-backend tool/field scrubbing. Codex emits codex-specific tool
    # `type`s (web_search, image_generation, namespace) and request fields
    # (client_metadata, prompt_cache_key) that strict OpenAI-compat
    # backends like LMStudio reject with HTTP 400. Frontier defaults to
    # pass-through; local defaults to keeping only `function`-type tools.
    if backend.supported_tool_types:
        before_tools = body.get("tools", []) or []
        before_types = {t.get("type") for t in before_tools
                        if isinstance(t, dict)}
        trace.tools_before = len(before_tools)
        # Expand codex 0.128+'s `type=namespace` MCP wrappers into
        # type=function entries first, so the Advisor MCP tool (and any
        # other MCP-registered function tools) survive scrub.
        body = expand_mcp_namespaces(
            body,
            prefix_inner=os.environ.get("TINYCTX_MCP_NAME_NO_PREFIX") != "1",
        )
        body = scrub_unsupported_tools(
            body, supported_types=backend.supported_tool_types)
        after_tools = body.get("tools", []) or []
        after_types = {t.get("type") for t in after_tools
                       if isinstance(t, dict)}
        trace.tools_after = len(after_tools)
        trace.tool_types_dropped = sorted(
            t for t in before_types - after_types if t)
    else:
        ts = body.get("tools", []) or []
        trace.tools_before = len(ts)
        trace.tools_after = len(ts)

    if backend.strip_request_fields:
        before_keys = set(body.keys())
        body = strip_unsupported_responses_fields(
            body, drop=backend.strip_request_fields)
        trace.fields_stripped = sorted(before_keys - set(body.keys()))
    if backend.inject_defaults:
        body = inject_responses_defaults(body, backend.inject_defaults)
        trace.fields_injected = sorted(backend.inject_defaults.keys())

    _log("route", session=sid, decision=decision.route, reason=decision.reason,
         is_compaction=decision.is_compaction, est_tokens=decision.est_input_tokens,
         turns=decision.turn_count, target=backend.base_url, model=backend.model,
         streak=streak, requested_model=requested_model)

    headers = _forward_headers(request, backend)
    is_stream = bool(body.get("stream", False))

    # Compaction debate path: when codex's handoff-summary fingerprint hit
    # AND we're routing to local AND debate is enabled AND history is long
    # enough to warrant the extra 3 calls. Bypasses the normal forward.
    if (decision.is_compaction
            and decision.route == "local"
            and CFG.compactor_debate
            and decision.est_input_tokens >= CFG.compactor_min_history_tokens):
        trace.compactor_used = True
        return await _compactor_response(body, backend, is_stream, sid,
                                         project_root=request.headers.get("x-codex-cwd"),
                                         trace=trace)

    if backend.wire_api == "responses":
        url = backend.base_url.rstrip("/") + "/responses"
        forward_body = body
    else:
        # Local backend speaks chat-completions. Convert the body and fix URL.
        url = backend.base_url.rstrip("/") + "/chat/completions"
        forward_body = normalize_for_chat(body)

    trace.target_url = url
    trace.is_stream = is_stream
    return await _forward(url, headers, forward_body, is_stream, sid, decision,
                          translate_tool_calls=backend.translate_tool_calls,
                          chat_to_responses=(backend.wire_api != "responses"),
                          trace=trace)


@APP.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Convenience: also accept chat-completions traffic and route the same way.
    Useful for tools that expect /v1/chat/completions (e.g. Aider, Cline).
    """
    raw = await request.body()
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid JSON: {e}")

    sid = _session_id(request, body)
    streak = _SESSION_ERROR_STREAK[sid]
    decision = decide(body, CFG, error_streak=streak)
    backend = _select_backend(decision)

    if backend.model:
        body["model"] = backend.model

    _log("route_chat", session=sid, decision=decision.route, reason=decision.reason,
         target=backend.base_url, model=backend.model)

    headers = _forward_headers(request, backend)
    url = backend.base_url.rstrip("/") + "/chat/completions"
    return await _forward(url, headers, body, bool(body.get("stream", False)),
                          sid, decision,
                          translate_tool_calls=backend.translate_tool_calls)


def _forward_headers(request: Request, backend: BackendCfg) -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    # Forward Authorization unless the backend has its own key env var.
    auth = request.headers.get("authorization")
    api_key = _resolve_api_key(backend, auth)
    if api_key:
        h["Authorization"] = api_key if api_key.lower().startswith(("bearer ", "basic ")) else f"Bearer {api_key}"
    # Forward codex-specific routing headers if present (helps upstream debugging).
    for k in ("openai-beta", "openai-version", "x-codex-session-id"):
        v = request.headers.get(k)
        if v:
            h[k] = v
    h.update(backend.headers)
    return h


async def _forward(url: str, headers: dict[str, str], body: dict[str, Any],
                   is_stream: bool, sid: str, decision: Decision,
                   *, translate_tool_calls: bool = False,
                   chat_to_responses: bool = False,
                   trace: RequestTrace | None = None) -> Any:
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    if is_stream:
        return StreamingResponse(
            _stream_proxy(url, headers, body, sid, decision, timeout,
                          translate_tool_calls=translate_tool_calls,
                          chat_to_responses=chat_to_responses,
                          trace=trace),
            media_type="text/event-stream",
        )
    started = time.time()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            _SESSION_ERROR_STREAK[sid] += 1
            _log("upstream_error", session=sid, error=str(e), url=url)
            if trace is not None:
                trace.status = 0
                trace.elapsed_s = round(time.time() - started, 3)
                trace.emit(CFG.log_dir)
            return JSONResponse({"error": {"message": str(e), "type": "tinyctx_upstream"}},
                                status_code=502)
        if r.status_code >= 400:
            _SESSION_ERROR_STREAK[sid] += 1
            if trace is not None:
                trace.status = r.status_code
                trace.elapsed_s = round(time.time() - started, 3)
                trace.emit(CFG.log_dir)
            return JSONResponse(content=_safe_json(r), status_code=r.status_code)
        _SESSION_ERROR_STREAK[sid] = 0
        payload = _safe_json(r)
        translated_calls = 0
        if translate_tool_calls and isinstance(payload, dict):
            new_payload = rebuild_response(payload)
            if new_payload is not payload:
                # rebuild swapped some message text → function_call items
                translated_calls = sum(1 for it in new_payload.get("output", [])
                                       if isinstance(it, dict)
                                       and it.get("type") == "function_call")
                payload = new_payload
        if trace is not None:
            trace.status = r.status_code
            trace.bytes_out = len(json.dumps(payload).encode("utf-8")) if isinstance(payload, dict) else 0
            trace.translated = translate_tool_calls
            trace.translated_calls = translated_calls
            trace.elapsed_s = round(time.time() - started, 3)
            trace.emit(CFG.log_dir)
        return JSONResponse(content=payload, status_code=r.status_code)


async def _stream_proxy(url: str, headers: dict[str, str], body: dict[str, Any],
                        sid: str, decision: Decision,
                        timeout: httpx.Timeout,
                        *, translate_tool_calls: bool = False,
                        chat_to_responses: bool = False,
                        trace: RequestTrace | None = None) -> AsyncIterator[bytes]:
    started = time.time()
    bytes_out = 0
    status = 200
    translator: StreamTranslator | ChatToResponsesTranslator | None
    if chat_to_responses:
        translator = ChatToResponsesTranslator()
    elif translate_tool_calls:
        translator = StreamTranslator()
    else:
        translator = None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                status = r.status_code
                if r.status_code >= 400:
                    _SESSION_ERROR_STREAK[sid] += 1
                    text = (await r.aread()).decode("utf-8", "replace")
                    _log("upstream_error", session=sid, status=r.status_code,
                         url=url, body=text[:2000])
                    yield f"event: error\ndata: {json.dumps({'status': r.status_code, 'body': text[:2000]})}\n\n".encode()
                    return
                _SESSION_ERROR_STREAK[sid] = 0
                async for chunk in r.aiter_raw():
                    bytes_out += len(chunk)
                    if translator is None:
                        yield chunk
                    else:
                        for out in translator.feed(chunk):
                            yield out
                if translator is not None:
                    for out in translator.flush():
                        yield out
    except httpx.HTTPError as e:
        _SESSION_ERROR_STREAK[sid] += 1
        _log("stream_error", session=sid, error=str(e))
        status = 0
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n".encode()
    finally:
        elapsed = round(time.time() - started, 3)
        _log("stream_done", session=sid, route=decision.route, bytes=bytes_out,
             translated=bool(translator),
             elapsed_s=elapsed)
        if trace is not None:
            trace.status = status
            trace.bytes_out = bytes_out
            trace.translated = bool(translator)
            trace.translated_calls = (translator._emitted_calls
                                       if translator is not None else 0)
            trace.elapsed_s = elapsed
            trace.emit(CFG.log_dir)


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except json.JSONDecodeError:
        return {"raw": r.text}


# ─── small counters used to populate RequestTrace transformation diffs ──

_REASONING_TYPES = {"reasoning", "reasoning_summary", "thinking"}


def _count_encrypted(body: dict[str, Any]) -> int:
    """Count how many reasoning items in `body` carry encrypted_content
    (about to be stripped by sanitize)."""
    n = 0
    for key in ("input", "messages"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("type") or it.get("role")
            if t in _REASONING_TYPES and "encrypted_content" in it:
                n += 1
            content = it.get("content")
            if isinstance(content, list):
                for c in content:
                    if (isinstance(c, dict) and c.get("type") in _REASONING_TYPES
                            and "encrypted_content" in c):
                        n += 1
    return n


def _count_dedup_placeholders(body: dict[str, Any]) -> int:
    return json.dumps(body, default=str).count(
        "[tinyctx: identical call deduped")


def _count_purge_placeholders(body: dict[str, Any]) -> int:
    return json.dumps(body, default=str).count(
        "[tinyctx: failed input purged")


async def _compactor_response(body: dict[str, Any], backend: BackendCfg,
                              is_stream: bool, sid: str,
                              *, project_root: str | None,
                              trace: RequestTrace | None = None) -> Any:
    """Run the 3-role debate, persist if configured, and return the merged
    summary in the wire shape codex expects.

    On any failure we fall back to forwarding the original request so the
    user never sees a hard crash from the compactor."""
    started = time.time()
    try:
        summary, telemetry, structured = await compact_with_debate(body, backend)
    except Exception as e:  # noqa: BLE001
        _log("compactor_failed", session=sid, error=str(e))
        structured = {"compartments": [], "facts": [], "open_questions": []}
        if trace is not None:
            trace.compactor_outcome = "failed_fallback"
            trace.elapsed_s = round(time.time() - started, 3)
            trace.emit(CFG.log_dir)
        # Fall through to a normal forward — proxy never blocks codex.
        url = backend.base_url.rstrip("/") + ("/responses"
                                              if backend.wire_api == "responses"
                                              else "/chat/completions")
        forward_body = body if backend.wire_api == "responses" else normalize_for_chat(body)
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        return await _forward(url, {"Content-Type": "application/json"},
                              forward_body, is_stream, sid,
                              Decision("local", "compactor_fallback"),
                              trace=None)

    _log("compactor_done", session=sid, telemetry=telemetry,
         summary_chars=len(summary), elapsed_s=round(time.time() - started, 3))

    if CFG.save_compactions and project_root:
        try:
            from pathlib import Path
            saved = save_compaction(Path(project_root), sid, summary,
                                    telemetry=telemetry,
                                    structured=structured)
            _log("compactor_saved", session=sid, path=str(saved))
        except Exception as e:  # noqa: BLE001
            _log("compactor_save_failed", session=sid, error=str(e))

    model_id = backend.model or body.get("model") or "tinyctx-compactor"
    if trace is not None:
        trace.target_model = model_id
        trace.target_url = backend.base_url
        trace.target_wire_api = "responses"
        trace.is_stream = is_stream
        trace.bytes_out = len(summary.encode("utf-8"))
        trace.compactor_outcome = telemetry.get("outcome", "judged")
        trace.elapsed_s = round(time.time() - started, 3)
        trace.status = 200
        trace.emit(CFG.log_dir)
    if is_stream:
        sse = build_responses_api_sse(summary, model_id)
        async def _emit() -> AsyncIterator[bytes]:
            yield sse
        return StreamingResponse(_emit(), media_type="text/event-stream")
    return JSONResponse(build_responses_api_payload(summary, model_id))


def main() -> None:
    import uvicorn
    uvicorn.run(APP, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
