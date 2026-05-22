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

from . import auto_scout
from .compactor import (
    build_responses_api_payload,
    build_responses_api_sse,
    compact_with_debate,
)
from .config import (
    BackendCfg,
    Config,
    effective_proactive_compact_threshold,
    effective_strip_request_fields,
    load_config,
)
from .continuity import save_compaction
from . import historian
from . import lingua
from .orchestration_runtime import apply_orchestration
from .read_delta import collapse_repeated_reads
from .router import (
    Decision,
    RouteContext,
    Router,
    count_turns,
    decide,
    estimate_tokens,
    is_compaction_request,
    _flatten_text,
)
from .sanitize import (
    CacheAwareMutator,
    cap_responses_fields,
    collect_failure_signals,
    dedup_tool_calls,
    drop_orphan_tool_outputs,
    inject_responses_defaults,
    normalize_for_chat,
    proactive_compact,
    purge_failed_tool_inputs,
    rewrite_input_roles,
    rewrite_model,
    expand_mcp_namespaces,
    inject_advisor_hint,
    scrub_unsupported_tools,
    strip_encrypted_content,
    strip_unsupported_responses_fields,
    trim_tools_for_frontier,
)
from . import post_stream as _post
from .request_phase import RequestPhase, set_phase as _phase_set
from . import retry_policy
from . import stall_watchdog as _stall
from . import stream_relay as _relay
from .tool_call_translator import ChatToResponsesTranslator, StreamTranslator, rebuild_response
from .trace import RequestTrace


CFG: Config = load_config()
APP = FastAPI(title="tinyctx", version="0.1.0")

# Mount the live dashboard at /dashboard. Five endpoints; vanilla HTML+JS;
# zero new deps. See tinyctx/dashboard.py.
try:
    from . import dashboard as _dashboard
    _dashboard.register(APP, CFG.log_dir)
except Exception as _e:  # noqa: BLE001 — dashboard must never block proxy boot
    sys.stderr.write(f"tinyctx dashboard register failed: {_e}\n")

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
    except Exception:  # noqa: BLE001 — _log must never raise
        # Why: _log is called everywhere including hot streaming paths.
        # A disk-write failure (full, read-only, perms) must not bubble
        # up — stderr emission above already happened.
        pass


def _extract_valid_tool_names(body: dict[str, Any] | Any) -> set[str] | None:
    """Pull tool names out of a request body's `tools` array. Returns None
    when the body has no `tools` field, or the field is empty or unusable —
    None signals the translator to skip name validation entirely (legacy
    pass-through). Handles two common entry shapes:
        {"type": "function", "name": "shell", ...}
        {"type": "function", "function": {"name": "shell", ...}}
    Unknown shapes are skipped silently (lenient), not fatal.
    """
    if not isinstance(body, dict):
        return None
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    names: set[str] = set()
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            fn = entry.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names or None


_PROACTIVE_SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing the OLDER turns of an in-progress coding session "
    "that is about to be truncated to fit within a context window. The "
    "RECENT turns will continue verbatim after your summary; the model "
    "reading your summary is the same one that wrote those turns. Your "
    "job is to give it enough memory — including SHELL ACTIVITY, FILE "
    "EDITS, and ERROR SIGNALS — to keep working without losing the thread.\n\n"
    "The input you receive is split into:\n"
    "  - '## Pre-extracted execution signals' (recent edits / shell / errors)\n"
    "  - '## Raw transcript of older turns'\n"
    "Use BOTH. Don't drop the execution signals — they are load-bearing.\n\n"
    "Output a 400-900 word markdown handoff with these sections:\n"
    "## Session goal\n"
    "  The user's actual goal. Be specific. Include any pivot the user made.\n"
    "## Recent file edits\n"
    "  Concrete paths touched and what changed. One line per edit, newest "
    "  last. Quote the pre-extracted signals when relevant.\n"
    "## Recent shell activity\n"
    "  Exact commands run and what they returned (success/failure + key "
    "  output line). Cover at least the 10 most recent. One line per "
    "  command. Drop pure noise (`echo`, `ls /tmp`); keep build/test/git/curl.\n"
    "## Errors / failures encountered\n"
    "  Distinct error patterns observed. Include the actual error text so "
    "  the next turn can reason about them.\n"
    "## Plan progress\n"
    "  Latest update_plan / TodoWrite content (verbatim if short). Items "
    "  checked off so far if extractable.\n"
    "## Acceptance / verification signals\n"
    "  Tests passed / failed (counts + names if visible). Build status. "
    "  Linter/type-check status if visible.\n"
    "## Open questions / blockers / next step\n"
    "  Pending work the model was about to do, or the question it was "
    "  about to ask. Be explicit so the next turn can resume.\n\n"
    "Rules:\n"
    "  - Concrete > abstract. Say \"removed RayNeo SLAM env knobs in\n"
    "    com.foo.Bar.kt:42\" not \"made some Kotlin changes\".\n"
    "  - Drop redundancy and chitchat.\n"
    "  - Do NOT invent anything not in the conversation.\n"
    "  - Preserve unique markers, file paths, error class names, command "
    "    names verbatim — the next model uses them as anchors.\n"
    "  - Keep it terse but COMPLETE on the execution signals."
)


def _make_local_summarizer(local: BackendCfg):
    """Return a synchronous summarizer callable suitable for
    sanitize.proactive_compact. The callable calls the local backend's
    /chat/completions endpoint to produce a real handoff summary instead
    of the deterministic '[N older turns omitted]' placeholder.

    Cache hits in proactive_compact's session-keyed cache reuse the
    summary, so this fires roughly once per session even on long runs.

    Failures fall back to the placeholder via proactive_compact's own
    exception handler — never blocks the request.
    """
    def _summarize(blob: str) -> str:
        url = local.base_url.rstrip("/") + "/chat/completions"
        api_key = os.environ.get(local.api_key_env or "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": local.model or "local",
            "messages": [
                {"role": "system", "content": _PROACTIVE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": blob[:120_000]},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            # Bumped from 1200 → 2000 to fit the structured multi-section
            # handoff (goal + edits + shell + errors + plan + verification
            # + next step). 1200 truncated the shell/errors sections in
            # practice and was the proximate cause of "execution state
            # lost after compaction".
            "max_tokens": 2000,
            "stream": False,
        }
        # Synchronous httpx — proactive_compact wraps us in asyncio.to_thread
        # at the proxy layer, so we don't block the event loop.
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0,
                                                write=15.0, pool=5.0)) as c:
            r = c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("local summarizer returned empty")
        return text.strip()
    return _summarize


def _get_dotted(d: dict[str, Any], path: str) -> Any:
    """Read a dotted-path value from a nested dict; None if missing."""
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _chat_to_responses_payload(chat: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-streaming OpenAI chat-completions response JSON into
    a Responses-API response JSON. Mirrors what ChatToResponsesTranslator
    does for the streaming path. DeepSeek-style `reasoning_content` is
    surfaced as a Responses-API reasoning item before the assistant
    message; tool calls are surfaced as function_call items.
    """
    rid = "resp_" + uuid4().hex[:24]
    msg = (chat.get("choices") or [{}])[0].get("message") or {}
    output: list[dict[str, Any]] = []

    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        output.append({
            "id": "rs_" + uuid4().hex[:24],
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": rc}],
        })

    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict) or tc.get("type") != "function":
            continue
        fn = tc.get("function") or {}
        output.append({
            "id": "fc_" + uuid4().hex[:24],
            "type": "function_call",
            "call_id": tc.get("id") or "call_" + uuid4().hex[:16],
            "name": fn.get("name") or "",
            "arguments": fn.get("arguments") or "",
            "status": "completed",
        })

    text = msg.get("content")
    if isinstance(text, str) and text:
        output.append({
            "id": "msg_" + uuid4().hex[:24],
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })

    usage = chat.get("usage") or {}
    out_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens",
                                  usage.get("prompt_tokens", 0) +
                                  usage.get("completion_tokens", 0)),
    }
    if isinstance(usage.get("completion_tokens_details"), dict):
        rt = usage["completion_tokens_details"].get("reasoning_tokens")
        if isinstance(rt, int):
            out_usage["output_tokens_details"] = {"reasoning_tokens": rt}

    return {
        "id": rid,
        "object": "response",
        "created_at": chat.get("created", int(time.time())),
        "status": "completed",
        "model": chat.get("model") or "",
        "output": output,
        "usage": out_usage,
    }


def _resolve_api_key(backend: BackendCfg, codex_auth: str | None) -> str | None:
    if backend.api_key_env:
        v = os.environ.get(backend.api_key_env)
        if v:
            return v
    # fall back to codex's own Authorization header (passthrough mode)
    if codex_auth:
        return codex_auth
    # Last resort: read codex's OAuth access_token from ~/.codex/auth.json so
    # non-codex clients (task-master, Aider, ...) can share codex's ChatGPT
    # subscription auth without setting OPENAI_API_KEY. Token rotation is
    # handled by the codex CLI itself; we just re-read on each request.
    try:
        with open(os.path.expanduser("~/.codex/auth.json")) as f:
            tok = json.load(f).get("tokens", {}).get("access_token")
        if tok:
            return tok
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        # Why: codex auth.json missing / malformed / wrong shape — all
        # are normal for clients not using ChatGPT subscription auth.
        # Returning None falls through to the no-key case in callers.
        pass
    return None


def _select_backend(decision: Decision) -> BackendCfg:
    return CFG.local if decision.route == "local" else CFG.frontier


def _build_frontier_retry_target(
        request: Request | None,
        body: dict[str, Any],
        reason: str,
) -> tuple[str, dict[str, str], dict[str, Any], Decision, BackendCfg]:
    """Rebuild url/headers/body for a retry that escalates to frontier.

    Used by the unified retry layer (retry_policy.classify_failure ==
    retry_escalate) to switch from local to frontier mid-request.
    Crucially, this can run after a local `wire_api=chat` normalization,
    so `body` may be in chat-completions shape (`messages` + local model).
    We coerce to frontier Responses shape and rewrite model.

    `request` may be None when called from a context that doesn't have
    the FastAPI Request handy (e.g. retry inside a stream producer).
    In that case the caller passes a pre-built headers dict separately;
    we still produce frontier-shaped url + decision so the dispatch
    loop can swap them in.
    """
    backend = CFG.frontier
    url = backend.base_url.rstrip("/") + "/responses"
    retry_body = _coerce_frontier_retry_body(body, backend)
    headers = (_forward_headers(request, backend) if request is not None
               else {"Content-Type": "application/json",
                     "Accept": "text/event-stream"})
    decision = Decision(
        "frontier",
        f"retry_escalate: {reason}",
        is_compaction=False,
    )
    return url, headers, retry_body, decision, backend


def _coerce_frontier_retry_body(
    body: dict[str, Any],
    backend: BackendCfg,
) -> dict[str, Any]:
    """Convert local retry payload into frontier-friendly Responses shape."""
    try:
        out = json.loads(json.dumps(body))
    except (TypeError, ValueError):
        out = dict(body) if isinstance(body, dict) else {}
    if not isinstance(out, dict):
        out = {}

    if backend.model:
        out["model"] = backend.model

    # Escalation can happen after local `wire_api=chat` conversion.
    # Frontier /responses expects `input`, not `messages`.
    if "messages" in out:
        if "input" not in out:
            out["input"] = out.get("messages")
        out.pop("messages", None)

    # Chat payloads use max_tokens; responses payloads use max_output_tokens.
    if "max_tokens" in out and "max_output_tokens" not in out:
        try:
            out["max_output_tokens"] = int(out["max_tokens"])
        except (TypeError, ValueError):
            pass
        out.pop("max_tokens", None)

    return out


def _session_id(request: Request, body: dict[str, Any]) -> str:
    sid = request.headers.get("x-codex-session-id")
    sid = sid or body.get("session_id") or body.get("metadata", {}).get("session_id")
    return sid or "global"


def _project_session_key(request: Request, sid: str) -> str:
    """Composite key scoping per-session state to per-project as well.

    codex.app currently does NOT send `x-codex-session-id`, so almost
    all traces end up with sid="global" — if we keyed proactive_compact
    cache, error_streak, and mutation-gate timing on sid alone, projects
    would cross-contaminate (project A's summary injected into project B,
    A's failure escalating B to frontier, etc.).

    This combines sid with a hash of `x-codex-cwd` (which codex DOES
    send and auto_scout already uses) to get true per-project isolation
    even when the upstream collapses sessions to "global".

    Falls back to plain `sid` when cwd header is absent (no regression
    for existing callers; the behavior is just unscoped same as before).
    """
    import hashlib
    cwd = request.headers.get("x-codex-cwd") or ""
    if not cwd:
        return sid
    cwd_hash = hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:8]
    return f"{cwd_hash}:{sid}"


def _conversation_session_key(proj_sid: str, body: dict[str, Any]) -> str:
    """Conversation-scoped composite key for per-thread state.

    Three modules rely on per-conversation isolation: synthetic_continue's
    injection counter, empty_response_guard's force-frontier flag, and
    stuck_loop's per-turn reminder gate. Earlier this keyed on
    `prompt_cache_key` alone; live trace showed that field drifts mid-
    conversation (OpenAI prompt-cache invalidation), which silently reset
    every per-conv counter and prevented the P2 budget cap from ever
    firing. See `tinyctx/conv_id.py` for the resolver and full rationale.
    """
    from .conv_id import resolve_conv_key
    return resolve_conv_key(proj_sid, body)


_STALL_WATCHDOG_TASK: asyncio.Task | None = None


@APP.on_event("startup")
async def _start_stall_watchdog_on_startup() -> None:
    """Spawn the mid-stream stall watchdog. On stall: set the phase,
    cancel the in-flight relay producer task (if registered), flag the
    next request to escalate to frontier, and record a forensic event.
    The cancel-and-retry primary path unblocks the wedged stream
    immediately; the force-frontier flag is belt+suspenders for the
    follow-up turn codex naturally retries."""
    global _STALL_WATCHDOG_TASK
    if not CFG.stall_watchdog_enabled:
        return

    async def _on_stall(proj_sid: str, conv_sid: str | None = None) -> None:
        # Prefer conv_sid so the force-frontier flag scopes to ONE
        # conversation; falling back to proj_sid only if conv_sid wasn't
        # captured (older mark_event call sites or chat-completions path).
        escalate_key = conv_sid if conv_sid else proj_sid
        try:
            _phase_set(proj_sid, RequestPhase.stalled, "")
        except Exception:  # noqa: BLE001 — phase emission is telemetry; stall handler must continue
            pass
        # Cancel-and-retry: if the relay producer task is registered for
        # this proj_sid, cancel it so the consumer wakes up, emits a
        # clean SSE terminator, and unblocks codex. The flag-only
        # fallback below still fires as belt+suspenders for the next
        # turn — codex's natural retry on incomplete-status will then
        # route to frontier.
        cancelled = False
        try:
            cancelled = _stall.cancel_active_task(proj_sid)
        except Exception:  # noqa: BLE001 — best-effort task cancel
            # Why: cancel_active_task may race with task already done.
            # Treat as "not cancelled" and continue with flag-only path.
            cancelled = False
        try:
            from . import empty_response_guard as _erg
            _erg.force_next_to_frontier(escalate_key, "mid_stream_stall")
            _phase_set(proj_sid, RequestPhase.escalated_to_frontier, "")
        except Exception:  # noqa: BLE001 — escalation hint; stall handler must continue
            # Why: setting the escalation flag is an optimization for
            # the NEXT turn; current turn already cancelled above.
            pass
        trigger_label = "stall_cancelled" if cancelled else "stall_kill"
        elapsed: float | None = None
        try:
            elapsed = _stall.seconds_since_event(proj_sid)
        except Exception:  # noqa: BLE001 — elapsed is for logging only
            elapsed = None
        try:
            _log(trigger_label, session=proj_sid, conv_sid=conv_sid,
                 escalate_key=escalate_key,
                 threshold_s=CFG.stall_threshold_s,
                 elapsed_silent_s=elapsed,
                 task_cancelled=cancelled)
        except Exception:  # noqa: BLE001 — _log already swallows internally; belt+suspenders
            pass
        if CFG.forensics_enabled:
            try:
                from . import forensics as _fx
                forensics_dir = CFG.log_dir.parent / "forensics"
                _fx.write_forensics_dump(
                    forensics_dir, proj_sid,
                    trigger=trigger_label,
                    response_buffer="",
                    extra={"threshold_s": CFG.stall_threshold_s,
                           "escalation": "force_next_to_frontier",
                           "escalate_key": escalate_key,
                           "conv_sid": conv_sid,
                           "task_cancelled": cancelled,
                           "elapsed_silent_s": elapsed},
                    max_dumps=CFG.forensics_max_dumps,
                )
            except Exception:  # noqa: BLE001 — forensics is best-effort
                # Why: forensics dump in the stall path must never crash
                # the watchdog. forensics module itself swallows errors.
                pass

    _STALL_WATCHDOG_TASK = _stall.start_watchdog(
        check_interval_s=CFG.stall_check_interval_s,
        threshold_s=CFG.stall_threshold_s,
        on_stall=_on_stall,
    )


@APP.on_event("shutdown")
async def _stop_stall_watchdog_on_shutdown() -> None:
    global _STALL_WATCHDOG_TASK
    task = _STALL_WATCHDOG_TASK
    _STALL_WATCHDOG_TASK = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutdown handler
        # Why: shutdown handler awaiting the cancelled watchdog. Both
        # CancelledError (expected) and any other exception (broken
        # watchdog state) are swallowed because we're already shutting
        # down — nothing useful to do with the error.
        pass


@APP.on_event("startup")
def _auto_register_mcp_servers_on_startup() -> None:
    """One-shot: detect graphify/gitnexus, register them into
    ~/.codex/config.toml. Idempotent across restarts. See mcp_registry
    module for the full contract. Toggle off via
    CFG.auto_register_mcp_servers = False (or env-equivalent)."""
    if not CFG.auto_register_mcp_servers:
        return
    try:
        from . import mcp_registry
        result = mcp_registry.bootstrap(log_fn=lambda ev, **fields: _log(ev, **fields))
        _log("mcp_registry_bootstrap_done", **result)
    except Exception as e:  # noqa: BLE001 — never let startup hook crash the proxy
        _log("mcp_registry_bootstrap_failed", error=str(e))


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
    # State that's per-conversation must scope to per-project too — otherwise
    # all "global" sessions across different repos cross-contaminate. See
    # `_project_session_key` for the rationale. We pass `proj_sid` to anything
    # that holds per-session state (proactive_compact cache, error_streak,
    # mutation gate timing); plain `sid` stays in trace records and logs so
    # the user-visible session id is the codex one (not the prefix-hashed
    # composite).
    proj_sid = _project_session_key(request, sid)
    # conv_sid scopes per-conversation state (synthetic_continue counter,
    # empty_response_guard flag, stuck_loop reminder gate). Derived from
    # `prompt_cache_key` which codex sends per-thread (stable across turns
    # of one conversation, distinct for advisor sub-threads). Falls back
    # to proj_sid when the field is absent so old behavior is preserved.
    conv_sid = _conversation_session_key(proj_sid, body)
    trace = RequestTrace(session_id=sid)
    trace.project_session_key = proj_sid
    _phase_set(proj_sid, RequestPhase.received, trace.request_id)
    streak = _SESSION_ERROR_STREAK[proj_sid]
    _phase_set(proj_sid, RequestPhase.classifying, trace.request_id)

    # Body-derived signals computed once, fed into guards + Router. We
    # used to compute these via `decide()` and then discard everything
    # except the decision; now we extract them directly and let the
    # Router (called below) consume them via RouteContext.
    _blob = _flatten_text(body)
    est_tokens = estimate_tokens(_blob)
    turns = count_turns(body)
    is_compaction = is_compaction_request(_blob)

    _phase_set(proj_sid, RequestPhase.routing, trace.request_id)

    # Tool-call frequency tracking — mine body.input for function_call
    # items (deduped by call_id) so the dashboard can show which MCP
    # servers / built-in tools are actually being used. Fire-and-forget
    # — never raises. See tinyctx/tool_metrics.py.
    try:
        from . import tool_metrics as _tm
        _tm.record_from_body(body)
    except Exception:  # noqa: BLE001 — tool_metrics is observability
        # Why: tool_metrics is observability for the dashboard; a
        # parse failure must not affect routing or request handling.
        pass

    requested_model = (body.get("model") or "").lower()
    trace.requested_model = requested_model
    if requested_model in ("tinyctx-local", "tinyctx-frontier"):
        trace.forced_by_client_model = True
        if requested_model == "tinyctx-frontier":
            # The agent invoked advisor — note timestamp so the stuck-loop
            # watchdog skips its grace window and doesn't nudge the parent
            # session right after the agent already escalated. Mark under
            # proj_sid so any conversation in this project (incl. the parent
            # that triggered the advisor) sees the grace; advisor's own
            # conv_sid is separate and its watchdog state stays isolated.
            try:
                from . import stuck_loop
                stuck_loop.mark_advisor_call(proj_sid)
            except Exception:  # noqa: BLE001 — instrumentation only
                pass

    # Pre-flight guard pipeline. Replaces the previously-inline calls
    # into empty_response_guard / stuck_loop / synthetic_continue /
    # soft_completion / plan_persistence. Each guard is a small class
    # in tinyctx/guards.py wrapping the existing module function. The
    # pipeline runs guards in priority order; ForceFrontierGuard sets
    # `guard_ctx.force_route` which is consumed by Router._force_route_rule
    # below. Per-guard trace fields and `_phase_set` calls are restored
    # from the result list to preserve the observability the old inline
    # code emitted.
    guard_force_route: str | None = None
    try:
        from .guards import (
            BudgetReminderGuard,
            ForceFrontierGuard,
            GuardContext,
            GuardPipeline,
            PlanPersistenceInjector,
            trace_guard_results,
            SoftCompletionGate,
            StuckLoopGuard,
        )
        cwd_hdr = request.headers.get("x-codex-cwd") or ""
        state_dir = CFG.log_dir.parent / "state"
        active_guards: list[Any] = []
        if CFG.empty_response_guard_enabled:
            active_guards.append(ForceFrontierGuard())
        # Budget reminder is always enabled (its skip-gates live inside
        # the guard); same gating shape as before.
        active_guards.append(BudgetReminderGuard(
            max_injections=CFG.max_continue_injections_per_session))
        if CFG.stuck_loop_watchdog_enabled:
            active_guards.append(StuckLoopGuard(
                turn_trigger=CFG.stuck_loop_turn_trigger,
                turn_gap=CFG.stuck_loop_turn_gap,
                advisor_grace_s=CFG.stuck_loop_advisor_grace_s))
        if CFG.soft_completion_gate_enabled:
            active_guards.append(SoftCompletionGate())
        if CFG.plan_persistence_enabled:
            active_guards.append(PlanPersistenceInjector(
                state_dir=state_dir,
                cwd=cwd_hdr,
                plan_ttl_s=CFG.plan_persistence_ttl_s,
                session_id=sid))
        guard_ctx = GuardContext(
            body=body, proj_sid=proj_sid, conv_sid=conv_sid,
            turn_count=turns,
            is_compaction=is_compaction,
            forced_by_client_model=trace.forced_by_client_model,
        )
        guard_results = GuardPipeline(active_guards).run(guard_ctx)
        body = guard_ctx.body
        guard_force_route = guard_ctx.force_route
        trace.guard_results = trace_guard_results(guard_results)

        # Apply per-guard side effects (trace fields, phase, log emit).
        # The route override itself is consumed by Router.decide via
        # `guard_force_route` — no decision rewrite here.
        for gr in guard_results:
            if gr.guard_name == "force_frontier":
                if gr.fired:
                    _phase_set(proj_sid, RequestPhase.empty_guarded,
                                trace.request_id)
                    _log("empty_response_guard_forced_frontier",
                          session=sid, proj_sid=proj_sid,
                          prev_completion_tokens=gr.additional_log.get(
                              "completion_tokens"),
                          prev_finish_reason=gr.additional_log.get(
                              "finish_reason"))
                elif gr.additional_log.get("exception_type"):
                    _log("empty_response_guard_error",
                          session=sid, error=gr.reason)
            elif gr.guard_name == "stuck_loop":
                if gr.fired:
                    trace.stuck_reminder_injected = True
                    trace.stuck_turn_count_at_inject = turns
                    _phase_set(proj_sid, RequestPhase.injecting,
                                trace.request_id)
                    _log("stuck_reminder_injected", session=sid,
                          proj_sid=proj_sid, turn_count=turns)
                elif gr.additional_log.get("exception_type"):
                    _log("stuck_loop_error", session=sid, error=gr.reason)
            elif gr.guard_name == "budget_reminder":
                if gr.fired:
                    _log("budget_exhausted_reminder_injected",
                          session=sid, proj_sid=proj_sid,
                          injection_count=gr.additional_log.get(
                              "injection_count"))
                elif gr.additional_log.get("exception_type"):
                    _log("budget_reminder_error", session=sid,
                          error=gr.reason)
            elif gr.guard_name == "soft_completion_gate":
                if gr.fired:
                    trace.soft_completion_gate_injected = True
                    trace.soft_completion_gate_pattern = (
                        gr.additional_log.get("pattern") or "")
                    _log("soft_completion_gate_injected", session=sid,
                          proj_sid=proj_sid,
                          pattern=gr.additional_log.get("pattern"))
                elif gr.additional_log.get("exception_type"):
                    _log("soft_completion_gate_error", session=sid,
                          error=gr.reason)
            elif gr.guard_name == "plan_persistence":
                if gr.fired:
                    # Two distinct logs depending on which sub-action fired.
                    if "saved" in (gr.reason or ""):
                        _log("plan_persistence_saved", session=sid,
                              cwd=cwd_hdr[:120],
                              turn_count=turns)
                    if "injected" in (gr.reason or ""):
                        _log("plan_persistence_injected",
                              session=sid, cwd=cwd_hdr[:120],
                              prev_turn_count=gr.additional_log.get(
                                  "prev_turn_count"),
                              updated=gr.additional_log.get("updated"))
                elif gr.additional_log.get("exception_type"):
                    _log("plan_persistence_error", session=sid,
                          error=gr.reason)

        # Single roll-up event so dashboards can see what fired in one
        # place without scanning across the per-guard log events above.
        _log("guards_pipeline_done", session=sid, proj_sid=proj_sid,
              conv_sid=conv_sid,
              fired=[gr.guard_name for gr in guard_results if gr.fired],
              skipped=[gr.guard_name for gr in guard_results
                       if not gr.fired])
    except Exception as e:  # noqa: BLE001 — pipeline must never block
        _log("guards_pipeline_error", session=sid, error=str(e))

    # Model-driven escalation (Anthropic Advisor Strategy alignment):
    # ask the LOCAL model itself whether this turn deserves the advisor.
    # Runs only when no higher-priority router rule will already escalate
    # — i.e. no compaction, no force_route, no explicit-frontier request,
    # no error_streak. We pass the classifier's score (sc.p) and reason
    # into the RouteContext; the Router's _classify_rule consumes it.
    # Failures are silent.
    classify_p = 0.0
    classify_reason = ""
    # Skip the classifier when another router rule will already escalate
    # to frontier (or pin to local), to avoid the ~200ms classifier RTT
    # whose result would be discarded. Mirrors the original pre-refactor
    # gates plus the new guard_force_route path.
    _streak_thr = getattr(CFG, "escalate_on_error_streak", 0) or 0
    _will_escalate_pre_classify = (
        guard_force_route is not None
        or (_streak_thr > 0 and streak >= _streak_thr)
    )
    if (not is_compaction
            and not trace.forced_by_client_model
            and guard_force_route is None):
        try:
            failure_signals = collect_failure_signals(body)
            from .guards import decision_from_failure_scan
            decision = decision_from_failure_scan(failure_signals)
            trace.failure_signal_score = int(decision.trace.get("score", 0))
            trace.failure_signals = [s.to_trace() for s in decision.signals]
            if decision.signals or decision.action != "ok":
                trace.guardrail_decisions.append(decision.to_trace())
            if decision.should_escalate:
                guard_force_route = "frontier"
                _phase_set(proj_sid, RequestPhase.escalated_to_frontier,
                           trace.request_id)
                _log(
                    "failure_signal_escalated_to_frontier",
                    session=sid,
                    proj_sid=proj_sid,
                    score=decision.trace.get("score", 0),
                    signals=[s.to_trace() for s in decision.signals],
                )
        except Exception as e:  # noqa: BLE001
            _log("failure_signal_scan_error", session=sid, error=str(e))
    _will_escalate_pre_classify = (
        guard_force_route is not None
        or (_streak_thr > 0 and streak >= _streak_thr)
    )
    if (CFG.self_classify_enabled
            and not is_compaction
            and not trace.forced_by_client_model
            and not _will_escalate_pre_classify):
        try:
            from . import self_classify
            api_key = (os.environ.get(CFG.local.api_key_env)
                       if CFG.local.api_key_env else None)
            sc = await self_classify.classify(
                body,
                local_base_url=CFG.local.base_url,
                local_model=CFG.local.model,
                api_key=api_key,
                timeout_s=CFG.self_classify_timeout_s,
                scope=proj_sid,
            )
            if sc is not None:
                trace.self_classify_p = sc.p
                trace.self_classify_reason = sc.reason
                trace.self_classify_cached = sc.cached
                if sc.escalate and sc.p >= CFG.self_classify_threshold:
                    classify_p = sc.p
                    classify_reason = sc.reason
                    trace.self_classify_overrode = True
                    _phase_set(proj_sid, RequestPhase.escalated_to_frontier,
                               trace.request_id)
        except Exception as e:  # noqa: BLE001 — classifier must never fail forward
            _log("self_classify_error", session=sid, error=str(e))

    # ─── Single consolidated route decision ──────────────────────────
    route_ctx = RouteContext(
        body=body,
        proj_sid=proj_sid,
        conv_sid=conv_sid,
        turn_count=turns,
        est_tokens=est_tokens,
        requested_model=requested_model,
        force_route=guard_force_route,
        error_streak=streak,
        is_compaction=is_compaction,
        classify_p=classify_p,
        classify_reason=classify_reason,
    )
    router = Router(CFG).with_codex_auth(request.headers.get("authorization"))
    decision = router.decide(route_ctx)
    backend = _select_backend(decision)

    trace.route = decision.route
    trace.route_reason = decision.reason
    trace.is_compaction = decision.is_compaction
    trace.est_input_tokens = decision.est_input_tokens
    trace.turn_count = decision.turn_count
    trace.error_streak = streak

    # Compaction boundary: codex emitted a handoff-summary request, so
    # the conversation's effective context is about to be rebuilt. Reset
    # per-conversation counters that should be measured against the
    # POST-compaction conversation, not the pre-compaction one. Without
    # this, a session that accumulated N injections pre-compaction would
    # trip the P2 budget cap (N + small_post_compact_count >= max) on
    # otherwise-healthy compacted sessions.
    if decision.is_compaction:
        try:
            from . import synthetic_continue as _syn_budget
            from . import stuck_loop as _stuck
            _syn_budget.reset_compaction_state(conv_sid, proj_sid=proj_sid)
            _stuck.reset_compaction_state(conv_sid, proj_sid=proj_sid)
            _log("compaction_state_reset", session=sid,
                 conv_sid=conv_sid, trigger="incoming_is_compaction")
        except Exception as e:  # noqa: BLE001
            _log("compaction_reset_error", session=sid, error=str(e))

    # (stuck_loop, budget_reminder, soft_completion guards now run
    # inside the GuardPipeline above — see tinyctx/guards.py.)

    # Sanitize before any model swap.
    if CFG.sanitize_encrypted_content:
        trace.encrypted_content_stripped = _count_encrypted(body)
        body = strip_encrypted_content(body)

    # Cache-aware gate for history mutations: only fire dedup/purge/historian
    # substitution / read_delta when the cache prefix is likely stale anyway
    # (TTL elapsed) or we're heading into a forced compaction (context-usage
    # threshold). Otherwise leave history untouched so prompt-cache reads
    # stay cheap.
    want_mutation = (CFG.dedup_tool_calls or CFG.purge_failed_tool_inputs
                     or CFG.historian_substitute or CFG.read_delta_enabled)
    trace.mutation_wanted = want_mutation
    if want_mutation:
        fire, gate_reason = _MUTATOR.should_apply(
            proj_sid,
            est_tokens=decision.est_input_tokens,
            max_tokens=int(body.get("metadata", {}).get("context_window")
                           or CFG.default_context_window),
        )
        trace.mutation_fired = fire
        trace.mutation_gate_reason = gate_reason
        _log("mutation_gate", session=sid, fire=fire, reason=gate_reason)
        if fire:
            # Order matters: read_delta MUST run before dedup_tool_calls.
            # dedup hashes by (name, arguments) and two Reads of the same
            # path have identical arguments — so dedup collapses the
            # earlier read's args+output to a placeholder, leaving
            # read_delta with only 1 candidate (no repeat detected).
            # read_delta is strictly better for re-reads because it
            # preserves the diff; dedup just throws old content away.
            # After read_delta runs, dedup still fires for non-Read
            # tools (shell, ls, etc.) where dedup is the right tool.
            if CFG.read_delta_enabled:
                body, rd_info = collapse_repeated_reads(
                    body,
                    min_bytes=CFG.read_delta_min_bytes,
                    max_diff_budget=CFG.read_delta_max_diff_budget,
                )
                trace.read_delta_applied = rd_info["applied"]
                trace.read_delta_candidates = rd_info["candidates"]
                trace.read_delta_replacements = rd_info["replacements"]
                trace.read_delta_bytes_saved = rd_info["bytes_saved"]
                trace.read_delta_paths = rd_info["paths"][:20]
                if rd_info["applied"]:
                    _log("read_delta", session=sid,
                         replacements=rd_info["replacements"],
                         bytes_saved=rd_info["bytes_saved"],
                         paths=rd_info["paths"][:20])
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
            _MUTATOR.mark_applied(proj_sid)

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

    # Proactive history truncation (last line of defense against "Codex
    # ran out of room"). Fires only when est_tokens crosses
    # CFG.effective_proactive_compact_threshold() (i.e. the raw threshold
    # minus CFG.proactive_compact_overhead_buffer to calibrate for tinyctx's
    # own ~25-30K of instructions/tools/scout overhead) AND the request is
    # NOT already a codex compaction request. Replaces the middle of
    # body.input with a tinyctx summary item; codex's client-side history
    # is unchanged so the UI still shows every turn. See
    # sanitize.proactive_compact for full rationale.
    #
    # Gated by `proactive_compact_only_on_frontier`: skip when route=local
    # since the local backend has 1M context (cost-free to overspend) and
    # the model benefits from full history. Only frontier needs the
    # 272k-ceiling defense + per-token cost discipline.
    # Effective threshold is derived from frontier.context_window so
    # swapping models (gpt-5.5 ↔ gemini ↔ smaller) auto-adjusts. Falls
    # back to the absolute config value if context_window is unset.
    pc_threshold = effective_proactive_compact_threshold(CFG)
    trace.proactive_compact_threshold_used = pc_threshold
    pc_should_run = (
        pc_threshold > 0
        and not decision.is_compaction
        and (decision.route == "frontier"
             or not CFG.proactive_compact_only_on_frontier)
    )
    if pc_should_run:
        summarizer = (
            _make_local_summarizer(CFG.local)
            if CFG.proactive_compact_use_summarizer
            else None
        )
        # proactive_compact is sync and the summarizer (when provided) does a
        # blocking httpx.Client call to the local model (~3-8s on cache miss,
        # 0s on cache hit). Run in a worker thread so we don't block the
        # asyncio event loop and stall other concurrent requests.
        body, pc_info = await asyncio.to_thread(
            proactive_compact,
            body,
            session_id=proj_sid,  # composite key — see _project_session_key
            est_tokens=decision.est_input_tokens,
            threshold_tokens=pc_threshold,
            recent_keep=CFG.proactive_compact_recent_keep,
            summarizer=summarizer,
        )
        trace.proactive_compact_applied = pc_info["applied"]
        trace.proactive_compact_reason = pc_info["reason"]
        if pc_info["applied"]:
            trace.proactive_compact_items_before = pc_info["items_before"]
            trace.proactive_compact_items_after = pc_info["items_after"]
            trace.proactive_compact_middle_compacted = pc_info.get("middle_items_compacted", 0)
            trace.proactive_compact_synthetic_calls = pc_info.get("synthetic_call_stubs", 0)
            _log("proactive_compact", session=sid, **pc_info)
            # Outgoing-side compaction boundary: tinyctx itself truncated
            # the middle of body.input. The post-truncation conversation
            # is effectively a fresh context for the upstream model, so
            # reset per-conversation injection budget + stuck-loop turn
            # baseline to avoid premature P2 cap on healthy sessions.
            try:
                from . import synthetic_continue as _syn_budget
                from . import stuck_loop as _stuck
                _syn_budget.reset_compaction_state(conv_sid, proj_sid=proj_sid)
                _stuck.reset_compaction_state(conv_sid, proj_sid=proj_sid)
                _log("compaction_state_reset", session=sid,
                     conv_sid=conv_sid, trigger="proactive_compact_applied")
            except Exception as e:  # noqa: BLE001
                _log("compaction_reset_error", session=sid, error=str(e))
    else:
        trace.proactive_compact_applied = False
        trace.proactive_compact_reason = (
            "skipped_local_route"
            if decision.route == "local" and CFG.proactive_compact_only_on_frontier
            else "disabled_or_compaction"
        )

    # Auto-scout: zero-config project context bootstrap. Reads
    # `x-codex-cwd` header (codex sends it), looks up
    # ~/.tinyctx/cache/<repo-hash>/scout.md, prepends it to instructions
    # if present. If absent, schedules a background build (graphify if
    # available, else in-tree fallback scanner) so the NEXT request gets
    # the context. Never blocks. Silent on failure. See auto_scout.py.
    if CFG.auto_scout:
        try:
            cwd_hdr = request.headers.get("x-codex-cwd")
            auto_scout.schedule_bootstrap(
                cwd_hdr,
                install_graphify=CFG.auto_scout_install_graphify,
            )
            scout_md = auto_scout.get_scout(cwd_hdr)
            if scout_md:
                body, was_injected = auto_scout.inject_into_body(body, scout_md)
                trace.scout_injected = was_injected
                if was_injected:
                    trace.scout_chars = len(scout_md)
        except Exception as e:  # noqa: BLE001 — auto-scout must never fail forward
            _log("auto_scout_error", session=sid, error=str(e))

    # Inject the bundled global agent rules (tinyctx/templates/AGENTS.md).
    # Idempotent: if codex.app already loaded ~/.codex/AGENTS.md the title
    # marker is already in instructions and we skip. Otherwise we prepend
    # the bundled version so a fresh `git clone tinyctx` on a new machine
    # gets the same rule baseline without the user manually copying files.
    if CFG.inject_global_agent_rules:
        try:
            from . import agent_rules
            body, was_injected = agent_rules.inject_into_body(body)
            trace.global_agent_rules_injected = was_injected
            if was_injected:
                trace.global_agent_rules_chars = agent_rules.template_chars()
        except Exception as e:  # noqa: BLE001
            _log("agent_rules_inject_error", session=sid, error=str(e))

    # Task Orchestrator: classify task type with local heuristics/planner,
    # suggest Skill/MCP usage, optionally inject a validated Dynamic Skill
    # playbook, and stamp TaskRecord metadata into request trace.
    if CFG.orchestrator_enabled:
        try:
            project_root_hdr = request.headers.get("x-codex-cwd") or ""
            body, task_record = apply_orchestration(
                body,
                cfg=CFG,
                trace=trace,
                session_id=sid,
                project_root=project_root_hdr,
            )
            if CFG.orchestrator_trace_decisions and task_record is not None:
                _log(
                    "orchestrator_applied",
                    session=sid,
                    task_id=task_record.task_id,
                    task_type=task_record.task_type,
                    injected=trace.orchestrator_injected,
                    dynamic_skill_hash=trace.orchestrator_dynamic_skill_hash,
                )
        except Exception as e:  # noqa: BLE001
            _log("orchestrator_error", session=sid, error=str(e))

    # Inject the advisor sub-agent usage hint into instructions BEFORE
    # rewrite_model — the inject function reads body.model to skip the
    # advisor's own sub-thread (model="tinyctx-frontier"). After this
    # call, rewrite_model overwrites body.model with the backend's id.
    #
    # Frontier-only optimization: skip the hint when route=frontier. The
    # hint teaches the cheap local model that it can escalate to a
    # frontier advisor — pointless on the frontier itself. Saves ~1-2k
    # tokens per frontier request.
    if decision.route == "frontier" and CFG.frontier_skip_advisor_hint:
        trace.advisor_hint_skipped = True
    else:
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

    strip_fields = effective_strip_request_fields(backend)
    if strip_fields:
        before_keys = set(body.keys())
        body = strip_unsupported_responses_fields(
            body, drop=strip_fields)
        trace.fields_stripped = sorted(before_keys - set(body.keys()))
    if backend.inject_defaults:
        body = inject_responses_defaults(body, backend.inject_defaults)
        trace.fields_injected = sorted(backend.inject_defaults.keys())
    if backend.cap_fields:
        # FORCE cap fields where the inbound value exceeds the cap.
        # Distinct from inject_defaults: caps OVERRIDE existing values
        # (codex sends max_output_tokens=128000 by default; we cap to
        # 16000 to prevent runaway DeepSeek output).
        before_caps = {p: _get_dotted(body, p) for p in backend.cap_fields}
        body = cap_responses_fields(body, backend.cap_fields)
        after_caps = {p: _get_dotted(body, p) for p in backend.cap_fields}
        capped = [p for p in backend.cap_fields
                  if before_caps[p] != after_caps[p]]
        if capped:
            trace.fields_capped = capped
            _log("cap_fields", session=sid,
                 capped={p: {"from": before_caps[p], "to": after_caps[p]}
                         for p in capped})

    # Frontier-only: LLMLingua-2 pre-escalation compression of bulky
    # tool-result payloads. Default off; gated by `frontier_lingua_enabled`.
    # Targets only function_call_output / tool_result items (skips
    # instructions / tools / user / assistant messages — all cache-critical).
    # Cache-aware: only fires when CacheAwareMutator already opened the gate
    # for THIS request (we reuse the same `fire` decision computed earlier).
    if (decision.route == "frontier"
            and CFG.frontier_lingua_enabled
            and lingua.is_available()):
        lingua_fire = trace.mutation_fired or want_mutation
        if lingua_fire:
            body, lg_info = lingua.compress_for_frontier(
                body,
                ratio=CFG.frontier_lingua_ratio,
                model_name=lingua.DEFAULT_MODEL,
            )
            trace.lingua_applied = lg_info["applied"]
            trace.lingua_items_compressed = lg_info["items_compressed"]
            trace.lingua_chars_before = lg_info["chars_before"]
            trace.lingua_chars_after = lg_info["chars_after"]
            if lg_info["applied"]:
                _log("lingua", session=sid,
                     items=lg_info["items_compressed"],
                     bytes_before=lg_info["chars_before"],
                     bytes_after=lg_info["chars_after"])

    # Frontier-only: trim the tools array to what was actually used in
    # the recent window + an essentials allowlist. Codex sends ~50 tools
    # (~10k tokens) every request; most sessions only call a handful.
    # Local backend skips this — 1M context absorbs the full catalog
    # cheaply and we don't want to surprise the local model.
    if decision.route == "frontier" and CFG.frontier_trim_tools:
        # Codex 0.128 also exposes any tool whose name starts with
        # "mcp__advisor__" — the advisor agent route — keep all of those
        # by name pattern, not just the literal essentials.
        tools_now = body.get("tools") or []
        advisor_tool_names = tuple(
            t.get("name","") for t in tools_now
            if isinstance(t, dict)
            and isinstance(t.get("name"), str)
            and t.get("name","").startswith("mcp__advisor__")
        )
        essentials_extended = tuple(CFG.frontier_tools_essentials) + advisor_tool_names
        body, tt_info = trim_tools_for_frontier(
            body,
            recent_window=CFG.frontier_tools_recent_window,
            essentials=essentials_extended,
        )
        trace.tools_trimmed_applied = tt_info["applied"]
        trace.tools_trimmed_before = tt_info["tools_before"]
        trace.tools_trimmed_after = tt_info["tools_after"]
        trace.tools_trimmed_dropped = tt_info["dropped_names"][:30]  # cap
        if tt_info["applied"]:
            _log("frontier_trim_tools", session=sid, **{
                k: v for k, v in tt_info.items() if k != "kept_names"
            })

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
        return await _compactor_response(body, backend, is_stream, proj_sid,
                                         project_root=request.headers.get("x-codex-cwd"),
                                         trace=trace)

    if backend.wire_api == "responses":
        url = backend.base_url.rstrip("/") + "/responses"
        forward_body = body
        # Older/community LMStudio responses adapters reject role=developer
        # with HTTP 400 ("Unexpected message role."). Rewrite to system on
        # the local route only; frontier (chatgpt.com/backend-api/codex)
        # supports developer natively. The chat-completions branch below
        # is already handled inside normalize_for_chat.
        if (decision.route == "local"
                and CFG.local_role_rewrite_enabled
                and CFG.local_role_rewrite_map):
            forward_body, _n_role_rw = rewrite_input_roles(
                forward_body, rewrite_map=CFG.local_role_rewrite_map)
            if _n_role_rw:
                _log("role_rewrite", session=sid,
                     count=_n_role_rw, map=CFG.local_role_rewrite_map)
    else:
        # Local backend speaks chat-completions. Convert the body and fix URL.
        url = backend.base_url.rstrip("/") + "/chat/completions"
        forward_body = normalize_for_chat(body, strip_tools=backend.strip_tools)

    trace.target_url = url
    trace.is_stream = is_stream

    # Measure what we're actually forwarding (post-transform). Lets us
    # quantify the win from sanitize/proactive_compact and find waste:
    # `est_input_tokens - forwarded_tokens_est` is the savings, and the
    # breakdown shows where the remaining tokens go.
    try:
        fb_serialized = json.dumps(forward_body, ensure_ascii=False, default=str)
        trace.forwarded_bytes = len(fb_serialized.encode("utf-8"))
        trace.forwarded_tokens_est = estimate_tokens(fb_serialized)
        breakdown: dict[str, int] = {}
        # instructions
        inst = forward_body.get("instructions") or ""
        breakdown["instructions"] = estimate_tokens(inst if isinstance(inst, str) else json.dumps(inst, ensure_ascii=False))
        # tools
        tools = forward_body.get("tools") or []
        breakdown["tools"] = estimate_tokens(json.dumps(tools, ensure_ascii=False, default=str))
        # input items
        inp = forward_body.get("input") or forward_body.get("messages") or []
        breakdown["input"] = estimate_tokens(_flatten_text(inp))
        # other = everything else
        other = {k: v for k, v in forward_body.items()
                 if k not in ("instructions", "tools", "input", "messages")}
        breakdown["other"] = estimate_tokens(json.dumps(other, ensure_ascii=False, default=str))
        trace.forwarded_breakdown = breakdown
    except Exception:  # noqa: BLE001 — instrumentation must never fail forward
        pass

    # Final preflight: drop orphan tool-output items whose call_id has no
    # matching call earlier in body.input. Catches residuals from
    # proactive_compact (tool_search_call elided from middle), client
    # reordering, or upstream bugs. chatgpt.com 400s on these.
    if CFG.drop_orphan_tool_outputs:
        forward_body, _orphan_info = drop_orphan_tool_outputs(forward_body)
        if _orphan_info["applied"]:
            _log("orphan_tool_output_dropped", session=sid,
                 dropped=_orphan_info["dropped"],
                 call_ids=_orphan_info["call_ids"])

    # Capture request snapshot for forensics. When an empty response or
    # high-confidence PUNT triggers later, write_forensics_dump will pair
    # this with the response. See tinyctx/forensics.py.
    if CFG.forensics_enabled:
        try:
            from . import forensics as _fx
            _fx.capture_request_snapshot(
                proj_sid=proj_sid,
                request_id=trace.request_id,
                url=url,
                body=forward_body,
                headers=headers,
                request_started_at=time.time(),
            )
        except Exception:  # noqa: BLE001 — forensics snapshot is best-effort
            # Why: snapshot capture failure must not block the request;
            # post-mortem just won't have the request body if it fires.
            pass

    _phase_set(proj_sid, RequestPhase.backend_streaming, trace.request_id)
    return await _forward(url, headers, forward_body, is_stream, proj_sid, decision,
                          translate_tool_calls=backend.translate_tool_calls,
                          chat_to_responses=(backend.wire_api != "responses"),
                          trace=trace,
                          cwd=request.headers.get("x-codex-cwd") or "",
                          conv_sid=conv_sid)


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
    proj_sid = _project_session_key(request, sid)
    conv_sid = _conversation_session_key(proj_sid, body)
    streak = _SESSION_ERROR_STREAK[proj_sid]
    decision = decide(body, CFG, error_streak=streak)
    backend = _select_backend(decision)

    if backend.model:
        body["model"] = backend.model

    _log("route_chat", session=sid, project_session=proj_sid,
         decision=decision.route, reason=decision.reason,
         target=backend.base_url, model=backend.model)

    headers = _forward_headers(request, backend)
    url = backend.base_url.rstrip("/") + "/chat/completions"
    return await _forward(url, headers, body, bool(body.get("stream", False)),
                          proj_sid, decision,
                          translate_tool_calls=backend.translate_tool_calls,
                          conv_sid=conv_sid)


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
                   trace: RequestTrace | None = None,
                   cwd: str = "",
                   conv_sid: str | None = None) -> Any:
    # write=180s gives headroom for multi-megabyte request bodies on
    # slow uplinks. With a 2MB body and a stalled TCP send-window, the
    # old write=60s would fire before any keepalive could rescue the
    # client. read=600s tolerates the upstream taking up to 10 minutes
    # to start streaming (large-context inference cold-start).
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=180.0, pool=10.0)
    # Transport-level retry. httpx retries ONLY on connection-level
    # failures (DNS, TCP connect, ConnectError, ConnectTimeout) and
    # NEVER on HTTP responses (200/4xx/5xx). Exactly the semantics we
    # want — short DeepSeek hiccups (e.g. 21:49:00 connect timeout
    # after which codex.app stalled for 2 hours) are retried silently
    # without exposing the user to error toasts. Genuine server
    # responses pass through untouched. Set retries=0 to disable.
    transport = httpx.AsyncHTTPTransport(retries=int(
        os.environ.get("TINYCTX_UPSTREAM_RETRIES", "1") or 1
    ))
    if is_stream:
        return StreamingResponse(
            _stream_proxy(url, headers, body, sid, decision, timeout,
                          transport=transport,
                          translate_tool_calls=translate_tool_calls,
                          chat_to_responses=chat_to_responses,
                          trace=trace,
                          cwd=cwd,
                          conv_sid=conv_sid),
            media_type="text/event-stream",
        )
    started = time.time()
    # ── Unified retry layer ──────────────────────────────────────────
    # Per user directive "凡是中断了都要加重试". The retry_policy
    # classifier decides retry_same / retry_escalate / propagate for each
    # failure. Bounded by max_total_retries_per_request as a hard cap.
    retry_state = retry_policy.RequestRetryState()
    cur_url, cur_headers, cur_body, cur_decision = url, headers, body, decision
    # Scope force-frontier escalation to the per-conversation key when
    # known so a failure in conv A doesn't bleed into conv B. Falls back
    # to proj_sid (`sid` here) for back-compat with callers that haven't
    # supplied conv_sid yet.
    erg_key = conv_sid if conv_sid else sid
    last_response_payload: Any = None
    last_response_status: int = 0
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        while True:
            retry_state.record_attempt()
            attempt_url = cur_url
            attempt_started = time.time()
            try:
                r = await client.post(attempt_url, headers=cur_headers, json=cur_body)
                http_status: int | None = r.status_code
                conn_error = False
                exc: Exception | None = None
            except httpx.HTTPError as e:
                r = None
                http_status = None
                conn_error = True
                exc = e
            if r is not None and r.status_code < 400:
                _SESSION_ERROR_STREAK[sid] = 0
                break  # success — fall through to payload handling
            # Failure path — classify and decide next action.
            retry_after = 0.0
            if r is not None:
                ra = r.headers.get("retry-after") or r.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else 0.0
                except (TypeError, ValueError):
                    retry_after = 0.0
            action = retry_policy.classify_failure(
                route=cur_decision.route,
                status=http_status,
                is_connection_error=conn_error,
                is_compaction=cur_decision.is_compaction,
                attempts_used=retry_state.attempts_used,
                max_total_retries=CFG.max_total_retries_per_request,
                upstream_retry_count=CFG.upstream_retry_count,
                retry_on_local_4xx_escalate_frontier=CFG.retry_on_local_4xx_escalate_frontier,
                retry_on_frontier_4xx=CFG.retry_on_frontier_4xx,
                retry_after_s=retry_after,
            )
            retry_state.last_action = action
            _SESSION_ERROR_STREAK[sid] += 1
            if action.decision == "propagate":
                if conn_error:
                    _log("upstream_error", session=sid,
                         error=str(exc), url=attempt_url,
                         attempts_used=retry_state.attempts_used,
                         policy_reason=action.reason)
                    if trace is not None:
                        trace.status = 0
                        trace.elapsed_s = round(time.time() - started, 3)
                        trace.emit(CFG.log_dir)
                    # Set force_next_to_frontier if classifier asked for it
                    if action.escalate_flag_reason:
                        try:
                            from . import empty_response_guard as _erg
                            _erg.force_next_to_frontier(
                                erg_key, action.escalate_flag_reason)
                        except Exception:  # noqa: BLE001 — escalation hint is next-turn optimization
                            # Why: missing this flag means next turn
                            # routes to default backend (no correctness
                            # loss; this turn's response already returned).
                            pass
                    return JSONResponse(
                        {"error": {"message": str(exc),
                                   "type": "tinyctx_upstream"}},
                        status_code=502)
                assert r is not None  # mypy/type narrowing
                if action.escalate_flag_reason:
                    try:
                        from . import empty_response_guard as _erg
                        _erg.force_next_to_frontier(
                            erg_key, action.escalate_flag_reason)
                    except Exception:  # noqa: BLE001 — escalation hint is next-turn optimization
                        pass
                if trace is not None:
                    trace.status = r.status_code
                    trace.elapsed_s = round(time.time() - started, 3)
                    trace.emit(CFG.log_dir)
                return JSONResponse(content=_safe_json(r),
                                    status_code=r.status_code)
            # Retry path — log and re-attempt.
            new_url = attempt_url
            new_headers = cur_headers
            new_decision = cur_decision
            if action.decision == "retry_escalate":
                new_url, new_headers, _b, new_decision, _backend = (
                    _build_frontier_retry_target(None, cur_body, action.reason))
                # Preserve codex routing headers (openai-beta,
                # x-codex-session-id, etc.) from the original request,
                # but REBUILD Authorization for the frontier backend.
                # Without this, a local-backend bearer (e.g. LMStudio's
                # sk-* key) leaks to chatgpt.com and triggers a 401 that
                # closes the request at bytes_out=0 — codex shows
                # "task interrupted" right after retry_attempted. See
                # rq_f372c3c35c47444db89e for the live trace.
                merged = dict(cur_headers)
                merged["Content-Type"] = "application/json"
                merged.setdefault("Accept", "text/event-stream")
                fb_key = _resolve_api_key(CFG.frontier, None)
                if fb_key:
                    merged["Authorization"] = (
                        fb_key if fb_key.lower().startswith(
                            ("bearer ", "basic ")) else f"Bearer {fb_key}")
                else:
                    merged.pop("Authorization", None)
                new_headers = merged
                # Set force_next_to_frontier so next turn also frontier
                if action.escalate_flag_reason:
                    try:
                        from . import empty_response_guard as _erg
                        _erg.force_next_to_frontier(
                            erg_key, action.escalate_flag_reason)
                    except Exception:  # noqa: BLE001 — escalation hint is next-turn optimization
                        pass
                retry_state.record_escalation()
            _log("retry_attempted",
                 session=sid,
                 attempt_number=retry_state.attempts_used,
                 original_status=http_status,
                 retry_target=action.decision,
                 original_url=attempt_url,
                 new_url=new_url,
                 reason=action.reason,
                 request_id=trace.request_id if trace is not None else "",
                 conn_error=conn_error,
                 elapsed_s=round(time.time() - attempt_started, 3))
            # Reset the stall watchdog's last-event timestamp on the
            # retry boundary so the new attempt gets a fresh threshold
            # window. Without this, a 14s LMStudio 400 followed by a
            # silent frontier wait would let stall_watchdog's countdown
            # use the OLD `mark_event` from the 400 arrival — and codex's
            # client-side idle timeout would fire first.
            try:
                _stall.mark_event(sid, conv_sid=conv_sid)
            except Exception:  # noqa: BLE001 — watchdog mark is advisory
                # Why: stall_watchdog mark-event resets the timer for the
                # new attempt. If it fails the retry still proceeds — the
                # OLD timer just stays in effect for one more cycle.
                pass
            if action.backoff_s > 0:
                try:
                    await asyncio.sleep(action.backoff_s)
                except Exception:  # noqa: BLE001 — sleep interruption is non-fatal
                    # Why: sleep can raise on cancellation; the retry
                    # loop continues regardless.
                    pass
            cur_url, cur_headers, cur_decision = new_url, new_headers, new_decision
            # loop continues — try again
        # success branch
        assert r is not None
        payload = _safe_json(r)
        translated_calls = 0
        # Non-streaming chat→responses translation. The streaming path in
        # _stream_proxy uses ChatToResponsesTranslator; this is the
        # equivalent for one-shot JSON responses (Vercel AI SDK's
        # generateObject hits the proxy with stream=false and expects a
        # Responses-API-shaped body back, even when the upstream is a
        # chat-completions backend like DeepSeek).
        if chat_to_responses and isinstance(payload, dict) and "choices" in payload:
            payload = _chat_to_responses_payload(payload)
        if translate_tool_calls and isinstance(payload, dict):
            valid_names = (_extract_valid_tool_names(body)
                           if CFG.unknown_tool_call_protection else None)
            new_payload = rebuild_response(payload, valid_tool_names=valid_names)
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
                        proj_sid: str, decision: Decision,
                        timeout: httpx.Timeout,
                        *, transport: httpx.AsyncBaseTransport | None = None,
                        translate_tool_calls: bool = False,
                        chat_to_responses: bool = False,
                        trace: RequestTrace | None = None,
                        cwd: str = "",
                        conv_sid: str | None = None) -> AsyncIterator[bytes]:
    # conv_sid is the per-conversation scope key for synthetic_continue
    # injection budget and empty_response_guard flagging. Falls back to
    # proj_sid when not provided by callers that haven't been migrated —
    # preserves old project-scoped behavior.
    erg_key = conv_sid if conv_sid is not None else proj_sid
    # Hoisted: every retry/escalation/phase-set log site below wants
    # request_id, and `trace` is the same object throughout the relay.
    request_id = trace.request_id if trace is not None else ""
    started = time.time()
    bytes_out = 0
    status = 200
    translator: StreamTranslator | ChatToResponsesTranslator | None
    valid_names = (_extract_valid_tool_names(body)
                   if CFG.unknown_tool_call_protection else None)
    auto_user_input_enabled = os.environ.get("TINYCTX_AUTO_USER_INPUT", "1") != "0"
    if chat_to_responses:
        translator = ChatToResponsesTranslator(valid_tool_names=valid_names)
    elif translate_tool_calls or auto_user_input_enabled:
        translator = StreamTranslator(valid_tool_names=valid_names)
    else:
        translator = None
    # codex.app's SSE parser raises "stream closed before response.completed"
    # if we close the stream after `event: error` without emitting a
    # synthetic terminator. We always end with response.completed so codex
    # observes a structurally valid SSE close — even when the upstream
    # itself errored. The completion has status=incomplete so codex still
    # surfaces the failure correctly.
    def _terminator_event(message: str, *, status_label: str = "incomplete") -> bytes:
        return _relay.build_terminator_event(message,
                                             body.get("model") if isinstance(body, dict) else None,
                                             status_label=status_label)

    # SSE keepalive: when upstream goes silent during streaming, emit
    # `: tinyctx keepalive` SSE comment lines every
    # CFG.stream_keepalive_interval_s seconds so codex.app's stream
    # parser sees ongoing bytes (preventing idle-timeout disconnects)
    # and any TCP middlebox keeps the connection alive. SSE comments
    # (lines starting with `:`) are ignored by spec-compliant clients.
    keepalive_interval = CFG.stream_keepalive_interval_s
    keepalives_emitted = 0

    # Soft-completion sniffer state. Reset the per-session output buffer
    # at start of this stream so we match within THIS turn only. The
    # flag (in soft_completion module) survives across streams until
    # the gate injection consumes it on the next request.
    if CFG.soft_completion_gate_enabled:
        try:
            from . import soft_completion
            soft_completion.reset_stream(proj_sid)
        except Exception:  # noqa: BLE001 — instrumentation must never fail forward
            pass

    # Stream-rewrite state. If enabled, intercept the first
    # `response.completed` SSE event in the OUTGOING stream, hold it
    # while the soft-completion classifier runs, and conditionally
    # inject a synthetic function_call to advisor BEFORE flushing the
    # held event. See tinyctx/stream_rewrite.py for rationale + risk.
    held_completion_buf = bytearray()
    holding_completion = [False]  # list-wrapped for nonlocal-like closure
    rewrite_enabled = CFG.soft_completion_stream_rewrite_enabled

    # Outgoing-bytes capture for forensics. accumulate_chunk only stores
    # raw upstream bytes; the bytes we YIELD to client (post-injection)
    # are different — and that's what codex actually parses. Capture
    # last 32KB of yielded bytes so forensics shows what codex saw.
    outgoing_capture = bytearray()
    OUTGOING_MAX = 32 * 1024

    def _capture_outgoing(b: bytes) -> bytes:
        """Append b to the outgoing capture (capped tail) and return b
        so the caller can `yield _capture_outgoing(...)` inline."""
        if not CFG.forensics_enabled:
            return b
        if b:
            outgoing_capture.extend(b)
            if len(outgoing_capture) > OUTGOING_MAX:
                # Trim from the front to keep last OUTGOING_MAX bytes
                del outgoing_capture[:len(outgoing_capture) - OUTGOING_MAX]
        return b

    def _intercept_completed(out_bytes: bytes) -> bytes:
        """Return bytes safe to yield to client. If the response.completed
        marker is detected, hold the marker (and any subsequent bytes
        of this and later chunks) in `held_completion_buf` so we can
        run the classifier first and conditionally inject synthetic
        events before flushing it. No-op when rewrite is disabled."""
        if not rewrite_enabled:
            return out_bytes
        if holding_completion[0]:
            # Already holding — append everything else, yield nothing
            held_completion_buf.extend(out_bytes)
            return b""
        try:
            from . import stream_rewrite as _sr
        except Exception:  # noqa: BLE001
            return out_bytes
        if not _sr.looks_like_response_completed(out_bytes):
            return out_bytes
        pre, completed_part = _sr.split_at_completed(out_bytes)
        holding_completion[0] = True
        held_completion_buf.extend(completed_part)
        return pre  # bytes BEFORE the marker — yield those

    upstream_failed = False
    upstream_failure_msg = ""
    try:
        if keepalive_interval > 0:
            # Producer/consumer with keepalive across BOTH phases:
            #  Phase 1 — request upload + wait for upstream response headers
            #  Phase 2 — stream response body
            # Codex.app's stream parser disconnects after ~60s of zero
            # bytes from the proxy. With a 2MB request body and slow
            # upstream (e.g. DeepSeek loading a 500K-token context),
            # phase 1 alone can exceed 60s — and the OLD code's keepalive
            # only fired in phase 2, so the client gave up before any
            # bytes flowed. Now the producer task runs the ENTIRE upstream
            # interaction (open + upload + read headers + stream body),
            # and the main coroutine yields keepalives whenever the queue
            # is silent for keepalive_interval seconds.
            #
            # P6: producer + consumer + supervisor live in stream_relay.
            # Status-error forensics is the only branch we still own here
            # because it needs `url` and the forensics writer.
            def _on_status_error(status_code: int, err_body: str) -> None:
                _SESSION_ERROR_STREAK[proj_sid] += 1
                _log("upstream_error", session=proj_sid,
                     status=status_code, url=url, body=err_body[:2000])
                if CFG.forensics_enabled and CFG.forensics_capture_errors:
                    try:
                        from . import forensics as _fx
                        forensics_dir = CFG.log_dir.parent / "forensics"
                        _fx.write_forensics_dump(
                            forensics_dir, proj_sid,
                            trigger=f"upstream_{status_code}",
                            response_buffer=err_body or "",
                            extra={"status": status_code, "url": url},
                            max_dumps=CFG.forensics_max_dumps,
                        )
                    except Exception:  # noqa: BLE001 — forensics dump is best-effort
                        # Why: post-mortem dump must not interfere with
                        # returning the upstream error to the client.
                        pass

            chunk_q: asyncio.Queue = asyncio.Queue()
            producer = _relay.StreamProducer(
                url=url, headers=headers, body=body,
                proj_sid=proj_sid, conv_sid=conv_sid,
                decision=decision, timeout=timeout, transport=transport,
                erg_key=erg_key, request_id=request_id,
                cfg=CFG, log=_log,
                build_frontier_retry_target=_build_frontier_retry_target,
                resolve_api_key=_resolve_api_key,
            )
            consumer = _relay.StreamConsumer(
                chunk_q=chunk_q, translator=translator,
                proj_sid=proj_sid, conv_sid=conv_sid,
                keepalive_interval=keepalive_interval,
                capture_outgoing=_capture_outgoing,
                intercept_completed=_intercept_completed,
                cfg=CFG, log=_log, url=url,
                on_status_error=_on_status_error,
            )
            supervisor = _relay.StallSupervisor(
                proj_sid, enabled=CFG.stall_watchdog_enabled)
            # Reset _SESSION_ERROR_STREAK on the SUCCESS-status branch;
            # on_status_error increments it on the failure branch.
            # The reset has to fire on the STATUS=(200, None) path that
            # stream_relay handles internally, so we hook it by reading
            # consumer.status / upstream_failed after the loop.
            async for out in _relay.relay_stream(
                    chunk_q=chunk_q, producer=producer, consumer=consumer,
                    supervisor=supervisor,
                    keepalive_interval=keepalive_interval):
                yield out
            # Mirror consumer state back into outer locals so the post-
            # stream / finally blocks below see the same values they
            # did when this logic lived inline.
            bytes_out = consumer.bytes_out
            status = consumer.status
            keepalives_emitted = consumer.keepalives_emitted
            upstream_failed = consumer.upstream_failed
            upstream_failure_msg = consumer.upstream_failure_msg
            if not upstream_failed and status == 200:
                _SESSION_ERROR_STREAK[proj_sid] = 0
            # ─── stream-rewrite synthesis ──────────────────────────
            # We held back the response.completed event. Decide whether
            # to inject a synthetic advisor function_call in front of
            # it, then flush. Runs only on the keepalive path because
            # the held-completion buffer is fed by _intercept_completed
            # which only sees bytes via the queue-based consumer.
            if rewrite_enabled and holding_completion[0] and not upstream_failed:
                try:
                    from . import soft_completion as _sc
                    from . import stream_rewrite as _sr
                    api_key = (os.environ.get(CFG.local.api_key_env)
                               if CFG.local.api_key_env else None)
                    buffer_snapshot = _sc._OUTPUT_BUFFER.get(proj_sid, "")
                    body_input = (body.get("input")
                                   if isinstance(body, dict) else None)
                    user_goal = _sc.extract_user_goal(body_input)
                    tracker = _sc.extract_progress_tracker(body_input)
                    tool_summary = _sc.extract_tool_summary(body_input)
                    # Synchronous classification — we're at stream
                    # end, no other bytes flowing. Bounded by
                    # self_classify_timeout_s (default 30s).
                    diag = await _sc.classify_at_stream_end_diag(
                        proj_sid,
                        local_base_url=CFG.local.base_url,
                        local_model=CFG.local.model,
                        api_key=api_key,
                        timeout_s=CFG.self_classify_timeout_s,
                        threshold=CFG.self_classify_threshold,
                        raw_buffer=buffer_snapshot,
                        user_goal=user_goal,
                        progress_tracker=tracker,
                        tool_summary=tool_summary,
                        force_frontier_threshold=(
                            CFG.soft_completion_auto_force_frontier_threshold
                            if CFG.soft_completion_auto_force_frontier_enabled
                            else 1.01),
                        short_text_threshold=CFG.soft_completion_short_text_threshold,
                        stop_text_threshold=CFG.soft_completion_stop_text_threshold,
                    )
                    if (diag.result is not None
                            and diag.result.soft_punt
                            and diag.result.p >= CFG.soft_completion_stream_rewrite_threshold):
                        text_excerpt = _sc._extract_text_from_buffer(
                            buffer_snapshot)
                        task_body = _sr.build_task_body(
                            text_excerpt,
                            diag.result.reason,
                            diag.result.p)
                        # Multi-strategy synthetic continue: rotate
                        # through codex builtins (shell / local_shell /
                        # update_plan) until codex actually dispatches
                        # one. spawn_agent was tried first but binary
                        # analysis confirmed codex silently drops it.
                        from . import synthetic_continue as _syn
                        inj_events, strategy = _syn.build_continue_injection(
                            erg_key,
                            max_injections=CFG.max_continue_injections_per_session,
                        )
                        if strategy["label"] == "budget_exhausted":
                            from . import empty_response_guard as _erg_budget
                            _erg_budget.force_next_to_frontier(
                                erg_key, "injection_budget_exhausted")
                            _log("soft_completion_stream_rewrite_budget_exhausted",
                                 session=proj_sid,
                                 p=diag.result.p,
                                 injection_count=_syn.injection_count(erg_key),
                                 max_injections=CFG.max_continue_injections_per_session)
                        else:
                            for evt in inj_events:
                                yield _capture_outgoing(evt)
                            _log("soft_completion_stream_rewrite_injected",
                                 session=proj_sid,
                                 p=diag.result.p,
                                 reason=diag.result.reason,
                                 strategy=strategy["label"],
                                 tool_name=strategy["tool_name"],
                                 task_chars=len(task_body),
                                 injection_count=_syn.injection_count(erg_key))
                        if trace is not None:
                            trace.soft_completion_gate_injected = True
                            trace.soft_completion_gate_pattern = (
                                f"stream-rewrite: {diag.result.reason}")[:80]
                        # Forensics: capture the PUNT that triggered
                        # this stream-rewrite. Same write_punt_forensics
                        # the bg_classify path uses.
                        if (CFG.forensics_enabled
                                and CFG.forensics_capture_punts
                                and diag.result.p >= CFG.forensics_punt_threshold):
                            try:
                                from . import forensics as _fx
                                forensics_dir = CFG.log_dir.parent / "forensics"
                                # Custom dump that ALSO includes the
                                # outgoing capture (what codex
                                # actually saw, post-injection).
                                raw = _sc._OUTPUT_BUFFER.get(proj_sid, "") or ""
                                fpath = _fx.write_forensics_dump(
                                    forensics_dir, proj_sid,
                                    trigger="punt_via_stream_rewrite",
                                    response_buffer=raw,
                                    classifier_verdict={
                                        "soft_punt": diag.result.soft_punt,
                                        "p": diag.result.p,
                                        "reason": diag.result.reason,
                                        "extracted_text_chars": diag.extracted_text_chars,
                                        "raw_buffer_chars": diag.raw_buffer_chars,
                                        "finish_reason": diag.finish_reason,
                                    },
                                    extra={
                                        "strategy": strategy["label"],
                                        "tool_name": strategy["tool_name"],
                                        "outgoing_to_codex_chars": len(outgoing_capture),
                                        "outgoing_to_codex_tail": (
                                            bytes(outgoing_capture[-4000:]).decode("utf-8", "replace")
                                            if outgoing_capture else ""),
                                    },
                                    max_dumps=CFG.forensics_max_dumps,
                                )
                                if fpath:
                                    _log("forensics_dump_written",
                                         session=proj_sid,
                                         trigger="punt_via_stream_rewrite",
                                         path=str(fpath))
                            except Exception:  # noqa: BLE001 — forensics dump is best-effort
                                pass
                    else:
                        _log("soft_completion_stream_rewrite_skipped",
                             session=proj_sid,
                             reason=("not_punt" if diag.result is None
                                     else f"p={diag.result.p:.2f}"
                                          f"<{CFG.soft_completion_stream_rewrite_threshold}"))
                except Exception as e:  # noqa: BLE001
                    _log("soft_completion_stream_rewrite_error",
                         session=proj_sid, error=str(e))
                # Always flush the held response.completed at the end
                if held_completion_buf:
                    yield _capture_outgoing(bytes(held_completion_buf))
        else:
            # keepalive disabled — original simple loop, no extra task overhead
            async with httpx.AsyncClient(
                    timeout=timeout, transport=transport) as client:
                async with client.stream(
                        "POST", url, headers=headers, json=body) as r:
                    status = r.status_code
                    if r.status_code >= 400:
                        _SESSION_ERROR_STREAK[proj_sid] += 1
                        text = (await r.aread()).decode("utf-8", "replace")
                        _log("upstream_error", session=proj_sid,
                             status=r.status_code, url=url, body=text[:2000])
                        yield (
                            f"event: error\ndata: "
                            f"{json.dumps({'status': r.status_code, 'body': text[:2000]})}"
                            f"\n\n").encode()
                        upstream_failed = True
                        upstream_failure_msg = (
                            f"upstream {r.status_code}: {text[:200]}")
                    else:
                        _SESSION_ERROR_STREAK[proj_sid] = 0
                        async for chunk in r.aiter_raw():
                            bytes_out += len(chunk)
                            if CFG.stall_watchdog_enabled:
                                _stall.mark_event(proj_sid, conv_sid=conv_sid)
                            # Soft-completion accumulator (no-keepalive path).
                            # LLM classifier runs at stream end.
                            if CFG.soft_completion_gate_enabled:
                                try:
                                    from . import soft_completion as _sc
                                    _sc.accumulate_chunk(proj_sid, chunk)
                                except Exception:  # noqa: BLE001 — accumulator is observability
                                    # Why: soft-completion accumulator
                                    # failure must not drop the chunk
                                    # for the downstream client.
                                    pass
                            # NOTE: stream rewrite intercept is wired in
                            # the keepalive path above; this no-keepalive
                            # branch (CFG.stream_keepalive_interval_s == 0)
                            # passes chunks through unmodified. Stream
                            # rewrite without keepalive would need its
                            # own intercept here too — left as a TODO
                            # since keepalive is on by default in tinyctx.
                            if translator is None:
                                yield chunk
                            else:
                                for out in translator.feed(chunk):
                                    yield out
                        if translator is not None:
                            for out in translator.flush():
                                yield out
    except _stall.StallCancelledError as e:
        # Watchdog cancelled the in-flight relay; emit terminator and
        # set force-frontier flag for the follow-up turn.
        # P7: dispatch into RelayErrorTerminator for the bookkeeping +
        # forensics; we still own the `yield` because it's the
        # generator's response to its parser.
        _term = _post.RelayErrorTerminator(
            cfg=CFG, log=_log,
            session_error_streak=_SESSION_ERROR_STREAK)
        res = _term.on_stall_cancelled(
            e, proj_sid=proj_sid, conv_sid=conv_sid,
            bytes_out=bytes_out, started=started, url=url)
        status = res.status
        yield res.error_event
        upstream_failed = res.upstream_failed
        upstream_failure_msg = res.upstream_failure_msg
    except httpx.HTTPError as e:
        _term = _post.RelayErrorTerminator(
            cfg=CFG, log=_log,
            session_error_streak=_SESSION_ERROR_STREAK)
        res = _term.on_http_error(
            e, proj_sid=proj_sid, conv_sid=conv_sid,
            bytes_out=bytes_out, started=started, url=url,
            erg_key=erg_key, request_id=request_id)
        status = res.status
        yield res.error_event
        upstream_failed = res.upstream_failed
        upstream_failure_msg = res.upstream_failure_msg
    # Always emit a structurally valid response.completed terminator so
    # codex.app's SSE parser doesn't raise "stream closed before
    # response.completed". For the success path we hope the upstream
    # already emitted its own response.completed; emitting a second one
    # is fine (codex's parser uses last-wins).
    try:
        if upstream_failed:
            yield _terminator_event(upstream_failure_msg or "tinyctx upstream failure")
    except Exception:  # noqa: BLE001 — never fail the finally
        pass
    finally:
        if CFG.stall_watchdog_enabled:
            try:
                _stall.clear(proj_sid)
            except Exception:  # noqa: BLE001 — watchdog cleanup in finally
                # Why: clear-in-finally must never raise. Stale entries
                # in stall_watchdog are auto-aged-out.
                pass
        elapsed = round(time.time() - started, 3)
        _log("stream_done", session=proj_sid, route=decision.route, bytes=bytes_out,
             translated=bool(translator),
             elapsed_s=elapsed,
             keepalives=keepalives_emitted)
        # P7: post-stream analysis (classifier spawn + empty-response
        # guard + forensics) all live in post_stream.PostStreamAnalyzer.
        # The analyzer preserves pre-P7 timing exactly: the LLM
        # classifier is spawned as a fire-and-forget bg task (never
        # blocks return); the empty-response guard runs synchronously.
        try:
            _ps_analyzer = _post.PostStreamAnalyzer(cfg=CFG, log=_log)
            await _ps_analyzer.analyze(_post.PostStreamContext(
                proj_sid=proj_sid,
                conv_sid=conv_sid,
                erg_key=erg_key,
                request_id=request_id,
                body=body if isinstance(body, dict) else {},
                cwd=cwd,
                bytes_out=bytes_out,
                status=status,
                upstream_failed=upstream_failed,
                keepalives_emitted=keepalives_emitted,
                elapsed=elapsed,
                started=started,
                url=url,
            ))
        except Exception as _e:  # noqa: BLE001
            _log("post_stream_analyze_error",
                 session=proj_sid, error=str(_e))
        if trace is not None:
            trace.status = status
            trace.bytes_out = bytes_out
            trace.translated = bool(translator)
            trace.translated_calls = (translator._emitted_calls
                                       if translator is not None else 0)
            trace.elapsed_s = elapsed
            trace.keepalives_emitted = keepalives_emitted
            trace.emit(CFG.log_dir)
        _phase_set(proj_sid,
                   RequestPhase.stalled if upstream_failed else RequestPhase.done,
                   request_id)


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
        forward_body = body if backend.wire_api == "responses" else normalize_for_chat(body, strip_tools=backend.strip_tools)
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=180.0, pool=10.0)
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
