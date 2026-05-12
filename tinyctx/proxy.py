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
from .config import BackendCfg, Config, effective_proactive_compact_threshold, load_config
from .continuity import save_compaction
from . import historian
from . import lingua
from .read_delta import collapse_repeated_reads
from .router import Decision, decide
from .sanitize import (
    CacheAwareMutator,
    cap_responses_fields,
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
from .request_phase import RequestPhase, set_phase as _phase_set
from . import retry_policy
from . import stall_watchdog as _stall
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
    except Exception:
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
    retry_escalate) to switch from local to frontier mid-request. We
    reuse the same `body` (frontier accepts every codex-emitted role/
    field/tool-type natively, so no transform needed) but rebuild the
    URL + auth headers from the frontier backend config.

    `request` may be None when called from a context that doesn't have
    the FastAPI Request handy (e.g. retry inside a stream producer).
    In that case the caller passes a pre-built headers dict separately;
    we still produce frontier-shaped url + decision so the dispatch
    loop can swap them in.
    """
    backend = CFG.frontier
    url = backend.base_url.rstrip("/") + "/responses"
    headers = (_forward_headers(request, backend) if request is not None
               else {"Content-Type": "application/json",
                     "Accept": "text/event-stream"})
    decision = Decision(
        "frontier",
        f"retry_escalate: {reason}",
        is_compaction=False,
    )
    return url, headers, body, decision, backend


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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            cancelled = False
        try:
            from . import empty_response_guard as _erg
            _erg.force_next_to_frontier(escalate_key, "mid_stream_stall")
            _phase_set(proj_sid, RequestPhase.escalated_to_frontier, "")
        except Exception:  # noqa: BLE001
            pass
        trigger_label = "stall_cancelled" if cancelled else "stall_kill"
        elapsed: float | None = None
        try:
            elapsed = _stall.seconds_since_event(proj_sid)
        except Exception:  # noqa: BLE001
            elapsed = None
        try:
            _log(trigger_label, session=proj_sid, conv_sid=conv_sid,
                 escalate_key=escalate_key,
                 threshold_s=CFG.stall_threshold_s,
                 elapsed_silent_s=elapsed,
                 task_cancelled=cancelled)
        except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
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
    decision = decide(body, CFG, error_streak=streak)
    backend = _select_backend(decision)
    _phase_set(proj_sid, RequestPhase.routing, trace.request_id)

    # Tool-call frequency tracking — mine body.input for function_call
    # items (deduped by call_id) so the dashboard can show which MCP
    # servers / built-in tools are actually being used. Fire-and-forget
    # — never raises. See tinyctx/tool_metrics.py.
    try:
        from . import tool_metrics as _tm
        _tm.record_from_body(body)
    except Exception:  # noqa: BLE001
        pass

    # Plan persistence: save any update_plan / TodoWrite tracker
    # currently in body.input to disk (per-cwd), and inject the
    # persisted plan when this is a fresh thread (turn_count==0)
    # — bridges context across codex thread boundaries.
    if CFG.plan_persistence_enabled:
        try:
            from . import plan_persistence as _pp
            cwd_hdr = request.headers.get("x-codex-cwd") or ""
            state_dir = CFG.log_dir.parent / "state"
            # Save current plan if any
            plan_now = _pp.extract_plan_text(body)
            if plan_now:
                saved = _pp.save_plan(state_dir, cwd_hdr, plan_now,
                                       session_id=sid,
                                       turn_count=decision.turn_count)
                if saved:
                    _log("plan_persistence_saved", session=sid,
                         cwd=cwd_hdr[:120], turn_count=decision.turn_count)
            # Inject persisted plan on fresh thread
            if decision.turn_count == 0:
                pdata = _pp.load_plan(state_dir, cwd_hdr,
                                       ttl_s=CFG.plan_persistence_ttl_s)
                if pdata is not None:
                    body, was_inj = _pp.inject_plan(body, pdata)
                    if was_inj:
                        _log("plan_persistence_injected",
                             session=sid, cwd=cwd_hdr[:120],
                             prev_turn_count=pdata.get("turn_count_at_save"),
                             updated=pdata.get("updated_at_iso"))
        except Exception as e:  # noqa: BLE001
            _log("plan_persistence_error", session=sid, error=str(e))

    # Empty-response guard: if the previous turn for this session
    # returned an effectively empty response (e.g., DeepSeek silently
    # degraded under long context), force this turn to frontier so
    # the user gets a real answer. One-shot per detection — flag is
    # consumed here. See tinyctx/empty_response_guard.py.
    if CFG.empty_response_guard_enabled:
        try:
            from . import empty_response_guard as _erg
            # Try conv-scoped key first; fall back to proj-scoped so flags
            # set by mid-stream stall or upstream-error escalation (which
            # don't have body access) still trigger frontier escalation.
            force_info = _erg.consume_force_frontier(conv_sid)
            if force_info is None and conv_sid != proj_sid:
                force_info = _erg.consume_force_frontier(proj_sid)
            elif force_info is not None and conv_sid != proj_sid:
                # Consuming any flag for this proj_sid clears ALL flags
                # under it (conv_sid-keyed + proj_sid-keyed). Without this,
                # a dangling proj_sid flag (e.g. exec_resume set it later
                # for the same project) would be consumed by a DIFFERENT
                # conversation's next request via the fallback above —
                # force-routing it for no reason.
                _erg.reset_state(proj_sid)
            if force_info is not None:
                decision = Decision(
                    "frontier",
                    f"empty-response guard: {force_info.get('reason', '?')[:80]}",
                    is_compaction=decision.is_compaction,
                    est_input_tokens=decision.est_input_tokens,
                    turn_count=decision.turn_count,
                )
                backend = CFG.frontier
                _phase_set(proj_sid, RequestPhase.empty_guarded, trace.request_id)
                _log("empty_response_guard_forced_frontier",
                     session=sid, proj_sid=proj_sid,
                     prev_completion_tokens=force_info.get("completion_tokens"),
                     prev_finish_reason=force_info.get("finish_reason"))
        except Exception as e:  # noqa: BLE001 — guard must never block
            _log("empty_response_guard_error", session=sid, error=str(e))

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

    # Model-driven escalation (Anthropic Advisor Strategy alignment):
    # ask the LOCAL model itself whether this turn deserves the advisor.
    # Only runs when:
    #   - feature is enabled
    #   - we'd otherwise route to local (don't second-guess explicit
    #     frontier escalation, force_route, error_streak, compaction)
    #   - this isn't a force_route / explicit-model override
    # Failures are silent — the classifier returns None and the
    # original heuristic decision stands. See tinyctx/self_classify.py.
    if (CFG.self_classify_enabled
            and decision.route == "local"
            and not decision.is_compaction
            and not trace.forced_by_client_model):
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
                    decision = Decision(
                        "frontier",
                        f"self-classify p={sc.p:.2f}: {sc.reason}",
                        is_compaction=decision.is_compaction,
                        est_input_tokens=decision.est_input_tokens,
                        turn_count=decision.turn_count,
                    )
                    backend = CFG.frontier
                    trace.self_classify_overrode = True
                    _phase_set(proj_sid, RequestPhase.escalated_to_frontier,
                               trace.request_id)
        except Exception as e:  # noqa: BLE001 — classifier must never fail forward
            _log("self_classify_error", session=sid, error=str(e))

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

    # Stuck-loop watchdog: when turn_count climbs past the trigger
    # without a recent advisor call, append a `<system-reminder>` to
    # the input tail asking the agent to either consult advisor or
    # surface its blocker. See tinyctx/stuck_loop.py for rationale.
    # Only fires for non-compaction main turns (advisor sub-threads
    # carry forced_by_client_model and don't need their own watchdog).
    if (CFG.stuck_loop_watchdog_enabled
            and not decision.is_compaction
            and not trace.forced_by_client_model):
        try:
            from . import stuck_loop
            # Key the reminder gate by conv_sid so a fresh conversation
            # (codex turn_count resets to 0) doesn't get blocked by a
            # stale `_LAST_REMINDER_TURN` from the previous conversation
            # (e.g. 175 → `0 - 175 = -175 < 50` would skip forever).
            # Advisor grace stays project-scoped so advisor activity in
            # any sub-thread quiets nudges across all conversations.
            body, was_injected = stuck_loop.maybe_inject_stuck_reminder(
                body, conv_sid, decision.turn_count,
                turn_trigger=CFG.stuck_loop_turn_trigger,
                turn_gap=CFG.stuck_loop_turn_gap,
                advisor_grace_s=CFG.stuck_loop_advisor_grace_s,
                advisor_scope_sid=proj_sid,
            )
            if was_injected:
                trace.stuck_reminder_injected = True
                trace.stuck_turn_count_at_inject = decision.turn_count
                _phase_set(proj_sid, RequestPhase.injecting, trace.request_id)
                _log("stuck_reminder_injected", session=sid,
                     proj_sid=proj_sid, turn_count=decision.turn_count)
        except Exception as e:  # noqa: BLE001 — watchdog must never block
            _log("stuck_loop_error", session=sid, error=str(e))

    # P2: injection-budget exhaustion reminder. When synthetic_continue
    # tripped its budget on the previous turn, append a one-shot
    # `<system-reminder>` warning the agent that tinyctx auto-continued
    # N times and may have been wrong. Keyed by conv_sid so a fresh
    # conversation in the same project starts with a clean budget
    # instead of inheriting the previous thread's exhausted counter.
    # The flag is consumed on use so this never repeats next turn.
    if not decision.is_compaction and not trace.forced_by_client_model:
        try:
            from . import synthetic_continue as _syn_budget
            inj_count = _syn_budget.injection_count(conv_sid)
            if (inj_count >= CFG.max_continue_injections_per_session
                    and inj_count > 0):
                body, was_budget_inj = (
                    _syn_budget.maybe_inject_budget_reminder(
                        body, conv_sid, inj_count))
                if was_budget_inj:
                    _log("budget_exhausted_reminder_injected",
                         session=sid, proj_sid=proj_sid,
                         injection_count=inj_count)
        except Exception as e:  # noqa: BLE001
            _log("budget_reminder_error", session=sid, error=str(e))

    # Soft-completion gate: if the previous turn ended with a "soft
    # punt to user" pattern (matched in the streaming sniffer), inject
    # an advisor-vet reminder requiring the agent to route any user-
    # facing question through advisor first. Per user directive: "如果
    # 非要提问，走 advisor 进行回答". See tinyctx/soft_completion.py.
    if (CFG.soft_completion_gate_enabled
            and not decision.is_compaction
            and not trace.forced_by_client_model):
        try:
            from . import soft_completion
            body, was_gated, gate_pattern = (
                soft_completion.maybe_inject_soft_completion_gate(
                    body, proj_sid))
            if was_gated:
                trace.soft_completion_gate_injected = True
                trace.soft_completion_gate_pattern = gate_pattern
                _log("soft_completion_gate_injected", session=sid,
                     proj_sid=proj_sid, pattern=gate_pattern)
        except Exception as e:  # noqa: BLE001
            _log("soft_completion_gate_error", session=sid, error=str(e))

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

    if backend.strip_request_fields:
        before_keys = set(body.keys())
        body = strip_unsupported_responses_fields(
            body, drop=backend.strip_request_fields)
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
        forward_body = normalize_for_chat(body)

    trace.target_url = url
    trace.is_stream = is_stream

    # Measure what we're actually forwarding (post-transform). Lets us
    # quantify the win from sanitize/proactive_compact and find waste:
    # `est_input_tokens - forwarded_tokens_est` is the savings, and the
    # breakdown shows where the remaining tokens go.
    try:
        from .router import estimate_tokens, _flatten_text
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
        except Exception:  # noqa: BLE001
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
                        except Exception:  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                pass
            if action.backoff_s > 0:
                try:
                    await asyncio.sleep(action.backoff_s)
                except Exception:  # noqa: BLE001
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
    if chat_to_responses:
        translator = ChatToResponsesTranslator(valid_tool_names=valid_names)
    elif translate_tool_calls:
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
        rid = "resp_" + uuid4().hex[:24]
        payload = {
            "type": "response.completed",
            "response": {
                "id": rid,
                "object": "response",
                "model": body.get("model") or "tinyctx",
                "status": status_label,
                "incomplete_details": {"reason": "tinyctx_proxy_terminator",
                                       "message": message[:500]},
                "output": [],
            },
        }
        return f"event: response.completed\ndata: {json.dumps(payload)}\n\n".encode()

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
            chunk_q: asyncio.Queue = asyncio.Queue()
            _STATUS = object()
            _SENTINEL = object()
            _ERR = object()
            # ── Unified retry layer ──────────────────────────────────
            # Per user directive 2026-05-11 "凡是中断了都要加重试".
            # The producer task itself runs a retry loop: when the
            # upstream returns 4xx/5xx or raises a connection error
            # BEFORE we've put any body chunks on the queue, the
            # producer consults `retry_policy.classify_failure` and
            # may re-issue the request — switching to frontier when
            # the policy says retry_escalate. The consumer below sees
            # only the FINAL outcome (status + body or error). Once
            # the producer has put a body chunk, retry is blocked
            # (partial content already in flight).
            retry_state = retry_policy.RequestRetryState()
            # Mutable per-attempt state so we can swap on escalate.
            attempt_url = [url]
            attempt_headers = [headers]
            attempt_body = [body]
            attempt_decision = [decision]

            async def _producer():
                produced_chunk = False
                try:
                    while True:
                        retry_state.record_attempt()
                        cur_attempt_url = attempt_url[0]
                        cur_attempt_headers = attempt_headers[0]
                        cur_attempt_body = attempt_body[0]
                        cur_decision = attempt_decision[0]
                        attempt_started = time.time()
                        http_status: int | None = None
                        conn_error = False
                        exc_caught: Exception | None = None
                        err_body_for_retry: str = ""
                        retry_after_s = 0.0
                        try:
                            async with httpx.AsyncClient(
                                    timeout=timeout, transport=transport) as client:
                                async with client.stream(
                                        "POST", cur_attempt_url,
                                        headers=cur_attempt_headers,
                                        json=cur_attempt_body) as r:
                                    http_status = r.status_code
                                    if r.status_code >= 400:
                                        err_body_for_retry = (
                                            await r.aread()).decode(
                                            "utf-8", "replace")
                                        ra = (r.headers.get("retry-after")
                                              or r.headers.get("Retry-After"))
                                        try:
                                            retry_after_s = float(ra) if ra else 0.0
                                        except (TypeError, ValueError):
                                            retry_after_s = 0.0
                                        # Fall through to the policy check
                                        # below — do NOT push STATUS yet.
                                    else:
                                        # Success — emit STATUS and stream
                                        # body chunks. From this point on
                                        # retry is impossible (partial
                                        # content in flight).
                                        await chunk_q.put(
                                            (_STATUS, (r.status_code, None)))
                                        async for chunk in r.aiter_raw():
                                            produced_chunk = True
                                            await chunk_q.put((None, chunk))
                                        # finished successfully — push
                                        # sentinel (return skips the
                                        # outer else clause) and exit.
                                        await chunk_q.put(
                                            (_SENTINEL, None))
                                        return
                        except Exception as e:  # noqa: BLE001
                            conn_error = True
                            exc_caught = e
                        # Failure path. Consult policy.
                        if produced_chunk:
                            # We already streamed bytes to the consumer
                            # for THIS attempt's response body — cannot
                            # retry. Propagate.
                            if exc_caught is not None:
                                raise exc_caught
                            await chunk_q.put(
                                (_STATUS, (http_status or 0, err_body_for_retry)))
                            await chunk_q.put((_SENTINEL, None))
                            return
                        action = retry_policy.classify_failure(
                            route=cur_decision.route,
                            status=http_status,
                            is_connection_error=conn_error,
                            is_compaction=cur_decision.is_compaction,
                            attempts_used=retry_state.attempts_used,
                            max_total_retries=CFG.max_total_retries_per_request,
                            upstream_retry_count=CFG.upstream_retry_count,
                            retry_on_local_4xx_escalate_frontier=(
                                CFG.retry_on_local_4xx_escalate_frontier),
                            retry_on_frontier_4xx=CFG.retry_on_frontier_4xx,
                            retry_after_s=retry_after_s,
                        )
                        retry_state.last_action = action
                        if action.decision == "propagate":
                            if action.escalate_flag_reason:
                                try:
                                    from . import empty_response_guard as _erg
                                    _erg.force_next_to_frontier(
                                        erg_key, action.escalate_flag_reason)
                                except Exception:  # noqa: BLE001
                                    pass
                            if exc_caught is not None:
                                # surface connection error to consumer
                                raise exc_caught
                            # surface upstream 4xx/5xx to consumer
                            await chunk_q.put(
                                (_STATUS, (http_status or 0,
                                            err_body_for_retry)))
                            await chunk_q.put((_SENTINEL, None))
                            return
                        # retry_same / retry_escalate
                        new_url = cur_attempt_url
                        new_headers = cur_attempt_headers
                        new_decision = cur_decision
                        if action.decision == "retry_escalate":
                            esc_url, esc_headers_proto, _b, esc_decision, _bk = (
                                _build_frontier_retry_target(
                                    None, cur_attempt_body, action.reason))
                            # Preserve codex routing headers (openai-beta,
                            # x-codex-session-id, ...) from the original
                            # request, but REBUILD Authorization for the
                            # frontier backend. Without this, a
                            # local-backend bearer (e.g. LMStudio's sk-*)
                            # leaks to chatgpt.com and triggers a 401 that
                            # closes the stream at bytes_out=0 — codex
                            # shows "task interrupted" right after
                            # retry_attempted. See rq_f372c3c35c47444db89e.
                            merged = dict(cur_attempt_headers)
                            merged["Content-Type"] = "application/json"
                            merged.setdefault("Accept", "text/event-stream")
                            fb_key = _resolve_api_key(CFG.frontier, None)
                            if fb_key:
                                merged["Authorization"] = (
                                    fb_key if fb_key.lower().startswith(
                                        ("bearer ", "basic "))
                                    else f"Bearer {fb_key}")
                            else:
                                merged.pop("Authorization", None)
                            new_url = esc_url
                            new_headers = merged
                            new_decision = esc_decision
                            if action.escalate_flag_reason:
                                try:
                                    from . import empty_response_guard as _erg
                                    _erg.force_next_to_frontier(
                                        erg_key, action.escalate_flag_reason)
                                except Exception:  # noqa: BLE001
                                    pass
                            retry_state.record_escalation()
                        _log("retry_attempted",
                             session=proj_sid,
                             attempt_number=retry_state.attempts_used,
                             original_status=http_status,
                             retry_target=action.decision,
                             original_url=cur_attempt_url,
                             new_url=new_url,
                             reason=action.reason,
                             request_id=request_id,
                             conn_error=conn_error,
                             elapsed_s=round(time.time() - attempt_started, 3))
                        # Reset the stall watchdog's last-event timestamp
                        # on the retry boundary so the new upstream attempt
                        # gets a fresh threshold window. Without this, the
                        # countdown still references the FAILED attempt's
                        # last event, and a silent retry can run out the
                        # 180s before the watchdog ever fires.
                        try:
                            _stall.mark_event(proj_sid, conv_sid=conv_sid)
                        except Exception:  # noqa: BLE001
                            pass
                        if action.backoff_s > 0:
                            try:
                                await asyncio.sleep(action.backoff_s)
                            except Exception:  # noqa: BLE001
                                pass
                        attempt_url[0] = new_url
                        attempt_headers[0] = new_headers
                        attempt_decision[0] = new_decision
                        # loop and re-attempt
                except asyncio.CancelledError:
                    # Stall watchdog (or shutdown) cancelled us. Surface
                    # as a synthetic StallCancelledError so the consumer
                    # can emit a clean SSE terminator and trigger the
                    # next-turn escalation. We MUST NOT re-raise here —
                    # the producer is a fire-and-forget task whose
                    # cancellation must be communicated to the consumer
                    # via the queue, not by killing the whole generator.
                    try:
                        elapsed = _stall.seconds_since_event(proj_sid)
                    except Exception:  # noqa: BLE001
                        elapsed = None
                    synthetic = _stall.StallCancelledError(
                        "stall_watchdog_cancelled_relay",
                        proj_sid=proj_sid,
                        conv_sid=conv_sid,
                        elapsed_silent_s=elapsed,
                    )
                    try:
                        await chunk_q.put((_ERR, synthetic))
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as exc:  # noqa: BLE001
                    await chunk_q.put((_ERR, exc))
                else:
                    await chunk_q.put((_SENTINEL, None))

            producer = asyncio.create_task(_producer())
            if CFG.stall_watchdog_enabled:
                try:
                    _stall.register_task(proj_sid, producer)
                except Exception:  # noqa: BLE001
                    pass
            # Initial keepalive: emit one SSE comment frame IMMEDIATELY
            # on stream open, BEFORE waiting for the first upstream byte.
            # The idle-loop keepalive below only fires after
            # `keepalive_interval` seconds of silence — with the default
            # 15s interval, codex's client-side idle timeout (or any TCP
            # middlebox) can disconnect first when the upstream takes
            # >15s to deliver its first byte (large-context cold-start,
            # post-retry frontier wait). Emitting one keepalive frame at
            # t≈0 guarantees codex sees activity within the first turn
            # of the event loop, independent of upstream latency. SSE
            # comments are ignored by spec-compliant clients.
            yield b": tinyctx keepalive\n\n"
            keepalives_emitted += 1
            try:
                while True:
                    try:
                        tag, payload = await asyncio.wait_for(
                            chunk_q.get(), timeout=keepalive_interval)
                    except asyncio.TimeoutError:
                        yield b": tinyctx keepalive\n\n"
                        keepalives_emitted += 1
                        continue
                    if tag is _SENTINEL:
                        break
                    if tag is _ERR:
                        raise payload  # type: ignore[misc]
                    if tag is _STATUS:
                        status_code, err_body = payload
                        status = status_code
                        if CFG.stall_watchdog_enabled:
                            _stall.mark_event(proj_sid, conv_sid=conv_sid)
                        if err_body is not None:
                            _SESSION_ERROR_STREAK[proj_sid] += 1
                            _log("upstream_error", session=proj_sid,
                                 status=status_code, url=url,
                                 body=err_body[:2000])
                            yield (
                                f"event: error\ndata: "
                                f"{json.dumps({'status': status_code, 'body': err_body[:2000]})}"
                                f"\n\n").encode()
                            upstream_failed = True
                            upstream_failure_msg = (
                                f"upstream {status_code}: {err_body[:200]}")
                            # Forensics dump for upstream errors —
                            # capture the request that triggered the
                            # 4xx/5xx + the upstream's error body
                            if (CFG.forensics_enabled
                                    and CFG.forensics_capture_errors):
                                try:
                                    from . import forensics as _fx
                                    forensics_dir = CFG.log_dir.parent / "forensics"
                                    _fx.write_forensics_dump(
                                        forensics_dir, proj_sid,
                                        trigger=f"upstream_{status_code}",
                                        response_buffer=err_body or "",
                                        extra={"status": status_code,
                                               "url": url},
                                        max_dumps=CFG.forensics_max_dumps,
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                            # Don't break — wait for SENTINEL so producer
                            # cleanly closes its async-with stack.
                        else:
                            _SESSION_ERROR_STREAK[proj_sid] = 0
                        continue
                    # tag is None → real response-body chunk
                    bytes_out += len(payload)
                    if CFG.stall_watchdog_enabled:
                        _stall.mark_event(proj_sid, conv_sid=conv_sid)
                    # Soft-completion accumulator: just buffer the bytes,
                    # the LLM behavioral classifier runs ONCE at stream
                    # end (see finally block below). We don't decide
                    # mid-stream — hot-path stays cheap.
                    if CFG.soft_completion_gate_enabled:
                        try:
                            from . import soft_completion as _sc
                            _sc.accumulate_chunk(proj_sid, payload)
                        except Exception:  # noqa: BLE001
                            pass
                    if translator is None:
                        out_bytes = _intercept_completed(payload)
                        if out_bytes:
                            yield _capture_outgoing(out_bytes)
                    else:
                        for out in translator.feed(payload):
                            out_bytes = _intercept_completed(out)
                            if out_bytes:
                                yield _capture_outgoing(out_bytes)
                if translator is not None and not upstream_failed:
                    for out in translator.flush():
                        out_bytes = _intercept_completed(out)
                        if out_bytes:
                            yield _capture_outgoing(out_bytes)

                # ─── stream-rewrite synthesis ──────────────────────
                # We held back the response.completed event. Decide
                # whether to inject a synthetic advisor function_call
                # in front of it, then flush.
                if rewrite_enabled and holding_completion[0]:
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
                                except Exception:  # noqa: BLE001
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
            finally:
                if not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                # Always unregister, even if the watchdog already
                # cancelled us — `unregister_task(task=producer)` is a
                # no-op when a fresh stream has already replaced the
                # registration, so we never clobber the next stream.
                if CFG.stall_watchdog_enabled:
                    try:
                        _stall.unregister_task(proj_sid, producer)
                    except Exception:  # noqa: BLE001
                        pass
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
                                except Exception:  # noqa: BLE001
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
        # Watchdog cancelled the in-flight relay because the upstream
        # went silent past CFG.stall_threshold_s. Emit a clean SSE
        # error event and let the always-fires terminator below close
        # the stream with status=incomplete so codex's SSE parser
        # accepts it. force_next_to_frontier was already set by the
        # stall callback — codex's follow-up turn routes frontier-side.
        _SESSION_ERROR_STREAK[proj_sid] += 1
        _log("stream_error", session=proj_sid, error=str(e),
             error_type="StallCancelledError",
             bytes_yielded=bytes_out,
             elapsed_silent_s=e.elapsed_silent_s,
             conv_sid=e.conv_sid)
        status = 0
        yield (
            f"event: error\ndata: "
            f"{json.dumps({'message': str(e), 'type': 'stall_cancelled'})}"
            f"\n\n").encode()
        upstream_failed = True
        upstream_failure_msg = f"stall_cancelled: {e!s}"
        # Forensics dump — capture what we saw so post-mortems can
        # confirm the watchdog fired correctly.
        if CFG.forensics_enabled and CFG.forensics_capture_errors:
            try:
                from . import forensics as _fx
                forensics_dir = CFG.log_dir.parent / "forensics"
                _fx.write_forensics_dump(
                    forensics_dir, proj_sid,
                    trigger="stall_cancelled_relay",
                    response_buffer="",
                    timing={"elapsed_s": round(time.time() - started, 3),
                            "elapsed_silent_s": e.elapsed_silent_s},
                    extra={"conv_sid": e.conv_sid,
                           "bytes_yielded": bytes_out,
                           "url": url},
                    max_dumps=CFG.forensics_max_dumps,
                )
            except Exception:  # noqa: BLE001
                pass
    except httpx.HTTPError as e:
        _SESSION_ERROR_STREAK[proj_sid] += 1
        _log("stream_error", session=proj_sid, error=str(e),
             error_type=type(e).__name__, bytes_yielded=bytes_out)
        status = 0
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n".encode()
        upstream_failed = True
        upstream_failure_msg = f"http error: {e!s}"

        # On transient stream error, set force-frontier flag so the NEXT
        # request from codex (whether codex auto-retries or user nudges)
        # bypasses the unstable backend and routes to gpt-5.5. Per user
        # directive: "先重试原来模型，再出错就升级" — first retry happens
        # implicitly via codex's natural behavior, second attempt then
        # escalates by virtue of this flag.
        is_transient = isinstance(e, (
            httpx.RemoteProtocolError, httpx.ReadTimeout,
            httpx.ReadError, httpx.ConnectError, httpx.WriteError))
        if (CFG.empty_response_guard_enabled
                and is_transient
                and CFG.upstream_retry_enabled):
            try:
                from . import empty_response_guard as _erg
                # Only escalate if THIS session has now had multiple
                # consecutive errors (≥ retry_count + 1). The first
                # error sets the streak; subsequent errors trip escalation.
                if _SESSION_ERROR_STREAK[proj_sid] > CFG.upstream_retry_count:
                    _erg.force_next_to_frontier(
                        erg_key,
                        f"stream_error escalate: {type(e).__name__} streak={_SESSION_ERROR_STREAK[proj_sid]}")
                    _phase_set(proj_sid, RequestPhase.escalated_to_frontier, request_id)
                    _log("stream_error_escalating_to_frontier", session=proj_sid,
                         streak=_SESSION_ERROR_STREAK[proj_sid])
                else:
                    _phase_set(proj_sid, RequestPhase.retrying, request_id)
                    _log("stream_error_will_retry_same_backend",
                         session=proj_sid, streak=_SESSION_ERROR_STREAK[proj_sid])
            except Exception:  # noqa: BLE001
                pass

        # Error forensics — capture request that triggered this stream
        # error so we can post-mortem the failure (network blip /
        # upstream timeout / TLS handshake issue / etc.)
        if CFG.forensics_enabled and CFG.forensics_capture_errors:
            try:
                from . import forensics as _fx
                forensics_dir = CFG.log_dir.parent / "forensics"
                _fx.write_forensics_dump(
                    forensics_dir, proj_sid,
                    trigger="stream_error",
                    response_buffer="",
                    timing={"elapsed_s": round(time.time() - started, 3)},
                    extra={"error": str(e)[:1000],
                           "error_type": type(e).__name__,
                           "url": url,
                           "bytes_yielded": bytes_out,
                           "session_error_streak": _SESSION_ERROR_STREAK[proj_sid]},
                    max_dumps=CFG.forensics_max_dumps,
                )
            except Exception:  # noqa: BLE001
                pass
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
            except Exception:  # noqa: BLE001
                pass
        elapsed = round(time.time() - started, 3)
        _log("stream_done", session=proj_sid, route=decision.route, bytes=bytes_out,
             translated=bool(translator),
             elapsed_s=elapsed,
             keepalives=keepalives_emitted)
        # Soft-completion: spawn LLM behavioral classifier as a fire-
        # and-forget background task. It judges whether THIS turn's
        # output was a soft-punt-to-user; result lands as a flag that
        # the next request's gate check consumes. Never blocks the
        # stream's return — codex.app gets its bytes immediately.
        if (CFG.soft_completion_gate_enabled
                and status == 200
                and bytes_out > 0
                and not upstream_failed):
            _phase_set(proj_sid, RequestPhase.post_stream_classifying, request_id)
            try:
                from . import soft_completion as _sc
                api_key = (os.environ.get(CFG.local.api_key_env)
                           if CFG.local.api_key_env else None)
                # Snapshot the buffer at SPAWN time, not at task-run time.
                # Without this, asyncio scheduling delay (event loop busy
                # serving the next stream) could cause the bg task to read
                # a buffer that the next stream had already `reset_stream`
                # and not yet refilled — leading to spurious "no_buffer"
                # / "short_text text_chars=0" skips. Closure-capture the
                # snapshot so the dict isn't consulted later.
                buffer_snapshot = _sc._OUTPUT_BUFFER.get(proj_sid, "")
                # Extract semantic context from the request body for
                # the new classifier. The body has the entire
                # conversation history; mine it for user goal +
                # progress tracker + tool summary so classifier can
                # judge "did the agent finish the user's actual goal"
                # rather than just pattern-match the response shape.
                body_input = body.get("input") if isinstance(body, dict) else None
                user_goal_snapshot = _sc.extract_user_goal(body_input)
                tracker_snapshot = _sc.extract_progress_tracker(body_input)
                tool_summary_snapshot = _sc.extract_tool_summary(body_input)

                async def _bg_classify():
                    _log("soft_completion_classify_started", session=proj_sid,
                         buffer_chars_at_spawn=len(buffer_snapshot),
                         user_goal_chars=len(user_goal_snapshot),
                         tracker_chars=len(tracker_snapshot),
                         tool_summary=tool_summary_snapshot[:120])
                    try:
                        diag = await _sc.classify_at_stream_end_diag(
                            proj_sid,
                            local_base_url=CFG.local.base_url,
                            local_model=CFG.local.model,
                            api_key=api_key,
                            timeout_s=CFG.self_classify_timeout_s,
                            threshold=CFG.self_classify_threshold,
                            raw_buffer=buffer_snapshot,
                            user_goal=user_goal_snapshot,
                            progress_tracker=tracker_snapshot,
                            tool_summary=tool_summary_snapshot,
                            force_frontier_threshold=(
                                CFG.soft_completion_auto_force_frontier_threshold
                                if CFG.soft_completion_auto_force_frontier_enabled
                                else 1.01),
                            short_text_threshold=CFG.soft_completion_short_text_threshold,
                            stop_text_threshold=CFG.soft_completion_stop_text_threshold,
                            conv_sid=conv_sid,
                        )
                        # Always log the outcome — even None paths, so
                        # silent-skip cases are diagnosable. One of the
                        # following branches always fires:
                        if diag.result is not None:
                            _log("soft_completion_classified",
                                 session=proj_sid,
                                 soft_punt=diag.result.soft_punt,
                                 p=diag.result.p,
                                 reason=diag.result.reason,
                                 extracted_text_chars=diag.extracted_text_chars)
                            # PUNT forensics dump for high-confidence
                            # cases — lets us inspect what the agent
                            # actually said when classifier flagged it.
                            if (CFG.forensics_enabled
                                    and CFG.forensics_capture_punts
                                    and diag.result.soft_punt
                                    and diag.result.p >= CFG.forensics_punt_threshold):
                                try:
                                    forensics_dir = CFG.log_dir.parent / "forensics"
                                    p = _sc.write_punt_forensics(
                                        proj_sid, forensics_dir, diag.result, diag,
                                        max_dumps=CFG.forensics_max_dumps)
                                    if p:
                                        _log("forensics_dump_written",
                                             session=proj_sid, trigger="punt",
                                             path=p)
                                except Exception as fe:  # noqa: BLE001
                                    _log("forensics_dump_error",
                                         session=proj_sid, error=str(fe))
                            # C-4 hybrid: actively poke the codex.app
                            # session via `codex exec resume` side
                            # process. Turns the auto_force_frontier flag
                            # from passive (waits on user input) into
                            # active (immediate one-shot turn). See
                            # tinyctx/exec_resume.py.
                            if (CFG.exec_resume_enabled
                                    and diag.result.soft_punt
                                    and diag.result.p >= CFG.exec_resume_min_p
                                    and cwd):
                                try:
                                    from . import exec_resume as _xr
                                    log_dir = CFG.log_dir.parent / "exec_resume_logs"
                                    tiers = list(CFG.exec_resume_prompt_tiers or [])
                                    rec = await _xr.poke(
                                        cwd=cwd,
                                        prompt=CFG.exec_resume_prompt,
                                        prompt_tiers=tiers or None,
                                        codex_binary=CFG.exec_resume_codex_binary,
                                        sandbox=CFG.exec_resume_sandbox,
                                        approval_policy=CFG.exec_resume_approval_policy,
                                        cooldown_s=CFG.exec_resume_cooldown_s,
                                        max_per_minute=CFG.exec_resume_max_per_minute,
                                        timeout_s=CFG.exec_resume_timeout_s,
                                        log_dir=log_dir,
                                        proj_sid=proj_sid,
                                    )
                                    _log("exec_resume_poke",
                                         session=proj_sid,
                                         status=rec.status,
                                         reason=rec.reason,
                                         pid=rec.pid,
                                         resolved_session_id=rec.session_id,
                                         log_path=rec.log_path,
                                         p=diag.result.p)
                                except Exception as xe:  # noqa: BLE001
                                    _log("exec_resume_poke_error",
                                         session=proj_sid, error=str(xe))
                        elif diag.skipped_reason:
                            _log("soft_completion_classify_skipped",
                                 session=proj_sid,
                                 reason=diag.skipped_reason,
                                 finish_reason=diag.finish_reason,
                                 extracted_text_chars=diag.extracted_text_chars,
                                 raw_buffer_chars=diag.raw_buffer_chars,
                                 raw_head=diag.raw_buffer_head,
                                 raw_tail=diag.raw_buffer_tail)
                        elif diag.backend_error:
                            _log("soft_completion_classify_backend_error",
                                 session=proj_sid,
                                 error=diag.backend_error,
                                 status=diag.backend_status,
                                 extracted_text_chars=diag.extracted_text_chars)
                        else:
                            # Parse failure — backend returned 200 but
                            # content didn't yield a verdict.
                            _log("soft_completion_classify_parse_failed",
                                 session=proj_sid,
                                 status=diag.backend_status,
                                 raw_preview=diag.raw_content_preview,
                                 extracted_text_chars=diag.extracted_text_chars)
                    except Exception as e:  # noqa: BLE001
                        _log("soft_completion_classify_error",
                             session=proj_sid, error=str(e))
                asyncio.create_task(_bg_classify())
            except Exception as e:  # noqa: BLE001
                _log("soft_completion_classify_spawn_error",
                     session=proj_sid, error=str(e))
        # Empty-response guard: parse upstream's usage block from buffer
        # tail; if completion_tokens too low + finish_reason normal,
        # flag this session so the NEXT request gets routed to frontier.
        # See tinyctx/empty_response_guard.py.
        if (CFG.empty_response_guard_enabled
                and status == 200
                and not upstream_failed):
            try:
                from . import empty_response_guard as _erg
                from . import soft_completion as _sc
                buf_for_check = _sc._OUTPUT_BUFFER.get(proj_sid, "")
                info = _erg.maybe_flag_empty_response(
                    erg_key, buf_for_check,
                    min_completion_tokens=CFG.empty_response_min_completion_tokens)
                if info is not None:
                    _log("empty_response_detected", session=proj_sid,
                         completion_tokens=info.get("completion_tokens"),
                         finish_reason=info.get("finish_reason"),
                         reason=info.get("reason"))
                    # Forensics dump — capture request + response so
                    # next-time root cause is recoverable. The 05:07
                    # turn 1780 empty response had no captured body
                    # and is forever unrecoverable.
                    if CFG.forensics_enabled:
                        try:
                            from . import forensics as _fx
                            forensics_dir = CFG.log_dir.parent / "forensics"
                            path = _fx.write_forensics_dump(
                                forensics_dir,
                                proj_sid,
                                trigger="empty_response",
                                response_buffer=buf_for_check,
                                timing={
                                    "elapsed_s": elapsed,
                                    "started_at": started,
                                },
                                extra={
                                    "bytes_out": bytes_out,
                                    "keepalives_emitted": keepalives_emitted,
                                    "completion_tokens": info.get("completion_tokens"),
                                    "finish_reason": info.get("finish_reason"),
                                    "url": url,
                                },
                                max_dumps=CFG.forensics_max_dumps,
                            )
                            if path:
                                _log("forensics_dump_written",
                                     session=proj_sid, trigger="empty_response",
                                     path=str(path), file=path.name)
                        except Exception as fe:  # noqa: BLE001
                            _log("forensics_dump_error",
                                 session=proj_sid, error=str(fe))
            except Exception as e:  # noqa: BLE001
                _log("empty_response_guard_error",
                     session=proj_sid, error=str(e))
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
        forward_body = body if backend.wire_api == "responses" else normalize_for_chat(body)
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
