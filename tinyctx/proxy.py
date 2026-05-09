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
    inject_responses_defaults,
    normalize_for_chat,
    proactive_compact,
    purge_failed_tool_inputs,
    rewrite_model,
    expand_mcp_namespaces,
    inject_advisor_hint,
    scrub_unsupported_tools,
    strip_encrypted_content,
    strip_unsupported_responses_fields,
    trim_tools_for_frontier,
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


_PROACTIVE_SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing the OLDER turns of an in-progress coding session "
    "that is about to be truncated to fit within a context window. The "
    "RECENT turns will continue verbatim after your summary; the model "
    "reading your summary is the same one that wrote those turns. Your "
    "job is to give it just enough memory to keep working without losing "
    "the thread.\n\n"
    "Output a 250-500 word markdown handoff with these sections:\n"
    "## What we were trying to do\n"
    "  The user's actual goal. Be specific. Include any pivot the user made.\n"
    "## Files & decisions\n"
    "  Concrete paths touched. Decisions made (e.g. \"keep the env-knob\" or\n"
    "  \"drop SLAM setup\"). Verbatim where possible.\n"
    "## Commands & outcomes\n"
    "  Exact commands run and what they returned (success/failure + key\n"
    "  output line). Do NOT include long stderr — one line per command.\n"
    "## What's left / next step\n"
    "  Pending work the model was about to do, or the question it was\n"
    "  about to ask. Be explicit so the next turn can resume.\n\n"
    "Rules:\n"
    "  - Concrete > abstract. Say \"removed RayNeo SLAM env knobs in\n"
    "    com.foo.Bar.kt:42\" not \"made some Kotlin changes\".\n"
    "  - Drop redundancy and chitchat.\n"
    "  - Do NOT invent anything not in the conversation.\n"
    "  - Keep it terse. The next model has tokens to spare elsewhere."
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
            "max_tokens": 1200,
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
    trace = RequestTrace(session_id=sid)
    trace.project_session_key = proj_sid
    streak = _SESSION_ERROR_STREAK[proj_sid]
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
        # The agent invoked advisor — note timestamp so the stuck-loop
        # watchdog skips its grace window and doesn't nudge the parent
        # session right after the agent already escalated.
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
        except Exception as e:  # noqa: BLE001 — classifier must never fail forward
            _log("self_classify_error", session=sid, error=str(e))

    trace.route = decision.route
    trace.route_reason = decision.reason
    trace.is_compaction = decision.is_compaction
    trace.est_input_tokens = decision.est_input_tokens
    trace.turn_count = decision.turn_count
    trace.error_streak = streak

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
            body, was_injected = stuck_loop.maybe_inject_stuck_reminder(
                body, proj_sid, decision.turn_count,
                turn_trigger=CFG.stuck_loop_turn_trigger,
                turn_gap=CFG.stuck_loop_turn_gap,
                advisor_grace_s=CFG.stuck_loop_advisor_grace_s,
            )
            if was_injected:
                trace.stuck_reminder_injected = True
                trace.stuck_turn_count_at_inject = decision.turn_count
                _log("stuck_reminder_injected", session=sid,
                     proj_sid=proj_sid, turn_count=decision.turn_count)
        except Exception as e:  # noqa: BLE001 — watchdog must never block
            _log("stuck_loop_error", session=sid, error=str(e))

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
    # CFG.proactive_compact_threshold AND the request is NOT already a
    # codex compaction request. Replaces the middle of body.input with a
    # tinyctx summary item; codex's client-side history is unchanged so
    # the UI still shows every turn. See sanitize.proactive_compact for
    # full rationale.
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

    return await _forward(url, headers, forward_body, is_stream, proj_sid, decision,
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
    proj_sid = _project_session_key(request, sid)
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
                          trace=trace),
            media_type="text/event-stream",
        )
    started = time.time()
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
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
        # Non-streaming chat→responses translation. The streaming path in
        # _stream_proxy uses ChatToResponsesTranslator; this is the
        # equivalent for one-shot JSON responses (Vercel AI SDK's
        # generateObject hits the proxy with stream=false and expects a
        # Responses-API-shaped body back, even when the upstream is a
        # chat-completions backend like DeepSeek).
        if chat_to_responses and isinstance(payload, dict) and "choices" in payload:
            payload = _chat_to_responses_payload(payload)
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
                        *, transport: httpx.AsyncBaseTransport | None = None,
                        translate_tool_calls: bool = False,
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
            soft_completion.reset_stream(sid)
        except Exception:  # noqa: BLE001 — instrumentation must never fail forward
            pass

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

            async def _producer():
                try:
                    async with httpx.AsyncClient(
                            timeout=timeout, transport=transport) as client:
                        async with client.stream(
                                "POST", url, headers=headers, json=body) as r:
                            if r.status_code >= 400:
                                err_body = (await r.aread()).decode(
                                    "utf-8", "replace")
                                await chunk_q.put(
                                    (_STATUS, (r.status_code, err_body)))
                                # NOTE: fall through (no early return). The
                                # try/except/else's `else:` puts SENTINEL
                                # only when the try-block runs to completion;
                                # an early return would skip it and the
                                # consumer would hang waiting for sentinel.
                            else:
                                await chunk_q.put(
                                    (_STATUS, (r.status_code, None)))
                                async for chunk in r.aiter_raw():
                                    await chunk_q.put((None, chunk))
                except Exception as exc:  # noqa: BLE001
                    await chunk_q.put((_ERR, exc))
                else:
                    await chunk_q.put((_SENTINEL, None))

            producer = asyncio.create_task(_producer())
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
                        if err_body is not None:
                            _SESSION_ERROR_STREAK[sid] += 1
                            _log("upstream_error", session=sid,
                                 status=status_code, url=url,
                                 body=err_body[:2000])
                            yield (
                                f"event: error\ndata: "
                                f"{json.dumps({'status': status_code, 'body': err_body[:2000]})}"
                                f"\n\n").encode()
                            upstream_failed = True
                            upstream_failure_msg = (
                                f"upstream {status_code}: {err_body[:200]}")
                            # Don't break — wait for SENTINEL so producer
                            # cleanly closes its async-with stack.
                        else:
                            _SESSION_ERROR_STREAK[sid] = 0
                        continue
                    # tag is None → real response-body chunk
                    bytes_out += len(payload)
                    # Soft-completion accumulator: just buffer the bytes,
                    # the LLM behavioral classifier runs ONCE at stream
                    # end (see finally block below). We don't decide
                    # mid-stream — hot-path stays cheap.
                    if CFG.soft_completion_gate_enabled:
                        try:
                            from . import soft_completion as _sc
                            _sc.accumulate_chunk(sid, payload)
                        except Exception:  # noqa: BLE001
                            pass
                    if translator is None:
                        yield payload
                    else:
                        for out in translator.feed(payload):
                            yield out
                if translator is not None and not upstream_failed:
                    for out in translator.flush():
                        yield out
            finally:
                if not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
        else:
            # keepalive disabled — original simple loop, no extra task overhead
            async with httpx.AsyncClient(
                    timeout=timeout, transport=transport) as client:
                async with client.stream(
                        "POST", url, headers=headers, json=body) as r:
                    status = r.status_code
                    if r.status_code >= 400:
                        _SESSION_ERROR_STREAK[sid] += 1
                        text = (await r.aread()).decode("utf-8", "replace")
                        _log("upstream_error", session=sid,
                             status=r.status_code, url=url, body=text[:2000])
                        yield (
                            f"event: error\ndata: "
                            f"{json.dumps({'status': r.status_code, 'body': text[:2000]})}"
                            f"\n\n").encode()
                        upstream_failed = True
                        upstream_failure_msg = (
                            f"upstream {r.status_code}: {text[:200]}")
                    else:
                        _SESSION_ERROR_STREAK[sid] = 0
                        async for chunk in r.aiter_raw():
                            bytes_out += len(chunk)
                            # Soft-completion accumulator (no-keepalive path).
                            # LLM classifier runs at stream end.
                            if CFG.soft_completion_gate_enabled:
                                try:
                                    from . import soft_completion as _sc
                                    _sc.accumulate_chunk(sid, chunk)
                                except Exception:  # noqa: BLE001
                                    pass
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
        upstream_failed = True
        upstream_failure_msg = f"http error: {e!s}"
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
        elapsed = round(time.time() - started, 3)
        _log("stream_done", session=sid, route=decision.route, bytes=bytes_out,
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
                buffer_snapshot = _sc._OUTPUT_BUFFER.get(sid, "")
                async def _bg_classify():
                    _log("soft_completion_classify_started", session=sid,
                         buffer_chars_at_spawn=len(buffer_snapshot))
                    try:
                        diag = await _sc.classify_at_stream_end_diag(
                            sid,
                            local_base_url=CFG.local.base_url,
                            local_model=CFG.local.model,
                            api_key=api_key,
                            timeout_s=CFG.self_classify_timeout_s,
                            threshold=CFG.self_classify_threshold,
                            raw_buffer=buffer_snapshot,
                        )
                        # Always log the outcome — even None paths, so
                        # silent-skip cases are diagnosable. One of the
                        # following branches always fires:
                        if diag.result is not None:
                            _log("soft_completion_classified",
                                 session=sid,
                                 soft_punt=diag.result.soft_punt,
                                 p=diag.result.p,
                                 reason=diag.result.reason,
                                 extracted_text_chars=diag.extracted_text_chars)
                        elif diag.skipped_reason:
                            _log("soft_completion_classify_skipped",
                                 session=sid,
                                 reason=diag.skipped_reason,
                                 extracted_text_chars=diag.extracted_text_chars,
                                 raw_buffer_chars=diag.raw_buffer_chars,
                                 raw_head=diag.raw_buffer_head,
                                 raw_tail=diag.raw_buffer_tail)
                        elif diag.backend_error:
                            _log("soft_completion_classify_backend_error",
                                 session=sid,
                                 error=diag.backend_error,
                                 status=diag.backend_status,
                                 extracted_text_chars=diag.extracted_text_chars)
                        else:
                            # Parse failure — backend returned 200 but
                            # content didn't yield a verdict.
                            _log("soft_completion_classify_parse_failed",
                                 session=sid,
                                 status=diag.backend_status,
                                 raw_preview=diag.raw_content_preview,
                                 extracted_text_chars=diag.extracted_text_chars)
                    except Exception as e:  # noqa: BLE001
                        _log("soft_completion_classify_error",
                             session=sid, error=str(e))
                asyncio.create_task(_bg_classify())
            except Exception as e:  # noqa: BLE001
                _log("soft_completion_classify_spawn_error",
                     session=sid, error=str(e))
        if trace is not None:
            trace.status = status
            trace.bytes_out = bytes_out
            trace.translated = bool(translator)
            trace.translated_calls = (translator._emitted_calls
                                       if translator is not None else 0)
            trace.elapsed_s = elapsed
            trace.keepalives_emitted = keepalives_emitted
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
