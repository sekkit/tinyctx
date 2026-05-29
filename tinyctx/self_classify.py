"""Model-driven advisor recommendation classifier.

Anthropic's Advisor Strategy puts the routing decision in the model's
hands: the executor invokes `advisor()` when IT decides strategic input
is needed. tinyctx's pure proxy can't observe a tool call before
forwarding, so we approximate that contract with a lightweight
PRE-FLIGHT classifier: the proxy asks the LOCAL model itself whether
this turn deserves advisor guidance. By default the answer becomes
telemetry and a local route reason; legacy full-turn escalation is opt-in.

Why this is the most-general and Anthropic-aligned approach:
  - The classifier IS the executor model (or a peer of the same class).
    No external trained scorer to maintain. No labeled training data.
  - Quality automatically tracks the local model's quality — when you
    upgrade local, the classifier upgrades for free.
  - The reason field is interpretable: every advisor recommendation lands
    in trace JSONL with the model's own justification.

Cost / latency:
  - ~300-token system prompt + ~5-turn tail + ~50-token JSON response
  - One call to local backend per fresh user query (skipped for tool-
    result roundtrips). DeepSeek-class: ~$0.000035 / call, ~200ms.
  - Result cached by `(scope, hash(instructions+last_user_msg))` for
    60s — repeated identical bodies (e.g., codex retries) hit cache.

Reference research (full discussion in PR description):
  - AutoMix (Madaan et al., NeurIPS 2023, arxiv:2310.12963): small
    model self-verifies, escalates if uncertain.
  - LLM-as-a-Judge (Zheng et al., NeurIPS 2023, arxiv:2306.05685):
    foundational — small models can reliably self-evaluate.
  - FrugalGPT (Chen et al., 2023, arxiv:2305.05176): cascade with a
    learned scorer (alternative path).
  - RouteLLM (Ong et al., COLM 2024, arxiv:2406.18665): preference-
    trained router (alternative path, requires training data).
"""
from __future__ import annotations

import hashlib
import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


_SYSTEM_PROMPT = """You are an advisor-need classifier inside an LLM gateway. Decide whether the local executor should ask a stronger advisor for a SHORT PLAN/CORRECTION, or continue locally. The work could be coding, debugging, research, writing, planning, analysis, code review, refactoring, data work, system design, or any other reasoning task.

Output EXACTLY one JSON object on a single line. No prose, no markdown fence, no commentary:
{"escalate": true|false, "p": 0.0-1.0, "reason": "<≤10 words>"}

CRITICAL: keep "reason" ≤10 words. Long reasons get truncated by the token cap and break parsing.

═══ Recommend advisor (true, p ≥ 0.7) when ANY apply ═══

Decision quality matters
  - Multiple valid approaches with real trade-offs (architecture, API contract, model/library/framework choice, data shape, study design, naming that propagates, build pipeline)
  - Hard-to-reverse decisions (production schemas, public interfaces, persisted formats, contracts, branding, release tags)
  - High-stakes correctness: security, auth, money handling, transactions, concurrency invariants, race conditions, data integrity, privacy, legal/compliance, safety

Stuck or failure signals
  - 2+ failed attempts at the same sub-problem
  - Tests fail after multiple fixes; reasoning loops with no convergence
  - Empirical results contradict prior assumptions or each other
  - "I don't know how to proceed" / asking for direction

Reasoning depth required
  - Subtle technical judgment where a short second opinion would materially change the plan: edge cases, off-by-one, FP precision, locale, timezone, encoding, threat model, ambiguous spec
  - Adversarial reasoning: what could go wrong, what's an attacker doing, what edge case breaks this

Ambiguous intent
  - User request has multiple plausible interpretations and picking the wrong one wastes ≥30 minutes of work
  - Domain term has conflicting common meanings; spec is silent on a load-bearing detail

═══ Don't escalate (false, p ≥ 0.7) when ═══

Mechanical / routine
  - Renames, formatting, comment edits, boilerplate generation, import sorting
  - File / code scanning, search, lookup
  - Syntax checks, fact retrieval, format conversion (json↔yaml, csv↔table, etc.)
  - Simple Q&A from memory or quick docs

Driven by just-read input
  - Tool result dictates the next obvious step (read this file → patch line N; ran test → fix the obvious failure; saw stack trace → trace one frame up)
  - User correction with explicit instruction ("change X to Y", "use this name", "remove that block")

Low-stakes / throwaway
  - Experimental scratch work, prototype, throwaway script
  - Documentation tweak, README polish, comment improvement

Continuation
  - Next obvious step in an already-chosen plan
  - Mechanical implementation of a recently-made decision

═══ Calibration ═══

If genuinely unsure → escalate=false with p in 0.3-0.5. The advisor is expensive; only call when a stronger reasoner would GENUINELY change the answer (not just say it more eloquently).

Don't recommend advisor just because the task has multiple steps, the input is long, or the topic is technical. Length, workflow breadth, and jargon are not difficulty.

Don't escalate to "be safe". Confidence and brevity in the local model is fine when the work is routine.

═══ Reason field rules ═══

- ≤10 words, noun-phrase fragment (no full sentence)
- State the LOAD-BEARING factor, not boilerplate

Good reasons (concise + specific):
  "architectural choice, concurrency trade-offs"
  "stuck: 3 failed test fixes"
  "ambiguous user intent on contract"
  "routine rename, single file"
  "tool result dictates next step"
  "syntax lookup, low stakes"
  "auth flow correctness, security stakes"
  "format conversion, deterministic"

Bad reasons (verbose, unspecific, repeats criteria):
  "This task involves multiple architectural approaches with real..." (truncates)
  "Hard task" (no specificity)
  "Complex code" (too vague)"""


@dataclass
class ClassifyResult:
    escalate: bool
    p: float
    reason: str
    cached: bool = False


@dataclass
class ActionSignature:
    action: str
    target: str
    confidence: float
    reason: str = ""


@dataclass
class ConsistencyResult:
    samples: list[ActionSignature]
    agreed: bool
    reason: str


_CACHE: dict[str, tuple[float, "ClassifyResult"]] = {}
_CACHE_TTL_S = 60.0


_ACTION_SYSTEM_PROMPT = """You are an action-signature sampler inside an LLM gateway. Given the current turn, predict the local executor's NEXT decisive action, not the final answer.

Output EXACTLY one JSON object on a single line. No prose:
{"action":"answer|inspect|edit|run|ask_user|plan|unknown","target":"<file/tool/domain/general>","confidence":0.0-1.0,"reason":"<≤8 words>"}

Rules:
- action=inspect when the next step is reading/searching code, logs, docs, or files.
- action=edit when the next step is patching files.
- action=run when the next step is tests, build, shell verification, or a command.
- action=ask_user only when human input is genuinely required.
- action=answer when the deliverable is a direct response.
- target should be the main file path, tool name, command family, or domain. Use "general" if unclear.
- Keep reason short. Do not solve the task."""

_ACTION_JSON_RE = re.compile(r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}', re.DOTALL)
_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]*)"')
_TARGET_RE = re.compile(r'"target"\s*:\s*"([^"]*)"')
_CONF_RE = re.compile(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)')
_ACTION_ALLOWED = {"answer", "inspect", "edit", "run", "ask_user", "plan", "unknown"}


def _cache_key(body: dict[str, Any], scope: str) -> str:
    """Compose a cache key from `scope` (typically project_session_key)
    and the hash of (instructions tail + last user message). Same body +
    same scope → same key → cache hit; different scope → no collision."""
    inst = body.get("instructions", "") or ""
    if isinstance(inst, str):
        inst = inst[-2000:]
    last_user = ""
    items = body.get("input") or body.get("messages") or []
    if isinstance(items, list):
        for it in reversed(items):
            if not isinstance(it, dict):
                continue
            r = it.get("role")
            t = it.get("type")
            if r == "user" or (t == "message" and r == "user"):
                content = it.get("content")
                if isinstance(content, str):
                    last_user = content[-2000:]
                elif isinstance(content, list):
                    for c in content:
                        if (isinstance(c, dict)
                                and c.get("type") in ("text", "input_text")):
                            last_user = (c.get("text") or "")[-2000:]
                            break
                break
    h = hashlib.sha256(
        f"{scope}\n{inst}\n{last_user}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{scope}:{h}"


def looks_like_user_query(body: dict[str, Any]) -> bool:
    """Skip pure tool-result roundtrips. Only fire on a fresh user
    query at the tail. Tool result follow-ups are decided by what was
    just read; the model already has its instructions for that case."""
    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list) or not items:
        return False
    last = items[-1]
    if not isinstance(last, dict):
        return False
    role = last.get("role")
    t = last.get("type")
    if t in ("function_call_output", "tool_result", "mcp_result"):
        return False
    if role == "user":
        return True
    if t == "message" and role == "user":
        return True
    return False


def _extract_input_tail(body: dict[str, Any], n: int = 5) -> str:
    """Build the user prompt sent to the classifier — last `n` items
    of conversation as compact tagged blocks. Caps each item at 1500
    chars to keep classifier prompt small."""
    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for it in items[-n:]:
        if not isinstance(it, dict):
            continue
        role = it.get("role") or it.get("type") or "?"
        content = it.get("content")
        if isinstance(content, str):
            txt = content
        elif isinstance(content, list):
            txt_parts = [c.get("text", "") for c in content
                         if isinstance(c, dict)
                         and c.get("type") in ("text", "input_text", "output_text")]
            txt = "\n".join(p for p in txt_parts if p)
        else:
            txt = ""
        if txt.strip():
            parts.append(f"<{role}>{txt[:1500]}</{role}>")
    return "\n".join(parts)


_JSON_RE = re.compile(
    r'\{[^{}]*"escalate"\s*:\s*(?:true|false)[^{}]*\}',
    re.DOTALL,
)
# Some reasoning models (qwen3-think, R1 derivatives) emit their CoT
# inline as `<think>…</think>` in `content` rather than splitting into
# `reasoning_content`. Strip closed think-blocks before parsing. We
# also tolerate an unclosed leading `<think>` (CoT cut off by the token
# cap with no JSON yet) — in that case the body is empty post-strip
# and the parser falls through to None.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^\s*<think>.*$", re.DOTALL | re.IGNORECASE)
# Fallback for truncated JSON (response cut off mid-reason): salvage
# escalate + p directly from the raw text without requiring a closing
# brace. Live trace showed verbose models occasionally bleed past the
# token cap.
_ESC_RE = re.compile(r'"escalate"\s*:\s*(true|false)')
_P_RE = re.compile(r'"p"\s*:\s*(-?\d+(?:\.\d+)?)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def _strip_thinking(text: str) -> str:
    """Remove `<think>…</think>` blocks that some reasoning-class
    backends leak into the content field. Handles both closed blocks
    and an unclosed leading think tag (truncated CoT)."""
    if not isinstance(text, str) or not text:
        return text
    # Remove all closed <think>…</think> blocks
    cleaned = _THINK_RE.sub("", text)
    # If what remains is just an unclosed <think>… (no JSON ever
    # emitted), drop it. The fallback parser will then return None
    # rather than scan think-content for spurious "escalate" matches.
    if _OPEN_THINK_RE.match(cleaned):
        return ""
    return cleaned


def _parse_response(text: str) -> ClassifyResult | None:
    """Lenient parser: first try a complete JSON object containing an
    `escalate` field (handles markdown fences and surrounding prose).
    If that fails, fall back to regex-salvage of `escalate` and `p`
    fields directly — handles the case where the model's response was
    truncated mid-reason by the token cap. Returns None if neither
    path can extract a verdict."""
    if not isinstance(text, str) or not text:
        return None
    text = _strip_thinking(text)
    if not text:
        return None
    m = _JSON_RE.search(text)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            esc = bool(d.get("escalate"))
            try:
                p = float(d.get("p", 0.5))
            except (ValueError, TypeError):
                p = 0.5
            p = max(0.0, min(1.0, p))
            reason = str(d.get("reason", ""))[:200]
            return ClassifyResult(escalate=esc, p=p, reason=reason)
    # Fallback: salvage from possibly-truncated text
    m_esc = _ESC_RE.search(text)
    if not m_esc:
        return None
    esc = m_esc.group(1) == "true"
    m_p = _P_RE.search(text)
    p = 0.5
    if m_p:
        try:
            p = float(m_p.group(1))
        except (ValueError, TypeError):
            p = 0.5
        p = max(0.0, min(1.0, p))
    m_r = _REASON_RE.search(text)
    reason = (m_r.group(1) if m_r else "")[:200]
    if not reason:
        # If even reason got truncated, leave a marker so trace shows
        # we salvaged from a partial response.
        reason = "[salvaged from truncated classifier response]"
    return ClassifyResult(escalate=esc, p=p, reason=reason)


def _normalize_signature_part(value: str, *, default: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return default
    value = re.sub(r"\s+", " ", value)
    return value[:120]


def _parse_action_signature(text: str) -> ActionSignature | None:
    if not isinstance(text, str) or not text:
        return None
    text = _strip_thinking(text)
    if not text:
        return None
    data: dict[str, Any] | None = None
    m = _ACTION_JSON_RE.search(text)
    if m:
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, dict):
            data = raw
    if data is None:
        m_action = _ACTION_RE.search(text)
        if not m_action:
            return None
        data = {
            "action": m_action.group(1),
            "target": (_TARGET_RE.search(text).group(1)
                       if _TARGET_RE.search(text) else "general"),
            "confidence": (_CONF_RE.search(text).group(1)
                           if _CONF_RE.search(text) else 0.5),
        }

    action = _normalize_signature_part(str(data.get("action", "")),
                                       default="unknown")
    if action not in _ACTION_ALLOWED:
        action = "unknown"
    target = _normalize_signature_part(str(data.get("target", "")),
                                       default="general")
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", ""))[:120]
    return ActionSignature(
        action=action,
        target=target,
        confidence=confidence,
        reason=reason,
    )


def summarize_consistency(
    samples: list[ActionSignature],
) -> ConsistencyResult | None:
    valid_samples = [
        sample for sample in samples
        if sample.confidence >= 0.5
        and sample.action != "unknown"
    ]
    if len(valid_samples) < 2:
        return None
    counts: dict[tuple[str, str], int] = {}
    for sample in valid_samples:
        key = (sample.action, sample.target)
        counts[key] = counts.get(key, 0) + 1
    best_key, best_count = max(counts.items(), key=lambda item: item[1])
    agreed = best_count > (len(valid_samples) / 2)
    if agreed:
        reason = f"agree {best_key[0]}:{best_key[1]} {best_count}/{len(valid_samples)}"
    else:
        parts = [
            f"{action}:{target}={count}"
            for (action, target), count in sorted(counts.items())
        ]
        reason = f"disagree {';'.join(parts)}"
    return ConsistencyResult(
        samples=valid_samples, agreed=agreed, reason=reason[:240])


async def _action_signature_call(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> ActionSignature | None:
    try:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        text = msg.get("content", "") or msg.get("reasoning_content", "") or ""
        return _parse_action_signature(text)
    except Exception:
        return None


async def sample_action_signatures(
    body: dict[str, Any],
    local_base_url: str,
    local_model: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 20.0,
    sample_count: int = 3,
) -> ConsistencyResult | None:
    """Sample local next-action signatures and return majority agreement.

    This is intentionally local-only and only meant for self-classify
    boundary turns. It never raises; parse/backend failures return None
    so callers can keep existing routing behavior.
    """
    if sample_count <= 1:
        return None
    if not looks_like_user_query(body):
        return None
    user_prompt = _extract_input_tail(body)
    if not user_prompt:
        return None

    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 256,
        "reasoning_effort": "low",
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = local_base_url.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        tasks = [
            _action_signature_call(
                client, url=url, payload=payload, headers=headers)
            for _ in range(sample_count)
        ]
        raw_samples = await asyncio.gather(*tasks, return_exceptions=True)

    samples = [
        sample for sample in raw_samples
        if isinstance(sample, ActionSignature)
    ]
    if len(samples) < 2:
        return None
    return summarize_consistency(samples)


async def classify(body: dict[str, Any],
                   local_base_url: str,
                   local_model: str,
                   *,
                   api_key: str | None = None,
                   timeout_s: float = 5.0,
                   scope: str = "") -> ClassifyResult | None:
    """Ask the local model whether to escalate. Returns None when:
      - body doesn't have a fresh user query at the tail
      - local model is unreachable / times out
      - response can't be parsed as the expected JSON shape

    Cached for 60s by (scope, hash(instructions+last_user_msg)).
    Never raises — returns None and lets the caller fall back to the
    routing heuristic. The HTTP call is the sole side effect.
    """
    if not looks_like_user_query(body):
        return None

    key = _cache_key(body, scope)
    now = time.time()
    cached = _CACHE.get(key)
    if cached:
        ts, res = cached
        if now - ts < _CACHE_TTL_S:
            return ClassifyResult(
                escalate=res.escalate, p=res.p,
                reason=res.reason, cached=True)

    user_prompt = _extract_input_tail(body)
    if not user_prompt:
        return None

    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        # 2048 is the right shape for reasoning-class local models
        # (qwen3.6, DeepSeek-R1 family). They burn 150-250 tokens on
        # hidden chain-of-thought (`reasoning_content`) BEFORE emitting
        # any visible content — at the old 120/200 cap, all budget went
        # to reasoning and `content` came back empty with
        # finish_reason=length. The JSON answer itself only needs
        # ~40 tokens; the headroom is for the CoT phase.
        # No frontier cost — this is a single local call cached 60s.
        # Hint flags below are best-effort: backends that don't honour
        # them ignore them silently.
        "max_tokens": 2048,
        "reasoning_effort": "low",
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = local_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s)) as client:
            r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:  # noqa: BLE001 — silent fallback
        return None

    result = _parse_response(text)
    if result is not None:
        _CACHE[key] = (now, result)
    return result


def clear_cache(scope_prefix: str | None = None) -> None:
    """Test/dev helper. Clears the entire cache or only entries whose
    key starts with `scope_prefix`."""
    if scope_prefix is None:
        _CACHE.clear()
        return
    for k in list(_CACHE):
        if k.startswith(scope_prefix):
            _CACHE.pop(k, None)
