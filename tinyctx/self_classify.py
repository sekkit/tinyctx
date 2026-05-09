"""Model-driven escalation classifier (Anthropic Advisor Strategy aligned).

Anthropic's Advisor Strategy puts the routing decision in the model's
hands: the executor invokes `advisor()` when IT decides strategic input
is needed. tinyctx's pure proxy can't observe a tool call before
forwarding, so we approximate that contract with a lightweight
PRE-FLIGHT classifier: the proxy asks the LOCAL model itself "should
this turn escalate?" and uses the answer to override its routing.

Why this is the most-general and Anthropic-aligned approach:
  - The classifier IS the executor model (or a peer of the same class).
    No external trained scorer to maintain. No labeled training data.
  - Quality automatically tracks the local model's quality — when you
    upgrade local, the classifier upgrades for free.
  - The reason field is interpretable: every escalation decision lands
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
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


_SYSTEM_PROMPT = """You are a routing classifier inside an LLM gateway. Decide whether the next turn should escalate to a stronger advisor model OR be handled by the cheaper local executor. The work could be coding, debugging, research, writing, planning, analysis, code review, refactoring, data work, system design, or any other reasoning task.

Output EXACTLY one JSON object on a single line. No prose, no markdown fence, no commentary:
{"escalate": true|false, "p": 0.0-1.0, "reason": "<≤10 words>"}

CRITICAL: keep "reason" ≤10 words. Long reasons get truncated by the token cap and break parsing.

═══ Escalate (true, p ≥ 0.7) when ANY apply ═══

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
  - Cross-file or cross-document synthesis (>3 files / sources)
  - Multi-step plans where early missteps compound (3+ dependent steps)
  - Subtle technical judgment: edge cases, off-by-one, FP precision, locale, timezone, encoding, threat model, ambiguous spec
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

Don't escalate just because the input is long or the topic is technical. Length and jargon are not difficulty.

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


_CACHE: dict[str, tuple[float, "ClassifyResult"]] = {}
_CACHE_TTL_S = 60.0


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
