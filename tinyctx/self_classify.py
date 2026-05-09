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


_SYSTEM_PROMPT = """You are a routing classifier embedded in a coding agent. Your only job: decide whether the next turn should escalate to a stronger advisor model (Opus-class) OR be handled by the local executor model.

Output ONLY a single JSON object on one line. No prose, no markdown fence:
{"escalate": true|false, "p": 0.0-1.0, "reason": "<≤12 words>"}

Critical: keep "reason" SHORT (≤12 words). Long reasons get truncated by the token cap and the JSON becomes invalid.

Escalate when ANY of these apply:
- 2+ valid architectural approaches with real trade-offs (data shape, API contract, retry semantics, concurrency, lock ordering)
- 2+ failed attempts at the same problem; user is stuck
- Non-trivial security or correctness decision (auth flow, schema migration, transaction boundary)
- User intent is ambiguous and the wrong interpretation will waste significant work
- Cross-file reasoning, deep analysis, or strategic planning required

Don't escalate when:
- Routine code edits (rename, add comment, format, simple bug fix)
- File reading / code scanning / syntax lookup / padding
- Continuation of an established approach (next obvious mechanical step)
- Tool result follow-up where next action is dictated by what was just read

Be honest. If unsure, set escalate=false with low p (0.3-0.5)."""


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
# Fallback for truncated JSON (response cut off mid-reason): salvage
# escalate + p directly from the raw text without requiring a closing
# brace. Live trace showed DeepSeek occasionally bleeds past the
# token cap on verbose-mode runs.
_ESC_RE = re.compile(r'"escalate"\s*:\s*(true|false)')
_P_RE = re.compile(r'"p"\s*:\s*(-?\d+(?:\.\d+)?)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def _parse_response(text: str) -> ClassifyResult | None:
    """Lenient parser: first try a complete JSON object containing an
    `escalate` field (handles markdown fences and surrounding prose).
    If that fails, fall back to regex-salvage of `escalate` and `p`
    fields directly — handles the case where the model's response was
    truncated mid-reason by the token cap. Returns None if neither
    path can extract a verdict."""
    if not isinstance(text, str) or not text:
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
        # 200 tokens is comfortably above the JSON object the prompt
        # requests (esc + p + ≤12-word reason ≈ 30-40 tokens) but
        # leaves headroom if the model ignores the brevity directive.
        # Bumped from 120 after a live trace showed truncation on
        # verbose models.
        "max_tokens": 200,
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
