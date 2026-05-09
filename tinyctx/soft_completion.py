"""Soft-completion gate v2 — LLM-based behavioral classifier.

v1 used regex patterns to catch specific phrasings ("what would you
like", etc.). That was brittle and language-limited. v2 asks the LOCAL
model itself "did the agent just soft-punt to the user?" — same
approach as self_classify, just inverted (post-stream rather than
pre-flight).

Why this exists
───────────────
Live trace 2026-05-10: stuck_loop watchdog saved the session from a
1300+ turn loop. The agent then went into a different failure: wrapped
up the bug fix, declared "All done", and asked "What would you like
to work on next?" without enumerating tracker / running verification /
calling advisor. §4 收工纪律 + §3 advisor gate (templates/AGENTS.md)
target "agent declares done"; neither bound to this *soft punt to
user* shape. User directive: 如果非要提问，走 advisor 进行回答.

Why LLM (not regex)
───────────────────
Regex catches "what would you like" but misses "I think we should
pause and let you decide" or "Hmm, this is up to you — どうしましょう".
LLM understands semantic intent: "is this a punt or a load-bearing
clarifying question?" — agnostic to language and phrasing.

Mechanism
─────────
1. **Accumulator**: per-session ring buffer (`_OUTPUT_BUFFER[proj_sid]`,
   capped at `_BUFFER_MAX` raw bytes) collects the streamed response.
   Reset at start of each stream.
2. **Stream-end classifier**: when the stream completes successfully,
   `classify_at_stream_end` is spawned as a background task. It:
     - extracts the assistant's text deltas from the SSE-wrapped buffer
     - feeds the last ~4KB of extracted text to the local model with
       a behavioral classifier prompt
     - parses the JSON verdict; sets the flag on `soft_punt: true && p ≥ τ`
3. **Gate**: on the next request to the same `proj_sid`, the proxy
   prepends a `<system-reminder>` to `body.input` requiring the agent
   to spawn_agent(role="advisor") with the would-be user question and
   tracker state, and act on advisor's `ask:`/`work:` verdict.

Cost / latency
──────────────
One extra local-model call per agent response (≈300-token prompt + 30-
token JSON output ≈ 200ms on DeepSeek-class). Runs ASYNCHRONOUSLY in
the background — does NOT add to user-perceived latency. Codex.app
typically waits seconds before the user types a follow-up, so the
flag is reliably set in time for the next request's gate-check.
Skipped for short outputs (<200 chars text, likely tool-call only).

Interaction with the rest
─────────────────────────
- stuck_loop fires on `turn_count > 80` without convergence (loop)
- soft_completion fires on user-facing question without completion
- §4 + advisor gate in AGENTS.md fire on "agent declares done"
A single session can hit any combination; they nudge in different
directions.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx


# ─── per-session state ─────────────────────────────────────────────────────

_SOFT_COMPLETION_FLAG: dict[str, dict[str, Any]] = defaultdict(dict)
_OUTPUT_BUFFER: dict[str, str] = defaultdict(str)
# Cap raw SSE buffer at 64KB. After SSE+JSON-overhead extraction this
# yields ~30-50KB of actual text — comfortably above the 4KB tail we
# feed the classifier.
_BUFFER_MAX = 65536


# ─── LLM behavioral classifier ─────────────────────────────────────────────

_CLASSIFIER_SYSTEM_PROMPT = """You are a behavioral classifier inside an LLM proxy. Read an assistant's final response from a coding/agent task, and decide whether it ended by SOFT-PUNTING to the user — i.e. asking a meta-question instead of completing the work or running its stated verifications.

Output EXACTLY one JSON object on a single line. No prose, no markdown:
{"soft_punt": true|false, "p": 0.0-1.0, "reason": "<≤8 words>"}

═══ SOFT-PUNT (true, p ≥ 0.7) ═══

The assistant ends the turn by:
  - asking the user "what next / what should I do / what would you like / which option / shall I continue / 接下来做什么" or any equivalent meta-question that hands the next decision back to the user
  - listing options ("Some options: A / B / C — let me know") and stopping
  - declaring "all done / wrapped up / final summary" while user-stated verification (build / run / test / deploy) was NOT actually executed
  - phrasing in any language with the same semantic shape

═══ NOT SOFT-PUNT (false, p ≥ 0.7) ═══

  - The response IS a tool call (function_call) — agent is still making progress.
  - Asking a load-bearing clarifying question that genuinely needs user input (ambiguous spec, conflicting requirements, missing credentials, irreversible-action confirmation).
  - Submitting a substantive technical answer / code / analysis as the answer itself.
  - Reporting a hard failure with specific blocking reason (e.g. "build failed: <log>; need user to fix env X before I can continue").
  - Mid-thought streaming text without any closing question.

═══ Calibration ═══

Be CONSERVATIVE. False positives cost ~5K frontier tokens (one extra advisor call). False negatives mean the user sees an incomplete task — the original problem we are trying to fix.

Reason field: ≤8 words, noun-phrase fragment. Examples:
  "asks user what to do next"          ← soft_punt:true
  "lists options without committing"   ← soft_punt:true
  "declares done but verification not run" ← soft_punt:true
  "running tool, still progressing"    ← soft_punt:false
  "load-bearing clarifying question"   ← soft_punt:false
  "substantive technical answer"       ← soft_punt:false"""


@dataclass
class ClassifyResult:
    soft_punt: bool
    p: float
    reason: str


_JSON_RE = re.compile(
    r'\{[^{}]*"soft_punt"\s*:\s*(?:true|false)[^{}]*\}', re.DOTALL)
_PUNT_RE = re.compile(r'"soft_punt"\s*:\s*(true|false)')
_P_RE = re.compile(r'"p"\s*:\s*(-?\d+(?:\.\d+)?)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')
# Some reasoning-class local models leak <think>…</think> into content.
# Same handling as self_classify._strip_thinking.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _THINK_RE.sub("", text)


def _parse_response(text: str) -> ClassifyResult | None:
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
            sp = bool(d.get("soft_punt"))
            try:
                p = float(d.get("p", 0.5))
            except (ValueError, TypeError):
                p = 0.5
            p = max(0.0, min(1.0, p))
            reason = str(d.get("reason", ""))[:200]
            return ClassifyResult(soft_punt=sp, p=p, reason=reason)
    # Fallback: salvage from possibly-truncated text
    m_sp = _PUNT_RE.search(text)
    if not m_sp:
        return None
    sp = m_sp.group(1) == "true"
    m_p = _P_RE.search(text)
    p = 0.5
    if m_p:
        try:
            p = max(0.0, min(1.0, float(m_p.group(1))))
        except (ValueError, TypeError):
            pass
    m_r = _REASON_RE.search(text)
    reason = (m_r.group(1) if m_r else "[salvaged]")[:200]
    return ClassifyResult(soft_punt=sp, p=p, reason=reason)


# ─── delta extraction from SSE-wrapped buffer ──────────────────────────────
# Both Responses-API ({"type":"response.output_text.delta","delta":"..."})
# and Chat-Completions ({"choices":[{"delta":{"content":"..."}}]}) emit
# the same JSON-string-escaped content via a `delta`/`content` field.
# Regex matches the simplest envelope; reasoning-content / tool-call
# events are skipped (they don't carry the same field shape).

_TEXT_DELTA_RE = re.compile(
    r'"(?:delta|content)"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _extract_text_from_buffer(buf: str) -> str:
    """Pull the assistant's text content out of an SSE-wrapped raw
    response buffer. Handles JSON string escapes (\\n, \\", \\\\).
    Falls back to returning the raw buffer if no delta fields match
    (e.g. the upstream wasn't streaming JSON-shaped events)."""
    if not buf:
        return ""
    matches = _TEXT_DELTA_RE.findall(buf)
    if not matches:
        # Probably non-SSE / non-JSON content. Hand it to the LLM raw —
        # it can read SSE-wrapped or plain text.
        return buf[-4000:]
    pieces: list[str] = []
    for raw in matches:
        try:
            # Decode JSON string escapes inside the delta.
            decoded = json.loads(f'"{raw}"')
            if isinstance(decoded, str):
                pieces.append(decoded)
        except (json.JSONDecodeError, ValueError):
            pieces.append(raw)
    text = "".join(pieces)
    # Last 4KB — soft-punt signals are in the closing portion of the
    # response, not the body. The configured local backends (DeepSeek
    # 1M, qwen3 256K+ builds) all comfortably fit this plus the
    # classifier system prompt; we don't optimize for tiny-context
    # builds.
    return text[-4000:]


# ─── streaming buffer ──────────────────────────────────────────────────────


def reset_stream(proj_sid: str) -> None:
    """Clear per-session output buffer at start of a new stream. Flag
    is NOT cleared — it survives across streams until the gate
    consumes it on the next request."""
    _OUTPUT_BUFFER.pop(proj_sid, None)


def accumulate_chunk(proj_sid: str, chunk: bytes) -> None:
    """Append a chunk to the per-session output buffer (capped tail).
    Hot-path safe — never raises, no LLM call here. Classification
    runs once at stream end via `classify_at_stream_end`."""
    if not chunk:
        return
    try:
        text = chunk.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return
    buf = _OUTPUT_BUFFER[proj_sid] + text
    if len(buf) > _BUFFER_MAX:
        buf = buf[-_BUFFER_MAX:]
    _OUTPUT_BUFFER[proj_sid] = buf


def get_flag(proj_sid: str) -> dict[str, Any] | None:
    """Return the active flag dict for a session, or None."""
    f = _SOFT_COMPLETION_FLAG.get(proj_sid)
    return f if f and f.get("active") else None


# ─── stream-end classifier (async) ─────────────────────────────────────────


@dataclass
class ClassifyDiag:
    """Outcome breakdown for a classify_at_stream_end call. Used by the
    proxy to log which path the function took (success / short-text /
    backend-error / parse-failed) so silent-None failures aren't a
    black box. Only `result` is non-None on a successful classification."""
    result: ClassifyResult | None = None
    skipped_reason: str = ""        # "short_text" / "no_buffer" / ""
    backend_error: str = ""         # str(exception) on backend failure
    backend_status: int = 0         # http status if response received
    raw_content_preview: str = ""   # first 200 chars of upstream content
    extracted_text_chars: int = 0   # how much text the classifier saw
    # Raw SSE buffer head + tail for debugging: when extraction yields
    # 0 chars but raw buffer is non-empty, we want to see what shape
    # the upstream actually sent.
    raw_buffer_chars: int = 0
    raw_buffer_head: str = ""
    raw_buffer_tail: str = ""


async def classify_at_stream_end(
        proj_sid: str,
        local_base_url: str,
        local_model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        threshold: float = 0.7,
) -> ClassifyResult | None:
    """Run the LLM behavioral classifier against the accumulated stream
    buffer for `proj_sid`. Sets the per-session flag if verdict is
    `soft_punt: true && p >= threshold`. Returns the result for trace
    logging (or None on early-skip / failure).

    For richer diagnostics use `classify_at_stream_end_diag` — this
    function is the thin wrapper that just returns the result, kept for
    backward-compatible callers / tests.

    Designed to be spawned as `asyncio.create_task(...)` from the
    stream-end finally block — it should not block the request return.
    Never raises — silent fallback (no flag set) on backend error."""
    diag = await classify_at_stream_end_diag(
        proj_sid, local_base_url, local_model,
        api_key=api_key, timeout_s=timeout_s, threshold=threshold)
    return diag.result


async def classify_at_stream_end_diag(
        proj_sid: str,
        local_base_url: str,
        local_model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        threshold: float = 0.7,
        raw_buffer: str | None = None,
) -> ClassifyDiag:
    """Same as `classify_at_stream_end` but returns a `ClassifyDiag`
    capturing why the call returned None (for logging in the proxy
    integration). Never raises.

    `raw_buffer` parameter: when supplied, classify against this
    snapshot instead of reading from the per-session module dict. The
    proxy passes its own snapshot at task-SPAWN time to avoid a race
    where the bg task was delayed by event-loop pressure (next stream
    serving) and ended up reading a buffer the next stream had already
    `reset_stream`'d. None falls back to dict read for test/dev paths."""
    diag = ClassifyDiag()

    raw = raw_buffer if raw_buffer is not None else _OUTPUT_BUFFER.get(proj_sid, "")
    diag.raw_buffer_chars = len(raw)
    if raw:
        # Capture head + tail to debug "raw non-empty but extracted 0"
        # cases. Head shows the SSE event format codex actually emits;
        # tail shows the closing portion (where soft-punts typically
        # appear). 400 chars each = ~800 chars of debug payload.
        diag.raw_buffer_head = raw[:400]
        diag.raw_buffer_tail = raw[-400:] if len(raw) > 400 else ""
    if not raw:
        diag.skipped_reason = "no_buffer"
        return diag

    text = _extract_text_from_buffer(raw)
    diag.extracted_text_chars = len(text)
    # Skip very short outputs — likely tool-call-only turns where the
    # classifier has nothing meaningful to judge.
    if len(text) < 200:
        diag.skipped_reason = "short_text"
        return diag

    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        # Same headroom as self_classify — reasoning-class local models
        # burn ~200-1500 tokens on hidden CoT before the JSON verdict.
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
        diag.backend_status = r.status_code
        r.raise_for_status()
        data = r.json()
        out = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        diag.raw_content_preview = (out or "")[:200]
    except Exception as exc:  # noqa: BLE001
        diag.backend_error = f"{type(exc).__name__}: {exc!s}"[:200]
        return diag

    result = _parse_response(out)
    if result is None:
        # Parse failed — leave the raw_content_preview populated for
        # debug. Proxy integration will log it.
        return diag

    diag.result = result
    if result.soft_punt and result.p >= threshold:
        _SOFT_COMPLETION_FLAG[proj_sid] = {
            "active": True,
            "matched_pattern": f"llm: {result.reason}"[:80],
            "p": result.p,
            "ts": time.time(),
        }
    return diag


# ─── gate injection ────────────────────────────────────────────────────────

_GATE_TEMPLATE = """\
<system-reminder>
[NOT USER INPUT — tinyctx soft-completion gate]

Your previous turn ended by soft-punting back to the user (classifier verdict: `{pattern}`) instead of completing the work and verifying outcomes. Per the user's directive, **any question to the user MUST first be vetted by advisor**.

Before processing this turn's content, you MUST do this — exact shape:

```
spawn_agent(role="advisor", task=\"\"\"
The executor wants to ask the user a meta-question instead of completing the work. Decide whether the question is genuinely needed, or whether the executor should keep working.

What I (executor) want to ask the user: <repeat verbatim the question you just asked / were about to ask>

Original user goal: <one-sentence restatement of the user's actual goal>

Progress tracker (every item enumerated):
  1. <item>  — completed (evidence: <commit hash / test name + result / file path / build log line>)
  2. <item>  — de-scoped (reason: <one-line reason>)
  …

Hard verifications I claim done: <list with concrete evidence>
Hard verifications I have NOT done: <list, especially user-stated ones like "build / test / run / verify">

Reply with EXACTLY one of:
  - ask: <one-line reason the question is genuinely needed and not answerable from the executor's own scope>
  - work: <bullet list of concrete next steps the executor should do without asking>
\"\"\")
wait_agent(...)
```

Action on advisor reply:
- If reply starts with `work:` — DO NOT ask the user. Execute the bullet items, then re-evaluate.
- If reply starts with `ask:` — proceed with the user-facing question, citing advisor in one line: "Advisor: ask — <reason>".

This gate fires once per detected soft-completion. It will not nag again until the next detection.
</system-reminder>"""


def maybe_inject_soft_completion_gate(
        body: dict[str, Any], proj_sid: str
) -> tuple[dict[str, Any], bool, str]:
    """If the soft-completion flag is set for this session, append a
    `<system-reminder>` to body.input requiring the agent to vet the
    would-be user question through advisor first. Clears the flag on
    injection (fire-once semantics).

    Returns `(body, was_injected, matched_pattern)`. The original body
    is not mutated."""
    flag = _SOFT_COMPLETION_FLAG.get(proj_sid)
    if not flag or not flag.get("active"):
        return body, False, ""
    items = body.get("input")
    if not isinstance(items, list):
        return body, False, ""
    pattern = str(flag.get("matched_pattern", "unknown"))
    new_items = list(items)
    new_items.append({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": _GATE_TEMPLATE.format(pattern=pattern),
        }],
    })
    out = dict(body)
    out["input"] = new_items
    # Consume the flag.
    _SOFT_COMPLETION_FLAG[proj_sid] = {"active": False}
    _OUTPUT_BUFFER.pop(proj_sid, None)
    return out, True, pattern


# ─── test/dev helpers ──────────────────────────────────────────────────────


def reset_state(proj_sid: str | None = None) -> None:
    """Clear per-session state. Test helper."""
    if proj_sid is None:
        _SOFT_COMPLETION_FLAG.clear()
        _OUTPUT_BUFFER.clear()
        return
    _SOFT_COMPLETION_FLAG.pop(proj_sid, None)
    _OUTPUT_BUFFER.pop(proj_sid, None)


def _set_flag_for_test(proj_sid: str, reason: str = "test", p: float = 1.0) -> None:
    """Force-set the flag for tests of the gate-injection layer
    independently of the LLM classifier."""
    _SOFT_COMPLETION_FLAG[proj_sid] = {
        "active": True, "matched_pattern": f"llm: {reason}",
        "p": p, "ts": time.time(),
    }


def state_snapshot(proj_sid: str) -> dict[str, Any]:
    """Inspect per-session sniffer state. Test/dev helper."""
    return {
        "flag": dict(_SOFT_COMPLETION_FLAG.get(proj_sid) or {}),
        "buffer_chars": len(_OUTPUT_BUFFER.get(proj_sid, "")),
    }
