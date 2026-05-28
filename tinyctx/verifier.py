"""Output-quality verifier — post-stream LLM-as-a-Verifier quality gate.

Why this exists
───────────────
tinyctx knows when a local-model response is too short
(empty_response_guard), when the agent is soft-punting to the user
(soft_completion), and when the stream stalls (stall_watchdog). But it
has NO signal for "the local model produced a plausible-length response
that is subtly wrong." This module fills that gap.

Design
──────
Inspired by the llm-as-a-verifier paper's criteria-decomposition
approach, adapted for online (single-trajectory) use:

  - 3 criteria: task_completion, output_quality, execution_evidence
  - Each scored 1–5 by the LOCAL model itself (text output, no logprobs)
  - Total < threshold (default 8/15) → force frontier on next request
  - No round-robin tournament (online, not batch)
  - No K>1 repeated verification in v1 (K=1; borderline re-check is
    future work)

Cost / latency
──────────────
One extra local-model call per completed LOCAL-routed turn (~800-token
prompt + ~80-token JSON output ≈ 250ms on DeepSeek-class). Runs
ASYNCHRONOUSLY via asyncio.create_task — does NOT add to user-perceived
latency. Skipped for tool_calls turns (most agent turns), frontier
responses, and short outputs.

Integration
───────────
Reuses soft_completion's buffer (_OUTPUT_BUFFER, _extract_text_from_buffer,
_extract_finish_reason, extract_user_goal, extract_tool_summary) — no
duplicate buffering. Flag consumed by VerifierGate (priority 35) in the
guard pipeline on the next request.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import httpx

# ─── constants ───────────────────────────────────────────────────────────────

_NS = "verifier"
_K_FLAG = "flag"

_DEFAULT_THRESHOLD = 8
_DEFAULT_TIMEOUT_S = 30.0
_SHORT_TEXT_FLOOR = 100

# ─── per-session flag ────────────────────────────────────────────────────────

_VERIFIER_FLAG: dict[str, dict] = {}


def get_flag(proj_sid: str) -> dict | None:
    f = _VERIFIER_FLAG.get(proj_sid)
    return f if f and f.get("active") else None


def consume_flag(proj_sid: str) -> dict | None:
    f = _VERIFIER_FLAG.pop(proj_sid, None)
    return f if f and f.get("active") else None


# ─── data structures ─────────────────────────────────────────────────────────


@dataclass
class VerdictCriteria:
    task_completion: int
    output_quality: int
    execution_evidence: int

    @property
    def total(self) -> int:
        return self.task_completion + self.output_quality + self.execution_evidence


@dataclass
class VerifyResult:
    criteria: VerdictCriteria
    passed: bool
    reason: str


@dataclass
class VerifyDiag:
    result: VerifyResult | None = None
    skipped_reason: str = ""
    backend_error: str = ""
    backend_status: int = 0
    raw_response_preview: str = ""
    extracted_text_chars: int = 0


# ─── verifier system prompt ──────────────────────────────────────────────────

_VERIFIER_SYSTEM_PROMPT = """You are an output-quality auditor inside an AI coding agent gateway. An agent just finished a turn using a LOCAL model. Your job: score the agent's output on THREE independent criteria, then return a JSON verdict.

You will receive:
  - finish_reason — how the stream ended (stop / length / tool_calls)
  - user_goal — what the user asked for
  - tool_summary — count + names of tool calls the agent made
  - assistant_text — the agent's final response (last ~4KB of text)

Score EACH criterion on a 1–5 scale:
  1 = completely failed / missing
  2 = mostly wrong / major gaps
  3 = mixed — some correct, some wrong or missing
  4 = mostly correct, minor issues
  5 = fully correct, complete, well-evidenced

═══ Criterion 1: task_completion ═══
Did the agent actually deliver what the user asked for? Compare the
assistant_text against the user_goal:
  - 5: all stated requirements met; output matches what was requested
  - 4: core goal met, one minor requirement missing or incomplete
  - 3: partially met — some parts done, some not attempted or wrong
  - 2: mostly missed the goal; output solves a different problem
  - 1: completely off-target; response irrelevant to the request

═══ Criterion 2: output_quality ═══
Is the output CORRECT and well-formed for its type?
  - 5: output is correct, well-structured, properly formatted
  - 4: minor formatting issues or one small technical error
  - 3: several small errors or structural problems
  - 2: significant errors that would cause problems if used
  - 1: output is broken — syntax errors, contradictions, unusable

For code: check syntax validity, API correctness, edge-case handling.
For text/plans: check logical coherence, factual plausibility, completeness.
For file modifications: check that paths, content, and intent align.

═══ Criterion 3: execution_evidence ═══
Are the agent's claims backed by concrete tool-call results?
  - 5: every claim is backed by a matching tool output / commit / test
  - 4: most claims backed; one unsupported assertion
  - 3: some evidence present, but gaps where key claims lack backing
  - 2: agent makes claims but tool_summary shows no relevant execution
  - 1: agent claims success/action but zero tool evidence in summary

If tool_summary is "no_tool_calls" and the agent is giving factual
answers (not executing code), score 4 (reasonable for Q&A without tools).

═══ Output format ═══
Return EXACTLY one JSON object on a single line. No markdown, no prose:
{"task_completion": N, "output_quality": N, "execution_evidence": N, "reason": "<≤12 words>"}

The reason field should summarize the MAIN factor behind the scores in ≤12 words.
Examples:
  "all 3 requirements met, tests passed"
  "missing file write, agent only talked about it"
  "code has syntax error in line 12"
  "correct analysis but no verification run"
  "agent answered a different question than asked"
  "tool evidence fully matches claims"

Begin your analysis now."""


# ─── JSON parser ─────────────────────────────────────────────────────────────

try:
    from .soft_completion import _strip_thinking as _strip_think
except ImportError:
    _THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

    def _strip_think(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return _THINK_RE.sub("", text)


_JSON_RE = re.compile(
    r'\{(?:[^{}]|\{[^{}]*\})*"task_completion"\s*:\s*\d+(?:[^{}]|\{[^{}]*\})*'
    r'"output_quality"\s*:\s*\d+(?:[^{}]|\{[^{}]*\})*'
    r'"execution_evidence"\s*:\s*\d+(?:[^{}]|\{[^{}]*\})*\}',
    re.DOTALL,
)
_TC_RE = re.compile(r'"task_completion"\s*:\s*(\d+)')
_OQ_RE = re.compile(r'"output_quality"\s*:\s*(\d+)')
_EE_RE = re.compile(r'"execution_evidence"\s*:\s*(\d+)')
_REASON_V_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def _clamp(v: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, v))


def _parse_verdict(text: str) -> VerdictCriteria | None:
    if not isinstance(text, str) or not text:
        return None
    text = _strip_think(text)
    if not text:
        return None

    m = _JSON_RE.search(text)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            tc = _clamp(int(d.get("task_completion", 3)))
            oq = _clamp(int(d.get("output_quality", 3)))
            ee = _clamp(int(d.get("execution_evidence", 3)))
            return VerdictCriteria(
                task_completion=tc, output_quality=oq, execution_evidence=ee)

    # Fallback: salvage individual fields from truncated text
    m_tc = _TC_RE.search(text)
    m_oq = _OQ_RE.search(text)
    m_ee = _EE_RE.search(text)
    if m_tc and m_oq and m_ee:
        try:
            return VerdictCriteria(
                task_completion=_clamp(int(m_tc.group(1))),
                output_quality=_clamp(int(m_oq.group(1))),
                execution_evidence=_clamp(int(m_ee.group(1))),
            )
        except (ValueError, TypeError):
            pass
    return None


def _extract_reason(text: str) -> str:
    m = _REASON_V_RE.search(text)
    return (m.group(1) if m else "[parse failed]")[:200]


# ─── main classifier ─────────────────────────────────────────────────────────


async def verify_at_stream_end(
    proj_sid: str,
    local_base_url: str,
    local_model: str,
    *,
    api_key: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    threshold: int = _DEFAULT_THRESHOLD,
    raw_buffer: str | None = None,
    user_goal: str = "",
    tool_summary: str = "",
    conv_sid: str | None = None,
    current_route: str = "",
) -> VerifyDiag:
    """Run the output-quality verifier against the accumulated stream
    buffer. Sets a per-session flag when total < threshold so that
    VerifierGate forces the next request to frontier.

    Never raises — silent fallback (no flag set) on error."""
    diag = VerifyDiag()

    if raw_buffer is not None:
        raw = raw_buffer
    else:
        from .soft_completion import _OUTPUT_BUFFER
        raw = _OUTPUT_BUFFER.get(proj_sid, "")
    if not raw:
        diag.skipped_reason = "no_buffer"
        return diag

    from .soft_completion import _extract_finish_reason
    finish = _extract_finish_reason(raw) or ""
    if finish == "tool_calls":
        diag.skipped_reason = "tool_calls_finish"
        return diag

    from .soft_completion import _extract_text_from_buffer
    text = _extract_text_from_buffer(raw)
    diag.extracted_text_chars = len(text)
    if len(text) < _SHORT_TEXT_FLOOR:
        diag.skipped_reason = "short_text"
        return diag

    sections = [
        f"finish_reason: {finish or 'unknown'}",
        f"text_chars: {len(text)}",
        "",
        "user_goal:",
        (user_goal or "(not available)"),
        "",
        "tool_summary:",
        (tool_summary or "no_tool_calls"),
        "",
        "assistant_text:",
        text,
    ]
    user_content = "\n".join(sections)

    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        # Reasoning-class local models (DeepSeek-v4-pro) burn tokens on
        # hidden CoT. 2048 + reasoning_effort=low ensures enough headroom
        # for the JSON verdict after the think budget. Same params as
        # soft_completion.classify_at_stream_end_diag.
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
        msg = data.get("choices", [{}])[0].get("message", {})
        out = msg.get("content", "") or ""
        # Reasoning-class models (DeepSeek-v4-pro) may emit the verdict
        # in `reasoning_content` when content is empty. See production
        # log: all 13 parse_failed had empty raw_response_preview.
        if not out.strip():
            rc = msg.get("reasoning_content", "") or ""
            if rc.strip():
                out = rc
        diag.raw_response_preview = (out or "")[:200]
    except Exception as exc:  # noqa: BLE001
        diag.backend_error = f"{type(exc).__name__}: {exc!s}"[:200]
        return diag

    criteria = _parse_verdict(out)
    if criteria is None:
        return diag

    total = criteria.total
    reason = _extract_reason(out)
    passed = total >= threshold

    diag.result = VerifyResult(criteria=criteria, passed=passed, reason=reason)

    if not passed:
        _VERIFIER_FLAG[proj_sid] = {
            "active": True,
            "total": total,
            "reason": reason,
            "criteria": {
                "task_completion": criteria.task_completion,
                "output_quality": criteria.output_quality,
                "execution_evidence": criteria.execution_evidence,
            },
            "ts": time.time(),
        }

    return diag


# ─── test / dev helpers ──────────────────────────────────────────────────────


def reset_state(proj_sid: str | None = None) -> None:
    if proj_sid is None:
        _VERIFIER_FLAG.clear()
        return
    _VERIFIER_FLAG.pop(proj_sid, None)


def _set_flag_for_test(proj_sid: str, total: int = 4, reason: str = "test",
                       task_completion: int = 2, output_quality: int = 1,
                       execution_evidence: int = 1) -> None:
    _VERIFIER_FLAG[proj_sid] = {
        "active": True,
        "total": total,
        "reason": reason,
        "criteria": {
            "task_completion": task_completion,
            "output_quality": output_quality,
            "execution_evidence": execution_evidence,
        },
        "ts": time.time(),
    }


def state_snapshot(proj_sid: str) -> dict:
    f = _VERIFIER_FLAG.get(proj_sid) or {}
    return {"flag": dict(f), "flag_active": bool(f and f.get("active"))}
