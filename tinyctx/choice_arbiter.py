"""Choice Arbiter: when the model asks the user "A or B?", intercept and
auto-decide via multi-agent debate, then inject the verdict as a synthetic
user reply on the next request.

Why this exists
───────────────
soft_completion detects "asks user which option" patterns (LLM classifier,
not regex) and the stream_rewrite injects a noop tool call to keep the
turn alive. But the noop gives the model zero information — it doesn't
know which option was picked, so it asks again. Loop.

The choice arbiter closes the loop:
  1. Judge (local model): "is this a choice-ask? extract the options."
  2. Debate (3 local model personas in parallel): each role advocates for
     one option from a different angle (pragmatist / analyst / skeptic).
  3. Synthesis judge (local model): reads all 3 drafts, picks the winner.
  4. Store verdict in session_state.
  5. Next request: ChoiceArbiterGuard injects the chosen option as a
     synthetic user message at the tail of body.input.

The model sees the user "reply" with the consensus decision and continues
without re-asking. The judge step uses a model (not hardcoded keywords)
so it's language- and phrasing-agnostic.

Cost
────
Judge: one local call (~300 tokens, ~200ms).
Debate: 3 parallel local calls (~80 tokens each, ~250ms total).
Synthesis: one local call (~100 tokens, ~150ms).
Total: ~600ms — much faster than the former single-frontier path.
Falls back to single frontier advisor_decide if all local calls fail.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import session_state


# ─── SessionState namespace ──────────────────────────────────────────────

_NS = "choice_arbiter"
_K_VERDICT = "verdict"

# Compaction clears pending verdicts — post-compaction is a fresh
# conversation, and a stale verdict from before would be confusing.
session_state.register_compaction_reset(_NS, [_K_VERDICT])


# ─── Judge: local model determines "is this a choice-ask?" ───────────────

_JUDGE_SYSTEM_PROMPT = """You are a conversation pattern detector inside an agent gateway. Analyze the assistant's final response and determine whether it requires the user to make a decision before work can continue.

Detect TWO patterns — both require auto-decision:

1. EXPLICIT CHOICE: The assistant presents 2+ concrete alternatives and asks the user to pick one.
   Examples: "Should I use approach A or B?" / "Option 1: ... Option 2: ... Which do you prefer?"

2. CONTINUATION CONFIRMATION: The assistant pauses and asks whether to continue the current task,
   resume a previous plan, or proceed with the next step. Any form of:
   - "请问你是想继续这个任务，还是有别的事情..." / "是否继续？" / "继续还是停止？"
   - "Shall I proceed?" / "Do you want me to continue?" / "Should I keep going?"
   - "Want me to start on X?" / "Ready to move to the next step?"
   For these, generate standard options: ["继续当前任务", "停止"]

NOT decision points (is_choice_ask: false):
- "Let me know if you have questions" — not asking for a decision
- "Is that OK?" where the agent clearly continues anyway
- Open-ended "What would you like?" with no listed options and no prior task context
- Ambiguous — the assistant might continue on its own without input

Output EXACTLY one JSON object on a single line. No prose, no markdown:
{"is_choice_ask": true|false, "question": "<the question being asked>", "options": ["<option 1>", "<option 2>", ...], "context_summary": "<1-sentence summary>"}

For continuation confirmations, always set options to ["继续当前任务", "停止"].
For non-choice-asks: {"is_choice_ask": false, "question": "", "options": [], "context_summary": ""}"""


@dataclass
class JudgeResult:
    is_choice_ask: bool
    question: str
    options: list[str]
    context_summary: str


_JSON_RE = re.compile(
    r'\{[^{}]*"is_choice_ask"\s*:\s*(?:true|false)[^{}]*\}', re.DOTALL)
_IS_CHOICE_RE = re.compile(r'"is_choice_ask"\s*:\s*(true|false)')
_QUESTION_RE = re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"')
_OPTIONS_RE = re.compile(r'"options"\s*:\s*\[(.*?)\]', re.DOTALL)
_OPTION_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_SUMMARY_RE = re.compile(r'"context_summary"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Reasoning-class models may leak <think>…</think>
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _THINK_RE.sub("", text)


def _parse_judge_response(text: str) -> JudgeResult | None:
    """Parse the judge model's JSON response. Returns None on parse failure."""
    if not isinstance(text, str) or not text:
        return None
    text = _strip_thinking(text)
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        # Fallback: salvage individual fields
        m_is = _IS_CHOICE_RE.search(text)
        if not m_is:
            return None
        is_choice = m_is.group(1) == "true"
        m_q = _QUESTION_RE.search(text)
        question = m_q.group(1) if m_q else ""
        m_s = _SUMMARY_RE.search(text)
        summary = m_s.group(1) if m_s else ""
        options: list[str] = []
        m_opts = _OPTIONS_RE.search(text)
        if m_opts:
            options = _OPTION_ITEM_RE.findall(m_opts.group(1))
        return JudgeResult(
            is_choice_ask=is_choice,
            question=question,
            options=options,
            context_summary=summary,
        )
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    return JudgeResult(
        is_choice_ask=bool(d.get("is_choice_ask")),
        question=str(d.get("question", "") or ""),
        options=[str(o) for o in (d.get("options") or []) if o],
        context_summary=str(d.get("context_summary", "") or ""),
    )


async def judge_and_extract(
    response_text: str,
    *,
    local_base_url: str,
    local_model: str,
    api_key: str | None = None,
    timeout_s: float = 15.0,
) -> JudgeResult | None:
    """Ask the local model: is this response a choice-ask? If yes, extract
    the question, options, and context summary. Returns None on failure."""
    if not response_text.strip():
        return None

    tail = response_text[-5000:] if len(response_text) > 5000 else response_text
    user_content = f"assistant_text:\n{tail}"

    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
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
        out = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None

    return _parse_judge_response(out)


# ─── Advisor: frontier decides which option to pick ─────────────────────

_ADVISOR_SYSTEM_PROMPT = """You are a technical advisor picking between options on behalf of a user. An executor agent was about to ask the user to choose, but you're stepping in to make the call automatically.

Your goal: pick the BEST option given the user's apparent intent, project context, and general software engineering principles. Prefer options that:
- Move the work forward (action over deliberation)
- Are safer / more reversible
- Follow established conventions in the codebase
- Match what a senior engineer would choose

Respond in under 80 words with this exact format:
CHOSEN: <the selected option, verbatim or slightly clarified>
REASON: <one sentence explaining why>
FALLBACK: <second-best option — only include if it's genuinely close>"""


async def advisor_decide(
    judge_result: JudgeResult,
    *,
    frontier_base_url: str,
    frontier_model: str,
    api_key: str | None = None,
    timeout_s: float = 60.0,
) -> str | None:
    """Call the frontier model to pick between options. Returns the advisor's
    verdict text, or None on failure."""
    if not judge_result.options:
        return None

    options_text = "\n".join(
        f"  {i+1}. {o}" for i, o in enumerate(judge_result.options))
    user_prompt = (
        f"Question: {judge_result.question}\n"
        f"Context: {judge_result.context_summary}\n"
        f"Options:\n{options_text}"
    )

    payload: dict[str, Any] = {
        "model": frontier_model,
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
    if api_key:
        headers["Authorization"] = (
            api_key if api_key.lower().startswith(("bearer ", "basic "))
            else f"Bearer {api_key}"
        )

    url = frontier_base_url.rstrip("/") + "/responses"
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s)) as client:
            async with client.stream(
                    "POST", url, json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    return None
                text = ""
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    line = (raw.decode("utf-8", "replace")
                            if isinstance(raw, bytes) else raw)
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
                            text += d
                    elif et == "response.output_text.done":
                        t = evt.get("text")
                        if isinstance(t, str) and t:
                            text = t
                return text.strip() or None
    except Exception:
        return None


# ─── Multi-agent debate: 3 personas + synthesis judge ───────────────────

_DEBATE_PERSONAS: list[tuple[str, str]] = [
    (
        "pragmatist",
        "You are the Pragmatist in a quick 3-way option debate. "
        "Pick the option that moves the current task forward the fastest. "
        "Bias: action over deliberation, fewest steps to goal. "
        "Reply in EXACTLY this format:\nPICK: <option verbatim>\nREASON: <≤10 words>",
    ),
    (
        "analyst",
        "You are the Analyst in a quick 3-way option debate. "
        "Pick the option that best matches the user's stated goal and task context. "
        "Bias: intent alignment, what the user actually wanted when they started. "
        "Reply in EXACTLY this format:\nPICK: <option verbatim>\nREASON: <≤10 words>",
    ),
    (
        "skeptic",
        "You are the Skeptic in a quick 3-way option debate. "
        "Pick the safest and most reversible option. "
        "Bias: caution, prefer options that preserve future choices. "
        "Reply in EXACTLY this format:\nPICK: <option verbatim>\nREASON: <≤10 words>",
    ),
]

_SYNTHESIS_SYSTEM = (
    "You are the final judge in a 3-persona debate about which option to choose. "
    "Rules: if 2+ agree → pick that option. If all disagree → pick Analyst's pick. "
    "Output ONLY one line: CHOSEN: <option verbatim from the options list>"
)

_PICK_RE = re.compile(r"PICK:\s*(.+)", re.IGNORECASE)
_CHOSEN_RE = re.compile(r"CHOSEN:\s*(.+)", re.IGNORECASE)


async def _local_role_call(
    client: httpx.AsyncClient,
    local_base_url: str,
    local_model: str,
    api_key: str | None,
    persona_name: str,
    system_prompt: str,
    question: str,
    options: list[str],
    context: str,
) -> tuple[str, str]:
    """Single local-model persona call. Returns (persona_name, pick_text)."""
    options_text = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
    user_msg = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Options:\n{options_text}"
    )
    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 80,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = local_base_url.rstrip("/") + "/chat/completions"
    try:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        text = (r.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "") or "")
        return persona_name, text.strip()
    except Exception:
        return persona_name, ""


async def debate_decide(
    judge_result: JudgeResult,
    *,
    local_base_url: str,
    local_model: str,
    local_api_key: str | None = None,
    debate_timeout_s: float = 15.0,
    synthesis_timeout_s: float = 10.0,
) -> str | None:
    """Run 3 local-model personas in parallel, then a synthesis judge.
    Returns the chosen option text, or None if all calls fail."""
    if not judge_result.options:
        return None
    question = judge_result.question or "Which option should be chosen?"
    context = judge_result.context_summary or ""
    options = judge_result.options

    # Phase 1: 3 personas in parallel
    async with httpx.AsyncClient(
            timeout=httpx.Timeout(debate_timeout_s)) as client:
        tasks = [
            _local_role_call(
                client, local_base_url, local_model, local_api_key,
                name, prompt, question, options, context,
            )
            for name, prompt in _DEBATE_PERSONAS
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    picks: dict[str, str] = {}  # persona_name → full response text
    for res in raw_results:
        if isinstance(res, Exception) or not res:
            continue
        pname, text = res
        if text:
            picks[pname] = text

    if not picks:
        return None

    # Phase 2: synthesis judge reads all 3 picks
    options_text = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
    debate_summary = "".join(
        f"[{n.upper()}]\n{t}\n\n" for n, t in picks.items()
    )
    synthesis_user = (
        f"Original question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Debate:\n{debate_summary}"
    )
    payload = {
        "model": local_model,
        "messages": [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": synthesis_user},
        ],
        "temperature": 0.0,
        "max_tokens": 60,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if local_api_key:
        headers["Authorization"] = f"Bearer {local_api_key}"
    url = local_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(synthesis_timeout_s)) as client:
            r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        text = (r.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "") or "")
        m = _CHOSEN_RE.search(text)
        if m:
            return m.group(1).strip()
        return text.strip() or None
    except Exception:
        # Synthesis failed — fall back to majority vote from persona picks
        option_votes: dict[str, int] = {}
        for text in picks.values():
            m = _PICK_RE.search(text)
            if m:
                pick = m.group(1).strip()
                option_votes[pick] = option_votes.get(pick, 0) + 1
        if option_votes:
            return max(option_votes, key=lambda k: option_votes[k])
        # Last resort: return analyst's pick or first option
        for preferred in ("analyst", "pragmatist", "skeptic"):
            if preferred in picks:
                m = _PICK_RE.search(picks[preferred])
                if m:
                    return m.group(1).strip()
        return options[0] if options else None


# ─── Verdict storage (session_state) ────────────────────────────────────

@dataclass
class Verdict:
    advisor_choice: str
    question: str
    options: list[str]
    ts: float


def store_verdict(conv_sid: str, verdict: Verdict) -> None:
    session_state.set(conv_sid, _NS, _K_VERDICT, {
        "advisor_choice": verdict.advisor_choice,
        "question": verdict.question,
        "options": verdict.options,
        "ts": verdict.ts,
    })


def consume_verdict(conv_sid: str) -> Verdict | None:
    raw = session_state.consume(conv_sid, _NS, _K_VERDICT)
    if raw is None:
        return None
    return Verdict(
        advisor_choice=str(raw.get("advisor_choice", "") or ""),
        question=str(raw.get("question", "") or ""),
        options=[str(o) for o in (raw.get("options") or [])],
        ts=float(raw.get("ts", 0) or 0),
    )


# ─── Injection: synthetic user message ──────────────────────────────────

def inject_verdict_into_body(
    body: dict[str, Any], verdict: Verdict
) -> tuple[dict[str, Any], bool]:
    """Inject the advisor's choice as a synthetic user message at the tail
    of body.input. Returns (new_body, was_injected). The original body is
    not mutated."""
    items = body.get("input")
    if not isinstance(items, list):
        return body, False

    text = (
        f"[tinyctx choice arbiter — the advisor picked an option on your "
        f"behalf. Treat this as the user's decision and act on it "
        f"immediately without re-asking.]\n\n"
        f"Question: {verdict.question}\n"
        f"Advisor's choice: {verdict.advisor_choice}"
    )
    new_items = list(items)
    new_items.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    })
    out = dict(body)
    out["input"] = new_items
    return out, True


# ─── Top-level intercept (called from proxy stream-rewrite site) ────────

async def intercept(
    response_text: str,
    *,
    conv_sid: str,
    local_base_url: str,
    local_model: str,
    local_api_key: str | None = None,
    frontier_base_url: str,
    frontier_model: str,
    frontier_api_key: str | None = None,
    judge_timeout_s: float = 15.0,
    advisor_timeout_s: float = 60.0,
) -> Verdict | None:
    """Run the full choice-arbiter pipeline: judge → advisor → store.
    Returns the Verdict if a choice-ask was detected AND advisor returned
    a decision. Returns None if not a choice-ask, or on any failure.

    Called from the stream-rewrite section of _stream_proxy. The caller
    should still inject the normal noop-continue to keep the turn alive;
    the verdict will be consumed by ChoiceArbiterGuard on the next request.
    """
    judge = await judge_and_extract(
        response_text,
        local_base_url=local_base_url,
        local_model=local_model,
        api_key=local_api_key,
        timeout_s=judge_timeout_s,
    )
    if judge is None or not judge.is_choice_ask:
        return None
    if not judge.options or len(judge.options) < 2:
        return None

    # Primary: 3-persona debate (local models, parallel, ~600ms)
    choice = await debate_decide(
        judge,
        local_base_url=local_base_url,
        local_model=local_model,
        local_api_key=local_api_key,
    )
    # Fallback: single frontier advisor
    if not choice:
        choice = await advisor_decide(
            judge,
            frontier_base_url=frontier_base_url,
            frontier_model=frontier_model,
            api_key=frontier_api_key,
            timeout_s=advisor_timeout_s,
        )
    if not choice:
        return None

    verdict = Verdict(
        advisor_choice=choice,
        question=judge.question,
        options=judge.options,
        ts=time.time(),
    )
    store_verdict(conv_sid, verdict)
    return verdict


# ─── test helpers ───────────────────────────────────────────────────────

def reset_state(conv_sid: str | None = None) -> None:
    if conv_sid is None:
        for sid in list(session_state.keys_with_prefix("")):
            session_state.clear(sid, _NS, _K_VERDICT)
        return
    session_state.clear(conv_sid, _NS, _K_VERDICT)
