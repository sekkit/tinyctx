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
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from . import session_state


# ─── SessionState namespace + compaction reset policy ─────────────────────

_NS = "soft_completion"
_K_FLAG = "flag"
_K_OUTPUT_BUFFER = "output_buffer"

# Compaction clears the stale stream buffer — post-compaction is a fresh
# request and any buffered SSE fragment from before is no longer relevant.
# The flag is one-shot (consumed on the next gate-check), so it doesn't
# need explicit compaction-reset; leaving it set across a compaction is
# fine — the next request still consumes it once.
session_state.register_compaction_reset(_NS, [_K_OUTPUT_BUFFER])


# ─── legacy dict views ────────────────────────────────────────────────────


class _FlagDictView:
    """Read/write proxy for the legacy `_SOFT_COMPLETION_FLAG` attribute.
    `view[proj_sid]` returns the underlying flag dict by reference (so
    callers / tests that read individual fields like `flag["active"]`
    still work). Missing entries return `None` from `.get(...)` to match
    the original `dict.get` semantics — note the underlying type was
    `defaultdict(dict)` but every reader uses `.get(...)` so `None` is
    the observable shape."""

    __slots__ = ()

    def __iter__(self) -> Iterator[str]:
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if _K_FLAG in by_ns.get(_NS, {}):
                yield sid

    def keys(self) -> list[str]:
        return list(iter(self))

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        snap = session_state.snapshot()
        out: list[tuple[str, dict[str, Any]]] = []
        for sid, by_ns in snap.items():
            val = by_ns.get(_NS, {}).get(_K_FLAG)
            if val is not None:
                out.append((sid, val))
        return out

    def values(self) -> list[dict[str, Any]]:
        return [v for _, v in self.items()]

    def __contains__(self, sid: Any) -> bool:
        return session_state.get(sid, _NS, _K_FLAG) is not None

    def __getitem__(self, sid: Any) -> dict[str, Any]:
        val = session_state.get(sid, _NS, _K_FLAG)
        if val is None:
            # Match defaultdict(dict) semantics: missing → new empty dict
            # stored back, returned by reference. _set_flag_for_test and
            # the classifier paths use direct assignment so the shape on
            # read is consistent.
            empty: dict[str, Any] = {}
            session_state.set(sid, _NS, _K_FLAG, empty)
            return empty
        return val

    def __setitem__(self, sid: Any, value: dict[str, Any]) -> None:
        session_state.set(sid, _NS, _K_FLAG, value)

    def __delitem__(self, sid: Any) -> None:
        session_state.clear(sid, _NS, _K_FLAG)

    def get(self, sid: Any, default: Any = None) -> Any:
        val = session_state.get(sid, _NS, _K_FLAG)
        return default if val is None else val

    def pop(self, sid: Any, default: Any = None) -> Any:
        existing = session_state.consume(sid, _NS, _K_FLAG)
        return existing if existing is not None else default

    def clear(self) -> None:
        for sid in list(iter(self)):
            session_state.clear(sid, _NS, _K_FLAG)


class _OutputBufferDictView:
    """Read/write proxy for the legacy `_OUTPUT_BUFFER` attribute. The
    buffer is a STRING per proj_sid (ring-buffer'd by the caller via the
    read-then-set-truncated pattern in `accumulate_chunk`). Missing reads
    return `""` to match the original `defaultdict(str)` semantics."""

    __slots__ = ()

    def __iter__(self) -> Iterator[str]:
        snap = session_state.snapshot()
        for sid, by_ns in snap.items():
            if _K_OUTPUT_BUFFER in by_ns.get(_NS, {}):
                yield sid

    def keys(self) -> list[str]:
        return list(iter(self))

    def items(self) -> list[tuple[str, str]]:
        snap = session_state.snapshot()
        out: list[tuple[str, str]] = []
        for sid, by_ns in snap.items():
            val = by_ns.get(_NS, {}).get(_K_OUTPUT_BUFFER)
            if val is not None:
                out.append((sid, val))
        return out

    def values(self) -> list[str]:
        return [v for _, v in self.items()]

    def __contains__(self, sid: Any) -> bool:
        return session_state.get(sid, _NS, _K_OUTPUT_BUFFER) is not None

    def __getitem__(self, sid: Any) -> str:
        val = session_state.get(sid, _NS, _K_OUTPUT_BUFFER)
        return "" if val is None else val

    def __setitem__(self, sid: Any, value: str) -> None:
        session_state.set(sid, _NS, _K_OUTPUT_BUFFER, value)

    def __delitem__(self, sid: Any) -> None:
        session_state.clear(sid, _NS, _K_OUTPUT_BUFFER)

    def get(self, sid: Any, default: Any = "") -> str:
        val = session_state.get(sid, _NS, _K_OUTPUT_BUFFER)
        return default if val is None else val

    def pop(self, sid: Any, default: Any = None) -> Any:
        existing = session_state.consume(sid, _NS, _K_OUTPUT_BUFFER)
        return existing if existing is not None else default

    def clear(self) -> None:
        for sid in list(iter(self)):
            session_state.clear(sid, _NS, _K_OUTPUT_BUFFER)


# ─── per-session state ─────────────────────────────────────────────────────
# Module-level shims that delegate to SessionState. Public-API-identical
# to the previous `defaultdict(dict)` / `defaultdict(str)` storage so
# tests and proxy call sites keep working unchanged.

_SOFT_COMPLETION_FLAG = _FlagDictView()
_OUTPUT_BUFFER = _OutputBufferDictView()
# Cap raw SSE buffer at 64KB. After SSE+JSON-overhead extraction this
# yields ~30-50KB of actual text — comfortably above the 4KB tail we
# feed the classifier.
_BUFFER_MAX = 65536


# ─── LLM behavioral classifier ─────────────────────────────────────────────

_CLASSIFIER_SYSTEM_PROMPT = """You are a completion auditor inside an agent gateway. The agent just finished a turn. Decide whether the agent has ACTUALLY COMPLETED the user's stated goal, or whether it stopped while material work the user expects is still undone.

You will receive structured context:
  - `user_goal` — what the user asked for (their most recent or canonical request).
  - `progress_tracker` — items the agent itself enumerated as the task plan (from update_plan / TodoWrite). Each item has a status. May be empty if the agent never set up a tracker.
  - `tool_summary` — count + last names of function_calls the agent made this session (commits, file edits, tests, builds).
  - `finish_reason` — how the stream ended: `tool_calls` (still acting), `stop` (response ended), `length` (truncated).
  - `assistant_text` — the visible text of the agent's final response.

Output EXACTLY one JSON object on a single line. No prose, no markdown:
{"soft_punt": true|false, "p": 0.0-1.0, "reason": "<≤10 words>"}

═══ Quick short-circuits (no semantic analysis needed) ═══

  - `finish_reason=tool_calls` → soft_punt:false p=0.95 (agent is acting, not stopping).
  - `assistant_text` < 50 chars AND `finish_reason=stop` → soft_punt:false p=0.9 (brief confirmation between tool calls, low signal).

═══ Core question (when finish_reason=stop and assistant_text is substantive) ═══

Compare the agent's claimed completion vs. what the user asked for:

**SOFT-PUNT (true, p ≥ 0.7)** — the agent stopped while material work for the user's goal is undone:
  • Meta-question to user — "what next / which option / shall I" — handing decision back instead of acting.
  • Options-and-wait — lists 2+ alternatives and stops.
  • Premature claim of done — says "all done / final summary" but the progress_tracker has unchecked items, OR the user's stated verification (build/test/deploy/run) wasn't executed.
  • Plan without action — states a multi-step plan but `finish_reason=stop` and tool_summary shows no recent tool calls executing the plan.
  • **Implicit de-scope** — declares partial completion while saying remaining work is "follow-up / for later / when needed / can be done if you want / optional / left as exercise". Agent is unilaterally trimming scope without the user's agreement.
  • Soft hand-back — "let me know if you want me to ...", "happy to expand on ...", "feel free to ask if ..." — these signal "I'm done, you decide" without explicit "what next?" framing.
  • Natural-stopping framing — "I think we're at a good stopping point", "this seems like a reasonable place to pause" — agent self-deciding to stop.

**NOT SOFT-PUNT (false, p ≥ 0.7)** — the agent stopped legitimately:
  • Deliverable IS the response — user asked a question, agent answered substantively. No further work implied.
  • Hard failure with a specific blocker — build failed / missing credential / ambiguous spec — and the agent reports concretely what the user must do.
  • Load-bearing clarification — the question genuinely cannot be answered from the executor's own scope (irreversible-action confirmation, conflicting requirements).
  • All progress_tracker items are completed-with-evidence (commits / test pass / build success), AND user-stated verification was executed.
  • Mid-progress status note — short text between tool calls, agent is clearly going to keep working.

═══ Calibration ═══

The cost asymmetry: a false positive runs one extra advisor call (~5K tokens). A false negative makes the user see an incomplete task and manually nudge — that's the original problem we're fixing. Lean toward true on borderline cases.

Use ALL signals jointly: don't just judge `assistant_text` in isolation. If the user asked for X and X has explicit verification step Y, and tool_summary doesn't show Y was run, and assistant_text says "all done" — that's a punt regardless of how confidently the agent phrased it. Conversely, if tracker is empty and tool_summary shows recent edits + commits, and assistant_text is a brief status, it's not a punt.

Reason field: ≤10 words, noun-phrase fragment. Be specific about WHICH signal triggered the verdict.

Examples:
  "tracker has 4 unchecked, agent declared done"      ← punt:true
  "verification step not in tool_summary"             ← punt:true
  "remaining work declared follow-up by agent"        ← punt:true
  "asks user which option"                            ← punt:true
  "plan stated, no tool calls in summary"             ← punt:true
  "user asked Q, agent gave full A"                   ← punt:false
  "all 5 tracker items completed-with-evidence"       ← punt:false
  "build failed, agent named specific blocker"        ← punt:false (legit)
  "tool_calls finish, mid-progress"                   ← punt:false (short-circuit)"""


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
            # Why: classifier emitted a non-numeric `p` — keep the
            # default of 0.5 (uncertain) and let downstream policy
            # decide based on `soft_punt` alone.
            pass
    m_r = _REASON_RE.search(text)
    reason = (m_r.group(1) if m_r else "[salvaged]")[:200]
    return ClassifyResult(soft_punt=sp, p=p, reason=reason)


# ─── finish_reason extraction ──────────────────────────────────────────────
# Both Responses-API and Chat-Completions emit `"finish_reason":"<value>"` in
# the last few SSE events of a stream. We scan from the buffer TAIL because
# the field appears once near the end. Values: `"stop"` (clean text end),
# `"tool_calls"` (agent emitted function call(s) — still progressing),
# `"length"` (max_tokens hit — truncated), or null/missing for in-progress
# events. The last non-null value wins.

_FINISH_REASON_RE = re.compile(
    r'"finish_reason"\s*:\s*"([^"]+)"')


def _extract_finish_reason(buf: str) -> str | None:
    """Find the LAST non-null `finish_reason` value in the buffer. None
    if no terminal finish_reason event was captured (e.g., truncated
    buffer, stream still in flight, or the event hadn't reached the
    captured tail yet). Most streams emit finish_reason in the very
    last `data:` event, so the tail check is reliable."""
    if not buf:
        return None
    matches = _FINISH_REASON_RE.findall(buf)
    if not matches:
        return None
    # Last non-empty/non-null match wins
    for v in reversed(matches):
        if v and v != "null":
            return v
    return None


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


# ─── context extraction (user goal / progress tracker / tool summary) ─────
# Codex sends the entire conversation history in body.input each turn, so
# we can mine it for the completion signals the LLM classifier needs.


def extract_user_goal(body_input: list[Any] | None, max_chars: int = 1500) -> str:
    """Pull the user's most recent message from body.input. This is the
    "did the agent actually finish what was asked" anchor for the
    semantic classifier. Returns empty string if no user message found.

    Falls back to scanning ALL user messages for the LAST one (most
    recent intent). Caps at `max_chars` so the classifier prompt
    stays bounded."""
    if not isinstance(body_input, list):
        return ""
    last_user = ""
    for it in reversed(body_input):
        if not isinstance(it, dict):
            continue
        role = it.get("role")
        t = it.get("type")
        if role != "user" and not (t == "message" and role == "user"):
            continue
        content = it.get("content")
        if isinstance(content, str):
            last_user = content
            break
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("text", "input_text", "output_text"):
                    txt = c.get("text") or ""
                    if isinstance(txt, str):
                        parts.append(txt)
            joined = "\n".join(p for p in parts if p)
            if joined:
                last_user = joined
                break
    return last_user[:max_chars].strip()


def extract_progress_tracker(body_input: list[Any] | None,
                              max_chars: int = 1500) -> str:
    """Mine the conversation for `update_plan` / `TodoWrite` tool-call
    outputs. Returns a compact text representation of the latest plan
    state (each item with status), or empty string if no tracker found.

    Codex codex tracks progress via `update_plan` MCP-style calls. The
    tool_call_output for these contains the plan items. We scan from
    the END of body.input backward so we get the most recent state."""
    if not isinstance(body_input, list):
        return ""
    # Walk backwards looking for the latest update_plan tool call result
    for it in reversed(body_input):
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        # Codex's two formats: function_call_output (paired with a
        # function_call) or a direct call/result item.
        name = it.get("name", "") or ""
        if t == "function_call" and name in ("update_plan", "TodoWrite"):
            # The arguments string contains the plan items
            args_str = it.get("arguments", "") or ""
            if isinstance(args_str, str) and args_str.strip():
                rendered = _render_plan_args(args_str, max_chars)
                if rendered:
                    return rendered
        if t == "function_call_output":
            call_id = it.get("call_id", "")
            if not call_id:
                continue
            # Find the paired function_call to check its name
            for prev in reversed(body_input):
                if not isinstance(prev, dict):
                    continue
                if (prev.get("type") == "function_call"
                        and prev.get("call_id") == call_id):
                    pname = prev.get("name", "")
                    if pname in ("update_plan", "TodoWrite"):
                        args_str = prev.get("arguments", "") or ""
                        rendered = _render_plan_args(args_str, max_chars)
                        if rendered:
                            return rendered
                    break
    return ""


def _render_plan_args(args_str: str, max_chars: int) -> str:
    """Parse update_plan's arguments JSON and render as a compact
    per-item bullet list. Falls back to truncated raw args on parse
    failure."""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        return args_str[:max_chars]
    if not isinstance(args, dict):
        return args_str[:max_chars]
    # codex update_plan typically: {"explanation": "...", "plan": [{"step": "...", "status": "pending|in_progress|completed"}, ...]}
    plan = args.get("plan") or args.get("todos") or args.get("items") or []
    if not isinstance(plan, list):
        return args_str[:max_chars]
    lines: list[str] = []
    for i, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            continue
        step = (item.get("step") or item.get("content")
                or item.get("text") or item.get("description") or "?")
        status = (item.get("status") or item.get("state") or "?")
        lines.append(f"  {i}. [{status}] {str(step)[:200]}")
    if not lines:
        return args_str[:max_chars]
    rendered = "\n".join(lines)
    return rendered[:max_chars]


def extract_tool_summary(body_input: list[Any] | None,
                          last_n: int = 12,
                          max_chars: int = 800) -> str:
    """Summarize recent function_call activity in the conversation.
    Returns "no_tool_calls" when nothing found, else a count + the
    last `last_n` tool names. The classifier uses this to verify
    whether the agent actually executed work or just talked about it."""
    if not isinstance(body_input, list):
        return "no_tool_calls"
    names: list[str] = []
    for it in body_input:
        if not isinstance(it, dict):
            continue
        if it.get("type") != "function_call":
            continue
        n = it.get("name", "") or "?"
        names.append(str(n))
    if not names:
        return "no_tool_calls"
    total = len(names)
    tail = names[-last_n:]
    summary = f"total_tool_calls={total}; last={tail}"
    return summary[:max_chars]


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
    except Exception:  # noqa: BLE001 — never break the streaming hot path
        # Why: decode on a hot streaming path must not raise. `errors=
        # "ignore"` already covers UTF-8 issues; this guards against
        # exotic input types passed by tests/mocks. Drop the chunk.
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
    backend-error / parse-failed / finish-tool-calls / finish-stop-short)
    so silent-None failures aren't a black box. Only `result` is non-None
    on a successful LLM classification."""
    result: ClassifyResult | None = None
    skipped_reason: str = ""        # "short_text" / "no_buffer" / "tool_calls_finish" / ""
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
    # Stream's terminal `finish_reason`. Used both as a short-circuit
    # (tool_calls → never classify; saves the LLM call) AND as input
    # to the LLM prompt (so it can flag plan-without-action correctly).
    finish_reason: str = ""


async def classify_at_stream_end(
        proj_sid: str,
        local_base_url: str,
        local_model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        threshold: float = 0.7,
        conv_sid: str | None = None,
        current_route: str = "",
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
        api_key=api_key, timeout_s=timeout_s, threshold=threshold,
        conv_sid=conv_sid, current_route=current_route)
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
        user_goal: str = "",
        progress_tracker: str = "",
        tool_summary: str = "",
        force_frontier_threshold: float = 1.01,  # 1.01 = effectively disabled
        short_text_threshold: int = 50,
        stop_text_threshold: int = 50,
        conv_sid: str | None = None,
        current_route: str = "",
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

    # finish_reason short-circuit: streams that ended with tool_calls
    # are agent-still-acting (never a punt). Saves an LLM call per
    # tool-call turn — major cost reduction since most agent turns are
    # tool calls.
    finish = _extract_finish_reason(raw) or ""
    diag.finish_reason = finish
    if finish == "tool_calls":
        diag.skipped_reason = "tool_calls_finish"
        return diag

    text = _extract_text_from_buffer(raw)
    diag.extracted_text_chars = len(text)
    # Short-text short-circuit. Two thresholds because finish=stop and
    # finish=length/incomplete have different signal density:
    #   - finish=stop: agent decided to end the turn. Even very short
    #     text ("Done." / "好的。") can be a real soft-punt to the user.
    #     User directive C: classify ALL stops, no lower bound (effective
    #     stop_text_threshold=1).
    #   - finish=length/incomplete: stream was truncated by the upstream,
    #     not the agent. Short text more likely indicates a partial
    #     fragment than a real punt — keep the legacy 50-char floor.
    floor = stop_text_threshold if finish == "stop" else short_text_threshold
    if len(text) < floor:
        diag.skipped_reason = "short_text"
        return diag

    # Compose user content: structured context the semantic classifier
    # uses to decide "did the agent actually finish the user's goal".
    # Each section is bounded so total prompt stays ~5KB.
    sections = [
        f"finish_reason: {finish or 'unknown'}",
        f"text_chars: {len(text)}",
        "",
        "user_goal:",
        (user_goal or "(not available)"),
        "",
        "progress_tracker:",
        (progress_tracker or "(no tracker found in conversation)"),
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
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
        # Auto-force frontier on high-confidence PUNT. The deterministic
        # fix that doesn't depend on codex parsing synthetic events or
        # agent self-discipline. Same flag mechanism as
        # empty_response_guard — next request to this session bypasses
        # routing heuristics and goes straight to frontier (gpt-5.5).
        # Bounded by the higher `soft_completion_auto_force_frontier_threshold`
        # so we don't escalate every borderline verdict.
        if result.p >= force_frontier_threshold:
            # Don't re-set force_frontier when already on frontier —
            # avoids infinite loop where every frontier response
            # triggers force_frontier again, killing cache hits.
            if current_route == "frontier":
                diag.skipped_reason = f"force_frontier skipped: already on frontier"
            else:
                try:
                    from . import empty_response_guard as _erg
                    flag_key = conv_sid if conv_sid else proj_sid
                    _erg.force_next_to_frontier(
                        flag_key,
                        f"soft_punt p={result.p:.2f}: {result.reason[:60]}")
                except Exception:  # noqa: BLE001 — guard must never break classifier
                    pass
    return diag


# ─── public helper for PUNT-triggered forensics ───────────────────────────


def write_punt_forensics(
        proj_sid: str,
        forensics_dir,
        result: ClassifyResult,
        diag: ClassifyDiag,
        max_dumps: int = 100,
) -> str | None:
    """Convenience: dump forensics when classifier returned a high-
    confidence PUNT. Lets the proxy do this without importing forensics
    + composing the args repeatedly. Returns dump path str or None."""
    try:
        from . import forensics as _fx
    except Exception:  # noqa: BLE001 — forensics is optional
        # Why: forensics module is optional/test-mocked. If import fails
        # we silently skip the dump — classification result is already
        # returned to the caller.
        return None
    raw = _OUTPUT_BUFFER.get(proj_sid, "") or ""
    path = _fx.write_forensics_dump(
        forensics_dir=forensics_dir,
        proj_sid=proj_sid,
        trigger=f"punt_p{int(result.p * 100):02d}",
        response_buffer=raw,
        classifier_verdict={
            "soft_punt": result.soft_punt,
            "p": result.p,
            "reason": result.reason,
            "extracted_text_chars": diag.extracted_text_chars,
            "raw_buffer_chars": diag.raw_buffer_chars,
            "finish_reason": diag.finish_reason,
        },
        max_dumps=max_dumps,
    )
    return str(path) if path else None


# ─── gate injection ────────────────────────────────────────────────────────

_GATE_TEMPLATE = """\
<system-reminder>
[NOT USER INPUT — tinyctx soft-completion gate]

Your previous turn ended without taking action (classifier verdict: `{pattern}`). You MUST resume the work in this turn — the next thing you emit must be either tool calls OR a tracker enumeration with evidence. Pick the path that matches what your previous turn actually said:

═══ PATH A — your previous turn LISTED STEPS / GAVE A PLAN ═══
("I'll do X, then Y, then Z" / numbered plan / bullet list of next actions)

→ Execute ALL the steps in sequence NOW, this turn. Use tools/commands directly.
→ Do NOT re-state the plan. Do NOT split it across more turns. Do NOT ask the user "should I proceed?".
→ If a step genuinely cannot be executed (missing credential, ambiguous spec), STOP at that step and report which one and why — but only after you've done every step that CAN be done.

═══ PATH B — your previous turn ASKED THE USER A META-QUESTION ═══
("what would you like next" / "options:" / "shall I continue" / equivalents)

→ You MAY NOT ask the user without first vetting the question through advisor:

```
spawn_agent(role="advisor", task=\"\"\"
The executor wants to ask the user a meta-question. Decide whether the question is genuinely needed, or whether the executor should keep working.

Question I want to ask: <repeat verbatim>
Original user goal: <one-sentence restatement>
Progress tracker:
  1. <item> — completed (evidence: <commit hash / test result / file path / build log line>)
  2. <item> — de-scoped (reason: <reason>)
  …
Hard verifications done: <list with evidence>
Hard verifications NOT done: <list>

Reply EXACTLY one of:
  - ask: <one-line reason the question is genuinely needed>
  - work: <bullet list of concrete steps to do without asking>
\"\"\")
wait_agent(...)
```

→ If advisor replies `work: …`, execute those steps (Path A applies).
→ If advisor replies `ask: …`, proceed with the question, citing "Advisor: ask — <reason>".

═══ PATH C — your previous turn DECLARED "DONE" / "FINAL SUMMARY" ═══
(but progress tracker still has unchecked items, or user-stated verification not run)

→ Enumerate every tracker item. Each must be `completed (evidence: …)` or `de-scoped (reason: …)`.
→ If user-stated verification (build / test / deploy / start) was NOT executed: run it NOW with tool calls.
→ Only after the tracker is genuinely empty AND verification ran successfully may you re-emit a Final summary.

═══ Universal rules ═══

- Plain-text plan/promise without tool calls = treated as soft-punt by tinyctx classifier; you'll loop here again.
- Asking the user without going through advisor = treated as soft-punt; same loop.
- The only exits are: (a) tool calls advancing the work, (b) tracker enumeration with evidence, (c) advisor `ask:` verdict.

This gate fires once per detected soft-completion. It will not nag again until the next detection.
</system-reminder>"""


def maybe_inject_soft_completion_gate(
        body: dict[str, Any], proj_sid: str
) -> tuple[dict[str, Any], bool, str]:
    """If the soft-completion flag is set for this session, append a
    `<system-reminder>` to top-level `instructions` requiring the agent
    to vet the would-be user question through advisor first. Clears the
    flag on injection (fire-once semantics).

    Returns `(body, was_injected, matched_pattern)`. The original body
    is not mutated."""
    flag = _SOFT_COMPLETION_FLAG.get(proj_sid)
    if not flag or not flag.get("active"):
        return body, False, ""
    pattern = str(flag.get("matched_pattern", "unknown"))
    out = dict(body)
    gate_text = _GATE_TEMPLATE.format(pattern=pattern)
    inst = out.get("instructions")
    if isinstance(inst, str) and inst.strip():
        out["instructions"] = inst.rstrip() + "\n\n" + gate_text
    else:
        out["instructions"] = gate_text
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
