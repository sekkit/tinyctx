"""Soft-completion gate: when the agent ends a turn by asking the user
a meta question ("what would you like to work on next?") instead of
completing the work and verifying outcomes, force the next turn to
route the question through advisor first.

Why this exists
───────────────
Live trace 2026-05-10: the stuck-loop watchdog (`stuck_loop.py`) saved
a session from a 1300+ turn debugging loop, but the agent then went
into a different failure mode — wrapped up the bug fix, declared
"All done" with 6 ✅, and asked "What would you like to work on next?"
without:
  - enumerating its progress tracker (4 items still ⭕)
  - running the user-stated verification step (rebuild APK + logcat
    + screencap)
  - calling the advisor completion gate

§4 收工纪律 and §3 advisor gate (in templates/AGENTS.md) both target
"agent declaring done", but neither bound to this *soft punt to user*
shape. User directive: "如果非要提问，走 advisor 进行回答" — if the
agent insists on asking, vet the question through advisor first.

Mechanism
─────────
1. **Sniffer**: as the proxy streams response bytes to the client, a
   small per-session ring buffer (`_OUTPUT_BUFFER[proj_sid]`, 8KB
   tail) accumulates the bytes. Pattern regexes scan the buffer
   incrementally (chunks may split a phrase, so buffering matters).
2. **Flag**: on first match, `_SOFT_COMPLETION_FLAG[proj_sid]` records
   `{"active": True, "matched_pattern": <name>, ts: <time>}`. Buffer
   is reset at next stream start so the flag fires once per turn.
3. **Gate**: on the *next* request to the same `proj_sid`, the proxy
   prepends a `<system-reminder>` to `body.input` requiring the agent
   to spawn_agent(role="advisor") with the would-be user question and
   tracker state, and act on advisor's `ask:`/`work:` verdict.

False positives
───────────────
The patterns are biased toward HIGH-PRECISION matches: phrases that
mostly only appear when the agent is genuinely punting (e.g., "what
would you like to work on next", "some options:"). One false-positive
per session costs an extra ~5K frontier tokens via advisor — annoying
but not broken. False negatives are worse: agent walks past the gate
and the user sees an incomplete task, which is the original problem.

Interaction with stuck_loop
───────────────────────────
Both modules live alongside each other and share the same `proj_sid`
keying. They serve different failure modes:
  - stuck_loop fires on `turn_count > 80` without convergence (loop)
  - soft_completion fires on user-facing question without tracker
    completion (premature handoff)
A single session can hit both gates; that's fine — they nudge in
different directions.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any


# ─── per-session state ─────────────────────────────────────────────────────

_SOFT_COMPLETION_FLAG: dict[str, dict[str, Any]] = defaultdict(dict)
_OUTPUT_BUFFER: dict[str, str] = defaultdict(str)

# Buffer cap. The longest pattern we look for is ~50 chars; 8KB gives
# plenty of slack for SSE-frame overhead and split-chunk recovery while
# bounding memory.
_BUFFER_MAX = 8192


# ─── patterns ──────────────────────────────────────────────────────────────
# Calibrated for HIGH PRECISION. Each pattern should fire only on
# phrasings that are nearly always "soft punt to user" rather than
# legitimate technical content. Add patterns conservatively.

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # English
    (re.compile(r"what would you like (?:me )?to (?:work|do|tackle|focus)",
                re.IGNORECASE), "en_what_would_you_like"),
    (re.compile(r"what (?:do you want|would you prefer|should we do) "
                r"(?:to do |to work on |next|now)",
                re.IGNORECASE), "en_what_do_you_want"),
    (re.compile(r"\n\s*(?:Some\s+)?[Oo]ptions:\s*\n",
                re.IGNORECASE), "en_options_list"),
    (re.compile(r"would you like (?:me )?to (?:continue|proceed|move on|"
                r"go ahead)", re.IGNORECASE), "en_would_you_like_to_proceed"),
    # Chinese — codex output is sometimes Chinese
    (re.compile(r"你(?:想|希望)(?:让我)?(?:接下来|继续|下一步)?"
                r"(?:做|处理|完成|tackle)什么"), "zh_what_next"),
    (re.compile(r"(?:接下来|下一步)(?:做|要做|想做)?什么"),
        "zh_next_what_to_do"),
    (re.compile(r"想(?:要我|让我)?(?:做|处理)(?:什么|哪个|哪一个)"),
        "zh_want_to_do_what"),
]


# ─── streaming sniffer ─────────────────────────────────────────────────────


def reset_stream(proj_sid: str) -> None:
    """Clear the per-session output buffer at start of a new stream.
    The flag itself is NOT cleared — it survives across streams until
    the gate consumes it on the next request."""
    _OUTPUT_BUFFER.pop(proj_sid, None)


def scan_chunk(proj_sid: str, chunk: bytes) -> str | None:
    """Accumulate a stream chunk into the per-session buffer and scan
    for soft-completion patterns. On first match, sets the per-session
    flag and returns the matched pattern name. Subsequent calls within
    the same stream are no-ops once the flag is set.

    Never raises — designed to be called from the hot streaming path.
    Decode errors silently skip; partial-utf8 chunks are tolerated."""
    if not chunk:
        return None
    flag = _SOFT_COMPLETION_FLAG.get(proj_sid) or {}
    if flag.get("active"):
        # already matched this stream; don't keep scanning
        return None
    try:
        text = chunk.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    buf = _OUTPUT_BUFFER[proj_sid] + text
    if len(buf) > _BUFFER_MAX:
        buf = buf[-_BUFFER_MAX:]
    _OUTPUT_BUFFER[proj_sid] = buf
    for pat, name in _PATTERNS:
        if pat.search(buf):
            _SOFT_COMPLETION_FLAG[proj_sid] = {
                "active": True,
                "matched_pattern": name,
                "ts": time.time(),
            }
            return name
    return None


def get_flag(proj_sid: str) -> dict[str, Any] | None:
    """Return the active flag dict for a session, or None."""
    f = _SOFT_COMPLETION_FLAG.get(proj_sid)
    return f if f and f.get("active") else None


# ─── gate injection ────────────────────────────────────────────────────────

_GATE_TEMPLATE = """\
<system-reminder>
[NOT USER INPUT — tinyctx soft-completion gate]

Your previous turn ended by asking the user a question (matched pattern: `{pattern}`) instead of completing the work and verifying outcomes. Per the user's directive, **any question to the user MUST first be vetted by advisor**.

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
    # Consume the flag: fire-once until next detection.
    _SOFT_COMPLETION_FLAG[proj_sid] = {"active": False}
    # Also reset buffer so we don't immediately re-trigger from stale tail.
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


def state_snapshot(proj_sid: str) -> dict[str, Any]:
    """Inspect per-session sniffer state. Test/dev helper."""
    return {
        "flag": dict(_SOFT_COMPLETION_FLAG.get(proj_sid) or {}),
        "buffer_chars": len(_OUTPUT_BUFFER.get(proj_sid, "")),
    }
