"""Bridge successful advisor outputs into the next request as a
synthetic user continuation.

Why
───
When the upstream model already called advisor successfully, tinyctx's
stream-rewrite path may still only inject a noop synthetic continue
(`shell true` / `local_shell true` / `update_plan []`). That keeps the
turn alive but can still leave the executor paused on a plan sentence.

This module stores a small one-shot "continue with this work" payload in
SessionState so a pre-flight guard can inject it on the NEXT request.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from . import session_state


_NS = "advisor_continuation"
_K_PENDING = "pending"

session_state.register_session_end_reset(_NS, [_K_PENDING])


@dataclass
class PendingWork:
    work_text: str
    source: str
    ts: float


def _clean_work_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:4000].strip()


def store_pending_work(conv_sid: str, work_text: str, *, source: str) -> bool:
    cleaned = _clean_work_text(work_text)
    if not conv_sid or not cleaned:
        return False
    session_state.set(conv_sid, _NS, _K_PENDING, {
        "work_text": cleaned,
        "source": source or "advisor_output",
        "ts": time.time(),
    })
    return True


def consume_pending_work(conv_sid: str) -> PendingWork | None:
    raw = session_state.consume(conv_sid, _NS, _K_PENDING)
    if raw is None:
        return None
    return PendingWork(
        work_text=str(raw.get("work_text", "") or ""),
        source=str(raw.get("source", "") or "advisor_output"),
        ts=float(raw.get("ts", 0) or 0),
    )


def inject_pending_work_into_body(
    body: dict[str, Any], pending: PendingWork
) -> tuple[dict[str, Any], bool]:
    items = body.get("input")
    if not isinstance(items, list):
        return body, False
    text = (
        "[tinyctx advisor continuation — the previous turn already got "
        "advisor guidance. Treat this as continuation context, not a new "
        "user request. Execute these steps directly and do not stop at a "
        "plan-only response.]\n\n"
        f"Continue with this work:\n{pending.work_text}"
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


def _find_string_field(blob: str, field_name: str) -> str:
    m = re.search(rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                  blob, re.DOTALL)
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"')
    except json.JSONDecodeError:
        return ""


_JUDGE_SYSTEM_PROMPT = """You are a work-extraction filter inside an agent gateway. Your job: look at text that an advisor sub-agent produced and decide whether it contains concrete, actionable work the executor should immediately continue with.

Return "YES" ONLY when the text contains at least one of:
- A numbered list of concrete implementation steps (e.g. "1. Add a retry... 2. Update the config...")
- A clear CHOSEN/FALLBACK verdict picking between options
- An explicit imperative continuation directive (e.g. "Continue by...", "Next, implement...", "Proceed with...")

Return "NO" when the text is:
- Pure analysis, explanation, or background without a concrete next step
- A summary of what was done with no forward work
- A vague suggestion ("you could consider...") without a firm directive
- An error message or empty output

Reply with EXACTLY one word: YES or NO. No punctuation, no explanation."""


async def extract_pending_work_from_outgoing_sse(
    text: str,
    *,
    local_base_url: str = "",
    local_model: str = "",
    api_key: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """Use a local-model judge to decide whether the SSE output from an
    advisor sub-agent call contains actionable continuation work. Returns
    the cleaned work text, or "" when the judge says NO or on any failure."""
    if not text:
        return ""
    if '"name":"ask_advisor"' not in text and '"namespace":"mcp__advisor__"' not in text:
        return ""
    output = _find_string_field(text, "output")
    if not output:
        return ""
    output = output.strip()
    if not output:
        return ""

    if not local_base_url or not local_model:
        return ""

    tail = output[-3000:] if len(output) > 3000 else output
    try:
        import httpx
        payload = {
            "model": local_model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"advisor_output:\n{tail}"},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = local_base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s)) as client:
            r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        verdict = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .upper()
        )
    except Exception:
        return ""

    if verdict.startswith("YES"):
        return _clean_work_text(output)
    return ""


def reset_state(conv_sid: str | None = None) -> None:
    if conv_sid is None:
        for sid in list(session_state.keys_with_prefix("")):
            session_state.clear(sid, _NS, _K_PENDING)
        return
    session_state.clear(conv_sid, _NS, _K_PENDING)
