"""Multi-subagent compactor for codex auto-compaction.

# Invariant: PRISTINE RECOMPUTATION ONLY
#
# Inspired by cortexkit/magic-context's age-tier caveman compression: every
# compaction must run on the raw conversation history, never on a previously
# compacted summary. Codex naturally feeds us pristine history every turn
# (it does not use previous_response_id; see openai/codex#4047), so today
# this happens for free. If/when we add an incremental compactor, it must
# still derive each pass from the original turns and never feed a prior
# summary back through the role drafts. Otherwise lossy drift compounds and
# silently degrades quality across compactions.
#
# The 'PRISTINE' marker in role outputs is intentionally distinctive so the
# guard test (tests/test_compactor.py::test_pristine_recomputation_guard)
# can detect any future regression where compactor output is fed back in.

When codex's context approaches the model's limit, codex emits a "handoff
summary" prompt asking the same model to compress the conversation. tinyctx
already detects this fingerprint (router.is_compaction_request) and reroutes
to the local 27B instead of paying frontier rates. But a single-pass local
summary loses nuance.

This module replaces that single pass with a 3-role debate + 1-judge merge:

    archaeologist   preserve verbatim facts: file paths, decisions, exact errors
    narrator        preserve intent and the storyline of attempts
    enumerator      list every concrete artifact (files touched, commands run, errors)
    judge           merge the three drafts into a canonical handoff summary

The 3 role drafts run in parallel via httpx.AsyncClient, so wall-clock time
is roughly one local-model call (not three). The judge runs after.

If a role call fails, the judge proceeds with the survivors. If 2 of 3 roles
fail OR the judge fails, we fall back to a deterministic concatenation. The
caller (proxy.py) catches any compactor exception and falls back to a
straight forward to the local backend, so this is at worst a quality
regression — never a hard failure.

Side effect: when continuity is enabled, the final summary is persisted via
tinyctx.continuity.save_compaction so a new session can recall it.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

import httpx

from .config import BackendCfg


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
                            re.DOTALL | re.IGNORECASE)
_BARE_JSON_RE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)


def parse_judge_output(text: str) -> tuple[str, dict[str, Any]]:
    """Split the judge's output into (markdown_summary, structured_dict).

    Tries fenced ```json``` first (preferred); falls back to a heuristic
    bare-object scan; finally degrades to (full_text, empty_structured).
    Always returns successfully — no exceptions on malformed output."""
    structured: dict[str, Any] = {
        "compartments": [],
        "facts": [],
        "open_questions": [],
    }

    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                structured.update(
                    {k: parsed.get(k) or structured[k]
                     for k in ("compartments", "facts", "open_questions")}
                )
            md = (text[: m.start()] + text[m.end():]).strip()
            return md or text.strip(), structured
        except json.JSONDecodeError:
            # Why: fenced block content isn't valid JSON — fall through
            # to the bare-object heuristic below rather than abort.
            pass

    # Heuristic bare-object: take the LAST {...} block and try to parse it.
    candidates = _BARE_JSON_RE.findall(text)
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and (
                "compartments" in parsed or "facts" in parsed
            ):
                structured.update(
                    {k: parsed.get(k) or structured[k]
                     for k in ("compartments", "facts", "open_questions")}
                )
                md = text.replace(cand, "").strip()
                return md or text.strip(), structured
        except json.JSONDecodeError:
            # Why: this candidate isn't valid JSON; try the next one.
            continue

    return text.strip(), structured


# ----------------------------------------------------------------- prompts

ARCHAEOLOGIST_PROMPT = (
    "You are an archaeologist. Read the conversation history and extract "
    "everything that should NEVER be lost when this session is compacted: "
    "exact file paths touched, exact commands run, exact error messages, "
    "concrete numbers, decisions the user made. Preserve verbatim where "
    "possible. Be terse, no filler. Use bulleted markdown."
)

NARRATOR_PROMPT = (
    "You are a narrator. Read the conversation history and write a 4-8 "
    "sentence storyline of what we attempted, in what order, and why. Capture "
    "the user's intent and any pivots. No bullets — flowing prose. Do not "
    "list files or commands; the archaeologist handles those."
)

ENUMERATOR_PROMPT = (
    "You are an enumerator. Read the conversation history and produce three "
    "lists: (1) every file modified or proposed for modification, (2) every "
    "test or command run with its outcome (one line each), (3) every "
    "unresolved issue or open question. Markdown lists, no narrative."
)

JUDGE_PROMPT = (
    "You are a handoff editor. You will receive three perspectives on a "
    "coding session: ARCHAEOLOGIST (verbatim facts), NARRATOR (storyline), "
    "and ENUMERATOR (artifact lists). Merge them into a structured handoff "
    "that another LLM can read to continue the task.\n\n"
    "Output two sections, in this exact order:\n\n"
    "(1) A markdown summary with these headings:\n"
    "    ## What we are doing and why\n"
    "    ## Files & decisions\n"
    "    ## Commands & outcomes\n"
    "    ## Open issues / next steps\n\n"
    "(2) After the markdown summary, a fenced JSON block (```json … ```)\n"
    "    containing this schema EXACTLY:\n"
    "    {\n"
    '      "compartments": [{"name": "...", "topic": "...", "summary": "...",\n'
    '                        "files": ["..."]}, ...],\n'
    '      "facts":        [{"claim": "...", "evidence": "..."}, ...],\n'
    '      "open_questions": ["...", "..."]\n'
    "    }\n\n"
    "Rules:\n"
    "- Compartments group related work by topic (e.g. 'auth-setup', "
    "'db-migration'). 1-5 of them.\n"
    "- Facts are atomic, independently-checkable claims (e.g. 'The JWT "
    "secret lives in .env not config.toml').\n"
    "- Drop redundancy across the three perspectives.\n"
    "- Do NOT invent anything none of the three mentioned.\n"
    "- Be terse and accurate."
)

ROLES: list[tuple[str, str]] = [
    ("archaeologist", ARCHAEOLOGIST_PROMPT),
    ("narrator", NARRATOR_PROMPT),
    ("enumerator", ENUMERATOR_PROMPT),
]


# --------------------------------------------------------- history extraction

def _flatten_history(body: dict[str, Any], *, max_chars: int = 90_000) -> str:
    """Build a single string of the conversation history that the role agents
    will analyze. We strip the 'Create a handoff summary...' instruction (the
    debate's job is precisely to *replace* that single-pass instruction)."""
    out: list[str] = []

    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list):
        return ""

    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role") or it.get("type") or ""
        content = it.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t in ("text", "input_text", "output_text"):
                        parts.append(str(c.get("text", "")))
            text = "\n".join(parts)
        elif role in ("function_call", "tool_use"):
            text = f"[tool call: {it.get('name','?')}({it.get('arguments','')})]"
        elif role in ("function_call_output", "tool_result"):
            out_v = it.get("output") or it.get("content") or ""
            text = f"[tool result: {out_v}]"
        if text.strip():
            out.append(f"<{role}>\n{text.strip()}\n</{role}>")

    blob = "\n\n".join(out)
    if len(blob) > max_chars:
        # Keep head + tail; the middle of the conversation is usually less
        # critical than the most recent turns.
        head = blob[: max_chars // 3]
        tail = blob[-(2 * max_chars // 3) :]
        blob = head + "\n\n... [middle truncated by tinyctx compactor] ...\n\n" + tail
    return blob


# ------------------------------------------------------------ HTTP plumbing

async def _local_call(client: httpx.AsyncClient, backend: BackendCfg,
                      system_prompt: str, user_prompt: str,
                      *, max_tokens: int = 1500,
                      temperature: float = 0.2) -> str:
    """One call to a local OpenAI-compat /chat/completions endpoint. Returns
    the assistant message text, or raises on failure."""
    import os as _os
    url = backend.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": backend.model or "local",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    # Send Authorization when the backend declares an api_key_env. LMStudio
    # ignored auth so this used to be a no-op; DeepSeek (and most hosted
    # OpenAI-compat backends) require it. Without this header every
    # compactor draft hits 401 and codex's auto-compact silently falls
    # back to a 43-char "[tinyctx compactor: all subagents failed]"
    # placeholder, which obliterates the model's memory and surfaces as
    # "earlier task details were compacted out" in the codex.app UI.
    if backend.api_key_env:
        api_key = _os.environ.get(backend.api_key_env)
        if api_key:
            headers["Authorization"] = (
                api_key if api_key.lower().startswith(("bearer ", "basic "))
                else f"Bearer {api_key}"
            )
    r = await client.post(url, json=payload, headers=headers,
                          timeout=backend.timeout_s or 180.0)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------- debate orchestration

async def _gather_drafts(client: httpx.AsyncClient, backend: BackendCfg,
                         history: str) -> dict[str, str]:
    """Run the three role drafts in parallel. Returns a {role: text} dict;
    failed roles are absent."""

    async def _one(role: str, sys_prompt: str) -> tuple[str, str | None]:
        try:
            text = await _local_call(client, backend, sys_prompt, history,
                                     max_tokens=1200)
            return role, text
        except Exception:  # noqa: BLE001 — swallow per-role failures
            return role, None

    results = await asyncio.gather(*[_one(r, p) for r, p in ROLES])
    return {role: text for role, text in results if text}


async def _judge(client: httpx.AsyncClient, backend: BackendCfg,
                 drafts: dict[str, str]) -> str:
    """Have the judge merge whatever drafts we got. If only one survives,
    the judge essentially polishes it."""
    parts = [f"<{role}>\n{text}\n</{role}>" for role, text in drafts.items()]
    user = "\n\n".join(parts) if parts else "(no drafts available)"
    return await _local_call(client, backend, JUDGE_PROMPT, user,
                             max_tokens=1800)


def _fallback_concat(drafts: dict[str, str]) -> str:
    """Last-resort merge if the judge call fails."""
    if not drafts:
        return "[tinyctx compactor: all subagents failed]"
    out = ["# Handoff summary (tinyctx fallback merge)"]
    for role, text in drafts.items():
        out.append(f"## {role}\n{text}")
    return "\n\n".join(out)


async def compact_with_debate(body: dict[str, Any], backend: BackendCfg
                              ) -> tuple[str, dict[str, Any]]:
    """End-to-end: take a Responses-API body whose instruction is codex's
    handoff-summary prompt, return (summary_text, telemetry).

    Telemetry includes per-role timings and which roles failed; useful for
    logging and stats.
    """
    history = _flatten_history(body)
    if not history:
        return "[tinyctx compactor: no history]", {"reason": "empty_history"}

    timings: dict[str, float] = {}
    started = time.time()

    async with httpx.AsyncClient() as client:
        t0 = time.time()
        drafts = await _gather_drafts(client, backend, history)
        timings["drafts_s"] = time.time() - t0

        if len(drafts) >= 2:
            try:
                t1 = time.time()
                merged = await _judge(client, backend, drafts)
                timings["judge_s"] = time.time() - t1
                outcome = "judged"
            except Exception:
                merged = _fallback_concat(drafts)
                outcome = "judge_failed_concat"
        elif len(drafts) == 1:
            # only one draft — let the judge polish it.
            try:
                t1 = time.time()
                merged = await _judge(client, backend, drafts)
                timings["judge_s"] = time.time() - t1
                outcome = "single_draft_polished"
            except Exception:
                merged = list(drafts.values())[0]
                outcome = "single_draft_raw"
        else:
            merged = "[tinyctx compactor: all role drafts failed]"
            outcome = "all_failed"

    timings["total_s"] = time.time() - started

    # Split judge output into markdown summary + structured compartments/facts.
    summary_md, structured = parse_judge_output(merged)
    telemetry = {
        "outcome": outcome,
        "drafts_completed": list(drafts.keys()),
        "timings": timings,
        "structured": {
            "compartments": len(structured.get("compartments") or []),
            "facts": len(structured.get("facts") or []),
            "open_questions": len(structured.get("open_questions") or []),
        },
    }
    return summary_md, telemetry, structured


# ----------------------------------------------- response shape construction

def build_responses_api_payload(summary: str, model: str) -> dict[str, Any]:
    """Wrap the merged summary in a Responses-API completed-response shape so
    the proxy can hand it back to codex without further translation."""
    rid = "resp_" + uuid.uuid4().hex[:24]
    item_id = "msg_" + uuid.uuid4().hex[:24]
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": summary}],
            }
        ],
        "usage": {"input_tokens": len(summary) // 4,
                  "output_tokens": len(summary) // 4,
                  "total_tokens": len(summary) // 2},
    }


def build_responses_api_sse(summary: str, model: str) -> bytes:
    """Emit a minimal SSE stream that codex will accept: response.created,
    one output_text.delta with the full summary, then response.completed."""
    rid = "resp_" + uuid.uuid4().hex[:24]
    item_id = "msg_" + uuid.uuid4().hex[:24]
    events = [
        ("response.created", {
            "type": "response.created",
            "response": {"id": rid, "model": model, "object": "response",
                         "status": "in_progress"},
        }),
        ("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": item_id, "type": "message", "role": "assistant",
                     "content": []},
        }),
        ("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": item_id, "output_index": 0, "content_index": 0,
            "delta": summary,
        }),
        ("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": item_id, "output_index": 0, "content_index": 0,
            "text": summary,
        }),
        ("response.completed", {
            "type": "response.completed",
            "response": {
                "id": rid, "model": model, "object": "response",
                "status": "completed",
                "output": [{"id": item_id, "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": summary}]}],
            },
        }),
    ]
    chunks = []
    for ev_type, payload in events:
        chunks.append(f"event: {ev_type}\ndata: {json.dumps(payload)}\n\n".encode())
    return b"".join(chunks)
