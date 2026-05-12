"""Rolling per-session compression of older conversation turns.

Magic Context's Historian, adapted for tinyctx. Two halves:

  update()         async background task — runs after every Nth turn,
                   compresses turns 1..len-recent_keep into a structured
                   digest (markdown summary + compartments/facts JSON),
                   persists to disk per session.

  apply_to_body()  sync, called from the proxy's mutation gate — replaces
                   the older turns in the request body with the digest as
                   a single system message. Saves dramatic per-turn token
                   cost on long sessions, BUT mutates history bytes so it
                   must be gated by CacheAwareMutator like dedup/purge.

Both halves are off by default (`historian_enabled`, `historian_substitute`
in config). The update half is harmless (just generates a sidecar file);
the substitute half changes what codex sees and should only be on when
prompt-cache savings outweigh churn.

Failure tolerance: every LLM call is wrapped in a try/except that logs
nothing and silently no-ops. The proxy never blocks on historian failure.
"""
from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .compactor import _flatten_history, _local_call, parse_judge_output
from .config import BackendCfg
from .scout import cache_dir


HISTORIAN_PROMPT = (
    "You are a session historian. Compress the following CONVERSATION "
    "HISTORY (everything before the most recent few turns) into a digest "
    "that another LLM can read to continue the task without ever seeing "
    "the raw older turns. Output two sections in this exact order:\n\n"
    "(1) A markdown summary with these headings:\n"
    "    ## What we are doing and why\n"
    "    ## Files & decisions\n"
    "    ## Commands & outcomes\n"
    "    ## Open issues / next steps\n\n"
    "(2) After the markdown, a fenced JSON block (```json … ```) with:\n"
    '    {"compartments": [{"name": "...", "topic": "...", "summary": "...",\n'
    '                       "files": ["..."]}, ...],\n'
    '     "facts":        [{"claim": "...", "evidence": "..."}, ...],\n'
    '     "open_questions": ["...", ...]}\n\n'
    "Constraints: be terse, drop redundancy, never invent. Make the digest "
    "self-contained — the next turn's LLM will see it INSTEAD of the raw "
    "older turns, not in addition to them."
)


@dataclass
class HistorianState:
    last_run_turn_count: int = 0
    last_digest_md: str = ""
    last_digest_structured: dict[str, Any] = field(
        default_factory=lambda: {"compartments": [], "facts": [],
                                 "open_questions": []})
    revision: int = 0
    project_root: Path | None = None


# Per-session, in-memory state. Cleared on process restart.
_STATE_BY_SID: dict[str, HistorianState] = {}


def get_state(sid: str) -> HistorianState:
    return _STATE_BY_SID.setdefault(sid, HistorianState())


def reset_session(sid: str) -> None:
    """Used by tests; not called in production."""
    _STATE_BY_SID.pop(sid, None)


def historian_dir(project_root: Path, session_id: str) -> Path:
    return cache_dir(project_root) / "sessions" / session_id


def _count_history_items(items: list) -> int:
    """Items that count as a "turn" for triggering: roles user/assistant
    and tool-call/tool-result types."""
    if not isinstance(items, list):
        return 0
    n = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role")
        t = it.get("type")
        if role in ("user", "assistant") or t in (
            "function_call", "function_call_output", "tool_use",
            "tool_result", "message",
        ):
            n += 1
    return n


# ----------------------------------------------------------- update half


async def update(
    sid: str,
    body: dict[str, Any],
    backend: BackendCfg,
    *,
    min_new_turns: int = 5,
    recent_keep: int = 4,
    project_root: Path | None = None,
    _llm_call: Callable[..., Awaitable[str]] = _local_call,
) -> bool:
    """Run one historian pass for `sid`. Returns True iff a new digest was
    written. Quiet on all failures."""
    state = get_state(sid)
    if project_root and state.project_root is None:
        state.project_root = project_root

    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list):
        return False
    n = _count_history_items(items)
    if n - state.last_run_turn_count < min_new_turns:
        return False
    if n <= recent_keep:
        return False

    cut = max(0, len(items) - recent_keep)
    old = items[:cut]
    if not old:
        return False

    user_prompt = _flatten_history({"input": old})
    if len(user_prompt) < 500:
        return False

    try:
        async with httpx.AsyncClient() as client:
            digest = await _llm_call(
                client, backend, HISTORIAN_PROMPT, user_prompt,
                max_tokens=2000,
            )
    except Exception:
        return False

    md, structured = parse_judge_output(digest)
    state.last_digest_md = md
    state.last_digest_structured = structured
    state.last_run_turn_count = n
    state.revision += 1

    root = project_root or state.project_root
    if root:
        try:
            sdir = historian_dir(root, sid)
            sdir.mkdir(parents=True, exist_ok=True)
            md_path = sdir / f"historian-{state.revision}.md"
            json_path = sdir / f"historian-{state.revision}.json"
            md_path.write_text(md)
            json_path.write_text(json.dumps(structured, indent=2,
                                            ensure_ascii=False))
            latest = sdir / "historian-latest.md"
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                os.symlink(md_path, latest)
            except (OSError, NotImplementedError):
                # Why: symlink unsupported (Windows, restricted fs) —
                # fall back to copy. The digest is already at md_path.
                try:
                    latest.write_text(md)
                except OSError:
                    # Why: both symlink and copy failed; the digest
                    # itself is persisted, recall can still find it.
                    pass
        except OSError:
            # Why: digest write failed (disk full, permissions). Skip
            # this digest cycle; historian retries next compaction.
            pass

    return True


# ----------------------------------------------------------- apply half


_DIGEST_TAG = "<tinyctx-historian-digest"


def apply_to_body(
    body: dict[str, Any], sid: str, *, recent_keep: int = 4
) -> dict[str, Any]:
    """Replace items[:len-recent_keep] with one system message containing
    the latest digest. Returns a new body (does not mutate input). Idempotent
    — if the digest tag is already present in the first few items, returns
    the body unchanged."""
    state = get_state(sid)
    if not state.last_digest_md:
        return body
    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list) or len(items) <= recent_keep:
        return body

    # Idempotence: already substituted?
    for it in items[:5]:
        if isinstance(it, dict):
            content = it.get("content")
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        text += str(c.get("text", ""))
            if _DIGEST_TAG in text:
                return body

    digest_block = (
        f"<tinyctx-historian-digest revision=\"{state.revision}\">\n"
        f"{state.last_digest_md}\n"
        f"</tinyctx-historian-digest>"
    )
    cut = len(items) - recent_keep
    new_items = (
        [{"role": "system", "content": digest_block}] + items[cut:]
    )
    out = deepcopy(body)
    if "input" in out:
        out["input"] = new_items
    elif "messages" in out:
        out["messages"] = new_items
    else:
        out["input"] = new_items
    return out


# --------------------------- background-task tracking (proxy convenience)

_BG_TASKS: set[asyncio.Task] = set()


def spawn_update(
    sid: str,
    body: dict[str, Any],
    backend: BackendCfg,
    *,
    min_new_turns: int = 5,
    recent_keep: int = 4,
    project_root: Path | None = None,
) -> asyncio.Task:
    """Fire-and-forget wrapper. Holds a strong reference so the task isn't
    GC'd before completion; callers get the Task back if they want to await
    it (tests do)."""
    coro = update(sid, body, backend,
                  min_new_turns=min_new_turns,
                  recent_keep=recent_keep,
                  project_root=project_root)
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task
