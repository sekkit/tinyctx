"""Cross-session plan persistence: when codex's `update_plan` tool is
called in any thread, save the latest plan state to disk keyed by
working directory. When a NEW codex thread starts on the same repo,
inject the persisted plan into instructions so the agent picks up
where the prior thread left off.

Why
───
Live trace 2026-05-10: a long-running codex thread (1866 turns,
760K context) was abandoned and a fresh thread opened on the same
repo. The new thread had `turn_count=0, input_tokens=15K` — fully
empty of prior context. Agent correctly said "no recoverable plan in
current context" but the user lost real work.

Codex sessions are isolated by design (codex.app's own architecture).
tinyctx can persist the plan ITSELF (independent of codex's session
state) and re-inject on new threads to bridge the gap.

Storage
───────
~/.tinyctx/state/plans/<cwd_hash>.json

Each file:
{
  "cwd": "<absolute path of working dir at save time>",
  "plan_text": "  1. [completed] Create viewmodel/...\n  2. [pending] ...",
  "updated_at": 1778383500.123,
  "turn_count_at_save": 1866,
  "session_id_at_save": "global"
}

Per-cwd, not per-session — that's the whole point.

TTL: 7 days. Older plans are not auto-injected (avoid stale-context
poisoning when starting a genuinely unrelated task).

Detection (when to save)
────────────────────────
On every request: scan body.input for the LATEST `update_plan` /
`TodoWrite` function_call. If found and content differs from on-disk,
write to disk.

Detection (when to inject)
──────────────────────────
On every request where `turn_count == 0` (codex says "fresh thread"):
  1. Load persisted plan for this cwd
  2. If exists AND TTL not expired
  3. Prepend a `<persisted-plan>` block to body.instructions
  4. Mark trace.persisted_plan_injected = True
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


# Default 7 days. Older persisted plans aren't auto-injected.
DEFAULT_TTL_S = 7 * 24 * 3600


# ─── storage path ─────────────────────────────────────────────────────────


def _cwd_hash(cwd: str) -> str:
    """Stable per-cwd hash."""
    if not cwd:
        return "default"
    return hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:16]


def plan_path(state_dir: Path, cwd: str) -> Path:
    return state_dir / "plans" / f"{_cwd_hash(cwd)}.json"


# ─── plan extraction (from body.input) ────────────────────────────────────
# Reuse soft_completion's extractor — same `update_plan` / `TodoWrite`
# call mining. Lazy import to avoid circular module load.


def extract_plan_text(body: dict[str, Any]) -> str:
    """Mine body.input for the latest update_plan / TodoWrite call's
    rendered plan items text. Returns empty string when no tracker
    found in this turn's history."""
    if not isinstance(body, dict):
        return ""
    items = body.get("input")
    if not isinstance(items, list):
        return ""
    try:
        from . import soft_completion
        return soft_completion.extract_progress_tracker(items)
    except Exception:  # noqa: BLE001
        return ""


# ─── save / load ──────────────────────────────────────────────────────────


def save_plan(state_dir: Path, cwd: str, plan_text: str,
               session_id: str = "", turn_count: int = 0) -> bool:
    """Atomically write the plan to disk for `cwd`. Skips when content
    is identical to existing on-disk file (avoid pointless write/fsync).
    Returns True if write happened, False if no-op."""
    if not plan_text or not plan_text.strip():
        return False
    path = plan_path(state_dir, cwd)
    # Skip if same content already on disk
    try:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("plan_text") == plan_text:
                return False
    except Exception:  # noqa: BLE001
        pass
    payload = {
        "cwd": cwd,
        "plan_text": plan_text,
        "updated_at": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime()),
        "session_id_at_save": session_id,
        "turn_count_at_save": turn_count,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        return False


def load_plan(state_dir: Path, cwd: str,
               ttl_s: float = DEFAULT_TTL_S) -> dict[str, Any] | None:
    """Read persisted plan for `cwd`. Returns None when no file, file
    unreadable, or TTL expired."""
    path = plan_path(state_dir, cwd)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    updated_at = float(data.get("updated_at", 0) or 0)
    if updated_at <= 0:
        return None
    if time.time() - updated_at > ttl_s:
        return None  # too stale
    return data


# ─── injection ────────────────────────────────────────────────────────────

_INJECT_TEMPLATE = """\
<persisted-plan source="tinyctx" updated="{updated}" cwd="{cwd}" prev_turn_count="{turn_count}">
A previous codex thread for this working directory had this progress tracker. Use it to resume work in this NEW thread, or ignore if the user is starting genuinely unrelated work:

{plan_text}

Resume rules:
- If the user's current request relates to the items above, continue from the first uncompleted item.
- If the request is unrelated, IGNORE this block — do not pollute the new task.
- This block is informational; it is NOT a system directive to "do everything in the plan automatically".
</persisted-plan>

"""


def inject_plan(body: dict[str, Any], plan_data: dict[str, Any]
                ) -> tuple[dict[str, Any], bool]:
    """Prepend a `<persisted-plan>` block to body.instructions. Returns
    (new_body, was_injected). Never mutates input."""
    if not isinstance(body, dict):
        return body, False
    plan_text = (plan_data or {}).get("plan_text", "")
    if not plan_text:
        return body, False
    inst = body.get("instructions", "") or ""
    if not isinstance(inst, str):
        return body, False
    block = _INJECT_TEMPLATE.format(
        updated=plan_data.get("updated_at_iso", "?"),
        cwd=(plan_data.get("cwd", "") or "?")[:200],
        turn_count=plan_data.get("turn_count_at_save", 0),
        plan_text=plan_text,
    )
    out = dict(body)
    out["instructions"] = block + inst
    return out, True


# ─── dashboard helpers ────────────────────────────────────────────────────


def list_plans(state_dir: Path) -> list[dict[str, Any]]:
    """List all persisted plans for dashboard display."""
    plans_dir = state_dir / "plans"
    if not plans_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(plans_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "file": p.name,
                "cwd": data.get("cwd", "?"),
                "updated_at_iso": data.get("updated_at_iso", "?"),
                "updated_at": data.get("updated_at", 0),
                "session_id_at_save": data.get("session_id_at_save", ""),
                "turn_count_at_save": data.get("turn_count_at_save", 0),
                "plan_chars": len(data.get("plan_text", "") or ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out


def clear_plan(state_dir: Path, cwd: str) -> bool:
    """Delete the persisted plan for `cwd`. Returns True if removed."""
    path = plan_path(state_dir, cwd)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
