"""Symphony-inspired task records for tinyctx request supervision."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    session_id: str = "global"
    project_root: str = ""
    title: str = ""
    state: str = "running"
    task_type: str = "unknown"
    acceptance: list[str] = field(default_factory=list)
    recommended_skills: list[str] = field(default_factory=list)
    recommended_mcp: list[str] = field(default_factory=list)
    dynamic_skill_hash: str | None = None
    proof: dict[str, list[str]] = field(default_factory=lambda: {
        "tests": [],
        "changed_files": [],
        "trace_ids": [],
    })
    blockers: list[dict[str, str]] = field(default_factory=list)


def infer_task_identity(
    body: dict[str, Any],
    session_id: str = "global",
    project_root: str = "",
) -> dict[str, str]:
    user_text = _last_user_text(body) or _flatten_text(body.get("input")) or "(unknown task)"
    title = _title(user_text)
    digest_src = "\n".join([session_id or "global", project_root or "", title])
    task_id = "tsk_" + hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]
    return {"task_id": task_id, "title": title}


def create_or_update_task(
    body: dict[str, Any],
    plan: Any = None,
    session_id: str = "global",
    project_root: str = "",
    state: str = "running",
) -> TaskRecord:
    identity = infer_task_identity(body, session_id=session_id, project_root=project_root)
    return TaskRecord(
        task_id=identity["task_id"],
        session_id=session_id,
        project_root=project_root,
        title=identity["title"],
        state=state,
        task_type=str(getattr(plan, "task_type", "unknown") or "unknown"),
        acceptance=_acceptance_for_plan(plan),
        recommended_skills=list(getattr(plan, "recommended_skills", []) or []),
        recommended_mcp=list(getattr(plan, "recommended_mcp", []) or []),
        dynamic_skill_hash=_dynamic_skill_hash(getattr(plan, "dynamic_skill", None)),
    )


def mark_blocked(
    record: TaskRecord,
    reason: str,
    recovery_action: str | None = None,
) -> TaskRecord:
    blocker = {"reason": reason}
    if recovery_action:
        blocker["recovery_action"] = recovery_action
    return replace(record, state="blocked", blockers=[*record.blockers, blocker])


def add_proof(
    record: TaskRecord,
    *,
    tests: list[str] | None = None,
    changed_files: list[str] | None = None,
    trace_ids: list[str] | None = None,
) -> TaskRecord:
    proof = {
        "tests": [*record.proof.get("tests", []), *(tests or [])],
        "changed_files": [*record.proof.get("changed_files", []), *(changed_files or [])],
        "trace_ids": [*record.proof.get("trace_ids", []), *(trace_ids or [])],
    }
    return replace(record, proof=proof)


def snapshot(records: list[TaskRecord]) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    tasks = []
    for record in records:
        by_state[record.state] = by_state.get(record.state, 0) + 1
        tasks.append({
            "task_id": record.task_id,
            "session_id": record.session_id,
            "project_root": record.project_root,
            "title": record.title,
            "state": record.state,
            "task_type": record.task_type,
            "recommended_skills": record.recommended_skills,
            "recommended_mcp": record.recommended_mcp,
            "dynamic_skill_hash": record.dynamic_skill_hash,
            "proof": record.proof,
            "blockers": record.blockers,
        })
    return {"total": len(records), "by_state": by_state, "tasks": tasks}


def _last_user_text(body: dict[str, Any]) -> str:
    src = body.get("input") or body.get("messages") or []
    if isinstance(src, str):
        return src
    if not isinstance(src, list):
        return ""
    for item in reversed(src):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        text = _flatten_text(item.get("content"))
        if text:
            return text
    return ""


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _title(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:80]


def _acceptance_for_plan(plan: Any) -> list[str]:
    acceptance = getattr(plan, "acceptance", None)
    if isinstance(acceptance, list):
        return [str(item) for item in acceptance]
    return []


def _dynamic_skill_hash(dynamic_skill: Any) -> str | None:
    if not dynamic_skill:
        return None
    if isinstance(dynamic_skill, dict):
        value = dynamic_skill.get("content_hash") or dynamic_skill.get("hash")
        return str(value) if value else None
    value = getattr(dynamic_skill, "content_hash", None)
    return str(value) if value else None
