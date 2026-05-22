"""Instruction injection for tinyctx task orchestration decisions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .task_orchestrator import TaskPlan


START = "<!-- tinyctx-orchestrator:start -->"
END = "<!-- tinyctx-orchestrator:end -->"


def inject_task_plan(
    body: dict[str, Any],
    plan: TaskPlan,
    *,
    dynamic_skill_text: str | None = None,
    max_chars: int = 2000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    instructions = body.get("instructions")
    if not isinstance(instructions, str):
        instructions = ""
    dynamic_hash = _dynamic_skill_hash(plan.dynamic_skill)
    if START in instructions:
        return deepcopy(body), {
            "injected": False,
            "chars": 0,
            "dynamic_skill_hash": dynamic_hash,
        }

    block = _render_block(plan, dynamic_skill_text=dynamic_skill_text)
    if max_chars > 0 and len(block) > max_chars:
        block = block[: max(0, max_chars - len(END) - 6)].rstrip() + "\n...\n" + END
    new_body = deepcopy(body)
    new_body["instructions"] = (
        instructions.rstrip() + "\n\n" + block if instructions.strip() else block
    )
    return new_body, {
        "injected": True,
        "chars": len(block),
        "dynamic_skill_hash": dynamic_hash,
    }


def _render_block(plan: TaskPlan, *, dynamic_skill_text: str | None) -> str:
    lines = [
        START,
        "Task orchestration guidance (current task only; lower priority than system/developer/AGENTS instructions).",
        f"Task type: {plan.task_type} (confidence {plan.confidence:.2f})",
        "Recommended skills: " + _join(plan.recommended_skills),
        "Recommended MCP: " + _join(plan.recommended_mcp),
        "Routing hint: " + (plan.routing_hint or "auto"),
        "Execution mode: " + getattr(plan, "execution_mode", "serial"),
        "Execution reason: " + getattr(plan, "execution_reason", ""),
    ]
    parallel_subtasks = list(getattr(plan, "parallel_subtasks", []) or [])
    if parallel_subtasks:
        lines.append("Parallel subtasks:")
        lines.extend(_format_parallel_subtask(item) for item in parallel_subtasks)
    if plan.constraints:
        lines.append("Constraints:")
        lines.extend(f"- {item}" for item in plan.constraints)
    if plan.rationale:
        lines.append("Rationale: " + plan.rationale)
    if dynamic_skill_text:
        lines.append("")
        lines.append(dynamic_skill_text.strip())
    lines.append(END)
    return "\n".join(lines)


def _join(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _format_parallel_subtask(item: dict[str, str]) -> str:
    title = str(item.get("title") or "Subtask").strip()
    agent = str(item.get("agent") or "worker").strip()
    prompt = str(item.get("prompt") or title).strip()
    return f"- {title} -> {agent}: {prompt}"


def _dynamic_skill_hash(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("content_hash") or value.get("hash")
        return str(raw) if raw else None
    raw = getattr(value, "content_hash", None)
    return str(raw) if raw else None
