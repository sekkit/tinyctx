"""Runtime glue for tinyctx task orchestration."""
from __future__ import annotations

from typing import Any, Callable

from .config import Config
from .dynamic_skill import (
    DynamicSkill,
    build_dynamic_skill,
    render_dynamic_skill,
    validate_dynamic_skill,
)
from .orchestration_injector import inject_task_plan
from .skill_catalog import default_catalog
from .task_orchestrator import TaskPlan, plan_task
from .task_supervisor import TaskRecord, create_or_update_task
from .trace import RequestTrace


DynamicPlanner = Callable[[str, str], Any]


def apply_orchestration(
    body: dict[str, Any],
    *,
    cfg: Config,
    trace: RequestTrace | None = None,
    session_id: str = "global",
    project_root: str = "",
    catalog: dict[str, Any] | None = None,
    local_planner: Callable[..., Any] | None = None,
    dynamic_skill_planner: DynamicPlanner | None = None,
) -> tuple[dict[str, Any], TaskRecord | None]:
    if not cfg.orchestrator_enabled:
        return body, None

    text = _task_text_from_body(body)
    active_catalog = catalog if catalog is not None else default_catalog()
    plan = plan_task(
        text,
        catalog=active_catalog,
        local_planner=local_planner,
        min_confidence=cfg.orchestrator_min_confidence,
    )

    dynamic_skill_text = ""
    if _should_inject_dynamic_skill(cfg, plan):
        dynamic_skill = _resolve_dynamic_skill(
            plan,
            task_text=text,
            planner=dynamic_skill_planner,
        )
        validation = validate_dynamic_skill(dynamic_skill)
        if validation["ok"]:
            plan.dynamic_skill = dynamic_skill
            dynamic_skill_text = render_dynamic_skill(
                dynamic_skill,
                max_chars=max(0, min(1200, cfg.orchestrator_inject_max_chars)),
            )

    out, inject_info = inject_task_plan(
        body,
        plan,
        dynamic_skill_text=dynamic_skill_text,
        max_chars=cfg.orchestrator_inject_max_chars,
    )
    record = create_or_update_task(
        out,
        plan=plan,
        session_id=session_id,
        project_root=project_root,
        state="running",
    )

    if trace is not None and cfg.orchestrator_trace_decisions:
        trace.orchestrator_injected = bool(inject_info.get("injected"))
        trace.orchestrator_task_type = plan.task_type
        trace.orchestrator_confidence = float(plan.confidence)
        trace.orchestrator_skills = list(plan.recommended_skills)
        trace.orchestrator_mcp = list(plan.recommended_mcp)
        trace.orchestrator_dynamic_skill_hash = (
            str(inject_info.get("dynamic_skill_hash") or "")
        )
        trace.orchestrator_rationale = str(plan.rationale or "")
        trace.orchestrator_execution_mode = str(plan.execution_mode or "serial")
        trace.orchestrator_execution_reason = str(plan.execution_reason or "")
        trace.orchestrator_parallel_subtasks = list(plan.parallel_subtasks or [])
        trace.task_id = record.task_id
        trace.task_title = record.title
        trace.task_state = record.state

    return out, record


def _should_inject_dynamic_skill(cfg: Config, plan: TaskPlan) -> bool:
    return (
        cfg.orchestrator_dynamic_skill_enabled
        and bool(plan.dynamic_skill_needed)
        and float(plan.confidence) >= float(cfg.orchestrator_dynamic_skill_min_confidence)
    )


def _resolve_dynamic_skill(
    plan: TaskPlan,
    *,
    task_text: str,
    planner: DynamicPlanner | None,
) -> DynamicSkill:
    if isinstance(plan.dynamic_skill, DynamicSkill):
        return plan.dynamic_skill
    if isinstance(plan.dynamic_skill, dict):
        return build_dynamic_skill(
            task_text,
            _skill_gap(plan),
            planner=lambda _task, _gap: plan.dynamic_skill,
        )
    return build_dynamic_skill(task_text, _skill_gap(plan), planner=planner)


def _skill_gap(plan: TaskPlan) -> str:
    reason = (plan.rationale or "").strip()
    if reason:
        return reason
    if plan.recommended_skills:
        return "Need stricter execution guardrails for this task."
    return "No matching local skill found for this task."


def _task_text_from_body(body: dict[str, Any]) -> str:
    parts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        parts.append(instructions.strip())
    parts.append(_flatten_text(body.get("input")))
    if not parts:
        return ""
    return "\n".join(part for part in parts if part).strip()


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role in {"system", "developer"}:
                continue
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, list):
                for fragment in content:
                    if not isinstance(fragment, dict):
                        continue
                    text = fragment.get("text") or fragment.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return text if isinstance(text, str) else ""
    return ""
