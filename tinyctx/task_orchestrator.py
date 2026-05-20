from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from tinyctx.skill_catalog import default_catalog


@dataclass
class TaskPlan:
    task_type: str
    confidence: float
    recommended_skills: list[str]
    recommended_mcp: list[str]
    dynamic_skill_needed: bool
    dynamic_skill: Any
    routing_hint: str
    constraints: list[str]
    rationale: str


def plan_task(
    body: str,
    catalog: dict[str, Any] | None = None,
    local_planner: Callable[..., Any] | None = None,
    min_confidence: float = 0.62,
) -> TaskPlan:
    active_catalog = catalog if catalog is not None else default_catalog()

    if local_planner is not None:
        try:
            planned = _coerce_plan(_call_planner(local_planner, body, active_catalog))
            if planned.confidence >= min_confidence:
                return planned
        except Exception:
            pass

    return _fallback_plan(body, min_confidence)


def _call_planner(
    local_planner: Callable[..., Any],
    body: str,
    catalog: dict[str, Any],
) -> Any:
    try:
        return local_planner(body, catalog)
    except TypeError:
        return local_planner(body)


def _coerce_plan(raw: Any) -> TaskPlan:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError("planner output must be a dict or JSON object string")

    return TaskPlan(
        task_type=str(raw.get("task_type") or "unknown"),
        confidence=_float_or_zero(raw.get("confidence")),
        recommended_skills=_string_list(raw.get("recommended_skills")),
        recommended_mcp=_string_list(raw.get("recommended_mcp")),
        dynamic_skill_needed=bool(raw.get("dynamic_skill_needed", False)),
        dynamic_skill=raw.get("dynamic_skill"),
        routing_hint=str(raw.get("routing_hint") or "auto"),
        constraints=_string_list(raw.get("constraints")),
        rationale=str(raw.get("rationale") or "local planner"),
    )


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _fallback_plan(body: str, confidence: float) -> TaskPlan:
    task_type = _classify(body)
    if task_type in {"coding", "debug"}:
        return TaskPlan(
            task_type=task_type,
            confidence=confidence,
            recommended_skills=["cc-tdd", "cc-work"],
            recommended_mcp=["context-mode"],
            dynamic_skill_needed=False,
            dynamic_skill=None,
            routing_hint="auto",
            constraints=["test-first", "use context-mode for large searches"],
            rationale=f"fallback rules matched {task_type} task",
        )
    if task_type == "design":
        return TaskPlan(
            task_type="design",
            confidence=confidence,
            recommended_skills=["cc-design"],
            recommended_mcp=["browser"],
            dynamic_skill_needed=False,
            dynamic_skill=None,
            routing_hint="auto",
            constraints=["verify visual changes in browser"],
            rationale="fallback rules matched design task",
        )
    if task_type == "research":
        return TaskPlan(
            task_type="research",
            confidence=confidence,
            recommended_skills=["cc-research"],
            recommended_mcp=["context-mode"],
            dynamic_skill_needed=False,
            dynamic_skill=None,
            routing_hint="auto",
            constraints=["use context-mode for large searches"],
            rationale="fallback rules matched research task",
        )
    return TaskPlan(
        task_type="unknown",
        confidence=0.0,
        recommended_skills=[],
        recommended_mcp=["context-mode"],
        dynamic_skill_needed=True,
        dynamic_skill=None,
        routing_hint="auto",
        constraints=["use context-mode for large searches"],
        rationale="fallback rules found no specific match",
    )


def _classify(body: str) -> str:
    text = (body or "").lower()
    if _contains_any(text, ["fix", "failing", "failure", "bug", "debug", "error", "traceback", "pytest"]):
        return "debug"
    if _contains_any(text, ["implement", "code", "refactor", "test", "feature", "function", "class"]):
        return "coding"
    if _contains_any(text, ["design", "ui", "mockup", "prototype", "figma", "visual", "browser"]):
        return "design"
    if _contains_any(text, ["research", "summarize", "architecture", "docs", "documentation", "analyze"]):
        return "research"
    return "unknown"


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)
