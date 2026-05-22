from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    execution_mode: str = "serial"
    execution_reason: str = "dependent task; execute in order"
    parallel_subtasks: list[dict[str, str]] = field(default_factory=list)


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

    execution_mode = _execution_mode(raw.get("execution_mode"))
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
        execution_mode=execution_mode,
        execution_reason=str(raw.get("execution_reason") or _default_execution_reason(execution_mode)),
        parallel_subtasks=_parallel_subtasks(raw.get("parallel_subtasks")),
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


def _execution_mode(value: Any) -> str:
    mode = str(value or "serial").strip().lower()
    if mode in {"serial", "parallel_subagents", "advisor_only"}:
        return mode
    return "serial"


def _default_execution_reason(mode: str) -> str:
    if mode == "parallel_subagents":
        return "independent subtasks can run in parallel"
    if mode == "advisor_only":
        return "single executor should consult advisor before proceeding"
    return "dependent task; execute in order"


def _parallel_subtasks(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value[:5]:
        if isinstance(item, str):
            title = item.strip()
            if title:
                out.append({"title": title, "agent": "worker", "prompt": title})
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        prompt = str(item.get("prompt") or item.get("task") or title).strip()
        agent = str(item.get("agent") or item.get("role") or "worker").strip()
        if title and prompt:
            out.append({"title": title, "agent": agent or "worker", "prompt": prompt})
    return out


def _fallback_plan(body: str, confidence: float) -> TaskPlan:
    task_type = _classify(body)
    execution_mode, execution_reason, parallel_subtasks = _execution_decision(body, task_type)
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
            execution_mode=execution_mode,
            execution_reason=execution_reason,
            parallel_subtasks=parallel_subtasks,
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
            execution_mode=execution_mode,
            execution_reason=execution_reason,
            parallel_subtasks=parallel_subtasks,
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
            execution_mode=execution_mode,
            execution_reason=execution_reason,
            parallel_subtasks=parallel_subtasks,
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
        execution_mode=execution_mode,
        execution_reason=execution_reason,
        parallel_subtasks=parallel_subtasks,
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


def _execution_decision(body: str, task_type: str) -> tuple[str, str, list[dict[str, str]]]:
    text = (body or "").lower()
    if task_type == "debug":
        return "serial", "dependent debugging loop; fix and verify in order", []
    explicit_parallel = _contains_any(
        text,
        [
            "parallel", "concurrent", "fan out", "subagent", "sub-agent",
            "并行", "同时", "分别",
        ],
    )
    independent = explicit_parallel or _contains_any(
        text,
        ["independent", "separately", "独立", "互不依赖"],
    )
    subtasks = _infer_parallel_subtasks(text)
    if independent and len(subtasks) >= 2:
        return (
            "parallel_subagents",
            "independent subtasks can run in parallel",
            subtasks,
        )
    return "serial", "dependent task; execute in order", []


def _infer_parallel_subtasks(text: str) -> list[dict[str, str]]:
    lanes = [
        ("API pass", ["api", "接口"], "Review API contracts and integration risks."),
        ("Test pass", ["test", "pytest", "测试"], "Review tests, gaps, and verification plan."),
        ("Docs pass", ["docs", "documentation", "文档"], "Review documentation and user-facing guidance."),
        ("Security pass", ["security", "auth", "安全", "权限"], "Review security, auth, and privacy risks."),
        ("Performance pass", ["performance", "latency", "性能", "延迟"], "Review performance and latency risks."),
        ("UI pass", ["ui", "ux", "visual", "界面"], "Review user experience and visual behavior."),
    ]
    out: list[dict[str, str]] = []
    for title, needles, prompt in lanes:
        if any(needle in text for needle in needles):
            out.append({"title": title, "agent": "worker", "prompt": prompt})
    if len(out) >= 2:
        return out[:5]
    return []
