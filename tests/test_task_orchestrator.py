from __future__ import annotations

import dataclasses
import json

from tinyctx.task_orchestrator import TaskPlan, plan_task


def test_task_plan_dataclass_fields_match_contract():
    assert dataclasses.is_dataclass(TaskPlan)

    field_names = {field.name for field in dataclasses.fields(TaskPlan)}

    assert field_names == {
        "task_type",
        "confidence",
        "recommended_skills",
        "recommended_mcp",
        "dynamic_skill_needed",
        "dynamic_skill",
        "routing_hint",
        "constraints",
        "rationale",
    }


def test_plan_task_uses_confident_dict_local_planner():
    def local_planner(body, catalog):
        return {
            "task_type": "review",
            "confidence": 0.91,
            "recommended_skills": ["cc-work"],
            "recommended_mcp": ["serena"],
            "dynamic_skill_needed": False,
            "dynamic_skill": None,
            "routing_hint": "local",
            "constraints": ["keep diff focused"],
            "rationale": f"planned for {body[:6]} using {bool(catalog)}",
        }

    plan = plan_task("review this implementation", local_planner=local_planner)

    assert plan.task_type == "review"
    assert plan.confidence == 0.91
    assert plan.recommended_skills == ["cc-work"]
    assert plan.recommended_mcp == ["serena"]
    assert plan.routing_hint == "local"
    assert plan.constraints == ["keep diff focused"]
    assert "planned for review" in plan.rationale


def test_plan_task_accepts_json_string_local_planner():
    payload = {
        "task_type": "docs",
        "confidence": 0.88,
        "recommended_skills": ["cc-work"],
        "recommended_mcp": ["context-mode"],
        "dynamic_skill_needed": True,
        "dynamic_skill": {"name": "docs-helper"},
        "routing_hint": "auto",
        "constraints": ["summarize only"],
        "rationale": "json planner",
    }

    plan = plan_task("document the workflow", local_planner=lambda *_: json.dumps(payload))

    assert plan.task_type == "docs"
    assert plan.dynamic_skill_needed is True
    assert plan.dynamic_skill == {"name": "docs-helper"}
    assert plan.rationale == "json planner"


def test_low_confidence_planner_falls_back_to_coding_rules():
    def low_confidence_planner(*_):
        return {"task_type": "unknown", "confidence": 0.2, "rationale": "unsure"}

    plan = plan_task("fix failing pytest for the router", local_planner=low_confidence_planner)

    assert plan.task_type == "debug"
    assert plan.recommended_skills == ["cc-tdd", "cc-work"]
    assert plan.recommended_mcp == ["context-mode"]
    assert "test-first" in plan.constraints
    assert "fallback" in plan.rationale


def test_invalid_planner_output_falls_back_to_design_rules():
    plan = plan_task(
        "design a high fidelity UI mockup and verify it in the browser",
        local_planner=lambda *_: "{not json",
    )

    assert plan.task_type == "design"
    assert plan.recommended_skills == ["cc-design"]
    assert plan.recommended_mcp == ["browser"]
    assert plan.routing_hint == "auto"


def test_planner_exception_falls_back_to_research_rules():
    def broken_planner(*_):
        raise RuntimeError("planner unavailable")

    plan = plan_task("research the docs and summarize architecture", local_planner=broken_planner)

    assert plan.task_type == "research"
    assert plan.recommended_skills == ["cc-research"]
    assert plan.recommended_mcp == ["context-mode"]
    assert "use context-mode for large searches" in plan.constraints
