from tinyctx.config import Config
from tinyctx.trace import RequestTrace


def test_apply_orchestration_injects_plan_and_populates_trace():
    from tinyctx.orchestration_runtime import apply_orchestration

    cfg = Config()
    trace = RequestTrace()
    body = {"instructions": "base", "input": "Implement dashboard tests"}

    out, record = apply_orchestration(
        body,
        cfg=cfg,
        trace=trace,
        session_id="sid",
        project_root="C:/repo",
    )

    assert "<!-- tinyctx-orchestrator:start -->" in out["instructions"]
    assert trace.orchestrator_injected is True
    assert trace.orchestrator_task_type in {"coding", "debug"}
    assert trace.orchestrator_skills
    assert trace.task_id == record.task_id
    assert record.state == "running"


def test_apply_orchestration_respects_disabled_config():
    from tinyctx.orchestration_runtime import apply_orchestration

    cfg = Config()
    cfg.orchestrator_enabled = False
    trace = RequestTrace()
    body = {"instructions": "base", "input": "Implement dashboard tests"}

    out, record = apply_orchestration(body, cfg=cfg, trace=trace)

    assert out == body
    assert record is None
    assert trace.orchestrator_injected is False


def test_apply_orchestration_can_inject_valid_dynamic_skill_from_planner():
    from tinyctx.orchestration_runtime import apply_orchestration

    cfg = Config()
    trace = RequestTrace()

    def planner(_body, _catalog):
        return {
            "task_type": "unknown",
            "confidence": 0.95,
            "recommended_skills": [],
            "recommended_mcp": ["context-mode"],
            "dynamic_skill_needed": True,
            "routing_hint": "auto",
            "constraints": ["verify output"],
            "rationale": "no matching local skill",
        }

    out, record = apply_orchestration(
        {"instructions": "base", "input": "Plan unusual migration"},
        cfg=cfg,
        trace=trace,
        local_planner=planner,
    )

    assert "tinyctx Dynamic Skill" in out["instructions"]
    assert trace.orchestrator_dynamic_skill_hash
    assert record.dynamic_skill_hash == trace.orchestrator_dynamic_skill_hash
