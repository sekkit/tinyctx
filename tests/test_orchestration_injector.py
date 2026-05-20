from tinyctx.task_orchestrator import TaskPlan


def test_inject_task_plan_appends_bounded_instruction_block():
    from tinyctx.orchestration_injector import inject_task_plan

    body = {"instructions": "base instructions", "input": "Implement a feature"}
    plan = TaskPlan(
        task_type="coding",
        confidence=0.9,
        recommended_skills=["cc-tdd", "cc-work"],
        recommended_mcp=["context-mode"],
        dynamic_skill_needed=False,
        dynamic_skill=None,
        routing_hint="auto",
        constraints=["test-first"],
        rationale="coding task",
    )

    out, info = inject_task_plan(body, plan, max_chars=800)

    assert out is not body
    assert "base instructions" in out["instructions"]
    assert "<!-- tinyctx-orchestrator:start -->" in out["instructions"]
    assert "Recommended skills: cc-tdd, cc-work" in out["instructions"]
    assert "Recommended MCP: context-mode" in out["instructions"]
    assert info["injected"] is True
    assert info["chars"] <= 800


def test_inject_task_plan_is_idempotent_and_can_include_dynamic_skill():
    from tinyctx.orchestration_injector import inject_task_plan

    plan = TaskPlan(
        task_type="unknown",
        confidence=0.8,
        recommended_skills=[],
        recommended_mcp=["context-mode"],
        dynamic_skill_needed=True,
        dynamic_skill={"content_hash": "abc123"},
        routing_hint="auto",
        constraints=[],
        rationale="no matching skill",
    )
    body = {"instructions": "base"}

    once, first = inject_task_plan(body, plan, dynamic_skill_text="## tinyctx Dynamic Skill\nScope: current task only.")
    twice, second = inject_task_plan(once, plan, dynamic_skill_text="SHOULD NOT DUPLICATE")

    assert first["dynamic_skill_hash"] == "abc123"
    assert second["injected"] is False
    assert twice["instructions"].count("tinyctx-orchestrator:start") == 1
    assert "SHOULD NOT DUPLICATE" not in twice["instructions"]
