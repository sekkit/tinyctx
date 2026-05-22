def test_infer_task_identity_is_stable_and_uses_user_text():
    from tinyctx.task_supervisor import infer_task_identity

    body = {
        "input": [
            {"role": "developer", "content": "follow repo rules"},
            {"role": "user", "content": "Implement visual config center with tests"},
        ]
    }

    first = infer_task_identity(body, session_id="s1", project_root="C:/repo")
    second = infer_task_identity(body, session_id="s1", project_root="C:/repo")

    assert first == second
    assert first["task_id"].startswith("tsk_")
    assert first["title"] == "Implement visual config center with tests"


def test_create_or_update_task_copies_plan_fields():
    from tinyctx.task_supervisor import create_or_update_task

    class Plan:
        task_type = "coding"
        recommended_skills = ["cc-tdd", "cc-work"]
        recommended_mcp = ["context-mode"]
        dynamic_skill = {"content_hash": "abc123"}
        execution_mode = "parallel_subagents"
        execution_reason = "independent checks"
        parallel_subtasks = [
            {"title": "tests", "agent": "reviewer", "prompt": "check tests"},
        ]

    record = create_or_update_task(
        {"input": "Fix the failing dashboard test"},
        plan=Plan(),
        session_id="sid",
        project_root="C:/repo",
    )

    assert record.state == "running"
    assert record.task_type == "coding"
    assert record.recommended_skills == ["cc-tdd", "cc-work"]
    assert record.recommended_mcp == ["context-mode"]
    assert record.dynamic_skill_hash == "abc123"
    assert record.execution_mode == "parallel_subagents"
    assert record.execution_reason == "independent checks"
    assert record.parallel_subtasks == [
        {"title": "tests", "agent": "reviewer", "prompt": "check tests"},
    ]
    assert "dashboard test" in record.title


def test_mark_blocked_and_add_proof_are_immutable():
    from tinyctx.task_supervisor import add_proof, create_or_update_task, mark_blocked

    original = create_or_update_task({"input": "Run smoke tests"})
    blocked = mark_blocked(original, "pytest failed", recovery_action="fix regression")
    proven = add_proof(
        blocked,
        tests=["pytest tests/test_dashboard.py -q"],
        changed_files=["tinyctx/dashboard.py"],
        trace_ids=["rq_123"],
    )

    assert original.state == "running"
    assert original.blockers == []
    assert blocked.state == "blocked"
    assert blocked.blockers == [{"reason": "pytest failed", "recovery_action": "fix regression"}]
    assert proven.proof["tests"] == ["pytest tests/test_dashboard.py -q"]
    assert proven.proof["changed_files"] == ["tinyctx/dashboard.py"]
    assert proven.proof["trace_ids"] == ["rq_123"]


def test_snapshot_summarizes_records_for_dashboard():
    from tinyctx.task_supervisor import create_or_update_task, snapshot

    records = [
        create_or_update_task({"input": "Design UI"}, state="running"),
        create_or_update_task({"input": "Write docs"}, state="done"),
    ]

    data = snapshot(records)

    assert data["total"] == 2
    assert data["by_state"] == {"running": 1, "done": 1}
    assert data["tasks"][0]["task_id"].startswith("tsk_")
    assert "title" in data["tasks"][0]
    assert data["tasks"][0]["execution_mode"] == "serial"
