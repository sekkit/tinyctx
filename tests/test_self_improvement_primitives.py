from tinyctx import (
    eval_harness,
    frontier,
    guardrail_registry,
    self_improvement,
    trajectory,
    workspace,
)


def test_workspace_creates_session_dirs_and_profile(tmp_path):
    ws = workspace.ensure_workspace("proj:/weird session", root=tmp_path)

    assert ws.session_id == "proj-weird-session"
    assert ws.public.is_dir()
    assert ws.private.is_dir()
    assert ws.logs.is_dir()

    path = workspace.save_context_profile(
        {"commands": ["pytest"], "budgets": {"tokens": 1000}},
        root=tmp_path,
    )
    loaded = workspace.load_context_profile(root=tmp_path)

    assert path.name == "context_profile.json"
    assert loaded["version"] == 1
    assert loaded["commands"] == ["pytest"]
    assert loaded["budgets"]["tokens"] == 1000
    assert workspace.list_session_ids(root=tmp_path) == ["proj-weird-session"]


def test_trajectory_records_and_summarizes_events(tmp_path):
    first = trajectory.record_event(
        "abc",
        "route_decision",
        root=tmp_path,
        phase="router",
        metrics={"passed": True, "tokens_saved": 42},
        tags=("replayable",),
    )
    trajectory.record_event(
        "abc",
        "sanitize_check",
        root=tmp_path,
        phase="sanitize",
        metrics={"passed": False, "error": "secret"},
    )

    events = trajectory.read_events("abc", root=tmp_path)
    summary = trajectory.summarize_events(events)

    assert first["session_id"] == "abc"
    assert [event["event"] for event in events] == ["route_decision", "sanitize_check"]
    assert summary["total"] == 2
    assert summary["failures"] == 1
    assert summary["by_phase"]["router"] == 1


def test_eval_harness_runs_cases_and_aggregates_failures():
    cases = [
        eval_harness.EvalCase("ok", {"value": 1}),
        eval_harness.EvalCase("bad", {"value": 0}),
        eval_harness.EvalCase("boom", {"value": -1}),
    ]

    def evaluator(case):
        if case.case_id == "boom":
            raise RuntimeError("exploded")
        return {
            "passed": case.input["value"] > 0,
            "metrics": {"score": float(case.input["value"] > 0)},
        }

    results = eval_harness.run_suite(cases, evaluator)
    aggregate = eval_harness.aggregate_results(results)

    assert [result.passed for result in results] == [True, False, False]
    assert results[2].error == "exploded"
    assert aggregate["total"] == 3
    assert aggregate["passed"] == 1
    assert aggregate["score"] == 1 / 3


def test_frontier_archives_candidates_and_selects_weighted_best(tmp_path):
    frontier.add_candidate(
        "abc",
        frontier.Candidate(
            candidate_id="router-v1",
            kind="router",
            payload={"threshold": 0.4},
            metrics={"quality": 0.8, "tokens_saved": 0.2},
        ),
        root=tmp_path,
    )
    frontier.add_candidate(
        "abc",
        frontier.Candidate(
            candidate_id="router-v2",
            kind="router",
            payload={"threshold": 0.7},
            metrics={"quality": 0.7, "tokens_saved": 0.9},
            parent_id="router-v1",
            generation=1,
        ),
        root=tmp_path,
    )

    candidates = frontier.read_candidates("abc", root=tmp_path, kind="router")
    best = frontier.best_candidate(candidates, {"quality": 1.0, "tokens_saved": 0.5})

    assert [candidate["candidate_id"] for candidate in candidates] == [
        "router-v1",
        "router-v2",
    ]
    assert best is not None
    assert best["candidate_id"] == "router-v2"
    assert best["weighted_score"] == 1.15


def test_self_improvement_loop_archives_eval_and_records_trajectory(tmp_path):
    candidate = frontier.Candidate(
        candidate_id="sanitize-v1",
        kind="sanitize",
        payload={"redact": True},
        metrics={"tokens_saved": 0.2},
    )
    cases = [
        eval_harness.EvalCase("safe", {"text": "hello"}),
        eval_harness.EvalCase("secret", {"text": "token"}),
    ]

    def evaluator(case):
        return {
            "passed": case.case_id == "safe",
            "score": 1.0 if case.case_id == "safe" else 0.0,
        }

    outcome = self_improvement.evaluate_candidate(
        "abc",
        candidate,
        cases,
        evaluator,
        root=tmp_path,
        min_pass_rate=0.5,
    )
    events = trajectory.read_events("abc", root=tmp_path)
    archived = frontier.read_candidates("abc", root=tmp_path, kind="sanitize")

    assert outcome["accepted"] is True
    assert outcome["aggregate"]["pass_rate"] == 0.5
    assert archived[0]["metrics"]["tokens_saved"] == 0.2
    assert archived[0]["metrics"]["failed"] == 1.0
    assert [event["event"] for event in events] == [
        "candidate_eval_started",
        "candidate_eval_completed",
    ]


def test_guardrail_registry_runs_stage_and_summarizes():
    registry = guardrail_registry.GuardrailRegistry()
    registry.register(
        guardrail_registry.GuardrailPlugin(
            name="has-session",
            stage="pre",
            check=lambda ctx: guardrail_registry.GuardCheckResult(
                name="has-session",
                passed=bool(ctx.get("session_id")),
            ),
        )
    )
    registry.register(
        guardrail_registry.GuardrailPlugin(
            name="no-secret",
            stage="pre",
            check=lambda ctx: guardrail_registry.GuardCheckResult(
                name="no-secret",
                passed="secret" not in str(ctx.get("body", "")),
                action="block",
                reason="secret-like body",
            ),
        )
    )

    results = registry.run({"session_id": "abc", "body": "contains secret"}, stage="pre")
    summary = guardrail_registry.summarize_results(results)

    assert [result.name for result in results] == ["has-session", "no-secret"]
    assert summary["failed"] == 1
    assert summary["blocking"] == ["no-secret"]


def test_proxy_log_records_trajectory_without_legacy_verbose(tmp_path, monkeypatch):
    from tinyctx import proxy

    monkeypatch.setenv("TINYCTX_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(proxy.CFG, "verbose", False)

    proxy._log(
        "route",
        session="proj:/abc",
        decision="local",
        est_tokens=123,
        body="should not be persisted",
    )

    events = trajectory.read_events("proj-abc", root=tmp_path)

    assert len(events) == 1
    assert events[0]["event"] == "route"
    assert events[0]["phase"] == "router"
    assert events[0]["metrics"]["est_tokens"] == 123
    assert events[0]["artifacts"]["body"] == "<redacted>"
