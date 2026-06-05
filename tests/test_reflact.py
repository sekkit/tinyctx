"""Tests for tinyctx.reflact — skill optimization by trajectory-driven edit training."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_skill():
    return """# AGENTS.md - Sample Skill

## Core Rules
- Always read the file before editing it.
- Run tests after every code change.
- Use type hints in all Python functions.

## Shell Commands
- Use `pytest -x --tb=short` for test runs.
- Use `pip install -e ".[dev]"` for dev setup.
"""


@pytest.fixture
def sample_trajectory_events_failure():
    """A failure trajectory: user asked for a change, agent forgot to run tests."""
    return [
        {"event": "request_received", "phase": "received", "ts": 1000.0,
         "metrics": {}, "artifacts": {}, "tags": []},
        {"event": "tool_call", "phase": "executing", "ts": 1001.0,
         "metrics": {"tool": "edit_file"}, "artifacts": {"file": "src/main.py"}, "tags": []},
        {"event": "response_sent", "phase": "done", "ts": 1002.0,
         "metrics": {"passed": False, "error": "TypeError: missing type hint"},
         "artifacts": {}, "tags": []},
    ]


@pytest.fixture
def sample_trajectory_events_success():
    """A success trajectory: agent correctly ran tests."""
    return [
        {"event": "request_received", "phase": "received", "ts": 2000.0,
         "metrics": {}, "artifacts": {}, "tags": []},
        {"event": "tool_call", "phase": "executing", "ts": 2001.0,
         "metrics": {"tool": "edit_file"}, "artifacts": {"file": "src/main.py"}, "tags": []},
        {"event": "tool_call", "phase": "executing", "ts": 2002.0,
         "metrics": {"tool": "bash", "cmd": "pytest -x --tb=short"},
         "artifacts": {"exit_code": 0}, "tags": []},
        {"event": "response_sent", "phase": "done", "ts": 2003.0,
         "metrics": {"passed": True}, "artifacts": {}, "tags": []},
    ]


# ── reflect ────────────────────────────────────────────────────────────

class TestFormatTrajectory:
    def test_formats_events(self, sample_trajectory_events_failure):
        from tinyctx.reflact import fmt_trajectory
        result = fmt_trajectory(sample_trajectory_events_failure)
        assert "request_received" in result
        assert "tool_call" in result
        assert "response_sent" in result
        assert "passed" in result.lower()

    def test_empty_events(self):
        from tinyctx.reflact import fmt_trajectory
        result = fmt_trajectory([])
        assert result == ""

    def test_respects_max_chars(self, sample_trajectory_events_failure):
        from tinyctx.reflact import fmt_trajectory
        # Force truncation
        long_events = sample_trajectory_events_failure * 500
        result = fmt_trajectory(long_events, max_chars=200)
        assert len(result) <= 300  # allow some slack for truncation markers

    def test_format_rollout_result(self, sample_trajectory_events_failure):
        from tinyctx.reflact import fmt_rollout_result
        result = fmt_rollout_result(
            sample_trajectory_events_failure,
            score=0.3, label="case_1",
        )
        assert "case_1" in result
        assert "0.300" in result


class TestAnalyzeFailures:
    def test_calls_optimizer_with_prompts(self, sample_skill, sample_trajectory_events_failure):
        from tinyctx.reflact import analyze_failures

        def mock_optimizer(system, user, options=None):
            return (json.dumps({
                "reasoning": "Agent forgot to run tests",
                "edits": [{
                    "op": "replace",
                    "target": "Run tests after every code change.",
                    "content": "Run tests after every code change. If tests fail, read the error output before fixing.",
                    "reasoning": "Agent edited code but didn't verify with tests.",
                }]
            }), {})

        result = analyze_failures(
            sample_skill,
            [{"events": sample_trajectory_events_failure, "score": 0.3, "label": "fail_1"}],
            optimizer=mock_optimizer,
            minibatch_size=1,
        )
        assert result["n_edits"] == 1
        assert result["source_type"] == "failure"
        assert len(result["patches"]) == 1
        assert result["patches"][0]["edits"][0]["op"] == "replace"

    def test_no_failures_returns_empty(self, sample_skill):
        from tinyctx.reflact import analyze_failures
        result = analyze_failures(
            sample_skill, [],
            optimizer=lambda s, u, o=None: ("{}", {}),
        )
        assert result["n_edits"] == 0

    def test_optimizer_error_graceful(self, sample_skill, sample_trajectory_events_failure):
        from tinyctx.reflact import analyze_failures
        result = analyze_failures(
            sample_skill,
            [{"events": sample_trajectory_events_failure, "score": 0.3, "label": "fail_1"}],
            optimizer=lambda s, u, o=None: (_ for _ in ()).throw(RuntimeError("API error")),
            minibatch_size=1,
        )
        assert result["n_edits"] == 0
        assert len(result["patches"]) == 1
        assert "error" in result["patches"][0]


class TestAnalyzeSuccesses:
    def test_calls_optimizer_for_successes(self, sample_skill, sample_trajectory_events_success):
        from tinyctx.reflact import analyze_successes

        def mock_optimizer(system, user, options=None):
            return (json.dumps({
                "reasoning": "Good pattern: running tests",
                "edits": [{
                    "op": "append",
                    "target": "",
                    "content": "After a successful test run, note the count in response.",
                    "reasoning": "Agent ran tests correctly, reinforce this pattern.",
                }]
            }), {})

        result = analyze_successes(
            sample_skill,
            [{"events": sample_trajectory_events_success, "score": 1.0, "label": "success_1"}],
            optimizer=mock_optimizer,
            minibatch_size=1,
        )
        assert result["n_edits"] == 1
        assert result["source_type"] == "success"

    def test_no_successes_returns_empty(self, sample_skill):
        from tinyctx.reflact import analyze_successes
        result = analyze_successes(
            sample_skill, [],
            optimizer=lambda s, u, o=None: ("{}", {}),
        )
        assert result["n_edits"] == 0


class TestMergeEdits:
    def test_deduplicates_by_op_and_target(self):
        from tinyctx.reflact import merge_edits

        patches = [
            {"source_type": "failure", "edits": [
                {"op": "replace", "target": "rule A", "content": "fix A"},
                {"op": "append", "target": "", "content": "add B"},
            ]},
            {"source_type": "failure", "edits": [
                {"op": "replace", "target": "rule A", "content": "fix A v2"},  # duplicate
                {"op": "delete", "target": "rule C", "content": ""},
            ]},
        ]
        result = merge_edits(patches)
        assert len(result) == 3  # 2 unique from first + 1 unique from second

    def test_respects_max_edits(self):
        from tinyctx.reflact import merge_edits
        patches = [
            {"source_type": "failure", "edits": [
                {"op": "replace", "target": f"rule_{i}", "content": f"fix_{i}"}
                for i in range(10)
            ]},
        ]
        result = merge_edits(patches, max_edits=5)
        assert len(result) == 5

    def test_empty_patches(self):
        from tinyctx.reflact import merge_edits
        assert merge_edits([]) == []


# ── update ─────────────────────────────────────────────────────────────

class TestApplyEdit:
    def test_append(self, sample_skill):
        from tinyctx.reflact import apply_edit
        edit = {"op": "append", "content": "# New rule\nBe concise."}
        new_skill, report = apply_edit(sample_skill, edit)
        assert report["status"] == "applied_append"
        assert "# New rule" in new_skill
        assert "Be concise" in new_skill
        # Original content preserved
        assert "Always read the file" in new_skill

    def test_replace_exact(self, sample_skill):
        from tinyctx.reflact import apply_edit
        target = "Run tests after every code change."
        edit = {"op": "replace", "target": target, "content": "Run tests BEFORE every code change."}
        new_skill, report = apply_edit(sample_skill, edit)
        assert report["status"] == "applied_replace"
        assert "Run tests BEFORE" in new_skill
        assert target not in new_skill

    def test_replace_target_not_found(self, sample_skill):
        from tinyctx.reflact import apply_edit
        edit = {"op": "replace", "target": "NONEXISTENT TEXT", "content": "x"}
        new_skill, report = apply_edit(sample_skill, edit)
        assert report["status"] == "skipped_replace_target_not_found"
        assert new_skill == sample_skill

    def test_delete(self, sample_skill):
        from tinyctx.reflact import apply_edit
        target = "Use type hints in all Python functions."
        edit = {"op": "delete", "target": target}
        new_skill, report = apply_edit(sample_skill, edit)
        assert report["status"] == "applied_delete"
        assert target not in new_skill

    def test_insert_after(self, sample_skill):
        from tinyctx.reflact import apply_edit
        target = "Run tests after every code change."
        edit = {"op": "insert_after", "target": target, "content": "  Also check type coverage."}
        new_skill, report = apply_edit(sample_skill, edit)
        assert report["status"] == "applied_insert_after"
        assert "check type coverage" in new_skill

    def test_slow_update_region_protected(self):
        from tinyctx.reflact.update import SLOW_UPDATE_START, SLOW_UPDATE_END
        skill = f"""# Rules
Rule A.
{SLOW_UPDATE_START}
Protected rule.
{SLOW_UPDATE_END}
Rule B."""
        edit = {"op": "replace", "target": "Protected rule.", "content": "Hacked rule."}
        from tinyctx.reflact import apply_edit
        new_skill, report = apply_edit(skill, edit)
        assert report["status"] == "skipped_protected_slow_update_region"
        assert "Protected rule." in new_skill
        assert "Hacked rule." not in new_skill

    def test_append_before_slow_update(self):
        from tinyctx.reflact.update import SLOW_UPDATE_START, SLOW_UPDATE_END
        skill = f"Rule A.\n{SLOW_UPDATE_START}\nProtected.\n{SLOW_UPDATE_END}"
        edit = {"op": "append", "content": "New rule."}
        from tinyctx.reflact import apply_edit
        new_skill, report = apply_edit(skill, edit)
        assert report["status"] == "applied_append_before_slow_update"
        assert "Rule A." in new_skill
        assert "New rule." in new_skill
        assert SLOW_UPDATE_START in new_skill


class TestApplyPatch:
    def test_multiple_edits_in_order(self, sample_skill):
        from tinyctx.reflact import apply_patch
        edits = [
            {"op": "replace", "target": "Run tests after every code change.",
             "content": "Run tests BEFORE every code change."},
            {"op": "append", "content": "Extra rule: use ruff for linting."},
        ]
        new_skill, reports = apply_patch(sample_skill, edits)
        assert len(reports) == 2
        assert "Run tests BEFORE" in new_skill
        assert "ruff for linting" in new_skill

    def test_empty_edits(self, sample_skill):
        from tinyctx.reflact import apply_patch
        new_skill, reports = apply_patch(sample_skill, [])
        assert new_skill == sample_skill
        assert reports == []


class TestDescribeEdit:
    def test_describe_replace(self):
        from tinyctx.reflact import describe_edit
        desc = describe_edit({"op": "replace", "target": "old rule", "content": "new rule"})
        assert "replace" in desc.lower()
        assert "old rule" in desc

    def test_describe_append(self):
        from tinyctx.reflact import describe_edit
        desc = describe_edit({"op": "append", "content": "new rule text"})
        assert "append" in desc.lower()
        assert "new rule text" in desc


# ── trainer ─────────────────────────────────────────────────────────────

class TestTrainSkill:
    def test_trainer_runs_epochs(self, sample_skill):
        from tinyctx.reflact import TrainConfig, train_skill

        def mock_optimizer(system, user, options=None):
            return (json.dumps({
                "reasoning": "test",
                "edits": [{"op": "append", "content": "Rule from training.", "target": ""}],
            }), {})

        def mock_rollout_fn(skill):
            # Return 10 fake rollouts with mixed scores
            results = []
            for i in range(10):
                results.append({
                    "events": [
                        {"event": "request", "phase": "received", "metrics": {}, "artifacts": {}},
                        {"event": "response", "phase": "done", "metrics": {"score": 0.5 + i * 0.05}, "artifacts": {}},
                    ],
                    "score": 0.5 + i * 0.05,
                    "label": f"case_{i}",
                })
            return results

        with tempfile.TemporaryDirectory() as tmp:
            skill_path = Path(tmp) / "test_skill.md"
            skill_path.write_text(sample_skill)

            cfg = TrainConfig(
                skill_path=str(skill_path),
                out_dir=str(Path(tmp) / "out"),
                epochs=2,
                accumulation=2,
                batch_size=4,
                edit_budget=3,
                minibatch_size=2,
                validation_split=0.3,
                verbose=False,
            )

            result = train_skill(
                cfg,
                optimizer=mock_optimizer,
                rollout_fn=mock_rollout_fn,
            )

            assert result.best_skill
            assert result.steps_completed > 0
            assert result.epochs_completed >= 1
            assert len(result.history) > 0
            # Check output files
            out_dir = Path(tmp) / "out"
            assert (out_dir / "best_skill.md").exists()
            assert (out_dir / "train_history.jsonl").exists()

    def test_empty_rollouts_handled(self, sample_skill):
        from tinyctx.reflact import TrainConfig, train_skill

        def mock_optimizer(system, user, options=None):
            return ("{}", {})

        def mock_rollout_fn(skill):
            return []  # no cases

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                skill_path=str(Path(tmp) / "empty.md"),
                out_dir=str(Path(tmp) / "out"),
                epochs=1, accumulation=1, batch_size=1,
                edit_budget=3, validation_split=0.3, verbose=False,
            )
            Path(cfg.skill_path).write_text(sample_skill)

            result = train_skill(cfg, optimizer=mock_optimizer, rollout_fn=mock_rollout_fn)
            assert result.best_skill == sample_skill  # unchanged


class TestLoadSkill:
    def test_loads_from_agents_md(self):
        from tinyctx.reflact import load_skill_from_agents_md
        with tempfile.TemporaryDirectory() as tmp:
            agents = Path(tmp) / "AGENTS.md"
            agents.write_text("# Custom rules\nDo X. Do Y.")
            result = load_skill_from_agents_md(Path(tmp))
            assert "Do X" in result

    def test_fallback_to_template(self):
        from tinyctx.reflact import load_skill_from_agents_md
        with tempfile.TemporaryDirectory() as tmp:
            result = load_skill_from_agents_md(Path(tmp))
            # Should load the bundled template
            assert result
            assert len(result) > 50
