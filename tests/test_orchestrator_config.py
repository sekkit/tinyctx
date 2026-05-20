from pathlib import Path


def test_config_orchestrator_defaults_are_enabled():
    from tinyctx.config import Config

    cfg = Config()

    assert cfg.orchestrator_enabled is True
    assert cfg.orchestrator_dynamic_skill_enabled is True
    assert cfg.orchestrator_min_confidence == 0.62
    assert cfg.orchestrator_dynamic_skill_min_confidence == 0.78
    assert cfg.orchestrator_inject_max_chars == 2000


def test_load_config_reads_orchestrator_section(monkeypatch, tmp_path: Path):
    from tinyctx.config import load_config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[orchestrator]
enabled = false
dynamic_skill_enabled = false
min_confidence = 0.7
dynamic_skill_min_confidence = 0.9
inject_max_chars = 1200
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(cfg_path))

    cfg = load_config()

    assert cfg.orchestrator_enabled is False
    assert cfg.orchestrator_dynamic_skill_enabled is False
    assert cfg.orchestrator_min_confidence == 0.7
    assert cfg.orchestrator_dynamic_skill_min_confidence == 0.9
    assert cfg.orchestrator_inject_max_chars == 1200


def test_request_trace_carries_orchestrator_fields():
    from dataclasses import asdict

    from tinyctx.trace import RequestTrace

    trace = RequestTrace()
    trace.orchestrator_injected = True
    trace.orchestrator_task_type = "coding"
    trace.orchestrator_skills = ["cc-tdd"]
    trace.orchestrator_mcp = ["context-mode"]
    trace.task_state = "running"

    payload = asdict(trace)

    assert payload["orchestrator_injected"] is True
    assert payload["orchestrator_task_type"] == "coding"
    assert payload["task_state"] == "running"
