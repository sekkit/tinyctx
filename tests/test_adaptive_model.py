from __future__ import annotations

from tinyctx import adaptive_model
from tinyctx.config import BackendCfg, Config
from tinyctx.router import Decision


def setup_function() -> None:
    adaptive_model.reset_state()


def test_backend_key_matches_decision_key_shape() -> None:
    backend = BackendCfg(
        base_url="http://local.test/v1",
        model="local-mdl",
        wire_api="chat",
    )
    decision = Decision(
        route="local",
        reason="test",
        target="http://local.test/v1/chat/completions",
        model="local-mdl",
        wire_api="chat",
    )
    assert adaptive_model.backend_key("local", backend) == adaptive_model.decision_key(decision)


def test_rolling_failure_rate_triggers_and_recovers() -> None:
    state = adaptive_model.AdaptiveRouteState()
    key = "local:chat:http://local/v1/chat/completions:m"

    state.record(key, ok=False, max_samples=4)
    state.record(key, ok=False, max_samples=4)
    health = state.record(key, ok=True, max_samples=4)

    assert health.calls == 3
    assert health.failures == 2

    unhealthy = state.health(key, min_calls=3, threshold=0.5)
    assert unhealthy.should_escalate is True

    state.record(key, ok=True, max_samples=4)
    state.record(key, ok=True, max_samples=4)
    recovered = state.health(key, min_calls=3, threshold=0.5)
    assert recovered.calls == 4
    assert recovered.failures == 1
    assert recovered.should_escalate is False


def test_local_health_uses_config_thresholds() -> None:
    cfg = Config()
    cfg.local.base_url = "http://local.test/v1"
    cfg.local.model = "local-mdl"
    cfg.local.wire_api = "chat"
    cfg.adaptive_model_min_calls = 2
    cfg.adaptive_model_failure_rate_threshold = 0.5
    cfg.adaptive_model_sample_size = 3

    decision = Decision(
        route="local",
        reason="test",
        target="http://local.test/v1/chat/completions",
        model="local-mdl",
        wire_api="chat",
    )
    adaptive_model.record_decision(decision, ok=False, max_samples=3)
    adaptive_model.record_decision(decision, ok=True, max_samples=3)

    health = adaptive_model.local_health(cfg)
    assert health.calls == 2
    assert health.failures == 1
    assert health.should_escalate is True
