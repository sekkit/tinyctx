"""Unit-test the router decisions, the compaction fingerprint, and the
encrypted_content sanitizer. No network required."""
from __future__ import annotations

import pytest

from tinyctx import adaptive_model
from tinyctx.config import Config
from tinyctx.router import decide, goal_control_signal, is_compaction_request
from tinyctx.sanitize import strip_encrypted_content


CFG = Config()


@pytest.fixture(autouse=True)
def _reset_adaptive_model_state():
    adaptive_model.reset_state()
    yield
    adaptive_model.reset_state()


def test_compaction_fingerprint_redirects_to_local():
    body = {
        "model": "gpt-5.5",
        "instructions": (
            "Create a handoff summary for another LLM that will resume the "
            "task. Be concise and structured."
        ),
        "input": [{"role": "user", "content": "..."}],
    }
    d = decide(body, CFG)
    assert d.is_compaction is True
    assert d.route == "local", d.reason


def test_short_query_stays_local():
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": [{"type": "input_text",
                       "text": "rename foo() to bar()"}]}]}
    d = decide(body, CFG)
    assert d.route == "local", d.reason
    assert d.is_compaction is False


def test_huge_history_does_NOT_auto_escalate_by_default():
    """Aligned with Anthropic Advisor Strategy
    (claude.com/blog/the-advisor-strategy): the EXECUTOR MODEL decides
    when to escalate, infrastructure does not auto-escalate by byte
    count. Default config has all size/turn thresholds disabled.

    A huge body stays on local; the model invokes spawn_agent(advisor)
    when it actually needs strategic input."""
    big = "x" * 4_000_000  # ~1.1M est tokens — way past any old threshold
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": big}]}
    d = decide(body, CFG)
    assert d.route == "local", (
        f"default routing must NOT auto-escalate by size; got {d.route}: {d.reason}"
    )


def test_size_escalation_can_be_re_enabled_for_small_local_backends():
    """Users with a 32k-context LMStudio backend can opt back in by
    setting context_safe_fraction > 0 in config."""
    cfg = Config()
    # Simulate small local backend with size-based escalation explicitly enabled
    cfg.local.context_window = 32_000
    cfg.local.context_safe_fraction = 0.85  # opt-in
    big = "x" * 200_000  # ~55k est tokens, well past 32k×0.85=27k
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": big}]}
    d = decide(body, cfg)
    assert d.route == "frontier", (
        f"explicitly enabled size escalation should fire; got {d.route}: {d.reason}"
    )


def test_turn_count_escalation_can_be_re_enabled():
    """Same opt-in story for turn count."""
    cfg = Config()
    cfg.escalate_turn_count = 15
    body = {
        "input": [
            {"role": "user", "content": f"turn {i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"reply {i}"}
            for i in range(40)
        ]
    }
    d = decide(body, cfg)
    assert d.route == "frontier"
    assert "turn_count" in d.reason


def test_goal_command_routes_to_frontier_for_contract_quality():
    cfg = Config()
    body = {
        "input": [{
            "role": "user",
            "content": (
                "/goal compile this into a GOAL.md with done_when, "
                "scorecard, and verification commands"
            ),
        }],
    }
    d = decide(body, cfg)
    assert d.route == "frontier", d.reason
    assert "goal-control" in d.reason


def test_goal_control_does_not_fire_on_tool_result_roundtrip():
    body = {
        "input": [
            {"role": "user", "content": "/goal fix the router"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "GOAL.md contains done_when and scorecard",
            },
        ],
    }
    assert goal_control_signal(body) == ""
    d = decide(body, Config())
    assert d.route == "local", d.reason


def test_goal_control_frontier_can_be_disabled():
    cfg = Config()
    cfg.goal_control_frontier_enabled = False
    body = {"input": [{"role": "user", "content": "/goal fix the router"}]}
    d = decide(body, cfg)
    assert d.route == "local", d.reason


def test_force_route_overrides_everything():
    cfg = Config()
    cfg.force_route = "frontier"
    body = {"input": [{"role": "user", "content": "tiny"}]}
    d = decide(body, cfg)
    assert d.route == "frontier"

    cfg.force_route = "local"
    body = {"input": [{"role": "user", "content": "x" * 1_000_000}]}
    d = decide(body, cfg)
    assert d.route == "local"


def test_strip_encrypted_content_removes_from_reasoning():
    body = {
        "input": [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "encrypted_content": "OPAQUE"},
            {"role": "assistant", "content": [
                {"type": "reasoning", "encrypted_content": "ALSO_OPAQUE"},
                {"type": "output_text", "text": "world"},
            ]},
        ],
        "include": ["reasoning.encrypted_content", "tool_calls"],
    }
    out = strip_encrypted_content(body)
    assert "encrypted_content" not in out["input"][1]
    nested = out["input"][2]["content"]
    assert all("encrypted_content" not in c for c in nested if isinstance(c, dict))
    assert "reasoning.encrypted_content" not in out["include"]
    # original unchanged (we deep-copy)
    assert body["input"][1]["encrypted_content"] == "OPAQUE"


def test_compaction_phrase_variants():
    # These are the three invariant phrases from codex's handoff prompt.
    assert is_compaction_request("Create a handoff summary for another LLM")
    assert is_compaction_request("for another LLM that will resume the task next")
    assert is_compaction_request("help the next LLM seamlessly continue the work")
    assert not is_compaction_request("rename foo to bar")
    assert not is_compaction_request("write a handoff document for the team")  # no fingerprint


def test_context_window_drives_escalation_when_set():
    """When cfg.local.context_window is set, the router uses it (× safe
    fraction) instead of the legacy absolute escalate_input_tokens."""
    from tinyctx.config import Config, BackendCfg
    cfg = Config()
    cfg.local = BackendCfg(
        base_url="x", model="m", wire_api="chat",
        context_window=1_000_000, context_safe_fraction=0.85,
    )
    # 100K input → well below 850K cap → local
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 360_000}]}]}
    d = decide(body, cfg)
    assert d.route == "local", d.reason

    # 900K input → above 850K cap → frontier
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 3_500_000}]}]}
    d = decide(body, cfg)
    assert d.route == "frontier", d.reason
    assert "of local ctx 1000000" in d.reason


def test_legacy_threshold_used_when_context_window_unset():
    """If context_window=0, fall back to absolute escalate_input_tokens."""
    from tinyctx.config import Config, BackendCfg
    cfg = Config()
    cfg.local = BackendCfg(base_url="x", model="m", wire_api="chat",
                           context_window=0)
    cfg.escalate_input_tokens = 60_000
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 250_000}]}]}
    d = decide(body, cfg)
    assert d.route == "frontier"
    assert "60000" in d.reason


# ─── P5: Router class (consolidated route decision) ──────────────────────


def _make_router_cfg(**overrides):
    """Build a Config with a known-shape local + frontier for Router tests."""
    from tinyctx.config import BackendCfg, Config
    cfg = Config()
    cfg.local = BackendCfg(
        base_url="http://local.test/v1",
        api_key_env=None,
        model="local-mdl",
        wire_api="chat",
        timeout_s=180.0,
        context_window=1_000_000,
        context_safe_fraction=0.85,
    )
    cfg.frontier = BackendCfg(
        base_url="http://frontier.test/v1",
        api_key_env=None,
        model="frontier-mdl",
        wire_api="responses",
        timeout_s=300.0,
        context_window=272_000,
    )
    cfg.escalate_on_error_streak = 2
    cfg.redirect_compaction_to_local = True
    cfg.self_classify_threshold = 0.7
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _ctx(**overrides):
    """Build a RouteContext with sensible defaults for Router tests."""
    from tinyctx.router import RouteContext
    defaults = dict(
        body={"input": [{"role": "user", "content": "hi"}]},
        proj_sid="proj",
        conv_sid="conv",
        turn_count=1,
        est_tokens=100,
        requested_model="",
        force_route=None,
        error_streak=0,
        is_compaction=False,
        classify_p=0.0,
        classify_reason="",
    )
    defaults.update(overrides)
    return RouteContext(**defaults)


def test_router_decision_dataclass_has_target_headers_wire_api():
    """The new Decision must expose target URL + model + headers + wire_api
    + timeout so proxy.py can forward without separately resolving backend."""
    from tinyctx.router import Decision
    d = Decision(
        route="local", reason="x",
        target="http://t/", model="m", headers={"a": "b"},
        wire_api="chat", timeout_s=10.0, is_compaction=False,
    )
    assert d.target == "http://t/"
    assert d.headers["a"] == "b"
    assert d.wire_api == "chat"
    assert d.timeout_s == 10.0


def test_router_default_returns_local_for_small_inputs():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx())
    assert d.route == "local"
    assert d.target.startswith("http://local.test")
    assert d.model == "local-mdl"
    assert d.wire_api == "chat"
    assert d.timeout_s == 180.0


def test_router_compaction_rule_routes_to_local():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(is_compaction=True))
    assert d.route == "local"
    assert d.is_compaction is True
    assert "compaction" in d.reason.lower()


def test_router_compaction_beats_force_route_to_frontier():
    """Compaction redirect_to_local must win over a force_route=frontier
    from the guard pipeline (matches current proxy.py behavior: compaction
    handoff goes local even when an empty-response flag is pending)."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(is_compaction=True, force_route="frontier"))
    assert d.route == "local"
    assert d.is_compaction is True


def test_router_force_route_rule_to_frontier():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(force_route="frontier"))
    assert d.route == "frontier"
    assert d.target.startswith("http://frontier.test")
    assert d.model == "frontier-mdl"
    assert d.wire_api == "responses"


def test_router_force_route_beats_explicit_local_model():
    """A force_route=frontier from the guard pipeline (set by
    ForceFrontierGuard) overrides a client-side `model=tinyctx-local`."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(force_route="frontier", requested_model="tinyctx-local"))
    assert d.route == "frontier"


def test_router_explicit_model_local():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(requested_model="tinyctx-local"))
    assert d.route == "local"
    assert "client" in d.reason.lower() or "explicit" in d.reason.lower()


def test_router_explicit_model_frontier():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(requested_model="tinyctx-frontier"))
    assert d.route == "frontier"


def test_router_goal_control_routes_to_frontier():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(body={
        "input": [{
            "role": "user",
            "content": "Forge a goal contract with done_when checks",
        }],
    }))
    assert d.route == "frontier"
    assert "goal-control" in d.reason


def test_router_explicit_local_beats_goal_control():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(
        requested_model="tinyctx-local",
        body={"input": [{"role": "user", "content": "/goal write GOAL.md"}]},
    ))
    assert d.route == "local"


def test_router_explicit_model_beats_capacity_and_classify():
    """If the client explicitly asked for tinyctx-frontier, honor that even
    on a small-token request (don't downgrade to local). Mirrors
    `forced_by_client_model` semantics in proxy.py."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(requested_model="tinyctx-frontier",
                       est_tokens=50, classify_p=0.0))
    assert d.route == "frontier"


def test_router_explicit_local_beats_capacity_escalation():
    """If client says tinyctx-local but est_tokens > capacity, the explicit
    request wins (client knows what they're doing — same precedence as
    current proxy.py where requested_model is applied AFTER decide())."""
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.local.context_window = 32_000
    cfg.local.context_safe_fraction = 0.85
    r = Router(cfg)
    d = r.decide(_ctx(requested_model="tinyctx-local", est_tokens=100_000))
    assert d.route == "local"


def test_router_error_streak_escalates_to_frontier():
    from tinyctx.router import Router
    r = Router(_make_router_cfg(escalate_on_error_streak=2))
    d = r.decide(_ctx(error_streak=2))
    assert d.route == "frontier"
    assert "error_streak" in d.reason


def test_router_error_streak_below_threshold_stays_local():
    from tinyctx.router import Router
    r = Router(_make_router_cfg(escalate_on_error_streak=2))
    d = r.decide(_ctx(error_streak=1))
    assert d.route == "local"


def test_router_adaptive_model_escalates_after_local_failures():
    from tinyctx import adaptive_model
    from tinyctx.router import Decision, Router

    adaptive_model.reset_state()
    cfg = _make_router_cfg(
        adaptive_model_enabled=True,
        adaptive_model_min_calls=3,
        adaptive_model_failure_rate_threshold=0.5,
        adaptive_model_sample_size=5,
    )
    local_decision = Decision(
        route="local",
        reason="seed",
        target="http://local.test/v1/chat/completions",
        model="local-mdl",
        wire_api="chat",
    )
    adaptive_model.record_decision(local_decision, ok=False, max_samples=5)
    adaptive_model.record_decision(local_decision, ok=False, max_samples=5)
    adaptive_model.record_decision(local_decision, ok=True, max_samples=5)

    d = Router(cfg).decide(_ctx())

    assert d.route == "frontier"
    assert "adaptive local failure rate" in d.reason
    adaptive_model.reset_state()


def test_router_explicit_local_beats_adaptive_model_escalation():
    from tinyctx import adaptive_model
    from tinyctx.router import Decision, Router

    adaptive_model.reset_state()
    cfg = _make_router_cfg(
        adaptive_model_enabled=True,
        adaptive_model_min_calls=1,
        adaptive_model_failure_rate_threshold=0.1,
    )
    local_decision = Decision(
        route="local",
        reason="seed",
        target="http://local.test/v1/chat/completions",
        model="local-mdl",
        wire_api="chat",
    )
    adaptive_model.record_decision(local_decision, ok=False, max_samples=5)

    d = Router(cfg).decide(_ctx(requested_model="tinyctx-local"))

    assert d.route == "local"
    adaptive_model.reset_state()


def test_router_capacity_rule_escalates_when_over_safe_cap():
    """est_tokens > local.context_window * safe_fraction → frontier."""
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.local.context_window = 32_000
    cfg.local.context_safe_fraction = 0.85
    r = Router(cfg)
    d = r.decide(_ctx(est_tokens=30_000))  # > 32_000 * 0.85 = 27_200
    assert d.route == "frontier"
    assert "30000" in d.reason or "ctx" in d.reason


def test_router_capacity_rule_disabled_when_safe_fraction_zero():
    """Default config (safe_fraction=0.0) must NOT auto-escalate by tokens."""
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.local.context_window = 1_000_000
    cfg.local.context_safe_fraction = 0.0  # disabled
    r = Router(cfg)
    d = r.decide(_ctx(est_tokens=2_000_000))
    assert d.route == "local"


def test_router_classify_rule_is_advisor_only_by_default():
    """classify_p >= threshold escalates to frontier by default (escalates_to_frontier=True)."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg(self_classify_threshold=0.7))
    d = r.decide(_ctx(classify_p=0.85, classify_reason="needs strategy"))
    assert d.route == "frontier"
    assert "0.85" in d.reason or "self-classify" in d.reason.lower()


def test_router_classify_rule_legacy_switch_escalates_to_frontier():
    """Legacy full-turn frontier routing remains available as opt-in."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg(
        self_classify_threshold=0.7,
        self_classify_escalates_to_frontier=True,
    ))
    d = r.decide(_ctx(classify_p=0.85, classify_reason="needs strategy"))
    assert d.route == "frontier"
    assert "0.85" in d.reason or "self-classify" in d.reason.lower()


def test_router_classify_rule_below_threshold_stays_local():
    from tinyctx.router import Router
    r = Router(_make_router_cfg(self_classify_threshold=0.7))
    d = r.decide(_ctx(classify_p=0.5))
    assert d.route == "local"


def test_router_priority_compaction_beats_classify():
    """Even with classify_p high, compaction must route local."""
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(is_compaction=True, classify_p=0.99))
    assert d.route == "local"


def test_router_priority_force_route_beats_error_streak_and_classify():
    from tinyctx.router import Router
    r = Router(_make_router_cfg())
    d = r.decide(_ctx(force_route="frontier", error_streak=0, classify_p=0.0))
    assert d.route == "frontier"


def test_router_headers_include_content_type_and_accept():
    """Decision.headers must carry at minimum Content-Type + Accept; if the
    backend has an api_key_env set and env var present, Authorization too."""
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    r = Router(cfg)
    d = r.decide(_ctx())
    assert d.headers.get("Content-Type") == "application/json"
    assert "text/event-stream" in d.headers.get("Accept", "")


def test_router_headers_include_authorization_when_env_set(monkeypatch):
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.frontier.api_key_env = "TINYCTX_TEST_FRONTIER_KEY"
    monkeypatch.setenv("TINYCTX_TEST_FRONTIER_KEY", "sk-frontier-abc")
    r = Router(cfg)
    d = r.decide(_ctx(requested_model="tinyctx-frontier"))
    assert d.headers.get("Authorization", "").startswith("Bearer sk-frontier")


def test_router_target_url_is_responses_for_responses_wire_api():
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.frontier.wire_api = "responses"
    r = Router(cfg)
    d = r.decide(_ctx(force_route="frontier"))
    assert d.target.rstrip("/").endswith("/responses")


def test_router_target_url_is_chat_for_chat_wire_api():
    from tinyctx.router import Router
    cfg = _make_router_cfg()
    cfg.local.wire_api = "chat"
    r = Router(cfg)
    d = r.decide(_ctx())
    assert d.target.rstrip("/").endswith("/chat/completions")


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(failed)
