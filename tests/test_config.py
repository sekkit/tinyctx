"""Regression-guard tests for tinyctx.config.

Locks the public-API surface of `Config` / `BackendCfg` / `load_config()` /
`effective_proactive_compact_threshold()` / `_env()`. Defaults here are
intentional (several were set by the live trace 2026-05-10 directive) so
the tests use precise equality assertions — a future drift will fail
loudly instead of silently changing behavior.

Coverage layers:
  * Config dataclass field defaults (lock numeric / boolean / string /
    container values)
  * BackendCfg field defaults and per-backend (local / frontier) overrides
  * `_env()` semantics (unset / "" / value)
  * `load_config()` layering: defaults < TOML file < TINYCTX_* env
  * `effective_proactive_compact_threshold()` boundary cases not already
    in test_dynamic_thresholds.py (specifically `frontier=None`-shaped
    inputs and the both-zero / both-positive edges)
  * Invariants relied on by other modules (essentials membership,
    tier-prompt sizing, etc.)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tinyctx.config import (
    BackendCfg,
    Config,
    _env,
    effective_proactive_compact_threshold,
    load_config,
)


# ---------------------------------------------------------------------------
# _env() helper
# ---------------------------------------------------------------------------

def test_env_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYCTX_TEST_ENV_X", raising=False)
    assert _env("TINYCTX_TEST_ENV_X", "fallback") == "fallback"


def test_env_returns_default_when_unset_and_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TINYCTX_TEST_ENV_X", raising=False)
    assert _env("TINYCTX_TEST_ENV_X") is None


def test_env_returns_default_when_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string is treated the same as unset — important so that
    `setenv("X", "")` doesn't accidentally clear out a real default."""
    monkeypatch.setenv("TINYCTX_TEST_ENV_X", "")
    assert _env("TINYCTX_TEST_ENV_X", "fallback") == "fallback"


def test_env_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYCTX_TEST_ENV_X", "real")
    assert _env("TINYCTX_TEST_ENV_X", "fallback") == "real"


# ---------------------------------------------------------------------------
# BackendCfg defaults
# ---------------------------------------------------------------------------

def test_backendcfg_minimal_defaults() -> None:
    """Constructing BackendCfg with only the required base_url should
    yield the documented schema-shaped defaults."""
    bc = BackendCfg(base_url="https://example.com")
    assert bc.base_url == "https://example.com"
    assert bc.api_key_env is None
    assert bc.model == ""
    assert bc.wire_api == "responses"
    assert bc.timeout_s == 300.0
    assert bc.headers == {}
    assert bc.context_window == 0
    assert bc.context_safe_fraction == 0.0
    assert bc.supported_tool_types == ("function",)
    assert bc.strip_request_fields == ("client_metadata", "prompt_cache_key")
    assert bc.inject_defaults == {"text.format.type": "text"}
    assert bc.translate_tool_calls is True
    assert bc.cap_fields == {}


def test_backendcfg_inject_defaults_per_instance_isolated() -> None:
    """The default_factory must give each instance its own dict so that
    mutating one BackendCfg's inject_defaults can't leak across configs."""
    a = BackendCfg(base_url="https://a")
    b = BackendCfg(base_url="https://b")
    a.inject_defaults["custom"] = "value"
    assert "custom" not in b.inject_defaults


# ---------------------------------------------------------------------------
# Config — top-level scalar defaults (regression locks)
# ---------------------------------------------------------------------------

def test_config_host_port_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear env that influences default_factory paths, just in case.
    monkeypatch.delenv("TINYCTX_FORCE_ROUTE", raising=False)
    monkeypatch.delenv("TINYCTX_VERBOSE", raising=False)
    cfg = Config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 4141


def test_config_log_dir_default_is_under_tinyctx_home() -> None:
    cfg = Config()
    assert cfg.log_dir == Path.home() / ".tinyctx" / "logs"


def test_config_routing_thresholds_disabled_by_default() -> None:
    """Anthropic Advisor Strategy: model decides escalation, not bytes."""
    cfg = Config()
    assert cfg.escalate_input_tokens == 0
    assert cfg.escalate_turn_count == 0
    assert cfg.escalate_on_error_streak == 2


def test_config_compaction_redirect_defaults_on() -> None:
    cfg = Config()
    assert cfg.redirect_compaction_to_local is True
    assert cfg.compactor_debate is True
    assert cfg.compactor_min_history_tokens == 4_000
    assert cfg.save_compactions is True


def test_config_historian_defaults() -> None:
    cfg = Config()
    assert cfg.historian_enabled is False
    assert cfg.historian_substitute is False
    assert cfg.historian_min_new_turns == 5
    assert cfg.historian_recent_keep == 4


def test_config_sanitize_defaults() -> None:
    cfg = Config()
    assert cfg.sanitize_encrypted_content is True
    assert cfg.dedup_tool_calls is False
    assert cfg.purge_failed_tool_inputs is False
    assert cfg.failed_input_after_turns == 4
    assert cfg.mutation_ttl_s == 300.0
    assert cfg.mutation_threshold == 0.65
    assert cfg.default_context_window == 1_000_000


def test_config_read_delta_defaults() -> None:
    cfg = Config()
    assert cfg.read_delta_enabled is True
    assert cfg.read_delta_min_bytes == 400
    assert cfg.read_delta_max_diff_budget == 0.85


def test_config_proactive_compact_defaults() -> None:
    cfg = Config()
    assert cfg.proactive_compact_threshold == 200_000
    assert cfg.proactive_compact_safe_fraction == 0.75
    assert cfg.proactive_compact_recent_keep == 8
    assert cfg.proactive_compact_use_summarizer is True
    assert cfg.proactive_compact_only_on_frontier is True


def test_config_frontier_trim_tools_defaults() -> None:
    """User directive 2026-05-10 flipped this to False. Locking it
    here prevents accidental re-enable of rare-tool starvation bugs."""
    cfg = Config()
    assert cfg.frontier_trim_tools is False
    assert cfg.frontier_tools_recent_window == 30
    assert cfg.frontier_skip_advisor_hint is True


def test_config_frontier_tools_essentials_membership() -> None:
    """Essentials must include the codex sub-agent primitives so
    spawn_agent / wait_agent are never silently dropped if a future
    edit re-enables trimming."""
    cfg = Config()
    essentials = set(cfg.frontier_tools_essentials)
    for required in (
        "shell",
        "apply_patch",
        "container.exec",
        "update_plan",
        "spawn_agent",
        "wait_agent",
        "close_agent",
        "resume_agent",
        "request_user_input",
        "send_input",
        "view_image",
        "image_view",
        "mcp__advisor__ask_advisor",
    ):
        assert required in essentials, f"{required!r} missing from essentials"


def test_config_self_classify_defaults() -> None:
    cfg = Config()
    assert cfg.self_classify_enabled is True
    assert cfg.self_classify_threshold == 0.7
    assert cfg.self_classify_timeout_s == 30.0


def test_config_stuck_loop_defaults() -> None:
    cfg = Config()
    assert cfg.stuck_loop_watchdog_enabled is True
    assert cfg.stuck_loop_turn_trigger == 80
    assert cfg.stuck_loop_turn_gap == 50
    assert cfg.stuck_loop_advisor_grace_s == 600.0


def test_config_soft_completion_defaults() -> None:
    cfg = Config()
    assert cfg.soft_completion_gate_enabled is True
    assert cfg.soft_completion_short_text_threshold == 50
    # User directive 2026-05-10: classify EVERY stop, even very short ones.
    assert cfg.soft_completion_stop_text_threshold == 1


def test_config_soft_completion_stream_rewrite_defaults() -> None:
    cfg = Config()
    assert cfg.soft_completion_stream_rewrite_enabled is True
    assert cfg.soft_completion_stream_rewrite_threshold == 0.85
    assert cfg.soft_completion_stream_rewrite_tool_name == "spawn_agent"
    assert cfg.soft_completion_stream_rewrite_extra_args == {"role": "advisor"}


def test_config_continue_injection_budget() -> None:
    cfg = Config()
    assert cfg.max_continue_injections_per_session == 20


def test_config_empty_response_guard_defaults() -> None:
    cfg = Config()
    assert cfg.empty_response_guard_enabled is True
    assert cfg.empty_response_min_completion_tokens == 5


def test_config_auto_force_frontier_defaults() -> None:
    cfg = Config()
    assert cfg.soft_completion_auto_force_frontier_enabled is True
    assert cfg.soft_completion_auto_force_frontier_threshold == 0.85


def test_config_forensics_defaults() -> None:
    cfg = Config()
    assert cfg.forensics_enabled is True
    assert cfg.forensics_capture_punts is True
    assert cfg.forensics_punt_threshold == 0.9
    assert cfg.forensics_capture_errors is True
    assert cfg.forensics_max_dumps == 100


def test_config_plan_persistence_defaults() -> None:
    cfg = Config()
    assert cfg.plan_persistence_enabled is True
    assert cfg.plan_persistence_ttl_s == 7 * 24 * 3600  # 7 days


def test_config_exec_resume_defaults() -> None:
    cfg = Config()
    assert cfg.exec_resume_enabled is True
    assert cfg.exec_resume_min_p == 0.85
    assert cfg.exec_resume_cooldown_s == 300
    assert cfg.exec_resume_max_per_minute == 3
    assert cfg.exec_resume_timeout_s == 60
    assert cfg.exec_resume_codex_binary == ""
    assert cfg.exec_resume_sandbox == "read-only"
    assert cfg.exec_resume_approval_policy == "never"
    assert isinstance(cfg.exec_resume_prompt, str)
    assert cfg.exec_resume_prompt.startswith("continue working from where")


def test_config_exec_resume_prompt_tiers_shape() -> None:
    """SPEC-style tiered prompts: must have at least 3 entries (gentle,
    firm, final) and each entry should be a non-trivial instruction —
    >= 80 chars protects against truncation regressions."""
    cfg = Config()
    tiers = cfg.exec_resume_prompt_tiers
    assert isinstance(tiers, list)
    assert len(tiers) >= 3
    for i, prompt in enumerate(tiers):
        assert isinstance(prompt, str), f"tier[{i}] must be str"
        assert len(prompt) >= 80, (
            f"tier[{i}] is suspiciously short ({len(prompt)} chars): {prompt!r}"
        )


def test_config_exec_resume_prompt_tiers_isolated_per_instance() -> None:
    """The default_factory yields a fresh list per Config; mutation on
    one instance must not leak across instances."""
    a = Config()
    b = Config()
    a.exec_resume_prompt_tiers.append("custom")
    assert "custom" not in b.exec_resume_prompt_tiers


def test_config_upstream_retry_defaults() -> None:
    cfg = Config()
    assert cfg.upstream_retry_enabled is True
    assert cfg.upstream_retry_count == 1
    assert cfg.upstream_retry_max_bytes_yielded == 4096


def test_config_stall_watchdog_defaults() -> None:
    cfg = Config()
    assert cfg.stall_watchdog_enabled is True
    assert cfg.stall_threshold_s == 180.0
    assert cfg.stall_check_interval_s == 30.0


def test_config_stream_keepalive_default_interval() -> None:
    cfg = Config()
    assert cfg.stream_keepalive_interval_s == 15.0


def test_config_inject_global_agent_rules_default() -> None:
    cfg = Config()
    assert cfg.inject_global_agent_rules is True


def test_config_auto_register_mcp_servers_default() -> None:
    cfg = Config()
    assert cfg.auto_register_mcp_servers is True


def test_config_auto_scout_defaults() -> None:
    cfg = Config()
    assert cfg.auto_scout is True
    assert cfg.auto_scout_install_graphify is False


def test_config_frontier_lingua_defaults() -> None:
    cfg = Config()
    assert cfg.frontier_lingua_enabled is False
    assert cfg.frontier_lingua_ratio == 0.5
    assert cfg.frontier_lingua_min_bytes == 800


def test_config_force_route_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYCTX_FORCE_ROUTE", raising=False)
    cfg = Config()
    assert cfg.force_route == "auto"


def test_config_verbose_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYCTX_VERBOSE", raising=False)
    cfg = Config()
    assert cfg.verbose is True


def test_config_force_route_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINYCTX_FORCE_ROUTE", "frontier")
    cfg = Config()
    assert cfg.force_route == "frontier"


def test_config_verbose_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYCTX_VERBOSE", "0")
    cfg = Config()
    assert cfg.verbose is False


# ---------------------------------------------------------------------------
# Local + frontier BackendCfg defaults (full subobject snapshot)
# ---------------------------------------------------------------------------

def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "TINYCTX_LOCAL_BASE_URL",
        "TINYCTX_LOCAL_API_KEY",
        "TINYCTX_LOCAL_MODEL",
        "TINYCTX_LOCAL_WIRE_API",
        "TINYCTX_LOCAL_TIMEOUT_S",
        "TINYCTX_LOCAL_MAX_OUTPUT_TOKENS",
        "TINYCTX_FRONTIER_BASE_URL",
        "TINYCTX_FRONTIER_API_KEY",
        "TINYCTX_FRONTIER_MODEL",
        "TINYCTX_FRONTIER_TIMEOUT_S",
        "TINYCTX_FRONTIER_CONTEXT_WINDOW",
    ):
        monkeypatch.delenv(k, raising=False)


def test_config_local_backend_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_env(monkeypatch)
    cfg = Config()
    assert cfg.local.base_url == "http://127.0.0.1:1234/v1"
    assert cfg.local.api_key_env == "TINYCTX_LOCAL_API_KEY"
    assert cfg.local.model == "qwen3.6-27b"
    assert cfg.local.wire_api == "chat"
    assert cfg.local.timeout_s == 180.0
    assert cfg.local.inject_defaults == {
        "text.format.type": "text",
        "max_output_tokens": 16000,
    }
    assert cfg.local.cap_fields == {"max_output_tokens": 16000}


def test_config_local_backend_max_output_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("TINYCTX_LOCAL_MAX_OUTPUT_TOKENS", "8000")
    cfg = Config()
    assert cfg.local.inject_defaults["max_output_tokens"] == 8000
    assert cfg.local.cap_fields["max_output_tokens"] == 8000


def test_config_local_base_url_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("TINYCTX_LOCAL_BASE_URL", "http://10.0.0.5:8080/v1")
    cfg = Config()
    assert cfg.local.base_url == "http://10.0.0.5:8080/v1"


def test_config_frontier_backend_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    cfg = Config()
    assert cfg.frontier.base_url == "https://chatgpt.com/backend-api/codex"
    assert cfg.frontier.api_key_env == "TINYCTX_FRONTIER_API_KEY"
    assert cfg.frontier.model == "gpt-5.5"
    assert cfg.frontier.wire_api == "responses"
    assert cfg.frontier.timeout_s == 300.0
    assert cfg.frontier.context_window == 272_000
    assert cfg.frontier.supported_tool_types == ()
    assert cfg.frontier.strip_request_fields == ("max_output_tokens",)
    assert cfg.frontier.inject_defaults == {}
    assert cfg.frontier.translate_tool_calls is False


def test_config_frontier_context_window_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("TINYCTX_FRONTIER_CONTEXT_WINDOW", "1000000")
    cfg = Config()
    assert cfg.frontier.context_window == 1_000_000


# ---------------------------------------------------------------------------
# effective_proactive_compact_threshold — extra edge cases
# (test_dynamic_thresholds.py covers the main matrix; these add the
# both-zero, missing-attribute, and integer-truncation edges.)
# ---------------------------------------------------------------------------

def test_effective_threshold_truncates_to_int() -> None:
    cfg = Config()
    cfg.frontier = BackendCfg(
        base_url="x",
        context_window=100,
    )
    cfg.proactive_compact_safe_fraction = 0.333
    cfg.proactive_compact_overhead_buffer = 0
    # 100 * 0.333 = 33.3 → int() truncates toward zero → 33.
    assert effective_proactive_compact_threshold(cfg) == 33


def test_effective_threshold_negative_safe_fraction_falls_back() -> None:
    """sf <= 0 must use the absolute fallback (the function uses `> 0`
    so a negative fraction is treated as disabled)."""
    cfg = Config()
    cfg.frontier = BackendCfg(base_url="x", context_window=200_000)
    cfg.proactive_compact_safe_fraction = -0.1
    cfg.proactive_compact_threshold = 50_000
    cfg.proactive_compact_overhead_buffer = 0
    assert effective_proactive_compact_threshold(cfg) == 50_000


def test_effective_threshold_handles_missing_context_window_attr() -> None:
    """The function uses getattr with a 0 default; if a custom
    BackendCfg-like is swapped in without context_window it should
    fall back instead of raising."""

    class Stub:
        pass  # no context_window attr

    cfg = Config()
    cfg.frontier = Stub()  # type: ignore[assignment]
    cfg.proactive_compact_safe_fraction = 0.75
    cfg.proactive_compact_threshold = 123_456
    cfg.proactive_compact_overhead_buffer = 0
    assert effective_proactive_compact_threshold(cfg) == 123_456


# ---------------------------------------------------------------------------
# load_config — layering (defaults < TOML < env)
# ---------------------------------------------------------------------------

def _isolate_load_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Point HOME at a clean tmp dir and clear TINYCTX_* knobs so the
    test starts from defaults."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TINYCTX_CONFIG", raising=False)
    monkeypatch.delenv("TINYCTX_LOG_DIR", raising=False)
    monkeypatch.delenv("TINYCTX_FORCE_ROUTE", raising=False)
    monkeypatch.delenv("TINYCTX_VERBOSE", raising=False)
    _clear_backend_env(monkeypatch)
    return tmp_path


def test_load_config_pure_defaults_when_no_file_no_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _isolate_load_config(monkeypatch, tmp_path)
    cfg = load_config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 4141
    # log_dir is resolved off TINYCTX_LOG_DIR or default; with HOME
    # redirected it lives under the tmp path.
    assert cfg.log_dir == home / ".tinyctx" / "logs"
    assert cfg.log_dir.is_dir()  # load_config() mkdir's it.
    assert cfg.local.base_url == "http://127.0.0.1:1234/v1"
    assert cfg.frontier.model == "gpt-5.5"


def test_load_config_log_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    custom = tmp_path / "custom_logs"
    monkeypatch.setenv("TINYCTX_LOG_DIR", str(custom))
    cfg = load_config()
    assert cfg.log_dir == custom
    assert custom.is_dir()


def test_load_config_toml_overrides_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[server]
host = "10.0.0.1"
port = 9999
proactive_compact_recent_keep = 4
frontier_trim_tools = true

[local]
base_url = "http://lan-host:1234/v1"
model = "custom-local"

[frontier]
base_url = "https://example.com/v1"
context_window = 500000

[routing]
escalate_input_tokens = 60000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(toml))
    cfg = load_config()
    # [server]
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 9999
    assert cfg.proactive_compact_recent_keep == 4
    assert cfg.frontier_trim_tools is True
    # [local]
    assert cfg.local.base_url == "http://lan-host:1234/v1"
    assert cfg.local.model == "custom-local"
    # [frontier]
    assert cfg.frontier.base_url == "https://example.com/v1"
    assert cfg.frontier.context_window == 500_000
    # [routing]
    assert cfg.escalate_input_tokens == 60_000


def test_load_config_env_overrides_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[local]
base_url = "http://from-toml:1234/v1"
model = "toml-model"

[frontier]
base_url = "https://from-toml.example.com/v1"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(toml))
    monkeypatch.setenv("TINYCTX_LOCAL_BASE_URL", "http://from-env:1234/v1")
    monkeypatch.setenv("TINYCTX_FRONTIER_BASE_URL", "https://from-env.example/v1")
    cfg = load_config()
    # env wins
    assert cfg.local.base_url == "http://from-env:1234/v1"
    assert cfg.frontier.base_url == "https://from-env.example/v1"
    # field that the env didn't touch keeps the TOML value
    assert cfg.local.model == "toml-model"


def test_load_config_unknown_toml_keys_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`load_config` only setattr's keys that already exist on the
    target dataclass — typos in user TOML must not crash startup."""
    _isolate_load_config(monkeypatch, tmp_path)
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[server]
this_key_does_not_exist = "ignored"

[local]
also_unknown = 42
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(toml))
    cfg = load_config()  # must not raise
    assert not hasattr(cfg, "this_key_does_not_exist")
    assert not hasattr(cfg.local, "also_unknown")


def test_load_config_force_route_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCTX_FORCE_ROUTE", "local")
    cfg = load_config()
    assert cfg.force_route == "local"


def test_load_config_verbose_env_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCTX_VERBOSE", "0")
    cfg = load_config()
    assert cfg.verbose is False


def test_load_config_verbose_env_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_load_config(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCTX_VERBOSE", "1")
    cfg = load_config()
    assert cfg.verbose is True


def test_load_config_missing_file_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pointing TINYCTX_CONFIG at a non-existent path should not raise
    — load_config gates with `path.is_file()`."""
    _isolate_load_config(monkeypatch, tmp_path)
    monkeypatch.setenv("TINYCTX_CONFIG", str(tmp_path / "does_not_exist.toml"))
    cfg = load_config()  # must not raise
    assert cfg.host == "127.0.0.1"


def test_load_config_returns_fresh_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two calls to load_config() must return independent Config
    objects — mutating one's nested BackendCfg must not leak."""
    _isolate_load_config(monkeypatch, tmp_path)
    a = load_config()
    b = load_config()
    assert a is not b
    assert a.local is not b.local
    a.local.model = "mutated"
    assert b.local.model == "qwen3.6-27b"


# ---------------------------------------------------------------------------
# P8: namespaced sub-dataclass views + back-compat property forwarders
# ---------------------------------------------------------------------------
#
# P0-P7 grew Config to 50+ top-level flags with no grouping. P8 adds
# *views* (cfg.routing, cfg.stall, cfg.retry, cfg.compact, cfg.guards,
# cfg.stuck_loop, cfg.forensics) that expose related flags under a
# single namespace WITHOUT moving them — old call sites that read
# `cfg.force_route`, `cfg.stall_threshold_s`, etc. directly keep working,
# and TOML parsing (which uses flat keys like `[server]` / `[routing]`)
# remains unchanged. The view objects forward both `get` and `set` so
# `cfg.routing.force_route = "x"` and `cfg.force_route = "x"` are
# equivalent.
#
# All assertions below rely ONLY on the view mechanism — defaults
# themselves are locked by the regression tests above.


def test_routing_namespace_reads_top_level_defaults() -> None:
    cfg = Config()
    assert cfg.routing.force_route == cfg.force_route
    assert cfg.routing.escalate_on_error_streak == cfg.escalate_on_error_streak
    assert cfg.routing.escalate_input_tokens == cfg.escalate_input_tokens
    assert cfg.routing.redirect_compaction_to_local is cfg.redirect_compaction_to_local
    assert cfg.routing.self_classify_threshold == cfg.self_classify_threshold


def test_routing_namespace_write_propagates_to_top_level() -> None:
    """Setting via the view must update the canonical top-level slot
    so existing call sites (router.py, proxy.py) see the new value."""
    cfg = Config()
    cfg.routing.force_route = "frontier"
    assert cfg.force_route == "frontier"
    cfg.routing.escalate_input_tokens = 12345
    assert cfg.escalate_input_tokens == 12345


def test_routing_top_level_write_visible_via_namespace() -> None:
    """Inverse direction: legacy setattr at top-level must be visible
    through the namespace view — TOML loader uses `setattr(cfg, k, v)`
    on flat keys and the view must reflect that."""
    cfg = Config()
    cfg.force_route = "local"
    assert cfg.routing.force_route == "local"
    cfg.redirect_compaction_to_local = False
    assert cfg.routing.redirect_compaction_to_local is False


def test_stall_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert cfg.stall.stall_threshold_s == cfg.stall_threshold_s
    assert cfg.stall.stall_check_interval_s == cfg.stall_check_interval_s
    assert cfg.stall.stall_watchdog_enabled is cfg.stall_watchdog_enabled
    cfg.stall.stall_threshold_s = 60.0
    assert cfg.stall_threshold_s == 60.0
    cfg.stall_watchdog_enabled = False
    assert cfg.stall.stall_watchdog_enabled is False


def test_retry_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert cfg.retry.upstream_retry_count == cfg.upstream_retry_count
    assert (
        cfg.retry.retry_on_local_4xx_escalate_frontier
        is cfg.retry_on_local_4xx_escalate_frontier
    )
    assert cfg.retry.retry_on_frontier_4xx is cfg.retry_on_frontier_4xx
    assert cfg.retry.max_total_retries_per_request == cfg.max_total_retries_per_request
    cfg.retry.upstream_retry_count = 5
    assert cfg.upstream_retry_count == 5


def test_compact_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert cfg.compact.proactive_compact_threshold == cfg.proactive_compact_threshold
    assert (
        cfg.compact.proactive_compact_overhead_buffer
        == cfg.proactive_compact_overhead_buffer
    )
    assert (
        cfg.compact.proactive_compact_only_on_frontier
        is cfg.proactive_compact_only_on_frontier
    )
    assert (
        cfg.compact.proactive_compact_safe_fraction
        == cfg.proactive_compact_safe_fraction
    )
    cfg.compact.proactive_compact_threshold = 100_000
    assert cfg.proactive_compact_threshold == 100_000


def test_guards_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert (
        cfg.guards.max_continue_injections_per_session
        == cfg.max_continue_injections_per_session
    )
    assert cfg.guards.local_role_rewrite_enabled is cfg.local_role_rewrite_enabled
    assert cfg.guards.local_role_rewrite_map == cfg.local_role_rewrite_map
    assert cfg.guards.drop_orphan_tool_outputs is cfg.drop_orphan_tool_outputs
    assert cfg.guards.unknown_tool_call_protection is cfg.unknown_tool_call_protection
    cfg.guards.max_continue_injections_per_session = 99
    assert cfg.max_continue_injections_per_session == 99


def test_stuck_loop_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert cfg.stuck_loop.turn_trigger == cfg.stuck_loop_turn_trigger
    assert cfg.stuck_loop.turn_gap == cfg.stuck_loop_turn_gap
    assert cfg.stuck_loop.advisor_grace_s == cfg.stuck_loop_advisor_grace_s
    cfg.stuck_loop.turn_trigger = 200
    assert cfg.stuck_loop_turn_trigger == 200


def test_forensics_namespace_reads_and_writes() -> None:
    cfg = Config()
    assert cfg.forensics.forensics_enabled is cfg.forensics_enabled
    assert cfg.forensics.forensics_capture_errors is cfg.forensics_capture_errors
    assert cfg.forensics.forensics_capture_punts is cfg.forensics_capture_punts
    cfg.forensics.forensics_enabled = False
    assert cfg.forensics_enabled is False


def test_namespace_views_are_per_instance() -> None:
    """Two Config instances must have independent namespace views —
    mutating one must NEVER bleed into the other."""
    a = Config()
    b = Config()
    a.routing.force_route = "frontier"
    assert b.routing.force_route == "auto"
    assert b.force_route == "auto"


def test_namespaces_present_after_load_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """load_config() must produce a Config whose namespaces work — i.e.
    TOML-driven changes to flat keys propagate to the view."""
    _isolate_load_config(monkeypatch, tmp_path)
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[routing]
force_route = "frontier"
escalate_on_error_streak = 5
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(toml))
    cfg = load_config()
    assert cfg.force_route == "frontier"
    assert cfg.routing.force_route == "frontier"
    assert cfg.routing.escalate_on_error_streak == 5


def test_namespace_unknown_attr_raises() -> None:
    """The namespace must NOT silently swallow typos — accessing a key
    that isn't part of its declared field set should raise just like
    a plain dataclass would."""
    cfg = Config()
    with pytest.raises(AttributeError):
        _ = cfg.routing.this_does_not_exist  # noqa: F841
