"""Dynamic threshold derivation — `effective_proactive_compact_threshold`
should track the configured frontier model's context_window so swapping
models (gpt-5.5 ↔ gemini ↔ smaller) auto-adjusts without manual config
edits.

User directive: 一些域值需要根据接入模型的实际能力，比如上下文大小去
动态设置。不要写死。
"""
from __future__ import annotations

from tinyctx.config import (
    BackendCfg,
    Config,
    effective_proactive_compact_threshold,
)


def _make_cfg(*, frontier_ctx: int = 0,
              safe_fraction: float = 0.75,
              absolute_fallback: int = 200_000) -> Config:
    cfg = Config()
    cfg.frontier = BackendCfg(
        base_url="https://example.com",
        model="x",
        wire_api="responses",
        context_window=frontier_ctx,
    )
    cfg.proactive_compact_safe_fraction = safe_fraction
    cfg.proactive_compact_threshold = absolute_fallback
    return cfg


def test_effective_threshold_derives_from_context_window():
    cfg = _make_cfg(frontier_ctx=272_000, safe_fraction=0.75)
    assert effective_proactive_compact_threshold(cfg) == 204_000


def test_effective_threshold_scales_with_large_model():
    """Gemini 2.5 has 2M context. Threshold should scale up proportionally."""
    cfg = _make_cfg(frontier_ctx=2_000_000, safe_fraction=0.75)
    assert effective_proactive_compact_threshold(cfg) == 1_500_000


def test_effective_threshold_scales_with_small_model():
    """Some 128k-context model — threshold drops accordingly."""
    cfg = _make_cfg(frontier_ctx=128_000, safe_fraction=0.75)
    assert effective_proactive_compact_threshold(cfg) == 96_000


def test_effective_threshold_falls_back_to_absolute_when_no_context():
    """When frontier.context_window=0 (unset), fall back to the
    absolute proactive_compact_threshold so existing setups don't
    silently disable themselves."""
    cfg = _make_cfg(frontier_ctx=0, safe_fraction=0.75,
                    absolute_fallback=200_000)
    assert effective_proactive_compact_threshold(cfg) == 200_000


def test_effective_threshold_falls_back_when_safe_fraction_zero():
    """User can disable auto-derivation by setting safe_fraction=0
    and falling back to the absolute value."""
    cfg = _make_cfg(frontier_ctx=272_000, safe_fraction=0.0,
                    absolute_fallback=150_000)
    assert effective_proactive_compact_threshold(cfg) == 150_000


def test_effective_threshold_zero_disables():
    """When neither path yields a positive number, return 0 so the
    proxy skips the compact gate entirely."""
    cfg = _make_cfg(frontier_ctx=0, safe_fraction=0.75,
                    absolute_fallback=0)
    assert effective_proactive_compact_threshold(cfg) == 0


def test_safe_fraction_can_be_tightened():
    """Conservative deployment: 0.6 of context window."""
    cfg = _make_cfg(frontier_ctx=272_000, safe_fraction=0.6)
    assert effective_proactive_compact_threshold(cfg) == int(272_000 * 0.6)


def test_safe_fraction_can_be_loosened():
    """Aggressive deployment: 0.9 of context window — risky but lets
    more history through."""
    cfg = _make_cfg(frontier_ctx=272_000, safe_fraction=0.9)
    assert effective_proactive_compact_threshold(cfg) == int(272_000 * 0.9)


def test_default_frontier_ships_with_codex_gpt55_context_window():
    """Out of the box, Config() should have a sensible frontier.context_window
    so users on the default codex+gpt-5.5 path get auto-derived
    thresholds without touching config.toml."""
    cfg = Config()
    # Default frontier.context_window matches codex.app's hardcoded
    # 272000 for gpt-5.5. If this changes, update the test.
    assert cfg.frontier.context_window == 272_000
    assert effective_proactive_compact_threshold(cfg) == int(272_000 * 0.75)
