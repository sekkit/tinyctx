"""Configuration loading for tinyctx.

Reads ~/.tinyctx/config.toml (or $TINYCTX_CONFIG) plus environment overrides.
Falls back to sane defaults that match the user's existing OpenAI Focus Proxy
naming so an ops-mode swap is a drop-in.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:  # graceful: tomllib only needed if a config file exists
        tomllib = None  # type: ignore


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str) -> bool | None:
    v = _env(name)
    if v is None:
        return None
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_proxy(name: str, default: str | None = None) -> str | None:
    if name not in os.environ:
        return default
    v = (os.environ.get(name) or "").strip()
    if v.lower() in {"", "0", "false", "none", "off", "direct"}:
        return None
    return v


# ---------------------------------------------------------------------------
# Namespaced views (P8)
#
# Config grew organically to 50+ top-level flags spanning routing, retry,
# stall detection, compaction, forensics, guards, etc. The flat layout
# makes discovery hard (which flags are related?). P8 introduces *view*
# proxies — `cfg.routing`, `cfg.stall`, `cfg.retry`, `cfg.compact`,
# `cfg.guards`, `cfg.stuck_loop`, `cfg.forensics` — that surface related
# flags under a single namespace.
#
# Critical: BACK-COMPAT IS NON-NEGOTIABLE. The views do NOT own the
# storage; they forward both reads and writes back to the canonical
# top-level Config attributes. This means:
#
#   * Existing call sites (`cfg.force_route`, `cfg.proactive_compact_threshold`,
#     ...) keep working unchanged.
#   * TOML deserialization (`setattr(cfg, k, v)` on flat keys from the
#     [server] / [routing] / [local] / [frontier] sections) keeps working.
#   * New code can use the more legible nested form: `cfg.routing.force_route`.
#   * ~/.tinyctx/config.toml requires NO user migration — the file format
#     is unchanged.
#
# The mapping from view-attribute → top-level-attribute is declared per
# view via the `_FIELD_MAP` class var. Unknown attributes raise
# AttributeError (no silent typo swallow).
# ---------------------------------------------------------------------------


class _NSView:
    """Base for namespaced views over a parent Config.

    Subclasses declare `_FIELD_MAP: dict[str, str]` mapping the view's
    short attribute name → the canonical Config attribute name. The
    view holds a weak-style reference to its parent Config and forwards
    attribute access in both directions.
    """

    _FIELD_MAP: dict[str, str] = {}

    __slots__ = ("_parent",)

    def __init__(self, parent: "Config") -> None:
        # __slots__ + bypass __setattr__ for the parent ref.
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name: str) -> Any:
        # Only called for misses on _FIELD_MAP / _parent / dunder paths.
        fmap = type(self)._FIELD_MAP
        if name in fmap:
            return getattr(self._parent, fmap[name])
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r} "
            f"(known: {sorted(fmap)})"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        fmap = type(self)._FIELD_MAP
        if name in fmap:
            setattr(self._parent, fmap[name], value)
            return
        # No silent extension — surface the typo.
        raise AttributeError(
            f"{type(self).__name__!s} has no settable attribute {name!r} "
            f"(known: {sorted(fmap)})"
        )

    def __repr__(self) -> str:  # debug-friendly
        items = ", ".join(
            f"{k}={getattr(self._parent, v, '?')!r}"
            for k, v in type(self)._FIELD_MAP.items()
        )
        return f"{type(self).__name__}({items})"


class RoutingView(_NSView):
    """Routing thresholds and overrides.

    Groups flags that influence the local-vs-frontier route decision
    (see `tinyctx/router.py`). Forwards to top-level Config attributes;
    TOML section `[routing]` already targets these flat keys.

    Fields:
      force_route: One of {"auto","local","frontier"} — admin pin.
      escalate_on_error_streak: After N consecutive tool failures, escalate.
      escalate_input_tokens: Absolute token threshold for size-based
        escalation. 0 = disabled (Advisor-Strategy default).
      redirect_compaction_to_local: Route codex's compaction handoff
        prompt to local. The highest-leverage cost win.
      goal_control_frontier_enabled: Route `/goal` setup/review/blocker
        control turns to frontier while ordinary goal execution stays local.
      adaptive_model_enabled: Track local backend health and route auto
        requests to frontier while recent local failures exceed threshold.
      self_classify_threshold: Local model's P(escalate) cutoff.
      self_classify_escalates_to_frontier: Legacy mode: convert that
        cutoff into a full-turn frontier route instead of advisor-only
        telemetry. Default false to keep frontier as short consultation.
      self_consistency_enabled: Run local action-signature samples on
        self-classify boundary turns.
    """

    _FIELD_MAP = {
        "force_route": "force_route",
        "escalate_on_error_streak": "escalate_on_error_streak",
        "escalate_input_tokens": "escalate_input_tokens",
        "redirect_compaction_to_local": "redirect_compaction_to_local",
        "goal_control_frontier_enabled": "goal_control_frontier_enabled",
        "adaptive_model_enabled": "adaptive_model_enabled",
        "adaptive_model_min_calls": "adaptive_model_min_calls",
        "adaptive_model_failure_rate_threshold": "adaptive_model_failure_rate_threshold",
        "adaptive_model_sample_size": "adaptive_model_sample_size",
        "self_classify_threshold": "self_classify_threshold",
        "self_classify_escalates_to_frontier": "self_classify_escalates_to_frontier",
        "self_consistency_enabled": "self_consistency_enabled",
        "self_consistency_boundary_low": "self_consistency_boundary_low",
        "self_consistency_boundary_high": "self_consistency_boundary_high",
        "self_consistency_sample_count": "self_consistency_sample_count",
        "self_consistency_timeout_s": "self_consistency_timeout_s",
        "image_prefer_frontier": "image_prefer_frontier",
    }


class StallView(_NSView):
    """Mid-stream stall watchdog + SSE keepalive.

    See `tinyctx/stall_watchdog.py` and `tinyctx/stream_relay.py`.

    Fields:
      stall_watchdog_enabled: Master switch.
      stall_threshold_s: Seconds without upstream events before declaring stall.
      stall_check_interval_s: Watchdog wakeup cadence.
      stream_keepalive_interval_s: SSE comment cadence to keep clients alive.
    """

    _FIELD_MAP = {
        "stall_watchdog_enabled": "stall_watchdog_enabled",
        "stall_threshold_s": "stall_threshold_s",
        "stall_check_interval_s": "stall_check_interval_s",
        "stream_keepalive_interval_s": "stream_keepalive_interval_s",
    }


class RetryView(_NSView):
    """Unified retry policy (see `tinyctx/retry_policy.py`).

    Fields:
      upstream_retry_enabled: Master switch for mid-stream retry.
      upstream_retry_count: Same-backend retry budget per request.
      upstream_retry_max_bytes_yielded: Above this, retry is unsafe
        (content already partially sent to client).
      retry_on_local_4xx_escalate_frontier: On local 400/422, try
        frontier (chatgpt.com is more permissive about codex shapes).
      retry_on_frontier_4xx: Off by default — frontier 4xx rarely
        succeeds on retry with the same body.
      max_total_retries_per_request: Hard safety cap across all retry
        kinds for a single request.
    """

    _FIELD_MAP = {
        "upstream_retry_enabled": "upstream_retry_enabled",
        "upstream_retry_count": "upstream_retry_count",
        "upstream_retry_max_bytes_yielded": "upstream_retry_max_bytes_yielded",
        "retry_on_local_4xx_escalate_frontier": "retry_on_local_4xx_escalate_frontier",
        "retry_on_frontier_4xx": "retry_on_frontier_4xx",
        "max_total_retries_per_request": "max_total_retries_per_request",
    }


class CompactView(_NSView):
    """Proactive history compaction thresholds and policy.

    See `effective_proactive_compact_threshold()` for the resolution
    algorithm. Threshold derivation prefers
    `frontier.context_window * safe_fraction` and falls back to the
    absolute `proactive_compact_threshold`.

    Fields:
      proactive_compact_threshold: Absolute fallback threshold (tokens).
      proactive_compact_overhead_buffer: Subtracted from the derived
        threshold to account for tinyctx's own injections.
      proactive_compact_only_on_frontier: Skip for local route (1M ctx).
      proactive_compact_safe_fraction: Fraction of frontier.context_window
        used as trigger. 0.0 = disable auto-derivation.
      proactive_compact_recent_keep: Number of recent turns kept verbatim.
      proactive_compact_use_summarizer: Use LLM call vs deterministic
        placeholder.
    """

    _FIELD_MAP = {
        "proactive_compact_threshold": "proactive_compact_threshold",
        "proactive_compact_overhead_buffer": "proactive_compact_overhead_buffer",
        "proactive_compact_only_on_frontier": "proactive_compact_only_on_frontier",
        "proactive_compact_safe_fraction": "proactive_compact_safe_fraction",
        "proactive_compact_recent_keep": "proactive_compact_recent_keep",
        "proactive_compact_use_summarizer": "proactive_compact_use_summarizer",
    }


class GuardsView(_NSView):
    """Request-body guards: continue-injection budget, role rewrites,
    orphan tool-output drop, unknown-tool-call protection.

    Fields:
      max_continue_injections_per_session: Per-session synthetic-continue
        budget before tinyctx stops nudging and force-frontiers.
      local_role_rewrite_enabled: Rewrite `role="developer"` -> `system`
        for strict local OpenAI-compat backends.
      local_role_rewrite_map: Source-role -> target-role mapping.
      drop_orphan_tool_outputs: Strip tool-output items whose call_id has
        no matching call earlier in body.input (chatgpt.com 400s on
        orphans).
      unknown_tool_call_protection: Replace local-model function_calls
        whose name isn't in the codex tool registry with a benign
        `shell ["echo", ...]` stub so the session self-corrects.
    """

    _FIELD_MAP = {
        "max_continue_injections_per_session": "max_continue_injections_per_session",
        "local_role_rewrite_enabled": "local_role_rewrite_enabled",
        "local_role_rewrite_map": "local_role_rewrite_map",
        "drop_orphan_tool_outputs": "drop_orphan_tool_outputs",
        "unknown_tool_call_protection": "unknown_tool_call_protection",
    }


class StuckLoopView(_NSView):
    """Stuck-loop watchdog (see `tinyctx/stuck_loop.py`).

    Note: the view uses SHORT names (`turn_trigger`, `turn_gap`,
    `advisor_grace_s`) while the underlying Config attributes keep their
    `stuck_loop_` prefix for back-compat with existing call sites.

    Fields:
      turn_trigger: Minimum turn_count before a session is "stuck".
      turn_gap: Minimum turns between consecutive watchdog reminders.
      advisor_grace_s: Skip nudge if advisor was called within this many
        seconds.
    """

    _FIELD_MAP = {
        "turn_trigger": "stuck_loop_turn_trigger",
        "turn_gap": "stuck_loop_turn_gap",
        "advisor_grace_s": "stuck_loop_advisor_grace_s",
    }


class ForensicsView(_NSView):
    """Forensic dump policy (see `tinyctx/forensics.py`).

    Fields:
      forensics_enabled: Master switch.
      forensics_capture_errors: Dump on upstream errors / stream errors.
      forensics_capture_punts: Dump on high-confidence PUNT verdicts.
    """

    _FIELD_MAP = {
        "forensics_enabled": "forensics_enabled",
        "forensics_capture_errors": "forensics_capture_errors",
        "forensics_capture_punts": "forensics_capture_punts",
    }


@dataclass
class BackendCfg:
    base_url: str
    api_key_env: str | None = None
    # Whether to fall back to the inbound Codex Authorization header /
    # ~/.codex/auth.json when `api_key_env` is unset or missing.
    # Local backends default this OFF so custom LMStudio/vLLM tunnels do
    # not accidentally receive ChatGPT-account bearer tokens.
    forward_authorization: bool = True
    model: str = ""
    wire_api: str = "responses"
    timeout_s: float = 300.0
    # Optional outbound HTTP proxy for this backend. Values without a scheme
    # are normalized to http:// by the transport builder.
    proxy: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # The model's advertised input-context window (tokens). Drives the
    # "escalate when local can't fit it" routing branch — see router.decide.
    # 0 = unknown / disabled (router falls back to the absolute
    # `escalate_input_tokens` threshold).
    #
    # Examples:
    #   DeepSeek-v4-flash / -pro: 1_000_000
    #   Qwen2.5-Coder 32B (LMStudio default): 32_768
    #   Qwen3.6-27B 256K build: 262_144
    #   gpt-5.5 via codex backend: 1_000_000
    context_window: int = 0
    # Escalate to frontier when est_tokens / context_window exceeds this
    # fraction.
    #
    # 0.0 = disabled (default, aligned with Anthropic Advisor Strategy
    # — the model decides when to escalate, not infrastructure-by-bytes).
    # Set to e.g. 0.85 in `~/.tinyctx/config.toml [local]` if you run a
    # small-context backend that genuinely can't fit a long body.
    context_safe_fraction: float = 0.0
    # Per-backend tool/field scrubbing. Codex emits tool entries with
    # codex-specific `type` values (web_search, image_generation, namespace)
    # and fields (client_metadata, prompt_cache_key) that strict OpenAI-
    # compat backends like LMStudio reject with HTTP 400. The default keep-
    # set covers every common local backend; OpenAI's own endpoint accepts
    # everything so the frontier backend's defaults are effectively no-ops.
    # codex 0.128+ wraps MCP-server tools in `type=namespace`; the proxy
    # expands those into top-level `type=function` entries via
    # `expand_mcp_namespaces` BEFORE this scrub runs, so the default of
    # `function` only is correct — namespace shells are gone by the time
    # we get here.
    supported_tool_types: tuple[str, ...] = ("function",)
    strip_request_fields: tuple[str, ...] = ("client_metadata", "prompt_cache_key")
    # Opt-in passthrough for LMCache-enabled local vLLM/SGLang stacks that
    # consume cache metadata before the OpenAI-compatible backend validates
    # the request. This does not import or manage LMCache directly.
    lmcache_passthrough: bool = False
    # Dotted-path defaults injected when codex omits a field a strict
    # backend requires. LMStudio 0.4 demands text.format on /v1/responses;
    # OpenAI's backend treats it as optional. Frontier overrides this to
    # an empty dict (no injection).
    inject_defaults: dict[str, Any] = field(default_factory=lambda: {
        "text.format.type": "text",
    })
    # Translate model-native tool-call XML (qwen3-coder pythonic, Hermes,
    # etc.) into OpenAI structured `function_call` items on the way back.
    # Default true for local because LMStudio 0.4 doesn't translate it
    # itself; false for frontier (the OpenAI backend already returns
    # structured items).
    translate_tool_calls: bool = True
    # Strip tools and tool_choice from requests sent to this backend.
    # Enable for vLLM / backends without --enable-auto-tool-choice.
    strip_tools: bool = False
    # Hard caps. Unlike `inject_defaults` (which only sets if missing),
    # these LOWER the field's value when it exceeds the cap. Useful when
    # codex sends `max_output_tokens=128000` (default high value) and we
    # need to cap at 16000 to prevent DeepSeek runaway thinking loops.
    # Dotted-path → int cap. Empty by default (no cap).
    cap_fields: dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 4141
    log_dir: Path = field(default_factory=lambda: Path.home() / ".tinyctx" / "logs")

    # Cheap path: local model. Default LMStudio. Override to vLLM/SGLang/etc.
    local: BackendCfg = field(default_factory=lambda: BackendCfg(
        base_url=_env("TINYCTX_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1") or "",
        api_key_env="TINYCTX_LOCAL_API_KEY",
        forward_authorization=False,
        model=_env("TINYCTX_LOCAL_MODEL", "qwen3.6-27b") or "",
        wire_api=_env("TINYCTX_LOCAL_WIRE_API", "chat") or "chat",
        timeout_s=float(_env("TINYCTX_LOCAL_TIMEOUT_S", "180") or 180),
        proxy=_env_proxy("TINYCTX_LOCAL_PROXY"),
        # Cap runaway output. Without this, DeepSeek/etc. can stream
        # 1+ MB of tokens for 80+ seconds (observed today: a 77k-input
        # turn produced 1.25 MB / 80 s before the upstream cut its own
        # connection mid-stream — manifested as "peer closed connection
        # without sending complete message body" stream_error and the
        # codex.app UI showed the turn as interrupted). 16 000 tokens
        # is generous: a long structured answer is ~3-8k tokens; the
        # cap only bites runaway thinking loops. Set this to 0 in
        # config.toml [local] to disable.
        inject_defaults={
            "text.format.type": "text",
            "max_output_tokens": int(_env("TINYCTX_LOCAL_MAX_OUTPUT_TOKENS", "16000") or 16000),
        },
        # Force-cap codex's max_output_tokens (default 128000) to our
        # safe limit. Without this the inject_defaults was a no-op
        # because codex always provides the field.
        cap_fields={
            "max_output_tokens": int(_env("TINYCTX_LOCAL_MAX_OUTPUT_TOKENS", "16000") or 16000),
        },
    ))

    # Frontier path: GPT-5.5 (or whatever upstream codex is registered with).
    # The OpenAI backend accepts every codex-emitted field+tool-type, so
    # default every scrubber+injector to pass-through for this backend.
    frontier: BackendCfg = field(default_factory=lambda: BackendCfg(
        base_url=_env("TINYCTX_FRONTIER_BASE_URL", "https://chatgpt.com/backend-api/codex") or "",
        api_key_env="TINYCTX_FRONTIER_API_KEY",  # if None, pass through codex auth
        model=_env("TINYCTX_FRONTIER_MODEL", "gpt-5.5") or "",
        wire_api="responses",
        timeout_s=float(_env("TINYCTX_FRONTIER_TIMEOUT_S", "300") or 300),
        proxy=_env_proxy("TINYCTX_FRONTIER_PROXY", "http://127.0.0.1:10809"),
        # Codex.app 0.128 hard-codes context_window=272000 for gpt-5.5
        # in its model catalog. Setting it here lets
        # `effective_proactive_compact_threshold` derive the right
        # compact threshold automatically (272k × 0.75 ≈ 204k). Override
        # in config.toml [frontier] section when using a different
        # frontier (gemini, opus, etc.).
        context_window=int(_env("TINYCTX_FRONTIER_CONTEXT_WINDOW", "272000") or 272000),
        supported_tool_types=(),     # empty = keep all
        # chatgpt.com/backend-api/codex (the default frontier) rejects
        # `max_output_tokens` with HTTP 400 ("Unsupported parameter").
        # Codex.app sends it anyway with a default of 128000. Strip on
        # this path so the request goes through. If you point frontier
        # at a non-codex OpenAI Responses endpoint that DOES accept
        # max_output_tokens, override `strip_request_fields=()` in
        # ~/.tinyctx/config.toml.
        strip_request_fields=("max_output_tokens",),
        inject_defaults={},          # empty = inject nothing
        translate_tool_calls=False,  # frontier already returns structured items
    ))

    # Routing thresholds.
    #
    # Aligned with Anthropic's Advisor Strategy: the EXECUTOR MODEL
    # decides when to escalate, infrastructure does not auto-escalate
    # based on byte counts. So `escalate_input_tokens` and
    # `escalate_turn_count` default to 0 = disabled. The model gets the
    # frontier (via spawn_agent / advisor tool) when IT decides it
    # needs strategic input — not because we counted bytes.
    #
    # If you run a small-context local backend (LMStudio 32k, Ollama
    # default) where the local truly cannot fit a long body, set these
    # to positive values in `~/.tinyctx/config.toml [routing]` and the
    # router will fall back to size-based escalation as a safety net.
    escalate_input_tokens: int = 0          # 0 = disabled (was 60_000)
    escalate_turn_count: int = 0            # 0 = disabled (was 15)
    escalate_on_error_streak: int = 2       # repeated tool-failure -> frontier (kept; Anthropic "when stuck")
    image_prefer_frontier: bool = True      # route image-bearing prompts to frontier (vision support)

    # When True, low-risk image-to-text tasks (OCR, describe, summarize)
    # are replaced with captions from `mm cat <file> -m accurate` BEFORE
    # routing. Accuracy-sensitive or ambiguous image tasks keep the
    # original image and still hit image→frontier. If `mm` is absent or
    # captioning fails, preprocessing is a passthrough.
    # See `tinyctx/multimodal_preprocess.py`.
    image_to_text_preprocess_enabled: bool = True
    # Cap per-image processing time. mm `-m accurate` calls a VLM
    # endpoint; on a slow caption backend this bounds the latency added
    # per attachment.
    image_to_text_timeout_s: float = 30.0
    # SHA256-keyed caption cache directory. Empty -> ~/.tinyctx/cache/mm-captions.
    image_to_text_cache_dir: str = ""

    # SmallCode-inspired adaptive model select: keep a rolling in-memory
    # health window for the local backend. If automatic requests have seen
    # enough recent local failures, route subsequent auto turns to frontier
    # until successes push the failure rate back below threshold. Explicit
    # `tinyctx-local` / force_route still wins.
    adaptive_model_enabled: bool = True
    adaptive_model_min_calls: int = 3
    adaptive_model_failure_rate_threshold: float = 0.3
    adaptive_model_sample_size: int = 20

    # If true, the compaction handoff prompt is always routed to the local model.
    # This is the highest-leverage cost win and the reason this proxy exists.
    redirect_compaction_to_local: bool = True

    # Goal-mode control-plane routing. Ordinary long-running `/goal`
    # execution stays on the cheap local model, but the high-leverage
    # judgment turns (contract creation, `done_when` validation,
    # completion audit, blockers/pivots) route to frontier. The router
    # only inspects the tail user/control message so a persistent GOAL.md
    # in history does not make every iteration expensive.
    goal_control_frontier_enabled: bool = True

    # If true, compaction is handled by tinyctx.compactor's 3-role debate +
    # judge merge instead of a single-pass forward. Costs 4× local calls but
    # produces meaningfully better summaries when history is long. The 3
    # role drafts run in parallel so wall-clock latency is ~2× single call.
    compactor_debate: bool = True

    # Below this estimated input-token count, skip the debate (overkill on
    # short conversations) and fall through to single-pass forward.
    compactor_min_history_tokens: int = 4_000

    # If true, every successful compaction is persisted under
    # ~/.tinyctx/cache/<repo>/sessions/<sid>/ so the next session can recall
    # via `tinyctx-recall` after the previous one ran out of context.
    save_compactions: bool = True

    # Rolling per-session Historian (Magic-Context-style). Two halves:
    #   historian_enabled     — run async update() after each turn that
    #                           crosses the threshold; persists a digest
    #                           sidecar but does NOT change what codex sees.
    #                           Default ON. Costs 1 local LLM call every
    #                           `historian_min_new_turns` turns.
    #   historian_substitute  — when fire-time mutation gate opens, replace
    #                           older turns in the body with the digest.
    #                           Mutates history bytes, but is guarded by
    #                           CacheAwareMutator to preserve hot cache.
    historian_enabled: bool = True
    historian_substitute: bool = True
    historian_min_new_turns: int = 5
    historian_recent_keep: int = 4

    # If true, scrub `encrypted_content` from prior reasoning items when the
    # request is being routed to a different model than the one that produced
    # them. Avoids the "encrypted content could not be decrypted" crash.
    sanitize_encrypted_content: bool = True

    # Final preflight defense: drop tool-output items (function_call_output,
    # tool_result, mcp_result, tool_search_output) whose call_id has no
    # matching call item earlier in body.input. chatgpt.com codex backend
    # 400s on orphans with "No tool call found for tool search output with
    # call_id ...". Orphans can leak through proactive_compact (when the
    # matching call is a tool_search_call we don't synthesize a stub for),
    # client reordering, or upstream bugs. Default ON.
    drop_orphan_tool_outputs: bool = True

    # Rewrite `body.input[*].role` values that strict local OpenAI-compat
    # backends reject. codex 0.128+ emits `role="developer"` for system-level
    # instructions; older / community LMStudio builds (and the remote tunnel
    # observed 2026-05-11) HTTP-400 with "Unexpected message role." Maps
    # default to {"developer": "system"} which is the universally-accepted
    # equivalent. Only applied on route=local + wire_api=responses; the chat-
    # completions path already normalizes via normalize_for_chat. Frontier is
    # never rewritten (it natively supports `developer`).
    local_role_rewrite_enabled: bool = True
    local_role_rewrite_map: dict[str, str] = field(
        default_factory=lambda: {"developer": "system"})

    # History-hygiene transforms (DCP-inspired). When enabled, the
    # CacheAwareMutator (sanitize.py) gates them so they fire only when the
    # prompt cache is likely stale anyway, preserving cache hits in the
    # common case. Inspired by cortexkit/magic-context.
    dedup_tool_calls: bool = True
    purge_failed_tool_inputs: bool = True
    failed_input_after_turns: int = 4
    # Result-level shrink: after a large tool result has sat in history for
    # N+ assistant turns, replace the bulky payload with a deterministic
    # summary so later turns don't keep paying for the full blob. Inspired by
    # Reasonix's turn-end tool-result compaction, but implemented as a
    # conservative later-turn history transform.
    result_shrink_enabled: bool = True
    result_shrink_after_turns: int = 1
    result_shrink_min_bytes: int = 12_000
    result_shrink_signal_lines: int = 8
    result_shrink_head_chars: int = 400
    result_shrink_tail_chars: int = 800
    mutation_ttl_s: float = 300.0      # default = Anthropic 5-min cache TTL
    mutation_threshold: float = 0.65   # context usage trigger (0..1)
    default_context_window: int = 1_000_000  # used when request doesn't say

    # Repeat-Read delta compression (alexgreensh/token-optimizer-style).
    # When the executor re-reads the same file across turns, replace
    # later occurrences in body.input with a unified diff against the
    # first read. Cuts a per-turn 5–50 KB stable-bytes payload down to
    # a few hundred bytes when the file barely changed (or to "unchanged"
    # when it didn't change at all). Gated by the same CacheAwareMutator
    # as dedup/purge — only fires when the cache prefix is likely stale.
    read_delta_enabled: bool = True
    # Outputs below this many chars are skipped (placeholder overhead
    # would dominate). 400 ≈ 100 tokens.
    read_delta_min_bytes: int = 400
    # If the diff would be larger than (original × this fraction), keep
    # the original — a re-write doesn't compress and we'd just churn
    # cache. 0.85 leaves a meaningful win required.
    read_delta_max_diff_budget: float = 0.85

    # Proactive history truncation (proxy-side defense against "Codex ran
    # out of room"). When est_input_tokens reaches this threshold AND the
    # request is NOT codex's own compaction request, the proxy rewrites
    # body.input to keep system items + most recent N turns plus a single
    # tinyctx-generated summary item replacing the older middle. The
    # upstream model sees a slim payload that fits; codex's client-side
    # history is unchanged so the UI still shows every turn.
    #
    # codex.app 0.128 hard-codes per-model context_window=272000 for
    # gpt-5.5 and ships auto_compact_token_limit=null. Auto-compact only
    # fires when the user adds `model_auto_compact_token_limit` at the TOP
    # LEVEL of config.toml (profile-scoped doesn't apply to the default
    # profile). Even with that fixed, the proxy needs its own backstop.
    #
    # 0 = disabled. Default 200_000 is the ABSOLUTE FALLBACK used only when
    # frontier.context_window isn't set. The preferred path is dynamic:
    # tinyctx multiplies the configured frontier context window by
    # `proactive_compact_safe_fraction` to get the effective threshold,
    # so swapping models adjusts the threshold automatically.
    #
    # Examples (with proactive_compact_safe_fraction=0.75):
    #   frontier.context_window=272000  (gpt-5.5)   → effective 204000
    #   frontier.context_window=128000  (small)     → effective  96000
    #   frontier.context_window=2000000 (gemini)    → effective 1500000
    #
    # If neither frontier.context_window nor a custom threshold is set,
    # we fall back to this absolute number.
    #
    # See `effective_proactive_compact_threshold(cfg)`.
    proactive_compact_threshold: int = 200_000
    # Fraction of frontier.context_window that proactive_compact uses as
    # its trigger. 0.75 leaves ~25% headroom for output tokens, tools,
    # and tinyctx's own additions (advisor hint, summary item). Set to
    # 0.0 to disable auto-derivation and force use of the absolute
    # `proactive_compact_threshold` value above.
    proactive_compact_safe_fraction: float = 0.75
    # tinyctx adds its own injections to every forwarded request:
    # bundled agent rules, advisor hint, scout context, tool catalog
    # padding, etc. Live trace 2026-05-10 measured a ~30K-token gap
    # between `est_input_tokens` (user body alone) and
    # `forwarded_tokens_est` (what upstream actually receives). The
    # proactive_compact gate is checked against `est_input_tokens`, so
    # without this buffer it under-estimates by exactly that overhead
    # and never trips until the upstream itself crosses the budget.
    # `effective_proactive_compact_threshold` subtracts this from the
    # derived threshold so the gate compares apples to apples.
    proactive_compact_overhead_buffer: int = 25_000
    # Number of recent turns kept verbatim during proactive truncation.
    proactive_compact_recent_keep: int = 8
    # When true, summary uses one local-model call per session-summary cache
    # miss. When false, uses a deterministic "N older turns omitted"
    # placeholder (fast, free, lossy).
    proactive_compact_use_summarizer: bool = True
    # When true, proactive_compact ONLY fires for the frontier route. The
    # local model has a 1M-context backend (DeepSeek-v4-flash) so compact
    # is unnecessary AND degrades quality (model loses context). For the
    # frontier path, every token costs real money and the 272k internal
    # ceiling matters — compact is essential. Default true per user
    # directive: "本地的小模型不用太节约 token; 主要是给 frontier 要尽量的优化".
    proactive_compact_only_on_frontier: bool = True

    # ----- Frontier-only token-budget optimizations ---------------------
    # Trim the `tools` array down to a working set when forwarding to
    # frontier. Codex 0.128 sends ~50 tools per request (~10k tokens) but
    # most sessions only use a handful. Strategy:
    #   keep tools whose name appears in the most recent `frontier_tools_recent_window`
    #   input items as a function_call.name, PLUS an essentials allowlist
    #   so the model never gets stuck without shell/apply_patch/advisor.
    # Disabled (false) on local because (a) 1M ctx absorbs the cost and
    # (b) we don't want to surprise the local model by silently dropping
    # tools mid-conversation.
    #
    # 2026-05-28: DEFAULT FLIPPED TO TRUE per user directive: features
    # should auto-enable when not configured. Essentials below keep
    # sub-agent and user-input primitives available even before first use.
    # To disable trimming, set `frontier_trim_tools = false` under
    # `[server]` in ~/.tinyctx/config.toml.
    frontier_trim_tools: bool = True
    frontier_tools_recent_window: int = 30  # how many recent input items to scan
    frontier_tools_essentials: tuple[str, ...] = (
        "shell", "apply_patch", "container.exec",
        "update_plan",
        "view_image", "image_view",
        # codex 0.128+ multi-agent / spawn_agent protocol — REQUIRED to
        # keep, otherwise the advisor agent (and any other sub-agent
        # role) becomes unreachable. Bug found 2026-05-10: agent had
        # NEVER called spawn_agent, so frontier_trim_tools dropped it
        # every turn (chicken-and-egg). User's `~/.codex/config.toml`
        # has multi_agent=true + [agents.advisor] registered, but the
        # tool literally wasn't in the request the executor saw on
        # frontier route. log line:
        #   "dropped_names": [..., "close_agent", "spawn_agent",
        #                     "wait_agent", "resume_agent", ...]
        "spawn_agent", "wait_agent", "close_agent", "resume_agent",
        # User-input request channel — codex needs this to surface
        # genuine clarifications (the only legit "ask user" path).
        # Stripping it forces the agent to fall back to plain text,
        # which the soft_completion classifier then flags as a punt.
        "request_user_input", "send_input",
        # MCP advisor and the user's most-used MCP tools
        "mcp__advisor__ask_advisor",
    )

    # Skip injecting the advisor-sub-agent usage hint when the request is
    # already routed to frontier. The hint exists to teach the cheap local
    # model that it CAN escalate to a frontier advisor — pointless on the
    # frontier itself, where the model IS the advisor. Saves ~1-2k tokens
    # per frontier request.
    frontier_skip_advisor_hint: bool = True

    # Advisor recommendation classifier: ask the LOCAL model itself
    # whether this turn deserves stronger strategic guidance. Aligned
    # with Anthropic Advisor Strategy (claude.com/blog/the-advisor-strategy):
    # frontier should be a short advisor consultation, not the default
    # executor for an entire agent loop. See tinyctx/self_classify.py.
    #
    # Default ON. This is a local-backend classifier only; by default it
    # records advisor telemetry and route reasons while keeping execution
    # local. Tool-result roundtrips are skipped, so this only fires on
    # fresh user queries.
    self_classify_enabled: bool = True
    # P(advisor-needed) >= this → advisor recommendation. Lower = more
    # aggressive recommendation.
    self_classify_threshold: float = 0.7
    # Legacy / emergency switch: when true, a self-classify hit routes
    # the whole turn to frontier. Default false keeps frontier usage near
    # advisor-strategy levels instead of making frontier the executor.
    self_classify_escalates_to_frontier: bool = True
    # Time budget for the classifier call. Reasoning-class local models
    # (qwen3.x-think, DeepSeek-R1 family) burn 200-1500 tokens on hidden
    # chain-of-thought before emitting the JSON verdict; at 50 tok/s that
    # is 4-30s wall-clock. 30s default lets most cases complete and falls
    # back gracefully (returns None → router uses other signals) on the
    # truly slow ones. If your local model is non-reasoning, you can drop
    # this to 5s without losing accuracy.
    self_classify_timeout_s: float = 30.0
    # Boundary-turn self-consistency: when the advisor classifier lands in
    # this probability band, sample local next-action signatures. Agreement
    # keeps the executor local; disagreement is the measured signal that
    # justifies a frontier route.
    self_consistency_enabled: bool = True
    self_consistency_boundary_low: float = 0.55
    self_consistency_boundary_high: float = 0.85
    self_consistency_sample_count: int = 3
    self_consistency_timeout_s: float = 20.0

    # Self-improvement: performance regression watchdog. Tracks per-request
    # metrics (route, status, latency, bytes) and periodically evaluates
    # against historical baselines. When degradation is detected (error rate
    # 2x baseline, latency 1.5x baseline), the SelfImprovementGuard escalates
    # the next request to frontier. Default ON. See tinyctx/self_improvement.py.
    self_improvement_enabled: bool = True
    self_improvement_eval_interval: int = 50
    self_improvement_stats_window: int = 50

    # Stuck-loop watchdog: after the agent has run N+ turns without
    # convergence (defined as: tracker not advancing, no advisor call
    # in the recent past), inject a `<system-reminder>` into the next
    # request's input asking the agent to either consult advisor or
    # surface its blocker to the user. See tinyctx/stuck_loop.py.
    #
    # Triggered by live trace 2026-05-10 where a single codex session
    # ran 1323 turns on the same sub-problem before being terminated
    # by codex.app's own context limits.
    stuck_loop_watchdog_enabled: bool = True
    # Minimum turn_count before considering a session stuck. Healthy
    # tasks usually finish < 80 turns; >150 is a strong stuck signal.
    stuck_loop_turn_trigger: int = 80
    # Min turns between consecutive reminders. Don't nag every turn —
    # give the model 50 turns to act on the previous nudge.
    stuck_loop_turn_gap: int = 50
    # If the agent already called advisor within this many seconds,
    # skip the watchdog nudge (it's already doing the right thing).
    stuck_loop_advisor_grace_s: float = 600.0

    # Graduated escalation ladder (REFINE → PIVOT → SEARCH → BLOCKER).
    # Prevents the binary "fail once → jump to dead frontier" stall by
    # escalating through levels with strategy-change reminders at each
    # step. Counters survive compaction; reset on session end.
    escalation_ladder_enabled: bool = True

    # Soft-completion gate: when an agent ends a turn by asking the user
    # a meta-question ("what would you like to work on next?") instead
    # of completing tracker items + running verification, the proxy
    # detects the pattern in the streamed output and on the next request
    # injects a `<system-reminder>` requiring the agent to vet the
    # would-be question through advisor first. See soft_completion.py.
    #
    # Triggered by live trace 2026-05-10 where stuck_loop saved a
    # session from a 1300+ turn loop, but the agent then soft-punted
    # to the user instead of completing the 4 tracker items.
    soft_completion_gate_enabled: bool = True

    # Short-text floor for the LLM classifier. Two values because
    # finish_reason=stop and finish_reason=length/incomplete carry
    # different signal density — see soft_completion.py docstring.
    #
    # `short_text_threshold` (50) applies to length / incomplete /
    # completed / null finish_reasons — partial / truncated streams
    # where a short fragment is rarely a real punt.
    #
    # `stop_text_threshold` (1) applies to finish_reason=stop. Per
    # user directive 2026-05-10: classify EVERY stop, even very short
    # ones ("Done." / "好的。"), because those CAN be real soft-punts
    # the original 50-char floor was missing. Tradeoff is ~2-5× more
    # LLM classifier calls per session — bounded by the per-turn
    # frequency of stop streams (most turns end on tool_calls anyway,
    # which still short-circuits before the threshold check).
    soft_completion_short_text_threshold: int = 50
    soft_completion_stop_text_threshold: int = 1

    # Output-quality verifier: post-stream LLM-as-a-Verifier quality
    # check on LOCAL-routed responses. Scores 3 criteria (task_completion,
    # output_quality, execution_evidence) 1–5 each. If total < threshold,
    # the next request for this session is forced to frontier.
    #
    # Only applies to local-route responses (frontier self-verification
    # would waste a second expensive call for marginal gain).
    verifier_enabled: bool = True
    # Total score threshold (max 15). Default 8/15 ≈ 53 % — catches
    # clearly-deficient outputs without flagging every slightly-above-
    # mediocre response.
    verifier_threshold: int = 8
    # Time budget for the verifier LLM call. Same backend as
    # self_classify, so reuses the self_classify_timeout_s default.
    verifier_timeout_s: float = 30.0

    # Stream rewriting: when soft_completion classifier returns PUNT
    # with confidence ≥ rewrite_threshold, intercept the upstream's
    # `response.completed` event, run the classifier synchronously,
    # and if punt → inject a synthetic function_call to the advisor
    # MCP tool BEFORE the response.completed. Codex sees the function
    # call, routes to advisor automatically. Bypasses the rule-based
    # gate (which depends on agent self-discipline).
    #
    # Risk profile: relies on codex parsing a synthetic function_call
    # event. If codex's namespace dispatcher rejects the call (e.g.
    # the codex 0.128 "unsupported call" issue noted in
    # ~/.codex/config.toml), the rewrite would surface as an error in
    # codex chat. Default ON; explicit config can disable if a codex
    # version rejects the synthetic event shape.
    soft_completion_stream_rewrite_enabled: bool = True
    # Confidence required for the synthetic function_call rewrite.
    # Higher than the gate threshold (0.7) since the rewrite is more
    # invasive — only act on high-confidence verdicts.
    soft_completion_stream_rewrite_threshold: float = 0.85
    # Function name for the synthetic call. Codex 0.128+ supports the
    # native sub-agent primitive `spawn_agent` — `~/.codex/config.toml`
    # `[agents.advisor]` registration makes `role="advisor"` route to
    # the configured advisor agent. The legacy `mcp__advisor__ask_advisor`
    # MCP tool hits codex's namespace dispatcher bug on 0.128 and
    # returns "unsupported call". Override here for older codex.
    soft_completion_stream_rewrite_tool_name: str = "spawn_agent"
    # Extra arguments merged into the synthetic call's `arguments` JSON
    # alongside `task`. For `spawn_agent` codex requires `role` to pick
    # which sub-agent to dispatch to.
    soft_completion_stream_rewrite_extra_args: dict[str, str] = field(
        default_factory=lambda: {"role": "advisor"})

    # P2: per-session synthetic-continue injection budget. After this
    # many synthetic continues for one session, `build_continue_injection`
    # returns the `budget_exhausted` sentinel; the proxy stops yielding
    # synthetic events, sets the force-frontier flag for the next turn,
    # and injects a one-shot `<system-reminder>` warning the agent that
    # tinyctx auto-continued N times and may have been wrong about the
    # task being incomplete. Inspired by openai/symphony SPEC §7.1
    # `agent.max_turns`.
    max_continue_injections_per_session: int = 20

    # Choice arbiter: when the soft_completion classifier detects an
    # "asks user which option" pattern and the stream_rewrite would
    # inject a noop continue, first run the choice arbiter pipeline:
    #   1. Judge (local model): confirm this is a choice-ask, extract options
    #   2. Advisor (frontier model): pick the best option
    #   3. Store verdict in session_state
    #   4. Next request: ChoiceArbiterGuard injects advisor's pick as
    #      synthetic user message, so the model sees the decision and
    #      continues without re-asking.
    # Default ON — cost is one local + one frontier call per detected
    # choice-ask (~1-2% of soft_punt events). See choice_arbiter.py.
    choice_arbiter_enabled: bool = True

    # Empty-response guard: detect when the local backend returns
    # essentially nothing (completion_tokens < threshold + normal stop)
    # and force the NEXT turn for that session to frontier. Caught
    # DeepSeek-v4-flash silently degrading at 724K context (live trace
    # 2026-05-10 turn 1780: 1 completion token + finish_reason=stop →
    # codex displayed empty response → user thought session crashed).
    # See tinyctx/empty_response_guard.py.
    empty_response_guard_enabled: bool = True
    # Threshold below which a finish_reason=stop response is "empty".
    # 5 tokens lets through brief acknowledgments ("OK", "Done.") but
    # catches the 1-token failure mode.
    empty_response_min_completion_tokens: int = 5

    # Auto-force frontier on high-confidence soft-punt verdicts. Layer
    # on top of empty_response_guard's flag mechanism: when the LLM-
    # based soft_completion classifier returns PUNT with p >= threshold,
    # also set the same `_FORCE_NEXT_TO_FRONTIER` flag. The next turn
    # for that session auto-routes to frontier (gpt-5.5 via chatgpt.com)
    # — the deterministic fix that doesn't depend on codex parsing
    # synthetic events or agent self-discipline.
    #
    # Codex-side rationale (verified by binary analysis): codex's
    # function_call dispatcher validates against internal state and
    # silently drops synthetic injections that aren't tied to real
    # model reasoning. So stream-rewrite is unreliable. Forcing the
    # NEXT request to frontier sidesteps that entirely.
    #
    # Costs: each fire is one extra frontier turn (~5-15K gpt-5.5
    # tokens, ~$0.05-0.15). One-shot per detection (consumed on use),
    # so won't escalate indefinitely.
    soft_completion_auto_force_frontier_enabled: bool = True
    # Higher than the gate's 0.7 threshold — only escalate to frontier
    # on high-confidence PUNT verdicts to limit cost.
    soft_completion_auto_force_frontier_threshold: float = 0.85

    # Forensics: when an empty response is detected OR a high-confidence
    # PUNT is classified, dump the FULL request body + response buffer +
    # timing to ~/.tinyctx/forensics/. Lets us post-mortem rare
    # failures (the 05:07 empty response had no captured request body
    # and is forever unrecoverable). See tinyctx/forensics.py.
    forensics_enabled: bool = True
    # Trigger forensics for high-confidence PUNT verdicts too. Each
    # dump is ~10-50KB; with max_dumps=100 cap, total bounded ~5MB.
    forensics_capture_punts: bool = True
    # Threshold for PUNT-triggered forensics. 0.9 keeps the volume low
    # (only the cleanest punts trigger).
    forensics_punt_threshold: float = 0.9
    # Trigger forensics on upstream errors / stream errors / disconnect.
    forensics_capture_errors: bool = True
    # Max forensic dumps to retain (oldest deleted past this count).
    forensics_max_dumps: int = 100

    # When the local model emits a function_call whose `name` is NOT in the
    # codex-side tool registry (typo, hallucinated tool, schema mismatch),
    # codex's dispatcher cannot resolve it and the session stalls. With this
    # flag ON, the translator drops the bad function_call SSE events and
    # replaces them with a synthetic `shell ["echo", "tinyctx: dropped ..."]`
    # call (codex always dispatches `shell`). The model sees the echo output
    # next turn and self-corrects. See tinyctx/tool_call_translator.py.
    # Set to False to fall back to today's pass-through behavior (escape
    # hatch in case the replacement causes regressions).
    unknown_tool_call_protection: bool = True

    # Cross-thread plan persistence: when codex's update_plan is called,
    # save the plan to disk keyed by working directory. When a new codex
    # thread opens on the same repo (turn_count==0), inject the persisted
    # plan into instructions so context isn't lost across thread switches.
    # See tinyctx/plan_persistence.py.
    plan_persistence_enabled: bool = True
    # TTL after which a persisted plan is no longer auto-injected.
    plan_persistence_ttl_s: int = 7 * 24 * 3600  # 7 days

    # C-4 hybrid: codex exec resume "poke". When the soft_completion
    # classifier returns a high-confidence PUNT, fire `codex exec resume
    # <session_id> "<exec_resume_prompt>"` in a side process. exec is
    # one-shot (no finish_reason=stop wait-for-user), so it forces a
    # new turn into the SAME session without requiring user input —
    # turning the empty_response_guard / auto_force_frontier flag from
    # passive (waits on user) into active (immediate side-process turn).
    # See tinyctx/exec_resume.py.
    #
    # Conservative defaults: read-only sandbox, approval=never. Agent
    # can think + plan + advisor + read-only commands but cannot modify
    # files. Once the loop is validated, set
    # `exec_resume_sandbox = "workspace-write"` in config.toml.
    exec_resume_enabled: bool = True
    # Min PUNT confidence to trigger. Aligned with
    # soft_completion_auto_force_frontier_threshold (0.85) so the same
    # high-bar verdict drives both the flag AND the active poke.
    exec_resume_min_p: float = 0.85
    # Per-session cooldown (seconds) before another poke for the same
    # session. Stops feedback loops if the poke itself triggers another
    # PUNT verdict.
    exec_resume_cooldown_s: int = 300
    # Global cap across all sessions to bound API spend if many sessions
    # PUNT at once.
    exec_resume_max_per_minute: int = 3
    # Prompt the side process sees as a new user message. Keep concise
    # so the agent doesn't fixate on the prompt and forgets the actual
    # work. Per user directive Q1=(b).
    exec_resume_prompt: str = (
        "continue working from where you left off; do not over-explain, "
        "just take the next action")
    # SPEC §12.3-style tiered prompts. The proxy passes this list to
    # `exec_resume.poke(prompt_tiers=...)`; `select_tier_prompt` picks
    # which one fires based on the per-session poke count:
    #   count 0..1 -> gentle  (tiers[0])
    #   count 2..4 -> firm    (tiers[1])
    #   count >= 5 -> final   (tiers[2]) — after which the next
    #                          would-be poke is skipped with reason
    #                          `tier_exhausted` and the next request
    #                          for that session is forced to frontier.
    exec_resume_prompt_tiers: list[str] = field(default_factory=lambda: [
        ("tinyctx auto-continue: please continue if there's remaining "
         "work for this session, otherwise summarize and stop."),
        ("tinyctx auto-continue (3rd nudge): you appear stuck. Run any "
         "pending verification, consult advisor if confused, or stop "
         "and surface a concrete blocker to the user."),
        ("tinyctx auto-continue (final nudge): the auto-continue budget "
         "for this session is nearly exhausted. Either complete the "
         "task now or stop and surface a clear blocker — the next stop "
         "will route to frontier instead of nudging again."),
    ])
    # Subprocess timeout (seconds). Killed at this point; partial log
    # preserved.
    exec_resume_timeout_s: int = 60
    # Override codex binary path. Empty = auto-detect (codex.app's
    # bundled binary, then $PATH).
    exec_resume_codex_binary: str = ""
    # Sandbox mode the poked turn runs under. read-only is safest;
    # workspace-write lets the agent actually finish the work it was
    # paused on (recommended once you trust the loop).
    exec_resume_sandbox: str = "read-only"
    # approval_policy override for the poked turn. `never` is the only
    # value that keeps the subprocess non-interactive — anything else
    # will deadlock waiting for user input.
    exec_resume_approval_policy: str = "never"

    # In-proxy retry on transient upstream stream errors. Per user
    # directive: "先重试原来模型，再出错就升级". On RemoteProtocolError /
    # ReadTimeout / ConnectError, retry the SAME backend up to N more
    # times. After exhausting, set force-frontier flag so the NEXT
    # request from codex auto-routes to gpt-5.5.
    #
    # Only retries when bytes_out is small (no meaningful content sent
    # to client yet) — otherwise duplicate-content risk.
    upstream_retry_enabled: bool = True
    upstream_retry_count: int = 1  # 1 retry = 2 attempts total
    # Max bytes already yielded to client at which retry is still safe.
    # Above this, we've sent real content and can't redo cleanly.
    upstream_retry_max_bytes_yielded: int = 4096

    # ── Unified retry policy (retry_policy.py) ─────────────────────────
    # Per user directive 2026-05-11: "凡是中断了都要加重试" — every
    # interruption should trigger retry. Layer on top of the legacy
    # mid-stream-error retry above. The classifier in retry_policy.py
    # consults these flags to decide whether to retry same / escalate /
    # propagate for each (route, status/exception) combination.
    #
    # Flow:
    #   1. local 400/422 → escalate to frontier (chatgpt.com accepts
    #      every codex-emitted role/field; usually fixes schema-shape
    #      rejections). Controlled by `retry_on_local_4xx_escalate_frontier`.
    #   2. local 5xx / connection drop → retry same up to
    #      `upstream_retry_count` then escalate to frontier.
    #   3. frontier 5xx / connection drop → retry same up to
    #      `upstream_retry_count` then propagate (no further escalation
    #      possible).
    #   4. frontier 4xx → propagate (chatgpt.com is strict; same body
    #      retry rarely helps). Controlled by `retry_on_frontier_4xx`
    #      — off by default.
    #   5. 401/403/404 (auth/forbidden/not-found) → propagate. These
    #      don't get easier on retry.
    #   6. 429 → retry same with bounded backoff (Retry-After respected,
    #      capped at 5s), then escalate.
    #   7. is_compaction → never retry. Codex self-retries with a
    #      different shape, and proxy-side retry doubles cost.
    #   8. All retries capped by `max_total_retries_per_request` as a
    #      hard safety bound (default 3).
    #
    # Each retry emits `retry_attempted` log event with attempt number,
    # status, retry_target, original_url, new_url. Escalation also marks
    # the session via empty_response_guard.force_next_to_frontier so
    # codex's subsequent turns on the same conversation auto-route
    # frontier — avoiding the ping-pong of "local fails → escalate →
    # next turn back on local with the same broken body → fails again".
    retry_on_local_4xx_escalate_frontier: bool = True
    retry_on_frontier_4xx: bool = False
    # Safety bound across all retry kinds for one request. Adding all
    # the per-bucket caps could in theory let one request burn ~6
    # attempts (local 5xx ×2 + frontier 5xx ×2 + connection ×2). Hard
    # cap stops that. Default 3 = initial + up to 2 retries.
    max_total_retries_per_request: int = 3

    # Mid-stream stall watchdog. While `empty_response_guard` catches
    # near-empty responses post-stream, it does NOT catch the case where
    # the upstream opens the SSE channel and then HANGS — no events, no
    # error, no close. Codex.app waits for `response.completed`; tinyctx
    # waits for the next byte; nothing fires. The watchdog polls each
    # session's last-event timestamp and forces escalation when the gap
    # exceeds `stall_threshold_s`. See tinyctx/stall_watchdog.py.
    #
    # Inspired by openai/symphony SPEC §8.5 Part A and codex's own
    # `stall_timeout_ms` (5 min default in codex.app).
    stall_watchdog_enabled: bool = True
    # Seconds without an upstream event before declaring a stall. 180s
    # is generous enough to absorb large-context cold starts (DeepSeek
    # at 500K input can take 60-90s before the first token) without
    # waiting forever for genuinely-dead sessions.
    stall_threshold_s: float = 180.0
    # How often the watchdog wakes up to check. 30s gives sub-minute
    # detection latency on top of the threshold without burning CPU.
    stall_check_interval_s: float = 30.0

    # SSE keepalive injector for long-running upstream streams. When the
    # upstream (DeepSeek / chatgpt.com / etc.) is silent for this many
    # seconds during streaming, the proxy emits a `: tinyctx keepalive`
    # SSE comment line so codex.app's stream parser sees ongoing bytes
    # and doesn't trip its idle timeout (or any TCP middlebox / firewall
    # idle disconnection). SSE comments (lines starting with `:`) are
    # ignored by spec-compliant clients.
    #
    # Default 15s — well below codex's `stream_idle_timeout_ms = 300000`
    # (5 min). Set to 0 to disable. Cap should be < client's idle
    # timeout / 2 to leave safety margin.
    stream_keepalive_interval_s: float = 15.0

    # Inject the bundled global agent rules (tinyctx/templates/AGENTS.md)
    # into every request's `body.instructions`. Idempotent — skipped
    # automatically if codex.app already loaded `~/.codex/AGENTS.md`
    # (the rules are then already in scope; no duplication). Lets a
    # fresh `git clone tinyctx` work on a new machine without the user
    # having to manually copy AGENTS.md into their codex/claude config
    # dir. See tinyctx/agent_rules.py.
    inject_global_agent_rules: bool = True

    # Inject a host-platform-specific tool/shell rules block
    # (tinyctx/platform_rules.py) so the upstream model picks correct
    # invocations for THIS machine: no .sh / grep / bash -c on Windows,
    # no apt-get / systemctl on macOS, etc. Detected once via
    # platform.system() at import, byte-stable per process so the
    # upstream prompt cache hits. Default on; disable only if your
    # deployment is genuinely cross-host (e.g. proxy on Linux serving
    # mixed-OS codex clients), in which case the global AGENTS.md is
    # the right place for any catch-all guidance.
    inject_platform_rules: bool = True

    # Auto-register external MCP servers (graphify, gitnexus) into
    # ~/.codex/config.toml on proxy startup. The registration is
    # idempotent (delegates to _codex_toml.append_mcp_block — line-exact
    # marker check + fcntl.flock to prevent races with the explicit
    # bootstrap modules below) and skipped entirely when the tools
    # aren't on PATH. See tinyctx/mcp_registry.py for the full contract.
    auto_register_mcp_servers: bool = True

    # Auto-install missing components (context-mode, gitnexus, graphify,
    # serena, caveman) on proxy startup. Each component's bootstrap is
    # idempotent — already-installed components are skipped. Set false to
    # disable (e.g. in air-gapped or managed deployments).
    auto_install_missing_components: bool = True

    # Caveman-style inline compression: apply prose-shortening regex rules
    # to tool descriptions and instructions in the proxy pipeline. Pure
    # Python, zero dependencies, no codex config changes needed. Default on
    # — typical 30-50% reduction on tool description tokens.
    caveman_compress_enabled: bool = True
    # Compress tool descriptions (the `description` field of each tool in
    # body.tools). This is the biggest per-tool token cost in the system
    # prompt — caveman rules cut verbosity while preserving code/paths/URLs.
    caveman_compress_tool_descriptions: bool = True
    # Compress body.instructions (the system prompt). Deterministic, so
    # unconfigured installs get the shorter byte-stable prompt by default.
    caveman_compress_instructions: bool = True

    # When frontier is unreachable, the proxy auto-falls-back to local
    # and injects a <system-reminder> so the model tells the user.
    # After cooldown expires, the next request retries frontier.
    # Exponential backoff per consecutive failure: 30→60→120→300s max.
    frontier_unreachable_cooldown_s: float = 30.0

    # Auto-scout: zero-config project context bootstrap.
    # When True, the proxy reads `x-codex-cwd` from each request and
    # ensures ~/.tinyctx/cache/<repo-hash>/scout.md exists for that
    # project — building it asynchronously the first time, injecting it
    # into request.instructions on subsequent requests. See
    # tinyctx/auto_scout.py for the full pipeline.
    auto_scout: bool = True
    # When True AND `graphify` is missing on PATH, attempt a one-shot
    # `pipx install graphifyy` during the first project bootstrap. Default
    # OFF: normal request handling must not install external packages. The
    # fallback in-tree scanner works without graphify.
    auto_scout_install_graphify: bool = False

    # ctx-pack: preemptively inject top-K project files (ranked by
    # compression-biased PageRank from the project's graph.json) into
    # request body.instructions.  Replaces N ctx_execute_file tool
    # calls with a single context block.  Default true; set false to
    # disable (e.g. when graph.json is stale or unavailable).
    ctx_pack_enabled: bool = True

    # Instant project structure snapshot: synchronous dir-tree overview
    # (~30ms) injected on every first request so the model has structural
    # context from turn 0.  Complements the richer but async scout/ctx-pack
    # pipeline.  Set false to disable.
    snapshot_enabled: bool = True

    # Task Orchestrator: local-model task typing + Skill/MCP recommendation
    # injection. Keeps the actual Codex tool execution unchanged; tinyctx
    # only appends current-task guidance to instructions.
    orchestrator_enabled: bool = True
    orchestrator_min_confidence: float = 0.62
    orchestrator_dynamic_skill_enabled: bool = True
    orchestrator_dynamic_skill_min_confidence: float = 0.78
    orchestrator_inject_max_chars: int = 2000
    orchestrator_trace_decisions: bool = True

    # LLMLingua-2 pre-escalation prompt compression for the frontier path.
    # Microsoft's LLMLingua-2 (microsoft/LLMLingua, MIT) compresses tool-
    # result payloads before forwarding to the frontier model. Empirically
    # 2-5× compression on long contexts with no quality loss on coding/QA.
    #
    # Default ON for zero-config installs. The hook first checks whether
    # the optional `llmlingua` package is importable; when absent it is a
    # no-op. Compression still runs behind the same cache-aware gate as
    # dedup/purge/read_delta.
    frontier_lingua_enabled: bool = True
    # Compression aggressiveness. 0.5 = keep ~50% of tokens. Conservative
    # 0.6-0.7 gives 1.5-2× shrink without quality loss. Below 0.4 starts
    # losing detail.
    frontier_lingua_ratio: float = 0.5
    # Items below this many chars aren't worth compressing.
    frontier_lingua_min_bytes: int = 800

    # If set, force every request through one of {"local", "frontier", "auto"}.
    # Useful for debugging.
    force_route: str = field(default_factory=lambda: _env("TINYCTX_FORCE_ROUTE", "auto") or "auto")

    # Verbose JSONL logging
    verbose: bool = field(default_factory=lambda: (_env("TINYCTX_VERBOSE", "1") or "1") == "1")

    # ── P8 namespaced views over related top-level flags ────────────────
    # These do NOT own storage; they forward attribute access to the
    # canonical fields above so old call sites (`cfg.force_route`, etc.)
    # and TOML parsing (flat keys) keep working untouched. New code can
    # use the more legible nested form (`cfg.routing.force_route`).
    # Constructed in `__post_init__`.
    routing: "RoutingView" = field(init=False, repr=False)
    stall: "StallView" = field(init=False, repr=False)
    retry: "RetryView" = field(init=False, repr=False)
    compact: "CompactView" = field(init=False, repr=False)
    guards: "GuardsView" = field(init=False, repr=False)
    stuck_loop: "StuckLoopView" = field(init=False, repr=False)
    forensics: "ForensicsView" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Bind one view-per-instance so mutating one Config cannot bleed
        # into another (see test_namespace_views_are_per_instance).
        self.routing = RoutingView(self)
        self.stall = StallView(self)
        self.retry = RetryView(self)
        self.compact = CompactView(self)
        self.guards = GuardsView(self)
        self.stuck_loop = StuckLoopView(self)
        self.forensics = ForensicsView(self)


def effective_proactive_compact_threshold(cfg: "Config") -> int:
    """Compute the actual proactive_compact threshold for the current
    runtime, derived from frontier.context_window when available.

    Resolution order:
      1. If `frontier.context_window > 0` AND
         `proactive_compact_safe_fraction > 0`:
         base = int(context_window * safe_fraction)
      2. Else: base = cfg.proactive_compact_threshold (absolute fallback)

    The returned value is `base - proactive_compact_overhead_buffer`,
    clamped to >= 0. The buffer accounts for tinyctx's own injections
    (agent rules, advisor hint, scout, tool catalog) that inflate the
    forwarded payload above the user-body `est_input_tokens` that the
    gate is checked against — without it the gate under-fires by
    roughly the overhead amount.

    Returning 0 disables proactive_compact entirely. The proxy treats 0
    as "skip the gate".
    """
    cw = getattr(cfg.frontier, "context_window", 0) or 0
    sf = cfg.proactive_compact_safe_fraction
    if cw > 0 and sf > 0:
        base = int(cw * sf)
    else:
        base = int(cfg.proactive_compact_threshold or 0)
    if base <= 0:
        return 0
    buffer = int(getattr(cfg, "proactive_compact_overhead_buffer", 0) or 0)
    return max(0, base - buffer)


def effective_strip_request_fields(backend: BackendCfg) -> tuple[str, ...]:
    """Return top-level request fields to drop for this backend."""
    fields = tuple(backend.strip_request_fields or ())
    if backend.lmcache_passthrough:
        fields = tuple(f for f in fields if f != "prompt_cache_key")
    return fields


def load_config() -> Config:
    """Layered config: file provides base values; env vars override.

    Order:
      1. dataclass field defaults (built-in)
      2. TOML file at $TINYCTX_CONFIG or ~/.tinyctx/config.toml (file overrides defaults)
      3. TINYCTX_* env vars (env overrides file)
    """
    cfg = Config()
    log_override = _env("TINYCTX_LOG_DIR")
    if log_override:
        cfg.log_dir = Path(log_override).expanduser()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Step 2 — file (config.toml)
    path = Path(_env("TINYCTX_CONFIG", str(Path.home() / ".tinyctx" / "config.toml")) or "")
    if path.is_file() and tomllib is not None:
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        for k, v in (data.get("server") or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        for section, target in (("local", cfg.local), ("frontier", cfg.frontier)):
            for k, v in (data.get(section) or {}).items():
                if hasattr(target, k):
                    setattr(target, k, v)
        for k, v in (data.get("routing") or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        orch_map = {
            "enabled": "orchestrator_enabled",
            "min_confidence": "orchestrator_min_confidence",
            "dynamic_skill_enabled": "orchestrator_dynamic_skill_enabled",
            "dynamic_skill_min_confidence": "orchestrator_dynamic_skill_min_confidence",
            "inject_max_chars": "orchestrator_inject_max_chars",
            "trace_decisions": "orchestrator_trace_decisions",
        }
        for k, v in (data.get("orchestrator") or {}).items():
            attr = orch_map.get(k)
            if attr and hasattr(cfg, attr):
                setattr(cfg, attr, v)

    # Step 2.5 — optimized policy overlay (from policy_search)
    # Loads ~/.tinyctx/policies/optimized.toml if it exists.  These values
    # override config.toml defaults but are themselves overridden by env vars
    # (step 3).  This is the "search offline, apply online" bridge.
    _policy_path = Path.home() / ".tinyctx" / "policies" / "optimized.toml"
    if _policy_path.is_file() and tomllib is not None:
        try:
            _policy_data: dict[str, Any] = tomllib.loads(_policy_path.read_text(encoding="utf-8"))
            # Map policy_search field names → Config field names where they differ
            _policy_field_map: dict[str, str] = {
                "max_continue_injections": "max_continue_injections_per_session",
            }
            for k, v in _policy_data.items():
                attr = _policy_field_map.get(k, k)
                if hasattr(cfg, attr):
                    setattr(cfg, attr, v)
        except Exception:  # noqa: BLE001 — policy file must never block boot
            pass

    # Step 3 — env overrides
    for env_key, attr in (
        ("TINYCTX_LOCAL_BASE_URL", ("local", "base_url")),
        ("TINYCTX_LOCAL_MODEL", ("local", "model")),
        ("TINYCTX_LOCAL_WIRE_API", ("local", "wire_api")),
        ("TINYCTX_LOCAL_PROXY", ("local", "proxy")),
        ("TINYCTX_FRONTIER_BASE_URL", ("frontier", "base_url")),
        ("TINYCTX_FRONTIER_MODEL", ("frontier", "model")),
        ("TINYCTX_FRONTIER_WIRE_API", ("frontier", "wire_api")),
        ("TINYCTX_FRONTIER_PROXY", ("frontier", "proxy")),
    ):
        v = _env_proxy(env_key) if attr[1] == "proxy" else _env(env_key)
        if v is not None or (attr[1] == "proxy" and env_key in os.environ):
            section, key = attr
            setattr(getattr(cfg, section), key, v)
    fr = _env("TINYCTX_FORCE_ROUTE")
    if fr is not None:
        cfg.force_route = fr
    sce = _env_bool("TINYCTX_SELF_CLASSIFY_ESCALATES_TO_FRONTIER")
    if sce is not None:
        cfg.self_classify_escalates_to_frontier = sce
    gcf = _env_bool("TINYCTX_GOAL_CONTROL_FRONTIER_ENABLED")
    if gcf is not None:
        cfg.goal_control_frontier_enabled = gcf
    ame = _env_bool("TINYCTX_ADAPTIVE_MODEL_ENABLED")
    if ame is not None:
        cfg.adaptive_model_enabled = ame
    ipf = _env_bool("TINYCTX_IMAGE_PREFER_FRONTIER")
    if ipf is not None:
        cfg.image_prefer_frontier = ipf
    vb = _env("TINYCTX_VERBOSE")
    if vb is not None:
        cfg.verbose = vb == "1"
    lmcache = _env_bool("TINYCTX_LOCAL_LMCACHE_PASSTHROUGH")
    if lmcache is not None:
        cfg.local.lmcache_passthrough = lmcache
    sie = _env_bool("TINYCTX_SELF_IMPROVEMENT_ENABLED")
    if sie is not None:
        cfg.self_improvement_enabled = sie
    sei = _env("TINYCTX_SELF_IMPROVEMENT_EVAL_INTERVAL")
    if sei is not None:
        cfg.self_improvement_eval_interval = int(sei)
    sisw = _env("TINYCTX_SELF_IMPROVEMENT_STATS_WINDOW")
    if sisw is not None:
        cfg.self_improvement_stats_window = int(sisw)
    return cfg
