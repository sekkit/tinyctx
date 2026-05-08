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


@dataclass
class BackendCfg:
    base_url: str
    api_key_env: str | None = None
    model: str = ""
    wire_api: str = "responses"
    timeout_s: float = 300.0
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
    # fraction. 0.85 leaves ~15% headroom for the model's output and
    # avoids the well-known long-context quality cliff.
    context_safe_fraction: float = 0.85
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


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 4141
    log_dir: Path = field(default_factory=lambda: Path.home() / ".tinyctx" / "logs")

    # Cheap path: local model. Default LMStudio. Override to vLLM/SGLang/etc.
    local: BackendCfg = field(default_factory=lambda: BackendCfg(
        base_url=_env("TINYCTX_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1") or "",
        api_key_env="TINYCTX_LOCAL_API_KEY",
        model=_env("TINYCTX_LOCAL_MODEL", "qwen3.6-27b") or "",
        wire_api=_env("TINYCTX_LOCAL_WIRE_API", "chat") or "chat",
        timeout_s=float(_env("TINYCTX_LOCAL_TIMEOUT_S", "180") or 180),
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
        # Codex.app 0.128 hard-codes context_window=272000 for gpt-5.5
        # in its model catalog. Setting it here lets
        # `effective_proactive_compact_threshold` derive the right
        # compact threshold automatically (272k × 0.75 ≈ 204k). Override
        # in config.toml [frontier] section when using a different
        # frontier (gemini, opus, etc.).
        context_window=int(_env("TINYCTX_FRONTIER_CONTEXT_WINDOW", "272000") or 272000),
        supported_tool_types=(),     # empty = keep all
        strip_request_fields=(),     # empty = keep all
        inject_defaults={},          # empty = inject nothing
        translate_tool_calls=False,  # frontier already returns structured items
    ))

    # Routing thresholds. Conservative defaults — escalate only when needed.
    escalate_input_tokens: int = 60_000     # estimated input tokens above this -> frontier
    escalate_turn_count: int = 15           # turn count above this -> frontier
    escalate_on_error_streak: int = 2       # repeated tool-failure -> frontier

    # If true, the compaction handoff prompt is always routed to the local model.
    # This is the highest-leverage cost win and the reason this proxy exists.
    redirect_compaction_to_local: bool = True

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
    #                           Safe to leave on. Costs 1 local LLM call
    #                           every `historian_min_new_turns` turns.
    #   historian_substitute  — when fire-time mutation gate opens, replace
    #                           older turns in the body with the digest.
    #                           Mutates history bytes → opt-in only.
    historian_enabled: bool = False
    historian_substitute: bool = False
    historian_min_new_turns: int = 5
    historian_recent_keep: int = 4

    # If true, scrub `encrypted_content` from prior reasoning items when the
    # request is being routed to a different model than the one that produced
    # them. Avoids the "encrypted content could not be decrypted" crash.
    sanitize_encrypted_content: bool = True

    # History-hygiene transforms (DCP-inspired). When enabled, the
    # CacheAwareMutator (sanitize.py) gates them so they fire only when the
    # prompt cache is likely stale anyway, preserving cache hits in the
    # common case. Inspired by cortexkit/magic-context.
    dedup_tool_calls: bool = False
    purge_failed_tool_inputs: bool = False
    failed_input_after_turns: int = 4
    mutation_ttl_s: float = 300.0      # default = Anthropic 5-min cache TTL
    mutation_threshold: float = 0.65   # context usage trigger (0..1)
    default_context_window: int = 1_000_000  # used when request doesn't say

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
    frontier_trim_tools: bool = True
    frontier_tools_recent_window: int = 30  # how many recent input items to scan
    frontier_tools_essentials: tuple[str, ...] = (
        "shell", "apply_patch", "container.exec",
        "update_plan",
        "view_image", "image_view",
        # MCP advisor and the user's most-used MCP tools
        "mcp__advisor__ask_advisor",
    )

    # Skip injecting the advisor-sub-agent usage hint when the request is
    # already routed to frontier. The hint exists to teach the cheap local
    # model that it CAN escalate to a frontier advisor — pointless on the
    # frontier itself, where the model IS the advisor. Saves ~1-2k tokens
    # per frontier request.
    frontier_skip_advisor_hint: bool = True

    # If set, force every request through one of {"local", "frontier", "auto"}.
    # Useful for debugging.
    force_route: str = field(default_factory=lambda: _env("TINYCTX_FORCE_ROUTE", "auto") or "auto")

    # Verbose JSONL logging
    verbose: bool = field(default_factory=lambda: (_env("TINYCTX_VERBOSE", "1") or "1") == "1")


def effective_proactive_compact_threshold(cfg: "Config") -> int:
    """Compute the actual proactive_compact threshold for the current
    runtime, derived from frontier.context_window when available.

    Resolution order:
      1. If `frontier.context_window > 0` AND
         `proactive_compact_safe_fraction > 0`:
         return int(context_window * safe_fraction)
      2. Else: return cfg.proactive_compact_threshold (absolute fallback)

    Returning 0 disables proactive_compact entirely. The proxy treats 0
    as "skip the gate".

    This lets users switch frontier models (gpt-5.5 → gemini-2.5 →
    smaller) and have the threshold track context_window automatically,
    without rewiring config every time.
    """
    cw = getattr(cfg.frontier, "context_window", 0) or 0
    sf = cfg.proactive_compact_safe_fraction
    if cw > 0 and sf > 0:
        return int(cw * sf)
    return int(cfg.proactive_compact_threshold or 0)


def load_config() -> Config:
    """Layered config: file provides base values; env vars override.

    Order:
      1. dataclass field defaults (built-in)
      2. TOML file at $TINYCTX_CONFIG or ~/.tinyctx/config.toml (file overrides defaults)
      3. TINYCTX_* env vars (env overrides file)
    """
    cfg = Config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Step 2 — file
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

    # Step 3 — env overrides
    for env_key, attr in (
        ("TINYCTX_LOCAL_BASE_URL", ("local", "base_url")),
        ("TINYCTX_LOCAL_MODEL", ("local", "model")),
        ("TINYCTX_LOCAL_WIRE_API", ("local", "wire_api")),
        ("TINYCTX_FRONTIER_BASE_URL", ("frontier", "base_url")),
        ("TINYCTX_FRONTIER_MODEL", ("frontier", "model")),
        ("TINYCTX_FRONTIER_WIRE_API", ("frontier", "wire_api")),
    ):
        v = _env(env_key)
        if v is not None:
            section, key = attr
            setattr(getattr(cfg, section), key, v)
    fr = _env("TINYCTX_FORCE_ROUTE")
    if fr is not None:
        cfg.force_route = fr
    vb = _env("TINYCTX_VERBOSE")
    if vb is not None:
        cfg.verbose = vb == "1"
    return cfg
