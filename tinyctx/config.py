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
        model=_env("TINYCTX_LOCAL_MODEL", "qwen3.6-27b") or "",
        wire_api=_env("TINYCTX_LOCAL_WIRE_API", "chat") or "chat",
        timeout_s=float(_env("TINYCTX_LOCAL_TIMEOUT_S", "180") or 180),
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

    # Model-driven escalation: ask the LOCAL model itself whether to
    # escalate this turn to frontier. Aligned with Anthropic Advisor
    # Strategy (claude.com/blog/the-advisor-strategy) — the executor
    # decides, not infrastructure-by-bytes. See tinyctx/self_classify.py
    # for the prompt and contract.
    #
    # Default ON. Cost depends on local backend speed; cached 60s by
    # per-project key so codex retries don't re-classify. Tool-result
    # roundtrips are skipped, so this only fires on fresh user queries.
    self_classify_enabled: bool = True
    # P(escalate) >= this → frontier. 0.7 matches the existing trained-
    # classifier threshold. Lower = more aggressive escalation.
    self_classify_threshold: float = 0.7
    # Time budget for the classifier call. Reasoning-class local models
    # (qwen3.x-think, DeepSeek-R1 family) burn 200-1500 tokens on hidden
    # chain-of-thought before emitting the JSON verdict; at 50 tok/s that
    # is 4-30s wall-clock. 30s default lets most cases complete and falls
    # back gracefully (returns None → router uses other signals) on the
    # truly slow ones. If your local model is non-reasoning, you can drop
    # this to 5s without losing accuracy.
    self_classify_timeout_s: float = 30.0

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

    # Auto-register external MCP servers (graphify, gitnexus) into
    # ~/.codex/config.toml on proxy startup. The registration is
    # idempotent (managed block between BEGIN/END markers, replaced
    # byte-for-byte on subsequent runs) and skipped entirely when the
    # tools aren't on PATH. See tinyctx/mcp_registry.py for the full
    # contract — what gets touched, what doesn't, and license notes
    # (gitnexus is PolyForm Noncommercial, logged on detection).
    auto_register_mcp_servers: bool = True

    # Auto-scout: zero-config project context bootstrap.
    # When True, the proxy reads `x-codex-cwd` from each request and
    # ensures ~/.tinyctx/cache/<repo-hash>/scout.md exists for that
    # project — building it asynchronously the first time, injecting it
    # into request.instructions on subsequent requests. See
    # tinyctx/auto_scout.py for the full pipeline.
    auto_scout: bool = True
    # When True AND `graphify` is missing on PATH, attempt a one-shot
    # `pipx install graphifyy` during the first project bootstrap. False
    # by default — auto-installing OS-level packages is intrusive even
    # if the user opted into "transparent" mode. The fallback in-tree
    # scanner works without this.
    auto_scout_install_graphify: bool = False

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
