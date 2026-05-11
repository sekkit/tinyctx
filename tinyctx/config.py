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
    dedup_tool_calls: bool = False
    purge_failed_tool_inputs: bool = False
    failed_input_after_turns: int = 4
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
    # 2026-05-10: DEFAULT FLIPPED TO FALSE per user directive ("不要 trim
    # tools, 所有 tools 都保留"). Background: even with the
    # spawn_agent/wait_agent essentials fix, trimming creates a class of
    # rare-tool-starvation bugs (any tool not in `essentials` AND not in
    # the recent window is silently invisible). User accepts the
    # ~10k-token-per-request cost in exchange for full tool availability.
    # To re-enable trimming, set `frontier_trim_tools = true` under
    # `[server]` in ~/.tinyctx/config.toml.
    frontier_trim_tools: bool = False
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
    # codex chat. Default OFF until validated against live codex.
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

    # Auto-register external MCP servers (graphify, gitnexus) into
    # ~/.codex/config.toml on proxy startup. The registration is
    # idempotent (delegates to _codex_toml.append_mcp_block — line-exact
    # marker check + fcntl.flock to prevent races with the explicit
    # bootstrap modules below) and skipped entirely when the tools
    # aren't on PATH. See tinyctx/mcp_registry.py for the full contract.
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

    # LLMLingua-2 pre-escalation prompt compression for the frontier path.
    # Microsoft's LLMLingua-2 (microsoft/LLMLingua, MIT) compresses tool-
    # result payloads before forwarding to the frontier model. Empirically
    # 2-5× compression on long contexts with no quality loss on coding/QA.
    #
    # Default OFF because:
    #   1. Heavy first-load (downloads ~hundreds of MB of model weights)
    #   2. Mutates wire bytes — must be cache-aware-gated like
    #      dedup/purge/read_delta to preserve prompt-cache hits
    #   3. Requires `pip install 'tinyctx[compress]'` (optional dep)
    #
    # Opt-in path: install dep + flip this flag in config.toml. The hook
    # will lazy-import llmlingua and gracefully no-op if missing.
    frontier_lingua_enabled: bool = False
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
