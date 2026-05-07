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
    # Per-backend tool/field scrubbing. Codex emits tool entries with
    # codex-specific `type` values (web_search, image_generation, namespace)
    # and fields (client_metadata, prompt_cache_key) that strict OpenAI-
    # compat backends like LMStudio reject with HTTP 400. The default keep-
    # set covers every common local backend; OpenAI's own endpoint accepts
    # everything so the frontier backend's defaults are effectively no-ops.
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

    # If set, force every request through one of {"local", "frontier", "auto"}.
    # Useful for debugging.
    force_route: str = field(default_factory=lambda: _env("TINYCTX_FORCE_ROUTE", "auto") or "auto")

    # Verbose JSONL logging
    verbose: bool = field(default_factory=lambda: (_env("TINYCTX_VERBOSE", "1") or "1") == "1")


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
