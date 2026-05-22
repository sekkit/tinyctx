"""Routing decisions and Responses-API request analysis.

Three responsibilities:
  1. Detect codex's compaction "handoff summary" prompt.
  2. Estimate input token count cheaply (4 chars ≈ 1 token).
  3. Decide local vs frontier based on heuristics, optionally augmented
     by a learned classifier (FrugalGPT-style scorer; see classifier.py).

We do NOT trust verbalized confidence from the model. We trust observable
properties of the request (length, turn count, prompt fingerprint, error
streak via session state).

 P5 (this module): the `Router` class consolidates the previously-scattered
route-decision chain that lived inline in `proxy.responses`. One call
`Router(cfg).decide(ctx)` returns a fully-resolved `Decision` carrying
the backend URL + model + headers + wire_api + timeout, so proxy.py
no longer has to re-derive any of that. Each rule is a small `_X_rule`
method returning a Decision or None; the first non-None wins. The order
of rules is the priority order (compaction beats force_route beats
explicit-model beats goal-control beats error_streak beats adaptive
backend-health beats capacity beats classify beats default).

The legacy `decide()` free function is preserved unchanged — many tests
and the chat-completions handler still call it directly and we don't
need to widen the migration just for P5.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazily-loaded learned classifier; None if no model present on disk.
_CLASSIFIER: Any = None
_CLASSIFIER_LOADED = False


def _maybe_load_classifier():
    global _CLASSIFIER, _CLASSIFIER_LOADED
    if _CLASSIFIER_LOADED:
        return _CLASSIFIER
    _CLASSIFIER_LOADED = True
    try:
        from .classifier import Model, model_path
        path = model_path()
        if path.is_file():
            _CLASSIFIER = Model.from_json(path.read_text())
    except Exception:
        _CLASSIFIER = None
    return _CLASSIFIER

# Codex emits a stable handoff-summary prompt during local compaction. Source:
# wasnotwas.com/writing/context-compaction/, openai/codex codex-rs/core.
# We match a few invariant phrases to be robust to minor wording changes.
_COMPACTION_FINGERPRINTS = [
    re.compile(r"create a handoff summary", re.IGNORECASE),
    re.compile(r"another LLM that will resume the task", re.IGNORECASE),
    re.compile(r"seamlessly continue the work", re.IGNORECASE),
]

_GOAL_CONTROL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("slash-goal", re.compile(r"(^|\s)/goal\b", re.IGNORECASE)),
    ("goal-artifact", re.compile(
        r"\b(GOAL\.md|SPEC\.md|done_when|scorecard|feedback_loop|"
        r"human_control_surface|control\.md)\b",
        re.IGNORECASE,
    )),
    ("goal-contract", re.compile(
        r"\b(goal|/goal)\b.{0,100}\b("
        r"contract|compile|forge|generate|draft|create|tighten|spec|"
        r"acceptance|success criteria|stop condition)\b",
        re.IGNORECASE,
    )),
    ("goal-judgment", re.compile(
        r"\b(complete|done|satisfy|verify|validate|audit|review|blocked|"
        r"stuck|pivot|escalat)\w*\b.{0,100}\b(goal|done_when|acceptance)\b",
        re.IGNORECASE,
    )),
]


@dataclass
class Decision:
    route: str          # "local" | "frontier"
    reason: str         # human-readable
    is_compaction: bool = False
    est_input_tokens: int = 0
    turn_count: int = 0
    # P5: resolved backend info — populated by Router.decide so proxy.py
    # doesn't need to re-derive. Optional/empty when constructed via the
    # legacy free-function `decide()` (proxy.py callers that build a
    # Decision directly during retry-escalate continue to work).
    target: str = ""
    model: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    wire_api: str = ""
    timeout_s: float = 0.0


@dataclass
class RouteContext:
    """Inputs for `Router.decide`. The proxy assembles this from the
    incoming request + per-session state (error_streak, classify result,
    force_route flag from the guard pipeline) and hands it to the Router.

    `classify_p` / `classify_reason` are populated by `self_classify` if
    enabled — the Router only CONSUMES the score, it doesn't run the
    classifier itself (which would force this module to be async).
    """
    body: dict[str, Any]
    proj_sid: str
    conv_sid: str
    turn_count: int = 0
    est_tokens: int = 0
    requested_model: str = ""
    force_route: str | None = None     # from GuardPipeline (ForceFrontierGuard)
    error_streak: int = 0
    is_compaction: bool = False
    classify_p: float = 0.0
    classify_reason: str = ""


def _flatten_text(node: Any) -> str:
    """Walk a Responses-API request body and concatenate all text content.
    Returns a single string for token estimation + fingerprinting.
    """
    out: list[str] = []
    def _walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, list):
            for y in x:
                _walk(y)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("text", "content", "input", "instructions", "system",
                        "user", "assistant", "value", "arguments", "output"):
                    _walk(v)
                elif isinstance(v, (list, dict)):
                    _walk(v)
            return
    _walk(node)
    return "\n".join(out)


def _tail_user_text(body: dict[str, Any]) -> str:
    """Return text only when the current request ends with a user turn.

    Goal contracts often stay in instructions/history for many turns. If
    we scanned the whole body, every autonomous `/goal` iteration would
    route to frontier. The control-plane heuristic must fire only on a
    fresh user/control request, not on tool-result roundtrips.
    """
    items = body.get("input") or body.get("messages") or []
    if isinstance(items, str):
        return items
    if not isinstance(items, list) or not items:
        return ""
    last = items[-1]
    if not isinstance(last, dict):
        return ""
    role = last.get("role")
    typ = last.get("type")
    if not (role == "user" or (typ == "message" and role == "user")):
        return ""
    return _flatten_text(last.get("content"))


def goal_control_signal(body: dict[str, Any]) -> str:
    """Classify `/goal` setup/review/blocker turns that deserve frontier.

    Returns a short reason label, or "" when the request should remain in
    the normal cheap executor path.
    """
    text = re.sub(r"\s+", " ", _tail_user_text(body)).strip()
    if not text:
        return ""
    for label, pattern in _GOAL_CONTROL_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def estimate_tokens(s: str) -> int:
    """Cheap token estimate: ~4 chars/token, +25% for code/symbols.
    Good enough for routing thresholds; not for billing.
    """
    if not s:
        return 0
    n = len(s)
    return int(n / 3.6)


def count_turns(body: dict[str, Any]) -> int:
    """Count assistant turns in the request history."""
    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list):
        return 0
    n = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role") or it.get("type") or ""
        if role in ("assistant", "message", "tool_use", "function_call"):
            n += 1
    return n


def is_compaction_request(text_blob: str) -> bool:
    return any(p.search(text_blob) for p in _COMPACTION_FINGERPRINTS)


def decide(body: dict[str, Any], cfg, *, error_streak: int = 0) -> Decision:
    blob = _flatten_text(body)
    est = estimate_tokens(blob)
    turns = count_turns(body)
    compact = is_compaction_request(blob)

    if cfg.force_route == "local":
        return Decision("local", "force_route=local", compact, est, turns)
    if cfg.force_route == "frontier":
        return Decision("frontier", "force_route=frontier", compact, est, turns)

    if compact and cfg.redirect_compaction_to_local:
        return Decision("local", "compaction handoff -> cheap path",
                        is_compaction=True, est_input_tokens=est, turn_count=turns)

    if getattr(cfg, "goal_control_frontier_enabled", True):
        goal_signal = goal_control_signal(body)
        if goal_signal:
            return Decision(
                "frontier",
                f"goal-control turn -> frontier ({goal_signal})",
                est_input_tokens=est,
                turn_count=turns,
            )

    if error_streak >= cfg.escalate_on_error_streak:
        return Decision("frontier", f"error_streak={error_streak} >= threshold",
                        est_input_tokens=est, turn_count=turns)

    if getattr(cfg, "adaptive_model_enabled", True):
        try:
            from . import adaptive_model
            health = adaptive_model.local_health(cfg)
            if health.should_escalate:
                return Decision(
                    "frontier",
                    "adaptive local failure rate "
                    f"{health.failures}/{health.calls}={health.failure_rate:.0%}",
                    est_input_tokens=est,
                    turn_count=turns,
                )
        except Exception:
            pass

    # Token-size and turn-count escalation are OPT-IN as of the
    # Advisor-Strategy alignment (claude.com/blog/the-advisor-strategy).
    # Anthropic's design has the EXECUTOR MODEL decide when to invoke
    # the advisor (via spawn_agent / the advisor tool); infrastructure
    # does not auto-escalate based on byte counts. So defaults below
    # are 0 = disabled. Users with a small-context local backend
    # (LMStudio 32k, etc.) can still re-enable by setting non-zero
    # values in ~/.tinyctx/config.toml.
    local_ctx = getattr(cfg.local, "context_window", 0) if hasattr(cfg, "local") else 0
    safe_frac = getattr(cfg.local, "context_safe_fraction", 0.0) or 0.0
    if local_ctx and safe_frac > 0:
        cap = int(local_ctx * safe_frac)
        if est >= cap:
            return Decision(
                "frontier",
                f"est_tokens={est} >= {cap} ({safe_frac:.0%} of local ctx {local_ctx})",
                est_input_tokens=est, turn_count=turns,
            )
    elif cfg.escalate_input_tokens and est >= cfg.escalate_input_tokens:
        return Decision("frontier", f"est_tokens={est} >= {cfg.escalate_input_tokens}",
                        est_input_tokens=est, turn_count=turns)

    if cfg.escalate_turn_count and turns >= cfg.escalate_turn_count:
        return Decision("frontier", f"turn_count={turns} >= {cfg.escalate_turn_count}",
                        est_input_tokens=est, turn_count=turns)

    # Optional: learned classifier. If a trained model exists at
    # ~/.tinyctx/classifier.json, use it as a second opinion. We escalate
    # only if BOTH heuristic-borderline AND the classifier votes frontier
    # — cautious by design while the model is being bootstrapped.
    model = _maybe_load_classifier()
    if model is not None:
        try:
            from .classifier import extract_features, FEATURE_ORDER
            feats = extract_features(body, est_tokens=est, turn_count=turns,
                                     error_streak=error_streak,
                                     is_compaction=compact)
            vec = feats.to_vector()
            prob = model.predict_proba(vec)
            if (prob >= 0.7
                    and getattr(cfg, "self_classify_escalates_to_frontier", False)):
                return Decision(
                    "frontier",
                    f"classifier p(escalate)={prob:.2f}",
                    est_input_tokens=est, turn_count=turns,
                )
            if prob >= 0.7:
                return Decision(
                    "local",
                    f"classifier p(escalate)={prob:.2f} advisor-only",
                    est_input_tokens=est, turn_count=turns,
                )
        except Exception:  # noqa: BLE001 — classifier is advisory, not load-bearing
            # Why: classifier missing or feature-extractor failed
            # (training-only deps). Fall through to default local route.
            pass

    return Decision("local", "small/short -> cheap path",
                    est_input_tokens=est, turn_count=turns)


# ─── P5: consolidated Router ─────────────────────────────────────────────


def _resolve_api_key_from_env(backend, codex_auth: str | None = None) -> str | None:
    """Same precedence as proxy._resolve_api_key (env var → forwarded
    Authorization → ~/.codex/auth.json). Kept here so Router.decide can
    bake Authorization into Decision.headers without a circular import on
    proxy.py. Proxy passes its inbound Authorization (if any) through
    `codex_auth`."""
    if getattr(backend, "api_key_env", None):
        v = os.environ.get(backend.api_key_env)
        if v:
            return v
    if codex_auth:
        return codex_auth
    try:
        with open(os.path.expanduser("~/.codex/auth.json")) as f:
            tok = json.load(f).get("tokens", {}).get("access_token")
        if tok:
            return tok
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        # Why: codex auth.json missing / malformed / wrong shape — all
        # are normal for clients not using ChatGPT subscription auth.
        # Returning None lets the caller fall through to "no api key".
        pass
    return None


class Router:
    """Single source of truth for route + backend + headers.

    Rules run in priority order; first non-None Decision wins. The
    `_default_rule` always returns, so `decide` is total. Each rule
    method is intentionally short (≤30 LOC) — they read from
    `ctx`/`self.cfg` and call the `_make_local_decision` /
    `_make_frontier_decision` builders to fill in target/headers/etc.
    """

    def __init__(self, cfg, *, codex_auth: str | None = None):
        self.cfg = cfg
        # Inbound Authorization header from the codex client, used as a
        # fallback when no api_key_env is set for the chosen backend.
        # Proxy injects this per-request via `with_codex_auth`.
        self._codex_auth = codex_auth

    def with_codex_auth(self, codex_auth: str | None) -> "Router":
        """Return a per-request shallow-copy bound to the inbound
        Authorization. Cheap; called once per request."""
        r = Router.__new__(Router)
        r.cfg = self.cfg
        r._codex_auth = codex_auth
        return r

    # — public —

    def decide(self, ctx: RouteContext) -> Decision:
        for rule in (
            self._compaction_rule,
            self._force_route_rule,
            self._explicit_model_rule,
            self._goal_control_rule,
            self._error_streak_rule,
            self._adaptive_model_rule,
            self._capacity_rule,
            self._classify_rule,
            self._default_rule,
        ):
            d = rule(ctx)
            if d is not None:
                return d
        # _default_rule always returns — unreachable.
        raise RuntimeError("router has no default rule")

    # — backend resolution helpers —

    def _backend_for(self, route: str):
        return self.cfg.local if route == "local" else self.cfg.frontier

    def _target_url(self, backend) -> str:
        base = backend.base_url.rstrip("/")
        if getattr(backend, "wire_api", "responses") == "responses":
            return base + "/responses"
        return base + "/chat/completions"

    def _build_headers(self, backend) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        api_key = _resolve_api_key_from_env(backend, self._codex_auth)
        if api_key:
            h["Authorization"] = (
                api_key if api_key.lower().startswith(("bearer ", "basic "))
                else f"Bearer {api_key}"
            )
        # Per-backend static header overlay (e.g. openai-beta).
        h.update(getattr(backend, "headers", {}) or {})
        return h

    def _make_local_decision(self, ctx: RouteContext, reason: str) -> Decision:
        b = self.cfg.local
        return Decision(
            route="local",
            reason=reason,
            is_compaction=ctx.is_compaction,
            est_input_tokens=ctx.est_tokens,
            turn_count=ctx.turn_count,
            target=self._target_url(b),
            model=b.model or "",
            headers=self._build_headers(b),
            wire_api=getattr(b, "wire_api", "responses"),
            timeout_s=float(getattr(b, "timeout_s", 0.0) or 0.0),
        )

    def _make_frontier_decision(self, ctx: RouteContext, reason: str) -> Decision:
        b = self.cfg.frontier
        return Decision(
            route="frontier",
            reason=reason,
            is_compaction=ctx.is_compaction,
            est_input_tokens=ctx.est_tokens,
            turn_count=ctx.turn_count,
            target=self._target_url(b),
            model=b.model or "",
            headers=self._build_headers(b),
            wire_api=getattr(b, "wire_api", "responses"),
            timeout_s=float(getattr(b, "timeout_s", 0.0) or 0.0),
        )

    # — rules (priority order) —

    def _compaction_rule(self, ctx: RouteContext) -> Decision | None:
        """1. Compaction handoff → local (highest priority).

        Codex emits a deterministic handoff-summary prompt during local
        compaction; redirect it to the cheap path regardless of any other
        signal. Beats force_route so an empty-response flag set on a
        previous turn doesn't waste a frontier call on a summary."""
        if ctx.is_compaction and getattr(self.cfg, "redirect_compaction_to_local", True):
            return self._make_local_decision(ctx, "compaction handoff -> cheap path")
        return None

    def _force_route_rule(self, ctx: RouteContext) -> Decision | None:
        """2. Force-route override → honor the pin.

        Two sources, in order of precedence:
          a) `ctx.force_route` — per-request, set by ForceFrontierGuard
             when the previous turn produced an empty response, hit a
             stall, or got an upstream error.
          b) `cfg.force_route` — admin-level config pin
             (`TINYCTX_FORCE_ROUTE` env / `force_route` in config.toml)
             that's typically "auto" (no pin) but operators use
             "frontier" / "local" to force-pin a session for debugging.

        Either source beats explicit `tinyctx-local` / `tinyctx-frontier`
        requests so the recovery / admin routing wins over a client
        preference that's either known to fail (per-request) or
        intentionally being overridden by config (admin)."""
        # Per-request takes precedence over CFG pin (recovery > admin).
        pinned = ctx.force_route or getattr(self.cfg, "force_route", "auto")
        if pinned == "frontier":
            src = "guard pipeline" if ctx.force_route else "cfg"
            return self._make_frontier_decision(
                ctx, f"{src} force_route=frontier")
        if pinned == "local":
            src = "guard pipeline" if ctx.force_route else "cfg"
            return self._make_local_decision(
                ctx, f"{src} force_route=local")
        return None

    def _explicit_model_rule(self, ctx: RouteContext) -> Decision | None:
        """3. Client-requested model id → honor verbatim.

        `model=tinyctx-local` / `tinyctx-frontier` is the in-band route
        override codex.app / advisor agents use. Beats capacity and
        classify so the client's explicit decision is respected even on
        a small request."""
        m = (ctx.requested_model or "").lower()
        if m == "tinyctx-local":
            return self._make_local_decision(
                ctx, "client requested tinyctx-local")
        if m == "tinyctx-frontier":
            return self._make_frontier_decision(
                ctx, "client requested tinyctx-frontier")
        return None

    def _error_streak_rule(self, ctx: RouteContext) -> Decision | None:
        """5. Repeated tool-failure streak → frontier.

        Anthropic "when stuck, escalate" — N consecutive failures on
        local means cheap model isn't making progress, so try the
        stronger model."""
        thr = getattr(self.cfg, "escalate_on_error_streak", 0) or 0
        if thr and ctx.error_streak >= thr:
            return self._make_frontier_decision(
                ctx, f"error_streak={ctx.error_streak} >= {thr}")
        return None

    def _adaptive_model_rule(self, ctx: RouteContext) -> Decision | None:
        """6. Rolling local backend failure rate → frontier.

        Ported from SmallCode's adaptive model select, narrowed to tinyctx's
        local/frontier topology. It only applies after explicit route pins
        and per-request recovery flags, so user intent remains dominant."""
        if not getattr(self.cfg, "adaptive_model_enabled", True):
            return None
        try:
            from . import adaptive_model
            health = adaptive_model.local_health(self.cfg)
        except Exception:
            return None
        if not health.should_escalate:
            return None
        return self._make_frontier_decision(
            ctx,
            "adaptive local failure rate "
            f"{health.failures}/{health.calls}={health.failure_rate:.0%} "
            ">= "
            f"{getattr(self.cfg, 'adaptive_model_failure_rate_threshold', 0.3):.0%}",
        )

    def _goal_control_rule(self, ctx: RouteContext) -> Decision | None:
        """4. `/goal` control-plane turns → frontier.

        tinyctx's best goal-mode shape keeps ordinary execution on the
        cheap model, but routes high-leverage control decisions to the
        strongest model: creating the goal contract, validating
        `done_when`, judging completion, and surfacing blockers/pivots.
        Detection is tail-user-only so a persistent GOAL.md in history
        does not make every iteration expensive.
        """
        if not getattr(self.cfg, "goal_control_frontier_enabled", True):
            return None
        signal = goal_control_signal(ctx.body)
        if not signal:
            return None
        return self._make_frontier_decision(
            ctx, f"goal-control turn -> frontier ({signal})")

    def _capacity_rule(self, ctx: RouteContext) -> Decision | None:
        """7. Local context capacity escalation → frontier.

        Belt-and-suspenders for small-context local backends (LMStudio
        32k, Ollama default). Disabled when context_safe_fraction is 0
        (the Advisor-Strategy-aligned default). Also honors the legacy
        absolute `escalate_input_tokens` threshold when no context_window
        is set on the local backend."""
        local = getattr(self.cfg, "local", None)
        if local is None:
            return None
        cw = getattr(local, "context_window", 0) or 0
        sf = getattr(local, "context_safe_fraction", 0.0) or 0.0
        if cw and sf > 0:
            cap = int(cw * sf)
            if ctx.est_tokens >= cap:
                return self._make_frontier_decision(
                    ctx,
                    f"est_tokens={ctx.est_tokens} >= {cap} "
                    f"({sf:.0%} of local ctx {cw})")
            return None
        legacy = getattr(self.cfg, "escalate_input_tokens", 0) or 0
        if legacy and ctx.est_tokens >= legacy:
            return self._make_frontier_decision(
                ctx, f"est_tokens={ctx.est_tokens} >= {legacy}")
        return None

    def _classify_rule(self, ctx: RouteContext) -> Decision | None:
        """8. Self-classify advisor recommendation.

        The local model itself classified this turn as needing the
        advisor. Threshold defaults to 0.7 and is configurable. Default
        behavior keeps the executor local and records the recommendation;
        legacy full-turn frontier routing is opt-in."""
        thr = getattr(self.cfg, "self_classify_threshold", 1.1) or 1.1
        if ctx.classify_p >= thr:
            tail = (f": {ctx.classify_reason}" if ctx.classify_reason else "")
            if not getattr(self.cfg, "self_classify_escalates_to_frontier", False):
                return self._make_local_decision(
                    ctx, f"self-classify p={ctx.classify_p:.2f} advisor-only{tail}")
            return self._make_frontier_decision(
                ctx, f"self-classify p={ctx.classify_p:.2f}{tail}")
        return None

    def _default_rule(self, ctx: RouteContext) -> Decision:
        """9. Default → local (cheap path).

        Small/short request with no escalation signal; this is the win
        path that justifies the proxy's existence."""
        return self._make_local_decision(ctx, "small/short -> cheap path")
