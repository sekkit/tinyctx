"""Routing decisions and Responses-API request analysis.

Three responsibilities:
  1. Detect codex's compaction "handoff summary" prompt.
  2. Estimate input token count cheaply (4 chars ≈ 1 token).
  3. Decide local vs frontier based on heuristics, optionally augmented
     by a learned classifier (FrugalGPT-style scorer; see classifier.py).

We do NOT trust verbalized confidence from the model. We trust observable
properties of the request (length, turn count, prompt fingerprint, error
streak via session state).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass
class Decision:
    route: str          # "local" | "frontier"
    reason: str         # human-readable
    is_compaction: bool = False
    est_input_tokens: int = 0
    turn_count: int = 0


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

    if error_streak >= cfg.escalate_on_error_streak:
        return Decision("frontier", f"error_streak={error_streak} >= threshold",
                        est_input_tokens=est, turn_count=turns)

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
            if prob >= 0.7:  # high-confidence escalate
                return Decision(
                    "frontier",
                    f"classifier p(escalate)={prob:.2f}",
                    est_input_tokens=est, turn_count=turns,
                )
        except Exception:
            pass

    return Decision("local", "small/short -> cheap path",
                    est_input_tokens=est, turn_count=turns)
