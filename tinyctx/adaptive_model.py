"""Rolling backend health used for SmallCode-style adaptive routing.

SmallCode's adaptive router tracks model failures and switches away from
the primary model when the observed failure rate gets high.  tinyctx has
only two execution routes, so the portable shape is narrower: watch the
local backend's recent outcomes and route future automatic turns to the
frontier while the local failure rate is unhealthy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdaptiveHealth:
    key: str
    calls: int = 0
    failures: int = 0
    failure_rate: float = 0.0
    should_escalate: bool = False


@dataclass
class AdaptiveRouteState:
    _samples: dict[str, deque[bool]] = field(default_factory=dict)

    def record(self, key: str, *, ok: bool, max_samples: int = 20) -> AdaptiveHealth:
        max_samples = max(1, int(max_samples or 1))
        window = self._samples.get(key)
        if window is None or window.maxlen != max_samples:
            previous = list(window or [])
            window = deque(previous[-max_samples:], maxlen=max_samples)
            self._samples[key] = window
        window.append(bool(ok))
        return self.health(key)

    def health(self, key: str, *, min_calls: int = 0,
               threshold: float = 1.1) -> AdaptiveHealth:
        window = self._samples.get(key) or ()
        calls = len(window)
        failures = sum(1 for ok in window if not ok)
        rate = failures / calls if calls else 0.0
        return AdaptiveHealth(
            key=key,
            calls=calls,
            failures=failures,
            failure_rate=rate,
            should_escalate=(
                calls >= max(0, int(min_calls or 0))
                and calls > 0
                and rate >= float(threshold)
            ),
        )

    def reset(self) -> None:
        self._samples.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "calls": h.calls,
                "failures": h.failures,
                "failure_rate": h.failure_rate,
            }
            for key in sorted(self._samples)
            for h in (self.health(key),)
        }


STATE = AdaptiveRouteState()


def backend_key(route: str, backend: Any) -> str:
    base = (getattr(backend, "base_url", "") or "").rstrip("/")
    model = getattr(backend, "model", "") or ""
    wire = getattr(backend, "wire_api", "") or ""
    if base:
        base = base + ("/responses" if wire == "responses" else "/chat/completions")
    return f"{route}:{wire}:{base}:{model}"


def decision_key(decision: Any, *, scope: str = "") -> str:
    route = getattr(decision, "route", "") or ""
    model = getattr(decision, "model", "") or ""
    wire = getattr(decision, "wire_api", "") or ""
    target = (getattr(decision, "target", "") or "").rstrip("/")
    base = f"{route}:{wire}:{target}:{model}"
    return f"{scope}:{base}" if scope else base


def record_decision(decision: Any, *, ok: bool, max_samples: int = 20,
                    scope: str = "",
                    ) -> AdaptiveHealth:
    return STATE.record(decision_key(decision, scope=scope),
                        ok=ok, max_samples=max_samples)


def local_health(cfg: Any, *, scope: str = "") -> AdaptiveHealth:
    key = backend_key("local", getattr(cfg, "local", None))
    if scope:
        key = f"{scope}:{key}"
    return STATE.health(
        key,
        min_calls=getattr(cfg, "adaptive_model_min_calls", 3),
        threshold=getattr(cfg, "adaptive_model_failure_rate_threshold", 0.3),
    )


def reset_state() -> None:
    STATE.reset()
