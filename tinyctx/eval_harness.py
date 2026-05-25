"""Replay/evaluation primitives for tinyctx policy candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


MetricMap = Mapping[str, float]
Evaluator = Callable[["EvalCase"], Mapping[str, Any]]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    input: Mapping[str, Any]
    expected: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _coerce_result(case_id: str, raw: Mapping[str, Any]) -> EvalResult:
    metrics = dict(raw.get("metrics") or {})
    if "score" in raw:
        score = float(raw["score"])
    elif "score" in metrics:
        score = float(metrics["score"])
    else:
        score = 1.0 if raw.get("passed") else 0.0
    passed = bool(raw.get("passed", score >= 1.0))
    error = raw.get("error")
    return EvalResult(
        case_id=case_id,
        passed=passed,
        score=score,
        metrics=metrics,
        error=str(error) if error else None,
    )


def run_suite(
    cases: Sequence[EvalCase],
    evaluator: Evaluator,
    *,
    max_cases: Optional[int] = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    selected = list(cases[:max_cases]) if max_cases is not None else list(cases)
    for case in selected:
        try:
            raw = evaluator(case)
            if not isinstance(raw, Mapping):
                raw = {"passed": bool(raw), "score": 1.0 if raw else 0.0}
            results.append(_coerce_result(case.case_id, raw))
        except Exception as exc:  # noqa: BLE001 - eval failures are data
            results.append(
                EvalResult(
                    case_id=case.case_id,
                    passed=False,
                    score=0.0,
                    error=str(exc),
                )
            )
    return results


def march_of_nines(score: float, optimal: float = 1.0) -> float:
    """Transform a score with the march-of-9s function.

    φ(s) = -log10(|s - s_opt| + ε)

    As scores approach optimal, tiny improvements are amplified in log space,
    forcing agents to pursue extreme optimization rather than settling for
    "good enough".  The ε = 1e-10 prevents log10(0).
    """
    return -math.log10(abs(score - optimal) + 1e-10)


def normalized_score(
    score: float,
    sota: float,
    worst: float,
    optimal: float = 1.0,
) -> float:
    """Compute AIRA-style Normalized Score using march-of-9s.

    NS = (φ(score) - φ(worst)) / (φ(sota) - φ(worst))

    Returns a value in [0, 1] where 0 = worst observed, 1 = SOTA.
    Can exceed 1.0 if score surpasses the SOTA reference.
    """
    phi_score = march_of_nines(score, optimal)
    phi_worst = march_of_nines(worst, optimal)
    phi_sota = march_of_nines(sota, optimal)
    denom = phi_sota - phi_worst
    if denom == 0:
        return 1.0 if score >= sota else 0.0
    return (phi_score - phi_worst) / denom


def aggregate_results(
    results: Sequence[EvalResult],
    *,
    sota_score: Optional[float] = None,
    worst_score: Optional[float] = None,
    optimal_score: float = 1.0,
) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        base: dict[str, Any] = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "score": 0.0}
        if sota_score is not None and worst_score is not None:
            base["normalized_score"] = 0.0
        return base
    passed = sum(1 for result in results if result.passed)
    mean_score = sum(result.score for result in results) / total
    out: dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "score": mean_score,
    }
    if sota_score is not None and worst_score is not None:
        out["normalized_score"] = normalized_score(
            mean_score, sota_score, worst_score, optimal_score,
        )
    return out
