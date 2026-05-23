"""Replay/evaluation primitives for tinyctx policy candidates."""

from __future__ import annotations

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


def aggregate_results(results: Sequence[EvalResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "score": 0.0}
    passed = sum(1 for result in results if result.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "score": sum(result.score for result in results) / total,
    }
