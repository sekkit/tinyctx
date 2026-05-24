"""Governed candidate evaluation loop for tinyctx policies."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import eval_harness, frontier, trajectory


def evaluate_candidate(
    session_id: Any,
    candidate: frontier.Candidate,
    cases: Sequence[eval_harness.EvalCase],
    evaluator: eval_harness.Evaluator,
    *,
    root: Optional[Path] = None,
    weights: Optional[Mapping[str, float]] = None,
    min_pass_rate: float = 1.0,
    max_cases: Optional[int] = None,
) -> dict[str, Any]:
    """Run a bounded eval and archive the candidate with merged metrics."""
    weights = dict(weights or {"score": 1.0, "pass_rate": 1.0})
    trajectory.record_event(
        session_id,
        "candidate_eval_started",
        root=root,
        phase="eval",
        artifacts={
            "candidate_id": candidate.candidate_id,
            "kind": candidate.kind,
            "case_count": len(cases),
        },
    )
    results = eval_harness.run_suite(cases, evaluator, max_cases=max_cases)
    aggregate = eval_harness.aggregate_results(results)
    metrics = dict(candidate.metrics)
    metrics.update({
        "score": float(aggregate["score"]),
        "pass_rate": float(aggregate["pass_rate"]),
        "passed": float(aggregate["passed"]),
        "failed": float(aggregate["failed"]),
    })
    archived = frontier.add_candidate(
        session_id,
        frontier.Candidate(
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            payload=candidate.payload,
            metrics=metrics,
            artifacts={
                **dict(candidate.artifacts),
                "eval_results": [
                    {
                        "case_id": result.case_id,
                        "passed": result.passed,
                        "score": result.score,
                        "error": result.error,
                    }
                    for result in results
                ],
            },
            parent_id=candidate.parent_id,
            generation=candidate.generation,
            created_ts=candidate.created_ts,
        ),
        root=root,
    )
    candidates = frontier.read_candidates(
        session_id,
        root=root,
        kind=candidate.kind,
    )
    best = frontier.best_candidate(candidates, weights)
    accepted = (
        bool(best)
        and best.get("candidate_id") == archived["candidate_id"]
        and aggregate["pass_rate"] >= min_pass_rate
    )
    trajectory.record_event(
        session_id,
        "candidate_eval_completed",
        root=root,
        phase="eval",
        metrics={
            "score": aggregate["score"],
            "pass_rate": aggregate["pass_rate"],
            "accepted": accepted,
        },
        artifacts={
            "candidate_id": candidate.candidate_id,
            "kind": candidate.kind,
            "best_candidate_id": best.get("candidate_id") if best else None,
        },
    )
    return {
        "candidate": archived,
        "aggregate": aggregate,
        "best": best,
        "accepted": accepted,
    }


# ─── post-stream performance tracking ───────────────────────────────────

_NS = "self_improvement"


def record_request(
    proj_sid: str,
    conv_sid: str,
    *,
    route: str,
    status: int,
    elapsed_s: float,
    bytes_out: int,
    upstream_failed: bool = False,
    cfg: "Any | None" = None,
) -> None:
    """Record a completed request's metrics for self-improvement analysis.

    Called from PostStreamAnalyzer.analyze() after every stream completes.
    Appends to the per-project rolling metrics window and triggers periodic
    evaluation when the request count crosses the eval_interval boundary.
    """
    enabled = getattr(cfg, "self_improvement_enabled", False) if cfg else False
    if not enabled:
        return
    stats_window = getattr(cfg, "self_improvement_stats_window", 50) if cfg else 50
    eval_interval = getattr(cfg, "self_improvement_eval_interval", 50) if cfg else 50

    try:
        trajectory.record_event(
            proj_sid,
            "request_completed",
            phase="post_stream",
            metrics={
                "route": route,
                "status": status,
                "elapsed_s": round(elapsed_s, 3),
                "bytes_out": bytes_out,
                "upstream_failed": int(upstream_failed),
            },
        )
    except Exception:
        pass

    from . import session_state
    session_state.append_bounded(
        proj_sid, _NS, "recent_metrics",
        {
            "route": route,
            "status": status,
            "elapsed_s": round(elapsed_s, 3),
            "bytes_out": bytes_out,
            "upstream_failed": int(upstream_failed),
            "ts": time.time(),
        },
        maxlen=stats_window,
    )

    count = session_state.increment(proj_sid, _NS, "request_count")
    if count % eval_interval == 0:
        _maybe_evaluate(proj_sid, conv_sid, cfg=cfg)


def _maybe_evaluate(
    proj_sid: str,
    conv_sid: str,
    *,
    cfg: "Any | None" = None,
) -> "dict[str, Any] | None":
    """Aggregate recent request metrics and evaluate against historical baselines.

    The first evaluation establishes a baseline (archived as candidate).
    Subsequent evaluations compare current metrics against the best previous
    candidate. If the current window fails the eval cases, a degradation flag
    is set for the SelfImprovementGuard to consume on the next request.
    """
    from . import session_state
    recent = session_state.get_history(proj_sid, _NS, "recent_metrics")
    if len(recent) < 5:
        return None

    total = len(recent)
    errors = sum(
        1 for r in recent
        if r.get("status", 200) >= 400 or r.get("status", 200) == 0 or r.get("upstream_failed", 0)
    )
    current_error_rate = errors / total if total > 0 else 0.0
    current_avg_latency = sum(r.get("elapsed_s", 0.0) for r in recent) / total

    route_counts: dict[str, int] = {}
    for r in recent:
        rte = r.get("route", "unknown")
        route_counts[rte] = route_counts.get(rte, 0) + 1

    from .frontier import read_candidates, best_candidate
    baselines = read_candidates(proj_sid, kind="performance")
    is_first_eval = len(baselines) == 0

    best = best_candidate(baselines, {"error_rate": -10.0, "avg_latency_s": -1.0}) if baselines else None
    baseline_error_rate = (best.get("metrics", {}).get("error_rate", current_error_rate)
                           if best else current_error_rate)
    baseline_latency = (best.get("metrics", {}).get("avg_latency_s", current_avg_latency)
                       if best else current_avg_latency)

    from .eval_harness import EvalCase
    cases = [
        EvalCase(
            case_id="error_rate",
            input={"observed_error_rate": current_error_rate},
            expected={"max_error_rate": max(baseline_error_rate * 2.0, 0.10)},
        ),
        EvalCase(
            case_id="latency",
            input={"observed_avg_latency_s": current_avg_latency},
            expected={"max_latency_s": max(baseline_latency * 1.5, baseline_latency + 30.0)},
        ),
    ]

    from .frontier import Candidate
    candidate = Candidate(
        candidate_id=f"perf-{int(time.time())}",
        kind="performance",
        payload={
            "window_size": total,
            "eval_count": len(baselines) + 1,
            "route_distribution": route_counts,
        },
        metrics={
            "error_rate": round(current_error_rate, 4),
            "avg_latency_s": round(current_avg_latency, 3),
        },
    )

    result = evaluate_candidate(
        proj_sid,
        candidate,
        cases,
        _perf_evaluator,
        min_pass_rate=1.0,
    )

    if not is_first_eval and not result.get("accepted", False):
        failed = []
        for r in result.get("candidate", {}).get("artifacts", {}).get("eval_results", []):
            if not r.get("passed"):
                failed.append(r.get("case_id"))
        session_state.set(proj_sid, _NS, "degraded", {
            "at": time.time(),
            "reasons": failed,
            "error_rate": current_error_rate,
            "avg_latency_s": current_avg_latency,
        })
    elif not is_first_eval:
        session_state.clear(proj_sid, _NS, "degraded")

    return result


def _perf_evaluator(case: "EvalCase") -> dict[str, Any]:
    """Evaluate a single performance case against its expected threshold."""
    if case.case_id == "error_rate":
        observed = float(case.input.get("observed_error_rate", 0.0))
        max_allowed = float(case.expected.get("max_error_rate", float("inf")))
        passed = observed <= max_allowed
        score = 1.0 - (observed / max_allowed) if max_allowed > 0 else 1.0
        return {
            "passed": passed,
            "score": max(0.0, min(1.0, score)),
            "metrics": {"observed": observed, "max_allowed": max_allowed},
        }
    if case.case_id == "latency":
        observed = float(case.input.get("observed_avg_latency_s", 0.0))
        max_allowed = float(case.expected.get("max_latency_s", float("inf")))
        passed = observed <= max_allowed
        score = 1.0 - (observed / max_allowed) if max_allowed > 0 else 1.0
        return {
            "passed": passed,
            "score": max(0.0, min(1.0, score)),
            "metrics": {"observed_s": observed, "max_allowed_s": max_allowed},
        }
    return {"passed": True, "score": 1.0}


def consume_degradation_flag(proj_sid: str) -> "dict[str, Any] | None":
    """Read-and-clear the per-project degradation flag.

    Called by SelfImprovementGuard during the pre-flight pipeline.
    Returns the degradation info dict if a flag was set, or None.
    """
    from . import session_state
    return session_state.consume(proj_sid, _NS, "degraded")
