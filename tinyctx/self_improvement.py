"""Governed candidate evaluation loop for tinyctx policies."""

from __future__ import annotations

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
