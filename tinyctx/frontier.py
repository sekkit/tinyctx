"""Versioned candidate frontier for tinyctx self-improvement experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .workspace import SessionWorkspace, ensure_workspace, safe_id


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    payload: Mapping[str, Any]
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    generation: int = 0
    created_ts: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": safe_id(self.candidate_id, "candidate"),
            "kind": self.kind,
            "payload": dict(self.payload),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "parent_id": self.parent_id,
            "generation": self.generation,
            "created_ts": self.created_ts,
        }


def archive_path(workspace: SessionWorkspace, kind: str = "context") -> Path:
    return workspace.candidates / f"{safe_id(kind)}.jsonl"


def score(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(float(metrics.get(name, 0.0)) * float(weight) for name, weight in weights.items())


def add_candidate(
    session_id: Any,
    candidate: Candidate,
    *,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    workspace = ensure_workspace(session_id, root=root)
    record = candidate.to_record()
    path = archive_path(workspace, candidate.kind)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def read_candidates(
    session_id: Any,
    *,
    root: Optional[Path] = None,
    kind: str = "context",
) -> list[dict[str, Any]]:
    workspace = ensure_workspace(session_id, root=root)
    path = archive_path(workspace, kind)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    best_score: Optional[float] = None
    for item in candidates:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
        current = score(metrics, weights)
        if best is None or best_score is None or current > best_score:
            best = dict(item)
            best["weighted_score"] = current
            best_score = current
    return best
