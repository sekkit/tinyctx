"""Append-only session trajectory ledger."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .workspace import SessionWorkspace, ensure_workspace


@dataclass(frozen=True)
class TrajectoryEvent:
    event: str
    session_id: str
    phase: str = "unknown"
    ts: float = field(default_factory=time.time)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=tuple)
    parent_id: Optional[str] = None
    event_id: Optional[str] = None

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_id": self.parent_id,
            "ts": self.ts,
            "session_id": self.session_id,
            "event": self.event,
            "phase": self.phase,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "tags": list(self.tags),
        }


def ledger_path(workspace: SessionWorkspace) -> Path:
    return workspace.logs / "trajectory.jsonl"


def record_event(
    session_id: Any,
    event: str,
    *,
    root: Optional[Path] = None,
    phase: str = "unknown",
    metrics: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    tags: Sequence[str] = (),
    parent_id: Optional[str] = None,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    workspace = ensure_workspace(session_id, root=root)
    entry = TrajectoryEvent(
        event=event,
        session_id=workspace.session_id,
        phase=phase,
        metrics=metrics or {},
        artifacts=artifacts or {},
        tags=tuple(tags),
        parent_id=parent_id,
        event_id=event_id,
    ).to_record()
    path = ledger_path(workspace)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def read_events(
    session_id: Any,
    *,
    root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    workspace = ensure_workspace(session_id, root=root)
    path = ledger_path(workspace)
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
    if limit is not None and limit >= 0:
        return records[-limit:]
    return records


def summarize_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, int] = {}
    by_event: dict[str, int] = {}
    failures = 0
    for item in events:
        phase = str(item.get("phase") or "unknown")
        event = str(item.get("event") or "unknown")
        by_phase[phase] = by_phase.get(phase, 0) + 1
        by_event[event] = by_event.get(event, 0) + 1
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        if metrics.get("passed") is False or metrics.get("error"):
            failures += 1
    return {
        "total": len(events),
        "failures": failures,
        "by_phase": by_phase,
        "by_event": by_event,
    }
