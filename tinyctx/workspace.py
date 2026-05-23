"""Filesystem contract for tinyctx self-improvement state."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: Any, default: str = "default") -> str:
    text = str(value or "").strip()
    text = _SAFE_ID_RE.sub("-", text).strip(".-")
    return text[:120] or default


def default_root() -> Path:
    override = os.environ.get("TINYCTX_WORKSPACE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tinyctx"


@dataclass(frozen=True)
class SessionWorkspace:
    root: Path
    session_id: str
    base: Path
    public: Path
    private: Path
    logs: Path
    evals: Path
    candidates: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "session_id": self.session_id,
            "base": str(self.base),
            "public": str(self.public),
            "private": str(self.private),
            "logs": str(self.logs),
            "evals": str(self.evals),
            "candidates": str(self.candidates),
        }


def ensure_workspace(session_id: Any, root: Optional[Path] = None) -> SessionWorkspace:
    root = Path(root) if root is not None else default_root()
    sid = safe_id(session_id)
    base = root / "sessions" / sid
    workspace = SessionWorkspace(
        root=root,
        session_id=sid,
        base=base,
        public=base / "public",
        private=base / "private",
        logs=base / "private" / "logs",
        evals=base / "private" / "evals",
        candidates=base / "private" / "candidates",
    )
    for path in (
        workspace.public,
        workspace.private,
        workspace.logs,
        workspace.evals,
        workspace.candidates,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return workspace


def context_profile_path(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else default_root()
    return base / "context_profile.json"


def default_context_profile() -> dict[str, Any]:
    return {
        "version": 1,
        "commands": [],
        "sensitive_paths": [],
        "mcps": {},
        "skills": {},
        "graphify": {},
        "sanitizer": {},
        "budgets": {},
    }


def load_context_profile(root: Optional[Path] = None) -> dict[str, Any]:
    path = context_profile_path(root)
    profile = default_context_profile()
    if not path.exists():
        return profile
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return profile
    if isinstance(loaded, dict):
        profile.update(loaded)
    return profile


def save_context_profile(
    profile: Mapping[str, Any],
    root: Optional[Path] = None,
) -> Path:
    path = context_profile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = default_context_profile()
    data.update(dict(profile))
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def list_session_ids(root: Optional[Path] = None) -> list[str]:
    base = Path(root) if root is not None else default_root()
    sessions = base / "sessions"
    if not sessions.exists():
        return []
    return sorted(
        path.name for path in sessions.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
