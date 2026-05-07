"""Tiny per-machine registry of projects tinyctx has touched.

Used by `tinyctx-dreamer run` to know which repos to refresh. Auto-populated
on first `tinyctx-scout init` / `tinyctx-keypin scan` for a project.

Implementation is one JSON file at ~/.tinyctx/projects.json:

    {"projects": ["/abs/path/repo1", "/abs/path/repo2", ...]}

We deliberately keep this tiny and sync — no SQLite, no concurrent writers.
The file is rewritten in-place; if a concurrent dreamer is running we just
race on a small file (worst case: a dropped registration that the user can
re-add).
"""
from __future__ import annotations

import json
from pathlib import Path


def _file() -> Path:
    return Path.home() / ".tinyctx" / "projects.json"


def _load() -> dict:
    p = _file()
    if not p.is_file():
        return {"projects": []}
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "projects" not in data:
            return {"projects": []}
        return data
    except (OSError, json.JSONDecodeError):
        return {"projects": []}


def _save(data: dict) -> None:
    p = _file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def register(project_root: Path) -> bool:
    """Add `project_root` (resolved absolute path) to the registry. Returns
    True if newly added, False if already present."""
    abs_path = str(Path(project_root).resolve())
    data = _load()
    if abs_path in data["projects"]:
        return False
    data["projects"].append(abs_path)
    data["projects"].sort()
    _save(data)
    return True


def unregister(project_root: Path) -> bool:
    abs_path = str(Path(project_root).resolve())
    data = _load()
    if abs_path not in data["projects"]:
        return False
    data["projects"] = [x for x in data["projects"] if x != abs_path]
    _save(data)
    return True


def all_projects() -> list[Path]:
    """Return every registered path that still exists on disk."""
    data = _load()
    return [Path(p) for p in data["projects"] if Path(p).is_dir()]


def is_registered(project_root: Path) -> bool:
    abs_path = str(Path(project_root).resolve())
    return abs_path in _load()["projects"]
