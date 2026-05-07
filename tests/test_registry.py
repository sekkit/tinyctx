"""Tests for tinyctx.registry."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import registry


def test_register_creates_file_and_dedup():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "repo"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            assert registry.register(proj) is True   # newly added
            assert registry.register(proj) is False  # duplicate
            data = json.loads((td_p / ".tinyctx" / "projects.json").read_text())
            assert str(proj.resolve()) in data["projects"]
            assert len(data["projects"]) == 1


def test_unregister_removes_existing():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "repo"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            registry.register(proj)
            assert registry.unregister(proj) is True
            assert registry.unregister(proj) is False
            assert registry.all_projects() == []


def test_all_projects_filters_missing_directories():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        present = td_p / "alive"; present.mkdir()
        gone = td_p / "deleted"; gone.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            registry.register(present)
            registry.register(gone)
            gone.rmdir()  # simulate user deleting a tracked repo
            paths = registry.all_projects()
            assert present.resolve() in [p.resolve() for p in paths]
            assert gone.resolve() not in [p.resolve() for p in paths]


def test_is_registered_handles_corrupt_file():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "repo"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            f = td_p / ".tinyctx" / "projects.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("{ not json")
            # Corrupt file should not crash; just behave as empty.
            assert registry.is_registered(proj) is False
            assert registry.all_projects() == []


def test_register_sorts_for_determinism():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        b = td_p / "b"; b.mkdir()
        a = td_p / "a"; a.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p)}):
            registry.register(b)
            registry.register(a)
            data = json.loads((td_p / ".tinyctx" / "projects.json").read_text())
            assert data["projects"] == sorted(data["projects"])


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
