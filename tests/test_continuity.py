"""Tests for tinyctx.continuity: persist + recall."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import continuity


def test_save_creates_per_session_file():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            p1 = continuity.save_compaction(proj, "sess-A", "first summary")
            p2 = continuity.save_compaction(proj, "sess-A", "second summary")
            assert p1.name == "compaction-1.md"
            assert p2.name == "compaction-2.md"
            assert "first summary" in p1.read_text()
            assert "second summary" in p2.read_text()


def test_save_writes_telemetry_in_header():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            p = continuity.save_compaction(
                proj, "sess-A", "summary",
                telemetry={"outcome": "judged", "timings": {"total_s": 12.4}},
            )
            text = p.read_text()
            assert "telemetry" in text
            assert "judged" in text


def test_save_updates_latest_pointer():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(proj, "sess-A", "v1")
            time.sleep(0.01)
            p2 = continuity.save_compaction(proj, "sess-B", "v2")
            latest = continuity.sessions_dir(proj) / "latest.md"
            assert latest.exists()
            # symlink or copy — both must point at v2's content
            assert "v2" in latest.read_text()


def test_recall_returns_latest_session_only_by_default():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(proj, "sess-A", "alpha")
            time.sleep(0.05)
            continuity.save_compaction(proj, "sess-B", "bravo")
            paths = continuity.recall(proj, all_sessions=False, limit=5)
            # only sess-B's compactions returned
            assert len(paths) == 1
            assert "bravo" in paths[0].read_text()


def test_recall_all_sessions():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(proj, "sess-A", "alpha")
            time.sleep(0.05)
            continuity.save_compaction(proj, "sess-B", "bravo")
            paths = continuity.recall(proj, all_sessions=True, limit=5)
            assert len(paths) == 2
            texts = "".join(p.read_text() for p in paths)
            assert "alpha" in texts and "bravo" in texts


def test_list_sessions_orders_by_recency():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(proj, "old", "x")
            time.sleep(0.05)
            continuity.save_compaction(proj, "new", "y")
            sessions = continuity.list_sessions(proj)
            assert [s[0] for s in sessions] == ["new", "old"]
            assert all(c == 1 for _, c, _ in sessions)


def test_save_with_structured_writes_json_sidecar():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        structured = {
            "compartments": [{"name": "auth-setup", "topic": "JWT setup",
                              "summary": "We added JWT.", "files": ["src/auth.py"]}],
            "facts": [{"claim": "secret in .env", "evidence": "user said"}],
            "open_questions": ["test coverage?"],
        }
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            md_path = continuity.save_compaction(
                proj, "sess-X", "## summary\nbody",
                telemetry={"outcome": "judged"},
                structured=structured,
            )
            json_path = md_path.with_suffix(".json")
            assert json_path.is_file()
            data = json.loads(json_path.read_text())
            assert data["facts"][0]["claim"] == "secret in .env"
            assert data["compartments"][0]["name"] == "auth-setup"


def test_latest_structured_returns_dict():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            assert continuity.latest_structured(proj) is None
            continuity.save_compaction(proj, "sess-A", "body",
                                       structured={"facts": [{"claim": "x"}]})
            data = continuity.latest_structured(proj)
            assert data is not None
            assert data["facts"][0]["claim"] == "x"


def test_recall_facts_only_via_cli():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(
                proj, "sess-A", "body",
                structured={"facts": [
                    {"claim": "foo is bar", "evidence": "user turn 3"},
                    {"claim": "bar is baz"},
                ], "compartments": [], "open_questions": []},
            )
            # Capture stdout
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = continuity.main([
                    "--root", str(proj), "--facts-only",
                ])
            out = buf.getvalue()
            assert rc == 0
            assert "foo is bar" in out
            assert "bar is baz" in out


def test_recall_compartment_via_cli():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            continuity.save_compaction(
                proj, "sess-A", "body",
                structured={
                    "compartments": [
                        {"name": "auth", "topic": "JWT",
                         "summary": "did the auth bit", "files": ["a.py"]},
                        {"name": "db", "topic": "schema",
                         "summary": "did the db bit", "files": ["b.py"]},
                    ],
                    "facts": [], "open_questions": [],
                },
            )
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = continuity.main([
                    "--root", str(proj), "--compartment", "auth",
                ])
            out = buf.getvalue()
            assert rc == 0
            assert "did the auth bit" in out
            assert "did the db bit" not in out


def test_recall_returns_empty_for_unknown_repo():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            assert continuity.recall(proj) == []
            assert continuity.list_sessions(proj) == []


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
