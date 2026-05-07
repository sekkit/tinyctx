"""Tests for tinyctx.scout. We don't hit a real LLM endpoint — we inject a
fake `_llm_call` so the build path is fully exercised offline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import scout


def _fake_llm(system_prompt, user_prompt, **kw):
    # Return a deterministic summary referencing what we got.
    return f"# Scout summary\nmodel={kw.get('model')}\nsymbols seen: {user_prompt.count('===')}"


def _write_graph(td: Path, files: dict[str, str]) -> tuple[Path, Path]:
    """Create a project tree and a tinyctx-shaped graph.json over it."""
    proj = td / "project"
    proj.mkdir()
    nodes = []
    for rel, content in files.items():
        p = proj / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        nodes.append({
            "id": rel,
            "wrapped_signature": 4,
            "wrapped_body": max(1, len(content) // 10),
            "deps": [],
        })
    # add a primitive each one depends on, so j0 isn't degenerate
    nodes.append({"id": "primitive", "wrapped_signature": 1, "wrapped_body": 0, "deps": []})
    for n in nodes[:-1]:
        n["deps"] = ["primitive"] * 3
    graph = td / "graph.json"
    graph.write_text(json.dumps({"nodes": nodes}))
    return proj, graph


def test_repo_hash_stable_across_calls():
    p = Path("/tmp/whatever")
    assert scout.repo_hash(p) == scout.repo_hash(p)
    assert scout.repo_hash(Path("/tmp/other")) != scout.repo_hash(p)


def test_file_for_node_handles_relpath_and_symbol():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        (td_p / "src").mkdir()
        f = td_p / "src" / "foo.py"
        f.write_text("x = 1")
        assert scout.file_for_node("src/foo.py", td_p) == f
        assert scout.file_for_node("src/foo.py:Foo.bar", td_p) == f
        assert scout.file_for_node("src/missing.py", td_p) is None


def test_file_for_node_handles_dotted_form():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        (td_p / "src").mkdir()
        (td_p / "src" / "auth.py").write_text("def login(): ...")
        # dotted: src.auth.login -> src/auth.py
        assert scout.file_for_node("src.auth.login", td_p) == td_p / "src" / "auth.py"


def test_build_scout_writes_artifacts(tmp_path=None):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {
            "src/auth.py": "def verify_token():\n    return True\n",
            "src/db.py": "def query():\n    return []\n",
            "src/util.py": "def helper():\n    pass\n",
        })
        # redirect cache to td (avoid touching real ~/.tinyctx)
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            sp = scout.build_scout(graph, proj, top_k=10,
                                   model="qwen-test",
                                   base_url="http://localhost:9999/v1",
                                   _llm_call=_fake_llm)
            assert sp.is_file()
            content = sp.read_text()
            assert "Scout summary" in content
            assert "qwen-test" in content

            mf = scout.manifest_path(proj)
            data = json.loads(mf.read_text())
            assert data["version"] == scout.CACHE_VERSION
            assert data["model"] == "qwen-test"
            assert data["top_k"] == 10
            assert len(data["ranked"]) >= 1
            # file_hashes recorded for source files we found
            assert any("auth.py" in k for k in data["file_hashes"])


def test_is_stale_detects_changed_file():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {
            "src/auth.py": "v1",
            "src/db.py": "v1",
        })
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=10, _llm_call=_fake_llm)
            stale, reason = scout.is_stale(proj)
            assert stale is False, reason

            # mutate a tracked file
            (proj / "src" / "auth.py").write_text("v2 changed")
            stale, reason = scout.is_stale(proj)
            assert stale is True
            assert "changed" in reason.lower()


def test_is_stale_detects_missing_manifest():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "project"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            stale, reason = scout.is_stale(proj)
            assert stale is True
            assert reason == "no manifest"


def test_status_reports_state():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            assert scout.status(proj)["state"] == "absent"
            scout.build_scout(graph, proj, top_k=10, _llm_call=_fake_llm)
            s = scout.status(proj)
            assert s["state"] == "fresh"
            assert s["nodes"] >= 1


def test_gather_scan_targets_deterministic_order():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {
            "src/a.py": "a",
            "src/b.py": "b",
            "src/c.py": "c",
        })
        t1 = scout.gather_scan_targets(graph, proj, top_k=5, max_file_chars=100)
        t2 = scout.gather_scan_targets(graph, proj, top_k=5, max_file_chars=100)
        assert [n.nid for n in t1] == [n.nid for n in t2]


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
