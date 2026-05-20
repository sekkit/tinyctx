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


# ---------------------------------------------------------------------------
# Below: extended coverage. Targets the public surface of tinyctx.scout that
# wasn't exercised by the original 8 tests. All filesystem work uses tmp_path
# (or TemporaryDirectory) plus a HOME override to keep the cache fully
# isolated from the user's real ~/.tinyctx.
# ---------------------------------------------------------------------------


def test_repo_hash_is_16_lowercase_hex():
    h = scout.repo_hash(Path("/tmp/some-repo"))
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_cache_paths_consistent_under_home_override(tmp_path):
    """cache_dir / manifest_path / scout_path must all sit under the same
    repo-hashed directory so tests asserting on one imply the others."""
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        cdir = scout.cache_dir(proj)
        mf = scout.manifest_path(proj)
        sp = scout.scout_path(proj)
        assert mf.parent == cdir
        assert sp.parent == cdir
        assert cdir.parent == tmp_path / ".tinyctx" / "cache"
        # the leaf is the repo_hash
        assert cdir.name == scout.repo_hash(proj)


def test_file_hash_returns_empty_for_missing_file(tmp_path):
    assert scout.file_hash(tmp_path / "does_not_exist.py") == ""


def test_file_hash_stable_and_changes_with_content(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    h1 = scout.file_hash(f)
    h2 = scout.file_hash(f)
    assert h1 == h2 and len(h1) == 16
    f.write_text("hello!")
    assert scout.file_hash(f) != h1


def test_file_for_node_dotted_resolves_typescript_and_go(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "client.ts").write_text("export {}")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "server.go").write_text("package pkg")
    assert scout.file_for_node("src.client.handler", tmp_path) == tmp_path / "src" / "client.ts"
    assert scout.file_for_node("pkg.server.Run", tmp_path) == tmp_path / "pkg" / "server.go"


def test_file_for_node_returns_none_for_pure_symbol(tmp_path):
    # No file path component, no dotted form that resolves to anything.
    assert scout.file_for_node("primitive", tmp_path) is None


def test_truncate_at_and_below_limit():
    assert scout._truncate("abc", 5) == "abc"
    assert scout._truncate("abcde", 5) == "abcde"
    out = scout._truncate("abcdef", 5)
    assert out.startswith("abcde")
    assert "[truncated]" in out


def test_gather_scan_targets_empty_graph_returns_empty(tmp_path):
    g = tmp_path / "empty.json"
    g.write_text(json.dumps({"nodes": []}))
    assert scout.gather_scan_targets(g, tmp_path, top_k=5, max_file_chars=100) == []


def test_gather_scan_targets_unresolvable_node_keeps_placeholder(tmp_path):
    """A node whose id can't be mapped to a file is preserved with file=None
    and empty sha/snippet — that's what build_user_prompt then skips."""
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"nodes": [
        {"id": "ghost_node", "wrapped_signature": 1, "wrapped_body": 0, "deps": []},
    ]}))
    out = scout.gather_scan_targets(g, tmp_path, top_k=5, max_file_chars=100)
    assert len(out) == 1
    assert out[0].nid == "ghost_node"
    assert out[0].file is None
    assert out[0].sha == ""
    assert out[0].snippet == ""


def test_gather_scan_targets_truncates_to_max_file_chars():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        big = "x" * 5000
        proj, graph = _write_graph(td_p, {"src/big.py": big})
        out = scout.gather_scan_targets(graph, proj, top_k=5, max_file_chars=1000)
        # Pick the node whose file ended up being big.py.
        big_nodes = [n for n in out if n.file and n.file.endswith("big.py")]
        assert big_nodes, [n.nid for n in out]
        n = big_nodes[0]
        assert "[truncated]" in n.snippet
        # _truncate keeps n chars + a trailing marker line; bound generously.
        assert len(n.snippet) <= 1000 + 64


def test_build_user_prompt_skips_snippetless_nodes_and_uses_separator():
    targets = [
        scout.ScannedNode(nid="real.py:fn", score=0.9, file="/abs/real.py",
                          sha="cafebabe00000000", snippet="def fn(): pass\n"),
        scout.ScannedNode(nid="ghost", score=0.5, file=None, sha="", snippet=""),
    ]
    prompt = scout.build_user_prompt(targets)
    assert "real.py:fn" in prompt
    assert "/abs/real.py" in prompt
    assert "ghost" not in prompt
    # Score formatted with 4 decimal places.
    assert "0.9000 :: real.py:fn" in prompt
    # Code fenced.
    assert "```\ndef fn(): pass\n" in prompt


def test_build_scout_raises_when_no_nodes(tmp_path):
    g = tmp_path / "empty.json"
    g.write_text(json.dumps({"nodes": []}))
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}):
        try:
            scout.build_scout(g, proj, top_k=5, _llm_call=_fake_llm)
        except RuntimeError as e:
            assert "no nodes" in str(e)
        else:
            raise AssertionError("expected RuntimeError on empty graph")


def test_build_scout_passes_kwargs_to_llm_call():
    """Verify the LLM call sees the configured base_url / model / api_key
    rather than scout.py's defaults. This is the hook auto_scout relies on."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x = 1\n"})
        captured = {}

        def fake(system, user, *, base_url, model, api_key=None, **kw):
            captured["base_url"] = base_url
            captured["model"] = model
            captured["api_key"] = api_key
            captured["system"] = system
            return "# summary"

        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=5,
                              base_url="http://example/v1",
                              model="custom-model",
                              api_key="sekret",
                              _llm_call=fake)
        assert captured == {
            "base_url": "http://example/v1",
            "model": "custom-model",
            "api_key": "sekret",
            "system": scout.SCOUT_SYSTEM_PROMPT,
        }


def test_build_scout_manifest_shape_and_file_hashes():
    """The manifest is a contract with is_stale + status — assert its shape."""
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {
            "src/a.py": "alpha\n",
            "src/b.py": "beta\n",
        })
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=10, model="m1",
                              base_url="http://example/v1", _llm_call=_fake_llm)
            mf = json.loads(scout.manifest_path(proj).read_text())
        # Top-level keys
        assert set(mf) == {
            "version", "project_root", "graph_path", "model", "base_url",
            "top_k", "ranked", "file_hashes", "built_at",
        }
        assert mf["version"] == scout.CACHE_VERSION
        assert mf["model"] == "m1"
        assert mf["base_url"] == "http://example/v1"
        # Each ranked entry is the documented dict shape.
        for r in mf["ranked"]:
            assert set(r) == {"id", "score", "file", "sha"}
        # file_hashes maps absolute path -> 16-hex sha.
        assert mf["file_hashes"], "expected at least one file hash"
        for path_str, sha in mf["file_hashes"].items():
            assert Path(path_str).is_absolute()
            assert len(sha) == 16


def test_is_stale_version_mismatch():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=5, _llm_call=_fake_llm)
            mf = scout.manifest_path(proj)
            data = json.loads(mf.read_text())
            data["version"] = scout.CACHE_VERSION + 99
            mf.write_text(json.dumps(data))
            stale, reason = scout.is_stale(proj)
            assert stale is True
            assert reason == "version mismatch"


def test_is_stale_corrupt_manifest():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            mf = scout.manifest_path(proj)
            mf.parent.mkdir(parents=True, exist_ok=True)
            mf.write_text("{ this is not json")
            stale, reason = scout.is_stale(proj)
            assert stale is True
            assert reason == "manifest unreadable"


def test_is_stale_detects_deleted_tracked_file():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {
            "src/a.py": "alpha",
            "src/b.py": "beta",
        })
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=10, _llm_call=_fake_llm)
            stale, _ = scout.is_stale(proj)
            assert stale is False
            # Delete a tracked file.
            (proj / "src" / "a.py").unlink()
            stale, reason = scout.is_stale(proj)
            assert stale is True
            assert reason.startswith("missing:")
            assert "a.py" in reason


def test_status_corrupt_manifest_reports_corrupt():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            mf = scout.manifest_path(proj)
            mf.parent.mkdir(parents=True, exist_ok=True)
            mf.write_text("{not-json")
            s = scout.status(proj)
            assert s["state"] == "corrupt"
            assert s["exists"] is True


def test_status_fresh_carries_manifest_metadata():
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=7, model="custom",
                              _llm_call=_fake_llm)
            s = scout.status(proj)
            assert s["state"] == "fresh"
            assert s["model"] == "custom"
            assert s["top_k"] == 7
            assert isinstance(s["built_at"], (int, float))
            assert s["nodes"] >= 1
            # Path strings are absolute.
            assert Path(s["scout_path"]).is_absolute()
            assert Path(s["cache_dir"]).is_absolute()


def test_cli_path_returns_1_when_no_cache(capsys):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            rc = scout.main(["path", "--root", str(proj)])
        assert rc == 1


def test_cli_path_prints_path_when_cache_exists(capsys):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=5, _llm_call=_fake_llm)
            rc = scout.main(["path", "--root", str(proj)])
            out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out.endswith("scout.md")
        assert Path(out).is_file()


def test_cli_show_prints_scout_md(capsys):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=5, model="show-test",
                              _llm_call=_fake_llm)
            rc = scout.main(["show", "--root", str(proj)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scout summary" in out
        assert "show-test" in out


def test_cli_show_returns_1_when_no_cache(capsys):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"
        proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            rc = scout.main(["show", "--root", str(proj)])
        assert rc == 1


def test_cli_status_json_output_is_valid_json(capsys):
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj, graph = _write_graph(td_p, {"src/a.py": "x"})
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            scout.build_scout(graph, proj, top_k=5, _llm_call=_fake_llm)
            rc = scout.main(["status", "--root", str(proj), "--json"])
            out = capsys.readouterr().out
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["state"] == "fresh"
        # --json sorts keys; the contract surface should include these.
        for k in ("cache_dir", "exists", "nodes", "scout_path", "state"):
            assert k in parsed


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
