"""Auto-scout: zero-config project-context bootstrap.

The proxy auto-runs scout for any codex request that arrives with
`x-codex-cwd`. First request: schedules background build, returns nothing.
Subsequent requests: prepend cached scout.md to instructions.

Tests use a tmp project directory and bypass the local-model call by
monkey-patching scout.build_scout (which is what calls DeepSeek)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def _make_project(tmp: Path) -> Path:
    """Create a tiny synthetic Python project."""
    proj = tmp / "fake-project"
    proj.mkdir()
    (proj / "main.py").write_text(
        "def hello():\n    print('hi')\n\nclass Foo:\n    pass\n"
    )
    (proj / "utils.py").write_text(
        "import os\n\ndef helper():\n    return os.environ\n"
    )
    (proj / "node_modules").mkdir()
    (proj / "node_modules" / "skipme.js").write_text("// should be skipped")
    (proj / ".git").mkdir()
    (proj / ".git" / "HEAD").write_text("ref: skipped")
    return proj


# ─── inject_into_body ──────────────────────────────────────────────────────


def test_inject_into_body_prepends_to_instructions():
    from tinyctx.auto_scout import inject_into_body
    body = {"instructions": "Hello, codex.", "input": []}
    out, injected = inject_into_body(body, "Project summary text.")
    assert injected is True
    inst = out["instructions"]
    assert inst.startswith("<!-- tinyctx auto-scout BEGIN -->")
    assert "Project summary text." in inst
    assert "<!-- tinyctx auto-scout END -->" in inst
    assert "Hello, codex." in inst
    # original body untouched
    assert body["instructions"] == "Hello, codex."


def test_inject_into_body_idempotent_with_marker():
    from tinyctx.auto_scout import inject_into_body
    body = {"instructions": "<!-- tinyctx auto-scout BEGIN -->\nfoo\n<!-- tinyctx auto-scout END -->\n\nrest"}
    out, injected = inject_into_body(body, "summary")
    assert injected is False
    assert out is body  # unchanged


def test_inject_into_body_no_op_for_empty_summary():
    from tinyctx.auto_scout import inject_into_body
    body = {"instructions": "x"}
    out, injected = inject_into_body(body, "")
    assert injected is False
    assert out is body


def test_inject_into_body_no_op_when_instructions_missing():
    from tinyctx.auto_scout import inject_into_body
    body = {"input": []}
    out, injected = inject_into_body(body, "summary")
    assert injected is False


def test_inject_into_body_no_op_when_instructions_not_string():
    from tinyctx.auto_scout import inject_into_body
    body = {"instructions": {"nested": "x"}}
    out, injected = inject_into_body(body, "summary")
    assert injected is False


# ─── get_scout (cache lookup) ──────────────────────────────────────────────


def test_get_scout_returns_none_when_no_cwd():
    from tinyctx.auto_scout import get_scout
    assert get_scout(None) is None
    assert get_scout("") is None


def test_get_scout_returns_none_when_no_cache():
    from tinyctx.auto_scout import get_scout
    with tempfile.TemporaryDirectory() as td:
        # tmp dir has no scout cache
        assert get_scout(td) is None


def test_get_scout_returns_cached_content(monkeypatch, tmp_path):
    """Pre-populate the cache then verify get_scout reads it back."""
    from tinyctx import auto_scout, scout
    proj = _make_project(tmp_path)
    cdir = scout.cache_dir(proj)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "scout.md").write_text("# cached project summary\n", encoding="utf-8")
    out = auto_scout.get_scout(proj)
    assert out is not None
    assert "cached project summary" in out


# ─── _fallback_scan (in-tree scanner) ──────────────────────────────────────


def test_fallback_scan_picks_up_python_files(tmp_path):
    from tinyctx.auto_scout import _fallback_scan
    proj = _make_project(tmp_path)
    g = _fallback_scan(proj)
    assert g is not None
    assert "nodes" in g and "edges" in g
    ids = {n["id"] for n in g["nodes"]}
    assert "main.py" in ids
    assert "utils.py" in ids


def test_fallback_scan_skips_node_modules_and_git(tmp_path):
    from tinyctx.auto_scout import _fallback_scan
    proj = _make_project(tmp_path)
    g = _fallback_scan(proj)
    assert g is not None
    ids = {n["id"] for n in g["nodes"]}
    # node_modules/skipme.js, .git/HEAD must be ignored
    assert not any("node_modules" in i for i in ids), f"got: {ids}"
    assert not any(".git" in i for i in ids), f"got: {ids}"


def test_fallback_scan_returns_none_on_empty_project(tmp_path):
    from tinyctx.auto_scout import _fallback_scan
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _fallback_scan(empty) is None


def test_fallback_scan_output_consumable_by_graphify_adapter(tmp_path):
    """The fallback scan must produce a shape graphify_adapter._from_flat
    can consume — that's what hands data to interest.py PageRank."""
    from tinyctx.auto_scout import _fallback_scan
    from tinyctx.graphify_adapter import _from_flat
    proj = _make_project(tmp_path)
    g = _fallback_scan(proj)
    nodes = _from_flat(g)
    assert len(nodes) >= 2
    # Each adapted node has id + size info
    ids = {n["id"] for n in nodes}
    assert "main.py" in ids
    assert "utils.py" in ids


# ─── schedule_bootstrap (bootstrap orchestration) ──────────────────────────


def test_schedule_bootstrap_no_op_when_scout_already_exists(tmp_path, monkeypatch):
    """If scout.md already exists, schedule_bootstrap is a no-op (no
    background task spawned)."""
    from tinyctx import auto_scout, scout
    proj = _make_project(tmp_path)
    cdir = scout.cache_dir(proj)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "scout.md").write_text("# already there\n")

    spawn_count = {"n": 0}

    def fake_to_thread(*args, **kwargs):
        spawn_count["n"] += 1
        async def _coro(): pass
        return _coro()
    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
    auto_scout.schedule_bootstrap(proj)
    assert spawn_count["n"] == 0


def test_schedule_bootstrap_no_op_for_invalid_cwd():
    from tinyctx import auto_scout
    # Should never raise
    auto_scout.schedule_bootstrap(None)
    auto_scout.schedule_bootstrap("")
    auto_scout.schedule_bootstrap("/nonexistent/path/does/not/exist")
    # If we reach here without exception, we pass


def test_schedule_bootstrap_dedupes_repeat_calls(tmp_path, monkeypatch):
    """Calling schedule_bootstrap twice for the same project must only
    spawn the background task once. Otherwise repeat requests would
    spam graphify subprocesses."""
    from tinyctx import auto_scout
    proj = _make_project(tmp_path)
    # Reset internal state
    auto_scout._BOOTSTRAPPED_OR_INFLIGHT.discard(str(proj.resolve()))

    spawn_count = {"n": 0}

    def fake_to_thread(*args, **kwargs):
        spawn_count["n"] += 1
        async def _coro(): pass
        return _coro()

    # Need a running event loop for create_task
    import asyncio
    async def runner():
        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
        auto_scout.schedule_bootstrap(proj)
        auto_scout.schedule_bootstrap(proj)
        auto_scout.schedule_bootstrap(proj)
        # let any pending tasks settle
        await asyncio.sleep(0)

    asyncio.new_event_loop().run_until_complete(runner())
    assert spawn_count["n"] == 1, f"expected 1 spawn, got {spawn_count['n']}"


# ─── _bootstrap_sync (full pipeline with scout patched) ────────────────────


def test_bootstrap_sync_uses_fallback_when_graphify_missing(tmp_path, monkeypatch):
    """With graphify NOT on PATH and install_graphify=False, the bootstrap
    must use the fallback scanner and successfully invoke scout.build_scout."""
    from tinyctx import auto_scout, scout
    proj = _make_project(tmp_path)

    monkeypatch.setattr("shutil.which", lambda cmd: None)

    # Replace scout.build_scout to avoid calling local LLM
    captured = {}
    def fake_build_scout(graph_path, project_root, **kw):
        captured["graph_path"] = Path(graph_path)
        captured["project_root"] = Path(project_root)
        # Verify the graph file we wrote is parseable + has our project's nodes
        graph = json.loads(Path(graph_path).read_text())
        assert "nodes" in graph
        ids = {n.get("id") for n in graph["nodes"]}
        assert "main.py" in ids
        # Persist a fake scout.md so tests can verify e2e
        sp = scout.scout_path(project_root)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("# fake scout output\n")
        return sp

    monkeypatch.setattr("tinyctx.scout.build_scout", fake_build_scout)
    auto_scout._BOOTSTRAPPED_OR_INFLIGHT.discard(str(proj.resolve()))

    auto_scout._bootstrap_sync(proj, install_graphify=False)

    assert captured.get("project_root") == proj.resolve() or \
           captured.get("project_root") == proj
    # scout.md should now exist
    assert scout.scout_path(proj).is_file()


def test_bootstrap_sync_silent_on_build_scout_failure(tmp_path, monkeypatch):
    """If the local-model call fails (LMStudio not running, etc.), we
    must not raise. The proxy never blocks on auto-scout."""
    from tinyctx import auto_scout, scout
    proj = _make_project(tmp_path)

    monkeypatch.setattr("shutil.which", lambda cmd: None)

    def crashing_build(graph_path, project_root, **kw):
        raise RuntimeError("local model unreachable")

    monkeypatch.setattr("tinyctx.scout.build_scout", crashing_build)
    auto_scout._BOOTSTRAPPED_OR_INFLIGHT.discard(str(proj.resolve()))

    # Must not raise
    auto_scout._bootstrap_sync(proj, install_graphify=False)
    # And no scout.md was produced
    assert not scout.scout_path(proj).is_file()


# ─── default config ────────────────────────────────────────────────────────


def test_auto_scout_defaults_to_enabled():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.auto_scout is True
    # auto-install pipx is intrusive; default off
    assert cfg.auto_scout_install_graphify is False
