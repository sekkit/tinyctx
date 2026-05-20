"""Tests for tinyctx.memory. We don't require mem0 to be installed; these
tests verify graceful degradation. The "live" mem0 path is only exercised
when the user opts into the [mem] extra."""
from __future__ import annotations

import io
import contextlib
import sys
import types
from unittest import mock

import pytest

from tinyctx import memory


# --------------------------------------------------------------- helpers


class _FakeMem:
    """Minimal in-memory fake of mem0.Memory for behavioral tests.

    Stores text+metadata under user_id and supports content-dedup so we
    can verify "same content twice doesn't double-store" behavior.
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def add(self, text, *, user_id="default", metadata=None):
        bucket = self._store.setdefault(user_id, [])
        # dedup by exact text — mirrors mem0's idempotent semantics
        for existing in bucket:
            if existing["memory"] == text:
                return {"results": [{"id": existing["id"], "event": "NOOP"}]}
        item = {"id": f"m{len(bucket)}", "memory": text,
                "metadata": metadata or {}}
        bucket.append(item)
        return {"results": [{"id": item["id"], "event": "ADD"}]}

    def search(self, query, *, user_id="default", limit=5):
        bucket = self._store.get(user_id, [])
        # "results" wrapped form
        hits = [{"memory": x["memory"], "score": 0.9}
                for x in bucket if query.lower() in x["memory"].lower()]
        return {"results": hits[:limit]}

    def get_all(self, *, user_id="default", limit=100):
        return list(self._store.get(user_id, []))[:limit]


def _install_fake_mem0(monkeypatch):
    """Install a fake `mem0` module so MemStore / is_available succeed."""
    fake_module = types.ModuleType("mem0")

    class _Memory:
        def __init__(self):
            self._inner = _FakeMem()

        # mem0.Memory.from_config classmethod-style entrypoint
        @classmethod
        def from_config(cls, _cfg):
            inst = cls()
            return inst

        def add(self, *a, **kw):
            return self._inner.add(*a, **kw)

        def search(self, *a, **kw):
            return self._inner.search(*a, **kw)

        def get_all(self, *a, **kw):
            return self._inner.get_all(*a, **kw)

    fake_module.Memory = _Memory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", fake_module)
    return fake_module


def test_is_available_matches_mem0_importability():
    """Manual oracle: tinyctx.memory.is_available() is True iff `import
    mem0` succeeds. Most CI environments don't have mem0 → False."""
    try:
        import mem0  # noqa: F401
        assert memory.is_available() is True
    except ImportError:
        assert memory.is_available() is False


def test_memstore_raises_clean_error_when_mem0_absent():
    """If mem0 isn't installed, MemStore() raises ImportError with a
    helpful message instead of crashing somewhere weird."""
    if memory.is_available():
        # mem0 IS installed; we can't test the absent path without faking it.
        return
    try:
        memory.MemStore()
    except ImportError as e:
        assert "tinyctx[mem]" in str(e) or "mem0ai" in str(e)
        return
    raise AssertionError("MemStore() did not raise ImportError")


def test_cli_available_subcommand():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = memory.main(["available"])
    out = buf.getvalue().strip()
    if memory.is_available():
        assert rc == 0
        assert out == "yes"
    else:
        assert rc == 1
        assert out.startswith("no")


def test_cli_add_without_mem0_returns_error():
    """`tinyctx-mem add ...` should not crash, just exit 1."""
    if memory.is_available():
        return
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["add", "user prefers tabs"])
    assert rc == 1
    assert "not installed" in buf_err.getvalue()


def test_cli_search_without_mem0_returns_error():
    if memory.is_available():
        return
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["search", "tabs"])
    assert rc == 1
    assert "not installed" in buf_err.getvalue()


def test_ingest_compaction_without_structured_returns_error():
    """If there's no structured compaction yet, ingest must fail cleanly."""
    if not memory.is_available():
        # Without mem0, we still hit the "not installed" path first.
        buf_err = io.StringIO()
        with contextlib.redirect_stderr(buf_err):
            rc = memory.main(["ingest-compaction", "--root", "/tmp"])
        assert rc == 1
        return
    # If mem0 IS installed, /tmp won't have a structured compaction.
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["ingest-compaction", "--root", "/tmp"])
    assert rc == 1


# ---------------------------------------------------- new behavioral tests


def test_memstore_store_retrieve_roundtrip(monkeypatch):
    """add() + search() should round-trip a stored note via MemStore."""
    _install_fake_mem0(monkeypatch)
    m = memory.MemStore()
    m.add("user prefers tabs over spaces", user_id="alice")
    hits = m.search("tabs", user_id="alice")
    assert isinstance(hits, list)
    assert len(hits) == 1
    assert hits[0]["memory"] == "user prefers tabs over spaces"


def test_memstore_dedup_on_identical_text(monkeypatch):
    """Adding the same content twice must not produce two stored entries."""
    _install_fake_mem0(monkeypatch)
    m = memory.MemStore()
    m.add("repeat me", user_id="bob")
    m.add("repeat me", user_id="bob")
    items = m.get_all(user_id="bob")
    assert len(items) == 1


def test_memstore_search_normalizes_results_dict(monkeypatch):
    """When the underlying mem0 returns {'results': [...]}, MemStore.search
    must unwrap it to a plain list."""
    _install_fake_mem0(monkeypatch)
    m = memory.MemStore()
    m.add("alpha beta gamma", user_id="u")
    hits = m.search("alpha", user_id="u")
    assert isinstance(hits, list) and not isinstance(hits, dict)


def test_memstore_search_handles_plain_list_return(monkeypatch):
    """mem0 sometimes returns a bare list (older API). MemStore.search
    must accept that shape too."""
    _install_fake_mem0(monkeypatch)
    m = memory.MemStore()

    def _list_search(query, *, user_id="default", limit=5):
        return [{"memory": "list-shaped"}]

    monkeypatch.setattr(m._mem, "search", _list_search)
    hits = m.search("anything")
    assert hits == [{"memory": "list-shaped"}]


def test_memstore_search_handles_unknown_shape(monkeypatch):
    """If mem0 returns something weird (e.g. None), search() returns []
    rather than raising."""
    _install_fake_mem0(monkeypatch)
    m = memory.MemStore()
    monkeypatch.setattr(m._mem, "search", lambda *a, **kw: None)
    assert m.search("x") == []


def test_memstore_falls_back_when_from_config_raises(monkeypatch):
    """If Memory.from_config() blows up, MemStore must fall back to the
    bare Memory() default rather than propagating the error."""
    fake = types.ModuleType("mem0")
    construct_log = []

    class _Memory:
        def __init__(self):
            construct_log.append("default")

        @classmethod
        def from_config(cls, _cfg):
            construct_log.append("from_config")
            raise RuntimeError("config rejected")

    fake.Memory = _Memory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", fake)
    m = memory.MemStore()
    assert "from_config" in construct_log
    assert "default" in construct_log
    # Internal _mem is a Memory() instance from the fallback path.
    assert m._mem.__class__ is _Memory


def test_cli_search_no_hits_emits_message_and_returns_1(monkeypatch):
    """`tinyctx-mem search` returns rc=1 with a "no hits" stderr line when
    there's nothing matching."""
    _install_fake_mem0(monkeypatch)
    monkeypatch.setattr(memory, "is_available", lambda: True)
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["search", "nonexistent-query-xyz"])
    assert rc == 1
    assert "(no hits)" in buf_err.getvalue()


def test_cli_search_renders_score_prefix(monkeypatch):
    """When mem0 returns hits with numeric `score`, the CLI prefixes
    each line with `[score]`."""
    _install_fake_mem0(monkeypatch)
    monkeypatch.setattr(memory, "is_available", lambda: True)
    # Pre-populate via direct add so search has a hit.
    m = memory.MemStore()
    m.add("the quick brown fox", user_id="default")
    # The CLI builds its own MemStore — but our fake `mem0` module is in
    # sys.modules so the new instance reuses the same fake (different
    # internal _FakeMem, however). To exercise the rendering, monkeypatch
    # MemStore so we get the *same* underlying store back.
    monkeypatch.setattr(memory, "MemStore", lambda *a, **kw: m)
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out):
        rc = memory.main(["search", "fox"])
    assert rc == 0
    out = buf_out.getvalue()
    assert "the quick brown fox" in out
    assert out.lstrip().startswith("[")  # score prefix present


def test_cli_stats_prints_counts(monkeypatch):
    """`tinyctx-mem stats` must report user_id and memory count."""
    _install_fake_mem0(monkeypatch)
    monkeypatch.setattr(memory, "is_available", lambda: True)
    m = memory.MemStore()
    m.add("note one", user_id="default")
    m.add("note two", user_id="default")
    monkeypatch.setattr(memory, "MemStore", lambda *a, **kw: m)
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out):
        rc = memory.main(["stats"])
    assert rc == 0
    out = buf_out.getvalue()
    assert "user_id: default" in out
    assert "memories: 2" in out


def test_cli_ingest_compaction_happy_path(monkeypatch):
    """When latest_structured returns a real payload, ingest pushes facts
    AND compartment summaries into mem0."""
    _install_fake_mem0(monkeypatch)
    monkeypatch.setattr(memory, "is_available", lambda: True)
    fake_data = {
        "facts": [
            {"claim": "app uses pytest"},
            {"claim": "  "},                 # empty after strip → skipped
            {"claim": "deploys via uv"},
        ],
        "compartments": [
            {"topic": "auth", "summary": "JWT-based"},
            {"topic": "", "summary": ""},     # empty → skipped
        ],
    }
    # Pre-import to ensure the relative import in _cmd_ingest_compaction
    # binds against this monkeypatched function.
    import tinyctx.continuity as cont
    monkeypatch.setattr(cont, "latest_structured", lambda root: fake_data)
    m = memory.MemStore()
    monkeypatch.setattr(memory, "MemStore", lambda *a, **kw: m)
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out):
        rc = memory.main(["ingest-compaction", "--root", "/tmp"])
    assert rc == 0
    items = m.get_all(user_id="default")
    # 2 valid facts + 1 valid compartment = 3 entries
    assert len(items) == 3
    texts = {item["memory"] for item in items}
    assert "app uses pytest" in texts
    assert "deploys via uv" in texts
    assert "auth: JWT-based" in texts


def test_cli_unknown_subcommand_exits_with_argparse_error():
    """argparse should exit 2 on unknown subcommand and not crash."""
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        with pytest.raises(SystemExit) as exc:
            memory.main(["totally-unknown"])
    assert exc.value.code == 2


if __name__ == "__main__":
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
