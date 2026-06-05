"""Tests for tinyctx.knowledge_base.

Fully offline and deterministic: no real LLM, no embeddings, no network.
The store is redirected to a tmp dir via TINYCTX_KB_DIR. These tests also
exercise the real tinyctx.retrieval_fanout integration (provider contract,
sensitive-text filtering, inject_context), since the KB is designed to plug
into that existing fan-out layer.
"""
from __future__ import annotations

import pytest

from tinyctx import knowledge_base as kb


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYCTX_KB_DIR", str(tmp_path / "kb"))
    # Keep chunking deterministic and small so multi-chunk paths are covered.
    monkeypatch.setenv("TINYCTX_KB_MAX_CHARS", "200")
    monkeypatch.setenv("TINYCTX_KB_OVERLAP", "20")
    yield


# --------------------------------------------------------------- primitives

def test_scope_hash_stable_and_distinct():
    assert kb.scope_hash("projA") == kb.scope_hash("projA")
    assert kb.scope_hash("projA") != kb.scope_hash("projB")


def test_chunk_text_short_returns_single():
    assert kb.chunk_text("hello world", max_chars=200, overlap=20) == ["hello world"]


def test_chunk_text_empty_returns_empty():
    assert kb.chunk_text("   ", max_chars=200, overlap=20) == []


def test_chunk_text_splits_long_with_progress():
    text = ("alpha beta gamma delta. " * 60).strip()
    chunks = kb.chunk_text(text, max_chars=120, overlap=20)
    assert len(chunks) > 1
    # every chunk is within a sane size bound (boundary-aware, not exact)
    assert all(len(c) <= 160 for c in chunks)
    # reassembled content still contains the load-bearing tokens
    assert "alpha" in chunks[0]


# ------------------------------------------------------------- ingest/search

def test_ingest_and_search_routes_to_rare_term_doc():
    kb.ingest("s", text="the quick brown zebra runs fast", doc_id="a")
    kb.ingest("s", text="the slow green turtle walks", doc_id="b")
    kb.ingest("s", text="a plain ordinary sentence here", doc_id="c")

    hits = kb.search("s", "zebra", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].doc_id == "a"
    assert 0.0 < hits[0].score <= 0.85


def test_search_empty_scope_returns_empty():
    assert kb.search("nope", "anything", top_k=5) == []


def test_search_no_overlap_returns_empty():
    kb.ingest("s", text="completely unrelated words", doc_id="a")
    assert kb.search("s", "xylophone qwertzuiop", top_k=5) == []


def test_scopes_are_isolated():
    kb.ingest("scopeA", text="unicorn rainbow sparkle", doc_id="a")
    kb.ingest("scopeB", text="dragon mountain fire", doc_id="b")

    assert kb.search("scopeA", "unicorn", top_k=5)
    assert kb.search("scopeB", "unicorn", top_k=5) == []
    assert kb.search("scopeB", "dragon", top_k=5)


def test_ingest_same_doc_id_replaces_chunks():
    kb.ingest("s", text="first version mentions falcon", doc_id="d")
    assert kb.search("s", "falcon", top_k=5)
    kb.ingest("s", text="second version mentions penguin", doc_id="d")
    # old content gone, new content present, still a single doc
    assert kb.search("s", "falcon", top_k=5) == []
    assert kb.search("s", "penguin", top_k=5)
    assert kb.stats("s")["docs"] == 1


def test_remove_deletes_doc():
    kb.ingest("s", text="ephemeral content lynx", doc_id="x")
    assert kb.remove("s", "x") is True
    assert kb.remove("s", "x") is False
    assert kb.search("s", "lynx", top_k=5) == []


def test_stats_and_list_docs():
    kb.ingest("s", text="alpha", doc_id="a")
    kb.ingest("s", text="beta", doc_id="b")
    st = kb.stats("s")
    assert st["docs"] == 2 and st["chunks"] >= 2
    ids = {d["doc_id"] for d in kb.list_docs("s")}
    assert ids == {"a", "b"}


# ------------------------------------------------------- file + degradation

def test_ingest_plain_text_file_without_markitdown(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("# Guide\nthe configuration uses a wombat parameter\n", encoding="utf-8")
    res = kb.ingest("s", path=f)
    assert res.n_chunks >= 1 and res.doc_id == "guide.md"
    assert kb.search("s", "wombat", top_k=5)


def test_convert_to_text_uses_injected_converter(tmp_path):
    f = tmp_path / "paper.bin"
    f.write_bytes(b"\x00\x01binary")
    res = kb.ingest("s", path=f, _converter=lambda p: "converted text with rare okapi token")
    assert res.n_chunks >= 1
    hits = kb.search("s", "okapi", top_k=5)
    assert hits and hits[0].doc_id == "paper.bin"


# ----------------------------------------------------------- secret safety

def test_ingest_skips_sensitive_filename(tmp_path):
    f = tmp_path / "secrets.txt"
    f.write_text("token=ghp_supersecretvalue", encoding="utf-8")
    res = kb.ingest("s", path=f)
    assert res.n_chunks == 0 and res.skipped == "sensitive path"
    assert kb.search("s", "token", top_k=5) == []


def test_ingest_drops_secret_looking_chunks():
    res = kb.ingest("s", text="api_key: sk-live-supersecret-value", doc_id="leak")
    assert res.n_chunks == 0 and "sensitive" in res.skipped
    assert kb.search("s", "api_key", top_k=5) == []


# ------------------------------------------------ retrieval_fanout provider

def test_provider_returns_retrieval_hits():
    from tinyctx.retrieval_fanout import RetrievalHit

    kb.ingest("s", text="the migration uses an aardvark strategy", doc_id="m.md")
    provider = kb.knowledge_base_provider("s", top_k=3)
    hits = provider("aardvark")
    assert hits and isinstance(hits[0], RetrievalHit)
    assert hits[0].source == "knowledge_base"
    assert hits[0].path == "m.md"
    assert "aardvark" in hits[0].snippet


def test_provider_plugs_into_run_fanout_and_inject():
    from tinyctx.retrieval_fanout import inject_context, run_fanout

    kb.ingest("s", text="caching layer relies on a marmot eviction policy", doc_id="cache.md")
    provider = kb.knowledge_base_provider("s", top_k=5)

    hits = run_fanout("marmot eviction", [provider], top_k=5)
    assert hits and any("marmot" in h.snippet for h in hits)

    body = {"instructions": "system", "input": []}
    out, injected = inject_context(body, hits, reason="knowledge_base scope=s")
    assert injected is True
    assert "marmot" in out["instructions"]
    assert "BEGIN TINYCTX RETRIEVAL FANOUT" in out["instructions"]


def test_provider_filters_secret_snippets_defence_in_depth():
    # Force a secret-looking chunk straight into the store (bypassing ingest
    # filtering) to prove the provider re-filters on the way out.
    from tinyctx import knowledge_base as _kb

    store = _kb._load_store("s")
    store["docs"]["raw"] = {"source": "inline", "file_hash": "x", "n_chunks": 1, "ingested_at": 0}
    store["chunks"].append({"doc_id": "raw", "ordinal": 0, "text": "password = hunter2-supersecret"})
    _kb._save_store("s", store)

    provider = _kb.knowledge_base_provider("s", top_k=5)
    assert provider("password") == []


# ----------------------------------------------------------------- CLI smoke

def test_cli_ingest_search_roundtrip(tmp_path, capsys):
    f = tmp_path / "notes.md"
    f.write_text("the deploy pipeline triggers a quokka job", encoding="utf-8")
    assert kb.main(["ingest", "cliscope", str(f)]) == 0
    assert kb.main(["search", "cliscope", "quokka"]) == 0
    out = capsys.readouterr().out
    assert "notes.md" in out
