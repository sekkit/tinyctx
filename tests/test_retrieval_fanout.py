"""Local retrieval fan-out primitives.

External providers stay disabled by default; this tests the safe local
merge/dedup/budget/injection layer that proxy code can call.
"""
from __future__ import annotations


def test_run_fanout_merges_dedupes_and_orders_hits():
    from tinyctx.retrieval_fanout import RetrievalHit, run_fanout

    def provider_a(query: str):
        return [
            RetrievalHit("scout", "tinyctx/router.py", "Router decides", 0.7),
            RetrievalHit("scout", "tinyctx/router.py", "Router decides", 0.5),
        ]

    def provider_b(query: str):
        return [
            RetrievalHit("ctx_pack", "tinyctx/proxy.py", "Proxy assembles", 0.9),
        ]

    hits = run_fanout("router proxy", [provider_a, provider_b], top_k=2)

    assert [(h.source, h.path) for h in hits] == [
        ("ctx_pack", "tinyctx/proxy.py"),
        ("scout", "tinyctx/router.py"),
    ]


def test_run_fanout_dedupes_across_providers_and_merges_sources():
    from tinyctx.retrieval_fanout import RetrievalHit, run_fanout

    def provider_a(query: str):
        return [RetrievalHit("scout", "tinyctx/router.py", "same", 0.7)]

    def provider_b(query: str):
        return [RetrievalHit("ctx_pack", "tinyctx/router.py", "same", 0.9)]

    hits = run_fanout("router", [provider_a, provider_b], top_k=5)

    assert len(hits) == 1
    assert hits[0].path == "tinyctx/router.py"
    assert hits[0].source == "ctx_pack,scout"
    assert hits[0].score == 0.9


def test_run_fanout_respects_char_budget():
    from tinyctx.retrieval_fanout import RetrievalHit, run_fanout

    def provider(query: str):
        return [
            RetrievalHit("scout", "a.py", "x" * 80, 1.0),
            RetrievalHit("scout", "b.py", "y" * 80, 0.9),
        ]

    hits = run_fanout("q", [provider], top_k=5, max_chars=120)

    assert len(hits) == 1
    assert hits[0].path == "a.py"


def test_inject_context_is_idempotent():
    from tinyctx.retrieval_fanout import RetrievalHit, inject_context

    body = {"instructions": "system", "input": []}
    hits = [RetrievalHit("scout", "tinyctx/router.py", "Router decides", 0.9)]

    once, injected = inject_context(body, hits, reason="self-consistency disagreement")
    twice, injected_again = inject_context(once, hits, reason="again")

    assert injected is True
    assert injected_again is False
    assert once == twice
    assert once["instructions"].count("BEGIN TINYCTX RETRIEVAL FANOUT") == 1
    assert "tinyctx/router.py" in once["instructions"]


def test_extract_query_text_reads_tail_user_message():
    from tinyctx.retrieval_fanout import extract_query_text

    body = {
        "input": [
            {"role": "user", "content": "old"},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "new request"}]},
        ]
    }

    assert extract_query_text(body) == "new request"


def test_mentioned_path_provider_reads_existing_file(tmp_path):
    from tinyctx.retrieval_fanout import mentioned_path_provider

    target = tmp_path / "tinyctx" / "router.py"
    target.parent.mkdir()
    target.write_text("class Router:\n    pass\n", encoding="utf-8")

    provider = mentioned_path_provider(tmp_path)
    hits = provider("Please inspect tinyctx/router.py")

    assert len(hits) == 1
    assert hits[0].source == "mentioned_path"
    assert hits[0].path == "tinyctx/router.py"
    assert "class Router" in hits[0].snippet


def test_mentioned_path_provider_skips_symlink_escape(tmp_path):
    from tinyctx.retrieval_fanout import mentioned_path_provider

    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("do not inject", encoding="utf-8")
    link = tmp_path / "leak.txt"
    link.symlink_to(outside)

    provider = mentioned_path_provider(tmp_path)

    assert provider("Please inspect leak.txt") == []


def test_mentioned_path_provider_skips_sensitive_path_names(tmp_path):
    from tinyctx.retrieval_fanout import mentioned_path_provider

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-live-secret\n", encoding="utf-8")
    secret_file = tmp_path / "secrets.txt"
    secret_file.write_text("token=ghp_secret\n", encoding="utf-8")

    provider = mentioned_path_provider(tmp_path)

    assert provider("Please inspect .env and secrets.txt") == []


def test_mentioned_path_provider_skips_secret_like_file_contents(tmp_path):
    from tinyctx.retrieval_fanout import mentioned_path_provider

    target = tmp_path / "config.txt"
    target.write_text("api_key: sk-live-secret\n", encoding="utf-8")

    provider = mentioned_path_provider(tmp_path)

    assert provider("Please inspect config.txt") == []


def test_inject_context_filters_sensitive_provider_hits():
    from tinyctx.retrieval_fanout import RetrievalHit, inject_context

    body = {"instructions": "system", "input": []}
    hits = [
        RetrievalHit("scout", "safe.py", "print('ok')", 0.9),
        RetrievalHit("provider", "config.txt", "api_key: sk-live-secret", 1.0),
    ]

    out, injected = inject_context(body, hits, reason="self-consistency disagreement")

    assert injected is True
    assert "safe.py" in out["instructions"]
    assert "sk-live-secret" not in out["instructions"]
    assert "config.txt" not in out["instructions"]


def test_scout_cache_provider_reads_cached_summary(tmp_path):
    from tinyctx.retrieval_fanout import scout_cache_provider

    provider = scout_cache_provider(lambda root: "## Architecture\nRouter decides route")
    hits = provider("router")

    assert len(hits) == 1
    assert hits[0].source == "scout_cache"
    assert "Router decides" in hits[0].snippet


def test_default_providers_include_local_sources_only(tmp_path):
    from tinyctx.retrieval_fanout import default_providers

    providers = default_providers(tmp_path)

    assert len(providers) >= 2
    assert all(callable(p) for p in providers)


def test_inject_for_disagreement_uses_mentioned_paths(tmp_path):
    from tinyctx.retrieval_fanout import inject_for_disagreement

    target = tmp_path / "tinyctx" / "router.py"
    target.parent.mkdir()
    target.write_text("class Router:\n    pass\n", encoding="utf-8")
    body = {
        "instructions": "system",
        "input": [{"role": "user", "content": "inspect tinyctx/router.py"}],
    }

    out, injected = inject_for_disagreement(body, root=tmp_path)

    assert injected is True
    assert "class Router" in out["instructions"]
