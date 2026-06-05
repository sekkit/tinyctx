"""Tests for tinyctx.knowledge_sources.

Fully offline and deterministic: every provider's HTTP is replaced with an
injected `_get` returning canned (status, text). No network is touched.
Also verifies graceful degradation (non-200 / garbage / empty -> []) and
that external_providers() stays OFF unless explicitly enabled.
"""
from __future__ import annotations

import json

import pytest

from tinyctx import knowledge_sources as ks


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYCTX_KB_SOURCES_DIR", str(tmp_path / "kbsrc"))
    # Ensure a polluted real environment can't enable sources during tests.
    for var in (
        "TINYCTX_KB_DEVDOCS", "TINYCTX_KB_DEVDOCS_URL", "TINYCTX_KB_DEVDOCS_SLUGS",
        "TINYCTX_KB_KIWIX", "TINYCTX_KB_KIWIX_URL", "TINYCTX_KB_KIWIX_BOOKS",
        "TINYCTX_KB_WIKIDATA", "TINYCTX_KB_WIKIDATA_URL", "TINYCTX_KB_WIKIDATA_LANG",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _get_const(status, text):
    def _g(url, params):
        return (status, text)
    return _g


# -------------------------------------------------------------------- devdocs

def test_devdocs_matches_entries_locally():
    index = json.dumps({"entries": [
        {"name": "os.path.join", "path": "python~3.12/os.path#join", "type": "os.path"},
        {"name": "json.loads", "path": "python~3.12/json#loads", "type": "json"},
    ]})
    p = ks.devdocs_provider("http://localhost:9292", ["python~3.12"],
                            _get=_get_const(200, index), top_k=5)
    hits = p("os.path.join")
    assert hits
    assert hits[0].source == "devdocs:python~3.12"
    assert "os.path.join" in hits[0].snippet
    assert "http://localhost:9292/python~3.12/" in hits[0].snippet


def test_devdocs_degrades_when_unreachable():
    p = ks.devdocs_provider("http://localhost:9292", ["python~3.12"],
                            _get=_get_const(0, ""), top_k=5)
    assert p("os.path.join") == []


def test_devdocs_no_slugs_returns_empty():
    p = ks.devdocs_provider("http://localhost:9292", [], _get=_get_const(200, "{}"))
    assert p("anything") == []


def test_devdocs_caches_index(tmp_path):
    calls = {"n": 0}

    def counting_get(url, params):
        calls["n"] += 1
        return (200, json.dumps({"entries": [
            {"name": "wombat.config", "path": "p", "type": "t"}]}))

    p = ks.devdocs_provider("http://h", ["slug"], _get=counting_get)
    assert p("wombat")
    assert p("wombat")
    # index.json fetched once, second query served from on-disk cache
    assert calls["n"] == 1


# ---------------------------------------------------------------------- kiwix

def test_parse_kiwix_xml_opensearch():
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        '<item><title>Segfault in C</title><link>A/seg.html</link>'
        '<description>A &lt;b&gt;segfault&lt;/b&gt; happens</description></item>'
        '</channel></rss>'
    )
    out = ks._parse_kiwix(xml)
    assert out and out[0]["title"] == "Segfault in C"
    assert out[0]["link"] == "A/seg.html"


def test_parse_kiwix_json():
    js = json.dumps({"results": [{"title": "T", "url": "A/x", "snippet": "yy"}]})
    out = ks._parse_kiwix(js)
    assert out == [{"title": "T", "link": "A/x", "desc": "yy"}]


def test_kiwix_provider_strips_html_and_degrades():
    xml = ('<rss><channel><item><title>Pointer crash</title>'
           '<link>A/p.html</link><description>use &lt;code&gt;gdb&lt;/code&gt;</description>'
           '</item></channel></rss>')
    p = ks.kiwix_provider("http://localhost:8080", _get=_get_const(200, xml), top_k=5)
    hits = p("crash")
    assert hits and hits[0].source == "kiwix"
    assert "<code>" not in hits[0].snippet and "gdb" in hits[0].snippet
    # degrade
    assert ks.kiwix_provider("http://localhost:8080", _get=_get_const(500, ""))("x") == []


# ------------------------------------------------------------------- wikidata

def test_wikidata_parses_entities():
    payload = json.dumps({"search": [
        {"id": "Q7259", "label": "Ada Lovelace",
         "description": "English mathematician",
         "concepturi": "http://www.wikidata.org/entity/Q7259"},
    ]})
    p = ks.wikidata_provider(_get=_get_const(200, payload), top_k=5)
    hits = p("Ada Lovelace")
    assert hits and hits[0].source == "wikidata"
    assert "Q7259" in hits[0].snippet and "mathematician" in hits[0].snippet
    assert hits[0].path == "http://www.wikidata.org/entity/Q7259"


def test_wikidata_degrades():
    assert ks.wikidata_provider(_get=_get_const(0, ""))("x") == []
    assert ks.wikidata_provider(_get=_get_const(200, "garbage"))("x") == []


def test_wikidata_extracts_entity_from_sentence():
    captured = []

    def cap_get(url, params):
        captured.append(params.get("search"))
        return (200, json.dumps({"search": [
            {"id": "Q7259", "label": "Ada Lovelace", "description": "mathematician"}]}))

    p = ks.wikidata_provider(_get=cap_get)
    hits = p("who was Ada Lovelace and what is she known for")
    assert hits
    assert "Q7259" in hits[0].snippet
    # searched the extracted proper noun, not the whole sentence
    assert any("Ada Lovelace" in (s or "") for s in captured)
    assert all("who was" not in (s or "") for s in captured)


def test_wikidata_disambiguates_by_context():
    # Same label "Transformer", different senses; query context = deep learning.
    payload = json.dumps({"search": [
        {"id": "Q11658", "label": "Transformer",
         "description": "electrical device that transfers energy"},
        {"id": "Q85810444", "label": "Transformer",
         "description": "deep learning model architecture"},
    ]})
    p = ks.wikidata_provider(_get=_get_const(200, payload))
    hits = p("explain the Transformer architecture in deep learning")
    assert hits
    # context-aware rerank promotes the ML sense despite API order
    assert "Q85810444" in hits[0].snippet


def test_wikidata_rerank_stable_without_context_signal():
    # No description overlap -> preserve original wbsearchentities order.
    cands = [
        {"id": "Q1", "label": "X", "description": "alpha"},
        {"id": "Q2", "label": "X", "description": "beta"},
    ]
    assert [e["id"] for e in ks._rerank_by_context("zzz nomatch", cands)] == ["Q1", "Q2"]


def test_http_get_sets_user_agent(monkeypatch):
    # Wikimedia rejects (403) requests without a User-Agent; ensure the
    # default getter always sends a descriptive one.
    import httpx
    captured = {}

    class _Resp:
        status_code = 200
        text = "ok"

    def fake_get(url, params=None, timeout=None, headers=None, follow_redirects=None):
        captured["headers"] = headers or {}
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)
    status, _ = ks._http_get("http://example", {}, timeout_s=5)
    assert status == 200
    assert "tinyctx" in captured["headers"].get("User-Agent", "")


# --------------------------------------------------------- external_providers

def test_external_providers_off_by_default():
    assert ks.external_providers(None) == []


def test_external_providers_enables_via_env(monkeypatch):
    monkeypatch.setenv("TINYCTX_KB_KIWIX", "1")
    monkeypatch.setenv("TINYCTX_KB_KIWIX_URL", "http://localhost:8080")
    monkeypatch.setenv("TINYCTX_KB_WIKIDATA", "true")
    names = [p.__name__ for p in ks.external_providers(None)]
    assert "kiwix_provider" in names
    assert "wikidata_provider" in names
    assert "devdocs_provider" not in names  # not enabled


def test_external_providers_devdocs_requires_url_and_slugs(monkeypatch):
    monkeypatch.setenv("TINYCTX_KB_DEVDOCS", "1")
    assert ks.external_providers(None) == []  # no url/slugs -> skipped
    monkeypatch.setenv("TINYCTX_KB_DEVDOCS_URL", "http://localhost:9292")
    monkeypatch.setenv("TINYCTX_KB_DEVDOCS_SLUGS", "python~3.12,go")
    names = [p.__name__ for p in ks.external_providers(None)]
    assert names == ["devdocs_provider"]


def test_external_providers_reads_cfg_object():
    class Cfg:
        kb_wikidata_enabled = True
    names = [p.__name__ for p in ks.external_providers(Cfg())]
    assert names == ["wikidata_provider"]


# -------------------------------------------------- retrieval_fanout integration

def test_provider_plugs_into_run_fanout_and_inject():
    from tinyctx.retrieval_fanout import inject_context, run_fanout

    payload = json.dumps({"search": [
        {"id": "Q176789", "label": "Mixture of Agents",
         "description": "ensemble LLM method", "concepturi": "http://wikidata.org/Q176789"},
    ]})
    provider = ks.wikidata_provider(_get=_get_const(200, payload))
    hits = run_fanout("mixture of agents", [provider], top_k=5)
    assert hits
    body = {"instructions": "system", "input": []}
    out, injected = inject_context(body, hits, reason="kb_sources test")
    assert injected is True
    assert "Mixture of Agents" in out["instructions"]


# ------------------------------------------------------------------ CLI smoke

def test_cli_reports_no_hits_when_source_disabled(capsys):
    # wikidata against an unreachable injected default -> real HTTP would be
    # attempted; instead exercise the arg path with a source that needs a URL.
    rc = ks.main(["devdocs", "anything"])  # no --url -> empty -> rc 1
    assert rc == 1


# ------------------------------------------------- dashboard config integration

def test_config_loads_knowledge_sources_section(tmp_path, monkeypatch):
    from tinyctx import config as cfgmod
    monkeypatch.setenv("TINYCTX_LOG_DIR", str(tmp_path / "logs"))
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[knowledge_sources]\n'
        'kb_wikidata_enabled = true\n'
        'kb_devdocs_enabled = true\n'
        'kb_devdocs_url = "http://localhost:9292"\n'
        'kb_devdocs_slugs = "python~3.12,go"\n',
        encoding="utf-8")
    monkeypatch.setenv("TINYCTX_CONFIG", str(toml))
    cfg = cfgmod.load_config()
    assert cfg.kb_wikidata_enabled is True
    assert cfg.kb_devdocs_enabled is True
    assert cfg.kb_devdocs_url == "http://localhost:9292"
    # external_providers consumes these config fields (config wins over env)
    names = [p.__name__ for p in ks.external_providers(cfg)]
    assert "devdocs_provider" in names
    assert "wikidata_provider" in names
    assert "kiwix_provider" not in names  # left unset


def test_config_default_none_falls_back_to_env(monkeypatch):
    # Unset in config (None default) -> external_providers uses env fallback.
    from tinyctx import config as cfgmod
    cfg = cfgmod.Config()
    assert cfg.kb_wikidata_enabled is None
    monkeypatch.setenv("TINYCTX_KB_WIKIDATA", "1")
    names = [p.__name__ for p in ks.external_providers(cfg)]
    assert "wikidata_provider" in names


def test_config_schema_exposes_knowledge_sources():
    from tinyctx.config_schema import (
        ALLOWED_SECTIONS, config_schema, validate_sections)
    assert "knowledge_sources" in ALLOWED_SECTIONS
    assert "knowledge_sources" in config_schema()["sections"]
    res = validate_sections({"knowledge_sources": {
        "kb_wikidata_enabled": True, "kb_devdocs_url": "http://localhost:9292"}})
    assert res["ok"] is True
    assert not res["warnings"]  # all KB fields recognised, not "unknown"
