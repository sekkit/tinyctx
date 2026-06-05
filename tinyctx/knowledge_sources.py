"""External knowledge-source providers for retrieval_fanout.

Four read-only knowledge sources, each exposed as a retrieval_fanout
Provider — Callable[[str], list[RetrievalHit]] — so they plug straight into
the existing run_fanout / inject_context layer:

  * devdocs          — self-hosted DevDocs (offline, all-language official
                       API docs). Local HTTP. The 'big reference library'
                       for a coding agent; complements in-tree pydoc_mcp.
  * kiwix            — self-hosted kiwix-serve over ZIM snapshots (Stack
                       Overflow / Wikipedia offline). Local HTTP.
  * wikidata         — Wikidata wbsearchentities (entity / fact lookup,
                       CC0). Public API by default; point base_url at a
                       local Wikibase if you run one.

Design rules (mirroring retrieval_fanout's stance that external fan-out
must stay EXPLICIT):

  * Every source is OFF by default. external_providers(cfg) returns only
    the ones explicitly enabled via config attr or env var.
  * Every provider degrades to [] on ANY error (service down, timeout, bad
    payload, parse failure) — never raises, never blocks beyond timeout_s.
  * All network I/O goes through an injectable `_get` seam, so tests run
    fully offline and deterministic. `_get(url, params) -> (status, text)`.
  * Results are RetrievalHit; retrieval_fanout.inject_context re-filters
    snippets through its secret guard (defence in depth).
  * DevDocs index.json is cached under ~/.tinyctx/cache/kb_sources/devdocs/.

This module imports retrieval_fanout lazily (inside the provider closures)
so there is no import cycle and retrieval_fanout stays a pure local module
with no dependency on this one. Wiring into the proxy is additive: pass
external_providers(cfg) as `extra_providers` to inject_for_disagreement.

CLI (manual verification without touching the proxy):
    tinyctx-kb-sources devdocs  "os.path.join" --url http://localhost:9292 --slug python~3.12
    tinyctx-kb-sources kiwix    "segfault"     --url http://localhost:8080
    tinyctx-kb-sources wikidata "Ada Lovelace"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


# Provider type matches retrieval_fanout.Provider without importing it.
Getter = Callable[[str, dict], Tuple[int, str]]

WIKIDATA_API_DEFAULT = "https://www.wikidata.org/w/api.php"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _tokens(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def _cache_root() -> Path:
    override = os.environ.get("TINYCTX_KB_SOURCES_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tinyctx" / "cache" / "kb_sources"


# ----------------------------------------------------------------- http seam

_DEFAULT_UA = "tinyctx/0.7 (knowledge-source provider; local-first codex proxy)"


def _http_get(url: str, params: dict, *, timeout_s: float,
              headers: Optional[dict] = None) -> Tuple[int, str]:
    """Default HTTP getter. Returns (status_code, text); (0, "") on any
    transport error. Never raises. Always sends a descriptive User-Agent —
    some services (Wikimedia/Wikidata) reject requests that lack one (403)."""
    try:
        import httpx

        h = {"User-Agent": _DEFAULT_UA}
        if headers:
            h.update(headers)
        r = httpx.get(url, params=params, timeout=timeout_s,
                      headers=h, follow_redirects=True)
        return r.status_code, r.text
    except Exception:
        return 0, ""


def _bind_getter(_get: Optional[Getter], *, timeout_s: float,
                 headers: Optional[dict] = None) -> Getter:
    if _get is not None:
        return _get
    return lambda url, params: _http_get(url, params, timeout_s=timeout_s,
                                         headers=headers)


def _make_hit(source: str, path: str, snippet: str, score: float):
    """Build a retrieval_fanout.RetrievalHit (imported lazily)."""
    from .retrieval_fanout import RetrievalHit

    return RetrievalHit(source=source, path=path, snippet=snippet, score=score)


def _rank_score(i: int, top_k: int, *, hi: float = 0.78, lo: float = 0.5) -> float:
    """Deterministic descending score in [lo, hi] by result rank. External
    knowledge ranks below a direct file match (mentioned_path = 1.0)."""
    if top_k <= 1:
        return hi
    return round(hi - (hi - lo) * (i / max(1, top_k - 1)), 4)


# -------------------------------------------------------------------- devdocs

def _devdocs_index(base_url: str, slug: str, getter: Getter,
                   *, ttl_s: float = 86400.0) -> list:
    """Fetch + cache a DevDocs docset index.json. Returns its `entries`
    list ([{name, path, type}, ...]) or [] on failure."""
    cache = _cache_root() / "devdocs"
    safe_slug = re.sub(r"[^A-Za-z0-9_.~-]+", "_", slug)
    cpath = cache / (safe_slug + ".json")
    now = time.time()
    if cpath.is_file():
        try:
            if now - cpath.stat().st_mtime < ttl_s:
                return json.loads(cpath.read_text(encoding="utf-8")).get("entries") or []
        except (OSError, ValueError):
            pass
    url = base_url.rstrip("/") + "/docs/" + slug + "/index.json"
    status, text = getter(url, {})
    if status != 200 or not text:
        # serve a stale cache if we have one
        if cpath.is_file():
            try:
                return json.loads(cpath.read_text(encoding="utf-8")).get("entries") or []
            except (OSError, ValueError):
                return []
        return []
    try:
        entries = json.loads(text).get("entries") or []
    except (ValueError, AttributeError):
        return []
    try:
        cache.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    except OSError:
        pass
    return entries


def devdocs_provider(
    base_url: str,
    slugs: list,
    *,
    top_k: int = 5,
    timeout_s: float = 4.0,
    _get: Optional[Getter] = None,
):
    """Provider over a self-hosted DevDocs instance. DevDocs has no server
    search API, so we fetch+cache each docset's index.json and match query
    tokens against entry names locally, returning the matched entries as
    pointers (name / type / URL)."""
    getter = _bind_getter(_get, timeout_s=timeout_s)
    slugs = [s for s in (slugs or []) if s]

    def _provider(query: str) -> list:
        q_terms = set(_tokens(query))
        if not q_terms or not base_url:
            return []
        scored: list = []
        for slug in slugs:
            for entry in _devdocs_index(base_url, slug, getter):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or ""
                etoks = set(_tokens(name))
                if not etoks:
                    continue
                overlap = q_terms & etoks
                if not overlap:
                    continue
                # prefer exact/substring; score by overlap fraction + bonus
                frac = len(overlap) / len(etoks)
                exact = 1.0 if (query or "").strip().lower() in name.lower() else 0.0
                scored.append((frac + exact, slug, entry))
        if not scored:
            return []
        scored.sort(key=lambda t: t[0], reverse=True)
        hits: list = []
        for i, (_, slug, entry) in enumerate(scored[:top_k]):
            name = entry.get("name") or ""
            etype = entry.get("type") or ""
            ref = base_url.rstrip("/") + "/" + slug + "/" + str(entry.get("path") or "")
            snippet = "%s  [%s/%s]\n%s" % (name, slug, etype, ref)
            hits.append(_make_hit("devdocs:" + slug, slug + "/" + str(entry.get("path") or ""),
                                  snippet, _rank_score(i, top_k)))
        return hits

    _provider.__name__ = "devdocs_provider"
    return _provider


# ---------------------------------------------------------------------- kiwix

def _parse_kiwix(text: str) -> list:
    """Best-effort parse of a kiwix-serve /search response. Tries JSON first
    (newer builds), then OpenSearch/RSS XML. Returns [{title, link, desc}]."""
    text = (text or "").strip()
    if not text:
        return []
    # JSON attempt
    if text[:1] in "{[":
        try:
            data = json.loads(text)
            results = data.get("results") if isinstance(data, dict) else data
            out: list = []
            for r in results or []:
                if isinstance(r, dict):
                    out.append({
                        "title": r.get("title") or r.get("label") or "",
                        "link": r.get("url") or r.get("link") or r.get("path") or "",
                        "desc": r.get("snippet") or r.get("description") or "",
                    })
            if out:
                return out
        except (ValueError, AttributeError):
            pass
    # XML attempt (OpenSearch RSS: channel/item with title/link/description)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1].lower()
        if tag in ("item", "entry"):
            fields: dict = {}
            for child in el:
                ctag = child.tag.rsplit("}", 1)[-1].lower()
                fields[ctag] = (child.text or "").strip() or fields.get(ctag, "")
                if ctag == "link" and not (child.text or "").strip():
                    fields["link"] = child.get("href", "") or fields.get("link", "")
            out.append({
                "title": fields.get("title", ""),
                "link": fields.get("link", "") or fields.get("id", ""),
                "desc": fields.get("description", "") or fields.get("summary", ""),
            })
    return out


def kiwix_provider(
    base_url: str,
    *,
    books: Optional[list] = None,
    top_k: int = 5,
    timeout_s: float = 4.0,
    _get: Optional[Getter] = None,
):
    """Provider over a self-hosted kiwix-serve full-text search."""
    getter = _bind_getter(_get, timeout_s=timeout_s)
    url = base_url.rstrip("/") + "/search" if base_url else ""

    def _provider(query: str) -> list:
        q = (query or "").strip()
        if not q or not base_url:
            return []
        params: dict = {"pattern": q, "pageLength": top_k, "format": "xml"}
        if books:
            params["books.name"] = books[0] if len(books) == 1 else books
        status, text = getter(url, params)
        if status != 200:
            return []
        results = _parse_kiwix(text)
        hits: list = []
        for i, r in enumerate(results[:top_k]):
            title = r.get("title") or ""
            if not title:
                continue
            link = r.get("link") or ""
            desc = re.sub(r"<[^>]+>", "", r.get("desc") or "")  # strip HTML
            snippet = "\n".join(x for x in [title, desc[:400]] if x)
            hits.append(_make_hit("kiwix", link or title, snippet, _rank_score(i, top_k)))
        return hits

    _provider.__name__ = "kiwix_provider"
    return _provider


# ------------------------------------------------------------------- wikidata

_PROPER_NOUN_RE = re.compile(r"[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*")


def _entity_terms(query: str) -> list:
    """wbsearchentities matches entity LABELS, not free text. Short query →
    search as-is; sentence → pull proper-noun-ish spans ('Ada Lovelace') so
    'who was Ada Lovelace' still resolves. Falls back to the whole query."""
    q = (query or "").strip()
    if not q:
        return []
    if len(q.split()) <= 3:
        return [q]
    cands = _PROPER_NOUN_RE.findall(q)
    return cands[:2] if cands else [q]


def _rerank_by_context(query: str, candidates: list) -> list:
    """Disambiguate same-name entities by query context.

    wbsearchentities matches LABELS and ranks by Wikidata's own popularity,
    ignoring the query's domain — so 'Transformer' returns the electrical
    device / the media franchise, not the ML model. We re-rank candidates by
    how well each entity's *description* overlaps the query's context words.
    Stable sort: entities with equal overlap keep wbsearchentities' original
    relevance order, so this only *promotes* a contextually-better match and
    never degrades the no-context case.
    """
    ctx = set(_tokens(query))
    if not ctx:
        return list(candidates)

    def overlap(e: dict) -> int:
        # Match against description (the distinguishing text); labels are the
        # same across same-name candidates, so they carry no signal here.
        return len(ctx & set(_tokens(e.get("description") or "")))

    return sorted(candidates, key=overlap, reverse=True)


def wikidata_provider(
    *,
    base_url: str = WIKIDATA_API_DEFAULT,
    language: str = "en",
    top_k: int = 5,
    timeout_s: float = 6.0,
    _get: Optional[Getter] = None,
):
    """Provider over Wikidata wbsearchentities (entity/fact lookup, CC0).

    Two-stage to handle same-name ambiguity: (1) extract entity name(s) from
    the query and over-fetch candidates; (2) re-rank by query context so the
    domain-correct sense wins (see _rerank_by_context)."""
    getter = _bind_getter(_get, timeout_s=timeout_s)
    fetch_limit = max(top_k, 10)  # over-fetch: the right sense is often not #1

    def _search(term: str) -> list:
        status, text = getter(base_url, {
            "action": "wbsearchentities",
            "search": term,
            "format": "json",
            "language": language,
            "uselang": language,
            "type": "item",
            "limit": fetch_limit,
        })
        if status != 200 or not text:
            return []
        try:
            return json.loads(text).get("search") or []
        except (ValueError, AttributeError):
            return []

    def _provider(query: str) -> list:
        candidates: list = []
        seen: set = set()
        for term in _entity_terms(query):
            for e in _search(term):
                if not isinstance(e, dict):
                    continue
                qid = e.get("id") or ""
                label = e.get("label") or ""
                if not (qid or label):
                    continue
                key = qid or label
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(e)

        ranked = _rerank_by_context(query, candidates)
        hits: list = []
        for i, e in enumerate(ranked[:top_k]):
            qid = e.get("id") or ""
            label = e.get("label") or ""
            desc = e.get("description") or ""
            snippet = "%s (%s): %s" % (label, qid, desc)
            path = e.get("concepturi") or e.get("url") or qid
            hits.append(_make_hit("wikidata", path, snippet, _rank_score(i, top_k)))
        return hits

    _provider.__name__ = "wikidata_provider"
    return _provider


# --------------------------------------------------------- config-gated build

def _flag(cfg: Any, attr: str, env: str, default: bool = False) -> bool:
    v = getattr(cfg, attr, None) if cfg is not None else None
    if v is not None:
        return bool(v)
    e = os.environ.get(env)
    if e is not None:
        return e.strip().lower() in ("1", "true", "yes", "on")
    return default


def _val(cfg: Any, attr: str, env: str, default: Any) -> Any:
    v = getattr(cfg, attr, None) if cfg is not None else None
    if v:
        return v
    return os.environ.get(env, default)


def external_providers(cfg: Any = None) -> list:
    """Build the list of ENABLED external providers from config/env.

    Everything is off unless explicitly enabled. Safe to call with cfg=None
    or a Config that lacks these attrs (getattr defaults to off), so this
    adds zero behavior until a user opts in. Pass the result as
    `extra_providers` to retrieval_fanout.inject_for_disagreement.
    """
    providers: list = []

    if _flag(cfg, "kb_devdocs_enabled", "TINYCTX_KB_DEVDOCS"):
        base = _val(cfg, "kb_devdocs_url", "TINYCTX_KB_DEVDOCS_URL", "")
        slugs_raw = _val(cfg, "kb_devdocs_slugs", "TINYCTX_KB_DEVDOCS_SLUGS", "")
        slugs = slugs_raw if isinstance(slugs_raw, list) else [
            s.strip() for s in str(slugs_raw).split(",") if s.strip()]
        if base and slugs:
            providers.append(devdocs_provider(base, slugs))

    if _flag(cfg, "kb_kiwix_enabled", "TINYCTX_KB_KIWIX"):
        base = _val(cfg, "kb_kiwix_url", "TINYCTX_KB_KIWIX_URL", "")
        books_raw = _val(cfg, "kb_kiwix_books", "TINYCTX_KB_KIWIX_BOOKS", "")
        books = books_raw if isinstance(books_raw, list) else [
            s.strip() for s in str(books_raw).split(",") if s.strip()]
        if base:
            providers.append(kiwix_provider(base, books=books or None))

    if _flag(cfg, "kb_wikidata_enabled", "TINYCTX_KB_WIKIDATA"):
        base = _val(cfg, "kb_wikidata_url", "TINYCTX_KB_WIKIDATA_URL", WIKIDATA_API_DEFAULT)
        lang = _val(cfg, "kb_wikidata_language", "TINYCTX_KB_WIKIDATA_LANG", "en")
        providers.append(wikidata_provider(base_url=base, language=lang))

    return providers


# --------------------------------------------------------------------- CLI

_SOURCES = {
    "devdocs": lambda a: devdocs_provider(a.url or "", _split(a.slug), top_k=a.top_k),
    "kiwix": lambda a: kiwix_provider(a.url or "", books=_split(a.book) or None, top_k=a.top_k),
    "wikidata": lambda a: wikidata_provider(
        base_url=a.url or WIKIDATA_API_DEFAULT, top_k=a.top_k),
}


def _split(v: Optional[str]) -> list:
    return [s.strip() for s in (v or "").split(",") if s.strip()]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tinyctx-kb-sources",
        description="Query one external knowledge source (manual check).")
    parser.add_argument("source", choices=sorted(_SOURCES))
    parser.add_argument("query")
    parser.add_argument("--url", default=None, help="base_url for devdocs/kiwix/wikidata")
    parser.add_argument("--slug", default=None, help="devdocs docset slug(s), comma-separated")
    parser.add_argument("--book", default=None, help="kiwix book name(s), comma-separated")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    args = parser.parse_args(argv)

    provider = _SOURCES[args.source](args)
    hits = provider(args.query)
    if not hits:
        sys.stderr.write("(no hits — source disabled, unreachable, or empty)\n")
        return 1
    for h in hits:
        head = h.snippet.splitlines()[0][:100] if h.snippet else ""
        print("%.3f  [%s]  %s" % (h.score, h.source, head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
