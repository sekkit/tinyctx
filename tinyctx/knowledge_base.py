"""Scoped knowledge base: per-scope local document retrieval for tinyctx.

Design rationale (full writeup in docs/knowledge-base-design.md):

  * A "scope" is any string id — a project root, a topic, a task. Each
    scope gets an ISOLATED store under ~/.tinyctx/cache/kb/<scope_hash>/,
    mirroring scout's per-repo cache discipline (scout.repo_hash + the
    ~/.tinyctx/cache/<hash>/ layout). Scoping — not one global "knowledge
    dump" — is deliberate: a codex agent wants the *current task's*
    documents, and a global index mostly adds retrieval noise that hurts
    accuracy.

  * Ingest: convert a file to text (markitdown if installed, else read it
    as plain text), chunk it, and persist the chunks as JSON. PDF / docx /
    pptx support is OPTIONAL and degrades cleanly to .md/.txt when
    markitdown isn't importable — exactly how tinyctx.memory degrades
    without mem0. No hard dependency is added to the default install.

  * Search: deterministic, dependency-free IDF-weighted bag-of-words
    scoring over the scope's chunks. No embeddings, no graph, no network —
    safe to run inline in the proxy and fully testable offline.

  * Provider: knowledge_base_provider(scope) adapts search() to the
    retrieval_fanout Provider contract — Callable[[str], list[RetrievalHit]]
    — so the KB plugs into the EXISTING fan-out merge/dedup/budget/inject
    layer (tinyctx.retrieval_fanout) with ZERO changes to that module.

Why bag-of-words first, not MiniRAG graph-RAG: keep the default path
local, zero-dependency, deterministic, and reversible. A MiniRAG /
LightRAG graph backend is a documented, OPT-IN upgrade (the
`tinyctx[kb-graph]` extra), NOT wired here — it pulls heavy deps and its
extraction quality on a 27B local model is unverified. See the design
doc's "Phase 2" before reaching for it.

Secret safety: ingestion skips sensitive files and drops chunks that look
like they carry secrets, reusing retrieval_fanout's is_sensitive_path /
contains_sensitive_text guards, so keys never enter the store; the
provider re-filters on the way out as defence in depth.

CLI:
    tinyctx-kb ingest <scope> <path>   [--doc-id ID]
    tinyctx-kb search <scope> <query>  [--top-k 5]
    tinyctx-kb list   <scope>
    tinyctx-kb remove <scope> <doc-id>
    tinyctx-kb stats  <scope>

Env vars:
    TINYCTX_KB_DIR        # override cache root (default ~/.tinyctx/cache/kb)
    TINYCTX_KB_MAX_CHARS  # chunk size in chars (default 1200)
    TINYCTX_KB_OVERLAP    # chunk overlap in chars (default 150)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


CACHE_VERSION = 1

# English words as whole tokens; each CJK ideograph as its own token so
# Chinese text still produces overlap signal without a real segmenter.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------- cache paths
# Read env at call time (not import time) so tests can redirect the store
# via TINYCTX_KB_DIR without reimporting the module.

def _kb_root() -> Path:
    override = os.environ.get("TINYCTX_KB_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tinyctx" / "cache" / "kb"


def scope_hash(scope: str) -> str:
    return hashlib.sha256(str(scope).encode("utf-8")).hexdigest()[:16]


def kb_dir(scope: str) -> Path:
    return _kb_root() / scope_hash(scope)


def store_path(scope: str) -> Path:
    return kb_dir(scope) / "store.json"


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# ------------------------------------------------------------------- records

@dataclass
class IngestResult:
    scope: str
    doc_id: str
    n_chunks: int
    skipped: str = ""  # non-empty reason means nothing was stored


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    ordinal: int
    text: str
    score: float


# -------------------------------------------------------------- optional deps

def markitdown_available() -> bool:
    """True if Microsoft markitdown is importable (enables PDF/docx/pptx).

    Absence is fine: convert_to_text() falls back to reading the file as
    UTF-8 text, so .md / .txt / source files always work.
    """
    try:
        import markitdown  # noqa: F401
        return True
    except Exception:
        return False


def convert_to_text(
    path: Path,
    *,
    _converter: Optional[Callable[[Path], str]] = None,
) -> str:
    """Convert a file to plain text/markdown.

    `_converter` is an injection seam for tests (and a future markitdown
    wrapper). When omitted: use markitdown if importable, else read the
    file as UTF-8 with errors replaced.
    """
    if _converter is not None:
        return _converter(path)
    if markitdown_available():
        try:
            from markitdown import MarkItDown  # type: ignore

            md = MarkItDown(enable_plugins=False)
            return md.convert(str(path)).text_content
        except Exception:
            pass  # fall through to plain read
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------- chunking

def chunk_text(
    text: str,
    *,
    max_chars: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list:
    """Split text into overlapping char windows, preferring to break on a
    newline / sentence / space boundary near the window end. Deterministic
    and progress-guaranteed (never loops)."""
    if max_chars is None:
        max_chars = _int_env("TINYCTX_KB_MAX_CHARS", 1200)
    if overlap is None:
        overlap = _int_env("TINYCTX_KB_OVERLAP", 150)
    max_chars = max(1, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars - 1))

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list = []
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chars
        if end >= n:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break
        window = text[start:end]
        # Prefer a clean boundary in the second half of the window.
        brk = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
        if brk < max_chars // 2:
            brk = len(window)  # no good boundary -> hard cut
        piece = text[start:start + brk].strip()
        if piece:
            chunks.append(piece)
        advance = brk - overlap
        if advance <= 0:
            advance = max_chars  # guarantee forward progress
        start += advance
    return chunks


def _tokens(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


# ------------------------------------------------------------------ store io

def _load_store(scope: str) -> dict:
    p = store_path(scope)
    if not p.is_file():
        return {"version": CACHE_VERSION, "scope": scope, "docs": {}, "chunks": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "scope": scope, "docs": {}, "chunks": []}
    data.setdefault("docs", {})
    data.setdefault("chunks", [])
    return data


def _save_store(scope: str, store: dict) -> None:
    d = kb_dir(scope)
    d.mkdir(parents=True, exist_ok=True)
    store["version"] = CACHE_VERSION
    store["scope"] = scope
    tmp = store_path(scope).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    tmp.replace(store_path(scope))


# -------------------------------------------------------------------- ingest

def ingest(
    scope: str,
    *,
    path: Optional[Path] = None,
    text: Optional[str] = None,
    doc_id: Optional[str] = None,
    _converter: Optional[Callable[[Path], str]] = None,
) -> IngestResult:
    """Ingest a file (`path`) or raw `text` into the scope's store.

    Re-ingesting the same doc_id replaces its chunks (incremental update).
    Sensitive files and secret-looking chunks are dropped, never stored.
    """
    from .retrieval_fanout import contains_sensitive_text, is_sensitive_path

    if path is not None:
        path = Path(path)
        if is_sensitive_path(str(path)):
            return IngestResult(scope, doc_id or path.name, 0, skipped="sensitive path")
        source = str(path)
        fhash = _file_hash(path)
        if doc_id is None:
            doc_id = path.name
        if text is None:
            text = convert_to_text(path, _converter=_converter)
    else:
        source = "inline"
        fhash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
        if doc_id is None:
            doc_id = "inline-" + fhash

    raw_chunks = chunk_text(text or "")
    safe_chunks = [c for c in raw_chunks if not contains_sensitive_text(c)]
    if not safe_chunks:
        return IngestResult(scope, doc_id, 0, skipped="empty or all-sensitive")

    store = _load_store(scope)
    # Drop any prior chunks for this doc_id (incremental replace).
    store["chunks"] = [c for c in store["chunks"] if c.get("doc_id") != doc_id]
    for i, chunk in enumerate(safe_chunks):
        store["chunks"].append({"doc_id": doc_id, "ordinal": i, "text": chunk})
    store["docs"][doc_id] = {
        "source": source,
        "file_hash": fhash,
        "n_chunks": len(safe_chunks),
        "ingested_at": round(time.time(), 3),
    }
    _save_store(scope, store)
    return IngestResult(scope, doc_id, len(safe_chunks))


def remove(scope: str, doc_id: str) -> bool:
    store = _load_store(scope)
    if doc_id not in store["docs"]:
        return False
    store["docs"].pop(doc_id, None)
    store["chunks"] = [c for c in store["chunks"] if c.get("doc_id") != doc_id]
    _save_store(scope, store)
    return True


def list_docs(scope: str) -> list:
    store = _load_store(scope)
    return [{"doc_id": k, **v} for k, v in sorted(store["docs"].items())]


def stats(scope: str) -> dict:
    store = _load_store(scope)
    return {
        "scope": scope,
        "scope_hash": scope_hash(scope),
        "dir": str(kb_dir(scope)),
        "docs": len(store["docs"]),
        "chunks": len(store["chunks"]),
        "markitdown": markitdown_available(),
    }


# -------------------------------------------------------------------- search

def search(scope: str, query: str, *, top_k: int = 5) -> list:
    """Deterministic IDF-weighted bag-of-words retrieval over the scope.

    Rare query terms dominate the score (high IDF); long chunks are length-
    normalised so they don't win by sheer size. Returns SearchHit list with
    scores normalised into (0, 0.85] so KB hits rank below a direct file
    match (mentioned_path scores 1.0) but can rise above the coarse
    scout_cache snippet (0.6).
    """
    store = _load_store(scope)
    chunks = store.get("chunks", [])
    if not chunks:
        return []
    q_terms = set(_tokens(query))
    if not q_terms:
        return []

    # Document frequency per token across chunks (chunks are our "documents").
    n_docs = len(chunks)
    chunk_tok_sets: list = []
    df: dict = {}
    for c in chunks:
        toks = _tokens(c.get("text", ""))
        tset = set(toks)
        chunk_tok_sets.append((toks, tset))
        for t in tset:
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        return math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1.0

    scored: list = []
    for c, (toks, tset) in zip(chunks, chunk_tok_sets):
        overlap = q_terms & tset
        if not overlap:
            continue
        raw = sum(idf(t) for t in overlap) / math.sqrt(len(toks) + 1)
        scored.append((raw, c))
    if not scored:
        return []

    max_raw = max(r for r, _ in scored)
    scored.sort(key=lambda rc: rc[0], reverse=True)
    out: list = []
    for raw, c in scored[: max(1, top_k)]:
        norm = (raw / max_raw) * 0.85 if max_raw > 0 else 0.0
        out.append(SearchHit(
            doc_id=c.get("doc_id", ""),
            ordinal=int(c.get("ordinal", 0)),
            text=c.get("text", ""),
            score=round(norm, 4),
        ))
    return out


# ---------------------------------------------------------------- provider

def knowledge_base_provider(
    scope: str,
    *,
    top_k: int = 5,
    max_snippet_chars: int = 1200,
) -> Callable[[str], list]:
    """Adapt search() to the retrieval_fanout Provider contract:
    Callable[[str], list[RetrievalHit]].

    Pass the returned callable to retrieval_fanout.run_fanout(...) alongside
    the existing providers. Self-contained: imports RetrievalHit lazily so
    there is no import cycle and the KB module loads standalone.
    """
    def _provider(query: str) -> list:
        from .retrieval_fanout import RetrievalHit, contains_sensitive_text

        hits = search(scope, query, top_k=top_k)
        out: list = []
        for h in hits:
            snippet = h.text[:max_snippet_chars]
            if contains_sensitive_text(snippet):
                continue
            out.append(RetrievalHit(
                source="knowledge_base",
                path=h.doc_id,
                snippet=snippet,
                score=h.score,
            ))
        return out

    _provider.__name__ = "knowledge_base_provider"
    return _provider


# --------------------------------------------------------------------- CLI

def _cmd_ingest(args: argparse.Namespace) -> int:
    p = Path(args.path)
    if not p.is_file():
        sys.stderr.write("no such file: %s\n" % p)
        return 1
    res = ingest(args.scope, path=p, doc_id=args.doc_id)
    if res.skipped:
        sys.stderr.write("skipped (%s)\n" % res.skipped)
        return 1
    print(json.dumps({"doc_id": res.doc_id, "chunks": res.n_chunks}))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    hits = search(args.scope, args.query, top_k=args.top_k)
    if not hits:
        sys.stderr.write("(no hits)\n")
        return 1
    for h in hits:
        head = h.text.strip().splitlines()[0][:100] if h.text.strip() else ""
        print("%.3f  %s#%d  %s" % (h.score, h.doc_id, h.ordinal, head))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    docs = list_docs(args.scope)
    if not docs:
        sys.stderr.write("(empty scope)\n")
        return 1
    for d in docs:
        print("%s  chunks=%s  source=%s" % (d["doc_id"], d.get("n_chunks"), d.get("source")))
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    ok = remove(args.scope, args.doc_id)
    print(json.dumps({"removed": ok, "doc_id": args.doc_id}))
    return 0 if ok else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(stats(args.scope), ensure_ascii=False))
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyctx-kb", description="Scoped local knowledge base")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a file into a scope")
    p_ing.add_argument("scope")
    p_ing.add_argument("path")
    p_ing.add_argument("--doc-id", dest="doc_id", default=None)
    p_ing.set_defaults(func=_cmd_ingest)

    p_se = sub.add_parser("search", help="search a scope")
    p_se.add_argument("scope")
    p_se.add_argument("query")
    p_se.add_argument("--top-k", dest="top_k", type=int, default=5)
    p_se.set_defaults(func=_cmd_search)

    p_ls = sub.add_parser("list", help="list docs in a scope")
    p_ls.add_argument("scope")
    p_ls.set_defaults(func=_cmd_list)

    p_rm = sub.add_parser("remove", help="remove a doc from a scope")
    p_rm.add_argument("scope")
    p_rm.add_argument("doc_id")
    p_rm.set_defaults(func=_cmd_remove)

    p_st = sub.add_parser("stats", help="show scope stats")
    p_st.add_argument("scope")
    p_st.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
