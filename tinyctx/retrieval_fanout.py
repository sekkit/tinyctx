"""Safe local retrieval fan-out primitives.

This module intentionally does not call external MCP/web providers.
It gives the proxy a deterministic merge/dedup/budget/injection layer;
runtime providers can be wired in behind explicit config later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Any


_BEGIN = "BEGIN TINYCTX RETRIEVAL FANOUT"
_END = "END TINYCTX RETRIEVAL FANOUT"
_SENSITIVE_PATH_RE = re.compile(
    r"(^|[/._-])("
    r"secret|secrets|credential|credentials|password|passwd|token|api[-_]?key|"
    r"private[-_]?key|id_rsa|id_dsa|id_ed25519"
    r")($|[/._-])",
    re.IGNORECASE,
)
_SENSITIVE_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "credentials.json",
    "secrets.json",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_SECRET_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"(?im)^\s*[\w.-]*(?:api[_-]?key|secret|password|credential|"
        r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|"
        r"bearer[_-]?token)"
        r"[\w.-]*\s*[:=]\s*\S+"
    ),
    re.compile(r"(?im)^\s*token\s*[:=]\s*\S+"),
)


@dataclass(frozen=True)
class RetrievalHit:
    source: str
    path: str
    snippet: str
    score: float = 0.0


Provider = Callable[[str], list[RetrievalHit]]


def _clean_mentioned_path(raw: str) -> str:
    rel = raw.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def is_sensitive_path(path: str) -> bool:
    normalized = (path or "").replace("\\", "/").strip().lower()
    if not normalized:
        return False
    parts = [p for p in normalized.split("/") if p]
    basename = parts[-1] if parts else normalized
    if basename in _SENSITIVE_BASENAMES or basename.startswith(".env."):
        return True
    if Path(basename).suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    return bool(_SENSITIVE_PATH_RE.search(normalized))


def contains_sensitive_text(text: str) -> bool:
    return any(pat.search(text or "") for pat in _SECRET_TEXT_PATTERNS)


def _safe_hit(hit: RetrievalHit) -> bool:
    return (
        not is_sensitive_path(hit.path)
        and not contains_sensitive_text(hit.snippet)
    )


def run_fanout(
    query: str,
    providers: list[Provider],
    *,
    top_k: int = 5,
    max_chars: int = 6000,
) -> list[RetrievalHit]:
    best: dict[tuple[str, str], tuple[RetrievalHit, set[str]]] = {}
    for provider in providers:
        try:
            hits = provider(query)
        except Exception:
            continue
        for hit in hits or []:
            if not isinstance(hit, RetrievalHit):
                continue
            key = (hit.path, hit.snippet.strip())
            prev = best.get(key)
            if prev is None:
                best[key] = (hit, {hit.source})
                continue
            prev_hit, sources = prev
            sources.add(hit.source)
            if hit.score > prev_hit.score:
                best[key] = (hit, sources)

    merged = [
        RetrievalHit(
            source=",".join(sorted(sources)),
            path=hit.path,
            snippet=hit.snippet,
            score=hit.score,
        )
        for hit, sources in best.values()
    ]
    ordered = sorted(merged, key=lambda h: h.score, reverse=True)
    out: list[RetrievalHit] = []
    used = 0
    for hit in ordered:
        cost = len(hit.source) + len(hit.path) + len(hit.snippet)
        if out and used + cost > max_chars:
            break
        if not out and cost > max_chars:
            snippet = hit.snippet[:max(0, max_chars - len(hit.source) - len(hit.path))]
            out.append(RetrievalHit(hit.source, hit.path, snippet, hit.score))
            break
        out.append(hit)
        used += cost
        if len(out) >= top_k:
            break
    return out


def inject_context(
    body: dict[str, Any],
    hits: list[RetrievalHit],
    *,
    reason: str,
) -> tuple[dict[str, Any], bool]:
    safe_hits = [hit for hit in hits if _safe_hit(hit)]
    if not safe_hits:
        return body, False
    instructions = body.get("instructions")
    if not isinstance(instructions, str):
        return body, False
    if _BEGIN in instructions:
        return body, False
    lines = [
        f"{_BEGIN}",
        f"Reason: {reason}",
        "",
    ]
    for hit in safe_hits:
        lines.extend([
            f"## {hit.path} ({hit.source}, score={hit.score:.2f})",
            hit.snippet.strip(),
            "",
        ])
    lines.append(_END)
    out = dict(body)
    out["instructions"] = "\n".join(lines).strip() + "\n\n" + instructions
    return out, True


def extract_query_text(body: dict[str, Any]) -> str:
    items = body.get("input") or body.get("messages") or []
    if not isinstance(items, list):
        return ""
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content[-2000:]
        if isinstance(content, list):
            for part in content:
                if (isinstance(part, dict)
                        and part.get("type") in ("text", "input_text")):
                    return str(part.get("text", ""))[-2000:]
    return ""


_PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z0-9_]+")


def mentioned_path_provider(
    root: str | Path,
    *,
    max_file_chars: int = 3000,
) -> Provider:
    base = Path(root)
    try:
        base_resolved = base.resolve()
    except OSError:
        base_resolved = base

    def _provider(query: str) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for raw in _PATH_RE.findall(query or ""):
            rel = _clean_mentioned_path(raw)
            if not rel or rel in seen or ".." in Path(rel).parts:
                continue
            if is_sensitive_path(rel):
                continue
            seen.add(rel)
            path = base / rel
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(base_resolved)
            except (OSError, ValueError):
                continue
            if not resolved.is_file():
                continue
            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if contains_sensitive_text(text):
                continue
            hits.append(RetrievalHit(
                source="mentioned_path",
                path=rel,
                snippet=text[:max_file_chars],
                score=1.0,
            ))
        return hits

    return _provider


def scout_cache_provider(
    getter: Callable[[str | Path | None], str | None] | None = None,
    *,
    root: str | Path | None = None,
    max_chars: int = 3000,
) -> Provider:
    if getter is None:
        from . import auto_scout
        getter = auto_scout.get_scout

    def _provider(query: str) -> list[RetrievalHit]:
        try:
            scout_md = getter(root)
        except Exception:
            return []
        if not scout_md:
            return []
        return [RetrievalHit(
            source="scout_cache",
            path="scout.md",
            snippet=scout_md[:max_chars],
            score=0.6,
        )]

    return _provider


def disabled_external_provider(name: str) -> Provider:
    def _provider(query: str) -> list[RetrievalHit]:
        return []
    _provider.__name__ = f"{name}_disabled_provider"
    return _provider


def default_providers(
    root: str | Path,
    *,
    include_external: bool = False,
) -> list[Provider]:
    providers: list[Provider] = [
        mentioned_path_provider(root),
        scout_cache_provider(root=root),
    ]
    if include_external:
        # Runtime MCP/web fan-out must remain explicit. The proxy process
        # cannot assume Serena/GitNexus/web credentials or latency budget.
        providers.extend([
            disabled_external_provider("serena"),
            disabled_external_provider("gitnexus"),
            disabled_external_provider("web"),
        ])
    return providers


def inject_for_disagreement(
    body: dict[str, Any],
    *,
    root: str | Path,
    top_k: int = 5,
    max_chars: int = 6000,
    extra_providers: "list[Provider] | None" = None,
) -> tuple[dict[str, Any], bool]:
    query = extract_query_text(body)
    if not query:
        return body, False
    providers = default_providers(root)
    if extra_providers:
        # Additive: external knowledge-source providers (knowledge_sources)
        # are appended only when the caller passes them. Default behaviour
        # (local-only fan-out) is unchanged.
        providers = providers + list(extra_providers)
    hits = run_fanout(
        query,
        providers,
        top_k=top_k,
        max_chars=max_chars,
    )
    return inject_context(
        body,
        hits,
        reason="self-consistency disagreement",
    )
