"""ctx_pack: preemptively inject key project files into request context.

Problem: codex calls ctx_execute_file for every file it needs to read,
accumulating 1-3s per call. With many files this adds latency and can hit
the 120s tool call timeout (seen in Understand-Anything analysis task).

Solution: on the first request of a session, read the top-K files ranked
by compression-biased PageRank (interest.py) from graph.json, and inject
their contents as a formatted markdown block into body.instructions.

One request replaces N tool calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import interest

_BEGIN_MARKER = "<!-- tinyctx ctx-pack BEGIN -->"
_END_MARKER = "<!-- tinyctx ctx-pack END -->"

# Configurable via env (overridable in config.toml later)
DEFAULT_MAX_FILES = 8
DEFAULT_MAX_TOTAL_CHARS = 32000


def build_pack(
    graph_path: Path,
    project_root: Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str | None:
    """Read top-K files from graph.json PageRank ranking, return a markdown
    block suitable for injection into instructions.  Returns None when no
    graph exists or no files could be read."""
    if not graph_path.is_file():
        return None
    try:
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    raw_nodes = graph_data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None

    # Build a node_id -> source_file mapping from the raw graphify output.
    # The interest.py pipeline expects wrapped_signature/wrapped_body/deps
    # which graphify doesn't produce; we use the raw source_file field as
    # fallback and rank by file size on disk.
    id_to_file: dict[str, str] = {}
    file_weights: dict[str, float] = {}
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id", "")
        sf = n.get("source_file", "")
        if not isinstance(nid, str) or not nid or not isinstance(sf, str) or not sf:
            continue
        id_to_file[nid] = sf
        # Weight heuristic: count occurrences in graph edges as a cheap
        # proxy for importance (don't need full PageRank).
        file_weights[sf] = file_weights.get(sf, 0.0) + 1.0

    # Also count incoming link targets
    links = graph_data.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            target = link.get("target", "")
            if target in id_to_file:
                sf = id_to_file[target]
                file_weights[sf] = file_weights.get(sf, 0.0) + 1.5

    # Sort by weight descending, deduplicate
    seen: set[str] = set()
    ranked_files: list[tuple[str, float]] = []
    for sf, weight in sorted(file_weights.items(), key=lambda x: x[1], reverse=True):
        norm = sf.replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)
        ranked_files.append((sf, weight))

    if not ranked_files:
        return None

    blocks: list[str] = []
    total_chars = 0
    for file_path, _weight in ranked_files[:max_files]:
        full_path = project_root / file_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        room = max_total_chars - total_chars
        if room < 200:
            break
        if len(content) > room:
            content = content[:room] + "\n-- truncated --"
        rel = str(file_path).replace("\\", "/")
        blocks.append(
            f"### `{rel}`\n"
            f"```{_lang_hint(rel)}\n"
            f"{content}\n"
            f"```"
        )
        total_chars += len(content)

    if not blocks:
        return None
    return "\n\n".join(blocks)


def inject_pack(body: dict[str, Any], pack_md: str) -> dict[str, Any]:
    """Inject a ctx_pack block into body.instructions.  Idempotent —
    if the pack is already present, the body is returned unchanged."""
    inst = body.get("instructions") or ""
    if not isinstance(inst, str):
        inst = str(inst)
    if _BEGIN_MARKER in inst:
        return body  # already injected

    block = (
        f"\n\n{_BEGIN_MARKER}\n"
        f"# Key project files (auto-injected by tinyctx ctx-pack)\n"
        f"{pack_md}\n"
        f"{_END_MARKER}\n\n"
    )
    new_body = dict(body)
    new_body["instructions"] = block + inst
    return new_body


def _extract_file_path(node_id: str) -> str | None:
    """Extract a file path from a graphify node id like 'src/foo.py:Foo.bar'."""
    if ":" in node_id:
        return node_id.split(":")[0]
    return node_id


def _lang_hint(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python", "pyi": "python",
        "ts": "typescript", "tsx": "tsx", "js": "javascript", "jsx": "jsx",
        "go": "go", "rs": "rust", "java": "java", "kt": "kotlin",
        "c": "c", "cc": "cpp", "cpp": "cpp", "h": "c", "hpp": "cpp",
        "cs": "csharp", "rb": "ruby", "php": "php", "swift": "swift",
        "sh": "bash", "bash": "bash", "toml": "toml", "yaml": "yaml",
        "yml": "yaml", "json": "json", "md": "markdown", "sql": "sql",
        "html": "html", "css": "css", "scss": "scss",
        "dockerfile": "dockerfile", "makefile": "makefile",
    }.get(ext, "")


def _fallback_rank(nodes: list[dict], top_k: int) -> list[tuple[str, float]]:
    """Fallback ranking by node size (largest first) when PageRank fails."""
    scored: list[tuple[str, float]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id", "")
        if not isinstance(nid, str) or not nid:
            continue
        sig = int(n.get("wrapped_signature", 0) or 0)
        body = int(n.get("wrapped_body", 0) or 0)
        scored.append((nid, float(sig + body)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
