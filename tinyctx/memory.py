"""Optional mem0 wrapper for cross-session user/project memory.

Why optional: mem0ai pulls in heavy deps (vector store, embedding model).
Most tinyctx users don't need it; those who do should `pip install
tinyctx[mem]`. Everything in this module degrades cleanly when mem0 isn't
importable — it just reports "mem0 not installed" instead of crashing.

Usage pattern (when mem0 IS installed):

    from tinyctx.memory import MemStore
    m = MemStore()                        # uses local backend by default
    m.add("user prefers tabs over spaces", user_id="alice")
    hits = m.search("indentation preference", user_id="alice")

CLI:
    tinyctx-mem available                 # check whether mem0 is installed
    tinyctx-mem add "user prefers terse PR descriptions"
    tinyctx-mem search "code review style"
    tinyctx-mem ingest-compaction         # push facts from latest compaction
    tinyctx-mem stats

We deliberately do NOT auto-inject mem0 hits at SessionStart. The user
opts in by querying explicitly. This avoids polluting codex's prompt
prefix (which would hurt prompt-cache hit rate) and conflicting with
codex's own resume logic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def is_available() -> bool:
    try:
        import mem0  # noqa: F401
        return True
    except ImportError:
        return False


class MemStore:
    """Thin wrapper over `mem0.Memory`. Constructed lazily — raises
    ImportError with a clear message if mem0 isn't installed."""

    def __init__(
        self,
        *,
        local_base_url: str | None = None,
        local_model: str | None = None,
        store_dir: Path | None = None,
    ):
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mem0 not installed. Run: pip install 'tinyctx[mem]' "
                "(or `pip install mem0ai`)"
            ) from e

        # Default config: route LLM calls through whatever local
        # OpenAI-compat backend tinyctx is already configured for.
        base_url = local_base_url or os.environ.get(
            "TINYCTX_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1")
        model = local_model or os.environ.get(
            "TINYCTX_LOCAL_MODEL", "qwen3.6-27b")
        sdir = store_dir or (Path.home() / ".tinyctx" / "mem0")
        sdir.mkdir(parents=True, exist_ok=True)

        try:
            self._mem = Memory.from_config({
                "llm": {
                    "provider": "openai",
                    "config": {
                        "openai_base_url": base_url,
                        "model": model,
                    },
                },
                # Leave embedder + vector_store at mem0's defaults to
                # avoid forcing extra deps. Users who want fully-local
                # embeddings can override via mem0's own config.
            })
        except Exception:
            # Fall back to mem0's default-default Memory() — works without
            # any local backend if mem0's defaults are reachable.
            self._mem = Memory()

    # ------------------------------------------------------------ ops

    def add(self, text: str, *, user_id: str = "default",
            metadata: dict | None = None) -> Any:
        return self._mem.add(text, user_id=user_id, metadata=metadata or {})

    def search(self, query: str, *, user_id: str = "default",
               limit: int = 5) -> list[dict]:
        out = self._mem.search(query, user_id=user_id, limit=limit)
        # Normalize: mem0 returns either a list or {"results": [...]}.
        if isinstance(out, dict) and "results" in out:
            return list(out["results"])
        return list(out) if isinstance(out, list) else []

    def get_all(self, *, user_id: str = "default", limit: int = 100) -> list[dict]:
        out = self._mem.get_all(user_id=user_id, limit=limit)
        if isinstance(out, dict) and "results" in out:
            return list(out["results"])
        return list(out) if isinstance(out, list) else []


# ----------------------------------------------------------- CLI

def _cmd_available(_args) -> int:
    print("yes" if is_available() else "no (run: pip install 'tinyctx[mem]')")
    return 0 if is_available() else 1


def _cmd_add(args) -> int:
    if not is_available():
        sys.stderr.write("mem0 not installed; pip install 'tinyctx[mem]'\n")
        return 1
    m = MemStore()
    res = m.add(args.text, user_id=args.user_id)
    print(json.dumps({"added": True, "result": str(res)}))
    return 0


def _cmd_search(args) -> int:
    if not is_available():
        sys.stderr.write("mem0 not installed; pip install 'tinyctx[mem]'\n")
        return 1
    m = MemStore()
    hits = m.search(args.query, user_id=args.user_id, limit=args.limit)
    if not hits:
        sys.stderr.write("(no hits)\n")
        return 1
    for h in hits:
        text = h.get("memory") or h.get("text") or h.get("content") or str(h)
        score = h.get("score")
        prefix = f"[{score:.3f}] " if isinstance(score, (int, float)) else ""
        print(f"{prefix}{text}")
    return 0


def _cmd_stats(args) -> int:
    if not is_available():
        sys.stderr.write("mem0 not installed; pip install 'tinyctx[mem]'\n")
        return 1
    m = MemStore()
    items = m.get_all(user_id=args.user_id, limit=10_000)
    print(f"user_id: {args.user_id}")
    print(f"memories: {len(items)}")
    return 0


def _cmd_ingest_compaction(args) -> int:
    """Take the facts/compartments from the latest compaction and push them
    into mem0 as durable user/project memories."""
    if not is_available():
        sys.stderr.write("mem0 not installed; pip install 'tinyctx[mem]'\n")
        return 1
    from .continuity import latest_structured
    data = latest_structured(Path(args.root).resolve())
    if not data:
        sys.stderr.write("(no structured compaction; needs compactor_debate output)\n")
        return 1
    m = MemStore()
    n = 0
    for f in data.get("facts") or []:
        claim = (f.get("claim") or "").strip()
        if not claim:
            continue
        m.add(claim, user_id=args.user_id, metadata={"source": "compaction"})
        n += 1
    for c in data.get("compartments") or []:
        topic = (c.get("topic") or c.get("name") or "").strip()
        summary = (c.get("summary") or "").strip()
        if not topic and not summary:
            continue
        text = f"{topic}: {summary}".strip(": ").strip()
        m.add(text, user_id=args.user_id,
              metadata={"source": "compaction", "kind": "compartment"})
        n += 1
    print(f"ingested {n} entries into mem0 user_id={args.user_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.memory")
    p.add_argument("--user-id", default="default")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("available", help="check whether mem0 is installed")\
        .set_defaults(_fn=_cmd_available)

    p_add = sub.add_parser("add", help="add a memory")
    p_add.add_argument("text")
    p_add.set_defaults(_fn=_cmd_add)

    p_search = sub.add_parser("search", help="search memories")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.set_defaults(_fn=_cmd_search)

    p_stats = sub.add_parser("stats", help="memory stats")
    p_stats.set_defaults(_fn=_cmd_stats)

    p_ingest = sub.add_parser("ingest-compaction",
                              help="push facts from latest compaction")
    p_ingest.add_argument("--root", default=".")
    p_ingest.set_defaults(_fn=_cmd_ingest_compaction)

    args = p.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
