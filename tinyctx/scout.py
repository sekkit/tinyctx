"""Project-context scout: a subagent-style local-model scan that produces a
small, byte-stable hierarchical summary of a repo's load-bearing symbols.

Two-layer design (to avoid the wasteful "summarize the whole repo at install
time" pattern):

    layer 1 (free, no LLM):  graphify or aider-style static graph build.
                             Already covered by tinyctx.graphify_adapter +
                             tinyctx.interest. Outputs a graph.json with
                             dependency edges and wrapped lengths.

    layer 2 (this module):   take only the top-K nodes ranked by
                             compression-biased PageRank (tinyctx.interest),
                             feed their source to a local 27B model via an
                             OpenAI-compatible /chat/completions endpoint,
                             and persist a 1-2K-token summary keyed by the
                             repo path. Cache invalidates when any scanned
                             file's content hash changes.

The output is intentionally byte-stable across rebuilds (sorted node order,
deterministic prompt, low temperature) so it can be safely glued into a
codex AGENTS.md preamble without breaking prompt cache.

CLI:
    python -m tinyctx.scout init    [--graph PATH] [--root .] [--top-k 40]
    python -m tinyctx.scout refresh [--root .]              # rebuild if stale
    python -m tinyctx.scout status  [--root .]              # show cache state
    python -m tinyctx.scout show    [--root .]              # print scout.md
    python -m tinyctx.scout path    [--root .]              # print cache path

Env vars (override config):
    TINYCTX_LOCAL_BASE_URL   # default http://127.0.0.1:1234/v1
    TINYCTX_LOCAL_MODEL      # default qwen3.6-27b
    TINYCTX_SCOUT_TOP_K      # default 40
    TINYCTX_SCOUT_MAX_FILE_CHARS  # default 4000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .interest import compression_pagerank, load_graph


CACHE_VERSION = 1
DEFAULT_TOP_K = int(os.environ.get("TINYCTX_SCOUT_TOP_K", "40"))
DEFAULT_MAX_FILE_CHARS = int(os.environ.get("TINYCTX_SCOUT_MAX_FILE_CHARS", "4000"))
DEFAULT_BASE_URL = os.environ.get("TINYCTX_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_MODEL = os.environ.get("TINYCTX_LOCAL_MODEL", "qwen3.6-27b")


# ---------------------------------------------------------------- cache paths

def repo_hash(project_root: Path) -> str:
    return hashlib.sha256(str(project_root.resolve()).encode()).hexdigest()[:16]


def cache_dir(project_root: Path) -> Path:
    return Path.home() / ".tinyctx" / "cache" / repo_hash(project_root)


def manifest_path(project_root: Path) -> Path:
    return cache_dir(project_root) / "manifest.json"


def scout_path(project_root: Path) -> Path:
    return cache_dir(project_root) / "scout.md"


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# ---------------------------------------------------------- node -> file path

def file_for_node(node_id: str, project_root: Path) -> Path | None:
    """Best-effort map a graph node id to a source file.
    Conventions handled (graphify-style):
      "src/foo.py:Foo.bar" -> src/foo.py
      "src/foo.py"         -> src/foo.py
      "src.foo.bar"        -> src/foo.py  (dotted, last component is symbol)
    """
    rel, _, _ = node_id.partition(":")
    candidate = project_root / rel
    if candidate.is_file():
        return candidate
    # try dotted form (heuristic): drop last component, slashify the rest
    if "/" not in rel and "." in rel:
        parts = rel.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            stem = "/".join(parts[:cut])
            for ext in (".py", ".ts", ".js", ".tsx", ".go", ".rs", ".java"):
                p = project_root / (stem + ext)
                if p.is_file():
                    return p
    return None


# --------------------------------------------------------- gather + summarize

@dataclass
class ScannedNode:
    nid: str
    score: float
    file: str | None
    sha: str
    snippet: str


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "\n... [truncated]\n"


def gather_scan_targets(graph_path: Path, project_root: Path,
                        *, top_k: int, max_file_chars: int) -> list[ScannedNode]:
    nodes = load_graph(str(graph_path))
    if not nodes:
        return []
    pr = compression_pagerank(nodes)
    # deterministic order: rank desc, then id asc
    ranked = sorted(pr.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    out: list[ScannedNode] = []
    for nid, score in ranked:
        f = file_for_node(nid, project_root)
        if f is None:
            out.append(ScannedNode(nid=nid, score=score, file=None, sha="",
                                   snippet=""))
            continue
        try:
            content = f.read_text(errors="replace")
        except OSError:
            # Why: candidate file vanished or is unreadable (permissions,
            # broken symlink). Emit a placeholder ScannedNode with empty
            # snippet so the caller still sees the node id.
            continue
        out.append(ScannedNode(
            nid=nid,
            score=score,
            file=str(f.resolve()),
            sha=file_hash(f),
            snippet=_truncate(content, max_file_chars),
        ))
    return out


SCOUT_SYSTEM_PROMPT = (
    "You are a code-context scout. You will be given the most load-bearing "
    "symbols of a project, ranked by compression-biased PageRank. Produce a "
    "compact hierarchical summary (under 2000 tokens) so another coding "
    "agent can orient itself without reading every file. Constraints:\n"
    "- Group entries by directory.\n"
    "- For each load-bearing symbol: 1-3 lines on what it does and why it is "
    "load-bearing. Note key invariants and contracts a caller must respect.\n"
    "- End with a 'Load-bearing primitives' section listing the top 5 by score.\n"
    "- Use markdown. Be terse. No filler, no apologies, no meta-commentary."
)


def build_user_prompt(targets: list[ScannedNode]) -> str:
    parts: list[str] = ["Top-ranked symbols (JSON-PageRank score · symbol id):"]
    for t in targets:
        if not t.snippet:
            continue
        rel = t.file or "?"
        parts.append(
            f"\n=== {t.score:.4f} :: {t.nid}\n"
            f"=== file: {rel}\n"
            f"```\n{t.snippet}\n```"
        )
    return "\n".join(parts)


# ------------------------------------------------------------- LLM invocation

def call_local_model(system_prompt: str, user_prompt: str,
                     *, base_url: str, model: str,
                     api_key: str | None = None,
                     timeout_s: float = 600.0,
                     max_tokens: int = 2500) -> str:
    """Talk to an OpenAI-compatible /chat/completions endpoint. Designed for
    LMStudio/vLLM/Ollama/SGLang. Low temperature for deterministic output."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    r = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected response shape: {data!r}") from e


# ----------------------------------------------------------- end-to-end build

def build_scout(graph_path: Path, project_root: Path, *,
                top_k: int = DEFAULT_TOP_K,
                max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
                base_url: str = DEFAULT_BASE_URL,
                model: str = DEFAULT_MODEL,
                api_key: str | None = None,
                _llm_call=call_local_model) -> Path:
    """Build a fresh scout summary and persist it. Returns the scout.md path."""
    targets = gather_scan_targets(graph_path, project_root,
                                  top_k=top_k, max_file_chars=max_file_chars)
    if not targets:
        raise RuntimeError(f"no nodes found in {graph_path}")

    user_prompt = build_user_prompt(targets)
    summary = _llm_call(SCOUT_SYSTEM_PROMPT, user_prompt,
                        base_url=base_url, model=model, api_key=api_key)

    cdir = cache_dir(project_root)
    cdir.mkdir(parents=True, exist_ok=True)
    scout_path(project_root).write_text(summary)

    # Auto-register so `tinyctx-dreamer run` knows this repo exists.
    try:
        from . import registry
        registry.register(project_root)
    except Exception as e:  # noqa: BLE001
        # Why: auto-registration is a convenience for `tinyctx-dreamer`;
        # scout itself is the primary contract. Log and continue so a
        # registry write failure (e.g. read-only home dir) doesn't
        # abort the scout build.
        import sys as _sys
        _sys.stderr.write(
            f"[scout] auto-register failed: {type(e).__name__}: {e}\n")
    manifest = {
        "version": CACHE_VERSION,
        "project_root": str(project_root.resolve()),
        "graph_path": str(Path(graph_path).resolve()),
        "model": model,
        "base_url": base_url,
        "top_k": top_k,
        "ranked": [{"id": t.nid, "score": t.score, "file": t.file, "sha": t.sha}
                   for t in targets],
        "file_hashes": {t.file: t.sha for t in targets if t.file},
        "built_at": time.time(),
    }
    manifest_path(project_root).write_text(json.dumps(manifest, indent=2))
    return scout_path(project_root)


def is_stale(project_root: Path) -> tuple[bool, str]:
    """Return (stale, reason). Stale if no manifest, version mismatch, or any
    tracked file's hash changed."""
    mf = manifest_path(project_root)
    if not mf.is_file():
        return True, "no manifest"
    try:
        data = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError):
        return True, "manifest unreadable"
    if data.get("version") != CACHE_VERSION:
        return True, "version mismatch"
    for path_str, old_hash in (data.get("file_hashes") or {}).items():
        p = Path(path_str)
        if not p.is_file():
            return True, f"missing: {path_str}"
        if file_hash(p) != old_hash:
            return True, f"changed: {path_str}"
    return False, "ok"


def status(project_root: Path) -> dict[str, Any]:
    mf = manifest_path(project_root)
    out: dict[str, Any] = {
        "project_root": str(project_root.resolve()),
        "cache_dir": str(cache_dir(project_root)),
        "scout_path": str(scout_path(project_root)),
        "exists": mf.is_file(),
    }
    if not mf.is_file():
        out["state"] = "absent"
        return out
    try:
        data = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError):
        out["state"] = "corrupt"
        return out
    stale, reason = is_stale(project_root)
    out["state"] = "stale" if stale else "fresh"
    out["reason"] = reason
    out["built_at"] = data.get("built_at")
    out["model"] = data.get("model")
    out["top_k"] = data.get("top_k")
    out["nodes"] = len(data.get("ranked") or [])
    return out


# ---------------------------------------------------------------------- CLI

def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=".",
                   help="project root (default: cwd)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.scout")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="build the scout cache")
    _add_root(pi)
    pi.add_argument("--graph", required=True,
                    help="path to a tinyctx-shaped graph.json")
    pi.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    pi.add_argument("--model", default=DEFAULT_MODEL)
    pi.add_argument("--base-url", default=DEFAULT_BASE_URL)

    pr = sub.add_parser("refresh", help="rebuild cache if stale")
    _add_root(pr)
    pr.add_argument("--graph", default=None,
                    help="graph.json (defaults to manifest's graph_path)")
    pr.add_argument("--force", action="store_true",
                    help="rebuild even if not stale")

    ps = sub.add_parser("status", help="show cache state")
    _add_root(ps)
    ps.add_argument("--json", action="store_true")

    psh = sub.add_parser("show", help="print scout.md")
    _add_root(psh)

    pp = sub.add_parser("path", help="print scout.md path (or empty if absent)")
    _add_root(pp)

    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    if args.cmd == "init":
        path = build_scout(Path(args.graph), root,
                           top_k=args.top_k, model=args.model,
                           base_url=args.base_url)
        print(path)
        return 0

    if args.cmd == "refresh":
        stale, reason = is_stale(root)
        if not stale and not args.force:
            print(f"fresh: {reason}", file=sys.stderr)
            return 0
        graph = args.graph
        if graph is None:
            mf = manifest_path(root)
            if not mf.is_file():
                print("no manifest; pass --graph", file=sys.stderr)
                return 1
            data = json.loads(mf.read_text())
            graph = data.get("graph_path")
            if not graph or not Path(graph).is_file():
                print(f"manifest's graph_path unreachable: {graph}",
                      file=sys.stderr)
                return 1
        path = build_scout(Path(graph), root)
        print(path)
        return 0

    if args.cmd == "status":
        s = status(root)
        if args.json:
            print(json.dumps(s, indent=2, sort_keys=True))
        else:
            print(f"state: {s.get('state')}")
            print(f"path:  {s['scout_path']}")
            if s.get("reason"):
                print(f"reason: {s['reason']}")
            if s.get("built_at"):
                print(f"built: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s['built_at']))}")
            if s.get("nodes"):
                print(f"nodes: {s['nodes']} (top-k={s.get('top_k')})")
        return 0

    if args.cmd == "show":
        sp = scout_path(root)
        if not sp.is_file():
            print("(no scout cached)", file=sys.stderr)
            return 1
        sys.stdout.write(sp.read_text())
        return 0

    if args.cmd == "path":
        sp = scout_path(root)
        if sp.is_file():
            print(sp)
            return 0
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
