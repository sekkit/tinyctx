"""Auto-scout: zero-config project-context bootstrap.

The proxy receives codex requests with an `x-codex-cwd` header pointing at
the project root. This module ensures a `scout.md` (project summary) exists
in tinyctx's cache for that project — building it asynchronously the first
time a project is seen, then injecting the cached summary into every
subsequent request's `instructions` field.

User-visible behavior: zero configuration. Just open codex.app in any
project and after the first turn the model has a high-quality summary
of the codebase in its system prompt.

Build pipeline (in priority order, each falls through silently on failure):

  1. `graphify .` — if the binary is on PATH (offered by safishamsi/graphify),
                    produces graphify-out/graph.json; we read that.
  2. (optional) `pipx install graphifyy` — only if config.auto_scout_install_graphify
                                            and pipx is available. One-shot.
  3. In-tree fallback scanner — walks source files (.py/.ts/.go/.rs/etc.),
                                emits a minimal flat graph.json compatible
                                with graphify_adapter._from_flat. Loses
                                edge/dependency info but the PageRank ranking
                                still surfaces large/load-bearing files.

Once a graph.json exists (real or synthetic), `scout.build_scout()` runs
the standard tinyctx flow: rank top-K nodes by compression-biased PageRank
(interest.py), summarize with the local model, persist as scout.md.

Failures are silent. The proxy NEVER blocks waiting for scout — it spawns
a background task and returns immediately. The first request gets nothing;
all subsequent requests for the same project get the injected summary.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import scout


# Marker bracketing the injected block so we can detect and avoid
# double-injection across multiple proxy passes.
_BEGIN_MARKER = "<!-- tinyctx auto-scout BEGIN -->"
_END_MARKER = "<!-- tinyctx auto-scout END -->"

# Source file extensions the fallback scanner picks up. Order doesn't matter.
_SOURCE_EXTS = (
    ".py", ".pyi",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".kts", ".swift",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".cs", ".rb", ".php", ".scala", ".dart", ".lua",
    ".sh", ".bash",
)
# Directories the fallback scanner skips (test files in node_modules, build
# output, vcs metadata, virtual envs, etc. — anything not source-of-truth).
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".pnpm", "vendor",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".cache", ".parcel-cache",
    "graphify-out", ".tinyctx",
})
_FALLBACK_MAX_FILES = int(os.environ.get("TINYCTX_AUTO_SCOUT_MAX_FILES", "200"))
_FALLBACK_MAX_FILE_BYTES = int(
    os.environ.get("TINYCTX_AUTO_SCOUT_MAX_FILE_BYTES", "50000"))
_GRAPHIFY_TIMEOUT_S = int(
    os.environ.get("TINYCTX_AUTO_SCOUT_GRAPHIFY_TIMEOUT_S", "120"))
_PIPX_INSTALL_TIMEOUT_S = int(
    os.environ.get("TINYCTX_AUTO_SCOUT_PIPX_TIMEOUT_S", "180"))

# Track repos we've already attempted, success or failure, so a permanent
# build failure doesn't get retried on every single request. Cleared by
# restarting the proxy.
_BOOTSTRAPPED_OR_INFLIGHT: set[str] = set()


def get_scout(cwd: str | Path | None) -> str | None:
    """Return cached scout.md content for the project at `cwd`, or None.
    Cheap (one stat + one read at most). Never raises."""
    if not cwd:
        return None
    try:
        root = Path(cwd).resolve()
        sp = scout.scout_path(root)
        if sp.is_file():
            return sp.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    return None


def inject_into_body(body: dict[str, Any], scout_md: str
                     ) -> tuple[dict[str, Any], bool]:
    """Prepend scout.md to body.instructions with idempotent markers.

    Returns (new_body, was_injected). If body.instructions already
    contains the BEGIN marker (e.g. the proxy is replaying a body, or
    multiple proxy hops happened), we do NOT re-inject."""
    if not scout_md or not isinstance(body, dict):
        return body, False
    inst = body.get("instructions")
    if not isinstance(inst, str):
        return body, False
    if _BEGIN_MARKER in inst:
        return body, False
    block = (
        f"{_BEGIN_MARKER}\n"
        "## Project context (auto-scouted by tinyctx)\n\n"
        f"{scout_md.strip()}\n"
        f"{_END_MARKER}\n\n"
    )
    new_body = dict(body)
    new_body["instructions"] = block + inst
    return new_body, True


def schedule_bootstrap(cwd: str | Path | None,
                       *,
                       install_graphify: bool = False) -> None:
    """Fire-and-forget: ensure a scout.md exists for `cwd`. If it already
    does, no-op. If it doesn't, schedule an async background build.

    Safe to call from any async context. Never raises. Never blocks."""
    if not cwd:
        return
    try:
        root = Path(cwd).resolve()
    except Exception:  # noqa: BLE001
        return
    if not root.is_dir():
        return
    sp = scout.scout_path(root)
    if sp.is_file():
        return  # already built
    repo_key = str(root)
    if repo_key in _BOOTSTRAPPED_OR_INFLIGHT:
        return  # already attempted (success or fail) since proxy startup
    _BOOTSTRAPPED_OR_INFLIGHT.add(repo_key)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    loop.create_task(
        asyncio.to_thread(_bootstrap_sync, root, install_graphify))


def _bootstrap_sync(root: Path, install_graphify: bool) -> None:
    """Build scout.md for one project. Runs in a worker thread so it
    doesn't block the proxy's event loop. All exceptions swallowed."""
    try:
        cdir = scout.cache_dir(root)
        cdir.mkdir(parents=True, exist_ok=True)

        graph_data = _try_graphify(root, install_graphify)
        if graph_data is None:
            graph_data = _fallback_scan(root)
        if graph_data is None:
            return  # nothing usable

        graph_path = cdir / "graph.json"
        graph_path.write_text(json.dumps(graph_data), encoding="utf-8")

        # Build scout.md using the USER'S configured local backend, not
        # scout.py's hardcoded LMStudio default. Otherwise scout.call_local_model
        # POSTs to http://127.0.0.1:1234/v1 (LMStudio) and 400s when the
        # user is actually running DeepSeek / Ollama / vLLM / whatever.
        try:
            from .config import load_config
            cfg = load_config()
            backend = cfg.local
            base_url = backend.base_url or scout.DEFAULT_BASE_URL
            model = backend.model or scout.DEFAULT_MODEL
            api_key = (os.environ.get(backend.api_key_env)
                       if backend.api_key_env else None)
        except Exception:  # noqa: BLE001
            base_url = scout.DEFAULT_BASE_URL
            model = scout.DEFAULT_MODEL
            api_key = None

        try:
            scout.build_scout(
                graph_path, root,
                top_k=scout.DEFAULT_TOP_K,
                base_url=base_url,
                model=model,
                api_key=api_key,
            )
        except Exception:  # noqa: BLE001 — local model unreachable, etc.
            return
    except Exception:  # noqa: BLE001 — defensive, never bubble up
        return


def _try_graphify(root: Path, install_graphify: bool) -> dict[str, Any] | None:
    """Run `graphify .` if the binary is on PATH; optionally one-shot
    install via pipx if `install_graphify` is true. Returns parsed
    graph.json on success, None otherwise."""
    cmd = shutil.which("graphify")
    if not cmd and install_graphify:
        pipx = shutil.which("pipx")
        if pipx:
            try:
                subprocess.run(
                    [pipx, "install", "graphifyy"],
                    capture_output=True,
                    timeout=_PIPX_INSTALL_TIMEOUT_S,
                    check=False,
                )
                cmd = shutil.which("graphify")
            except Exception:  # noqa: BLE001
                cmd = None
    if not cmd:
        return None
    try:
        subprocess.run(
            [cmd, "."],
            cwd=str(root),
            capture_output=True,
            timeout=_GRAPHIFY_TIMEOUT_S,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    # graphify writes to graphify-out/graph.json by convention
    gj = root / "graphify-out" / "graph.json"
    if not gj.is_file():
        return None
    try:
        return json.loads(gj.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _fallback_scan(root: Path) -> dict[str, Any] | None:
    """Build a minimal flat graph.json by walking source files. No real
    dependency analysis — produces a `{nodes, edges}` shape compatible
    with `graphify_adapter._from_flat`. Top-K ranking via PageRank still
    selects large / dense files reasonably well even without edges."""
    nodes: list[dict[str, Any]] = []
    n_files = 0
    try:
        for path in sorted(root.rglob("*")):
            if n_files >= _FALLBACK_MAX_FILES:
                break
            if not path.is_file():
                continue
            # skip if any parent dir is in skip-set
            if any(p.name in _SKIP_DIRS for p in path.parents):
                continue
            if path.suffix.lower() not in _SOURCE_EXTS:
                continue
            try:
                text = path.read_text(
                    encoding="utf-8", errors="replace"
                )[:_FALLBACK_MAX_FILE_BYTES]
            except Exception:  # noqa: BLE001
                continue
            if not text.strip():
                continue
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                continue
            nodes.append({
                "id": rel,
                "text": text,
                "deps": [],
            })
            n_files += 1
    except Exception:  # noqa: BLE001
        return None
    if not nodes:
        return None
    return {"nodes": nodes, "edges": []}
