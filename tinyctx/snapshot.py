"""Instant project structure snapshot — zero-setup, synchronous, ~30ms.

Before graphify/scout/ctx-pack are ready (async bootstrap), inject a
lightweight directory-tree overview so the model has structural context
from turn 0.  Complements the richer but slower scout/ctx-pack pipeline.

Adapted from GitDiagram's "5-minute overview" philosophy — show the
skeleton first, fill in muscle later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_BEGIN_MARKER = "<!-- tinyctx snapshot BEGIN -->"
_END_MARKER = "<!-- tinyctx snapshot END -->"

# Directories to skip entirely
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".venv", "venv", ".env", "build", "dist",
    ".next", ".nuxt", "target", "vendor", ".cache",
    "graphify-out", "tmp", ".claude", ".codex",
})

# Files to skip at the top level
_SKIP_FILES = frozenset({
    ".DS_Store", "Thumbs.db", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "Gemfile.lock",
    "Pipfile.lock",
})

DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_DEPTH = 3

# Dir -> top file counts to show
_MAX_TOP_FILES = 8
_MAX_TOP_DIRS = 10


def build_snapshot(
    project_root: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str | None:
    """Walk project_root and return a lightweight tree-as-markdown summary.
    Returns None when the root doesn't exist or is empty."""
    if not project_root.is_dir():
        return None

    lines: list[str] = []
    total = 0

    def _walk(dir_path: Path, prefix: str, depth: int) -> None:
        nonlocal total
        if depth > max_depth or total >= max_chars:
            return
        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except OSError:
            return

        dirs: list[Path] = []
        files: list[Path] = []
        for entry in entries:
            if entry.name.startswith(".") and entry.name not in (".env",):
                continue
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                dirs.append(entry)
            else:
                if entry.name in _SKIP_FILES:
                    continue
                files.append(entry)

        shown_dirs = dirs[:_MAX_TOP_DIRS]
        shown_files = files[:_MAX_TOP_FILES]
        remaining_dirs = len(dirs) - len(shown_dirs)
        remaining_files = len(files) - len(shown_files)

        for i, d in enumerate(shown_dirs):
            is_last = (i == len(shown_dirs) - 1) and not shown_files and remaining_dirs == 0
            connector = "└── " if is_last else "├── "
            size_hint = _dir_size_hint(d)
            line = f"{prefix}{connector}{d.name}/{size_hint}"
            lines.append(line)
            total += len(line) + 1
            if total >= max_chars:
                return
            ext_prefix = prefix + ("    " if is_last else "│   ")
            _walk(d, ext_prefix, depth + 1)

        for i, f in enumerate(shown_files):
            is_last = (i == len(shown_files) - 1) and remaining_files == 0 and remaining_dirs == 0
            connector = "└── " if is_last else "├── "
            try:
                fsize = f.stat().st_size
            except OSError:
                fsize = 0
            line = f"{prefix}{connector}{f.name}  ({_human_size(fsize)})"
            lines.append(line)
            total += len(line) + 1
            if total >= max_chars:
                return

        if remaining_files > 0 or remaining_dirs > 0:
            parts: list[str] = []
            if remaining_files > 0:
                parts.append(f"{remaining_files} files")
            if remaining_dirs > 0:
                parts.append(f"{remaining_dirs} dirs")
            line = f"{prefix}└── ... ({', '.join(parts)} omitted)"
            lines.append(line)
            total += len(line) + 1

    root_name = project_root.name or str(project_root)
    lines.append(f"# `{root_name}/` project structure")
    lines.append("")
    lines.append(f"{root_name}/")
    _walk(project_root, "", 0)
    lines.append("")

    return "\n".join(lines)


def inject_snapshot(body: dict[str, Any], snap_md: str) -> dict[str, Any]:
    """Inject a snapshot block into body.instructions. Idempotent."""
    inst = body.get("instructions") or ""
    if not isinstance(inst, str):
        inst = str(inst)
    if _BEGIN_MARKER in inst:
        return body
    block = (
        f"\n\n{_BEGIN_MARKER}\n"
        f"{snap_md}\n"
        f"{_END_MARKER}\n\n"
    )
    new_body = dict(body)
    new_body["instructions"] = block + inst
    return new_body


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _dir_size_hint(d: Path) -> str:
    """Quick entry count — don't recursively sum (too slow for 30ms budget)."""
    try:
        count = sum(1 for _ in d.iterdir())
        return f" ({count} entries)"
    except OSError:
        return ""
