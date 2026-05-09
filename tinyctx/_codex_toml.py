"""Shared helpers for safely patching ~/.codex/config.toml.

Multiple bootstrap modules (gitnexus / serena / caveman / others) need to
append `[mcp_servers.<name>]` blocks to the user's codex config without
breaking existing comments / ordering. We never re-serialize TOML —
we append text and detect prior installs by an exact section-header
match, while holding an OS-level file lock to prevent race-induced
duplicates from concurrent bootstrap runs (tinyctx-up async fires
several in parallel).

Three operations:

  append_mcp_block(path, marker, block, dry_run=False)
    Idempotent + lock-protected. Appends `block` to `path` if `marker`
    (the literal `[mcp_servers.<name>]` line) is not already present
    on its own line in the file.

  strip_mcp_block(text, marker)
    Removes the contiguous block starting at `marker` until the next
    `[section]` header or EOF. Also walks back over preceding `#`
    comment lines we wrote, so the file is left clean after uninstall.

  has_mcp_block(path, marker)
    Cheap presence check (no lock).
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows
    _HAS_FCNTL = False


@contextlib.contextmanager
def _file_lock(path: Path):
    """Best-effort exclusive lock on `<path>.lock`. fcntl.flock on POSIX;
    no-op on Windows. The lock file is separate from the actual config
    so we can lock even when the config doesn't yet exist."""
    if not _HAS_FCNTL:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".tinyctx-lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path),
                 os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


def _marker_present(text: str, marker: str) -> bool:
    """Detect `marker` as a real TOML section header — must appear on a
    line of its own (allowing leading whitespace) so substring noise
    inside comments doesn't trigger false positives, and we don't miss
    a real header just because it has trailing whitespace."""
    for line in text.splitlines():
        if line.strip() == marker:
            return True
    return False


def append_mcp_block(
    path: Path,
    marker: str,
    block: str,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Idempotently append `block` to `path` when `marker` is absent.

    Holds an exclusive POSIX file lock on `<path>.tinyctx-lock` for the
    full check-then-write sequence so concurrent bootstrap processes
    can't both see "absent" and both append (which produced duplicate
    `[mcp_servers.gitnexus]` headers in the wild).
    """
    try:
        with _file_lock(path):
            existed = path.is_file()
            if existed:
                text = path.read_text(encoding="utf-8", errors="replace")
                if _marker_present(text, marker):
                    return True, "already configured"
            if dry_run:
                return True, f"DRY-RUN would append to {path}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if existed:
                current = path.read_text(encoding="utf-8", errors="replace")
                # Re-check after lock — in case another process snuck in.
                if _marker_present(current, marker):
                    return True, "already configured (after-lock recheck)"
                if not current.endswith("\n"):
                    current += "\n"
                path.write_text(current + block, encoding="utf-8")
            else:
                path.write_text(block.lstrip(), encoding="utf-8")
    except OSError as e:
        return False, f"write failed: {e}"
    return True, f"appended block to {path}"


def strip_mcp_block(text: str, marker: str) -> str:
    """Remove contiguous block starting at `marker` line until next
    `[section]` header (or EOF), and walk back over leading `#` comments
    we wrote (so an `# Added by tinyctx` header is also removed)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == marker:
            # Walk back over any preceding tinyctx-comment lines we wrote.
            j = len(out)
            while j > 0 and out[j - 1].lstrip().startswith("#"):
                j -= 1
            # Drop trailing blank line before the comment block as well.
            if j > 0 and out[j - 1].strip() == "":
                j -= 1
            out = out[:j]
            # Skip forward until next bracket section OR EOF.
            i += 1
            while i < n:
                stripped = lines[i].strip()
                if stripped.startswith("[") and stripped != marker:
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def has_mcp_block(path: Path, marker: str) -> bool:
    """Cheap presence check — exact-line match (not substring)."""
    if not path.is_file():
        return False
    try:
        return _marker_present(
            path.read_text(encoding="utf-8", errors="replace"), marker)
    except OSError:
        return False
