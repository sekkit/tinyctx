"""Merkle-tree root hash over scout's tracked file_hashes.

Pure data structure — no I/O. Given a flat ``{file_path: file_hash}`` map
(scout's existing manifest field), compute a hierarchical Merkle DAG with:

  * a single ``root`` SHA-256 hex digest covering every tracked entry
  * a per-directory hash so callers can pinpoint *which* dir changed when
    two manifests differ

Borrowed-as-idea (not code) from zilliztech/claude-context, where the same
shape lives at ``packages/core/src/sync/merkle.ts`` + ``synchronizer.ts:24-44``
and persists to ``~/.context/merkle/<md5(absPath)>.json``. tinyctx scout's
flat ``manifest.json`` already covers the typical < 50K-file case; this
module adds a top-level integrity hash and per-dir hashes so future
incremental flows (and external diffing tools) can fast-path "is anything
different?" without re-reading file contents.

Algorithm:
  1. group entries by directory (POSIX-normalized parent of each path)
  2. each leaf = SHA-256(b"FILE\\0<basename>\\0<file_hash>")
  3. each interior dir = SHA-256(b"DIR\\0<basename>\\0" + sorted child hashes
     joined by b"\\n")
  4. root = the hash of the synthetic root dir ""

The format is intentionally byte-stable across platforms (POSIX paths,
sorted children, fixed separators) so two scout caches built on different
machines from the same content produce identical roots.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath


_LEAF_TAG = b"FILE\x00"
_DIR_TAG = b"DIR\x00"
_SEP = b"\x00"
_CHILD_SEP = b"\n"


@dataclass(frozen=True)
class MerkleTree:
    """Result of :func:`compute`. Hashes are 16-hex-char SHA-256 prefixes
    (matches scout's existing ``file_hash()`` truncation)."""

    root: str
    """Hex digest of the synthetic root dir."""

    dirs: dict[str, str] = field(default_factory=dict)
    """``{posix_dir_path: hex_digest}``. Includes ``""`` for the root."""

    files: dict[str, str] = field(default_factory=dict)
    """``{posix_file_path: file_hash}`` — echoed from input for diffing."""

    def diff(self, other: "MerkleTree") -> dict[str, list[str]]:
        """Return ``{"added": [...], "removed": [...], "changed": [...]}``
        with file paths. Pure set arithmetic — no tree walk required since
        we keep ``files`` flat."""
        a, b = set(self.files), set(other.files)
        added = sorted(b - a)
        removed = sorted(a - b)
        changed = sorted(p for p in a & b if self.files[p] != other.files[p])
        return {"added": added, "removed": removed, "changed": changed}


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


def _norm(path: str) -> str:
    """POSIX-normalize a path string for stable hashing across OSes."""
    # PurePosixPath drops trailing slashes and collapses redundant ``//``;
    # str() round-trips ``""`` to ``"."`` which we don't want here.
    if not path:
        return ""
    return str(PurePosixPath(path)).lstrip("/")


def compute(file_hashes: dict[str, str]) -> MerkleTree:
    """Build a :class:`MerkleTree` from scout's flat ``file_hashes`` map.

    Empty input yields a deterministic empty-tree root (the tag-only digest
    of the synthetic root dir). Order of keys in ``file_hashes`` does not
    affect the result.
    """
    files = {_norm(p): h for p, h in file_hashes.items() if p}
    # children[parent_dir] = list of (basename, child_hash, is_dir)
    # Track directories explicitly so empty dirs survive (rare but possible
    # if a future caller wants to seed structure without files).
    children: dict[str, list[tuple[str, str, bool]]] = {}
    all_dirs: set[str] = {""}

    for path, fh in sorted(files.items()):
        # Walk parents to register every dir on the path.
        parts = path.split("/")
        basename = parts[-1]
        parents = parts[:-1]
        # Build cumulative parent dir paths.
        cumulative = ""
        for part in parents:
            parent_of_this = cumulative
            cumulative = part if not cumulative else f"{cumulative}/{part}"
            all_dirs.add(cumulative)
            # Record dir-as-child link only once.
            sib = children.setdefault(parent_of_this, [])
            if not any(name == part and is_dir for name, _, is_dir in sib):
                # placeholder; hash filled in below after recursion
                sib.append((part, "", True))
        # Register the file leaf.
        leaf_hash = _digest(_LEAF_TAG, basename.encode("utf-8"), _SEP, fh.encode("ascii"))
        children.setdefault(cumulative, []).append((basename, leaf_hash, False))

    # Hash dirs bottom-up by depth descending.
    dirs: dict[str, str] = {}

    def _hash_dir(dir_path: str) -> str:
        if dir_path in dirs:
            return dirs[dir_path]
        # Resolve any unresolved child-dir hashes lazily.
        resolved: list[tuple[str, str]] = []
        for name, child_hash, is_dir in children.get(dir_path, []):
            if is_dir:
                full = name if not dir_path else f"{dir_path}/{name}"
                child_hash = _hash_dir(full)
            resolved.append((name, child_hash))
        resolved.sort(key=lambda t: t[0])
        body = _CHILD_SEP.join(
            f"{name}\x00{h}".encode("utf-8") for name, h in resolved
        )
        own_basename = dir_path.rsplit("/", 1)[-1] if dir_path else ""
        digest = _digest(_DIR_TAG, own_basename.encode("utf-8"), _SEP, body)
        dirs[dir_path] = digest
        return digest

    for d in sorted(all_dirs, key=lambda x: -x.count("/")):
        _hash_dir(d)
    root = _hash_dir("")

    return MerkleTree(root=root, dirs=dirs, files=files)


def root_only(file_hashes: dict[str, str]) -> str:
    """Convenience: just the root hex digest. Equivalent to
    ``compute(file_hashes).root`` but doesn't materialize per-dir hashes."""
    return compute(file_hashes).root
