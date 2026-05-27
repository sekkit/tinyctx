"""Replace image attachments in request body with text captions.

Runtime side of the mm-cli integration. When
``CFG.image_to_text_preprocess_enabled`` is True, the proxy invokes
``preprocess()`` on the request body BEFORE the router sees it: every
content item whose type is ``input_image`` / ``image_url`` is replaced
with a ``input_text`` item containing a caption produced by
``mm cat <tmpfile> -m accurate --format json``.

The effect: the post-processed body no longer trips the
``image_prefer_frontier`` rule, so the turn stays on the cheap local
backend. Cost: one VLM/LLM caption call per image (bounded by
``image_to_text_timeout_s``) and a small SHA-256 lookup against a
caption cache on disk.

Failure mode is "passthrough": any exception (mm missing, decode
error, timeout, malformed JSON) leaves the offending image item
unchanged so the existing `image_detected → frontier` routing still
kicks in. Best-effort — never raises.

Cache layout::

    ~/.tinyctx/cache/mm-captions/<sha256[:2]>/<sha256>.txt

One caption per image hash. Cache files are plain text; safe to delete
to force re-caption.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional


_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[\w./+-]+);base64,(?P<data>[A-Za-z0-9+/=]+)$",
    re.DOTALL,
)


def _which_mm() -> str:
    """Locate the mm binary using the same search rules as mm_bootstrap.
    Returns "" when not found."""
    forced = os.environ.get("TINYCTX_MM_BIN") or ""
    if forced and os.path.isfile(forced) and os.access(forced, os.X_OK):
        return forced
    found = shutil.which("mm")
    if found:
        return found
    for d in (
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ):
        candidate = os.path.join(d, "mm")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def _resolve_cache_dir(cfg_value: str) -> Path:
    if cfg_value:
        return Path(cfg_value).expanduser()
    home = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
    return home / "cache" / "mm-captions"


def _cache_get(cache_dir: Path, digest: str) -> Optional[str]:
    p = cache_dir / digest[:2] / f"{digest}.txt"
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _cache_put(cache_dir: Path, digest: str, caption: str) -> None:
    try:
        sub = cache_dir / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"{digest}.txt").write_text(caption, encoding="utf-8")
    except OSError:
        pass  # best-effort — caching is an optimization


def _extract_image_bytes(item: dict[str, Any]) -> tuple[bytes, str]:
    """Pull raw image bytes + MIME from a content item.

    Supports two shapes:
      * data URI: ``{"image_url": "data:image/png;base64,..."}``
      * dict URL: ``{"image_url": {"url": "data:image/png;base64,..."}}``

    Returns ``(b"", "")`` when the item points at an http(s) URL — we
    don't fetch external URLs here (different blast radius, network
    egress). Callers should treat that as "passthrough — leave the item
    alone."
    """
    url = item.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        return b"", ""

    m = _DATA_URI_RE.match(url.strip())
    if not m:
        # Likely an http(s) URL — out of scope.
        return b"", ""
    try:
        data = base64.b64decode(m.group("data"), validate=False)
    except (ValueError, TypeError):
        return b"", ""
    mime = m.group("mime") or "image/png"
    return data, mime


_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}


def _run_mm_caption(
    mm_bin: str,
    image_bytes: bytes,
    mime: str,
    *,
    timeout_s: float,
) -> str:
    """Write bytes to a temp file with a sensible extension, invoke
    ``mm cat <file> -m accurate --format json``, parse caption from
    the JSON output. Returns "" on any failure."""
    ext = _EXT_BY_MIME.get(mime.lower(), ".bin")
    fd, tmp_path = tempfile.mkstemp(prefix="tinyctx-mm-", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(image_bytes)
        proc = subprocess.run(
            [mm_bin, "cat", tmp_path, "-m", "accurate", "--format", "json"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if proc.returncode != 0:
        return ""
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return ""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout[:2000]  # treat raw text as a caption fallback

    # mm `cat -m accurate --format json` returns either a dict with a
    # "description" / "caption" / "text" field, or a list-of-messages
    # OpenAI-Responses shape. Try the common keys first.
    if isinstance(payload, dict):
        for k in ("description", "caption", "text", "summary", "content"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Sometimes nested: payload["result"]["caption"] etc.
        for nested_key in ("result", "data", "output"):
            sub = payload.get(nested_key)
            if isinstance(sub, dict):
                for k in ("description", "caption", "text"):
                    v = sub.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    elif isinstance(payload, list) and payload:
        # OpenAI-Responses content list — collect text parts.
        chunks: list[str] = []
        for it in payload:
            if isinstance(it, dict):
                t = it.get("text") or it.get("content")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks)
    return ""


def _caption_for(
    image_bytes: bytes,
    mime: str,
    *,
    mm_bin: str,
    cache_dir: Path,
    timeout_s: float,
) -> tuple[str, str, bool]:
    """Return (caption, sha256, cache_hit). caption is "" on failure."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    cached = _cache_get(cache_dir, digest)
    if cached is not None:
        return cached, digest, True
    caption = _run_mm_caption(mm_bin, image_bytes, mime, timeout_s=timeout_s)
    if caption:
        _cache_put(cache_dir, digest, caption)
    return caption, digest, False


_IMG_TYPES = {"input_image", "image_url", "image"}


def preprocess(
    body: dict[str, Any],
    *,
    enabled: bool = True,
    mm_bin: str = "",
    cache_dir: Optional[Path] = None,
    timeout_s: float = 30.0,
    log: Optional[Callable[..., None]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Walk body.input/messages, replace each image attachment with a
    text caption produced by mm. Returns ``(new_body, stats)`` where
    ``stats`` is a dict with counters useful for tracing.

    When ``enabled`` is False, returns ``(body, {"enabled": False})``
    without inspecting anything.

    Never raises — passthrough on any failure.
    """
    stats: dict[str, Any] = {
        "enabled": enabled,
        "images_seen": 0,
        "images_captioned": 0,
        "cache_hits": 0,
        "errors": 0,
        "skipped_remote_url": 0,
    }
    if not enabled or not isinstance(body, dict):
        return body, stats

    bin_path = mm_bin or _which_mm()
    if not bin_path:
        stats["errors"] += 1
        stats["error_reason"] = "mm binary not found"
        return body, stats

    if cache_dir is None:
        cache_dir = _resolve_cache_dir("")

    new_body = deepcopy(body)
    container_key = "input" if isinstance(new_body.get("input"), list) else (
        "messages" if isinstance(new_body.get("messages"), list) else "")
    if not container_key:
        return new_body, stats

    changed = False
    for item in new_body[container_key]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[Any] = []
        for c in content:
            if not (isinstance(c, dict) and c.get("type") in _IMG_TYPES):
                new_content.append(c)
                continue
            stats["images_seen"] += 1
            image_bytes, mime = _extract_image_bytes(c)
            if not image_bytes:
                stats["skipped_remote_url"] += 1
                new_content.append(c)
                continue
            try:
                caption, digest, cache_hit = _caption_for(
                    image_bytes, mime,
                    mm_bin=bin_path,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                )
            except Exception:  # noqa: BLE001 — passthrough on any error
                stats["errors"] += 1
                new_content.append(c)
                continue
            if not caption:
                stats["errors"] += 1
                new_content.append(c)
                continue
            if cache_hit:
                stats["cache_hits"] += 1
            stats["images_captioned"] += 1
            changed = True
            text_marker = (
                f"[image attachment ({mime}, sha256={digest[:12]}); "
                f"caption: {caption}]"
            )
            new_content.append({"type": "input_text", "text": text_marker})
        if new_content != content:
            item["content"] = new_content
    if log is not None and (stats["images_captioned"] or stats["errors"]):
        try:
            log("multimodal_preprocess", **stats)
        except Exception:  # noqa: BLE001 — logging must never fail forward
            pass
    if not changed:
        return body, stats  # avoid returning a copy when nothing changed
    return new_body, stats
