"""Headroom-aware tool output compression.

Integrates the headroom-ai library into tinyctx's sanitize pipeline.
Headroom provides content-type-aware compression (SmartCrusher for JSON,
CodeCompressor, SearchCompressor, etc.) that can drastically shrink
bulky tool outputs before they reach the LLM.

Headroom extras are auto-installed on demand:
  - SmartCrusher (JSON → CSV+schema): works out of the box, zero extras
  - CodeCompressor (AST-aware): needs headroom-ai[code] (tree-sitter)
  - Kompress-base (ML text): needs headroom-ai[ml] (transformers+HuggingFace)

When tree-sitter or transformers are missing, this module triggers a
ONE-TIME background install via `uv pip install headroom-ai[code,ml]`.
Until it completes, code/text/log compression falls through (no-op);
SmartCrusher for JSON arrays works regardless. Re-import after install
picks up the new capabilities without a proxy restart.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from typing import Any

logger = logging.getLogger("tinyctx.headroom")

# ── import guard ──────────────────────────────────────────────────────

_headroom_available: bool | None = None
_ContentRouter: Any = None

# Tree-sitter and transformers availability (for code/text compression).
# None = unchecked, True = installed, False = missing.
_tree_sitter_available: bool | None = None
_transformers_available: bool | None = None

# Background install lock — ensures we only spawn one install thread.
_install_lock = threading.Lock()
_install_in_progress = False
_install_log: list[str] = []

# Capability log gate — set by proxy to True after first emit.
_logged_caps = False


def _check_headroom() -> bool:
    global _headroom_available, _ContentRouter
    if _headroom_available is None:
        try:
            from headroom.transforms import ContentRouter
            _ContentRouter = ContentRouter
            _headroom_available = True
        except ImportError:
            _headroom_available = False
    return _headroom_available


# ── capability detection ──────────────────────────────────────────────

def _check_code_available() -> bool:
    """Check if tree-sitter is installed for code-aware compression."""
    global _tree_sitter_available
    if _tree_sitter_available is None:
        try:
            import tree_sitter  # noqa: F401
            _tree_sitter_available = True
        except ImportError:
            _tree_sitter_available = False
            _maybe_auto_install("code")
    return _tree_sitter_available


def _check_ml_available() -> bool:
    """Check if transformers is installed for ML-based text compression."""
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers  # noqa: F401
            _transformers_available = True
        except ImportError:
            _transformers_available = False
            _maybe_auto_install("ml")
    return _transformers_available


def _maybe_auto_install(extra: str) -> None:
    """Trigger a one-shot background install of headroom extras.

    Only fires once per process lifetime (guarded by _install_lock).
    The install runs in a daemon thread so it never blocks the proxy.
    """
    global _install_in_progress, _install_log
    with _install_lock:
        if _install_in_progress:
            return  # already installing
        _install_in_progress = True
        _install_log.append(f"auto_install_headroom_{extra}_started")

    def _install():
        global _tree_sitter_available, _transformers_available
        try:
            # Find uv — prefer the one that manages our venv
            uv_bin = _find_uv()
            cmd = [uv_bin, "pip", "install", "--python", sys.executable,
                   f"headroom-ai[{extra}]"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                _install_log.append(f"auto_install_headroom_{extra}_ok")
                # Re-detect after install
                try:
                    import tree_sitter  # noqa: F401
                    _tree_sitter_available = True
                except ImportError:
                    pass
                try:
                    import transformers  # noqa: F401
                    _transformers_available = True
                except ImportError:
                    pass
            else:
                _install_log.append(
                    f"auto_install_headroom_{extra}_failed: {result.stderr[:200]}")
        except Exception as e:
            _install_log.append(f"auto_install_headroom_{extra}_error: {e}")

    t = threading.Thread(target=_install, daemon=True, name="headroom-auto-install")
    t.start()


def _find_uv() -> str:
    """Find the uv binary — prefer the one that built this venv."""
    import shutil
    uv = shutil.which("uv")
    if uv:
        return uv
    # Fallback: try the known location
    candidates = [
        "/Users/sekkit/.local/bin/uv",
        str(Path(sys.executable).parent / "uv"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "uv"  # last resort


from pathlib import Path


def capability_report() -> dict[str, Any]:
    """Return a diagnostic dict of what's available and what's installing."""
    return {
        "headroom_installed": _check_headroom(),
        "tree_sitter_available": _check_code_available() if _check_headroom() else False,
        "transformers_available": _check_ml_available() if _check_headroom() else False,
        "install_in_progress": _install_in_progress,
        "install_log": list(_install_log),
    }


# ── tool output extraction ────────────────────────────────────────────

_TOOL_RESULT_TYPES = frozenset({
    "function_call_output",
    "tool_result",
    "mcp_result",
})


def _tool_output_text(item: dict[str, Any]) -> str:
    """Extract text content from a tool-output item."""
    output = item.get("output")
    if output is not None:
        if isinstance(output, str):
            return output
        if isinstance(output, (list, dict)):
            try:
                return json.dumps(output, ensure_ascii=False)
            except Exception:
                return str(output)
        return str(output)

    content = item.get("content")
    if content is not None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t in ("text", "input_text", "output_text"):
                        parts.append(str(c.get("text", "")))
            return "\n".join(parts)
        return str(content)

    return ""


# ── main compression function ─────────────────────────────────────────


def compress_tool_outputs(
    body: dict[str, Any],
    *,
    model: str = "gpt-4o",
    enabled: bool = True,
    min_chars: int = 200,
    max_chunk_chars: int = 128_000,
) -> dict[str, Any]:
    """Walk body.input and compress tool output items via headroom's ContentRouter.

    Content-type dispatch:
      - JSON arrays → SmartCrusher (always works, zero extra deps)
      - Source code   → CodeCompressor (needs tree-sitter, auto-installed)
      - Search/log    → respective compressor (always works)
      - Plain text    → Kompress-base (needs transformers, auto-installed)

    Missing capabilities are auto-installed in a background thread on
    first detection. Until install completes, those content types pass
    through uncompressed — no errors, no blocking.

    Returns the (possibly mutated) body dict. When headroom is not
    installed or enabled=False, returns body unchanged.
    """
    if not enabled:
        return body
    if not _check_headroom():
        return body

    items = body.get("input")
    if not isinstance(items, list):
        return body

    # Trigger capability checks (these spawn background installs if needed)
    _check_code_available()
    _check_ml_available()

    router = _get_router()

    compressed_count = 0
    chars_before_total = 0
    chars_after_total = 0
    strategies: dict[str, int] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in _TOOL_RESULT_TYPES:
            continue

        text = _tool_output_text(item)
        if len(text) < max(1, min_chars):
            continue

        if len(text) > max_chunk_chars:
            compressed, strategy = _compress_large(text, router, max_chunk_chars)
        else:
            compressed, strategy = _compress_single(text, router)

        if compressed != text:
            chars_before_total += len(text)
            chars_after_total += len(compressed)
            strategies[strategy] = strategies.get(strategy, 0) + 1
            _write_compressed(item, compressed)
            compressed_count += 1

    if compressed_count > 0:
        savings_pct = (
            round(100 * (1 - chars_after_total / max(chars_before_total, 1)))
            if chars_before_total > 0
            else 0
        )
        logger.info(
            "headroom_compress: %d outputs compressed, %d→%d chars (%d%% saved), "
            "strategies: %s",
            compressed_count,
            chars_before_total,
            chars_after_total,
            savings_pct,
            dict(strategies),
        )

    # Log install progress if any
    if _install_log:
        for msg in _install_log:
            if msg.endswith("_started"):
                logger.info("headroom auto-install started: %s", msg)
        _install_log.clear()

    return body


def _get_router() -> Any:
    """Return a configured ContentRouter instance."""
    global _ContentRouter
    return _ContentRouter()


def _compress_single(text: str, router: Any) -> tuple[str, str]:
    """Compress a single text payload. Returns (compressed_text, strategy_name)."""
    try:
        result = router.compress(text)
        strategy = getattr(result, "strategy_used", "unknown")
        strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)
        compressed = result.compressed
        if isinstance(compressed, str) and compressed:
            if compressed.startswith('"') and compressed.endswith('"'):
                try:
                    uncompressed = json.loads(compressed)
                    if isinstance(uncompressed, str) and len(uncompressed) > 0:
                        compressed = uncompressed
                except (json.JSONDecodeError, TypeError):
                    pass
            return compressed, strategy_name
    except Exception:
        logger.debug("headroom compress failed, keeping original", exc_info=True)

    return text, "passthrough"


def _compress_large(text: str, router: Any, chunk_size: int) -> tuple[str, str]:
    """Compress a very large payload by splitting into chunks."""
    chunks: list[str] = []
    strategies: set[str] = set()
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        compressed, strategy = _compress_single(chunk, router)
        chunks.append(compressed)
        strategies.add(strategy)

    strategy_summary = "+".join(sorted(strategies)) if strategies else "passthrough"
    return "\n".join(chunks), strategy_summary


def _write_compressed(item: dict[str, Any], compressed: str) -> None:
    """Write compressed output back to the item, preferring 'output' field."""
    if "output" in item and isinstance(item["output"], str):
        item["output"] = compressed
    elif "content" in item and isinstance(item["content"], str):
        item["content"] = compressed
    elif "output" in item and isinstance(item["output"], (list, dict)):
        item["output"] = compressed
    else:
        if "output" in item:
            item["output"] = compressed
        elif "content" in item:
            item["content"] = compressed
