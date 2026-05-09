"""LLMLingua-2 pre-escalation prompt compression hook.

Microsoft's LLMLingua-2 (microsoft/LLMLingua, MIT) is a small classifier
that predicts which prompt tokens are safely droppable while keeping the
target model's output unchanged. Empirically gets 2-5× compression on
long contexts with no quality regression on coding/QA tasks.

tinyctx fires this hook BEFORE forwarding to the frontier, on the
expensive items only (tool outputs, large message text). It NEVER
touches `instructions`, tool schemas, or system messages — those are
prompt-cache-critical and the savings are paid for by the cache,
not by compression. It also skips compaction handoff prompts (the
compactor itself produces compressed output already).

Lazy import: llmlingua is heavy (~3 GB transformers cache on first
load). We import it only when the hook is actually configured to fire,
and gracefully no-op when the dep is missing — `pip install
'tinyctx[compress]'` upgrades a no-op into a real compression call.

Cache-aware: like dedup/purge/read_delta, LLMLingua mutates wire bytes,
so it MUST run under `CacheAwareMutator`. The proxy gates it on the
same TTL/threshold. Default off — opt-in via
`frontier_lingua_enabled = true` in config.

CLI:
    python -m tinyctx.lingua status        # show whether llmlingua importable + model state
    python -m tinyctx.lingua test           # run a self-check on a sample blob
    python -m tinyctx.lingua warmup         # download model weights ahead of first use
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Empirically the smallest LLMLingua-2 checkpoint that still works well
# for coding/QA contexts. Override via env if you want the heavier
# multilingual variant (`microsoft/llmlingua-2-bert-base-multilingual...`).
DEFAULT_MODEL = os.environ.get(
    "TINYCTX_LINGUA_MODEL",
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")

# Compression aggressiveness. 0.5 means "keep ~50% of tokens".
# Range: 0.1 (very aggressive) — 0.9 (very mild).
DEFAULT_RATIO = float(os.environ.get("TINYCTX_LINGUA_RATIO", "0.5"))

# Items below this size aren't worth compressing (model load + classify
# overhead would dominate). 4 chars/token → 800 chars ≈ 200 tokens.
MIN_BYTES = int(os.environ.get("TINYCTX_LINGUA_MIN_BYTES", "800"))

# Don't shrink ANY field below this many characters — preserves at least
# enough context for the model to follow the conversation.
FLOOR_BYTES = int(os.environ.get("TINYCTX_LINGUA_FLOOR_BYTES", "200"))

# When the compressor's output > original × this fraction, keep original.
SAVING_BUDGET = float(os.environ.get("TINYCTX_LINGUA_BUDGET", "0.85"))

_TOOL_RESULT_TYPES = {"function_call_output", "tool_result", "mcp_result"}

TINYCTX_HOME = Path(
    os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "lingua.log"

# Cached PromptCompressor (single global; first construction loads weights).
_COMPRESSOR_CACHE: Any = None
_COMPRESSOR_INIT_FAILED = False


def _log(msg: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def is_available() -> bool:
    """True iff llmlingua is importable. Cheap (importlib lookup)."""
    try:
        import importlib
        return importlib.util.find_spec("llmlingua") is not None
    except Exception:  # noqa: BLE001
        return False


def _get_compressor(model_name: str = DEFAULT_MODEL) -> Any | None:
    """Lazy-init shared PromptCompressor. Returns None on failure."""
    global _COMPRESSOR_CACHE, _COMPRESSOR_INIT_FAILED
    if _COMPRESSOR_CACHE is not None:
        return _COMPRESSOR_CACHE
    if _COMPRESSOR_INIT_FAILED:
        return None
    try:
        from llmlingua import PromptCompressor  # type: ignore[import-untyped]
    except ImportError:
        _COMPRESSOR_INIT_FAILED = True
        return None
    try:
        _COMPRESSOR_CACHE = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map="cpu",  # default conservative; user can override
        )
    except Exception as e:  # noqa: BLE001
        _log(f"PromptCompressor init failed: {e}")
        _COMPRESSOR_INIT_FAILED = True
        return None
    return _COMPRESSOR_CACHE


def _compress_one(text: str, *, ratio: float = DEFAULT_RATIO,
                  model_name: str = DEFAULT_MODEL) -> tuple[str, dict]:
    """Compress one text blob. Returns (new_text, info)."""
    info = {"original_chars": len(text), "compressed_chars": len(text),
            "compressed": False, "reason": "no-op"}
    if len(text) < MIN_BYTES:
        info["reason"] = "below_min_bytes"
        return text, info

    pc = _get_compressor(model_name)
    if pc is None:
        info["reason"] = "compressor_unavailable"
        return text, info

    try:
        # rate=1-ratio so users specify "fraction kept", not "fraction dropped".
        # llmlingua-2 expects rate in (0,1].
        result = pc.compress_prompt(
            text,
            rate=max(0.1, min(0.95, ratio)),
            force_tokens=["\n", "?", "."],
        )
        compressed = (result.get("compressed_prompt")
                      if isinstance(result, dict) else str(result))
    except Exception as e:  # noqa: BLE001
        _log(f"compress failed: {e}")
        info["reason"] = f"runtime_error: {e}"
        return text, info

    if not isinstance(compressed, str) or not compressed:
        info["reason"] = "empty_result"
        return text, info
    if len(compressed) >= int(len(text) * SAVING_BUDGET):
        info["reason"] = "saving_below_budget"
        return text, info
    if len(compressed) < FLOOR_BYTES:
        info["reason"] = "below_floor_bytes"
        return text, info

    info["compressed_chars"] = len(compressed)
    info["compressed"] = True
    info["reason"] = "ok"
    return compressed, info


def _flatten_to_text(payload: Any) -> tuple[str, str]:
    """Return (text, original_shape). Shape is one of:
    'string', 'list-text-items', 'json'. Used so we can re-encode the
    compressed text back into the same shape without breaking the wire."""
    if isinstance(payload, str):
        return payload, "string"
    if isinstance(payload, list):
        parts: list[str] = []
        for c in payload:
            if isinstance(c, dict):
                t = c.get("type")
                if t in ("text", "input_text", "output_text"):
                    parts.append(str(c.get("text", "")))
        return "\n".join(parts), "list-text-items"
    try:
        return json.dumps(payload, ensure_ascii=False), "json"
    except Exception:  # noqa: BLE001
        return str(payload), "json"


def _set_compressed(item: dict, key: str, new_text: str,
                    shape: str) -> None:
    """Write back the compressed text in the original shape."""
    if shape == "list-text-items":
        item[key] = [{"type": "output_text", "text": new_text}]
    else:
        item[key] = new_text


def compress_for_frontier(body: dict[str, Any], *,
                          ratio: float = DEFAULT_RATIO,
                          model_name: str = DEFAULT_MODEL,
                          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Walk `body.input` and shrink large tool-result payloads.

    NEVER touches:
      - body.instructions  (cache-critical)
      - body.tools         (cache-critical)
      - assistant text     (would distort what the model already said)
      - user messages      (preserve verbatim user intent)

    Only target: function_call_output / tool_result / mcp_result items.
    Returns (new_body, info).
    """
    info: dict[str, Any] = {
        "applied": False, "items_examined": 0, "items_compressed": 0,
        "chars_before": 0, "chars_after": 0, "skipped": [],
    }
    if not is_available():
        info["skipped"].append("llmlingua_not_installed")
        return body, info

    items = body.get("input") or body.get("messages")
    if not isinstance(items, list):
        return body, info

    out = deepcopy(body)
    container_key = "input" if isinstance(out.get("input"), list) else "messages"
    out_items = out[container_key]

    compressed_n = 0
    chars_before = 0
    chars_after = 0
    skips: list[str] = []

    for item in out_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in _TOOL_RESULT_TYPES:
            continue
        info["items_examined"] += 1
        # locate payload field
        for fld in ("output", "content"):
            if fld not in item:
                continue
            payload = item[fld]
            text, shape = _flatten_to_text(payload)
            if not text or len(text) < MIN_BYTES:
                continue
            new_text, sub_info = _compress_one(
                text, ratio=ratio, model_name=model_name)
            chars_before += len(text)
            chars_after += len(new_text)
            if sub_info["compressed"]:
                _set_compressed(item, fld, new_text, shape)
                compressed_n += 1
            else:
                skips.append(sub_info["reason"])
            break  # don't double-process output and content for same item

    info["applied"] = compressed_n > 0
    info["items_compressed"] = compressed_n
    info["chars_before"] = chars_before
    info["chars_after"] = chars_after
    info["skipped"] = skips
    return out, info


# ─────────────────── CLI ─────────────────────────


def _print_status() -> None:
    avail = is_available()
    print(f"llmlingua importable: {'yes' if avail else 'no'}")
    print(f"model:                {DEFAULT_MODEL}")
    print(f"default ratio:        {DEFAULT_RATIO}")
    print(f"min bytes:            {MIN_BYTES}")
    if not avail:
        print("Install: pip install 'tinyctx[compress]'")


def _cmd_test() -> int:
    sample = ("This is a longer-than-MIN_BYTES sample. " * 60)
    new, info = _compress_one(sample, ratio=0.5)
    print(json.dumps(info, indent=2))
    print(f"\noriginal len: {len(sample)}")
    print(f"new len:      {len(new)}")
    return 0 if info["compressed"] or not is_available() else 1


def _cmd_warmup() -> int:
    if not is_available():
        print("llmlingua not installed; nothing to warm up", file=sys.stderr)
        return 1
    print(f"loading {DEFAULT_MODEL} (downloads weights on first run)...")
    pc = _get_compressor()
    if pc is None:
        print("init failed; see ~/.tinyctx/logs/lingua.log")
        return 1
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.lingua")
    p.add_argument("cmd", nargs="?", default="status",
                   choices=["status", "test", "warmup"])
    args = p.parse_args(argv)
    if args.cmd == "status":
        _print_status()
        return 0
    if args.cmd == "test":
        return _cmd_test()
    if args.cmd == "warmup":
        return _cmd_warmup()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
