"""Hierarchical patch merge — LLM-driven aggregation of edit proposals.

Analogous to gradient aggregation in neural network training: merges
independently-generated patches from the Reflect stage into a single
coherent patch via hierarchical LLM calls.

Patches from failure analysis take priority over success patches.
Within each level, batches are merged in parallel via thread pool.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

_MERGE_SYSTEM = """You are an expert skill merger. Your job is to combine multiple proposed
skill patches into a single coherent patch.

Rules:
1. Remove duplicate edits (same operation on same target).
2. If two edits conflict, pick the more specific one.
3. If two edits can be combined into a stronger single edit, do so.
4. Preserve the original "op", "target", and "content" fields.
5. Never invent new edits — only merge existing ones.

Return JSON:
{
  "edits": [
    {"op": "replace|append|delete|insert_after", "target": "...", "content": "...", "reasoning": "merged from edits [0,3]"}
  ],
  "reasoning": "how you merged the patches"
}"""


def _merge_batch(
    skill_content: str,
    patches: list[dict[str, Any]],
    *,
    optimizer: OptimizerFn,
    max_tokens: int = 4096,
    level: int = 1,
    label: str = "",
) -> dict[str, Any]:
    """Call optimizer LLM to merge a batch of patches into one."""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to Merge ({len(patches)} total, level {level})"
        f"{' (' + label + ')' if label else ''}\n{patches_text}"
    )

    try:
        response, _meta = optimizer(_MERGE_SYSTEM, user, {
            "max_tokens": max_tokens,
            "stage": f"merge_l{level}",
        })
    except Exception:
        response = ""

    if response:
        merged = _parse_merge_response(response, level)
        if merged:
            return merged

    # Fallback: concatenate all edits from all patches
    all_edits: list[dict[str, Any]] = []
    for patch in patches:
        for edit in patch.get("edits", []):
            if isinstance(edit, dict):
                edit.setdefault("merge_level", level)
                all_edits.append(edit)
    return {"reasoning": f"fallback concat (level {level})", "edits": all_edits, "merge_level": level}


def _parse_merge_response(response: str, level: int) -> dict[str, Any] | None:
    """Extract merged edits from optimizer response."""
    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]*"edits"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    edits = data.get("edits", [])
    if not isinstance(edits, list):
        return None
    for e in edits:
        e.setdefault("merge_level", level)
    data.setdefault("merge_level", level)
    return data


def merge_patches(
    skill_content: str,
    failure_patches: list[dict[str, Any]],
    success_patches: list[dict[str, Any]],
    *,
    optimizer: OptimizerFn,
    batch_size: int = 4,
    max_tokens: int = 4096,
    workers: int = 4,
    verbose: bool = False,
) -> dict[str, Any]:
    """Hierarchically merge failure and success patches into a single coherent patch.

    1. Merge failure patches hierarchically (they get priority).
    2. Merge success patches hierarchically.
    3. Merge the two resulting patches together.

    Args:
        skill_content: current skill document
        failure_patches: list of patches from failure analysis
        success_patches: list of patches from success analysis
        optimizer: callable (system, user, options) → (response, metadata)
        batch_size: number of patches to merge per LLM call
        max_tokens: max tokens per merge call
        workers: max parallel merge workers
        verbose: print progress

    Returns:
        dict with "edits" list and "reasoning" field
    """
    # Merge failures
    if failure_patches:
        failure_merged = _hierarchical_merge(
            skill_content, failure_patches, optimizer,
            batch_size=batch_size, max_tokens=max_tokens,
            workers=workers, verbose=verbose, label="failures",
        )
    else:
        failure_merged = {"edits": [], "reasoning": "no failure patches"}

    # Merge successes
    if success_patches:
        success_merged = _hierarchical_merge(
            skill_content, success_patches, optimizer,
            batch_size=batch_size, max_tokens=max_tokens,
            workers=workers, verbose=verbose, label="successes",
        )
    else:
        success_merged = {"edits": [], "reasoning": "no success patches"}

    # Merge failure + success results
    both = [failure_merged, success_merged]
    if failure_merged.get("edits") and success_merged.get("edits"):
        return _merge_batch(
            skill_content, both, optimizer=optimizer,
            max_tokens=max_tokens, level=99, label="final",
        )

    # Only one side has edits
    return failure_merged if failure_merged.get("edits") else success_merged


def _hierarchical_merge(
    skill_content: str,
    patches: list[dict[str, Any]],
    optimizer: OptimizerFn,
    *,
    batch_size: int = 4,
    max_tokens: int = 4096,
    workers: int = 4,
    verbose: bool = False,
    label: str = "",
) -> dict[str, Any]:
    """Merge N patches hierarchically, bottom-up.

    Each level merges `batch_size` patches into one via LLM call.
    Same-level batches execute in parallel.
    """
    if not patches:
        return {"edits": [], "reasoning": "empty"}

    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0

    while len(current) > 1:
        level += 1
        # Build batches
        batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(current), batch_size):
            batches.append(current[i : i + batch_size])

        if verbose:
            print(f"    [merge {label}] level {level}: {len(current)} patches → {len(batches)} batches")

        # Merge batches in parallel
        merged: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
            futures = {}
            for i, batch in enumerate(batches):
                fut = ex.submit(
                    _merge_batch, skill_content, batch,
                    optimizer=optimizer, max_tokens=max_tokens,
                    level=level, label=f"{label}-batch{i}",
                )
                futures[fut] = i

            # Collect results in order
            results_by_idx: dict[int, dict[str, Any]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_by_idx[idx] = fut.result()
                except Exception:
                    results_by_idx[idx] = {"edits": [], "reasoning": f"batch {idx} failed"}

            for i in range(len(batches)):
                merged.append(results_by_idx.get(i, {"edits": [], "reasoning": f"batch {i} missing"}))

        current = merged

    return current[0]
