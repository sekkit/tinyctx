"""LLM-driven edit ranking and selection (gradient clipping).

Analogous to gradient clipping in neural network training: ranks candidate
edits by importance using the optimizer LLM, keeps only the top-L.

When the number of edits exceeds the budget, the optimizer judges which
edits are most impactful. Edits within budget pass through unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .update import describe_edit

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

_RANKER_SYSTEM = """You are an expert skill editor. Your job is to review a set of proposed
edits to an agent's skill document and rank them by importance.

For each edit, assign a priority score from 1 (least important) to 10 (most
important). Consider:
  - How many failure cases would this edit prevent?
  - Is there concrete evidence from trajectories for this edit?
  - Would this edit generalize beyond the specific cases seen?
  - Does this edit conflict with or duplicate another edit?

Return JSON:
{
  "rankings": [{"index": 0, "score": 8, "reasoning": "..."}, ...],
  "reasoning": "overall assessment"
}

The `index` field must match the index in the edit list below. Include EVERY
edit in your rankings. Order does not matter (we sort by score)."""


def rank_and_select(
    skill_content: str,
    edits: list[dict[str, Any]],
    max_edits: int,
    *,
    optimizer: OptimizerFn,
    optimizer_max_tokens: int = 4096,
) -> list[dict[str, Any]]:
    """Use optimizer LLM to rank edits by importance, then keep top-max_edits.

    If the edit pool is within budget, returns it unchanged (no LLM call).
    Otherwise, asks the optimizer to score each edit from 1-10, sorts
    descending by score, and returns the top-max_edits.

    Args:
        skill_content: current skill document
        edits: list of edit dicts (merged, deduplicated)
        max_edits: budget cap
        optimizer: callable (system, user, options) -> (response, metadata)
        optimizer_max_tokens: max tokens for the ranking call

    Returns:
        selected edits list (len ≤ max_edits)
    """
    if len(edits) <= max_edits:
        return edits

    # Build edit descriptions for the optimizer
    edit_descs: list[str] = []
    for i, edit in enumerate(edits):
        edit_descs.append(f"[{i}] {describe_edit(edit)}")

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Proposed Edits ({len(edits)} total, budget={max_edits})\n\n"
        + "\n".join(edit_descs)
        + f"\n\nRank all {len(edits)} edits by importance (score 1-10). "
        f"Return JSON with `rankings` array."
    )

    try:
        response, _meta = optimizer(_RANKER_SYSTEM, user, {
            "max_tokens": optimizer_max_tokens,
            "stage": "select",
        })
    except Exception:
        # Fallback: return first max_edits
        return edits[:max_edits]

    if not response:
        return edits[:max_edits]

    rankings = _parse_rankings(response, len(edits))
    if not rankings:
        return edits[:max_edits]

    # Sort edits by score descending, pick top max_edits
    indexed = list(enumerate(edits))
    indexed.sort(key=lambda ie: rankings.get(ie[0], 5), reverse=True)
    selected = [edit for _idx, edit in indexed[:max_edits]]

    # Preserve original order within selected set
    original_indices = {id(e): i for i, e in enumerate(edits)}
    selected.sort(key=lambda e: original_indices.get(id(e), 0))

    return selected


def _parse_rankings(response: str, n_edits: int) -> dict[int, int]:
    """Extract {index: score} mapping from optimizer response.

    Robust against markdown fences and partial JSON.
    """
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
        # Try to find a JSON object
        import re
        match = re.search(r'\{.*"rankings"\s*:\s*\[.*?\].*?\}', text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    rankings_list = data.get("rankings", [])
    if not isinstance(rankings_list, list):
        return {}

    result: dict[int, int] = {}
    for item in rankings_list:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", item.get("id", -1))
        score = item.get("score", item.get("priority", 5))
        try:
            idx = int(idx)
            score = int(float(score))
            score = max(1, min(10, score))
            if 0 <= idx < n_edits:
                result[idx] = score
        except (ValueError, TypeError):
            continue

    return result
