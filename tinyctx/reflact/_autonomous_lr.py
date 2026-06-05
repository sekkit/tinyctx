"""Autonomous learning rate — optimizer decides its own edit budget.

The optimizer examines the proposed edits, current skill, and rollout
performance to decide how many edits to apply this step.

Unlike fixed schedulers, this adapts to the quality and quantity of edits:
  - Many high-confidence edits + low score → apply more
  - Few speculative edits + high score → apply fewer
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .update import describe_edit

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

_AUTONOMOUS_LR_SYSTEM = """You are an optimization rate controller. Given a set of proposed edits
to an agent skill document and the current rollout performance, decide how
many edits to apply this optimization step.

Consider:
- Low rollout score → be more aggressive (apply more edits)
- High rollout score → be conservative (apply fewer edits, fine-tune)
- Many high-quality edits with strong evidence → apply more
- Edits that seem speculative or overlapping → apply fewer
- If the skill is already long, prefer replace/delete over append

Return JSON:
{
  "learning_rate": <integer 0-N, number of edits to apply>,
  "reasoning": "why you chose this number"
}"""


def decide_autonomous_learning_rate(
    *,
    skill_content: str,
    edits: list[dict[str, Any]],
    rollout_score: float,
    rollout_n: int,
    optimizer: OptimizerFn,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Ask the optimizer to choose the edit budget for this step.

    Returns dict with at least {learning_rate: int, reasoning: str}.
    learning_rate is clamped to [0, len(edits)].
    """
    available = len(edits)
    if available == 0:
        return {"learning_rate": 0, "reasoning": "no edits available"}

    item_lines = [
        f"[{idx}] {describe_edit(edit)}"
        for idx, edit in enumerate(edits)
    ]

    user = (
        f"## Current Skill ({len(skill_content)} chars)\n{skill_content[:800]}...\n\n"
        f"## Step Performance\n"
        f"rollout_score={rollout_score:.4f}\n"
        f"rollout_n={rollout_n}\n"
        f"proposed_edits={available}\n\n"
        f"## Proposed Edits\n"
        + "\n".join(item_lines)
        + f"\n\nDecide how many of these {available} proposed edits should be applied."
    )

    try:
        response, _meta = optimizer(_AUTONOMOUS_LR_SYSTEM, user, {
            "max_tokens": max_tokens,
            "stage": "autonomous_lr",
        })
    except Exception:
        return {"learning_rate": min(available, 3), "reasoning": "optimizer error, default=3"}

    text = response.strip() if response else ""
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"learning_rate"\s*:\s*\d+[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"learning_rate": min(available, 3), "reasoning": "parse failed, default=3"}
        else:
            # Try to extract just a number
            numbers = re.findall(r'\b(\d+)\b', text)
            if numbers:
                val = int(numbers[0])
                return {"learning_rate": max(0, min(val, available)),
                        "reasoning": "extracted from response"}
            return {"learning_rate": min(available, 3), "reasoning": "no number found, default=3"}

    lr = data.get("learning_rate", 3)
    try:
        lr = int(float(str(lr)))
    except (ValueError, TypeError):
        lr = 3
    return {
        "learning_rate": max(0, min(lr, available)),
        "reasoning": str(data.get("reasoning", "")),
    }
