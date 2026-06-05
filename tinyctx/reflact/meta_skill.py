"""Meta-skill — cross-epoch optimizer guidance.

Analogous to meta-learning in neural network training: periodically asks the
optimizer to reflect on its own optimization behavior across epochs and write
a meta-skill document. This document is then fed into future optimizer calls
so the optimizer learns to optimize better over time.

Run every `meta_skill_interval` epochs (default: every epoch after the first).
"""

from __future__ import annotations

from typing import Any, Callable

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

_META_SKILL_SYSTEM = """You are a meta-optimizer. Your job is to analyse the optimization history
across multiple epochs and produce a short guidance document for the optimizer
LLM that will run future optimization steps.

Focus on:
1. What patterns in the trajectories repeatedly lead to useful edits?
2. What kinds of edits get rejected and should be avoided?
3. What constraints or rules should the optimizer follow to produce better edits?
4. Is the skill growing too fast (verbose) or too conservatively?

Write concise guidance (50-200 words) that will be prepended to future optimizer
system prompts. The guidance should be actionable and concrete, not abstract.

Return JSON:
{
  "meta_skill": "the guidance text",
  "observations": ["list of key observations from the training history"],
  "recommendations": ["list of specific recommendations for the optimizer"]
}"""


def run_meta_skill(
    *,
    skill_content: str,
    history: list[dict[str, Any]],
    current_meta_skill: str = "",
    optimizer: OptimizerFn,
    max_tokens: int = 4096,
    epoch: int = 0,
) -> dict[str, Any]:
    """Generate or refine a meta-skill document.

    Args:
        skill_content: current skill document
        history: training history records from completed steps
        current_meta_skill: previously generated meta-skill (to refine)
        optimizer: callable (system, user, options) → (response, metadata)
        max_tokens: max completion tokens
        epoch: current epoch number (for logging)

    Returns:
        dict with {meta_skill, observations, recommendations}
    """
    # Summarize history for the meta-optimizer
    history_lines: list[str] = []
    for rec in history[-30:]:  # last 30 steps
        action = rec.get("action", "?")
        edits = rec.get("n_edits", 0)
        applied = rec.get("n_applied", 0)
        score = rec.get("current_score", 0)
        best = rec.get("best_score", 0)
        n_fail = rec.get("n_failures", 0)
        n_succ = rec.get("n_successes", 0)
        history_lines.append(
            f"  step={rec['step']:3d} {action:22s} "
            f"score={score:.3f} best={best:.3f} "
            f"edits={edits}/{applied} failures={n_fail} successes={n_succ}"
        )

    user = (
        f"## Current Skill ({len(skill_content)} chars)\n"
        f"First 500 chars: {skill_content[:500]}\n\n"
        f"## Training History ({len(history)} steps, epoch {epoch})\n"
        + "\n".join(history_lines)
    )

    if current_meta_skill:
        user += f"\n\n## Current Meta-Skill (refine this)\n{current_meta_skill}"
    else:
        user += "\n\nNo current meta-skill exists. Write the first version."

    try:
        response, _meta = optimizer(_META_SKILL_SYSTEM, user, {
            "max_tokens": max_tokens,
            "stage": "meta_skill",
        })
    except Exception:
        return {"meta_skill": current_meta_skill, "observations": [], "recommendations": []}

    if not response:
        return {"meta_skill": current_meta_skill, "observations": [], "recommendations": []}

    return _parse_meta_skill(response, current_meta_skill, epoch)


def _parse_meta_skill(response: str, fallback: str, epoch: int) -> dict[str, Any]:
    """Extract meta-skill content from optimizer response."""
    import json as _json
    import re

    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        match = re.search(r'\{.*"meta_skill".*\}', text, re.DOTALL)
        if match:
            try:
                data = _json.loads(match.group(0))
            except _json.JSONDecodeError:
                return {"meta_skill": fallback, "observations": [], "recommendations": []}
        else:
            return {"meta_skill": fallback, "observations": [], "recommendations": []}

    meta = str(data.get("meta_skill", fallback)).strip()
    obs = data.get("observations", [])
    recs = data.get("recommendations", [])
    return {
        "meta_skill": meta or fallback,
        "observations": obs if isinstance(obs, list) else [],
        "recommendations": recs if isinstance(recs, list) else [],
        "epoch": epoch,
    }


def format_meta_skill_context(meta_skill_text: str) -> str:
    """Format a meta-skill text for prepending to optimizer prompts.

    Returns empty string if the meta-skill is empty or None.
    """
    if not meta_skill_text or not meta_skill_text.strip():
        return ""
    return (
        "## Meta-Skill Guidance (from previous optimization epochs)\n\n"
        f"{meta_skill_text.strip()}\n\n"
        "---\n"
    )
