"""ReflACT — skill optimization by trajectory-driven edit training.

Full-feature SkillOpt-style ReflACT pipeline adapted for tinyctx:

  1. Rollout   — execute eval cases with current skill, collect trajectories
  2. Reflect   — optimizer LLM analyzes failure/success trajectories → edit proposals
  3. Aggregate — hierarchical LLM merge of patches into coherent patch
  4. Select    — LLM ranking + clip to edit_budget (cosine/linear/constant/autonomous scheduler)
  5. Update    — apply edits to skill document (patch or full-rewrite mode)
  6. Evaluate  — validate candidate against holdout → accept/reject

Plus:
  - Cross-epoch meta-skill learning
  - Step buffer: rejected edit history fed to optimizer
  - Autonomous learning rate mode

Key design choices vs upstream SkillOpt:
  - Zero new dependencies — optimizer calls reuse tinyctx's backend
  - Trajectories from existing trajectory.py JSONL
  - Skill document is the AGENTS.md template (agent_rules.py)
  - Inline prompt templates (no external prompt file dependency)

Usage:
    from tinyctx.reflact import train_skill, TrainConfig

    cfg = TrainConfig(skill_path="/path/to/AGENTS.md", out_dir="/tmp/out", epochs=3)
    result = train_skill(cfg, optimizer=my_optimizer, rollout_fn=my_rollout_fn)
"""

from .trainer import TrainConfig, TrainResult, train_skill, load_skill_from_agents_md
from .reflect import (
    analyze_failures,
    analyze_successes,
    fmt_trajectory,
    fmt_rollout_result,
    merge_edits,
)
from .update import apply_edit, apply_patch, describe_edit
from .scheduler import build_scheduler
from .select import rank_and_select
from .aggregate import merge_patches as hierarchical_merge_patches
from .comparison import (
    build_comparison_pairs,
    format_comparison_context,
    run_slow_update,
    inject_empty_slow_update,
    replace_slow_update_content,
    extract_slow_update_content,
)
from .meta_skill import run_meta_skill, format_meta_skill_context

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train_skill",
    "load_skill_from_agents_md",
    # Reflect
    "analyze_failures",
    "analyze_successes",
    "fmt_trajectory",
    "fmt_rollout_result",
    "merge_edits",
    # Aggregate
    "hierarchical_merge_patches",
    # Select
    "rank_and_select",
    # Scheduler
    "build_scheduler",
    # Comparison (slow update / longitudinal pairs)
    "build_comparison_pairs",
    "format_comparison_context",
    "run_slow_update",
    "inject_empty_slow_update",
    "replace_slow_update_content",
    "extract_slow_update_content",
    # Meta-skill
    "run_meta_skill",
    "format_meta_skill_context",
    # Update
    "apply_edit",
    "apply_patch",
    "describe_edit",
]
