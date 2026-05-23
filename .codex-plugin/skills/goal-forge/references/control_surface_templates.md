# Human Control Surface Templates

Use these templates when a `/goal` may run for hours, may be reviewed by a human or sidecar agent in parallel, consumes scarce resources, or may need tunable priorities during execution.

The control surface is not a second `GOAL.md`. It is a compact operator panel. Select only the knobs relevant to the goal.

## CONTROL.md

```md
# CONTROL

## Status Contract

status_file: PLAN.md
attempt_log: ATTEMPTS.md
durable_notes: NOTES.md
update_memory_after: every_experiment
check_control_before: phase_change, strategic_pivot, expensive_step

## Human Priorities

primary_priority: [quality | speed | stability | cost | behavior_preservation | visual_polish | evidence_quality]
secondary_priority: [optional]

## Scope Knobs

allowed_files:
- [path or glob]

protected_files:
- [path or glob]

max_blast_radius: [module / package / app / explicit description]

## Resource Knobs

max_runtime_per_step: [duration or none]
max_parallel_jobs: [number or none]
network_allowed: [true | false | approval_required]
external_api_allowed: [true | false | approval_required]

## Decision Gates

require_approval_for:
- strategic_pivot
- destructive_change
- dependency_change
- schema_or_migration_change
- public_api_change
- scope_expansion

## Sidecar Inputs

sidecar_apply_cadence: [before_phase_change | between_runs_only | manual_approval_only]
nudge_file: [path or none]
human_overlay_file: [path or none]
review_queue_file: [path or none]

## Latest Human Nudge

[Short note or path to detailed nudge.]
```

## Knob Taxonomy

Use the taxonomy below to choose goal-specific knobs. Do not include every category by default.

### Visibility

- `current_phase`
- `active_hypothesis`
- `next_action`
- `latest_result`
- `blockers`
- `confidence_level`

### Cadence

- `update_memory_after`
- `pause_after`
- `check_control_before`
- `report_status_every`

### Scope

- `allowed_files`
- `protected_files`
- `no_touch_zones`
- `max_blast_radius`
- `acceptable_refactors`

### Quality Bar

- `primary_priority`
- `acceptance_thresholds`
- `regression_tolerance`
- `review_required_for`

### Resource Budget

- `max_runtime_per_step`
- `max_cost`
- `max_parallel_jobs`
- `network_allowed`
- `external_api_allowed`
- `scarce_resource_locks`

### Decision Gates

- `require_approval_for_strategic_pivots`
- `require_approval_for_destructive_changes`
- `require_approval_for_schema_changes`
- `require_approval_for_dependency_changes`
- `require_approval_for_scope_expansion`

### Sidecar Inputs

- `human_overlay_file`
- `nudge_file`
- `review_queue_file`
- `external_findings_file`
- `sidecar_apply_cadence`

## Goal Block Snippet

```xml
<human_control_surface>
Create and maintain `CONTROL.md` as the compact human operator panel for this goal.

Before each phase change, strategic pivot, expensive step, or sidecar ingestion, reread `CONTROL.md`. If it changed, summarize the relevant change in `PLAN.md` and adapt before proceeding.

The initial `CONTROL.md` should include only knobs relevant to this goal:
- status files and update cadence
- human priorities
- scope/protected files
- resource limits or locks
- decision gates requiring approval
- sidecar inputs and apply cadence

Do not treat `CONTROL.md` as permission to ignore `GOAL.md`; it can narrow priorities, pause work, or require approval, but it cannot silently weaken `done_when` or scorecard thresholds.
</human_control_surface>
```
