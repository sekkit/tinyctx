# Working Memory Templates

Use these templates when scaffolding long-running or exploratory `/goal` work. Rename them only when the target repo already has a clear equivalent convention.

## PLAN.md

```md
# PLAN

## Goal

[One-sentence goal summary.]

## Current Strategy

[The current approach Codex is following.]

## Phases

- [ ] Inspect relevant code and constraints.
- [ ] Implement the smallest viable change.
- [ ] Run the fast feedback check.
- [ ] Refine until the scorecard threshold is met.
- [ ] Run final verification.

## Open Decisions

- [Decision or blocker that may require user input.]
```

## ATTEMPTS.md

```md
# ATTEMPTS

| Time | Attempt | Evidence | Result | Next Adjustment |
| --- | --- | --- | --- | --- |
| [timestamp] | [What changed or was tried.] | [Command, metric, diff, or observation.] | [Passed/failed/partial.] | [What to do differently next.] |
```

## NOTES.md

```md
# NOTES

## Chronological Notes

- [timestamp] [Discovery, blocker, hypothesis, or useful context.]
```

## Update Cadence

- Update `PLAN.md` when the strategy or phase changes.
- Update `ATTEMPTS.md` after each meaningful implementation attempt, failed experiment, or successful scoring improvement.
- Update `NOTES.md` whenever Codex discovers durable context that should survive compaction.
