# Self-Improvement Implementation Plan

## Goal

Build tinyctx self-improvement as a governed release loop rather than live
self-mutation:

1. Capture replayable session evidence.
2. Evaluate candidate context policies against held-out replays.
3. Keep a scored frontier of policy variants.
4. Promote only candidates that pass guardrails and regressions.

## P0 Slice

This slice adds small, testable primitives that other modules can adopt without
rewiring proxy streaming paths yet:

- `workspace`: filesystem-backed `.tinyctx` session/profile contract.
- `trajectory`: append-only event ledger for route/sanitize/compact/tool/memory
  decisions.
- `eval_harness`: deterministic replay/evaluation runner with aggregate metrics.
- `frontier`: versioned candidate archive with lineage and weighted scoring.
- `guardrail_registry`: plugin-style guardrail checks for staged evaluation.
- `self_improvement`: governed candidate eval loop that records trajectories and
  archives scored candidates.

## Follow-On Wiring

- Record proxy/router/sanitize/compaction events into the trajectory ledger.
- Surface workspace, trajectory, eval, frontier, and guard summaries in the
  dashboard.
- Convert existing guard checks into registry plugins where it clarifies
  ordering and reporting.
- Add golden replay suites for context savings, sanitizer leaks, memory recall,
  graphify relevance, and MCP/tool failures.

## Implemented Runtime Wiring

- Proxy `_log` records session-scoped events into the trajectory ledger even
  when legacy verbose JSONL logging is disabled.
- `/dashboard/self-improvement` exposes context profile, known sessions,
  trajectory summaries, and frontier candidates for a selected session.
- `self_improvement.evaluate_candidate()` runs bounded evals, records start/end
  events, archives the scored candidate, and reports whether it is the current
  weighted winner.
