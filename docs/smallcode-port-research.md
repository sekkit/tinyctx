# SmallCode Port Research

Source: sibling clone `smallcode` at commit `456713f`.

## Fit Summary

SmallCode is a full JavaScript coding agent optimized for 8B-35B local models. tinyctx is a Python wire proxy for Codex, so only proxy-safe, deterministic features should be ported. Features that require owning the agent loop, editing tools, or a custom TUI are poor fits.

## Already Covered In tinyctx

- Context budget: proactive compaction, read-delta, result shrink, frontier tool trimming, LLMLingua opt-in.
- Tool robustness: Responses/chat translation, XML/tool-call parser, unknown-tool replacement, orphan output drop.
- Escalation: error streaks, failure-signal scan, self-classify advisor recommendation, goal-control frontier routing.
- Memory/context: continuity, mem0 wrapper, scout, graphify/gitnexus adapters, historian.
- Loop recovery: stuck-loop reminders, soft-completion gate, empty-response guard, exec-resume.
- Observability: request traces, stats, dashboard, forensics.

## Ported

### Adaptive Model Select

SmallCode tracks model call/fail counts and switches to stronger models when the primary model gets unhealthy. tinyctx now ports the safe subset:

- Tracks a rolling local-backend outcome window in `tinyctx/adaptive_model.py`.
- Records non-streaming and streaming backend successes/failures.
- Routes future automatic requests to frontier when local failure rate exceeds `adaptive_model_failure_rate_threshold`.
- Keeps compaction, force route, and explicit `tinyctx-local` / `tinyctx-frontier` precedence intact.
- Exposes trace fields: calls, failures, failure rate, triggered.

## Rejected For Now

- BoneScript / MarrowScript: SmallCode-specific runtime layers, not useful for a Codex proxy.
- Patch-first editing, semantic merge, read-before-write: Codex owns editing tools; tinyctx should not rewrite tool semantics.
- TUI, plugin/session shell features: agent-shell concerns outside tinyctx's proxy boundary.
- Trace-to-test: useful and portable, but lower leverage than adaptive routing; should be a later offline CLI addition.
