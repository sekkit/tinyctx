# tinyctx Functional Test Points

This document is the functional coverage map for the current test suite. It is
organized by product capability and links each coverage point to the test file
that guards it. Keep this document updated whenever a new `tests/test_*.py`
file is added.

## Routing and Proxy

- **Route decision engine** — local/frontier decisions, compaction routing, token/turn thresholds, adaptive local-failure routing, classifier fallback, advisor-only self-classify recommendations, and model capability defaults. Covered by `tests/test_router.py`, `tests/test_dynamic_thresholds.py`, `tests/test_classifier.py`, `tests/test_self_classify.py`, `tests/test_adaptive_model.py`.
- **Proxy forwarding contract** — `/v1/responses`, `/v1/chat/completions`, forced route model IDs, encrypted reasoning scrub, local/frontier fake-backend routing, retry policy, and compactor fallback behavior. Covered by `tests/test_proxy_integration.py`, `tests/test_proxy_retry.py`, `tests/test_proxy_compactor_integration.py`.
- **Frontier optimization path** — frontier-only tool trimming, advisor-hint suppression, and proactive compaction discipline for expensive models. Covered by `tests/test_frontier_optimizations.py`, `tests/test_proactive_compact.py`.
- **LMCache passthrough path** — opt-in preservation of `prompt_cache_key` for external vLLM/SGLang+LMCache stacks while default strict-backend stripping remains unchanged. Covered by `tests/test_lmcache_passthrough.py`.

## Context Compression and Continuity

- **Compaction quality and payload shape** — multi-role compactor debate, judge merge, fallback concat, structured sidecar parsing, Responses payload/SSE output, and pristine recomputation guard. Covered by `tests/test_compactor.py`.
- **Session continuity** — persisted compaction summaries, `latest.md`, structured JSON sidecars, recall CLI modes, per-session ordering, and empty-repo behavior. Covered by `tests/test_continuity.py`.
- **Rolling history management** — historian update/substitution, proactive compact cache behavior, read-delta collapse, key file detection, and compression-biased ranking. Covered by `tests/test_historian.py`, `tests/test_proactive_compact.py`, `tests/test_read_delta.py`, `tests/test_keypin.py`, `tests/test_interest.py`.
- **Project/session isolation** — project-scoped session keys, per-project proactive cache isolation, error-streak isolation, conversation fingerprints, and stable prompt-cache identity. Covered by `tests/test_project_isolation.py`, `tests/test_conv_id.py`.
- **Scout and memory context** — auto-scout bootstrap/injection, scout cache/status CLI, graph ranking, mem0 ingestion, and registry-backed dreamer runs. Covered by `tests/test_auto_scout.py`, `tests/test_scout.py`, `tests/test_memory.py`, `tests/test_dreamer.py`, `tests/test_skill_catalog.py`.
- **Plan persistence** — saving and injecting cross-thread plans with TTL and repo scoping. Covered by `tests/test_plan_persistence.py`.

## Stream Reliability and Recovery

- **Streaming relay and terminators** — valid SSE forwarding, response completion terminators, upstream error handling, post-stream handling, and stream rewrite synthesis. Covered by `tests/test_stream_relay.py`, `tests/test_stream_terminator.py`, `tests/test_post_stream.py`, `tests/test_stream_rewrite.py`.
- **Keepalive and request phases** — keepalive events, request lifecycle phase tracking, and active-state observability. Covered by `tests/test_keepalive.py`, `tests/test_request_phase.py`.
- **Soft completion and stuck-loop recovery** — LLM-based punt detection, auto-force frontier flags, synthetic continue events, budget reminders, advisor grace windows, and state reset after compaction. Covered by `tests/test_soft_completion.py`, `tests/test_synthetic_continue.py`, `tests/test_stuck_loop.py`.
- **Empty response and stall recovery** — empty/length-truncated response detection, force-frontier flags, stall timestamp tracking, cancellation, and dashboard state export. Covered by `tests/test_empty_response_guard.py`, `tests/test_stall_watchdog.py`.
- **Exec resume poke** — resolving Codex session IDs, rate limits, prompt tiers, subprocess spawning, and dry-run/error behavior. Covered by `tests/test_exec_resume.py`.

## Bootstrap and Integrations

- **Codex configuration patching** — TOML marker handling, profile/provider install blocks, hook registration, idempotence, dry-run, uninstall, and concurrent appends. Covered by `tests/test_codex_toml.py`, `tests/test_codex_profile_bootstrap.py`, `tests/test_scout_hook_bootstrap.py`.
- **Advisor integration** — Advisor MCP schema, JSON-RPC handling, auth resolution, SSE accumulation, HTTP/network error surfacing, advisor bootstrap install/uninstall/status, and dynamic agent-rule injection. Covered by `tests/test_advisor.py`, `tests/test_advisor_bootstrap.py`, `tests/test_agent_rules.py`.
- **Advisor continuation** — pending work extraction from advisor SSE output, LLM judge verdict pipeline, work injection into request body, guard firing/consuming logic, and forensics regression fixtures. Covered by `tests/test_advisor_continuation.py`, `tests/test_advisor_continuation_regression.py`.
- **Choice arbiter** — text/thinking stripping, YES/NO verdict parsing, salvage fallback, and guard integration with stored verdict. Covered by `tests/test_choice_arbiter.py`, `tests/test_choice_arbiter_integration.py`.
- **Multimodal preprocessing** — image-to-caption pipeline, binary detection/bootstrap, cache hits, remote-URL passthrough, and MM failure fallback. Covered by `tests/test_multimodal_preprocess.py`, `tests/test_mm_bootstrap.py`.
- **Pydoc MCP server** — package resolution, symbol documentation, depth control, output truncation, JSON-RPC dispatch, stdio serve loop, and bootstrap install/detect. Covered by `tests/test_pydoc_mcp.py`, `tests/test_pydoc_mcp_bootstrap.py`.
- **Response verifier** — LLM-as-verifier pipeline, background verification tasks, integration with proxy stream output, and end-to-end pipeline shape. Covered by `tests/test_verifier.py`, `tests/test_verifier_integration.py`, `tests/test_verifier_pipeline.py`.
- **External MCP/bootstrap tooling** — graphify, serena, gitnexus, caveman, and MCP registry install/detect/config behavior. Covered by `tests/test_graphify_bootstrap.py`, `tests/test_graphify_adapter.py`, `tests/test_serena_bootstrap.py`, `tests/test_gitnexus_bootstrap.py`, `tests/test_caveman_bootstrap.py`, `tests/test_mcp_registry.py`.
- **Tool-call translation** — XML function-call parsing, chat-to-responses rebuilds, stream translator state, unknown-tool protection, advisor auto-answer, text-choice interception, and deterministic Responses fixture replay. Covered by `tests/test_tool_call_translator.py`, `tests/test_responses_fixture_replay.py`.
- **Orchestration and task guidance** — dynamic skill validation/rendering, orchestration injection/runtime, orchestrator config, task planning, and task supervision. Covered by `tests/test_dynamic_skill.py`, `tests/test_orchestration_injector.py`, `tests/test_orchestration_runtime.py`, `tests/test_orchestrator_config.py`, `tests/test_task_orchestrator.py`, `tests/test_task_supervisor.py`.
- **Tool metrics and integration workflow** — tool-call frequency snapshots, call-id deduping, path coverage, and end-to-end integration workflow shape. Covered by `tests/test_tool_metrics.py`, `tests/test_path_coverage_e2e.py`, `tests/test_integration_workflow.py`.
- **Task orchestration and dynamic skills** — orchestrator config, task supervision, orchestration runtime/injection, dynamic skill loading, and skill catalog discovery. Covered by `tests/test_orchestrator_config.py`, `tests/test_task_orchestrator.py`, `tests/test_task_supervisor.py`, `tests/test_orchestration_runtime.py`, `tests/test_orchestration_injector.py`, `tests/test_dynamic_skill.py`, `tests/test_skill_catalog.py`.
- **Lingua compression** — optional token compression policy, input filtering, and frontier compression hooks. Covered by `tests/test_lingua.py`.
- **Registry helpers** — project registration and lookup for recurring maintenance flows. Covered by `tests/test_registry.py`.

## Observability and Operations

- **Prompt cache extraction** — provider-specific prompt-cache token field extraction across DeepSeek, OpenAI Responses, OpenAI Chat, and Anthropic wire shapes. Covered by `tests/test_prompt_cache_extract.py`.
- **Trace and stats** — JSONL trace emission, filters, compact/verbose rendering, quality scoring, route aggregates, and no-data behavior. Covered by `tests/test_trace.py`, `tests/test_stats.py`.
- **Dashboard surfaces** — HTML dashboard, JSON state, aggregate rollups, recent events, manual escalation API, request-phase state, and visual configuration center APIs. Covered by `tests/test_dashboard.py`, `tests/test_dashboard_api.py`, `tests/test_dashboard_config_api.py`.
- **Forensics capture** — request/response dumps for empty responses, punts, upstream errors, retention limits, and redaction/shape guarantees. Covered by `tests/test_forensics.py`.
- **Operational hygiene** — hardcoded-path guards and platform/path coverage. Covered by `tests/test_no_hardcoded_paths.py`, `tests/test_path_coverage_e2e.py`.

## Configuration and Safety

- **Configuration model** — `Config`/`BackendCfg` defaults, TOML and environment overrides, namespaced views, fresh-instance isolation, context-window thresholds, retry/stall/forensics/guard defaults, visual config schema/presets, atomic TOML persistence, and verbose/log-dir behavior. Covered by `tests/test_config.py`, `tests/test_config_io.py`, `tests/test_config_schema.py`.
- **Sanitization and history hygiene** — unsupported request fields/tools, Responses defaults/caps, encrypted content stripping, chat normalization, deduping repeated tool calls, failed-input purging, cache-aware mutation gates, and orphan tool-output handling. Covered by `tests/test_sanitize.py`, `tests/test_sanitize_dedup.py`.
- **Guard rails** — policy guards, malformed request handling, and safety defaults. Covered by `tests/test_guards.py`.
- **Escalation ladder** — graduated REFINE→PIVOT→SEARCH→BLOCKER escalation with per-session failure/pivot counters, strategy-change reminders, and frontier force-routing at PIVOT level. Covered by `tests/test_escalation.py`.
- **Unified session state** — counters, flags, timestamps, bounded history, namespace isolation, snapshots, compaction resets, and falsy-session no-ops. Covered by `tests/test_session_state.py`.
- **Self-improvement primitives** — filesystem session workspaces, context profiles, replayable trajectory ledgers, eval aggregation, candidate frontier scoring, and staged guardrail registries. Covered by `tests/test_self_improvement_primitives.py`.

## Documentation Coverage Guard

- **Coverage map completeness** — every `tests/test_*.py` file must be listed here, and required functional sections must remain present. Covered by `tests/test_functional_test_points_doc.py`.
