# tinyctx architecture

This is the design map for tinyctx. Read it before diving into `proxy.py`
(which is ~2100 LOC of HTTP entry + lifecycle wiring). After P0-P8 the
heavy logic now lives in focused modules; `proxy.py` is the glue.

Audience: contributors who need to add a guard, a retry trigger, a
forensics dump, or a piece of per-conversation state without reading
the whole tree.

---

## 1. What tinyctx is

tinyctx is a local-first routing proxy between the OpenAI Codex CLI
(or any Responses-API client) and one or more LLM backends. Codex talks
to it as if it were `chatgpt.com/backend-api/codex`; tinyctx looks at
the body and decides whether the turn should go to a cheap local
backend (DeepSeek-v4-flash, LMStudio + Qwen3.6-27B, Ollama, vLLM) or
to the real frontier model (gpt-5.5). Sanitization, retries, stall
detection, history compaction and stream translation all happen
between codex and the chosen backend.

The goal is to land ~99% of turns on the cheap backend without the
user noticing — the frontier is reserved for hard decisions the local
model can't make. The mechanisms that make this safe are
interruption-resilience: every place an upstream can fail (connection
drop, 4xx, 5xx, mid-stream silence, empty response, soft-punt to user,
runaway turn count) has an explicit detector and recovery path.

The proxy is single-process FastAPI under uvicorn, single event loop,
no shared mutable state across processes. Per-conversation state lives
in `SessionState` (an in-memory namespaced dict). On-disk state is
limited to logs (`~/.tinyctx/logs/`), forensics dumps
(`~/.tinyctx/forensics/`) and scout/plan-persistence caches
(`~/.tinyctx/cache/`, `~/.tinyctx/state/`).

---

## 2. Request lifecycle

A `POST /v1/responses` request walks through these phases. Each phase
points to the module that owns it. `proxy.responses(...)` is the glue
that calls them in order.

```text
                              HTTP entry
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ proxy.responses             │   proxy.py
                    │  - parse body               │
                    │  - snapshot raw_body        │
                    │  - x-codex-session-id +     │
                    │    x-codex-cwd → proj_sid   │
                    └─────────────┬───────────────┘
                                  │
                  conv_sid = resolve_conv_key(proj_sid, body)
                                  │                 conv_id.py
                                  ▼
                    ┌─────────────────────────────┐
                    │ body-derived signals        │   router.py
                    │  - _flatten_text            │   (count_turns,
                    │  - estimate_tokens          │    estimate_tokens,
                    │  - count_turns              │    is_compaction_request)
                    │  - is_compaction            │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ GuardPipeline               │   guards.py
                    │   1. ForceFrontierGuard     │   (consumes
                    │   2. BudgetReminderGuard    │    empty_response_guard
                    │   3. StuckLoopGuard         │    flags, may inject
                    │   4. SoftCompletionGate     │    <system-reminder>
                    │   5. PlanPersistenceInjector│    into body.input)
                    │  → guard_force_route        │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ self_classify (async)       │   self_classify.py
                    │ local model rates           │
                    │ p(escalate)                 │
                    │  → classify_p, reason       │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Router.decide(RouteContext) │   router.py
                    │  Priority rules:            │
                    │    compaction               │
                    │    force_route              │
                    │    explicit_model           │
                    │    error_streak             │
                    │    capacity                 │
                    │    classify                 │
                    │    default → local          │
                    │  → Decision(route, target,  │
                    │     model, headers,         │
                    │     wire_api, reason)       │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ sanitize pipeline           │   sanitize.py
                    │  strip_encrypted_content    │
                    │  read_delta.collapse        │
                    │  dedup_tool_calls           │
                    │  purge_failed_tool_inputs   │
                    │  historian.apply_to_body    │
                    │  proactive_compact          │
                    │  auto_scout.inject          │
                    │  agent_rules.inject         │
                    │  inject_advisor_hint        │
                    │  rewrite_model              │
                    │  expand_mcp_namespaces      │
                    │  scrub_unsupported_tools    │
                    │  inject_responses_defaults  │
                    │  cap_responses_fields       │
                    │  trim_tools_for_frontier    │
                    │  drop_orphan_tool_outputs   │
                    │  normalize_for_chat (if     │
                    │    wire_api != responses)   │
                    └─────────────┬───────────────┘
                                  │
                          forensics request snapshot
                                  │                 forensics.py
                                  ▼
                    ┌─────────────────────────────┐
                    │ _forward / _stream_proxy    │   proxy.py
                    │                             │
                    │ non-stream:                 │
                    │   retry loop using          │
                    │   retry_policy.             │   retry_policy.py
                    │   classify_failure          │
                    │                             │
                    │ stream:                     │
                    │   relay_stream(             │   stream_relay.py
                    │     producer = StreamProducer,
                    │     consumer = StreamConsumer,
                    │     supervisor = StallSupervisor)
                    │                             │
                    │ stall_watchdog observes     │   stall_watchdog.py
                    │ silence; cancels producer   │
                    │ on threshold; consumer      │
                    │ emits terminator            │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Stream-rewrite synthesis    │   stream_rewrite.py
                    │ (hold response.completed,   │   soft_completion.py
                    │  classify, optionally       │   synthetic_continue.py
                    │  inject synthetic           │
                    │  function_call, flush)      │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ PostStreamAnalyzer          │   post_stream.py
                    │  - spawn bg classifier      │
                    │  - empty_response_guard     │   empty_response_guard.py
                    │  - forensics dumps          │   forensics.py
                    │  - exec_resume poke         │   exec_resume.py
                    │ RelayErrorTerminator        │
                    │  (StallCancelledError,      │
                    │   httpx.HTTPError)          │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                            SSE bytes to codex
                                  │
                                  ▼
                          RequestTrace.emit      trace.py
```

A single request can fire multiple retries, cancel a producer mid-stream,
and still emit one structurally valid SSE close. The bytes the client
sees are the consumer's output, separately captured for forensics.

---

## 3. Module map

`tinyctx/` after P0-P8 contains ~50 modules. The table groups them by
role. "core path" = touched by every (or most) requests; "support" =
CLI tools, bootstraps, optional features.

| Module                       | Responsibility                                                                 | Role        |
| ---------------------------- | ------------------------------------------------------------------------------ | ----------- |
| `proxy.py`                   | FastAPI server + request lifecycle glue (~2100 LOC; HTTP entry only)           | core path   |
| `router.py`                  | `Router.decide(ctx)`, RouteContext, Decision, 7 priority rules                 | core path   |
| `guards.py`                  | `GuardPipeline` + 5 pre-flight guards                                          | core path   |
| `session_state.py`           | Unified per-conv namespaced state store                                        | core path   |
| `conv_id.py`                 | `resolve_conv_key` — stable fingerprint resistant to `prompt_cache_key` drift  | core path   |
| `sanitize.py`                | 15+ body transforms (~1640 LOC); CacheAwareMutator gate; proactive_compact     | core path   |
| `retry_policy.py`            | Pure `classify_failure(...)` → retry_same / retry_escalate / propagate         | core path   |
| `stream_relay.py`            | StreamProducer + StreamConsumer + StallSupervisor + `relay_stream`             | core path   |
| `stall_watchdog.py`          | Background task; cancel in-flight producer on mid-stream silence               | core path   |
| `stream_rewrite.py`          | Intercept `response.completed`; inject synthetic continue when classifier punts | core path  |
| `synthetic_continue.py`      | Build noop tool-call SSE events; rotate strategies; enforce per-conv budget    | core path   |
| `soft_completion.py`         | Output-buffer accumulator + LLM-based punt classifier; gate injection         | core path   |
| `self_classify.py`           | Local model rates `p(escalate)` for the current turn (pre-flight)              | core path   |
| `empty_response_guard.py`    | Detect near-empty stream tail; flag next turn to frontier                     | core path   |
| `stuck_loop.py`              | High turn-count watchdog; inject `<system-reminder>`                          | core path   |
| `plan_persistence.py`        | Save/load codex `update_plan` per cwd; inject `<persisted-plan>`              | core path   |
| `post_stream.py`             | PostStreamAnalyzer + RelayErrorTerminator + ForensicsPolicy                   | core path   |
| `forensics.py`               | Write full request/response JSON dumps on triggered failures                  | core path   |
| `request_phase.py`           | Per-request lifecycle phase enum + setter                                     | core path   |
| `tool_call_translator.py`    | XML→struct (qwen-pythonic), chat→Responses SSE, 3-layer auto-answer           | core path   |
| `tool_metrics.py`            | Per-tool call counters mined from body.input (dashboard fuel)                 | core path   |
| `agent_rules.py`             | Prepend bundled global AGENTS.md into `body.instructions`                     | core path   |
| `auto_scout.py`              | Lazy scout build + inject `scout.md` into instructions on later turns         | core path   |
| `config.py`                  | TOML + env loading; namespaced sub-views (RoutingView, StallView, …)          | core path   |
| `trace.py`                   | `RequestTrace` dataclass + `tinyctx-trace` CLI                                | core path   |
| `mcp_registry.py`            | Auto-register graphify/gitnexus into `~/.codex/config.toml` on startup        | support     |
| `dashboard.py`               | `/dashboard` SSE/JSON endpoints; vanilla HTML page                            | support     |
| `compactor.py`               | 3-role debate + judge compaction (when `compactor_debate=true`)               | support     |
| `continuity.py`              | Persist compaction summaries; `tinyctx-recall` CLI                            | support     |
| `historian.py`               | Async rolling per-session digest; opt-in substitute on the wire               | support     |
| `read_delta.py`              | 2nd+ Read of same path → unified diff                                         | support     |
| `lingua.py`                  | LLMLingua-2 pre-escalation compression (opt-in)                               | support     |
| `interest.py`                | Compression-biased PageRank ranker (paper §5.1)                               | support     |
| `graphify_adapter.py`        | graphify graph.json → interest.py shape                                       | support     |
| `scout.py`                   | One-shot project summary builder (local LLM call)                             | support     |
| `scout_hook_bootstrap.py`    | Register SessionStart hook in `~/.codex/hooks.json`                           | support     |
| `keypin.py`                  | Mine codex rollouts for Read-frequency; emit `keyfiles.md`                    | support     |
| `memory.py`                  | mem0 wrapper for cross-session memory                                         | support     |
| `dreamer.py`                 | Periodic maintenance CLI (scout + keypin + graphify + mem ingest + GC)        | support     |
| `stats.py`                   | JSONL log → route mix / cost / quality grade                                  | support     |
| `registry.py`                | Per-machine list of touched projects                                          | support     |
| `classifier.py`              | Pure-Python logistic regression for escalation scoring (opt-in)               | support     |
| `advisor.py`                 | Stdio MCP server: `ask_advisor(question)` route to frontier                   | support     |
| `advisor_bootstrap.py`       | Auto-wire advisor MCP into codex config                                       | support     |
| `codex_profile_bootstrap.py` | Auto-write tinyctx provider + normal/goal Codex profiles                     | support     |
| `gitnexus_bootstrap.py`      | Auto-install gitnexus + register MCP                                          | support     |
| `graphify_bootstrap.py`      | Auto-install graphify + per-project codex skill wire                          | support     |
| `serena_bootstrap.py`        | Auto-install serena-agent + register MCP                                      | support     |
| `caveman_bootstrap.py`       | Vendor caveman repo + register caveman-shrink MCP                             | support     |
| `exec_resume.py`             | Side-process `codex exec resume` poker for stuck sessions                     | support     |
| `_codex_toml.py`             | Shared helper for idempotent `~/.codex/config.toml` edits                     | support     |

---

## 4. Interruption-resilience stack

The historical reason most of these modules exist. The original
inspiration (commit `586af76`) was openai/symphony's 5 interruption
patterns; tinyctx now has 7. Each one fires on a different failure
mode the user would otherwise see as a hang or stuck session.

1. **`retry_policy` / retry layer** — `retry_policy.classify_failure(...)`
   is a pure classifier that converts a failure (HTTP status, connection
   error, route, attempt count) into one of `retry_same`,
   `retry_escalate`, `propagate`. The proxy's `_forward` loop and
   `StreamProducer.run` both consult it. Local 5xx retries the same
   backend; local 400/422 escalates to frontier (chatgpt.com tolerates
   shapes LMStudio rejects); frontier 5xx retries; frontier 4xx
   propagates but sets `force_next_to_frontier` so the next turn doesn't
   silently land back on local.

2. **`stall_watchdog`** — background asyncio task; every
   `check_interval_s` it wakes and checks each session's last upstream
   event. If silent for `threshold_s` (default 180s), it cancels the
   registered in-flight producer task. The producer turns that into a
   synthetic `StallCancelledError` pushed onto its queue; the consumer
   emits a clean `event: error` + terminator and the next turn force-routes
   to frontier.

3. **`stream_keepalive`** — `relay_stream` yields `: tinyctx keepalive`
   SSE comment frames every `keepalive_interval_s` of upstream silence,
   plus one initial keepalive BEFORE the producer task runs. Codex.app
   disconnects after ~60s of zero bytes; a cold-start local backend can
   easily exceed that on the first request body upload.

4. **`synthetic_continue`** — when `soft_completion` classifies the
   stream-end response as a "soft punt to user" with high confidence,
   `stream_rewrite` holds back `response.completed` and
   `synthetic_continue.build_continue_injection` emits a noop tool call
   (`shell ["true"]`, `local_shell`, or `update_plan`) BEFORE flushing
   the held terminator. Codex dispatches it, returns the output, model
   resumes mid-thought. Strategy rotates per-conv; bounded by a
   per-conv injection budget.

5. **`empty_response_guard`** — after stream end, parse the SSE buffer
   tail for `usage.completion_tokens`. If it's very small (default <5)
   with `finish_reason ∈ {stop, length}`, the upstream silently
   degraded. Set the per-conv `force_next_to_frontier` flag; the next
   request's `ForceFrontierGuard` consumes it and pins the route.

6. **`stuck_loop`** — if `turn_count >= turn_trigger` (default 80) and
   no advisor call in the last `advisor_grace_s` (default 600s), inject
   a tail `<system-reminder>` telling the agent to either call advisor
   or surface its blocker. Recency-position injection so it survives
   500K-token context. One nudge per `turn_gap` (default 50) turns.

7. **`proactive_compact`** — in `sanitize.py`. When
   `est_tokens >= effective_proactive_compact_threshold(cfg)` and the
   request is NOT itself a codex compaction request, replace the middle
   of `body.input` with a summary item. Threshold defaults to a fraction
   of `frontier.context_window` minus an overhead buffer. The summary is
   either a deterministic placeholder or a real local-model summary
   (cached per-conv so back-to-back turns reuse it). Preserves user goal,
   active progress tracker, recent execution signals.

A request can trigger several of these on one turn (e.g. mid-stream
stall → cancel → empty-response on the retry → force frontier on the
next request).

---

## 5. State management

### SessionState (P1)

`tinyctx/session_state.py` is a namespaced in-memory store:

```python
_STATE[conv_sid][namespace][key] = value
```

Each module that needs per-conversation state declares its namespace and
registers a *compaction reset whitelist*:

```python
session_state.register_compaction_reset(
    "synthetic_continue",
    ["injection_count", "budget_reminder_fired"],
)
```

When a compaction request lands (codex's handoff-summary prompt, OR
tinyctx's own `proactive_compact` applied), `session_state.reset_compaction(conv_sid)`
clears every key registered for compaction reset under any namespace.
Other keys survive — e.g. `synthetic_continue.strategy_idx` is a
rotation index independent of conversation length; clearing it would
reset the round-robin pointlessly.

### conv_sid as fingerprint, not prompt_cache_key

`conv_id.resolve_conv_key(proj_sid, body)` derives a stable per-conversation
key from three drift-resistant signals: `client_metadata.x-codex-installation-id`,
the requested model name, and a hash of the first developer-role input
text. Earlier versions keyed on `prompt_cache_key` alone; live traces
showed that field drifts mid-conversation (OpenAI prompt-cache invalidation
on body-size growth, tool-list shifts), which silently reset every per-conv
counter and prevented the synthetic_continue budget cap from ever firing.

The new fingerprint is intentionally coarser-than-needed: two `/clear`
conversations in the same install/model/mode collide. The cost is
that the synthetic_continue budget cap fires slightly early in that
edge case — graceful degradation. Finer-than-actual keys (the old
behavior) caused the cap to never fire — the bug.

### proj_sid vs conv_sid

`proj_sid` = hash(cwd) + ":" + codex session id. Used for per-project
state that should NOT cross projects but CAN cross conversations
(error_streak, stall_watchdog last_event, mutation gate timing).

`conv_sid` = `proj_sid` + ":fp:" + fingerprint (or `:` + pck, or bare
proj_sid as fallback). Used for per-conversation counters
(synthetic_continue.injection_count, empty_response_guard.force_next_to_frontier,
stuck_loop.last_reminder_turn).

When in doubt: per-CONVERSATION state uses `conv_sid`; per-PROJECT
state uses `proj_sid`. Stall watchdog escalation key prefers `conv_sid`
when the caller has it (so one conversation's stall doesn't bleed into
another in the same project).

---

## 6. Test architecture

1219 tests across ~60 test files. Full run is ~30s, no network. The
key shape:

- **P0 integration sagas** (`tests/test_integration_workflow.py`,
  `tests/test_proxy_integration.py`,
  `tests/test_proxy_compactor_integration.py`,
  `tests/test_path_coverage_e2e.py`) — end-to-end scenarios with fake
  upstreams that exercise compaction handoff, retry escalate, mid-stream
  stall, empty response → force frontier, soft punt → synthetic
  continue. These are the contract layer for the proxy.
- **Per-module unit tests** — one `test_<module>.py` per source module.
  Focused on the module's public surface; reach into module-private
  state only through the `_SessionStateDictView` legacy shims.
- **TDD workflow** — the P0-P8 refactor used TDD throughout: write the
  failing integration test against the new API shape first
  (`tests/test_router.py`, `tests/test_guards.py`,
  `tests/test_stream_relay.py`, `tests/test_post_stream.py`,
  `tests/test_session_state.py`, `tests/test_config.py`), then
  extract until the test passes without modifying it.

The integration sagas are the boundary that says "behavior is
preserved across the refactor"; the per-module tests are the boundary
that says "this component does what it claims".

---

## 7. Key conventions

- **`conv_sid` scoping (P1).** Module that maintains per-conversation
  state takes `conv_sid` as the primary key. Falls back to `proj_sid`
  when conv_sid is unavailable (chat-completions path, older callers).
- **Back-compat shims (`_SessionStateDictView`).** When P2/P3 migrated
  modules from `_X_PER_SESSION: dict` module-locals into SessionState,
  the original attribute name was preserved as a dict-shaped proxy
  over the new storage. Tests poke `module._FORCE_NEXT_TO_FRONTIER[sid]`
  and still work; new code uses `session_state.get(...)` directly.
- **Forensics dump trigger naming.** Trigger names are snake_case
  noun-phrases: `empty_response`, `stream_error`, `stall_cancelled_relay`,
  `punt_via_stream_rewrite`, `upstream_<status>`, `punt_p{NN}`.
  `ForensicsPolicy.SUPPORTED_TRIGGERS` enumerates the static set;
  `is_supported_trigger` accepts the dynamic-suffix families.
- **Log event naming.** `_log(event, **fields)` events use snake_case
  verbs at the head: `route`, `retry_attempted`, `stall_kill`,
  `stream_done`, `empty_response_detected`, `compaction_state_reset`.
  Errors use `<module>_error` (`historian_spawn_failed`,
  `auto_scout_error`). The dashboard groups events by these names.
- **Pristine recomputation invariant** (compactor.py). The Historian
  background update always works from the pristine raw body, never
  from previously-mutated history. Mirrored by `raw_body =
  json.loads(json.dumps(body))` snapshot at the top of
  `proxy.responses`.

---

## 8. Cookbook — adding new functionality

### Adding a new guard

1. Add a class to `tinyctx/guards.py` with `name`, `priority`,
   `apply(ctx: GuardContext) -> GuardResult`. Lower priority runs
   first.
2. Wrap your detection function (often it already exists as a
   module-level helper); set `ctx.body = new_body` and
   `ctx.force_route = "frontier"` as needed.
3. Register it in `proxy.responses` inside the `active_guards` list
   under a CFG flag.
4. Add a per-guard log-emit branch inside the `for gr in guard_results`
   loop so the dashboard can attribute fired/skipped events.
5. Test: `tests/test_guards.py` — add a class that exercises both
   fired and skipped paths.

### Adding a new retry trigger

`retry_policy.classify_failure` is pure. Add a new branch keyed on
the failure shape (status, route, is_compaction) returning a
`RetryAction(decision, reason, ...)`. The proxy's retry loop
consumes it automatically — no proxy.py change needed for the
classification itself. If the new action needs a fresh CFG flag
(e.g. "retry_on_frontier_410"), thread it through
`classify_failure`'s kwargs and into `Config`.

### Adding a new forensics dump

1. Add a new trigger string to `ForensicsPolicy.SUPPORTED_TRIGGERS`
   (or extend `is_supported_trigger` if it's a dynamic family like
   `upstream_<NNN>`).
2. Decide which gating flag it should respect — `forensics_enabled`
   (always), plus optionally `forensics_capture_errors` or
   `forensics_capture_punts`. Update `_gating_passes`.
3. Call `forensics_policy.dump(trigger=..., proj_sid=..., ...)`
   from the detection site.

### Adding a compaction-resettable state key

1. Pick a namespace (your module name) and key.
2. At module load: `session_state.register_compaction_reset("my_module", ["my_key"])`.
3. Read/write via `session_state.get(conv_sid, "my_module", "my_key")`
   / `set(...)`. The proxy clears it automatically at compaction
   boundaries (both incoming codex compaction AND tinyctx proactive
   compaction).
4. If you want a back-compat dict view for tests, copy the
   `_SessionStateDictView` pattern from
   `empty_response_guard._SessionStateDictView`.

---

## 9. Refactor history

The post-Symphony resilience phases (`586af76` and later) built up
the interruption-prevention stack; the P0-P8 refactor that followed
restructured the proxy without changing behavior.

| Phase | Commit    | Summary                                                              |
| ----- | --------- | -------------------------------------------------------------------- |
| —     | `586af76` | feat(symphony): import 5 interruption-resilience patterns from openai/symphony |
| —     | `1f7e0b0` | fix: 6 cross-conversation state-leak + calibration bugs from live trace |
| —     | `0af6011` | feat(proxy): comprehensive retry policy for all upstream interruptions |
| —     | `ad7b2f1` | fix(stall_watchdog): actually cancel stalled tasks instead of flagging |
| P0    | `fb62128` | test: integration test moat for 5 saga lifecycles                    |
| P1    | `89bd3d5` | refactor: SessionState abstraction + pilot synthetic_continue migration |
| P2    | `9e76694` | refactor: migrate 3 modules to SessionState                          |
| P3    | `9c6a4cb` | refactor: migrate exec_resume / soft_completion / request_phase to SessionState |
| P4    | `c5f402c` | refactor: GuardPipeline + 5 wrapper guards                           |
| P5    | `dc4c0c2` | refactor: consolidate route decision into Router.decide(ctx)         |
| P6    | `ca9c3c4` | refactor: split _stream_proxy into stream_relay.py components        |
| P7    | `dd3a644` | refactor: extract post_stream.py (analyzer + terminator + forensics policy) |
| P8    | `fe0f559` | refactor: namespaced config sub-views (back-compat preserved)        |

Each refactor phase had a paired test contract (the P0 sagas are the
overall invariant; per-phase tests cover the new API surface). 1219
tests pass at the end of each phase.

Symphony reference: openai/symphony SPEC §7-8 was the source of the
phase enum (P3), the stall-watchdog cancel-and-retry pattern
(`ad7b2f1`), the early SSE keepalive, and the synthetic continue
shape (`586af76`).
