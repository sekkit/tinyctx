# tinyctx

A thin context-routing proxy for the OpenAI Codex CLI that lets you run **mostly on a local-class model** (DeepSeek-v4-flash, LMStudio + Qwen3.6-27B, vLLM, Ollama, …) and **only escalate to a frontier model** (GPT-5.5) when truly needed.

> **Compose-first.** tinyctx itself is ~10K LOC of glue across 28 modules. The hard parts — code knowledge graph, LSP-backed symbolic ops, sandboxed tool execution, persistent memory, prompt compression — are delegated to mature OSS projects, all auto-installed and auto-wired by `scripts/install.sh`.

## Why this exists

Codex CLI sends the entire conversation history on every turn (it does not use `previous_response_id` — see [openai/codex#4047](https://github.com/openai/codex/issues/4047)). When the rolling history hits 90% of the model's context window, codex triggers a compaction step that calls the same expensive frontier model to produce a "handoff summary." That summary call is *also* billed at frontier rates. And because codex resends the whole history each turn, prompt-cache hits depend on byte-stable prefixes — which the compaction prompt and reasoning items routinely break.

Three load-bearing observations from the literature:
1. **Prefix reuse, not algorithmic compression, is the dominant cost lever.** LMCache reports ~92% prefix-reuse in Claude Code traffic, climbing to ~98% in execution phase. With Anthropic prompt-caching's 10× read discount, that's ~5–10× cost reduction by itself.
2. **The compaction summary is the easiest call to redirect.** It's a stateless transform of conversation history into a condensed handoff. A 27B-class model does it well enough; a frontier model is wasted there. ([openai/codex#20972](https://github.com/openai/codex/issues/20972) is the open feature request for exactly this.)
3. **Reasoning items are model-bound.** When you swap models mid-session, prior `encrypted_content` from the other model's reasoning is undecryptable and crashes the request ([openai/codex#17541](https://github.com/openai/codex/issues/17541)). Any router that swaps backends must scrub these.

tinyctx sits between codex and the model, owns those three problems, and delegates everything else.

## Architecture

```
codex CLI
  │
  │  Responses API (HTTP, wire_api = "responses")
  ▼
┌─────────────────────────────────────────────────────────────┐
│  tinyctx-proxy   (FastAPI, ~10K LOC across 28 modules)      │
│                                                             │
│  • Compaction interceptor   (handoff-prompt → local model)  │
│  • Cascade router           (heuristic + optional classifier)│
│  • encrypted_content scrub  (when crossing model boundaries)│
│  • Cache-discipline         (CacheAwareMutator gates)       │
│  • read_delta               (repeat-Read → unified diff)    │
│  • Tool-call XML translator (qwen-pythonic → structured)    │
│  • Chat→Responses SSE bridge (DeepSeek/Ollama/etc.)         │
│  • Codex 0.128 wire-compat  (3 fixes for breaking changes)  │
│  • LLMLingua-2 hook         (frontier compression, opt-in)  │
└─────────────────────────────────────────────────────────────┘
  │                                       │
  │ default ~95%                          │ escalate ~5%
  ▼                                       ▼
Local cheap path                        GPT-5.5
(DeepSeek-v4-flash by default;          (via chatgpt.com/backend-api/codex
 LMStudio+Qwen3.6-27B, Ollama, vLLM,     or icodeeasy.cc/v1)
 SGLang all supported via config)

In parallel, codex talks directly to MCP servers / skills (all auto-installed
by scripts/install.sh):

  • gitnexus           — tree-sitter knowledge-graph MCP server
                         (abhigyanpatwari/GitNexus, PolyForm-Noncommercial)
  • graphify           — tree-sitter knowledge-graph codex SKILL
                         (safishamsi/graphify, MIT — AGENTS.md + PreToolUse hook)
  • serena             — LSP-backed symbolic ops MCP server
                         (oraios/serena, MIT)
  • caveman-shrink     — output / tool-description compression MCP middleware
                         (JuliusBrussee/caveman, MIT)
  • context-mode       — sandboxed tool execution
                         (mksglu/context-mode, Elastic 2.0)
  • mem0               — persistent project memory across sessions
                         (mem0ai/mem0, Apache 2.0; opt-in via tinyctx-mem)
  • advisor (built-in) — frontier-class consultation MCP (Anthropic Advisor Strategy)
  • scout SessionStart hook — injects ~/.tinyctx/cache/<repo>/scout.md as
                              additionalContext on turn 1 (no manual AGENTS.md edit)
```

## What we build vs. what we wire

| Layer | Tool | Provenance | License |
|-------|------|------------|---------|
| Wire intercept | **tinyctx-proxy** | this repo | MIT |
| Compression-biased ranker | **tinyctx.interest** | this repo, after [arxiv 2603.20396 §5.1](https://arxiv.org/abs/2603.20396) | MIT |
| Repeat-Read delta | **tinyctx.read_delta** | this repo | MIT |
| Code knowledge graph (MCP) | gitnexus | upstream OSS | PolyForm-Noncommercial |
| Code knowledge graph (skill) | graphify | upstream OSS | MIT |
| Symbolic ops (LSP MCP) | serena | upstream OSS | MIT |
| Tool sandbox (MCP) | context-mode | upstream OSS | Elastic 2.0 |
| Output / tool-desc compression (MCP) | caveman-shrink | upstream OSS | MIT |
| Pre-escalation compression (Python lib) | LLMLingua-2 (microsoft/LLMLingua) | upstream OSS | MIT |
| Memory | mem0ai (optional) | upstream OSS | Apache 2.0 |
| Inference (cheap, default) | DeepSeek API (`deepseek-v4-flash`) | upstream OSS / API | MIT-equivalent |
| Inference (cheap, alt) | LMStudio + Qwen3.6-27B | upstream OSS | Apache 2.0 |
| Inference (frontier) | OpenAI/codex backend (gpt-5.5) | upstream | proprietary |

If something isn't pulling its weight, replace it. The only piece tinyctx assumes lives forever is its own ~5,000 LOC proxy code, because the four jobs it does aren't covered by any OSS project we found.

## Quick start

```bash
cd ~/dev/tinyctx
./scripts/install.sh      # full setup: venv + all MCP servers + codex config
./scripts/start.sh        # starts the proxy on 127.0.0.1:4141
codex --profile tinyctx   # uses the proxy as model_provider
```

`install.sh` is fully idempotent and auto-installs every external dep. It will, in order:

1. Create `.venv` and install tinyctx-proxy
2. Copy starter `~/.tinyctx/config.toml` if absent
3. **gitnexus** — `npm install -g gitnexus@1.6.3` + register `[mcp_servers.gitnexus]`
4. **serena** — `uv tool install serena-agent` (bootstraps uv via curl if missing) + register `[mcp_servers.serena]`
5. **caveman-shrink** — `git clone` to `~/.tinyctx/vendor/caveman` + register `[mcp_servers.caveman-shrink]`
6. **mem0** — `pip install mem0ai` (skip with `TINYCTX_MEM0_DISABLE=1`)
7. **graphify** — `uv tool install graphifyy` + per-project `graphify codex install` (writes AGENTS.md + .codex/hooks.json)
8. **scout SessionStart hook** — registers `scripts/scout-session-start.sh` in `~/.codex/hooks.json` (co-exists with existing entries)

Each step writes a per-step log under `~/.tinyctx/logs/*-bootstrap.log`. None block proxy startup; all are safe to re-run.

See [docs/install.md](docs/install.md) for the long version, [docs/test.md](docs/test.md) for the test plan, and [docs/features.md](docs/features.md) for the full feature inventory.

## Picking a local backend

The `[local]` section of `~/.tinyctx/config.toml` decides where the cheap path lands. Four supported shapes — pick one in `examples/config.toml`, copy the chosen block into `~/.tinyctx/config.toml`, then `tinyctx-up restart`:

| Backend | wire_api | Pros | Cons |
|---|---|---|---|
| **DeepSeek API** (`deepseek-v4-flash`/`-pro`) ← default | `chat` | ~5–7 s/turn, very cheap, 1M context, `reasoning_content` preserved | needs API key |
| **LMStudio** + qwen3.6-27b on local GPU | `responses` | free, native Responses API, `reasoning_content` round-trips cleanly | slow on consumer Macs (~3 min/turn at xhigh reasoning); needs the chat template that emits qwen-pythonic XML |
| **Ollama** | `chat` | free, easy install | no `reasoning_content` for non-reasoning models |
| **vLLM / SGLang** | `responses` | self-hosted at scale | infra overhead |

### Using a paid API backend (DeepSeek, OpenAI, OpenRouter, …)

Keys never go into committed config. They live in `~/.tinyctx/secrets.env`, chmod 600. `scripts/start.sh` sources the file at proxy launch (and the launchd plist runs `start.sh`, so secrets survive a reboot):

```bash
cp examples/secrets.env.example ~/.tinyctx/secrets.env
chmod 600 ~/.tinyctx/secrets.env
$EDITOR ~/.tinyctx/secrets.env       # paste the real key

# Then in ~/.tinyctx/config.toml, swap to the DeepSeek (or OpenRouter, etc.)
# block under [local]. The api_key_env line tells the proxy which env var
# to look up; secrets.env has set it for the launchd-spawned process.

tinyctx-up restart
```

Verify the key landed in the right place:

```bash
tinyctx-trace --watch &      # in one shell
codex --profile tinyctx exec "say hi" </dev/null    # in another
```

The `request_trace` event will show `target_url=https://api.deepseek.com/v1/...` and `translated=True` (chat→responses round-trip).

## Sanitize / wire-compat pipeline

Every request to the proxy passes through (in order):

| Transform | What it does |
|---|---|
| `strip_encrypted_content` | Drop `encrypted_content` from prior reasoning items so cross-model swaps don't crash ([openai/codex#17541](https://github.com/openai/codex/issues/17541)). |
| `expand_mcp_namespaces` | codex 0.128+ wraps MCP tools in `type=namespace`; expand into `type=function` so they survive scrub. |
| `dedup_tool_calls` _(opt-in)_ | Identical-args duplicates → placeholder, keep latest verbatim. |
| `purge_failed_tool_inputs` _(opt-in)_ | After N assistant turns past a failed tool call, replace the bulky input with a placeholder. Error stays visible. |
| `read_delta.collapse_repeated_reads` | **2nd+ Read of same path** → unified diff vs first read (or "unchanged" marker). Default on. |
| `historian.apply_to_body` _(opt-in)_ | Replace pre-`recent_keep` turns with a local-model-generated digest. |
| `proactive_compact` | When `est_tokens ≥ context_window × safe_fraction`: replace middle of `body.input` with summary item; bucket-cached so back-to-back turns reuse it. |
| `inject_advisor_hint` | Append Advisor Strategy usage guidance to `body.instructions` (skipped on frontier route). |
| `expand_mcp_namespaces` + `scrub_unsupported_tools` | Drop tool entries strict OpenAI-compat backends reject. |
| `strip_unsupported_responses_fields` | Drop `client_metadata` / `prompt_cache_key` etc. |
| `inject_responses_defaults` | Add `text.format.type="text"` etc. that LMStudio requires but codex omits. |
| `cap_responses_fields` | Force-cap `max_output_tokens` to prevent runaway thinking loops (production-validated fix). |
| `lingua.compress_for_frontier` _(opt-in)_ | Run LLMLingua-2 over bulky tool-result payloads on the frontier path. |
| `trim_tools_for_frontier` | Reduce ~50 codex tools → working set (essentials + recently-used). |
| `normalize_for_chat` | Responses-API → chat-completions: tool calls, reasoning_content stub for DeepSeek thinking mode, input_image flatten, orphan call repair. |

Inspect any one request via `tinyctx-trace --request <rid> -v`.

All history-mutating transforms (dedup / purge / historian / read_delta / lingua) are gated by `CacheAwareMutator` so they only fire when the prompt-cache prefix is likely stale anyway (Anthropic 5-min TTL or context-usage threshold).

## Compression-biased context ranking

`tinyctx/interest.py` is a faithful adaptation of §5.1 of Aksenov, Bodnia, Freedman, Mulligan, *Compression Is All You Need: Modeling Mathematics* ([arxiv 2603.20396](https://arxiv.org/abs/2603.20396)) to a code corpus. The paper's central empirical finding is that human mathematics lives in the polynomial-growth (A_n, log-density) regime: a small number of hierarchically-nested macros buys exponential expansion. The proposed ranking is PageRank with **teleportation biased toward high-compression nodes** — formally, T_0(u) = unwrapped/wrapped (reductive compression) and I_0(u) = body/signature (deductive compression), combined as J_0 = β·T_0 + (1−β)·I_0.

For a coding agent, the same principle picks the load-bearing abstractions (utility modules, well-named interfaces, deeply-nested but terse APIs) that should be primed into the prompt. It's a strictly stronger ranking than uniform-personalization PageRank (which is what aider's repomap does) because it uses the corpus's own compression structure as the inductive bias.

```bash
# Build a graph for the repo (gitnexus or graphify), then rank symbols:
graphify .
.venv/bin/python -m tinyctx.interest graphify-out/graph.json "auth token verification"
```

## Modules

```
tinyctx/                          ~10K LOC, 28 modules
  proxy.py                  FastAPI server + routing pipeline
  router.py                 Routing decision (heuristic + optional classifier)
  sanitize.py               13 transforms (above table)
  read_delta.py             Repeat-Read collapse to unified diff
  lingua.py                 LLMLingua-2 pre-escalation compression hook
  config.py                 Layered config: defaults < TOML file < TINYCTX_* env
  compactor.py              3-role debate + judge merge → markdown summary + facts/compartments
  continuity.py             Persist compactions; --facts-only / --compartment recall
  historian.py              Rolling per-session digest (async update + opt-in substitute)
  scout.py                  Two-layer init scan: graphify + local-LLM subagent + content-hash cache
  scout_hook_bootstrap.py   Auto-register SessionStart hook in ~/.codex/hooks.json
  keypin.py                 Scan codex rollouts for Read-frequency; emit byte-stable keyfiles.md
  memory.py                 Optional mem0 wrapper for cross-session user/project memory
  interest.py               Compression-biased PageRank ranker (paper §5.1)
  graphify_adapter.py       graphify graph.json -> interest.py shape
  classifier.py             Pure-Python logistic regression for escalation scoring
  stats.py                  JSONL log -> route mix / cost / quality grade (S/A/B/C/D/F)
  registry.py               Per-machine list of projects tinyctx has touched
  dreamer.py                Periodic maintenance CLI: scout + keypin + graphify + mem ingest + GC
  trace.py                  RequestTrace dataclass + CLI viewer (compact / verbose / watch)
  advisor.py                Anthropic Advisor Strategy MCP server (built-in)
  tool_call_translator.py   XML→struct + chat→responses SSE + 3-layer auto-answer for request_user_input
  _codex_toml.py            Shared helper: idempotent ~/.codex/config.toml MCP block injection
  gitnexus_bootstrap.py     Auto-install gitnexus + register MCP
  graphify_bootstrap.py     Auto-install graphify + per-project codex skill wire
  serena_bootstrap.py       Auto-install serena-agent + register MCP
  caveman_bootstrap.py      Vendor caveman repo + register caveman-shrink MCP

scripts/
  install.sh                Idempotent installer (venv + 6 OSS MCP/skills + scout hook)
  start.sh / start-bg.ps1   Boot the proxy
  smoke_codex.sh            Real codex CLI smoke test against fake backends
  scout-session-start.sh    SessionStart hook script (auto-registered)
  cm-hook-shim              context-mode hook adapter
  tinyctx-up                Convenience wrapper: start / stop / restart / status / logs
  path_coverage.py          Read JSONL traces and report which code paths fired in production

.codex-plugin/              Codex marketplace plugin manifest (plugin.json + hooks)

tests/                      28 test files, 368 tests, no network required
```

CLIs:

```bash
tinyctx-proxy                                       # boot the proxy (also: scripts/start.sh)
tinyctx-scout init  --graph tinyctx-graph.json      # build the project context cache (calls local LLM once)
tinyctx-scout status                                # is the cache fresh?
tinyctx-scout refresh                               # rebuild only if any tracked file changed
tinyctx-recall                                      # print most recent compaction summary for this repo
tinyctx-recall --list                               # list sessions and compaction counts
tinyctx-recall --facts-only                         # just the structured facts list
tinyctx-recall --compartment auth-setup             # one named compartment from latest compaction
tinyctx-keypin scan                                 # rebuild keyfiles.md from codex rollout Read freq
tinyctx-keypin show                                 # print the latest keyfiles.md
tinyctx-mem available                               # check whether mem0 is installed
tinyctx-mem ingest-compaction                       # push facts from latest compaction into mem0
tinyctx-mem search "code style"                     # cross-session memory recall
tinyctx-dreamer list                                # show projects registered for periodic maintenance
tinyctx-dreamer run --gc                            # rebuild scout + keyfiles + graphify wiring + GC
tinyctx-dreamer install-launchd                     # install macOS daily 03:00 timer
tinyctx-stats                                       # routing summary from logs
tinyctx-stats --quality                             # S/A/B/C/D/F quality report (6-dim weighted)
tinyctx-trace                                       # last 10 requests, compact table
tinyctx-trace --watch                               # tail today's request_trace JSONL
tinyctx-trace --request rq_xxx -v                   # full detail for one request
tinyctx-interest <graph.json> "<query>"             # rank symbols by compression-biased PageRank
tinyctx-gitnexus  status / install / uninstall      # gitnexus MCP bootstrap
tinyctx-graphify  status / install / uninstall      # graphify codex-skill bootstrap
tinyctx-serena    status / install / uninstall      # serena MCP bootstrap
tinyctx-caveman   status / install / uninstall      # caveman-shrink MCP bootstrap
tinyctx-scout-hook status / install / uninstall     # SessionStart hook registration
tinyctx-lingua    status / test / warmup            # LLMLingua-2 hook diagnostics
python -m tinyctx.classifier train <labeled.jsonl>  # train escalation scorer (optional)
python -m tinyctx.classifier predict est_tokens=80000 turn_count=20
python -m tinyctx.graphify_adapter <graphify.json>  # convert a graphify export
```

## Multi-subagent compaction (and what happens when a session runs out)

When codex hits ~90% of its context window it triggers an internal compaction step that asks the model to write a "handoff summary" of the conversation so far. By default that call costs frontier rates. tinyctx already redirects that call to the local 27B (the highest-leverage cost win, and `openai/codex#20972` is the open feature request). What it now *also* does — when `compactor_debate = true` — is replace the single-pass local summary with a **3-role debate + judge merge**:

```
                     parallel local calls (~one wall-clock LLM round)
   archaeologist  →  preserves verbatim facts, file paths, exact errors
   narrator       →  preserves intent and the storyline of attempts
   enumerator     →  lists every concrete artifact (files, commands, errors)
                                 │
                                 ▼
                          judge (one local call)
                          merges the three drafts into a canonical
                          handoff summary with fixed sections + structured
                          JSON (compartments / facts / open_questions)
```

Per-role drafts run in parallel via `httpx.AsyncClient`, so wall-clock latency is roughly one local-model call, not three. Failure tolerance is graded: if 1 of 3 roles fails the judge merges the survivors; if the judge itself fails we deterministically concatenate the role drafts; if all role calls fail the proxy falls back to a straight forward to the local backend without the debate. **codex never sees a hard failure from the compactor.**

Every successful compaction is persisted to `~/.tinyctx/cache/<repo-hash>/sessions/<sid>/compaction-N.md` (+ structured `.json` sidecar). When a session truly runs out of room and codex makes you `/clear`, recall the last compaction:

```bash
tinyctx-recall                # print the most recent compaction
tinyctx-recall --list         # show all sessions for this repo
tinyctx-recall --all-sessions # print compactions across every prior session
```

## Two-layer project scout

The first time codex enters a repo, you don't want it spending frontier tokens just to figure out where things live. tinyctx ships a **two-layer scout**:

```
Layer 1  (free, no LLM)        graphify or gitnexus → graph.json
Layer 2  (one-shot, local LLM) tinyctx-scout init → ~/.tinyctx/cache/<repo>/scout.md
                                ranks the top-K load-bearing nodes via the
                                compression-biased PageRank from interest.py,
                                then asks the local 27B for a hierarchical
                                ≤2000-token summary. Cached by file content
                                hash; rebuild only when scanned files change.
SessionStart hook              scripts/scout-session-start.sh injects scout.md
                                as additionalContext at codex startup, so the
                                summary is in context for turn 1 — without
                                ever blowing main context with raw scan output.
                                Auto-registered by scripts/install.sh
                                (tinyctx.scout_hook_bootstrap).
```

Workflow:

```bash
# 1. Build a graph (gitnexus, graphify, or any tree-sitter indexer).
graphify .                                              # produces graphify-out/graph.json

# 2. One-shot scout build (talks to local model at $TINYCTX_LOCAL_BASE_URL).
tinyctx-scout init --graph graphify-out/graph.json --top-k 40
# → ~/.tinyctx/cache/<repo-hash>/scout.md       (≤2K tokens, byte-stable)
#   ~/.tinyctx/cache/<repo-hash>/manifest.json  (file content hashes)

# 3. (Optional) verify staleness next session.
tinyctx-scout status

# 4. SessionStart hook auto-injects scout.md into every codex session.
#    No AGENTS.md edit required — the hook emits {"additionalContext": "..."}
#    that codex consumes natively.
```

This is the **subagent shape** by construction: scout.py runs as an isolated local-model call with a fixed short system prompt, persists a digest, and the parent (codex) only ever sees the digest — never the raw 50-file scan that went into producing it.

## Borrowed from cortexkit/magic-context + token-optimizer

A separate analysis of [magic-context](https://github.com/cortexkit/magic-context) (OpenCode plugin, MIT) and [alexgreensh/token-optimizer](https://github.com/alexgreensh/token-optimizer) (Claude Code plugin, PolyForm-Noncommercial) influenced eight independent reimplementations across releases. We did not vendor any code (different host, different license assumptions) — these are independent reimplementations of the underlying ideas:

| From upstream | tinyctx implementation |
|---|---|
| Cache-aware deferred drops (queue ops, fire only on TTL/threshold) | `sanitize.CacheAwareMutator` — gates dedup/purge/historian-substitution/read_delta/lingua so the prompt-cache prefix stays warm in the common case |
| Pristine recomputation invariant (compress always from raw, never from prior summary) | Documented invariant in `compactor.py` + `test_pristine_recomputation_guard` |
| Pinned key files via read-frequency (dreamer.pin_key_files) | `tinyctx-keypin scan` — scans codex rollouts, ranks files by Read-tool frequency |
| Compartments + facts structured compaction | `compactor.parse_judge_output` produces compartments / facts / open_questions; `continuity.save_compaction` writes a `.json` sidecar; `tinyctx-recall --facts-only` / `--compartment <name>` |
| User memory promotion (mem0-style cross-session) | `tinyctx.memory` — thin wrapper over [mem0ai](https://github.com/mem0ai/mem0); auto-installed by `install.sh` |
| Historian (rolling per-turn compression of older history) | `tinyctx.historian` — async background `update()` after every Nth turn writes a digest sidecar; opt-in `historian_substitute` replaces older turns on the wire (gated by CacheAwareMutator) |
| Dreamer (periodic background consolidation) | `tinyctx-dreamer run` — scout refresh + keypin scan + graphify per-project wire + (optional) mem0 ingest for every registered project, plus session-cache GC; `install-launchd` / `install-cron` wires it to a daily timer |
| **Repeat-Read delta compression** (token-optimizer's `delta_diff.py`) | `tinyctx.read_delta` — same-path Read tool result on 2nd+ occurrence → unified diff vs first read (or "unchanged" marker). Default on, gated by CacheAwareMutator. |
| **Quality-grade scoring** (token-optimizer's quality dashboard) | `tinyctx-stats --quality` — 6-dim weighted score (routing efficiency / compaction discipline / token compression / read-delta savings / tool-trim savings / reliability), maps to S–F |

What we deliberately did NOT borrow: temporal markers (niche for code), cross-harness SQLite pool (we run only on codex), smart notes with conditional surfacing (too workflow-specific).

## Advisor Strategy (executor-driven frontier consultation)

After the v0.5 routing fixes (per-backend `context_window` + `escalate_turn_count = 9999`), frontier hit rate dropped to ~0% on real sessions — DeepSeek-v4-flash with 1M context absorbed everything. Cheap, but the frontier got no use even on prompts that genuinely benefit from `gpt-5.5`'s reasoning.

The Advisor Strategy ([Anthropic blog](https://claude.com/blog/the-advisor-strategy)) threads a third path: the executor decides per-turn whether to consult the frontier as a **tool**, instead of all-or-nothing per-request escalation.

```
99% of turns                     ←── executor (DeepSeek-v4-flash)
when stuck on a decision         ──→ ask_advisor(question, context)
                                       ↓
                                   consult frontier (gpt-5.5)
                                   return 200-500 word guidance
                                       ↓
executor resumes with advice     ←──
```

### Two implementations — pick one based on codex version

**Codex 0.128+ (recommended): codex agent route**

Reverse-engineering Codex.app 0.128.0-alpha.1's binary revealed that codex's namespace MCP dispatcher returns `unsupported call` for any LLM-emitted `mcp__<server>__<tool>` invocation in alpha builds — the LLM sees the tool, calls it, and codex's `core/src/tools/router.rs` rejects the dispatch. Reverse engineering also showed codex has a **stable, fully-implemented `spawn_agent` system** (multi_agent feature, true by default) that's a perfect Advisor Strategy fit — the LLM forks a sub-thread bound to a config file with its own model + system prompt. So the supported route on 0.128+ is to register advisor as a codex agent, not as an MCP tool.

```toml
# ~/.codex/config.toml
[agents.advisor]
description = "Consult a more capable advisor model (gpt-5.5 / Opus-class) for HARD decisions when stuck. Use when: (1) torn between architectural choices with real consequences, (2) tried 2+ failed approaches and need a fresh perspective, (3) about to make a non-trivial security/correctness decision, (4) user intent ambiguous and the wrong interpretation will waste significant work."
config_file = "agents/advisor.toml"
```

```toml
# ~/.codex/agents/advisor.toml
model = "tinyctx-frontier"            # forced-route alias → tinyctx ships the call to gpt-5.5
model_reasoning_effort = "high"
sandbox_mode = "read-only"
web_search = "disabled"

developer_instructions = """
You are an expert advisor for a coding agent...
[full system prompt — see examples/agent-advisor.toml]
"""
```

The executor then calls `spawn_agent(role="advisor", task="<question>")` and awaits with `wait_agent`. codex starts a sub-thread, the sub-thread's `model="tinyctx-frontier"` makes tinyctx force-route to the frontier (every call shows up in `tinyctx-trace --watch` with `forced_by_client_model=true`), gpt-5.5 returns its 200-500 word guidance, the executor reads it, life continues.

End-to-end verified on Codex.app 0.128.0-alpha.1: PING → advisor sub-thread → gpt-5.5 → PONG response in ~7s with `forced_by_client_model=true` confirmed in trace.

**Codex 0.125 (fallback): MCP server route**

For older codex builds, `tinyctx/advisor.py` is a stdio MCP server that exposes `ask_advisor(question, context, previous_attempts)`. Same wire shape as the agent route — the call goes through the tinyctx proxy with `model="tinyctx-frontier"` and the auth flow inherits codex's chatgpt token from `~/.codex/auth.json`.

```toml
# ~/.codex/config.toml (legacy, codex 0.125)
[mcp_servers.advisor]
type = "stdio"
command = "/path/to/tinyctx/.venv/bin/python"
args = ["-m", "tinyctx.advisor"]

[mcp_servers.advisor.env]
TINYCTX_PROXY_URL = "http://127.0.0.1:4141/v1"
TINYCTX_ADVISOR_MODEL = "tinyctx-frontier"
TINYCTX_ADVISOR_TIMEOUT_S = "180"
```

The MCP server still works for direct CLI use (`echo '{"method":"tools/call",...}' | python -m tinyctx.advisor`). It just doesn't reach the executor on codex 0.128+.

### Bonus tinyctx fixes shipped along the way

While debugging the advisor path on codex 0.128, three independent codex 0.128 ↔ DeepSeek wire incompatibilities surfaced and got fixed:

- **`expand_mcp_namespaces`** (`tinyctx/sanitize.py`) — codex 0.128 wraps MCP tools in `type: "namespace"` shells. The proxy now expands them into top-level `type=function` entries (names like `mcp__advisor__ask_advisor`).
- **`_flatten_tool_output`** — codex 0.128 returns rich tool outputs with `input_image` items mixed in (base64 PNGs from screenshots). DeepSeek's chat-completions API rejects these with HTTP 400. We now flatten to text + `[image attached]` placeholders.
- **`reasoning_content` stub** — codex 0.128's `type=reasoning` items ship empty (real thinking text is server-only). DeepSeek's thinking-mode endpoint then 400s on the next turn unless every assistant message carries some `reasoning_content`. The proxy stubs an empty string so DeepSeek's strict check passes.

Each one was triggered live during a real session (700+ turns) and identified via wire-body capture, codex.app binary reverse engineering, and DeepSeek error message inspection.

Anthropic's blog reports +2.7 SWE-bench points at -11.9% cost using this pattern (Sonnet+Opus pairing). The DeepSeek+gpt-5.5 pairing should see similar shape — the executor handles 99% of turns at deepseek pricing, and burns the frontier only on the 1-3 hard decisions per task that actually need it.

## Auto-answer for `request_user_input` blocking prompts

codex 0.128 emits a `request_user_input` tool call when the model wants the user to click a choice — which **blocks** the session until the user clicks. tinyctx ships a 3-layer auto-answer chain (default on, env switch `TINYCTX_AUTO_USER_INPUT=0` to disable):

| Layer | Trigger | Cost |
|---|---|---|
| 1. Explicit tool call | model calls `request_user_input` | 1 advisor (frontier) call |
| 2. Plain-text choice prompt | regex catches `请选择 A → ... B → ...` style at message tail | 1 advisor call |
| 3. Subtle "awaiting input" prose | DeepSeek classifier (~$0.0001) decides; if YES → escalate | local classifier + maybe 1 advisor |

The classifier-driven layer means even unstructured "let me know what to do next" messages get auto-resolved without dragging a paid frontier call through every zero-tool-call assistant turn.

## Repeat-Read delta compression

When the executor re-reads the same file across turns, the second-and-later `function_call_output` items in the wire body carry the full file contents again — typically 5-50 KB of stable bytes. `tinyctx.read_delta` detects this and replaces later occurrences with a unified diff (or "unchanged" marker if identical), gated by `CacheAwareMutator`.

Inspired by [token-optimizer/delta_diff.py](https://github.com/alexgreensh/token-optimizer/blob/main/skills/token-optimizer/scripts/delta_diff.py) (PolyForm-Noncommercial — idea borrowed, not code; reimplemented from scratch).

Detected read patterns: named tools (`Read`, `read_file`, `view`, …), `shell` with cat/head/tail/sed -n, MCP `*__read/__view/__cat`. Skipped: outputs < 400 chars (overhead), error-looking outputs, diffs that don't compress (<15% saving). Default on (`read_delta_enabled = true`).

## LLMLingua-2 pre-escalation compression (opt-in)

For frontier requests with bulky tool-result payloads, `tinyctx.lingua` can run [Microsoft's LLMLingua-2](https://github.com/microsoft/LLMLingua) over those payloads before forwarding. Empirically 2-5× compression on long contexts with no quality regression on coding/QA tasks.

```toml
# ~/.tinyctx/config.toml
[routing]
frontier_lingua_enabled = true
frontier_lingua_ratio = 0.5      # keep ~50% of tokens
```

```bash
pip install 'tinyctx[compress]'  # adds llmlingua dep
tinyctx-lingua warmup            # pre-download model weights (one-time)
```

NEVER touches `instructions`, `tools`, user messages, or assistant messages — those are prompt-cache-critical. Only targets `function_call_output` / `tool_result` items. Cache-aware-gated alongside dedup/purge/read_delta.

## Status

- **v0.7.0** — works end-to-end with fake backends and against real codex CLI 0.125.0 / Codex.app 0.128.0-alpha.1.
- **368 tests across 28 test files, all passing**.
- Advisor Strategy verified live on Codex.app 0.128 via the agent route (PING/PONG round-trip, frontier hit confirmed in trace).
- Production: 700+ turns/day, 95%+ local hit rate, sub-1% upstream errors.
- 7 OSS upstream dependencies wired (gitnexus, graphify, serena, caveman, context-mode, mem0, LLMLingua); all auto-installed by `scripts/install.sh`.
- Total original code: ~10,400 LOC across 28 modules.

## Troubleshooting: `hook: ... Failed` in codex output

If you're seeing lines like

```
hook: SessionStart
hook: SessionStart Failed
hook: PreToolUse Failed
```

even though codex produces correct output, the cause is **codex 0.125's hook protocol expects stdout to be a single valid JSON object**, and your hook command emits an empty stdout (or non-JSON).

This commonly happens when migrating from older harnesses where [`mksglu/context-mode`](https://github.com/mksglu/context-mode) is wired into `~/.codex/hooks.json` — context-mode writes its session DB and exits silently (sometimes via SIGKILL with no output at all), which codex correctly classifies as `Failed`.

**Fix**: replace the raw `context-mode hook codex …` commands with the bundled `scripts/cm-hook-shim`, which backgrounds context-mode (so codex doesn't wait for it) and emits a minimal `{}` to stdout (codex's "no additional context" sentinel).

```bash
# 1. Symlink the shim onto PATH:
ln -sf "$(pwd)/scripts/cm-hook-shim" ~/.local/bin/cm-hook-shim

# 2. Patch ~/.codex/hooks.json to call the shim:
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".codex" / "hooks.json"
data = json.loads(p.read_text()) if p.exists() else {"hooks": {}}
data["hooks"] = {
  "PreToolUse":  [{"hooks": [{"type":"command","command":"cm-hook-shim pretooluse"}]}],
  "PostToolUse": [{"hooks": [{"type":"command","command":"cm-hook-shim posttooluse"}]}],
  "SessionStart":[{"hooks": [{"type":"command","command":"cm-hook-shim sessionstart"}]}],
}
p.write_text(json.dumps(data, indent=2))
PY

# 3. Verify on next codex run — you'll see "hook: ... Completed" instead.
```

To bypass context-mode entirely (no DB writes), set `CM_HOOK_DISABLE=1` in your environment — the shim still emits `{}` and returns 0 so codex stays happy.

## Disabling / uninstalling individual integrations

Every bootstrap supports a `TINYCTX_<NAME>_DISABLE=1` env var to skip auto-install, plus an `uninstall` subcommand to remove its codex config block (binary stays — uninstall the binary separately if you want):

```bash
tinyctx-gitnexus  uninstall                          # removes [mcp_servers.gitnexus]
tinyctx-serena    uninstall
tinyctx-caveman   uninstall
tinyctx-graphify  uninstall --project /path/to/repo  # removes per-project AGENTS.md section
tinyctx-scout-hook uninstall                          # removes scout SessionStart entry

# Skip a specific integration on next install.sh / tinyctx-up:
export TINYCTX_GITNEXUS_DISABLE=1
export TINYCTX_SERENA_DISABLE=1
export TINYCTX_CAVEMAN_DISABLE=1
export TINYCTX_GRAPHIFY_DISABLE=1
export TINYCTX_SCOUT_HOOK_DISABLE=1
export TINYCTX_MEM0_DISABLE=1
```
