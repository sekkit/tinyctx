# tinyctx

A thin context-routing proxy for the OpenAI Codex CLI that lets you run **mostly on a local 27B-class model** (Qwen3.6-27B by default) and **only escalate to a frontier model** (GPT-5.5) when truly needed.

> **Compose-first.** tinyctx itself is ~500 LOC of glue. The hard parts — code knowledge graph, LSP-backed symbolic ops, sandboxed tool execution, persistent memory, prompt compression — are delegated to mature OSS projects already wired in.

## Why this exists

Codex CLI sends the entire conversation history on every turn (it does not use `previous_response_id` — see [openai/codex#4047](https://github.com/openai/codex/issues/4047)). When the rolling history hits 90% of the model's context window, codex triggers a compaction step that calls the same expensive frontier model to produce a "handoff summary." That summary call is *also* billed at frontier rates. And because codex resends the whole history each turn, prompt-cache hits depend on byte-stable prefixes — which the compaction prompt and reasoning items routinely break.

Three load-bearing observations from the literature:
1. **Prefix reuse, not algorithmic compression, is the dominant cost lever.** LMCache reports ~92% prefix-reuse in Claude Code traffic, climbing to ~98% in execution phase. With Anthropic prompt-caching's 10× read discount, that's ~5–10× cost reduction by itself.
2. **The compaction summary is the easiest call to redirect.** It's a stateless transform of conversation history into a condensed handoff. A 27B model does it well enough; a frontier model is wasted there. ([openai/codex#20972](https://github.com/openai/codex/issues/20972) is the open feature request for exactly this.)
3. **Reasoning items are model-bound.** When you swap models mid-session, prior `encrypted_content` from the other model's reasoning is undecryptable and crashes the request ([openai/codex#17541](https://github.com/openai/codex/issues/17541)). Any router that swaps backends must scrub these.

tinyctx sits between codex and the model, owns those three problems, and delegates everything else.

## Architecture

```
codex CLI
  │
  │  Responses API (HTTP, wire_api = "responses")
  ▼
┌─────────────────────────────────────────────────────────────┐
│  tinyctx-proxy   (FastAPI, ~500 LOC, the only original code)│
│                                                             │
│  • Compaction interceptor   (handoff-prompt → local model)  │
│  • Cascade router           (heuristic + optional classifier)│
│  • encrypted_content scrub  (when crossing model boundaries)│
│  • Cache-discipline         (byte-stable system + tools)    │
│  • Optional pre-escalation compression (LLMLingua-2 hook)   │
└─────────────────────────────────────────────────────────────┘
  │                                       │
  │ default ~80%                          │ escalate ~20%
  ▼                                       ▼
Local Qwen3.6-27B                       GPT-5.5
(via LMStudio at 127.0.0.1:1234         (via chatgpt.com/backend-api/codex
 or vLLM/SGLang)                         or icodeeasy.cc/v1)

In parallel, codex talks directly to MCP servers (these are existing OSS,
tinyctx just provides the install script and a sane default config):

  • graphify           — code knowledge graph + multimodal indexing
                         (safishamsi/graphify, MIT)
  • serena             — LSP-backed symbolic operations (find_symbol,
                         find_referencing_symbols, replace_body)
                         (oraios/serena, MIT)
  • context-mode       — sandboxed tool execution (already installed)
                         (mksglu/context-mode, Elastic 2.0)
  • mem0               — persistent project memory across sessions
                         (mem0ai/mem0, Apache 2.0)
```

## What we build vs. what we wire

| Layer | Tool | Provenance | License |
|-------|------|------------|---------|
| Wire intercept | **tinyctx-proxy** | this repo | MIT |
| Compression-biased ranker | **tinyctx.interest** | this repo, after [arxiv 2603.20396 §5.1](https://arxiv.org/abs/2603.20396) | MIT |
| Code graph | graphify | upstream OSS | MIT |
| Symbolic ops (LSP) | serena | upstream OSS | MIT |
| Tool sandbox | context-mode | already installed | Elastic 2.0 |
| Output / tool-desc compression | [caveman](https://github.com/JuliusBrussee/caveman) (`caveman-shrink` MCP) | upstream OSS | MIT |
| Memory | mem0 (optional) | upstream OSS | Apache 2.0 |
| Inference (cheap) | LMStudio + Qwen3.6-27B | upstream OSS | Apache 2.0 |
| Inference (frontier) | OpenAI/codex backend | upstream | proprietary |

If something isn't pulling its weight, replace it. The only piece tinyctx assumes lives forever is its own ~500 LOC proxy, because the four jobs it does aren't covered by any OSS project we found.

## Quick start

```bash
cd ~/dev/tinyctx
./scripts/install.sh      # installs python deps, graphify, serena, configures codex
./scripts/start.sh        # starts the proxy on 127.0.0.1:4141
codex --profile tinyctx   # uses the proxy as model_provider
```

See [docs/install.md](docs/install.md) for the long version and [docs/test.md](docs/test.md) for the test plan.

## Troubleshooting: `hook: ... Failed` in codex output

If you're seeing lines like

```
hook: SessionStart
hook: SessionStart Failed
hook: PreToolUse Failed
```

even though codex produces correct output, the cause is **codex 0.125's
hook protocol expects stdout to be a single valid JSON object**, and your
hook command emits an empty stdout (or non-JSON).

This commonly happens when migrating from older harnesses where
[`mksglu/context-mode`](https://github.com/mksglu/context-mode) is wired
into `~/.codex/hooks.json` — context-mode writes its session DB and exits
silently (sometimes via SIGKILL with no output at all), which codex
correctly classifies as `Failed`.

**Fix**: replace the raw `context-mode hook codex …` commands with the
bundled `scripts/cm-hook-shim`, which backgrounds context-mode (so codex
doesn't wait for it) and emits a minimal `{}` to stdout (codex's "no
additional context" sentinel).

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

To bypass context-mode entirely (no DB writes), set
`CM_HOOK_DISABLE=1` in your environment — the shim still emits `{}` and
returns 0 so codex stays happy.

## Picking a local backend

The `[local]` section of `~/.tinyctx/config.toml` decides where the cheap
path lands. Four supported shapes — pick one in `examples/config.toml`,
copy the chosen block into `~/.tinyctx/config.toml`, then `tinyctx-up
restart`:

| Backend | wire_api | Pros | Cons |
|---|---|---|---|
| **LMStudio** + qwen3.6-27b on local GPU | `responses` | free, native Responses API, `reasoning_content` round-trips cleanly | slow on consumer Macs (~3 min/turn at xhigh reasoning); needs the chat template that emits qwen-pythonic XML |
| **DeepSeek API** (`deepseek-v4-flash`/`-pro`) | `chat` | ~5–7 s/turn, very cheap, `reasoning_content` preserved through proxy | costs $$ per turn (small); needs API key |
| **Ollama** | `chat` | free, easy install | no `reasoning_content` for non-reasoning models |
| **vLLM / SGLang** | `responses` | self-hosted at scale | infra overhead |

### Using a paid API backend (DeepSeek, OpenAI, OpenRouter, …)

Keys never go into committed config. They live in `~/.tinyctx/secrets.env`,
chmod 600. `scripts/start.sh` sources the file at proxy launch (and the
launchd plist runs `start.sh`, so secrets survive a reboot):

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

The `request_trace` event will show `target_url=https://api.deepseek.com/v1/...`
and `translated=True` (chat→responses round-trip).

## Compression-biased context ranking

`tinyctx/interest.py` is a faithful adaptation of §5.1 of Aksenov, Bodnia,
Freedman, Mulligan, *Compression Is All You Need: Modeling Mathematics*
([arxiv 2603.20396](https://arxiv.org/abs/2603.20396)) to a code corpus.
The paper's central empirical finding is that human mathematics lives in
the polynomial-growth (A_n, log-density) regime: a small number of
hierarchically-nested macros buys exponential expansion. The proposed
ranking is PageRank with **teleportation biased toward high-compression
nodes** — formally, T_0(u) = unwrapped/wrapped (reductive compression) and
I_0(u) = body/signature (deductive compression), combined as J_0 = β·T_0 +
(1−β)·I_0.

For a coding agent, the same principle picks the load-bearing abstractions
(utility modules, well-named interfaces, deeply-nested but terse APIs)
that should be primed into the prompt. It's a strictly stronger ranking
than uniform-personalization PageRank (which is what aider's repomap
does) because it uses the corpus's own compression structure as the
inductive bias.

```bash
# Build a graphify graph for the repo, then rank symbols for a query:
graphify .
.venv/bin/python -m tinyctx.interest graphify-out/graph.json "auth token verification"
```

The format expected for `graph.json` is small and explicit; see the
docstring of `tinyctx/interest.py`. Plugs into a graphify export with one
adapter step (every node needs `wrapped_signature`, `wrapped_body`,
`deps`).

## Components

```
tinyctx/
  proxy.py              FastAPI server. /v1/responses + /v1/chat/completions, SSE-preserving forward
  router.py             Route decision: heuristic + optional learned classifier
  sanitize.py           encrypted_content scrub + chat normalization + DCP dedup/purge + cache-aware gate
  config.py             Layered config: defaults < TOML file < TINYCTX_* env
  compactor.py          3-role debate + judge merge → markdown summary + structured compartments/facts
  continuity.py         Persist compactions (.md + .json sidecar); --facts-only / --compartment recall
  historian.py          Rolling per-session compression: async update + on-wire substitution (gated)
  scout.py              Two-layer init scan: graphify + local-LLM subagent + content-hash cache
  keypin.py             Scan codex rollouts for Read-frequency; emit byte-stable keyfiles.md
  memory.py             Optional mem0 wrapper for cross-session user/project memory
  interest.py           Compression-biased PageRank ranker (paper §5.1)
  graphify_adapter.py   graphify graph.json -> interest.py shape
  classifier.py         Pure-Python logistic regression for escalation scoring
  stats.py              JSONL log -> route mix / cost / latency report
  registry.py           Per-machine list of projects tinyctx has touched (used by dreamer)
  dreamer.py            Periodic maintenance CLI: scout refresh + keypin scan + mem ingest + GC

scripts/
  install.sh            Idempotent installer (venv + graphify + serena + caveman vendor)
  start.sh              Boot the proxy
  smoke_codex.sh        Real codex CLI smoke test against fake backends

.codex-plugin/          Codex marketplace plugin manifest (plugin.json + hooks)

tests/                  6 files, 34 tests, no network required
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
tinyctx-dreamer run --gc                            # rebuild scout + keyfiles for every registered project
tinyctx-dreamer install-launchd                     # install macOS daily 03:00 timer
tinyctx-stats                                       # routing summary from logs
tinyctx-interest <graph.json> "<query>"             # rank symbols by compression-biased PageRank
python -m tinyctx.graphify_adapter <graphify.json>  # convert a graphify export
python -m tinyctx.classifier train <labeled.jsonl>  # train escalation scorer (optional)
python -m tinyctx.classifier predict est_tokens=80000 turn_count=20
```

## Multi-subagent compaction (and what happens when a session runs out)

When codex hits ~90% of its context window it triggers an internal
compaction step that asks the model to write a "handoff summary" of the
conversation so far. By default that call costs frontier rates. tinyctx
already redirects that call to the local 27B (the highest-leverage cost
win, and `openai/codex#20972` is the open feature request). What it now
*also* does — when `compactor_debate = true` — is replace the single-pass
local summary with a **3-role debate + judge merge**:

```
                     parallel local calls (~one wall-clock LLM round)
   archaeologist  →  preserves verbatim facts, file paths, exact errors
   narrator       →  preserves intent and the storyline of attempts
   enumerator     →  lists every concrete artifact (files, commands, errors)
                                 │
                                 ▼
                          judge (one local call)
                          merges the three drafts into a canonical
                          handoff summary with fixed sections
```

Per-role drafts run in parallel via `httpx.AsyncClient`, so wall-clock
latency is roughly one local-model call, not three. Failure tolerance is
graded: if 1 of 3 roles fails the judge merges the survivors; if the judge
itself fails we deterministically concatenate the role drafts; if all role
calls fail the proxy falls back to a straight forward to the local backend
without the debate. **codex never sees a hard failure from the compactor.**

Every successful compaction is persisted to
`~/.tinyctx/cache/<repo-hash>/sessions/<sid>/compaction-N.md`. When a
session truly runs out of room and codex makes you `/clear`, recall the
last compaction into the new session:

```bash
tinyctx-recall                # print the most recent compaction
tinyctx-recall --list         # show all sessions for this repo
tinyctx-recall --all-sessions # print compactions across every prior session
```

We deliberately do NOT auto-inject compaction summaries on SessionStart —
that would conflict with codex's own resume behaviour. Recall is opt-in.

Config switches (file overrides default; env overrides file):

```toml
[routing]
redirect_compaction_to_local    = true     # default; the load-bearing flag
compactor_debate                = true     # default; off = single-pass local
compactor_min_history_tokens    = 4000     # below this, skip the debate
save_compactions                = true     # persist for tinyctx-recall
```

## Two-layer project scout

The first time codex enters a repo, you don't want it spending frontier
tokens just to figure out where things live. tinyctx ships a **two-layer
scout** that solves this without doing the wasteful "summarize everything
at install time" thing:

```
Layer 1  (free, no LLM)        graphify / aider repomap → graph.json
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
```

Workflow:

```bash
# 1. Build a static graph (no LLM, free).
graphify .                                              # or any tree-sitter indexer
python -m tinyctx.graphify_adapter graphify-out/graph.json --out tinyctx-graph.json

# 2. One-shot scout build (talks to local LMStudio/vLLM at $TINYCTX_LOCAL_BASE_URL).
tinyctx-scout init --graph tinyctx-graph.json --top-k 40
# → ~/.tinyctx/cache/<repo-hash>/scout.md       (≤2K tokens, byte-stable)
#   ~/.tinyctx/cache/<repo-hash>/manifest.json  (file content hashes)

# 3. (Optional) verify staleness next session.
tinyctx-scout status
# state: fresh
# nodes: 40 (top-k=40)

# 4. Codex's SessionStart hook (registered by the marketplace plugin) auto-
#    injects scout.md as additionalContext on every codex run. If any tracked
#    file changed since last build, a background refresh kicks off; the (now
#    slightly stale) cache is still used for this session.
```

This is the **subagent shape** by construction: scout.py runs as an isolated
local-model call with a fixed short system prompt, persists a digest, and the
parent (codex) only ever sees the digest — never the raw 50-file scan that
went into producing it.

## Borrowed from cortexkit/magic-context

A separate analysis of [magic-context](https://github.com/cortexkit/magic-context) (OpenCode plugin, MIT) influenced seven additions across the v0.4 → v0.5 releases. We did not vendor any code (different host, different license assumptions) — these are independent reimplementations of the underlying ideas:

| From magic-context | tinyctx implementation |
|---|---|
| Cache-aware deferred drops (queue ops, fire only on TTL/threshold) | `sanitize.CacheAwareMutator` — gates dedup/purge/historian-substitution so the prompt-cache prefix stays warm in the common case |
| Pristine recomputation invariant (compress always from raw, never from prior summary) | Documented invariant in `compactor.py` + `test_pristine_recomputation_guard` |
| Pinned key files via read-frequency (dreamer.pin_key_files) | `tinyctx-keypin scan` — scans codex rollouts, ranks files by Read-tool frequency |
| Compartments + facts structured compaction | `compactor.parse_judge_output` produces compartments / facts / open_questions; `continuity.save_compaction` writes a `.json` sidecar; `tinyctx-recall --facts-only` / `--compartment <name>` |
| User memory promotion (mem0-style cross-session) | `tinyctx.memory` — thin wrapper over [mem0ai](https://github.com/mem0ai/mem0); optional dep `pip install 'tinyctx[mem]'` |
| Historian (rolling per-turn compression of older history) | `tinyctx.historian` — async background `update()` after every Nth turn writes a digest sidecar; opt-in `historian_substitute` replaces older turns on the wire (gated by CacheAwareMutator) |
| Dreamer (periodic background consolidation) | `tinyctx-dreamer run` — scout refresh + keypin scan + (optional) mem0 ingest for every registered project, plus session-cache GC; `install-launchd` / `install-cron` wires it to a daily timer |

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

Implementation: `tinyctx/advisor.py` is a stdio MCP server. It exposes `ask_advisor(question, context, previous_attempts)` to the executor. Each call is routed back through the running tinyctx proxy with `model="tinyctx-frontier"` (client-forced route), so:

- existing frontier auth/config is reused (no duplicate key plumbing)
- every advisor call shows up in `tinyctx-trace --watch` with `forced_by_client_model=true`
- swapping the frontier (e.g. to a different provider) requires zero advisor changes

Wire it into `~/.codex/config.toml`:

```toml
[mcp_servers.advisor]
type = "stdio"
command = "/path/to/tinyctx/.venv/bin/python"
args = ["-m", "tinyctx.advisor"]

[mcp_servers.advisor.env]
TINYCTX_PROXY_URL = "http://127.0.0.1:4141/v1"
TINYCTX_ADVISOR_MODEL = "tinyctx-frontier"
TINYCTX_ADVISOR_TIMEOUT_S = "180"
```

Auth fallback: if `TINYCTX_ADVISOR_API_KEY` is unset, `advisor.py` reads `~/.codex/auth.json` and forwards the codex `access_token` as the Authorization bearer — same source codex itself uses, so the advisor inherits the user's existing chatgpt login automatically.

**Codex namespace handling** (added in this fix):
- **Codex 0.128+** wraps MCP-server tools at the wire level inside `type: "namespace"` shells (`{"type": "namespace", "name": "mcp__advisor__", "tools": [{"type": "function", "name": "ask_advisor", ...}]}`). Without expansion, those wrappers get dropped by the function-only scrub and the executor never sees `ask_advisor`. The proxy now runs `expand_mcp_namespaces` before scrub: namespace shells become top-level `type=function` entries with names like `mcp__advisor__ask_advisor`, so DeepSeek (and any other chat-completions backend) can see and call them. Set `TINYCTX_MCP_NAME_NO_PREFIX=1` if a future codex build expects the bare inner tool name instead.
- **Codex 0.128.0-alpha.1 known dispatcher limitation**: even after the executor model invokes the expanded tool, codex's internal `core/src/tools/router.rs` returns `unsupported call: <name>` for namespace-expanded MCP tools — both with and without the namespace prefix. The wire expansion is verified correct end-to-end (the executor genuinely emits the function call; the dispatch hop on codex's side is what fails). Watch the `code_mode` / `tool_search_always_defer_mcp_tools` features for codex's fix; tinyctx's expansion will work the moment codex's dispatcher accepts the call.
- **Codex 0.125** doesn't expose MCP tools to the executor at all (everything goes through `tool_search` which isn't surfaced to the model in current builds). Upgrading to 0.128+ is required for advisor visibility, even though the dispatcher round-trip is still pending.

Anthropic's blog reports +2.7 SWE-bench points at -11.9% cost using this pattern (Sonnet+Opus pairing). The DeepSeek+gpt-5.5 pairing should see similar shape — the executor handles 99% of turns at deepseek pricing, and burns the frontier only on the 1-3 hard decisions per task that actually need it.

## Status

- v0.5.0 — works end-to-end with fake backends and against real codex CLI 0.125.0 / Codex.app 0.128.0-alpha.1.
- **140 tests across 17 files, all passing** (incl. 21 advisor tests + 3 namespace expansion tests + the proxy+compactor integration test).
- 6 OSS upstream dependencies wired (graphify, serena, caveman, mem0, LMStudio, magic-context-as-inspiration); 0 reinvented.
- Total original code: ~4,400 LOC across 18 modules.
