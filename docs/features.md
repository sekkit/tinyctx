# tinyctx 功能清单

基于当前源码（`tinyctx/` 18 个模块 + `scripts/` + `.codex-plugin/`）逐模块梳理的完整功能列表。
不包含 README 已经覆盖的"为什么存在"叙述，只列**实际代码做了什么**。

---

## 1. 架构总览

```
codex CLI ──HTTP──▶ tinyctx-proxy (FastAPI, 127.0.0.1:4141)
                      │
                      ├─ /v1/responses    主路径
                      ├─ /v1/chat/completions   兼容路径
                      ├─ /                简介页
                      └─ /v1/models       三个 alias 模型
                      │
                      ├─ 路由判定          → router.decide
                      ├─ Sanitize 管线    → sanitize.* + read_delta + historian
                      ├─ Compaction 拦截 → compactor (3-role debate)
                      ├─ Tool-call 翻译   → tool_call_translator
                      └─ Trace 落盘       → trace.RequestTrace
                      │
                      ▼ 80%               ▼ 20%
                    本地 27B            frontier (gpt-5.5)
                    (LMStudio / DeepSeek / Ollama)
```

辅助 CLI：`tinyctx-scout` / `-recall` / `-keypin` / `-mem` / `-trace` / `-stats` / `-dreamer` / `-interest`。

---

## 2. Wire-level 代理（[proxy.py](tinyctx/proxy.py) · [router.py](tinyctx/router.py)）

### 2.1 路由判定（`router.decide`）

| 触发 | 决策 | 依据 |
|---|---|---|
| `force_route=local/frontier` | 强制 | 配置或环境变量 |
| 检测到 codex compaction handoff prompt | local | `_COMPACTION_FINGERPRINTS` 三个正则 |
| `error_streak ≥ escalate_on_error_streak` | frontier | per-session 错误连击计数 |
| local rolling failure rate ≥ `adaptive_model_failure_rate_threshold` | frontier | SmallCode-style adaptive model select（默认 3 calls / 30% / 20-sample window） |
| `est_tokens ≥ local.context_window × safe_fraction` | frontier | 优先使用 backend 自带上下文窗口（动态），没设则退回 `escalate_input_tokens` 绝对值 |
| `turn_count ≥ escalate_turn_count` | frontier | 对话轮数门槛 |
| 学得分类器 `p(escalate) ≥ 0.7` | local/advisor-only（默认）或 frontier（legacy 开关） | 可选，见 §11 |
| 默认 | local | "small/short → cheap path" |

`is_compaction_request()` 是高杠杆关键：把 codex 的 handoff summary 调用本身从 frontier 改路到 local，省掉一大笔 frontier 计费。

### 2.2 客户端强制路由

请求里 `model` 字段为：
- `tinyctx-local` → 强制走本地
- `tinyctx-frontier` → 强制走 frontier（advisor sub-agent 用这个）
- 其他 → 走默认路由

### 2.3 上游转发

- `httpx.AsyncClient` 流式转发，保留 SSE
- 连接级重试：`AsyncHTTPTransport(retries=N)`（DNS / TCP / connect timeout 自动重试一次，HTTP 错误透传）
- 读超时 600s，写 60s
- 流式失败时自动注入 `event: response.completed` SSE 终结帧（`status=incomplete`），防止 codex.app 的 SSE 解析器抛 "stream closed before response.completed"

### 2.4 Per-session 状态

- `_SESSION_ERROR_STREAK`：错误连击；HTTP ≥400 / 网络错 +1，正常 +0；高于阈值触发 frontier 升级
- `adaptive_model.STATE`：按 backend key 维护 local 最近 N 次成功/失败；失败率过高时自动把 `auto` 请求切到 frontier，显式 `tinyctx-local` 仍优先
- `_MUTATOR`：`CacheAwareMutator`，控制 history-mutation 的 cache 友好门控（见 §4.4）
- `_PROACTIVE_SUMMARY_CACHE`：proactive_compact 的 bucket 化 summary 缓存

---

## 3. Sanitize 管线（[sanitize.py](tinyctx/sanitize.py)）

按 proxy 调用顺序：

### 3.1 `strip_encrypted_content`
扫 `input` / `messages` 里的 reasoning / reasoning_summary / thinking 项，删 `encrypted_content`。这是 [openai/codex#17541](https://github.com/openai/codex/issues/17541) 的正解 —— 不删就跨模型解不开报错。

### 3.2 History-hygiene transforms（cache-aware 门控）

| 函数 | 行为 |
|---|---|
| `dedup_tool_calls` | 同 `(name, arguments)` 哈希的多次 tool_call，保留最后一次，前面的 args/output 替换为占位符 |
| `purge_failed_tool_inputs` | 失败的 tool_call 在 `after_turns` 个 assistant 轮后，把巨大的 input（如失败 patch）替换成占位符；error 输出保留 |
| `read_delta.collapse_repeated_reads` | 同 path 的 Read 重读 → 第二次起换成 unified diff 或 "unchanged" 标记，见 §6 |
| `historian.apply_to_body` | 把老 turns 替换成 `<tinyctx-historian-digest>` 单条 system，见 §5 |

### 3.3 `expand_mcp_namespaces`
codex 0.128+ 把 MCP 工具裹在 `type: "namespace"` shell 里，这里展开成顶层 `type: "function"` 条目，每个内部工具名前缀 `mcp__<server>__<tool>`。否则下一步的 scrub 会把 namespace 整个丢掉，advisor 等 MCP 工具就消失了。

### 3.4 `scrub_unsupported_tools`
按 backend 的 `supported_tool_types`（默认 `("function",)`）过滤工具，丢掉 `web_search` / `image_generation` / 残留的 `namespace` 等本地后端不接受的类型。空集合 = 透传。

### 3.5 `strip_unsupported_responses_fields`
丢掉 codex 发的、严格后端拒收的字段：默认 `client_metadata`、`prompt_cache_key`。

### 3.6 `inject_responses_defaults`
按 dotted-path 注入缺省字段（仅当不存在时）。例如 LMStudio 0.4 必填 `text.format.type`，本地默认注入 `"text"`。

### 3.7 `cap_responses_fields`
**强制下限**字段封顶。和 inject 的区别：cap 会**覆盖**已有值。生产用例：codex.app 默认发 `max_output_tokens=128000`，cap 到 16000，防止 DeepSeek 跑 80 秒 / 1.6 MB 的 thinking-loop。

### 3.8 `inject_advisor_hint`
往 `instructions` 末尾追加约 1500 字节的 advisor 使用指南（Anthropic Advisor Strategy 完整对齐版本：四类触发场景 / 频率 / advice 处理 / reconcile-call）。
- 跳过 `model="tinyctx-frontier"`（advisor 自身）
- 幂等：检测 `_ADVISOR_HINT_MARKER` 防重复注入
- `frontier_skip_advisor_hint=true` 时 frontier 路径整段跳过（省 1-2K tokens）
- 环境变量 `TINYCTX_INJECT_ADVISOR_HINT=0` 全局禁用

### 3.9 `normalize_for_chat`
Responses-API → chat-completions 的 lossy 转换（仅当 backend `wire_api="chat"`）：
- 携带 reasoning text → 下条 assistant 的 `reasoning_content`
- `_TOOL_RESULT_TYPES` 通配（function_call_output / tool_result / mcp_result）做 call/result 配对
- 孤儿 `function_call`（call_id 没有对应 output）合成占位符 tool message —— 防止 chat API 拒收并避免模型"忘记调用过 X"
- `_flatten_tool_output` 拍平 list 形式的输出，把 `input_image` 替换成 `[image attached]` 占位符
- `max_output_tokens` → `max_tokens` 翻译，确保 cap 实际生效
- 强制每个 assistant 消息带 `reasoning_content`（即使空），满足 DeepSeek thinking mode 严格校验
- 合并连续 assistant tool_calls 到单条消息

### 3.10 `proactive_compact`（见 §5.3）
### 3.11 `trim_tools_for_frontier`（见 §7.5）

### 3.12 `CacheAwareMutator`
门控所有 history-mutating transforms（dedup / purge / read_delta / historian-substitute）。规则：
- 首轮（session 没记录过）→ 不 fire
- `context_usage ≥ threshold`（默认 0.65）→ fire
- 距上次 fire ≥ `ttl_seconds`（默认 300，对齐 Anthropic 5min cache TTL）→ fire
- 否则 defer
保留 prompt-cache 命中是首要目标，mutation 只在 cache 反正快过期时打开。

---

## 4. Compaction 处理

### 4.1 `compactor.compact_with_debate`（[compactor.py](tinyctx/compactor.py)）
3-role 并行 + 1-judge 合并：

| 角色 | 作用 |
|---|---|
| archaeologist | 保留 verbatim：file path、命令、报错、决策 |
| narrator | 4-8 句 storyline，意图 + pivot |
| enumerator | 三类清单：改动文件 / 命令+结果 / 待解 issue |
| judge | 合并三份草稿，输出 markdown summary + 一个 fenced JSON（compartments / facts / open_questions） |

- 三份 role draft 走 `httpx.AsyncClient` 并发，wall-clock ≈ 1× 本地调用
- 容错：1 个 role 失败 → judge 合并幸存者；2+ role 失败 → 单 draft polish；judge 失败 → `_fallback_concat`；全失败 → `[tinyctx compactor: ...]` 占位
- proxy 层再包一层 try：compactor 抛任何异常都回落到普通 forward，**永不让 codex 看到硬错**
- **PRISTINE recomputation invariant**：每次 compaction 必须基于原始 history，绝不用前一次的 summary 作输入；防止跨 compaction 的 lossy drift

### 4.2 `parse_judge_output`
拆判官输出为 `(markdown_summary, structured_dict)`。先尝试 `_JSON_FENCE_RE`，回退 `_BARE_JSON_RE`，最后兜底返回 `(text, empty_structured)`。永不抛异常。

### 4.3 Responses-API wire shape
- `build_responses_api_payload`：非流式 JSON（带 `usage` 估算）
- `build_responses_api_sse`：流式 SSE（response.created → output_item.added → output_text.delta → output_text.done → response.completed）

### 4.4 持久化（[continuity.py](tinyctx/continuity.py)）
路径布局：
```
~/.tinyctx/cache/<repo-hash>/sessions/<sid>/compaction-N.md
                                          /compaction-N.json   # structured sidecar
                                  /latest.md                   # symlink to latest
```

CLI `tinyctx-recall`：
- 默认打印当前 repo 最近一次 compaction
- `--list` 列所有 session + 计数
- `--all-sessions` 跨 session
- `--limit N` 限制输出数
- `--facts-only` 只打印 structured.facts
- `--compartment NAME` 只打印某个 compartment

---

## 5. 历史压缩

### 5.1 `historian` rolling 压缩（[historian.py](tinyctx/historian.py)）

两半：
- **update half**：异步后台。每过 `min_new_turns` 轮触发一次，对 `items[:-recent_keep]` 调本地 LLM 压成 markdown digest + structured 侧车，落盘到 `cache/<repo>/sessions/<sid>/historian-N.md`。无副作用，可常开。
- **apply half**（`apply_to_body`）：同步，把 body 的老 items 替换成单条 `<tinyctx-historian-digest>` system 消息。改 history 字节，**默认关闭**，需要 `historian_substitute=true` + cache-aware 门控放行。
- 幂等：检测前 5 条里有 `_DIGEST_TAG` 就直接返回原 body
- `spawn_update`：fire-and-forget Task，强引用计数避免 GC

### 5.2 `read_delta` 重复 Read 压缩（[read_delta.py](tinyctx/read_delta.py)，本次新增）

把同 path 多次 Read 的 result 替换成 unified diff：
- **检测**：三类 read tool
  1. 命名工具：Read / read_file / view / view_file / fs_read / file_read / container.read / open_file
  2. shell / exec_command / container.exec / bash + 首 token 在 `{cat, head, tail, less, more, bat}`，或 `sed -n`
  3. `mcp__*__*` 末段含 `read` / `view` / `cat`
- **基线**：第一次 Read 保留全文；第二次起 diff vs 第一次（baseline 稳定，model 看到一致参考帧）
- **三种结果**：
  - 内容一致 → `[tinyctx: re-read of <path> — unchanged since first read]`
  - 内容变化 → header + `difflib.unified_diff`（n=3）
  - diff > 原文 × 0.85 → 不替换（写得不偿失）
- **跳过**：output < `min_bytes`(400)、看起来是错误（`no such file` / `permission denied` / 等）
- **形态保护**：list-form output 维持 list 形态，单 string output 保留 string；不 mutate 入参
- 配置：`read_delta_enabled` / `read_delta_min_bytes` / `read_delta_max_diff_budget`
- 接入 `CacheAwareMutator` 同一门控

### 5.3 `proactive_compact` 主动截断（[sanitize.py](tinyctx/sanitize.py)）

防 `Codex ran out of room` 的最后一道防线。

- 门控：`est_tokens ≥ effective_proactive_compact_threshold(cfg)`
- 阈值动态推导：`frontier.context_window × proactive_compact_safe_fraction`（默认 0.75）。换 frontier 模型自动跟随
- 切片：保留 head（system/developer 项） + 单条 summary message + tail（最后 `recent_keep` 项，默认 8）
- summary 来源：
  - 本地 summarizer 调用（`proactive_compact_use_summarizer=true`，默认开）
  - 失败/关闭时退回确定性 placeholder
- **bucket 化缓存**：`_PROACTIVE_SUMMARY_CACHE` 用 `(sid, len(middle) // 20)` 做 key。同 bucket 复用 summary，bucket flip 才重算 → 单 session 大量请求只跑一次 summarizer（避免每轮调用 + 每轮看到新 summary 的"漂移性遗忘"）
- **增量 seeding**：bucket flip 时把上一桶 summary 拼到 prompt 前，让新 summary 是旧 + 新增内容的 refine（保留连续性）
- **孤儿修复**：tail 切割可能留下 `function_call_output` 但 call 在被截断的 middle 里 → chatgpt 后端会 400。 `proactive_compact` 给孤儿 output 前面合成 stub `function_call`（name=`tinyctx_compacted_call`）
- **frontier-only**：默认仅 frontier 路径开（`proactive_compact_only_on_frontier=true`）；本地 1M 上下文不需要、且 model 受益于完整 history
- 跳过 codex 自己的 compaction request（"don't compact a compact"）
- `clear_proactive_cache()`：测试辅助

---

## 6. Tool-call 翻译（[tool_call_translator.py](tinyctx/tool_call_translator.py)）

### 6.1 XML → 结构化 `function_call`
qwen3-coder / Barubary chat template 输出 `<tool_call><function=NAME><parameter=KEY>...</parameter></function></tool_call>` XML，codex CLI 0.125 不接受。

- `parse_tool_call_block`：宽松 regex 解析每个完整 block，参数值过 `_coerce_value`（true/false/null/JSON 数组对象 → 原生类型，其他保留字符串）
- `rebuild_response`：非流式 JSON 响应，扫 `output[]` 把含 XML 的 `output_text` part 拆出 `function_call` items（多个 call → 多个 item），保留剩余文本，幂等
- `StreamTranslator`：流式 SSE 状态机。缓冲 delta，识别完整 `<tool_call>` 块，发出 `output_item.added` (function_call) → `function_call_arguments.delta` → `function_call_arguments.done` → 关 message item
- `_PARTIAL_OPEN_RE` 检测半 buffer 的开标签，避免过早 emit text

### 6.2 chat-completions SSE → Responses-API SSE
`ChatToResponsesTranslator` 把本地 chat 后端的 SSE 翻成 codex 期望的 Responses-API SSE：
- text content → `response.output_text.delta`
- `reasoning_content` → 单独的 `reasoning` output item（output_index=0）+ summary
- `tool_calls` → `function_call` output item（按 index 累积 arguments）
- finish_reason → 关闭 item + `response.completed`
- `_partial` 缓冲非完整 SSE event

### 6.3 Auto-answer for "user input" prompts
codex 0.128 的 `request_user_input` 函数会阻塞 session 直到用户点击。`TINYCTX_AUTO_USER_INPUT=1`（默认开）启用三层拦截：

| 层 | 触发 | 处理 |
|---|---|---|
| 1. 显式调用 | model 调 `request_user_input` 工具 | `_try_auto_answer_user_input`：解析 arguments，咨询 advisor（gpt-5.5），把决策附到 assistant 文本，丢掉原 function_call |
| 2. 文本式选项 | regex 在 tail 里匹配 cue（"请选择"等）+ ≥2 enum 项 | `_try_auto_answer_text_choice`：调 advisor 选一个 |
| 3. 分类器兜底 | zero-tool-call 的 final 文本看起来像 "等待用户" | `_classify_final_answer` 调本地 DeepSeek (~$0.0001) 分类，YES 才调 advisor |

三层从 cheap 到 expensive 渐进。advisor 决策追加到 `_text_buf`，session 不阻塞。

---

## 7. Token-budget 优化（frontier 专属）

### 7.1 `proactive_compact` — §5.3
### 7.2 `frontier_skip_advisor_hint` — §3.8
### 7.3 `cap_fields` — §3.7
### 7.4 `proactive_compact_only_on_frontier` — §5.3
### 7.5 `trim_tools_for_frontier`（[sanitize.py](tinyctx/sanitize.py)）

codex 0.128 每请求带 ~50 个工具（~10K tokens），实际只用几个。
- 扫最近 `recent_window`（默认 30）个 input items 里 `function_call.name`
- 保留：用过的工具 ∪ `frontier_tools_essentials`（shell / apply_patch / container.exec / update_plan / view_image / image_view / mcp__advisor__ask_advisor / 所有 `mcp__advisor__*`）
- < 5 工具不动
- 默认 frontier 开、本地关（1M ctx 吸收得起）

---

## 8. 知识 / 上下文层

### 8.1 Compression-biased PageRank（[interest.py](tinyctx/interest.py)）

[arxiv 2603.20396](https://arxiv.org/abs/2603.20396) §5.1 在代码图上的实现：

| 步骤 | 含义 |
|---|---|
| `compute_unwrapped` | 递归算 unwrapped 长度 = wrapped + sum(deps unwrapped)，处理环 |
| `compute_compression` | T0(u) = unwrapped/wrapped（reductive），I0(u) = body/sig（deductive） |
| `compute_j0` | J0 = β·T0_norm + (1-β)·I0_norm，log-scale min-max 归一化（paper 提到跨 10²-10¹⁰⁴ 数量级） |
| `compression_pagerank` | 偏置 teleportation 的 PageRank，转移 P(v,u) = α·w(u,v)/W(u) + (1-α)·J0(v)/Z |
| `rank_for_query` | 加 query token 子串匹配作 personalization seed（aider repomap 风格），与 J0 multiplicative blend |

CLI `tinyctx-interest <graph.json> "<query>"`：打印 top-K 节点。

### 8.2 graphify 适配器（[graphify_adapter.py](tinyctx/graphify_adapter.py)）

把 graphify / NetworkX node-link / flat 三种 graph.json 形态统一转成 interest.py 期望的 shape。
- `_extract_id`：扫 id/name/qualified_name/fullname/path
- `_extract_text`：扫 code/text/snippet/summary/body/content/label
- `_extract_deps`：扫 deps/dependencies/references/imports/calls/neighbors
- `_split_signature_body`：第一行 = sig，剩下 = body
- 自动去重

CLI `python -m tinyctx.graphify_adapter <in.json> --out tinyctx-graph.json`。

### 8.3 Project Scout（[scout.py](tinyctx/scout.py)）

两层 init scan：
- **Layer 1（free）**：graphify 静态扫，produce graph.json
- **Layer 2（一次性 LLM）**：interest.py 排 top-K → 收集源码 snippet（`max_file_chars` 截断）→ 调本地 27B 产 ≤2K tokens 的层级化 markdown summary → 写 `~/.tinyctx/cache/<repo>/scout.md`

byte-stable 设计（节点排序确定性 + 低温）→ 可塞进 prompt-cache 前缀不破坏命中。

`is_stale()`：扫 manifest 里每个 tracked 文件的内容哈希，任何变更 → stale。

CLI 子命令：`init` / `refresh` / `status` / `show` / `path`。

SessionStart hook：[scripts/scout-session-start.sh](scripts/scout-session-start.sh) 把 scout.md 注入为 codex `additionalContext`。

### 8.4 Pinned Key Files（[keypin.py](tinyctx/keypin.py)）

扫 `~/.codex/sessions/**/rollout-*.jsonl`，统计 Read-tool（含 `mcp__*__read/view/get`）调用频率，按 file path 计数，filter 到当前 project，写 byte-stable `keyfiles.md`：

| Reads | File |
|------:|:-----|
| 47 | `tinyctx/proxy.py` |
| 31 | `tests/test_proxy_integration.py` |
| ... | ... |

CLI：`tinyctx-keypin scan` / `show`。`scan` 自动 `registry.register(root)` 让 dreamer 带上这个 repo。

scout（结构性 PageRank）和 keypin（行为频率）互补 —— 一个理论 load-bearing，一个实测 load-bearing。

---

## 9. 跨 session 记忆（[memory.py](tinyctx/memory.py)）

可选 mem0ai 包装：
- `is_available()`：lazy import 检查
- `MemStore.add / search / get_all`：归一化 mem0 不一致的返回 shape（list vs `{"results": []}`）
- 默认配置走 tinyctx 现有的 local backend（OpenAI 兼容 endpoint）
- store_dir 默认 `~/.tinyctx/mem0/`

CLI `tinyctx-mem`：
- `available` — 检查 mem0 是否安装
- `add "fact"` / `search "query"` / `stats`
- `ingest-compaction` — 把最新 compaction 的 facts + compartments 灌入 mem0（`source: "compaction"` metadata）

故意不在 SessionStart 自动注入，避免破坏 prompt-cache。

---

## 10. Advisor sub-agent（[advisor.py](tinyctx/advisor.py)）

Anthropic Advisor Strategy 实现 —— stdio MCP server。

- 暴露一个工具 `ask_advisor(question, context, previous_attempts)`
- 调用走 tinyctx 代理本身（`TINYCTX_PROXY_URL/responses`）+ `model="tinyctx-frontier"`，所以路由 / 鉴权 / trace 全部复用主流量管线
- 鉴权 token 解析顺序：`TINYCTX_ADVISOR_API_KEY` → 读 `~/.codex/auth.json` 的 `tokens.access_token` → 空
- system prompt 完整对齐 Anthropic：100 词以内、enumerated steps、不 recap、不 hedge、reconcile 协议
- 输出格式严格：`1. ...` `2. ...` `Risks: ...`
- `_consume_responses_stream`：扫 SSE，处理 `output_text.delta` / `output_text.done` / `error` / `response.failed`，把 tinyctx proxy 注入的 `{"status":N,"body":...}` 还原成可读错误
- JSON-RPC 协议实现：`initialize` / `notifications/initialized` / `tools/list` / `tools/call`

Codex 0.128+ 推荐路径：注册成 codex agent（`spawn_agent(role="advisor")`），`agents/advisor.toml` 里 `model="tinyctx-frontier"`。0.125 fallback 路径：MCP server。

`request_user_input` 拦截链（§6.3）也调它。

---

## 11. 学得分类器（[classifier.py](tinyctx/classifier.py)）

FrugalGPT-style 升级评分器：
- 11 维特征：est_tokens / log_est_tokens / turn_count / error_streak / is_compaction / tool_call_count / max_message_chars / code_density / has_apply_patch / has_image / instructions_chars
- 纯 Python logistic regression + SGD（不依赖 sklearn / numpy）
- z-score 标准化 + L2 正则
- `extract_features(body)` 接 router.decide 上下游
- 加载顺序：`~/.tinyctx/classifier.json` 存在 → router 把它当第二意见，仅在 `prob ≥ 0.7` 时升级 frontier
- 训练数据 schema：`{"features": {...}, "label": 0|1}` per line，bootstrap 用 stats 日志（`decision == "frontier"` 当 label）
- CLI `python -m tinyctx.classifier train <jsonl>` / `predict key=value...`

---

## 12. 可观测性

### 12.1 `RequestTrace`（[trace.py](tinyctx/trace.py)）

Per-request 数据类，统一 emit 一条 `request_trace` JSONL 事件。30+ 字段：
- 基础：request_id / session_id / started_at / route / route_reason / is_compaction / est_input_tokens / turn_count / error_streak / requested_model / forced_by_client_model
- Sanitize 差异：encrypted_content_stripped / tools_before/after / tool_types_dropped / fields_stripped / fields_injected / fields_capped
- Mutation gate：mutation_wanted / fired / gate_reason / deduped_calls / purged_inputs / historian_substituted
- forward target：target_url / target_wire_api / target_model
- Response：status / is_stream / bytes_out / translated / translated_calls
- Compactor：compactor_used / outcome
- Proactive compact：applied / reason / items_before/after / middle_compacted / synthetic_calls / threshold_used
- Forwarded breakdown：forwarded_bytes / forwarded_tokens_est / forwarded_breakdown(instructions/tools/input/other)
- Frontier opt：advisor_hint_skipped / tools_trimmed_*
- Read delta：read_delta_applied / candidates / replacements / bytes_saved / paths

### 12.2 `tinyctx-trace` CLI

- 默认：最近 10 条紧凑表格
- `-v`：每条多行
- `--last N` / `--all`
- `--watch`：tail-follow 当天日志，新事件实时打印
- `--request rid` / `--session sid` 过滤
- `--since YYYY-MM-DD`
- `--json` 原 JSONL 透传

### 12.3 `tinyctx-stats` CLI（[stats.py](tinyctx/stats.py)）

默认报表 —— 路由汇总：
- total / by_route / by_reason(top 10) / est_input_tokens_routed / stream_bytes_returned / avg_stream_seconds / compaction_redirects / upstream_errors / stream_errors

`--quality`（本次新增）—— 6 维评分，输出 S/A/B/C/D/F：

| 维度 | 权重 | 含义 |
|---|---|---|
| routing efficiency | 25% | local share |
| compaction discipline | 20% | 1 - (codex_compaction + proactive_compact) share |
| token compression | 20% | 100 × (1 - avg(forwarded/est)) |
| read-delta savings | 15% | replacements / candidates |
| tool-trim savings | 10% | frontier 请求里 trim 应用比例 |
| reliability | 10% | 1 - error_rate |

无信号维度跳过并按比例重新分配权重，小流量会话不会被零分拖死。

### 12.4 上行细颗粒事件
proxy._log 还会写：`route` / `route_chat` / `mutation_gate` / `cap_fields` / `proactive_compact` / `read_delta` / `compactor_done` / `compactor_save_failed` / `historian_spawn_failed` / `frontier_trim_tools` / `stream_done` / `upstream_error` / `stream_error`。

---

## 13. 维护

### 13.1 Project Registry（[registry.py](tinyctx/registry.py)）

`~/.tinyctx/projects.json` 单文件 list；`scout init` / `keypin scan` 自动注册当前 repo；`register` / `unregister` / `is_registered` / `all_projects`。

### 13.2 Dreamer 周期任务（[dreamer.py](tinyctx/dreamer.py)）

CLI `tinyctx-dreamer run`：对每个注册项目跑：
1. `tinyctx-scout refresh`（仅当 graph.json 存在）
2. `tinyctx-keypin scan`
3. `tinyctx-mem ingest-compaction`（`--ingest-mem` 时）
4. GC（`--gc`）：删 `~/.tinyctx/cache/<repo>/sessions/<sid>/` 中 mtime 超 `--retention-days`(默认 30) 的目录

子进程隔离 + `subprocess.run` 300s timeout，单个失败不影响其他项目。

CLI 子命令：`run` / `list` / `register` / `unregister` / `install-launchd`（macOS 03:00 daily）/ `install-cron`（Linux）。

---

## 14. CLI 入口点（pyproject.toml `[project.scripts]`）

| 命令 | 模块入口 |
|---|---|
| `tinyctx-proxy` | `tinyctx.proxy:main` — 启动 FastAPI |
| `tinyctx-scout` | `tinyctx.scout:main` — 项目摘要 |
| `tinyctx-stats` | `tinyctx.stats:main` — 路由 + quality 报表 |
| `tinyctx-trace` | `tinyctx.trace:main` — 请求级追踪 |
| `tinyctx-recall` | `tinyctx.continuity:main` — 召回 compaction |
| `tinyctx-keypin` | `tinyctx.keypin:main` — 高频文件扫描 |
| `tinyctx-mem` | `tinyctx.memory:main` — mem0 包装 |
| `tinyctx-dreamer` | `tinyctx.dreamer:main` — 周期维护 |
| `tinyctx-interest` | `tinyctx.interest:main` — 压缩偏置 PageRank |
| `python -m tinyctx.advisor` | stdio MCP server |
| `python -m tinyctx.graphify_adapter` | graph.json 转换 |
| `python -m tinyctx.classifier` | 训练 / 预测 |

shell 脚本（[scripts/](scripts)）：
- `install.sh` — 幂等安装器（venv + graphify + serena + caveman vendor）
- `start.sh` / `start-bg.ps1` / `start.ps1` — 启动 proxy
- `tinyctx-up` — `start / stop / restart / status / logs` 包装
- `cm-hook-shim` — context-mode 静默 hook 适配
- `scout-session-start.sh` — codex SessionStart hook
- `smoke_codex.sh` — 真实 codex CLI smoke test

---

## 15. 配置（[config.py](tinyctx/config.py)）

层级合并：默认值 < TOML 文件 < `TINYCTX_*` 环境变量。

### 15.1 `BackendCfg` 字段（local / frontier 各一份）
- `base_url` / `api_key_env` / `model` / `wire_api` / `timeout_s` / `headers`
- `context_window` + `context_safe_fraction` — 路由动态阈值依据
- `supported_tool_types` — tool scrub 白名单（local 默认 `("function",)`，frontier 默认 `()` 透传）
- `strip_request_fields` — 字段剔除清单（codex 字段如 `client_metadata`）
- `inject_defaults` — dotted-path → 缺省值（`text.format.type` / `max_output_tokens`）
- `cap_fields` — dotted-path → 强制上限
- `translate_tool_calls` — 是否运行 tool_call_translator（local true / frontier false）

### 15.2 Routing 层
`escalate_input_tokens` / `escalate_turn_count` / `escalate_on_error_streak` / `redirect_compaction_to_local` / `compactor_debate` / `compactor_min_history_tokens` / `save_compactions` / `historian_*`(4 项) / `sanitize_encrypted_content`

### 15.3 Mutation 层
`dedup_tool_calls` / `purge_failed_tool_inputs` / `failed_input_after_turns` / `mutation_ttl_s` / `mutation_threshold` / `default_context_window` / `read_delta_*`(3 项)

### 15.4 Proactive compact 层
`proactive_compact_threshold` / `_safe_fraction` / `_recent_keep` / `_use_summarizer` / `_only_on_frontier`

### 15.5 Frontier 优化层
`frontier_trim_tools` / `frontier_tools_recent_window` / `frontier_tools_essentials` / `frontier_skip_advisor_hint`

### 15.6 Misc
`force_route` / `verbose` / `host` / `port` / `log_dir`

`effective_proactive_compact_threshold(cfg)` — 把 `frontier.context_window × safe_fraction` 算成实际阈值；后端换模型自动跟随。

---

## 16. Codex 集成（[.codex-plugin/](.codex-plugin)）

`plugin.json`（marketplace 清单）：
- `model_providers.tinyctx`：base_url=127.0.0.1:4141/v1，wire_api="responses"，retries 4/10
- `profiles.tinyctx`：使用上面 provider，model=`tinyctx-auto`，context_window=400000，auto_compact_token_limit=64000
- `profiles.tinyctx-goal`：长时间 `/goal` 专用 profile，启用 `features.goals`，context_window=1050000，auto_compact_token_limit=997500，`approval_policy=never`，`sandbox_mode=danger-full-access`
- 兼容 codex >= 0.124.0
- 加载 `hooks/hooks.json`

`scripts/cm-hook-shim`：针对 context-mode 在 codex 0.125 hook 协议下静默退出导致 `hook: ... Failed` 的修复 —— 把 context-mode 后台化、stdout 输出 `{}` 让 codex 视为 "no additional context" 通过。

---

## 17. 测试覆盖（截至本次提交）

23 个测试文件，285 用例，全部通过：

| 文件 | 数量 | 覆盖 |
|---|---|---|
| test_advisor.py | 21 | MCP server / call_advisor / SSE 解析 |
| test_classifier.py | – | 特征抽取 + SGD |
| test_compactor.py | – | 3-role debate / pristine guard / fallback |
| test_continuity.py | – | save / recall / list_sessions |
| test_dreamer.py | – | run / register / GC |
| test_dynamic_thresholds.py | – | proactive_compact 动态阈值 |
| test_frontier_optimizations.py | – | trim_tools / skip_advisor_hint |
| test_graphify_adapter.py | – | 三种 graph 形态 |
| test_historian.py | – | update + apply + 幂等 |
| test_interest.py | – | T0/I0/J0/PageRank |
| test_keypin.py | – | rollout 扫描 + filter_to_project |
| test_memory.py | – | mem0 wrapper（mock） |
| test_proactive_compact.py | – | bucket cache + incremental seeding |
| test_proxy_compactor_integration.py | – | proxy ↔ compactor 全链路 |
| test_proxy_integration.py | – | 路由 / sanitize / forward |
| test_read_delta.py | 26 | classify / collapse / 6 类边界 |
| test_registry.py | – | register / unregister |
| test_router.py | – | decide 全分支 |
| test_sanitize_dedup.py | 50+ | dedup / purge / cache_aware / advisor_hint / mcp_namespace |
| test_scout.py | – | gather / build / staleness |
| test_stats.py | 16 | 默认报表 + quality 评分 |
| test_stream_terminator.py | – | response.completed 终结帧 |
| test_tool_call_translator.py | – | XML / chat→responses / auto-answer |
| test_trace.py | – | RequestTrace.emit / 过滤 / watch |

---

## 18. 数据 / 缓存目录布局

```
~/.tinyctx/
├── config.toml                   主配置
├── secrets.env                   API key (chmod 600)
├── projects.json                 dreamer registry
├── classifier.json               学得模型（可选）
├── logs/
│   ├── tinyctx-YYYYMMDD.jsonl    每日日志（route / trace / stream_done / ...）
│   ├── dreamer.log
│   └── dreamer.err
├── cache/
│   └── <repo-hash>/
│       ├── manifest.json         scout 文件哈希清单
│       ├── scout.md              ≤ 2K tokens 项目摘要
│       ├── keyfiles.md           Read 频率 top-N
│       └── sessions/
│           ├── <session-id>/
│           │   ├── compaction-N.md       compaction 持久化
│           │   ├── compaction-N.json     structured 侧车
│           │   ├── historian-N.md        rolling digest
│           │   └── historian-N.json
│           └── latest.md         symlink 到最近的
└── mem0/                         mem0 vector store（可选）
```

---

## 19. 总特性一览（速查）

按"省 token / 防 bug / 增能力 / 可观测"四类：

### 省 token
- ✅ 路由：99% 走本地 27B
- ✅ Compaction handoff 改路本地（最高杠杆）
- ✅ proactive_compact 主动截断
- ✅ trim_tools_for_frontier 工具裁剪
- ✅ skip_advisor_hint on frontier
- ✅ dedup_tool_calls / purge_failed_tool_inputs
- ✅ historian rolling digest（可选）
- ✅ read_delta 重读压缩
- ✅ scout 一次性项目摘要 + cache
- ✅ keypin / interest 选择性引用
- ✅ cap_fields 防 max_output_tokens 超限
- ✅ bucket-key + incremental-seed summary cache

### 防 bug（codex ↔ 后端兼容）
- ✅ encrypted_content scrub
- ✅ MCP namespace 展开
- ✅ unsupported tool/field scrub
- ✅ inject_defaults（LMStudio text.format）
- ✅ orphan function_call 占位符
- ✅ input_image 拍平
- ✅ reasoning_content stub（DeepSeek）
- ✅ proactive_compact 孤儿修复
- ✅ SSE 终结帧（stream closed before response.completed 修复）
- ✅ chat ↔ responses 双向翻译
- ✅ XML tool-call → 结构化
- ✅ qwen / Hermes 模型本地兼容
- ✅ context-mode hook shim

### 增能力
- ✅ 3-role debate compactor + structured facts/compartments
- ✅ PRISTINE recomputation invariant
- ✅ Compaction 持久化 + tinyctx-recall
- ✅ Compression-biased PageRank（论文 §5.1）
- ✅ Two-layer scout
- ✅ keypin（行为频率）+ scout（结构）双信号
- ✅ Advisor Strategy MCP server
- ✅ request_user_input 三层自动拦截
- ✅ 学得分类器（FrugalGPT 风格）
- ✅ mem0 跨 session 记忆（可选）
- ✅ Dreamer 周期维护
- ✅ Project registry

### 可观测
- ✅ RequestTrace 30+ 字段 per-request
- ✅ tinyctx-trace 5 种视图（compact / verbose / watch / filter / json）
- ✅ tinyctx-stats 默认报表 + `--quality` S-F 评分
- ✅ JSONL 日志（按天滚动）
- ✅ Per-event mutation_gate / proactive_compact / read_delta / compactor_done

---

## 20. 不做什么

为防过度膨胀，明确不内置：

- ❌ 自己的代码图谱抽取（用 graphify / serena）
- ❌ 自己的 LSP（用 serena MCP）
- ❌ 自己的 sandbox 工具执行（用 mksglu/context-mode）
- ❌ 自己的 vector store（mem0 可选）
- ❌ HTML dashboard（tinyctx-stats CLI 够用）
- ❌ 自动恢复 compaction 到 SessionStart（避免冲突 codex resume；用 `tinyctx-recall` 显式拉）
- ❌ 自动注入 mem0 命中到 prompt（破坏 cache prefix）
- ❌ 自实现 LLMLingua / 语义压缩（保持 byte-stable）
- ❌ 跨 host 工具（仅 codex CLI / Codex.app 0.125+ / 0.128+）
- ❌ 二次 cache 自管（依赖 Anthropic prompt cache 的 5min TTL）

---

_本文档基于 commit `dc05668` 之后的代码状态生成；新增模块 / 新增字段请同步更新。_
