# tinyctx 深入分析（2026-05-21）

> 范围：只读分析 `README.md`、`docs/features.md`、`docs/architecture.md`、`tinyctx/proxy.py`、`tinyctx/sanitize.py`。本轮未改业务代码，仅新增分析产物。

## 1. 总体架构判断

tinyctx 的核心定位是 Codex CLI 与不同上游模型之间的 wire-level 代理：Codex 仍按 Responses API 形态发请求，tinyctx 在 HTTP 边界完成路由、清洗、压缩、重试、SSE 翻译、可观测性记录与部分上下文治理。README 的 Architecture 章节把 `tinyctx-proxy` 放在 Codex CLI 与 Local cheap path / GPT-5.5 frontier path 中间，并列出 compaction interceptor、cascade router、encrypted_content scrub、cache discipline、read_delta、XML tool-call translator、Chat→Responses SSE bridge、Codex wire-compat、LLMLingua hook 等能力。`docs/features.md` 则把能力拆成 wire-level 代理、sanitize 管线、compaction、历史压缩、tool-call 翻译、frontier 专属 token 优化、observability 等模块。

关键设计取向是：默认多数请求走便宜本地后端，只有复杂任务、显式强制或失败升级时走 frontier；同时尽量在不破坏 Codex wire contract 的前提下兼容 chat-completions 型本地后端。`docs/architecture.md` 强调代理是 HTTP 协议边界，不是 agent runtime；per-conversation 状态集中在 `SessionState`，持久化状态在 `~/.tinyctx/cache/` 与 `~/.tinyctx/state/`。这解释了为什么 resume、compact、dashboard 与 retry 都围绕 request/session id 建立最小共享状态。

## 2. 路由（routing）

路由入口在 `tinyctx/proxy.py`。请求进入后，proxy 会提取用户强制模型、估算 token、判断是否 compaction，再调用路由决策。代码证据显示 `proxy.py` 会识别 `tinyctx-local`、`tinyctx-frontier` 这类客户端模型名，设置 `trace.forced_by_client_model = True`；否则使用 `decision.route` 与 `decision.reason` 写入 `trace.route` / `trace.route_reason`。`docs/features.md` 的 wire-level 章节说明 `router.decide` 支持 `force_route=local/frontier`、配置或环境变量强制、本地/前沿级联以及可选 classifier 二次判断。

路由不是单纯按模型名转发。它还会受 compaction 请求、error streak、用户显式 escalation、advisor/frontier hint、classifier 概率、token 规模等因素影响。`docs/features.md` 提到 `extract_features(body)` 接在 `router.decide` 上下游，`~/.tinyctx/classifier.json` 存在时作为第二意见，且只有概率达到阈值才升级 frontier。README 从产品层描述默认约 95% 走 local，约 5% escalate 到 GPT-5.5 frontier。

需要注意的是，路由决策发生在 sanitize 之后/附近但在真正转发之前。`tinyctx/proxy.py` 注释写明 “Sanitize before any model swap”，说明模型边界切换前会先去除或规整不兼容字段，避免将 frontier 专属或 Codex 内部字段直接送给本地 chat 后端。

## 3. local 与 frontier 差异

local path 的目标是低成本、大上下文、兼容 OpenAI-like chat/completions 后端。README 列出 DeepSeek-v4-flash、LMStudio、Ollama、vLLM、SGLang 等本地/便宜后端，并强调 Chat→Responses SSE bridge。`docs/features.md` 的 tool-call 翻译章节说明本地 chat-completions SSE 会被转换成 Responses API SSE，以便 Codex 仍看到 expected event shape。

frontier path 的目标是高质量、复杂任务、advisor 或显式升级。README 把 frontier 指向 `chatgpt.com/backend-api/codex` 或 `icodeeasy.cc/v1`，配置里有 frontier model、context window、timeout、api key env 等。`docs/features.md` 明确 token-budget 优化是 frontier 专属：`proactive_compact` 默认只在 frontier 路径打开，因为本地后端可能有 1M 上下文且完整 history 更有价值；frontier 则因为成本、context window 和后端限制更需要主动截断。

兼容性差异体现在 sanitize：本地 chat 后端通常不支持 Responses API 的部分字段、工具类型、encrypted_content、复杂 reasoning 字段或 Codex 特有 metadata；frontier 默认更接近 Codex Responses API，但仍有 chatgpt backend 对工具 schema、字段大小、auth 的特殊限制。因此 `tinyctx/sanitize.py` 同时承担字段剥离、默认值注入、工具修剪、chat normalization 与 frontier trimming。

## 4. compact 与 “Context automatically compacted” 后行为

tinyctx 有两类 compact：一类是 Codex 自己发出的 compaction handoff request；另一类是 tinyctx 在转发前做的 proactive compact。

对 Codex 原生 compact，`docs/features.md` 的 Compaction 处理章节描述 `compactor.compact_with_debate`：使用多角色 draft + judge 合并，失败时按层级降级，最终保证有 fallback 占位文本。`docs/features.md` 还说明 Responses-API wire shape 会构造 compact 响应，并通过 `continuity.py` 持久化 compaction summary。README 的 “Multi-subagent compaction” 解释当 session 耗尽、出现自动 compact 后，tinyctx 会尝试把 compact 摘要保存下来，供新 session 继续使用。

对 proactive compact，`tinyctx/sanitize.py` 是关键实现点，`docs/features.md` 的 §5.3 明确规则：当估算 token 超过 `effective_proactive_compact_threshold(cfg)` 且请求本身不是 Codex compact request 时，保留 head（system/developer 项）、插入一条 summary message、保留 tail（默认 recent_keep 8）。summary 可来自本地 summarizer，失败或关闭时使用确定性 placeholder。它还维护 `_PROACTIVE_SUMMARY_CACHE` bucket 缓存，以 `(sid, len(middle)//20)` 复用 summary，避免每轮重算和 summary 漂移。

“Context automatically compacted” 之后最重要的恢复路径是：compact summary 被保存到 per-repo cache，而不是只绑定原 session。`docs/architecture.md` / `docs/features.md` 描述 `continuity.py` 的布局为 `~/.tinyctx/cache/<repo-hash>/sessions/<session-id>/compaction-N.md` 和 `latest.md`。这意味着 `/clear` 或新会话后 session_id 变化，但同仓库仍可通过 latest summary 继续恢复。风险点是：如果 compact 发生在流式失败、retry 或 local 后端不兼容附近，必须确认 summary 是否真的被保存、dashboard 是否记录 `compactor_done` / `compactor_save_failed`。

## 5. resume 恢复路径

resume 相关能力不是单一功能，而是由 continuity、SessionStart scout、exec_resume 与 session state 组合形成。`docs/architecture.md` 模块表列出 `exec_resume.py` 是 side-process `codex exec resume` poker，用于 stuck sessions；同文档状态章节说明 per-conversation state 迁移到 `SessionState`。`docs/features.md` 的 compaction 持久化章节说明 compaction 摘要保存到 repo scoped cache，并通过 latest.md 暴露最近一次摘要。

恢复路径可以分成三层：第一层是 Codex 自身的 resume/rollout；第二层是 tinyctx 通过 `continuity.py` 保存 compact summary，让新 session 获取上一轮上下文；第三层是 hooks/scout 在 SessionStart 注入项目上下文。README 的 scout SessionStart hook 描述会把 `~/.tinyctx/cache/<repo>/scout.md` 作为 additionalContext 注入第一轮。风险是：这些恢复路径分别来自不同入口，若某一路失败，用户可能只看到“resume 后缺上下文”，但根因可能是 compact 未保存、latest 指向旧摘要、SessionStart hook 输出协议失败，或 exec_resume 仅 poke 了 stuck stream 而没有补充语义上下文。

## 6. retry：stream 与 non-stream 的差异

`docs/architecture.md` 把 retry 拆成 retry_policy 与 proxy forwarding loop：`retry_policy.classify_failure(...)` 根据 status、body、异常、route、attempt count 输出 `retry_same`、`retry_escalate` 或 `propagate`；`proxy.py` 的 `_forward` loop 和 stream relay 根据分类执行。non-stream 请求可以在收到完整响应或异常后重写 body 并重新发起；stream 请求则更复杂，因为上游可能已经开始输出，tinyctx 必须处理 producer/consumer、mid-stream silence、empty response、soft completion 和 cancel/retry。

`docs/architecture.md` 提到 stream path 使用 `relay_stream(producer=StreamProducer, consumer=StreamConsumer)`，并有 PostStreamAnalyzer、stream-rewrite synthesis、stall kill、retry_attempted 等 trace 事件。关键差异是：non-stream retry 的 body 重写发生在完整请求-响应边界，容易保证“还没把坏响应交给 Codex”；stream retry 可能需要取消卡住的 producer，再合成或重放 Responses SSE 事件，避免 Codex 端看到半截协议。文档还提到 “cancel-and-retry primary path unblocks the wedged stream”，以及 “stall → cancel → empty-response on retry → force frontier” 这类升级链路。

需要特别关注 body 重写差异：non-stream 可以直接用新的 body / route 再调用 `_forward`；stream 则必须考虑是否已经消费了原始 request body、是否保留 previous_response_id、是否要合成 empty output 或 tool call translation。调试时应分别造 4xx/5xx、连接断开、mid-stream silence、empty response 四类场景，不能用 non-stream 成功推断 stream 安全。

## 7. dashboard、recent/state 与可观测性

dashboard 是 tinyctx 的运行态观测面。`docs/architecture.md` 模块表列出 `dashboard.py` 提供 `/dashboard` SSE/JSON endpoints 和 vanilla HTML page；`tool_metrics.py` 从 `body.input` 挖 per-tool counters 供 dashboard 使用。`docs/features.md` observability 部分列出日志事件包括 `route`、`route_chat`、`mutation_gate`、`cap_fields`、`proactive_compact`、`read_delta`、`compactor_done`、`compactor_save_failed`、`historian_spawn_failed` 等，并聚合 total、by_route、by_reason、estimated input tokens、stream bytes、avg stream seconds、compaction redirects、upstream/stream errors。

recent/state 的价值在于把一次请求的生命周期与最终 route/retry/compact 结果串起来。`docs/architecture.md` 强调 `RequestTrace` 记录 request_id、session_id、started_at、route、route_reason、is_compaction、est_input_tokens、turn_count、error_streak、requested_model、forced_by_client_model 等。request_phase / task_state 则用于 active-state observability：用户看到卡住时，dashboard 应能区分是在 upstream streaming、retry waiting、compaction summarizing、historian spawning，还是已经 soft-completion。

目前风险是文档中的 orchestrator 面板仍有设计稿色彩，`docs/visual-config-plan.md` 和自动 Skill/MCP 编排设计提到“最近决策”“current task state”“proof-of-work”“blockers”，但需核对实际 `dashboard.py` 是否已经完整显示。对本次指定文件而言，能确认的是 proxy trace 与 docs 中定义了这些观测事件；不能仅凭文档断言 UI 已完整实现所有字段。

## 8. orchestrator、request_phase 与 task_state

orchestrator 目标是对用户任务进行分类并推荐 skill/MCP/route hint。`docs/features.md` / 设计文档描述 `task_orchestrator.py` 的 `plan_task(body, cfg, catalog) -> TaskPlan`，字段包括 `task_type`、`confidence`、`recommended_skills`、`recommended_mcp`、`dynamic_skill_needed`、`routing_hint`、`constraints`、`rationale`。当前会话开头的 tinyctx 注入也体现了这一点：任务被识别为 debug，推荐 cc-tdd、cc-work、context-mode，并带有 test-first、large-search 约束。

request_phase 与 task_state 不是同一个层级。request_phase 更像单次 HTTP 请求生命周期：received、sanitized、routed、forwarding、streaming、retrying、completed/failed。task_state 更像跨请求的工作项状态：planning、executing、blocked、verifying、done。二者对应关系应通过 session_id/request_id 连接，而不是一一相等。调试时应检查 dashboard recent 是否把 request_phase 事件归到正确 task_state，否则用户会看到“任务还在执行”但底层请求已经完成，或相反。

## 9. developer/system 指令在 input 与 instructions 的差异

Responses API 通常区分顶层 `instructions` 与 `input` 消息列表。`tinyctx/sanitize.py` 的 `normalize_for_chat` 是兼容本地 chat 后端的关键：它需要把 Responses 形态转换为 chat messages，又不能丢失 system/developer 指令。`docs/features.md` 的 sanitize 章节列出 `normalize_for_chat`，说明会把 Responses body 规整为 chat-completions 可接受的结构；同章节还列出 `inject_responses_defaults`、`strip_unsupported_responses_fields`、`scrub_unsupported_tools` 等。

这里的高风险点是：developer/system 指令如果原本在 `instructions`，而 local chat backend 只读 `messages`，就必须在 normalize 时注入到消息前缀；但如果同时 input 中已有 system/developer，又可能重复或改变优先级。frontier 侧更接近 Responses API，保留 instructions 可能是正确的；local 侧则必须合并。调试 local responses 兼容性时，应分别构造：只有 instructions、只有 input system/developer、二者都有、以及 compact 后 head 保留 system/developer 的场景，确认模型实际看到的指令顺序。

## 10. local responses 后端兼容性

README 强调 tinyctx 能接 DeepSeek、LMStudio、Ollama、vLLM、SGLang 等，但这些后端大多是 OpenAI Chat Completions 或类 OpenAI 接口，不天然支持 Codex Responses API。兼容层由 `tinyctx/sanitize.py`、`tool_call_translator.py`、stream relay、field capping 共同完成。`docs/features.md` 提到 `strip_unsupported_responses_fields`、`inject_responses_defaults`、`cap_responses_fields`、`normalize_for_chat`，以及 XML → structured `function_call`、chat-completions SSE → Responses API SSE。

这意味着 local path 的“成功”有两个层次：HTTP 层能返回 200，不代表 Codex 端能消费；必须检查 SSE event shape、tool call id、function_call_output 配对、previous_response_id、output_text / output_item 顺序等。`proactive_compact` 的孤儿修复也说明 Responses 协议对 function_call 与 output 配对很敏感：tail 截断后如果留下 output 而 call 被截掉，会导致后端 400，因此会合成 `tinyctx_compacted_call` stub。

## 11. frontier 鉴权回退

frontier 配置支持 `api_key_env`，README/配置说明中提到默认 frontier 可以走 `chatgpt.com/backend-api/codex`，如果 `api_key_env` 为 None 则透传 Codex auth；也可切到 `icodeeasy.cc/v1` 或其他 paid API backend。`docs/architecture.md` 中配置片段显示 frontier 有 `api_key_env="TINYCTX_FRONTIER_API_KEY"`、model、wire_api、timeout、context_window 等字段，同时注释提到默认 chatgpt backend 对某些字段和工具类型有限制。

调试鉴权回退时要区分三件事：一是使用用户环境变量显式 API key；二是复用 Codex 原请求里的 bearer/access token；三是当 frontier auth 不可用时是否 downgrade 到 local、retry_escalate 到备用 frontier，还是直接 propagate。当前指定文件能支撑“存在 api_key_env / access_token / frontier fallback 设计”的结论，但具体 fallback 顺序需要进一步读 `config.py`、`router.py`、auth 相关逻辑和测试。

## 12. plugin 与 hook 对执行过程的影响

README 明确 tinyctx 安装/接入多个 MCP、skills 和 hooks：gitnexus、graphify、serena、caveman-shrink、context-mode、mem0、advisor、scout SessionStart hook。它们并不都经过 tinyctx HTTP 代理；文档说 Codex 会并行直接与 MCP servers / skills 通信。因此 plugin/hook 影响执行过程，但影响面不同：HTTP 请求会被 tinyctx 路由和清洗；工具调用、上下文注入、hook additionalContext 可能在 Codex 侧先发生。

README 的 Troubleshooting 章节说明 Codex 0.125 hook 协议要求 stdout 是单个合法 JSON 对象；旧 context-mode hook 若 stdout 为空会被 Codex 标记 `hook: ... Failed`，解决方案是 `cm-hook-shim` 后台执行 context-mode 并输出 `{}`。这解释了为什么 hook 失败可能不影响模型回答，但会影响上下文记录、dashboard/session DB 或 PreToolUse/PostToolUse 侧观测。

plugin 对执行过程的另一个影响是工具描述膨胀和上下文污染。README 提到 caveman-shrink 做 output/tool-description compression，context-mode 做 sandboxed tool execution，scout hook 在首轮注入项目上下文。这些都会改变 body.input 或 instructions 的体积，从而影响 route token estimate、proactive compact 阈值、cache-aware mutator 是否触发。

## 13. 主要风险与待确认点

1. compact 后恢复是否稳定：需要验证 `continuity.py` 是否在真实 Codex “Context automatically compacted” 流程中总能写入 latest.md，并被下一 session 读取。
2. stream retry 是否协议安全：特别是 cancel 后 retry、empty retry、retry_escalate 到 frontier 时，Responses SSE 是否完整闭合。
3. local chat 后端是否真正读到 developer/system：需要测试 `instructions` 与 `input` 合并顺序，不应只看 HTTP body 合法。
4. dashboard UI 与 trace 数据可能不一致：文档列出的 recent/task_state/orchestrator 面板不一定都已落地。
5. frontier auth fallback 细节不足：指定文件能看到配置与意图，但具体失败路径需继续读 `config.py`、`router.py`、auth 相关逻辑和测试。

## 14. 下一步调试计划

### A. compact / resume 专项

1. 构造一次 Codex compaction handoff 请求，确认 proxy 识别为 compaction 并走 local compactor。
2. 检查日志是否出现 `compactor_done`，失败时是否出现 fallback 占位。
3. 验证 `~/.tinyctx/cache/<repo-hash>/sessions/<session-id>/compaction-N.md` 与 `latest.md` 是否更新。
4. 新开 session / resume 后确认 latest summary 是否被注入或可被恢复路径读取。
5. 人工制造 compactor 失败，验证 fallback 与 dashboard 可观测性。

### B. local responses 兼容性专项

1. 针对只有 `instructions`、只有 `input system/developer`、二者都有三种 body，记录 normalize 后 chat messages。
2. 用 fake local SSE 返回普通文本、XML tool call、空响应、半截 tool call，检查 Codex Responses SSE 是否闭合。
3. 构造 function_call_output 孤儿场景，确认 `tinyctx_compacted_call` stub 防 400。
4. 对 DeepSeek/Ollama/vLLM/SGLang 分别跑 smoke，记录字段剥离和 tool schema 差异。

### C. retry 专项

1. non-stream：模拟 429、500、connection reset、空 body，确认 `retry_same` / `retry_escalate` / `propagate`。
2. stream：模拟 mid-stream silence、producer 卡死、半截 SSE、retry 后空响应。
3. 对比 retry 前后 body：route、model、previous_response_id、instructions、input 是否被意外改写。
4. 确认 dashboard recent 中能看到 `retry_attempted`、`stall_kill`、最终 route。

### D. dashboard / orchestrator 专项

1. 打开 `/dashboard` JSON/SSE endpoints，核对 recent 是否包含 request_id、session_id、route_reason、request_phase。
2. 检查 task orchestrator 决策是否进入 trace：task_type、skills、mcp、routing_hint、constraints。
3. 设计一个长任务，观察 task_state 是否随 request_phase 推进，避免 UI stale。
4. 对 hook 成功/失败各跑一次，确认 PreToolUse/PostToolUse/SessionStart 是否影响 dashboard 事件。

### E. frontier 鉴权专项

1. 无 `TINYCTX_FRONTIER_API_KEY` 时确认是否透传 Codex auth。
2. 设置无效 API key，确认是 fallback local、retry backup frontier，还是 propagate。
3. 切到 `icodeeasy.cc/v1` 或 OpenAI-compatible paid backend，确认 wire_api 与字段修剪匹配。
4. 记录 auth 失败在 dashboard/recent 中的可观测字段，补齐缺失日志。

