# antoinezambelli/forge 架构与 tinyctx 适配判断

分析对象：`C:\Dev\_external\forge`，commit `655e1f6`，包版本 `forge-guardrails 0.7.0`。

## 结论摘要

`forge` 不应该被 tinyctx 整体引入。两者目标相邻但边界不同：

- `forge` 是自托管 LLM tool-calling 的可靠性层，重点是让小模型更稳定地产生合法工具调用。
- `tinyctx` 是 Codex CLI Responses API 的上下文路由代理，重点是成本、上下文窗口、Codex wire compatibility、压缩和模型边界切换。

tinyctx 值得吸收的是 `forge` 的工程化 guardrail 形态、评测体系和少量后端运行时治理；不建议吸收它的 WorkflowRunner、Chat Completions proxy、raw asyncio HTTP server 或完整 workflow DSL。

## forge 的技术架构

`forge` 是 Python 3.12+ 包，核心依赖很少：`pydantic>=2.0` 和 `httpx>=0.27`，可选 Anthropic SDK。项目约 154 个 git 文件，其中 Python 源码集中在 `src/forge/`，测试数量约 886 个。

主要模块分层：

- `forge.core`
  - `workflow.py`：`ToolSpec`、`ToolDef`、`ToolCall`、`TextResponse`、`Workflow`，用 Pydantic 动态模型校验工具参数。
  - `runner.py`：`WorkflowRunner`，拥有 agentic loop。
  - `inference.py`：统一处理 compaction、message folding、response validation、retry。
  - `steps.py`：required step、prerequisite、terminal tool 的状态跟踪。
  - `slot_worker.py`：单推理槽的优先级队列和抢占/取消。
- `forge.guardrails`
  - `response_validator.py`：校验未知工具、畸形参数、裸文本响应。
  - `guardrails.py`：可嵌入外部 loop 的 middleware facade。
  - `step_enforcer.py`：强制 required steps 和 prerequisites。
  - `error_tracker.py`：连续重试和连续工具错误计数。
- `forge.prompts`
  - `templates.py`：把工具 schema 注入 prompt，并从错误格式中 rescue 工具调用。
  - 支持 fenced JSON、Mistral `[TOOL_CALLS]...`、Qwen XML `<tool_call>...</tool_call>` 等格式恢复。
- `forge.context`
  - `manager.py`：上下文预算、阈值回调、触发压缩。
  - `strategies.py`：`NoCompact`、`SlidingWindowCompact`、`TieredCompact`。
- `forge.clients`
  - Ollama、llamafile、Anthropic adapters，另有 sampling defaults。
- `forge.proxy`
  - OpenAI-compatible `/v1/chat/completions` proxy，单请求级别应用 validation、rescue parsing、retry、synthetic respond tool。
- `forge.server`
  - 可启动/管理本地后端，检测 context budget，按硬件/后端推导预算。

## forge 的核心功能

1. 工具调用恢复与校验

   小模型经常把工具调用吐成 JSON code fence、XML 或模型家族特定格式。`forge` 会恢复成 OpenAI `tool_calls`，再校验工具名和参数形状。

2. Retry nudge

   当模型输出裸文本、未知工具或畸形参数时，`forge` 不直接失败，而是注入纠偏消息并重试，默认最多 3 次。

3. Synthetic `respond` tool

   当请求带工具但模型其实应该自然语言回复时，`forge` 注入一个 `respond(message=...)` 工具，让模型始终保持“选择一个工具”的结构化决策。

4. Workflow step enforcement

   `WorkflowRunner` 可定义 required steps、tool prerequisites 和 terminal tool，避免模型提前结束或跳过步骤。

5. Context compaction

   `TieredCompact` 按消息类型优先级裁剪：先删 nudge / prerequisite / retry，再处理旧 tool result，再处理 text / reasoning，并保留系统提示、原始用户输入和关键 tool call。

6. 后端与单槽治理

   支持 Ollama、llama-server、Llamafile、Anthropic；能探测上下文长度和 VRAM，另有 `SlotWorker` 管理共享 GPU 推理槽。

7. 评测体系

   README 声称有 26-scenario v0.7.0 eval suite；仓库内有大量 domain-style 单测，覆盖参数转换、数据缺口恢复、grounded synthesis、proxy、client、runner、context、guardrails。

## tinyctx 当前对应能力

tinyctx 的定位更贴近 Codex 专用反向代理：

- Responses API 代理和 SSE 兼容。
- compaction interceptor：把 Codex handoff summary 转本地模型。
- cascade router：本地便宜路径为默认，必要时升级 frontier。
- `encrypted_content` scrub：跨模型边界清除不可复用 reasoning blob。
- CacheAwareMutator：保护 prompt-cache prefix 稳定性。
- `read_delta`：重复 Read 转 diff。
- `tool_call_translator`：将 Qwen/pythonic/XML 风格工具调用转换为 Responses API 结构化事件。
- Chat-to-Responses SSE bridge。
- soft completion、synthetic continue、advisor、stall watchdog、retry policy、forensics、trace/stats。
- graphify/gitnexus/serena/mem0/LLMLingua 等 compose-first 外部能力。

所以 tinyctx 已覆盖一部分 `forge` 的“工具调用格式修复”和“上下文治理”，但这些能力分散在 stream/proxy/translator/retry/sanitize 模块中，还没有像 `forge.guardrails` 一样形成统一中间件层。

## 是否需要吸收

### 建议吸收：高优先级

1. Guardrails 统一抽象

   tinyctx 可以借鉴 `forge.guardrails`，把“解析、校验、纠偏、重试、记录错误”统一成一个明确 pipeline，而不是继续分散在 `tool_call_translator.py`、`stream_relay.py`、`sanitize.py`、`retry_policy.py`。

   建议形态：

   - `GuardrailCheck` / `GuardrailDecision` 数据结构。
   - validator 只描述问题，不直接改流。
   - repairer 负责从 XML / fenced JSON / malformed chunks 恢复 Responses function_call。
   - retry/nudge policy 单独决策。
   - trace 记录每次修复原因和置信度。

2. 评测套件思路

   `forge` 最大可借鉴点不是具体代码，而是 eval 文化。tinyctx 应建立自己的 regression scenarios：

   - Codex wire-compat 版本变化。
   - Qwen XML 工具调用恢复。
   - unknown tool/drop/rewrite。
   - compaction handoff 本地化质量。
   - encrypted_content 跨模型 scrub。
   - prompt-cache prefix 稳定性。
   - stuck loop / synthetic continue / soft completion。

3. 错误计数与 retry budget

   tinyctx 已有 retry/stall/watchdog，但可吸收 `ErrorTracker` 这种小而清晰的状态机，用于避免单 session 在同一类修复上无限循环。

### 建议吸收：中优先级

4. 后端上下文预算探测

   `forge.server` 的 budget detection / hardware detection 对 tinyctx 有帮助，但应做成配置建议和启动前检查，不应该让 tinyctx 默认管理用户的 Ollama/LMStudio/vLLM 进程。

5. 单槽/优先级队列

   对本地 GPU 后端，`SlotWorker` 的思路可用于 tinyctx 内部任务队列：compaction、scout、advisor、dreamer 优先级应不同。建议只对本地辅助调用生效，不要默认串行化 Codex 主请求。

6. 消息类型优先级

   `TieredCompact` 的“消息类型元数据 + 裁剪优先级”值得借鉴，用来整理 tinyctx 的 compaction/read_delta/LLMLingua 前置压缩逻辑。不过 tinyctx 不应直接采用 Forge 的裁剪策略，因为 Codex Responses history 和 tool result 语义更脆。

### 不建议吸收

1. WorkflowRunner / workflow DSL

   tinyctx 不应该拥有 Codex 的 agent loop。Codex 已经是 orchestrator，tinyctx 插手 required steps / terminal tool 会制造双重控制流。

2. OpenAI Chat Completions proxy

   tinyctx 的核心兼容面是 Responses API。已有 Chat-to-Responses bridge 即可；完整 `/v1/chat/completions` proxy 会扩大维护面。

3. Synthetic `respond` tool

   这对小模型 tool-calling 很有效，但对 Codex Responses API 风险较高：会改变 Codex 期望的事件序列和工具语义。tinyctx 已有 soft completion / synthetic continue，更适合 Codex 场景。

4. raw asyncio HTTP server

   tinyctx 已用 FastAPI/uvicorn。除非性能 profiling 证明框架开销是瓶颈，否则不值得替换。

5. Python 3.12+ / Pydantic 2 强迁移

   `forge` 可以要求 Python 3.12；tinyctx 当前是 Python 3.9+。为了一个 guardrail 抽象提高运行时门槛不划算。Pydantic 可局部用于 schema 校验，但不应成为核心迁移前提。

## 建议路线图

1. 新增 `tinyctx/guardrails.py`

   先只包裹现有 translator/sanitize/retry 行为，不改变外部行为。

2. 新增 `tests/scenarios/`

   用 JSON fixture 描述输入 Responses chunks、期望输出事件、trace 分类和是否触发重试。

3. 扩展 `trace.py`

   给每次 tool-call repair、unknown tool rewrite、synthetic continue、soft completion 增加结构化字段。

4. 增加本地后端 probe

   做成 `tinyctx doctor` 或 install-time check：报告模型上下文、是否单槽、建议的 compaction threshold，而不是自动接管进程。

5. 内部任务队列实验

   只覆盖 compaction/scout/dreamer/advisor，避免影响 Codex 主交互延迟。

## 最终判断

tinyctx 需要 `forge` 的一部分“可靠性工程思想”，不需要 `forge` 的产品形态。

最值得拿的是：

- guardrail pipeline；
- eval/ablation 报告体系；
- retry/error tracker；
- 本地后端 budget probe；
- 可选的内部单槽优先级队列。

最不该拿的是：

- WorkflowRunner；
- Chat Completions proxy；
- synthetic `respond` tool；
- raw asyncio server；
- Python 3.12/Pydantic 全面迁移。

## 专家讨论后的落地方案

多专家讨论后的共识更收敛：

- 借 forge 的 guardrail 脑子，不借 forge 的 Chat Completions wire shape。
- tinyctx 必须继续拥有 Responses API / SSE event 序列化主权。
- `function_call`、`response.output_item.added`、`response.function_call_arguments.delta/done`、`response.completed` 的事件顺序不能被 `tool_calls` 语义污染。
- `encrypted_content` scrub、prompt-cache prefix 稳定性、Chat-to-Responses bridge 仍是 tinyctx 自研边界。

推荐 pipeline：

```text
sanitize request
  -> upstream
  -> parse / normalize
  -> validate
  -> repair
  -> retry / escalate
  -> ErrorTracker
  -> trace / eval
```

第一阶段已经按这个方向落地为“小薄层”，没有新建大框架：

- `tinyctx/guards.py`
  - 新增 `FailureSignal`。
  - 新增 `GuardrailDecision`。
  - 新增 `GuardrailErrorTracker`。
  - 新增 `decision_from_failure_scan()`，把现有 `sanitize.collect_failure_signals()` 输出转成协议无关决策。
  - 新增 `trace_guard_results()`，把现有 preflight guard pipeline 结果转成 trace-safe 结构。
- `tinyctx/trace.py`
  - 新增 `guard_results`、`guardrail_decisions`、`failure_signal_score`、`failure_signals`。
- `tinyctx/proxy.py`
  - 现有 failure-signal frontier escalation 改为通过 `GuardrailDecision` 记录。
  - preflight `GuardPipeline` 结果写入 trace。
- `tests/fixtures/responses/`
  - 新增 Responses fixture replay 起点：sync text、sync XML tool-call repair、structured function_call passthrough、malformed XML no-rewrite、multi tool-call order、unknown-tool correction、stream text、stream fragmented tool-call。
- `tests/test_responses_fixture_replay.py`
  - 新增 deterministic replay harness，包含 SSE event parser 和 ordered subsequence assertions，后续可扩展成 tinyctx 自己的 forge-style eval suite。

已验证：

```text
PYTHONPATH=. pytest tests/test_guards.py tests/test_responses_fixture_replay.py tests/test_tool_call_translator.py tests/test_functional_test_points_doc.py -q
92 passed

PYTHONPATH=. pytest tests/test_sanitize_dedup.py tests/test_router.py tests/test_proxy_retry.py::TestForwardRetryNonStream::test_failure_signal_storm_forces_frontier_route -q
96 passed
```

后续优先级：

1. 扩展 fixture matrix：multi tool-call、unknown tool、orphan output、duplicate call、empty response、mid-stream stall。
2. 把 `GuardrailErrorTracker` 接入 retry/stall/unknown-tool 的 per-session 限流。
3. 增加 ablation 报告：每个 guard 单独关掉后对 wire-valid rate、retry success rate、P95 latency、frontier escalation rate 的影响。
