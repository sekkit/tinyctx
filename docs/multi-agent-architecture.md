# 多 Agent 框架整合架构图与开发指南

配套文档：[多 Agent 框架整合方案](multi-agent-framework-integration.md)（决策与原则）。本文档是"怎么画"和"怎么做"：

- §1–§4：四张架构图，覆盖拓扑 / 请求生命周期 / model alias 决策 / Tier 2 代码图整合
- §5：与现有 tinyctx 模块的对应表
- §6：Tier 1 开发指南（范例 + README 改动）
- §7：Tier 2 开发指南（`code_graph_tool.py` + 三份桥接）
- §8：测试与验收清单
- §9：版本里程碑

---

## 1. 架构图一：Provider / Consumer 拓扑

这是文档全篇的中心图。一切其它细节都从这里推导。

```
═══════════════════════════════════════════════════════════════════════
                    consumer layer (siblings, not nested)
═══════════════════════════════════════════════════════════════════════

   EvoAgentX        codex          CrewAI       SWE-Agent      PraisonAI
   (TextGrad/       (terminal      (Agent/Task  (ACI / Docker  (Agent/Team
    Evaluator)       coding         /Crew)       patch loop)    /Workflow)
                     agent)
       │              │              │              │              │
       └──────────────┴──────────────┴──────────────┴──────────────┘
                                     │
                            LiteLLM / OpenAI 兼容协议
                            base_url=http://127.0.0.1:4141/v1
                                     │
                                     ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                  tinyctx-proxy   (provider)                    │
   │              FastAPI, ~10K LOC, 28 modules                     │
   │                                                                 │
   │  HTTP endpoints:                                                │
   │    /v1/responses             主路径 (Responses API)            │
   │    /v1/chat/completions      兼容路径                          │
   │    /v1/models                三个 alias 模型                   │
   │    /                         简介页                            │
   │                                                                 │
   │  请求处理管线 (proxy.py):                                       │
   │    ├── _extract_alias    (model 字段 → 路由意图)                │
   │    ├── sanitize          (scrub foreign encrypted_content)     │
   │    ├── read_delta        (repeat Read → unified diff)          │
   │    ├── historian         (压缩老 turn)                         │
   │    ├── router.decide     (cascade: local vs frontier)          │
   │    ├── llmlingua-2       (frontier 升级前压缩, opt-in)          │
   │    ├── CacheAwareMutator (prefix 字节稳定，保 cache hit)        │
   │    ├── compactor         (handoff summary 拦截改本地)          │
   │    ├── tool_call_translator (qwen-pythonic ↔ structured)       │
   │    └── trace.RequestTrace (落盘到 ~/.tinyctx/cache/<repo>/)    │
   └───────────────────────────────────────────────────────────────┘
                            │                       │
                     ~95% local              ~5% frontier
                ┌───────────┴──────────┐    ┌──────┴────────────┐
                │ DeepSeek-v4-flash    │    │ GPT-5.5           │
                │ LMStudio + Qwen3.6   │    │ (chatgpt.com/     │
                │ Ollama / vLLM /      │    │  backend-api 或   │
                │ SGLang               │    │  icodeeasy.cc/v1) │
                └──────────────────────┘    └───────────────────┘

═══════════════════════════════════════════════════════════════════════
   并行通道：各 consumer 通过 MCP 协议直连工具（不经过 tinyctx-proxy）
═══════════════════════════════════════════════════════════════════════

   gitnexus  graphify  serena  caveman-shrink  context-mode  mem0  advisor
   (代码图)  (代码图)  (LSP)   (输出压缩)      (沙盒)        (记忆) (前沿咨询)

   这些 MCP server 由 ~/.tinyctx/install.sh 自动安装并注册到 codex；
   CrewAI / PraisonAI 用其原生 MCP 适配器接入；EvoAgentX / SWE-Agent
   通过 Tier 2 的 code_graph_tool.py 间接使用 graphify/gitnexus。
```

**关键观察**：

1. 上下层是 **HTTP 协议关系**，不是 agent 调用关系——所以没有 loop 嵌套问题。
2. **codex 是 consumer 不是中间层**。它的协议表面是终端 UI，没有"被上层 agent 调用"的接口。
3. **tinyctx 不识别上层 agent 语义**。它只看请求级别信号（token、turn、内容启发式）+ consumer 提供的 model alias。

---

## 2. 架构图二：单次请求生命周期

一个 consumer 发出请求到拿到响应的完整路径。

```
   consumer (例: EvoAgentX 的一次 forward pass)
         │
         │  POST http://127.0.0.1:4141/v1/responses
         │  {
         │    "model":   "tinyctx-frontier" | "gpt-5.5" | "tinyctx-local",
         │    "input":   [...],         // Responses API 输入项
         │    "tools":   [...],         // 可选工具 schema
         │    "stream":  true
         │  }
         ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                     tinyctx-proxy / proxy.py                   │
   │                                                                 │
   │  [1] _extract_alias(body.model)                                │
   │      → AliasIntent.FORCE_FRONTIER                              │
   │      → AliasIntent.FORCE_LOCAL                                 │
   │      → AliasIntent.DECIDE                                      │
   │                                                                 │
   │  [2] sanitize.scrub_reasoning(body)                            │
   │      丢掉来自其它模型的 encrypted_content（避免解密崩溃）        │
   │      参考: openai/codex#17541                                  │
   │                                                                 │
   │  [3] read_delta.fold_repeat_reads(body)                        │
   │      连续 Read 同文件 → 折叠成 unified diff                      │
   │                                                                 │
   │  [4] historian.compress_old_turns(body, budget)                │
   │      老 turn 文本走本地小模型压缩，新 turn 原样保留              │
   │                                                                 │
   │  [5] router.decide(est_tokens, turn_count, body, alias)        │
   │      ├─ alias == FORCE_FRONTIER  → Decision(route="frontier")  │
   │      ├─ alias == FORCE_LOCAL     → Decision(route="local")     │
   │      └─ alias == DECIDE          → cascade logic 决策          │
   │                                                                 │
   │  [6] if Decision.route == "frontier":                          │
   │        llmlingua_compress(body)   # opt-in, 升级前压缩          │
   │                                                                 │
   │  [7] CacheAwareMutator(body)                                   │
   │      保证 prefix 字节稳定，最大化 prompt cache 命中              │
   │                                                                 │
   │  [8] tool_call_translator(body)                                │
   │      qwen-pythonic XML ↔ structured tool call 互转              │
   │                                                                 │
   │  [9] _proxy_request(backend, body)  # 实际发出 HTTP            │
   │      ├─ local:    DeepSeek / LMStudio / Ollama / vLLM         │
   │      └─ frontier: GPT-5.5 (chatgpt.com 或 icodeeasy 后端)      │
   │                                                                 │
   │  [10] _stream_proxy(...)  # Chat → Responses SSE 桥接           │
   │       backend 返回 chat completions stream → 翻译成            │
   │       Responses API 的 stream 格式回吐给 consumer               │
   │                                                                 │
   │  [11] trace.RequestTrace.record(                               │
   │         decision, cost_usd, latency_ms,                        │
   │         cache_hit_rate, route, alias, ...)                     │
   │       落盘到 ~/.tinyctx/cache/<repo>/trace/<ts>.jsonl          │
   └───────────────────────────────────────────────────────────────┘
         │
         │  SSE stream (Responses API 格式)
         │  data: {"type":"response.output_text.delta","delta":"..."}
         │  data: {"type":"response.completed","usage":{...}}
         ▼
   consumer 收到响应，继续自己的 agent 循环
```

---

## 3. 架构图三：Model Alias 决策树

consumer 通过填 `model` 字段表达路由意图，**这是 consumer 和 tinyctx 之间的唯一通信通道**。

```
consumer 发出请求时的 body.model 字段
                  │
        ┌─────────┼─────────┬──────────────────────┐
        │         │         │                      │
   "tinyctx- "tinyctx-  "gpt-5.5"            其它任何值
   frontier"  local"    "claude-opus-4"     (例: "deepseek-chat")
        │         │     "tinyctx-cascade"          │
        │         │         │                      │
        ▼         ▼         ▼                      ▼
   强制升级    强制本地   cascade router      identity passthrough
   前沿        本地       决策                (按 model name 直接路由)
                          │                        │
                          ▼                        │
            ┌─────────────────────────────┐        │
            │  router.decide() 内部逻辑    │        │
            │                              │        │
            │  signal: 启发式              │        │
            │   - est_input_tokens         │        │
            │   - turn_count               │        │
            │   - 是否含 reasoning_content │        │
            │   - 是否含 image/audio       │        │
            │   - 关键词 (architecture,    │        │
            │     security, design...)     │        │
            │                              │        │
            │  signal: 学习分类器(可选)     │        │
            │   - 训练自 ~/.tinyctx/cache/ │        │
            │     <repo>/trace/*.jsonl     │        │
            │   - 小逻辑回归               │        │
            │                              │        │
            │  rule: 升级阈值              │        │
            │   - 启发式 score ≥ T_high    │        │
            │   - 或 alias 阶段已要求      │        │
            └──────────────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Decision(                Decision(
          route="local",            route="frontier",
          reason="..."              reason="..."
        )                        )
              │                         │
              ▼                         ▼
        本地 backend            前沿 backend
```

**consumer 该怎么填 model**：

| 上游场景 | 建议 model 值 | 原因 |
|---|---|---|
| EvoAgentX forward pass | `tinyctx-local` | 大量重复跑，必须便宜 |
| EvoAgentX gradient generator | `tinyctx-frontier` | 单次高质量推理，前沿划算 |
| CrewAI 普通 agent | `gpt-5.5`（让 router 决） | 工作量混合，router 判断 |
| CrewAI manager agent (HIERARCHICAL) | `tinyctx-frontier` | 规划决策质量敏感 |
| SWE-Agent localization 阶段 | `tinyctx-local` | 文件浏览，本地够 |
| SWE-Agent patch synthesis | `tinyctx-frontier` | 生成正确 patch 是关键 |
| PraisonAI / 一般 consumer | `gpt-5.5`（让 router 决） | 默认路径 |

---

## 4. 架构图四：Tier 2 代码知识图整合

四个 consumer 共享一个统一的 `code_graph_tool.py` adapter，下面接 graphify 或 gitnexus。

```
                consumer 层（各自的工具/插件机制）
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │  EvoAgentX                                                         │
   │    examples/evoagentx_symbol_grounded_evaluator.py                │
   │    class SymbolGroundednessEvaluator(BaseEvaluator):              │
   │      def evaluate(prediction, label):                             │
   │        names = _extract_identifiers(prediction)                   │
   │        missing = [n for n in names                                │
   │                   if not graph.symbol_exists(n)]                  │
   │        return {                                                    │
   │          "symbol_groundedness": 1 - len(missing)/len(names),      │
   │          "hallucinated_symbols": missing,                         │
   │        }                                                           │
   │                                                                    │
   │  CrewAI / PraisonAI                                                │
   │    examples/crewai_tools.py                                       │
   │    @tool("codebase_query")                                        │
   │    def codebase_query(q: str) -> str: ...                         │
   │    @tool("codebase_reverse_callers")                              │
   │    def codebase_reverse_callers(symbol: str) -> str: ...          │
   │                                                                    │
   │  SWE-Agent                                                         │
   │    examples/swe-agent-tools/codegraph.sh (ACI bundle 引用)        │
   │    #!/usr/bin/env bash                                             │
   │    exec python -m tinyctx.code_graph_tool "$@"                    │
   │                                                                    │
   └──────────────────────────────────────────────────────────────────┘
                              │
                              │  统一 Python API
                              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │            tinyctx/code_graph_tool.py    (~150 LOC, 新模块)       │
   │                                                                    │
   │   class CodeGraphTool:                                            │
   │     def __init__(self, repo_root: Path,                           │
   │                  backend: Literal["auto","graphify","gitnexus"]   │
   │                            = "auto"): ...                          │
   │                                                                    │
   │     def query(self, question: str) -> list[Hit]:                  │
   │       """语义查询，返回排序后的符号命中。"""                       │
   │                                                                    │
   │     def path(self, from_symbol: str, to_symbol: str)              │
   │              -> list[Edge] | None:                                │
   │       """两个符号间最短依赖路径。"""                               │
   │                                                                    │
   │     def reverse_callers(self, symbol: str) -> list[str]:          │
   │       """谁调用了 symbol？(grep -r 的反向)。"""                    │
   │                                                                    │
   │     def symbol_exists(self, name: str) -> bool:                   │
   │       """name 是否真实定义？给 EvoAgentX evaluator 用。"""        │
   │                                                                    │
   │     # CLI 入口（给 SWE-Agent bash 调用）：                        │
   │     # python -m tinyctx.code_graph_tool query "..."               │
   │     # python -m tinyctx.code_graph_tool reverse-callers <symbol>  │
   └──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │  backend = "auto"             │
              │  优先 graphify, 退化 gitnexus │
              ▼                               ▼
     ┌──────────────────┐            ┌──────────────────┐
     │ graphify (MIT)    │            │ gitnexus         │
     │                   │            │ (PolyForm-NC)    │
     │ codex skill 形态  │            │ MCP server 形态  │
     │ ~/.codex/skills/  │            │ npx gitnexus     │
     │ graphify/         │            │                  │
     │                   │            │                  │
     │ tree-sitter +     │            │ tree-sitter +    │
     │ NetworkX graph    │            │ Memgraph         │
     │                   │            │                  │
     │ 已被 install.sh   │            │ 已被 install.sh  │
     │ 自动装            │            │ 自动装            │
     └──────────────────┘            └──────────────────┘
```

---

## 5. 与现有 tinyctx 模块的对应表

新增和修改的代码与现有模块的关系，方便实现时定位上下文。

| 现有模块 | 角色 | 新增/修改交互 |
|---|---|---|
| [tinyctx/proxy.py](../tinyctx/proxy.py) | FastAPI 主入口 | 不修改，所有 consumer 共享同一管线 |
| [tinyctx/router.py](../tinyctx/router.py) | cascade 决策 | 不修改（Tier 3 才动） |
| [tinyctx/config.py](../tinyctx/config.py) | TOML 配置加载 | 可选新增 `[code_graph]` 段（backend 选择） |
| [tinyctx/graphify_adapter.py](../tinyctx/graphify_adapter.py) | graphify skill 引导 | `code_graph_tool.py` 的 graphify backend 调用此处接口 |
| [tinyctx/gitnexus_bootstrap.py](../tinyctx/gitnexus_bootstrap.py) | gitnexus MCP 装配 | `code_graph_tool.py` 的 gitnexus backend 复用其安装路径 |
| [tinyctx/mcp_registry.py](../tinyctx/mcp_registry.py) | MCP server 注册 | 不修改 |
| [tinyctx/trace.py](../tinyctx/trace.py) | RequestTrace 落盘 | Tier 3 可加 `consumer` 标签字段 |
| [tinyctx/interest.py](../tinyctx/interest.py) | 压缩-偏置内容排名 | 不修改 |
| [scripts/install.sh](../scripts/install.sh) | 安装脚本 | 不修改（graphify/gitnexus 已经被装） |
| `examples/` | 范例目录 | **集中新增** Tier 1 和 Tier 2 范例 |

---

## 6. Tier 1 开发指南：零代码 / 范例

零核心代码改动。三份范例 + README 一节。**总改动 ~140 LOC**。

### 6.1 `examples/evoagentx-config.yaml`

EvoAgentX 把 `executor` 和 `gradient_generator` 拆开走不同 alias，这是 Tier 1 最有杠杆的优化。

```yaml
# EvoAgentX 接入 tinyctx 的最小配置示例
#
# 用法：
#   pip install evoagentx
#   cp examples/evoagentx-config.yaml ~/.evoagentx/config.yaml
#   ./scripts/start.sh                         # 启动 tinyctx
#   python -m evoagentx.optimize my_task.yaml  # 走 tinyctx

# 执行器：跑 forward pass，在 benchmark 上重复跑几十次
# 必须便宜，强制本地
executor:
  provider: litellm
  model: tinyctx-local
  base_url: http://127.0.0.1:4141/v1
  api_key: sk-tinyctx-placeholder   # tinyctx 不校验，placeholder 即可
  temperature: 0.0

# 梯度生成器：textual gradient critique，单次推理质量敏感
# 强制前沿
gradient_generator:
  provider: litellm
  model: tinyctx-frontier
  base_url: http://127.0.0.1:4141/v1
  api_key: sk-tinyctx-placeholder
  temperature: 0.7

# 评估器：跑 benchmark 评分（如 MBPP / GSM8K），跟 executor 一致
evaluator:
  provider: litellm
  model: tinyctx-local
  base_url: http://127.0.0.1:4141/v1
  api_key: sk-tinyctx-placeholder
  temperature: 0.0

# 优化循环：先试 5 步 local-only，若收敛不到目标 score 再升 frontier gradient
textgrad:
  max_steps: 10
  eval_every_n_steps: 2
  rollback_on_regression: true
  escalate_gradient_to_frontier_after_step: 5
```

### 6.2 `examples/crewai_simple_crew.py`

```python
"""CrewAI 接入 tinyctx 的最小范例。

所有 agent 共享同一个 LLM 对象 → 全部请求走 tinyctx，
按 model alias 自动决定 local / frontier。
"""

from crewai import Agent, Crew, LLM, Process, Task

# 默认 cascade：让 tinyctx router 看 turn 内容自己决定
default_llm = LLM(
    model="gpt-5.5",
    base_url="http://127.0.0.1:4141/v1",
    api_key="sk-tinyctx-placeholder",
)

# 强制前沿：架构决策类 agent，质量敏感
frontier_llm = LLM(
    model="tinyctx-frontier",
    base_url="http://127.0.0.1:4141/v1",
    api_key="sk-tinyctx-placeholder",
)

architect = Agent(
    role="System Architect",
    goal="设计模块切分与接口契约",
    backstory="资深架构师，擅长找出隐藏的耦合",
    llm=frontier_llm,                # 架构决策走前沿
    tools=[],                        # Tier 2 时改成 [codebase_query, codebase_path]
)

developer = Agent(
    role="Implementation Engineer",
    goal="按规格写代码",
    backstory="工程师，关注边界条件和测试",
    llm=default_llm,                 # 由 router 决定
    tools=[],
)

reviewer = Agent(
    role="Code Reviewer",
    goal="找回归风险，验证调用图未破",
    backstory="代码审查员，重视影响面分析",
    llm=default_llm,                 # Tier 2 时改成 [codebase_reverse_callers]
    tools=[],
)

design_task = Task(
    description="为新功能 X 设计模块切分。",
    expected_output="模块清单 + 接口签名。",
    agent=architect,
)

implement_task = Task(
    description="按设计写代码。",
    expected_output="补丁文件。",
    agent=developer,
)

review_task = Task(
    description="审查补丁影响面。",
    expected_output="OK / 需要改的清单。",
    agent=reviewer,
)

crew = Crew(
    agents=[architect, developer, reviewer],
    tasks=[design_task, implement_task, review_task],
    process=Process.SEQUENTIAL,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
```

### 6.3 `examples/swe-agent-config.yaml`

```yaml
# SWE-Agent 接入 tinyctx 的最小配置示例
#
# 用法：
#   pip install sweagent
#   sweagent run --config examples/swe-agent-config.yaml \
#                --problem-statement.path my_issue.md

agent:
  model:
    name: openai/gpt-5.5             # alias，让 tinyctx router 决策
    api_base: http://127.0.0.1:4141/v1
    api_key_env: OPENAI_API_KEY      # 复用 codex 已配的 key
    temperature: 0.0
    per_instance_cost_limit: 3.00    # 单 task 上限，按 tinyctx 计算后的成本算
    total_cost_limit: 50.00          # 整轮跑批上限

  tools:
    bundles:
      - name: defaults
      # Tier 2 时加上：
      # - name: code_graph
      #   path: ./examples/swe-agent-tools

  templates:
    system_template: |
      You are a coding agent. Use bash to navigate, edit, and test.
      Prefer code_graph queries over grep when looking for callers.

  history_processors:
    - last_n_observations: 5         # 配合 tinyctx historian 减重复
```

### 6.4 `examples/praisonai_simple_team.py`（按需补，当前 README 一行说明即可）

PraisonAI 跟 CrewAI 同构，集成模式相同。**目前不预先写范例**——README 加一行：

> PraisonAI 使用方式跟 CrewAI 一致：`LLM(model=..., base_url="http://127.0.0.1:4141/v1")` 共享给所有 Agent。参考 `examples/crewai_simple_crew.py` 翻译即可。

### 6.5 README 改动

在 README.md 的"Quick start"和"What we build vs. what we wire"之间，加一节：

```markdown
## Other agent frameworks

tinyctx 也可以作为任何走 LiteLLM 的上游 agent 框架的底层 provider，
让上游框架自动继承 tinyctx 的路由 / 压缩 / 缓存优化。

| 框架 | 范例 | 备注 |
|---|---|---|
| EvoAgentX | `examples/evoagentx-config.yaml` | TextGrad 循环建议 executor 走 local、gradient 走 frontier |
| CrewAI    | `examples/crewai_simple_crew.py` | 架构决策 agent 用 `tinyctx-frontier`，其它用默认 cascade |
| SWE-Agent | `examples/swe-agent-config.yaml` | `api_base` 指向 tinyctx proxy 即可 |
| PraisonAI | (参考 CrewAI 范例) | LLM 类用法与 CrewAI 一致 |

详细决策与拓扑：[多 Agent 框架整合方案](docs/multi-agent-framework-integration.md)
开发指南：[架构图与开发文档](docs/multi-agent-architecture.md)
```

### 6.6 Tier 1 验收

- [ ] 三份范例文件存在
- [ ] README 加了对应小节并链接到 docs/
- [ ] 至少手工跑通其中一个（推荐 EvoAgentX，文档里说的 30–100× 省钱差距最大，最容易看出效果）
- [ ] `tests/test_examples_smoke.py` 加一条 smoke test：解析三份配置文件能正确读出 `base_url=127.0.0.1:4141`

---

## 7. Tier 2 开发指南：统一 Code Graph Adapter

### 7.1 `tinyctx/code_graph_tool.py` 设计

```python
"""统一的代码知识图 adapter。

为所有上游 agent 框架（EvoAgentX / CrewAI / SWE-Agent / PraisonAI）
提供一个稳定的 Python 接口，下面接 graphify（首选）或 gitnexus（回退）。

设计原则：
- 接口稳定：上游不需要知道用了哪个 backend
- 失败安全：图查询失败返回空列表 / False，不抛异常
- 零状态：单次调用，调用方自己缓存（图本身可能很大）
- 也提供 CLI 入口：python -m tinyctx.code_graph_tool query "..."
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Hit:
    symbol: str
    file: str
    line: int
    score: float


@dataclass(frozen=True)
class Edge:
    from_symbol: str
    to_symbol: str
    kind: str   # "call" | "import" | "inherit" | ...


Backend = Literal["auto", "graphify", "gitnexus"]


class CodeGraphTool:
    def __init__(self, repo_root: Path | str, backend: Backend = "auto") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.backend = self._pick_backend(backend)

    def query(self, question: str, *, limit: int = 20) -> list[Hit]:
        """自然语言或符号名查询，返回排序后的命中。"""
        return self._dispatch("query", question, limit=limit)

    def path(self, from_symbol: str, to_symbol: str) -> list[Edge] | None:
        """两个符号间最短依赖路径，无路径返回 None。"""
        return self._dispatch("path", from_symbol, to_symbol)

    def reverse_callers(self, symbol: str) -> list[str]:
        """谁调用 symbol？返回 file:line 列表。"""
        return self._dispatch("reverse_callers", symbol)

    def symbol_exists(self, name: str) -> bool:
        """name 是否真实定义？给 EvoAgentX evaluator 用。"""
        try:
            return self._dispatch("symbol_exists", name)
        except Exception:
            return False  # 失败时保守地认为存在（不误判幻觉）

    # ── 内部 ─────────────────────────────────────────────────

    def _pick_backend(self, requested: Backend) -> str:
        if requested != "auto":
            return requested
        # 优先 graphify（MIT），退化 gitnexus（PolyForm-NC）
        if self._graphify_available():
            return "graphify"
        if self._gitnexus_available():
            return "gitnexus"
        raise RuntimeError(
            "no code graph backend available; "
            "run scripts/install.sh to install graphify or gitnexus"
        )

    def _graphify_available(self) -> bool:
        # graphify 是 codex skill，检查 ~/.codex/skills/graphify/ 是否存在
        return (Path.home() / ".codex" / "skills" / "graphify").exists()

    def _gitnexus_available(self) -> bool:
        return subprocess.run(
            ["which", "gitnexus"], capture_output=True
        ).returncode == 0

    def _dispatch(self, op: str, *args, **kwargs):
        if self.backend == "graphify":
            return _call_graphify(self.repo_root, op, args, kwargs)
        if self.backend == "gitnexus":
            return _call_gitnexus(self.repo_root, op, args, kwargs)
        raise AssertionError(f"unknown backend: {self.backend}")


# ── backend 调用 ─────────────────────────────────────────────


def _call_graphify(repo: Path, op: str, args: tuple, kwargs: dict):
    """通过 subprocess 调 graphify CLI，按 op 翻译参数。"""
    cmd = ["python", "-m", "graphify", op, "--repo", str(repo), *map(str, args)]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return [] if op != "symbol_exists" else False
    return json.loads(result.stdout)


def _call_gitnexus(repo: Path, op: str, args: tuple, kwargs: dict):
    """通过 stdio MCP 调 gitnexus，按 op 翻译参数。"""
    # 实际实现走 MCP client (mcp.ClientSession)，此处略
    raise NotImplementedError("gitnexus backend pending")


# ── CLI 入口 ─────────────────────────────────────────────────


def _cli() -> None:
    """python -m tinyctx.code_graph_tool <op> [args...]"""
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m tinyctx.code_graph_tool <op> [args...]",
              file=sys.stderr)
        sys.exit(2)
    op, *rest = sys.argv[1:]
    tool = CodeGraphTool(repo_root=Path.cwd())
    method = getattr(tool, op.replace("-", "_"))
    print(json.dumps(method(*rest), default=str))


if __name__ == "__main__":
    _cli()
```

### 7.2 测试方案 `tests/test_code_graph_tool.py`

```python
"""code_graph_tool 的测试。

策略：
- 单元测试用 mock subprocess，验证参数翻译
- 集成测试在 tinyctx 自己的 repo 上跑（已经有 graphify 装好），
  验证 known symbol 存在、known caller 关系成立
- 失败安全：backend 不可用时所有接口返回安全默认值
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tinyctx.code_graph_tool import CodeGraphTool, Hit


def test_pick_backend_auto_prefers_graphify(tmp_path):
    with patch.object(CodeGraphTool, "_graphify_available", return_value=True):
        with patch.object(CodeGraphTool, "_gitnexus_available", return_value=True):
            tool = CodeGraphTool(tmp_path, backend="auto")
            assert tool.backend == "graphify"


def test_symbol_exists_false_safe_when_backend_fails(tmp_path):
    with patch.object(CodeGraphTool, "_graphify_available", return_value=True):
        tool = CodeGraphTool(tmp_path, backend="graphify")
        with patch("tinyctx.code_graph_tool._call_graphify",
                   side_effect=RuntimeError("boom")):
            # 失败时保守认为符号存在，避免误判幻觉
            assert tool.symbol_exists("anything") is False


@pytest.mark.integration
def test_query_known_symbol_in_tinyctx_repo():
    """在 tinyctx 自己的 repo 上跑，验证 'RouterDecision' 能被查到。"""
    repo = Path(__file__).parent.parent
    tool = CodeGraphTool(repo, backend="auto")
    hits = tool.query("RouterDecision")
    assert any("router.py" in h.file for h in hits)


@pytest.mark.integration
def test_reverse_callers_of_known_function():
    repo = Path(__file__).parent.parent
    tool = CodeGraphTool(repo, backend="auto")
    callers = tool.reverse_callers("router.decide")
    assert len(callers) > 0
    assert any("proxy.py" in c for c in callers)
```

### 7.3 三个 consumer 的桥接

#### `examples/crewai_tools.py`

```python
"""把 code_graph_tool 暴露成 CrewAI / PraisonAI 的 @tool。"""

import json
from pathlib import Path

from crewai import tool

from tinyctx.code_graph_tool import CodeGraphTool

_graph = CodeGraphTool(repo_root=Path.cwd())


@tool("codebase_query")
def codebase_query(question: str) -> str:
    """按自然语言或符号名查询代码库知识图。

    适用场景：
    - 找一个类的定义位置：codebase_query("class AuthSession")
    - 找返回特定类型的函数：codebase_query("functions returning Token")
    - 找跟某个模块相关的代码：codebase_query("token refresh logic")

    比 grep 准（基于 tree-sitter），比 LLM 搜索快（~100ms 对 LLM 调用）。
    """
    hits = _graph.query(question, limit=20)
    return json.dumps([h.__dict__ for h in hits])


@tool("codebase_reverse_callers")
def codebase_reverse_callers(symbol: str) -> str:
    """谁调用了这个符号？删除或改签名前必查。

    例：codebase_reverse_callers("validate_token")
    返回所有 file:line 调用点。
    """
    return "\n".join(_graph.reverse_callers(symbol))


@tool("codebase_path")
def codebase_path(from_symbol: str, to_symbol: str) -> str:
    """两个符号间的最短依赖路径。理解重构影响面用。"""
    path = _graph.path(from_symbol, to_symbol)
    if path is None:
        return f"no path from {from_symbol} to {to_symbol}"
    return json.dumps([e.__dict__ for e in path])
```

#### `examples/evoagentx_symbol_grounded_evaluator.py`

```python
"""EvoAgentX 的第 6 维评估器：符号是否落地。

加进 evaluator 列表后，TextGrad gradient generator 会在批评里看到
"prompt 让模型生成了不存在的函数 X"，下一轮 prompt 自然收紧。
"""

import re
from pathlib import Path

from evoagentx.evaluators import BaseEvaluator

from tinyctx.code_graph_tool import CodeGraphTool

# Python 标识符的简单提取（实际场景可换 AST 提取更准）
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BUILTIN = {"print", "len", "str", "int", "list", "dict", "True", "False", "None"}


class SymbolGroundednessEvaluator(BaseEvaluator):
    def __init__(self, repo_root: Path | str):
        self.graph = CodeGraphTool(repo_root)

    def evaluate(self, prediction: str, label: str) -> dict:
        names = {n for n in _IDENT.findall(prediction) if n not in _BUILTIN}
        if not names:
            return {"symbol_groundedness": 1.0, "hallucinated_symbols": []}
        missing = [n for n in names if not self.graph.symbol_exists(n)]
        return {
            "symbol_groundedness": 1.0 - len(missing) / len(names),
            "hallucinated_symbols": missing,
        }
```

#### `examples/swe-agent-tools/codegraph.sh`

```bash
#!/usr/bin/env bash
# SWE-Agent ACI 工具：调 tinyctx 的 code_graph_tool。
#
# 在 examples/swe-agent-config.yaml 的 tools.bundles 里引用：
#   - name: code_graph
#     path: examples/swe-agent-tools
#
# 用法（agent 在 bash 里调）：
#   codegraph query "validate_token"
#   codegraph reverse-callers "AuthSession"
#   codegraph path "main" "validate_token"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: codegraph <query|reverse-callers|path|symbol-exists> [args...]" >&2
  exit 2
fi

exec python -m tinyctx.code_graph_tool "$@"
```

### 7.4 Tier 2 验收

- [ ] `tinyctx/code_graph_tool.py` 单元测试通过（mock backend）
- [ ] 集成测试在本仓库跑通（auto backend → graphify，查 `RouterDecision` 能命中）
- [ ] `python -m tinyctx.code_graph_tool query "RouterDecision"` 命令行工作
- [ ] `examples/crewai_tools.py` 在示例 crew 里能正确被 agent 调用
- [ ] `examples/evoagentx_symbol_grounded_evaluator.py` 在一个 prompt-with-fake-symbol 测试上输出 `hallucinated_symbols: [...]`
- [ ] `examples/swe-agent-tools/codegraph.sh` 可执行且返回正确 JSON

---

## 8. 测试与回归

### 8.1 单元测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_code_graph_tool.py` | adapter 接口、backend 选择、失败安全 |
| `tests/test_examples_smoke.py` | 三份 Tier 1 配置 yaml/py 可解析 |
| `tests/test_evoagentx_evaluator.py` | symbol groundedness 评分正确性 |

### 8.2 集成测试

在 tinyctx 自己的 repo 上跑（图已存在）：

```bash
# 1. 启动 proxy
./scripts/start.sh &

# 2. 跑一个最小的 EvoAgentX optimize（不需要真的 LLM，只看请求出站）
PYTHONPATH=. python -c "
from examples.evoagentx_symbol_grounded_evaluator import SymbolGroundednessEvaluator
e = SymbolGroundednessEvaluator('.')
print(e.evaluate('use router.decide and nonexistent_fn_xyz', 'unused'))
# 期望：{'symbol_groundedness': 0.5, 'hallucinated_symbols': ['nonexistent_fn_xyz']}
"

# 3. 跑 CrewAI 范例（如果装了 crewai 包）
python examples/crewai_simple_crew.py
```

### 8.3 回归保护

- `tests/test_no_hardcoded_paths.py` 已经在防硬编码路径，新模块也需通过
- `tinyctx/trace.py` 在所有 consumer 请求落盘，Tier 1+2 不应改变 trace 字段
- 新增的 `code_graph_tool.py` **不能**被 import 进 `proxy.py`（保持 proxy 跟 graph 解耦）

---

## 9. 版本里程碑

| 版本 | 范围 | 改动量 | 估时 |
|---|---|---|---|
| v0.x+1 | Tier 1 三份范例 + README + smoke 测试 | ~140 LOC | 0.5–1 天 |
| v0.x+2 | `code_graph_tool.py` + 单元测试 + 集成测试 | ~250 LOC | 1.5–2 天 |
| v0.x+3 | 三份桥接范例 + 文档更新 | ~150 LOC | 1 天 |
| v0.x+4 | 按需触发 Tier 3 子项（参考决策文档 §5）| 视真实数据 | — |

---

## 10. 不在本文档范围

- **怎么改 router**（cascade 决策逻辑微调）→ 见 [features.md](features.md) §router
- **怎么改 sanitize / read_delta / compactor** → 见 [features.md](features.md) 对应章节
- **整合方案的"为什么"和拒绝理由** → 见 [multi-agent-framework-integration.md](multi-agent-framework-integration.md)
- **graphify / gitnexus 自身的内部架构** → 见各自上游 repo

本文档只回答"怎么画 + 怎么做"。
