# 多 Agent 框架整合方案：tinyctx 作为底层省 token provider

调研来源：[yun193/Multi-Agent-Semi-Automated-Software-Engineer](https://github.com/yun193/Multi-Agent-Semi-Automated-Software-Engineer) 提到的三层 agent 编排栈，外加 [PraisonAI](https://github.com/MervinPraison/PraisonAI) 的对比。本文档回答两个问题：

1. 上游 agent 框架（EvoAgentX / CrewAI / SWE-Agent / PraisonAI / …）能不能跟 tinyctx 整合？
2. 整合的话，graphify / gitnexus 这两个代码知识图扮演什么角色？

结论先行：**值得做，但只做 Tier 1+2 的最小集合**。再扩就会变成上层框架的克隆，而那不是 tinyctx 该做的事。

---

## 1. 结构性定位

### 1.1 正确的拓扑：provider/consumer，不是 agent/agent 嵌套

yun193 那个项目把 EvoAgentX → CrewAI → SWE-Agent 画成**纵向 agent 调用栈**——上层 agent 调用下层 agent。这种拓扑在 tinyctx 视角下有 loop 嵌套灾难（谁重试、谁拥有 trajectory、谁的 cost limit 生效），三份独立调研报告都警告过。

tinyctx 视角下的正确拓扑是**provider/consumer 而不是 agent/agent**：

```
  consumer 层    ┌─ EvoAgentX  (TextGrad / 多维 evaluator)
                 ├─ codex      (终端交互式 coding agent)
                 ├─ CrewAI     (团队协作编排)
                 ├─ SWE-Agent  (沙盒补丁循环)
                 ├─ PraisonAI  (production-ready 多 agent)
                 └─ (任何走 LiteLLM 或 OpenAI 兼容协议的工具)
                              │
                              ▼  LiteLLM `base_url` / OpenAI 兼容
  provider 层         tinyctx-proxy
                  (route / compact / cache / scrub / MCP 编排)
                              │
                              ▼
  backend 层    DeepSeek-v4-flash · Qwen3.6-27B · vLLM · GPT-5.5
```

codex 跟 EvoAgentX/CrewAI/SWE-Agent/PraisonAI 是**兄弟关系**，都消费 tinyctx。上下层是 HTTP 协议关系，不是 agent 调用关系——因此没有 loop 嵌套问题，tinyctx 的 stateless 假设天然成立。

> **关键澄清**：codex 不能当"中间层"塞在 EvoAgentX 和 tinyctx 之间。codex 是 LLM **consumer**（终端 UI 应用），不是 LLM **provider**（程序化可调用的 `(prompt) → response` 协议）。它的输出是流式终端 UI + reasoning content + tool call，不是可被上层 agent 框架程序化调用的接口。

### 1.2 收敛事实

所有调研过的上游框架**全部通过 LiteLLM 出站**（PraisonAI、CrewAI、EvoAgentX 直接走 LiteLLM；SWE-Agent 也是），全部支持自定义 `base_url`。tinyctx 不需要 fork 任何一个，只要把它们的模型端点指向 `http://127.0.0.1:4141/v1`，每一个上游框架就自动继承：

- cascade 路由（默认本地 27B，必要时升级 GPT-5.5）
- compaction 拦截（handoff summary 走本地）
- `read_delta` / `CacheAwareMutator` / `encrypted_content` scrub
- LLMLingua-2 升级前压缩

### 1.3 tinyctx 作为底层 provider 的设计原则

这套拓扑要保持健康，下面这条边界必须守住：

> **tinyctx 不识别上层 agent 框架的语义。**
>
> 不要去判断"这是 EvoAgentX 的 gradient 调用"或"这是 CrewAI 的 manager 调用"。tinyctx 只看请求级别信号（token 估计、turn count、内容启发式）。上层框架自己通过 **model alias** 表达需求：
>
> - `model=tinyctx-frontier` → 强制升级前沿
> - `model=tinyctx-local` → 强制本地
> - `model=gpt-5.5` / `claude-opus-4` / 其它 → 走 cascade router 决策
>
> 这样上层框架可以独立演化，tinyctx 不需要为任何一个的 API 变化打补丁。这是 tinyctx 之所以是 ~10K LOC 而不是 100K LOC 的根本原因。

---

## 2. 四个框架的核心事实摘要

调研结论的浓缩版。详细架构分析见每个项目自己的 README 和源码。

### 2.1 EvoAgentX

- **本质**：自演进多 agent 框架。`WorkFlowGraph`（NetworkX DAG）+ `Agent` 基类 + `Evaluator`（benchmark 评分） + 多种优化器（TextGrad / Mipro / SEW / AFlow / MapElites）。
- **TextGrad 循环**：forward pass → benchmark 评分 → LLM 生成"梯度式批评" → prompt 重写 → 回滚机制。
- **典型成本**：一次 10 步优化 ≈ batch_size × 10 × (forward 3K + gradient 8K) ≈ 1.1M tokens。
- **模型抽象**：`OpenaiModel` / `LiteLLMModel` / `AliyunModel` / `OpenRouterModel` / `SiliconFlowModel`，全部走 LiteLLM 或兼容协议。
- **记忆**：`ShortTermMemory`（in-memory） + `LongTermMemory`（RAG-backed） + `MemoryManager`。

### 2.2 CrewAI

- **本质**：纯 Python 编排框架（零 LangChain 依赖）。三层概念：`Agent`（角色 + 工具 + 记忆）、`Task`（工作单元）、`Crew`（协调容器，支持 Sequential / Hierarchical）。2025+ 加入 `Flow`（事件驱动状态机）。
- **LLM**：单一抽象层 LiteLLM，每次调用走 `LiteLLM.completion()`。原生支持 `base_url`。
- **工具**：`@tool` 装饰器或 `BaseTool` 子类。原生支持 MCP（`mcps=[]` 字段 + `MCPServerAdapter`）。
- **典型成本**：8-agent crew × 5 task ≈ 8–40 次 LLM 调用 + 经理 agent 规划 + 记忆操作。**没有原生 prompt cache**（LiteLLM 不暴露该参数）。
- **2025 重构**：四套独立 memory 合并为统一 Cognitive Memory（语义 + 时效 + 重要度加权）。

### 2.3 SWE-Agent

- **本质**：完整自治 agent，专为 SWE-bench 设计。核心是 **ACI**（Agent-Computer Interface）+ Docker 沙盒（SWE-ReX 部署） + YAML 驱动的 step 循环。
- **主循环**（`sweagent/agent/agents.py:1265-1294`）：`while not done: step()`，每个 step = 一次 LLM 调用 + 工具执行。
- **ACI 工具**：`bash` / `str_replace_editor` / `view` / `find_file`，全部带超时和输出截断。`find_file` / `search_dir` 是 grep-based。
- **模型层**：`AbstractModel` 基类，`GenericAPIModel` 调用 `litellm.completion`。`InstanceStats` 追踪 `num_steps` / `num_api_calls` / `total_cost`。
- **典型成本**：SWE-bench 单任务 0.50–3.00 USD。简单 bug 3–5 步，复杂重构 20–50 步。
- **重要提示**：README 自承 `mini-swe-agent`（100 行简化版）在相同 benchmark 上表现相当。完整版被自家承认"过度工程化"。

### 2.4 PraisonAI

- **本质**：production-ready 多 agent 框架。核心是 `Agent`（mixins 组合：ChatMixin / ExecutionMixin / MemoryMixin / ToolExecutionMixin / SessionManagerMixin）+ `AgentTeam`（编排容器）+ 可选 `Workflow` DAG（`Task` / `Route` / `Parallel` / `Loop` / `Repeat`）。
- **LLM 层**：包了 LiteLLM 一层（`praisonaiagents/llm/llm.py`），加 context window 管理、token tracking、failover、custom tool schema。`LLM(model=, base_url=, api_key=)` 三参数直接通。
- **工具**：100+ 内置（web search、code exec、GitHub、email、scheduling、file ops、db…）。@tool 装饰器 + BaseTool 子类两条路子。
- **MCP**：原生支持 stdio / HTTP / WebSocket / SSE，是四个框架里最成熟的。
- **记忆**：file-based / ChromaDB / Mem0 / MongoDB 四后端可选，per-agent 记忆 + 可选 `MultiAgentLedger` 共享团队记忆。
- **典型成本**：5-agent × 10 task ≈ 50+ LLM 调用，跟 CrewAI 同量级。无原生 prompt cache（LiteLLM 限制）。
- **跟 CrewAI 的关系**：从 tinyctx 视角**功能等价**。差异都在框架内部（工具数量、MCP 成熟度、内存架构），跟集成无关。所以本文档 Tier 1/2 对它的处理跟 CrewAI 完全相同，不单独拉一节范例。

---

## 3. Tier 1：零代码、十倍降本（必做）

三个框架共享同一个 `base_url`：`http://127.0.0.1:4141/v1`。在 `examples/` 下加三份最小配置，**不动 tinyctx 内核**。

### 3.1 EvoAgentX

`examples/evoagentx-config.yaml`（草案）：

```yaml
executor:
  provider: litellm
  model: deepseek/deepseek-v4-flash
  base_url: http://127.0.0.1:4141/v1
  # tinyctx router 会把这些请求识别为 local-class，直走本地后端

gradient_generator:
  provider: litellm
  model: tinyctx-frontier      # 强制 alias，让 router 升级到 gpt-5.5
  base_url: http://127.0.0.1:4141/v1
  # TextGrad 的"梯度生成"是优化循环里推理最重的一步，
  # 用 frontier 跑收敛更快，整体反而更便宜
```

**为什么这样分**：forward pass 在所有 example 上重复跑，应当便宜；gradient generation 是单次高质量推理，值得 frontier。EvoAgentX 报告原文："Flip it: local executor, optional frontier gradients"。

### 3.2 CrewAI

`examples/crewai_simple_crew.py`（草案）：

```python
from crewai import Agent, Task, Crew, Process, LLM

shared_llm = LLM(
    model="gpt-5.5",                           # 任意 alias，tinyctx 决定真实后端
    base_url="http://127.0.0.1:4141/v1",
    api_key="sk-tinyctx-placeholder",          # tinyctx 不校验
)

architect = Agent(role="Architect",     goal="...", llm=shared_llm, tools=[...])
developer = Agent(role="Developer",     goal="...", llm=shared_llm, tools=[...])
reviewer  = Agent(role="Code Reviewer", goal="...", llm=shared_llm, tools=[...])

crew = Crew(
    agents=[architect, developer, reviewer],
    tasks=[...],
    process=Process.HIERARCHICAL,
)
result = crew.kickoff()
```

`crew.kickoff()` 期间所有 agent 调用都流经 tinyctx 代理，router 自动按 turn 类型决策走 local 还是 frontier。

### 3.3 SWE-Agent

`examples/swe-agent-config.yaml`（草案）：

```yaml
model:
  name: openai/gpt-5.5
  api_base: http://127.0.0.1:4141/v1
  api_key_env: OPENAI_API_KEY                  # 复用 codex 已配的 key
  temperature: 0.0
  per_instance_cost_limit: 3.0
  total_cost_limit: 50.0

tools:
  # 保留原 ACI，但 find_file / search_dir 会在 Tier 2 替换
  bundle: bash_only
```

**收益估计**：SWE-Agent 在 codex backend 上跑一次 SWE-bench 任务 0.5–3 USD，通过 tinyctx 走 95% 本地后大概 0.05–0.30 USD，降一个数量级。

### 3.4 PraisonAI

跟 CrewAI 同构，单独列例子是冗余的。文档里 README 一行带过即可：

> PraisonAI 使用方式跟 CrewAI 一致：`LLM(model=..., base_url="http://127.0.0.1:4141/v1")` 共享给所有 Agent。参考 `examples/crewai_simple_crew.py` 翻译即可。

如果有用户报需要原生范例再补 `examples/praisonai_simple_team.py`（~30 LOC），当前不预设。

### 3.5 改动清单（Tier 1）

| 路径 | 类型 | 行数 |
|---|---|---|
| `examples/evoagentx-config.yaml` | 新文件 | ~30 |
| `examples/crewai_simple_crew.py` | 新文件 | ~60 |
| `examples/swe-agent-config.yaml` | 新文件 | ~25 |
| `README.md` | 加一节 "Other agent frameworks"（含 PraisonAI 一行说明） | ~25 |

合计 ~140 行纯配置 + 文档。零核心代码改动。零风险。

---

## 4. Tier 2：graphify / gitnexus 作为三个框架的代码地基

Tier 1 解决"成本"，Tier 2 解决"能力"。三个框架的代码理解都很弱，这是 tinyctx 不做就没人做的事。

### 4.1 现状对比

| 框架 | 它当前的"代码搜索" | 接 graphify/gitnexus 后 |
|---|---|---|
| EvoAgentX Evaluator | 只看 benchmark 指标，无法识别幻觉符号 | 加一维"是否调用了不存在的符号"——评分从 5 维变 6 维 |
| CrewAI architect / reviewer | LLM 驱动的 websearch / file_read，~3 次 LLM 调用 ~0.05 USD | `@tool` 装饰的 `codebase_query` / `codebase_path`，~0.1s，~0 成本 |
| SWE-Agent ACI | `find_file` / `search_dir` 都是 grep，localization 慢且易迷路 | localization 阶段先查图、命中再 grep；典型 50 步轨迹省 5–10 步 |

### 4.2 设计：统一 adapter

新建 `tinyctx/code_graph_tool.py`（~150 LOC），对外暴露一个稳定签名：

```python
class CodeGraphTool:
    """Thin wrapper over graphify (preferred, MIT) or gitnexus (fallback)."""

    def __init__(self, repo_root: Path, backend: Literal["auto", "graphify", "gitnexus"] = "auto"):
        ...

    def query(self, question: str) -> list[dict]:
        """Free-form semantic query. Returns ranked symbol hits."""

    def path(self, from_symbol: str, to_symbol: str) -> list[str] | None:
        """Shortest dependency path between two symbols."""

    def reverse_callers(self, symbol: str) -> list[str]:
        """Who calls this? (Inverse of grep -r 'symbol(')."""

    def symbol_exists(self, name: str) -> bool:
        """Used by EvoAgentX evaluator: is this name actually defined?"""
```

`backend="auto"` 时优先 graphify（MIT，已经在 `install.sh` 装），退化到 gitnexus（PolyForm-NC）。两者都已经被 [tinyctx/graphify_adapter.py](../tinyctx/graphify_adapter.py) 和 [tinyctx/gitnexus_bootstrap.py](../tinyctx/gitnexus_bootstrap.py) 引导。

### 4.3 每个框架的桥接

**EvoAgentX**：在 `examples/evoagentx_symbol_grounded_evaluator.py` 写一个新 Evaluator：

```python
from tinyctx.code_graph_tool import CodeGraphTool

class SymbolGroundednessEvaluator:
    """6th dimension on top of completeness/correctness/quality/security/coverage."""

    def __init__(self, graph: CodeGraphTool):
        self.graph = graph

    def evaluate(self, prediction: str, label: str) -> dict:
        # Extract identifiers from generated code, check each against the graph
        names = _extract_identifiers(prediction)
        missing = [n for n in names if not self.graph.symbol_exists(n)]
        return {
            "symbol_groundedness": 1.0 - len(missing) / max(len(names), 1),
            "hallucinated_symbols": missing,
        }
```

接入 TextGrad 反馈循环：幻觉符号会直接作为下一轮 gradient 的输入，prompt 自动收紧。

**CrewAI**：`examples/crewai_tools.py`：

```python
from crewai import tool
from tinyctx.code_graph_tool import CodeGraphTool

_graph = CodeGraphTool(repo_root=Path.cwd())

@tool("codebase_query")
def codebase_query(question: str) -> str:
    """Query the codebase knowledge graph deterministically.

    Use this for:
    - "Which modules depend on auth.py?"
    - "Show the call chain from main() to process_user_input()."
    - "Find all functions returning AuthToken."
    """
    hits = _graph.query(question)
    return json.dumps(hits[:20])

@tool("codebase_reverse_callers")
def codebase_reverse_callers(symbol: str) -> str:
    """Who calls this symbol? Use before deleting or signature-changing."""
    return "\n".join(_graph.reverse_callers(symbol))
```

把这两个工具挂到 architect 和 reviewer agent 上即可。

**SWE-Agent**：不需要改 SWE-Agent 源码。在 `examples/swe-agent-config.yaml` 的 `tools` 段加一个 custom bundle，包一个 bash 脚本，脚本里调 `python -m tinyctx.code_graph_tool query "$@"`。SWE-Agent 的 ACI 把它当普通 bash 命令看待。

### 4.4 改动清单（Tier 2）

| 路径 | 类型 | 行数 |
|---|---|---|
| `tinyctx/code_graph_tool.py` | 新模块 | ~150 |
| `examples/crewai_tools.py` | 新文件 | ~40 |
| `examples/evoagentx_symbol_grounded_evaluator.py` | 新文件 | ~50 |
| `examples/swe-agent-tools/codegraph.sh` | 新脚本 | ~20 |
| `tests/test_code_graph_tool.py` | 新测试 | ~60 |

合计 ~320 LOC。中风险：依赖 graphify/gitnexus 已安装（`install.sh` 已经装，所以是 tinyctx 用户的默认状态）。

---

## 5. Tier 3：基础设施延展（按需触发，不预先做）

Tier 3 不是新功能，是**让 tinyctx 在多 agent 框架场景下更省心的小幅延展**。下面三条都是"等真实使用数据出现再决定"，不预先实现。

### 5.1 在 router 里给 frontier alias 加一组"高 token 大轮"启发

EvoAgentX TextGrad 循环 + CrewAI 8-agent crew 这种场景，单次请求 token 通常远超 codex 普通 turn。如果观察到这类请求频繁触发"误判走本地导致质量崩"，可以在 [tinyctx/router.py](../tinyctx/router.py) 加一条规则：当请求带显式 `tinyctx-frontier` alias 或 turn 总 token > N 时直升前沿。**触发条件**：trace 里出现明显误判尾巴。

### 5.2 在 trace 里打 consumer-framework 标签

如果同时有多个上游框架在用 tinyctx，trace 不知道哪条请求来自哪个框架，事后分析成本归属不便。可以让上游框架通过 `User-Agent` 或自定义 header 自报家门，tinyctx 在 [tinyctx/trace.py](../tinyctx/trace.py) 里多落一个 `consumer` 字段。**纯被动观察**，不影响路由决策。**触发条件**：用户报需要按框架拆分账单。

### 5.3 PraisonAI / EvoAgentX 自家的 prompt cache 缺失

四个框架里**没有一个**有原生 prompt cache（LiteLLM 不暴露该参数）。tinyctx 已经在做 `CacheAwareMutator` 保证 prefix 稳定；上游框架的多轮 agent 调用其实可以共享更长的 prefix。**触发条件**：实际接入后量到上游框架 cache hit 率 < 50% 且优化空间清晰。

### 不在 Tier 3 的方向

原文档曾把"EvoAgentX Evaluator 反哺 tinyctx classifier"列在这里。**已删除**。原因见 §6.4 和 §7：那是把上层 agent 框架塞进 tinyctx 内部的反模式，违反 §1.3 的设计原则（tinyctx 不识别上层语义）。

---

## 6. 明确拒绝的方向

三份独立调研报告全部指向了同一组反模式。集中列在此处，避免之后被诱惑。

### 6.1 把任何一个框架包成 MCP server 塞进 codex

**为什么诱人**：看起来 codex 多了一个"调用 crew"或"调用 swe-agent"的工具。

**为什么必须拒绝**：

- 双层 agent loop 灾难：谁负责重试？谁拥有 trajectory？谁的 cost limit 生效？
- 上下文碎片化：被包装的 agent 需要自己的系统提示和工具定义，从 codex 的对话历史里被切出来后，要么吞掉所有中间推理只返回结果（codex 失去可观测性），要么把中间步骤透回 codex（破坏 codex 的 SSE 流模型）。
- MCP 边界没有更便宜：等价效果用 Tier 1 的 `base_url` 已经免费拿到，MCP 包装只多一层 RPC 延迟和日志丢失。

### 6.2 `install.sh` 自动装这三个框架

**为什么诱人**：开箱即用。

**为什么必须拒绝**：

- tinyctx 的自我定位是 ~10K LOC 的薄代理。SWE-Agent 拖进来 `swerex` + Docker + 一堆 model adapter，install 表面积爆炸式增长。
- 用户场景差异巨大：很多 codex 用户根本不需要多 agent 框架。
- Tier 1 的范例已经足够低门槛：用户自己 `pip install crewai` / `pip install sweagent` / `pip install evoagentx`，配置一行 `base_url`，就拿到全部好处。

### 6.3 让 router 看穿框架内部的 agent 调度

**为什么诱人**：理论上"按 agent 角色路由"很性感（architect 走 frontier、formatter 走 local）。

**为什么必须拒绝**：

- 路由决策必须在 HTTP 请求边界做完。tinyctx 的 router 是 stateless 的，引入 agent 角色感知会破坏这个核心假设。
- 因果断裂：router 在 crew 启动前决策，crew 内部 agent 角色是动态分派的（特别是 HIERARCHICAL 模式），router 无法预测。
- 委托复杂度：如果 manager agent 把任务 A 派给 agent 1（local）、任务 B 派给 agent 2（frontier），agent 1 中途超预算，该怎么办？没有干净的语义。

让 crew 内部自己决策（通过共享 `base_url` + 局部 model 覆盖），router 只看请求级别的信号。

### 6.4 用 TextGrad / EvoAgentX 自动优化 tinyctx 自己的 prompt

这是一条**结构性拒绝**，理由七条，每一条独立成立：

1. **tinyctx 没有可优化的 prompt 表面**。codex 的系统提示在 codex 二进制里，tinyctx 看不见也改不动。tinyctx 自己写 prompt 的地方只有 compaction summary 一处——一处 prompt 不构成 TextGrad 的优化场域。

2. **没有可测的 oracle**。TextGrad 收敛需要 reward signal（MBPP 准确率、GSM8K pass rate 这种硬数）。compaction summary 的 reward 是模糊的（"下一轮用户没抱怨"、"prompt cache 命中率没掉"），没法当 gradient 的反向信号。EvoAgentX 在 MBPP/GSM8K 上能跑通是因为 benchmark 本身就是 oracle，tinyctx 这个领域**结构上没有 oracle**。

3. **compaction 是无状态的一次性变换**，输入空间是无穷多样的对话历史。TextGrad 在零历史单步任务上很快 plateau——self-improvement loop 在这类任务上的上限文献早有结论。

4. **成本不对称**。TextGrad 一次优化跑 10–50 个 LLM turn。tinyctx 全部存在意义就是**降 LLM turn 成本**。花 50× 的代价优化一个 prompt 只有在"训一次部署很久"才划算（EvoAgentX 的目标场景）。tinyctx 的 prompt 跟模型行为深度耦合——DeepSeek 升一个小版本，前一轮 TextGrad 的优化结果就废了。

5. **tinyctx 想做的事都有更便宜的现成方法**：

   | 需求 | TextGrad | tinyctx 现有方案 |
   |---|---|---|
   | 压缩-偏置的内容排名 | 不擅长 | [interest.py](../tinyctx/interest.py) 闭式公式（arxiv 2603.20396 §5.1）|
   | local vs frontier 决策 | 太重 | router 的小逻辑回归 + 启发式 |
   | 改 compaction prompt | 高成本无 oracle | trace.py 上跑 A/B，按 cache hit / token cost 挑赢家 |
   | 评估 prompt 改动是否更好 | 需要 benchmark | trace 回放真实历史 session |

6. **真正能让 tinyctx 学到东西的方法是 trace replay**。[tinyctx/trace.py](../tinyctx/trace.py) 已经落盘每一次 RequestTrace。改 compaction prompt 的正确流程是：改 → 选 100 条历史 trace 回放 → 量 token cost / cache hit / 用户后续手动修正次数 → 赢家上线。这是工业界 A/B，不是 TextGrad。

7. **违反 §1.3 设计原则**。把 EvoAgentX 拿进 tinyctx 内部做优化，意味着 tinyctx 开始识别"什么是 TextGrad 调用、什么是 forward pass"，破坏 stateless 假设和"不识别上层语义"的边界。

**澄清边界**：上面拒绝的是"把 TextGrad/EvoAgentX 拿进 tinyctx 内部"。**EvoAgentX 作为 tinyctx 的上游 consumer（Tier 1）仍然是强烈推荐的**——它的 TextGrad 循环跑在用户自己的 EvoAgentX 项目里，请求出站走 tinyctx，那是双赢。consumer 跟"内嵌进 tinyctx"是两件完全不同的事。

### 6.5 把 SWE-Agent 的 loop 移植进 codex

**为什么诱人**：SWE-Agent 在 SWE-bench 上表现好。

**为什么必须拒绝**：

- codex 自己有更成熟的交互循环，10 年以上的产品打磨：多轮调试、reasoning_content、session 状态、SSE 流式输出。SWE-Agent 是 batch agent，"运行一次解一个 task"。
- 两套 UX 哲学完全不兼容。挑一个，不要混。
- SWE-Agent 自家文档（README）承认 mini-swe-agent（100 行简化版）在 benchmark 上表现相当。"完整版"被自家认为过度工程化，没必要照搬。

### 6.6 把 CrewAI 当 codex 的替代品

**为什么诱人**：CrewAI 是个完整的 Python orchestration。

**为什么必须拒绝**：

- CrewAI 没有实时终端 I/O，是 batch。
- 没有 reasoning_content / encrypted_content 支持。
- 钩子只在 agent execute 边界，不能在 step 中间介入。
- 记忆模型是跨 session 的 SQL，与 codex 的 per-session compaction + interest ranking 假设冲突。

CrewAI 是**自治多 agent 协作的编排引擎**，不是**人在回路的交互式 coding agent**。

---

## 7. 落地路线图

| 版本 | 内容 | 改动量 |
|---|---|---|
| v0.x+1 | Tier 1：`examples/{evoagentx,crewai,swe-agent}-config.*` + README 一节（含 PraisonAI 类比说明） | ~140 LOC，纯文档/配置 |
| v0.x+2 | Tier 2：`tinyctx/code_graph_tool.py` + 三份桥接范例 + 测试 | ~320 LOC |
| v0.x+3 | 按需触发 Tier 3 三条延展（frontier 启发、consumer 标签、prompt cache 调优）| 视真实数据 |

---

## 8. 一句话总结

tinyctx 是**底层省 token provider**，上面坐着 codex / EvoAgentX / CrewAI / SWE-Agent / PraisonAI 一排兄弟 consumer。上下层是 HTTP 协议关系，不是 agent 调用嵌套。整合到 Tier 1+2 为止，再扩就把上层语义吸进来了，那不是 tinyctx 该做的事。
