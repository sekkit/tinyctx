# Reasonix 技术评估（面向 tinyctx）

## 结论

`esengine/deepseek-reasonix` 里有若干值得吸收的技术方案，但不适合整套移植到 `tinyctx`。

原因很直接：

- `Reasonix` 是一个完整的 agent runtime / CLI / TUI
- `tinyctx` 是一个 HTTP 协议边界上的路由与兼容代理

因此我们应该借它的“局部机制”，而不是借它的“总体架构”。

推荐采用的方向：

1. 更细粒度的失败信号驱动升级
2. 更系统的 tool-call repair
3. 结果级 tool-result shrink
4. 成本与升级透明化

不推荐直接引入的方向：

1. Cache-First Loop 主架构
2. DeepSeek-native runtime 假设
3. TUI / session log / memory 主存储模型


## 一、可借鉴方案总览

### 1. 失败信号驱动升级

Reasonix 的价值点：

- 不只看 HTTP error
- 还看“模型挣扎”的结构化信号
- 在同一 turn 内完成升级

当前 tinyctx 已有相近能力：

- `retry_policy`：按 4xx / 5xx / connection error 决策
- `self_classify`：复杂任务时升级 frontier
- `soft_completion`：输出质量差时打 gate / force frontier
- `empty_response_guard`：空响应后强制下一轮 frontier

Reasonix 值得补强的点：

- 把“修复器触发”也记为升级信号
- 把“工具重复调用风暴”记为升级信号
- 把“参数结构修复次数”记为升级信号

### 2. Tool-call repair 四段式

Reasonix 的 repair pipeline：

1. `flatten`
2. `scavenge`
3. `truncation`
4. `storm`

对 tinyctx 的映射：

- `flatten`：适合加在工具 schema 暴露阶段
- `scavenge`：适合加在 `tool_call_translator.py`
- `truncation`：适合加在 stream / non-stream 响应修复层
- `storm`：适合加在 preflight / sanitize / retry 周边

优先级最高的是：

- `storm`
- `flatten`

因为它们对当前 tinyctx 的收益最大，且侵入性最低。

### 3. 结果级 auto-compaction

Reasonix 会在 turn 结束后，把过大的 tool result 压到一个 token cap 内。

tinyctx 已有相关能力：

- `historian`
- `proactive_compact`
- `read_delta`
- `trim_tools_for_frontier`

但当前更多是：

- 按整段 history 压缩
- 按 repeated read 压缩
- 按 frontier tool set 缩减

Reasonix 值得借的是：

- **针对单个大 tool result 的结果级 shrink**

这适合落在：

- `sanitize.py`
- `post_stream.py`

作为“后续轮次引用时只保留摘要”的附加机制。

### 4. 成本透明化

Reasonix 把以下内容直接暴露出来：

- cache hit
- 当前是否升级到高价模型
- cost signal

tinyctx 已有：

- dashboard
- trace
- stats

但还可以补：

- 每 turn 升级原因标签
- 最近 N 轮 upgrade / retry / guard 分布
- 估算 token 节省来源
- 组件级 repair 命中次数


## 二、不适合直接采用的部分

### 1. Cache-First Loop 主架构

Reasonix 的核心架构是：

- Immutable Prefix
- Append-only Log
- Volatile Scratch

这个方案非常强，但它要求：

- agent runtime 自己控制 prompt 组装
- session log 自己可重放
- tools / messages / scratch 区严格由 runtime 接管

tinyctx 当前并不控制 Codex 内部 agent loop。

所以：

- 原则可以借
- 架构不能直接搬

### 2. DeepSeek-native 假设

Reasonix 从设计开始就是为 DeepSeek 成本结构和行为模式优化。

tinyctx 的目标是：

- local / frontier 双后端
- 兼容 LMStudio / vLLM / Ollama / DeepSeek / OpenAI Codex backend

因此不能把某个模型族的行为假设写死进主架构。

### 3. 自带 runtime / TUI / memory 主模型

Reasonix 的很多能力来自它自己是完整客户端。

tinyctx 更适合：

- 继续保持代理边界清晰
- 只吸收“协议层 / 请求层 / 可观测性层”有收益的机制


## 三、建议落地路线

### P1：失败信号升级增强

新增统一失败信号计数器：

- tool-call repair 命中
- 截断修复命中
- 重复 identical tool call 命中
- local schema reject 命中

达到阈值后：

- 当前 turn 直接升级 frontier
- trace 记录具体触发原因

建议落点：

- `retry_policy.py`
- `proxy.py`
- `post_stream.py`

### P2：tool-call storm 抑制

目标：

- 同一会话内，短窗口重复相同 `(tool, args)` 的调用被识别
- 不再无限重试
- 改为注入反思 / 升级 / blocker

建议落点：

- `sanitize.py`
- `tool_call_translator.py`
- `session_state.py`

### P3：复杂 schema flatten

目标：

- 当 tool schema 过深 / leaf param 过多时
- 对模型暴露扁平版本
- 调用前再还原嵌套结构

收益：

- 减少工具参数缺失和错误结构
- 降低 local backend tool-call repair 压力

### P4：结果级 tool-result shrink

目标：

- 对超大 tool result 单独摘要
- 后续轮次保留摘要而非全文
- 必要时允许 re-read

建议落点：

- `sanitize.py`
- `post_stream.py`

### P5：dashboard 成本透明化

新增展示：

- recent upgrade reasons
- repair hit counters
- result shrink counters
- estimated token savings by source


## 四、对 tinyctx 的推荐结论

推荐吸收：

1. failure-signal auto-escalation
2. tool-call storm suppression
3. schema flatten / re-nest
4. per-result shrink
5. 更强 observability

不推荐吸收：

1. Cache-First Loop 主体
2. runtime/TUI/session architecture
3. DeepSeek-only 设计假设


## 五、下一步建议

建议按这个顺序推进：

1. 先做 `tool-call storm suppression`
2. 再做 `failure-signal auto-escalation`
3. 再做 `schema flatten`
4. 最后做 `result-level shrink`

原因：

- 前两项最能直接改善“空转 / 反复尝试 / 低价值本地挣扎”
- 后两项更偏稳态优化

