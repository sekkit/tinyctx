---
slugid: measured-escalation--0529
stage: spec
date: 2026-05-29
needs_tech: true
source: user-brainstorm
---

# tinyctx measured escalation spec

## 一句话目标

tinyctx 保持 "~99% local, frontier only when worth it" 的身份：把升级策略从"预测难度再升级"推进到"测量分歧再升级"，同时把会话中断从默认停下来问用户，改成能续跑就续跑、必须输入时才收集输入。

## 背景

已知：
- `tinyctx/router.py` 是 local/frontier 最终路由点。
- `tinyctx/self_classify.py` 已有本地 advisor-needed 预检 classifier。
- `tinyctx/choice_arbiter.py` 已有本地多角色 debate 与 frontier fallback。
- `tinyctx/verifier.py` 已有 post-stream 本地质量 verifier，`tinyctx/guards.py` 已有 `VerifierGate`。
- `tinyctx/soft_completion.py`、`tinyctx/stream_rewrite.py`、`tinyctx/proxy.py` 已有 soft completion 检测、SSE completion 拦截、synthetic continue 注入。
- `tinyctx/session_state.py` 提供 per-session namespaced state。
- `tinyctx/dashboard.py` 已有 FastAPI/SSE dashboard surface。

推测：
- 本地模型在边界任务上的自相矛盾，和最终回答错误存在正相关。
- 信息缺口类失败是 local-first 体系里主要准确性损失来源之一。
- 将 verifier 低分接回 router/continuation，比单纯记录日志更能减少用户手动救场。
- 动态输入 UI 能显著减少密码、密钥、选项确认类中断。

待确认：
- Codex/OpenAI 客户端允许 SSE completion 被 park 的最大时长。
- dashboard 是否总在用户可见位置；否则需要桌面通知或 push 唤起。
- 历史 trace 中 `self_classify.p` 边界带与真实失败率的相关性。

## 方向 A：提高任务准确性

### A1. 自一致性采样作为升级信号

只对 `self_classify.p` 落在边界带的 turn 做本地冗余采样。采样目标不是完整执行任务，而是让本地模型输出"下一步决定性动作签名"。动作签名一致时继续 local；动作签名分歧时才升级 frontier。

接入点：
- `tinyctx/config.py`：新增边界带、样本数、开关。
- `tinyctx/self_classify.py`：新增动作签名采样与一致性判定。
- `tinyctx/proxy.py`：在 self-classify 后、router 前运行边界带采样。
- `tinyctx/router.py`：新增 self-consistency rule，优先于 `_classify_rule`。

成本取舍：
- 推测：边界 turn 增加 2-3 次本地小调用，远低于 frontier 调用成本。
- 准确性收益待 trace replay 验证；上线默认应保留窄边界带，避免全量采样。

### A2. 升级门控议会

把 `choice_arbiter.py` 现有本地多角色 debate 固化为 proposer / critic / verifier 角色；触发源来自 A1 分歧、verifier 低分、error streak，不每 turn 运行。

接入点：
- `tinyctx/choice_arbiter.py`
- `tinyctx/guards.py`
- `tinyctx/session_state.py`

成本取舍：
- 本地议会成本是 k 次 local 调用。
- frontier advisor 仅在本地议会僵持时兜底。

### A3. 信息源发散 / 按需检索 fan-out

当 turn 引用 scout 缓存里没有的符号/文件，或 A1/A2 发现分歧时，并行检索 scout cache、Serena、GitNexus、可选 web，去重后取 top-k 走 additionalContext 注入。

接入点：
- `tinyctx/auto_scout.py`
- `tinyctx/scout.py`
- `tinyctx/serena_bootstrap.py`
- `tinyctx/gitnexus_bootstrap.py`
- `tinyctx/context_mode_bootstrap.py`

成本取舍：
- 推测：本地索引检索成本低；对"因缺上下文而猜"的场景收益最大。
- web 检索必须保持显式开关，避免不可控延迟与外部依赖。

### A4. verifier 闭环

把 verifier 分项低分转成动作：
- `execution_evidence` 低：优先 synthetic continue，要求补验证。
- `task_completion` 低：下一 turn force frontier 或进入本地议会。
- `output_quality` 低：优先本地改写；连续低分再升级。

接入点：
- `tinyctx/verifier.py`
- `tinyctx/guards.py`
- `tinyctx/proxy.py`
- `tinyctx/synthetic_continue.py`

成本取舍：
- 推测：比单纯 total threshold 更少误升级。
- 需要避免 verifier 自身噪声造成循环续跑。

## 方向 B：减少会话中断

### B5. 流内动态输入表单

检测到缺密钥、缺密码、缺令牌等输入时，不直接结束 turn；在 SSE 中 park completion，通过 dashboard 弹出表单，用户提交后将值以内存 TTL 方式回填为 synthetic tool result/user turn，让模型继续。

接入点：
- `tinyctx/soft_completion.py`
- `tinyctx/stream_rewrite.py`
- `tinyctx/dashboard.py`
- `tinyctx/session_state.py`
- `tinyctx/sanitize.py`

成本取舍：
- 待确认：SSE park 上限。
- 密钥必须只存内存、TTL、永不 trace、UI 掩码。

### B6. 中断路由器

把 soft completion 末尾的"问用户"分类：
- `self_answerable`：直接 synthetic continue。
- `secret_input`：B5 password 表单。
- `choice`：选项按钮或 choice arbiter。
- `external_action`：用户必须离开系统做外部动作。
- `human_judgement`：真正打扰用户。

接入点：
- `tinyctx/soft_completion.py`
- `tinyctx/choice_arbiter.py`
- `tinyctx/stream_rewrite.py`

成本取舍：
- 推测：先做分类能减少误把"需要密码"当普通 soft punt 续跑。

### B7. Park-and-resume

输入收集慢时释放当前 SSE 流，保存可恢复状态；值到达后以新 turn 注入恢复。

接入点：
- `tinyctx/session_state.py`
- `tinyctx/proxy.py`
- `tinyctx/dashboard.py`

成本取舍：
- 比 B5 更可靠，但实现面更大；需要恢复语义与 trace 关联。

## 本轮实现范围

已实现：
- A1：self-consistency 配置、动作签名采样、边界带采样、router 分歧升级/一致保留 local。
- A2：`choice_arbiter.py` 固定 proposer / critic / verifier；本地议会无多数或非法输出时返回 `None`，由 `intercept()` 走 frontier advisor 兜底。
- A3：新增安全本地 retrieval fan-out 层，支持 mentioned path 与 scout cache provider，跨 provider 去重并合并 source；Serena/GitNexus/web 作为显式禁用占位，不伪装已接入。
- A4：verifier 分项低分接回 `VerifierGate`；低 execution evidence 触发本地验证续跑，低 task completion 升 frontier，低 output quality 先本地改写、连续低分再升 frontier。
- B5/B6/B7：soft completion 增加 interrupt taxonomy；`secret_input` 创建 pending input request，dashboard 提供 password 表单/API，提交值只存内存并由 guard 在后续 turn 注入后消费；慢输入走 park/resume 语义。

刻意保留为待确认/后续：
- 不实装长时间 SSE hold-stream；当前以 park/resume 兜底。
- 不默认启用 web/Serena/GitNexus 外部检索 provider。
- 不把 `gpt-5.5` 这类公开模型 id 当显式 frontier override；只有 `tinyctx-frontier` 是强制 frontier。

## 成功标准

- A1 默认只作用于边界带。
- 非边界 turn 行为保持兼容。
- `self_classify_escalates_to_frontier` 仍是 legacy/full-turn 开关。
- 分歧信号进入 trace/router reason，可审计。
- pending input status/snapshot/dashboard 不泄漏 secret 值。
- local-first 红线保持：显式 `tinyctx-frontier`、真实分歧、低质量闭环或硬规则才升 frontier。
- 全量测试通过，`graphify update .` 已运行。
