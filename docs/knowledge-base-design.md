# tinyctx 知识库设计（Scoped Knowledge Base）

> 状态：Phase 1 已落地（`tinyctx/knowledge_base.py` + `tests/test_knowledge_base.py`）。
> Phase 2（PDF/MiniRAG 后端）与 proxy 接入为设计待执行项，见文末路线。

## 1. 定位

tinyctx 的信息源是三个维度，知识库是其中"深度·单篇/作用域"一维：

| 维度 | 形态 | 模块 |
|------|------|------|
| 广度·联网·动态 | web 搜索（SearXNG/GPT-Researcher，未实现） | — |
| 广度·离线·静态 | 知识库大全（DevDocs/Kiwix，未实现） | — |
| **深度·作用域** | **本设计：按 scope 的文档知识库** | `knowledge_base.py` |

**核心决策：scoped，不是 global。** 一个"无所不包"的全局知识库对编程 agent 价值存疑——codex 要的是*当前任务相关*的精准上下文，全局索引主要带来检索噪声，反伤准确性。所以每个 scope（项目根 / 主题 / 任务）一个隔离的小知识库。这与 `memory.py` 的克制一致（mem0 记忆刻意不自动注入，避免污染 prompt cache）。

## 2. 架构：作为 retrieval_fanout 的一个 provider

tinyctx 已有 `retrieval_fanout.py`：确定性的 merge/dedup/budget/inject 层，provider 契约是 `Callable[[str], list[RetrievalHit]]`，外部 provider 默认禁用、显式 config 开启。知识库**不另造轮子**，而是提供一个符合该契约的 provider：

```
knowledge_base.search(scope, query)  ->  list[SearchHit]
        │  knowledge_base_provider(scope) 适配
        ▼
RetrievalHit(source="knowledge_base", path=doc_id, snippet, score)
        │  与 mentioned_path / scout_cache 等 provider 一起
        ▼
retrieval_fanout.run_fanout(query, providers)  →  inject_context(body)
```

零改动 `retrieval_fanout.py` / `proxy.py`。KB provider 由调用方显式传入。

## 3. Phase 1（已实现）：纯本地词袋检索

**为什么先做词袋而非 graph-RAG**：守住 local-first 红线——默认路径零依赖、确定性、可离线测试、可回滚。

- **scope 隔离缓存**：`~/.tinyctx/cache/kb/<scope_hash>/store.json`（复刻 `scout.repo_hash` + `~/.tinyctx/cache/<hash>/` 布局）。`TINYCTX_KB_DIR` 可覆盖。
- **摄入**：文件→文本→切片→JSON。`.md/.txt/源码`直接读；PDF/docx 需 markitdown（可选，未装则降级，仿 `memory.is_available`）。同 `doc_id` 重摄入即增量替换。
- **检索**：IDF 加权词袋，稀有词主导、长 chunk 长度归一。纯 Python、确定性。中文按字成 token（无需分词器）。score 归一到 `(0, 0.85]`——低于直接文件命中（1.0）、可高于 scout_cache 粗快照（0.6）。
- **密钥安全**：摄入跳过敏感文件名、丢弃疑似密钥 chunk；provider 出口再过滤一次（defence in depth）。全部复用 `retrieval_fanout.is_sensitive_path / contains_sensitive_text`。
- **CLI**：`tinyctx-kb ingest|search|list|remove|stats <scope> ...`

### 接口
```python
ingest(scope, *, path=None, text=None, doc_id=None) -> IngestResult
search(scope, query, *, top_k=5) -> list[SearchHit]
knowledge_base_provider(scope, *, top_k=5) -> Provider   # 接 retrieval_fanout
remove(scope, doc_id) -> bool
list_docs(scope) / stats(scope)
```

## 4. 成本阶梯（守 local-first）

```
mentioned_path(0依赖) → scout_cache → knowledge_base(本地词袋,0依赖) → [Phase2: markitdown PDF] → [Phase2: MiniRAG 建图,重]
```
默认路径零增量成本；重后端逐级 opt-in，只在前一级不够时才进下一级。

## 5. Phase 2（设计，未实现）：可选增强后端

均为 opt-in，默认关闭，不进默认安装：

- **PDF/Office 摄入**：`pip install tinyctx[kb]`（markitdown）。`convert_to_text` 已留 `_converter` 注入点，装了即生效。
- **MiniRAG/LightRAG graph 后端**：`pip install tinyctx[kb-graph]`。
  - 用于"大文档/反复查询"场景；小文档保持词袋。
  - LLM=本地 27B、embedding=本地 bge-m3、**砍掉 alphaxiv 的 Gemini 生成层**（codex 自己生成），只取检索片段。
  - `待确认`：LightRAG 官方推荐 ≥32B 做实体抽取，27B 略低于推荐线 → 优先评估 **MiniRAG**（专为 SLM 设计）。
  - `待确认`：大文档建图延迟（分钟级）→ 必须异步（仿 `auto_scout.schedule_bootstrap` 的 `create_task` + 去重 set），不阻塞 turn。

## 6. proxy 接入（设计，未实现 — L2 需确认后再做）

接入会触碰主流程，按规则需先确认：

1. `config.py`：加 `kb_enabled: bool = False`、`kb_scope_strategy`（按 `x-codex-cwd` 取项目根作 scope）等字段。
2. `proxy.py`：在已有 `retrieval_fanout` 调用处（proxy.py:~1279），当 `kb_enabled` 时把 `knowledge_base_provider(scope)` 加入 providers 列表。
3. 触发收窄：仅当请求引用了已摄入的 scope 文档时才查（避免无谓注入）。

回滚：以上均为加性改动；删除 KB 字段与 provider 注入即还原。

## 7. 风险与待确认

- `推测`：全局大全知识库对编程 agent 边际价值低且可能因噪声反伤准确性 → 本设计坚持 scoped。
- `推测`：词袋检索对语义近义召回弱于向量/图谱 → 作为零成本默认够用；语义场景走 Phase 2。
- `待确认`：MiniRAG vs LightRAG 在 27B 上的建图质量与延迟，需实测定收录上限。

## 8. 已落地清单

- `tinyctx/knowledge_base.py` — Phase 1 自建作用域库全部能力 + CLI（`tinyctx-kb`）
- `tinyctx/knowledge_sources.py` — 4 个外部知识源 provider（见 §9）+ CLI（`tinyctx-kb-sources`）
- `tests/test_knowledge_base.py`（19）+ `tests/test_knowledge_sources.py`（22）— 离线确定性
- `pyproject.toml` — `tinyctx-kb` / `tinyctx-kb-sources` 脚本 + `kb` / `kb-graph` 可选依赖
- **已接线（加性 opt-in）**：`retrieval_fanout.inject_for_disagreement` 增加 `extra_providers` 参数；`proxy.py` 在 self-consistency 分歧注入处传 `knowledge_sources.external_providers(CFG)`。默认全关，零默认行为变化。
- **dashboard 可视化配置（已接入）**：`config.py` 加 `kb_*` 字段（默认 `None`=回退 env、配置优先）+ `load_config` 读 `[knowledge_sources]` section；`config_schema.py` 加 `knowledge_sources` section（dashboard Config Center 自动渲染 + 校验）。`external_providers(cfg)` 同时认 config 字段与 env。

## 9. 外部知识源（"知识库大全"第五路，已落地）

与 §1–§5 的"自建作用域库"互补：这是 tinyctx **只查不建**的外部成熟知识库，同样作为 retrieval_fanout 的 provider。全部默认关闭、优雅降级（服务不可达/超时/解析失败→`[]`，绝不阻塞）、HTTP 可注入（离线可测）、密钥经 inject_context 过滤。

| 源 | 定位 | 形态 | 启用 env |
|----|------|------|---------|
| **DevDocs** | 全语言官方文档（编程收益最直接） | 自托管 HTTP，本地匹配 index.json + 缓存 | `TINYCTX_KB_DEVDOCS=1` + `_URL` + `_SLUGS` |
| **Kiwix** | SO/Wikipedia 离线快照 | 自托管 kiwix-serve（XML/JSON best-effort） | `TINYCTX_KB_KIWIX=1` + `_URL` |
| **Wikidata** | 实体/事实校验（CC0） | wbsearchentities（公网/可指本地 Wikibase） | `TINYCTX_KB_WIKIDATA=1` |

> 范围决策：只保留**完全免费 + 无需 apikey**的源。Semantic Scholar 已移除（无 key 限流严重、是唯一有 key 概念的源）。DevDocs/Kiwix 纯本地自托管；Wikidata 公网 CC0、无 key、可指向本地 Wikibase。

手动验证：`tinyctx-kb-sources <source> "<query>" [--url ...] [--slug ...]`

**Wikidata 同名实体歧义（已解决）**：`wbsearchentities` 按标签匹配 + Wikidata 热度排序，不看 query 语境（"Transformer" 返回电气变压器/变形金刚）。解法（`_rerank_by_context`）：① 整句先提取专有名词候选（"who was Ada Lovelace"→"Ada Lovelace"）；② 多取候选（limit≥10）；③ 用 query 上下文词与候选 **description** 的重叠重排，stable sort 保底（无信号时退回原序，绝不劣化）。实测："…Transformer architecture in deep learning" → `Q85810444 machine-learning model architecture` 升至第一。零额外请求、零依赖、确定性。`待确认`：依赖 description 含上下文词；正确实体须落在 wbsearchentities 返回的候选集内；中文上下文弱。

`待确认`：DevDocs index.json 的确切 URL 布局随版本可能不同；Kiwix `format=json` 支持度（已做 XML 回退）；Wikidata 公网 wbsearchentities（轻量）与"重场景 100GB qEndpoint SPARQL"是两条路——后者需自然语言→SPARQL，未做。
