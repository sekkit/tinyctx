# tinyctx 产品技术方案图表

> 图表依据：`README.md` 的架构说明，以及 `tinyctx/proxy.py`、`tinyctx/router.py`、`tinyctx/compactor.py`、`tinyctx/continuity.py` 的实现路径。以下 Mermaid 块可直接在支持 Mermaid 的 Markdown 预览器中渲染。

## 1. 产品技术全景架构

```mermaid
flowchart TB
    user["开发者 / Codex 用户"]
    codex["Codex CLI<br/>wire_api = responses"]
    proxy["tinyctx-proxy<br/>FastAPI / Responses API"]

    local["Local cheap path<br/>Qwen3.6-27B via LMStudio / vLLM / SGLang"]
    frontier["Frontier path<br/>GPT-5.5 / Codex backend / compatible API"]

    cache["~/.tinyctx/cache<br/>session continuity / scout / traces"]
    secrets["~/.tinyctx/config.toml<br/>~/.tinyctx/secrets.env"]

    subgraph ProxyJobs["tinyctx-proxy 核心职责"]
        compact["Compaction interceptor<br/>handoff prompt -> local debate"]
        route["Cascade router<br/>heuristic + optional classifier"]
        scrub["encrypted_content scrub<br/>跨模型边界清理"]
        stable["Cache discipline<br/>byte-stable system + tools"]
        compress["Optional pre-escalation compression<br/>LLMLingua-2 hook"]
    end

    subgraph MCP["Codex 直连 MCP / 外部能力"]
        graphify["graphify<br/>代码知识图谱 / 多模态索引"]
        serena["serena<br/>LSP 符号操作"]
        ctxmode["context-mode<br/>沙箱执行 / 输出索引"]
        mem0["mem0<br/>跨会话记忆"]
        caveman["caveman-shrink<br/>工具描述 / 输出压缩"]
    end

    user --> codex
    codex -->|"HTTP /v1/responses"| proxy
    proxy --> ProxyJobs
    route -->|"default cheap path"| local
    route -->|"escalate only when needed"| frontier
    proxy --> cache
    proxy --> secrets
    codex -.-> MCP
```

## 2. 请求路由决策流

```mermaid
flowchart TD
    start(["请求进入<br/>/v1/responses 或 /v1/chat/completions"])
    parse["解析 JSON<br/>计算 session_id / project_session_key"]
    estimate["估算 tokens / turn_count<br/>识别 compaction handoff prompt"]
    force{"force_route?"}
    compact{"命中 compaction<br/>且允许本地重定向?"}
    errors{"error_streak 达阈值?"}
    ctx{"超过本地上下文<br/>安全阈值?"}
    turns{"turn_count 达阈值?"}
    classifier{"可选 classifier<br/>建议升级?"}
    local["route = local<br/>cheap path"]
    frontier["route = frontier<br/>frontier path"]
    mutate["请求变换<br/>strip encrypted_content / dedup tools / historian / advisor"]
    compactor{"本地 compactor debate<br/>启用且历史足够长?"}
    debate["3-role debate + judge<br/>生成 handoff summary"]
    forward["按 backend wire_api 转发<br/>responses 或 chat-completions"]
    response(["返回 Codex"])

    start --> parse --> estimate --> force
    force -->|"local"| local
    force -->|"frontier"| frontier
    force -->|"unset"| compact
    compact -->|"yes"| local
    compact -->|"no"| errors
    errors -->|"yes"| frontier
    errors -->|"no"| ctx
    ctx -->|"yes"| frontier
    ctx -->|"no"| turns
    turns -->|"yes"| frontier
    turns -->|"no"| classifier
    classifier -->|"yes"| frontier
    classifier -->|"no / unavailable"| local

    local --> mutate --> compactor
    compactor -->|"yes"| debate --> response
    compactor -->|"no"| forward --> response
    frontier --> mutate --> forward
```

## 3. Responses API 处理时序

```mermaid
sequenceDiagram
    participant C as Codex CLI
    participant P as tinyctx-proxy
    participant R as router.decide
    participant S as sanitize / mutation gate
    participant L as Local backend
    participant F as Frontier backend
    participant K as Cache / continuity

    C->>P: POST /v1/responses
    P->>R: estimate tokens, turns, compaction, error streak
    R-->>P: Decision(local/frontier, reason)
    P->>S: strip encrypted_content, rewrite model, scrub tools

    alt local compaction debate
        P->>L: archaeologist / narrator / enumerator drafts
        L-->>P: role drafts
        P->>L: judge merge
        L-->>P: merged summary + structured facts
        P->>K: save compaction-N.md/json and latest.md
        P-->>C: completed Responses payload
    else normal local route
        P->>L: forward responses or translated chat-completions
        L-->>P: model response
        P-->>C: response
    else frontier route
        P->>F: forward sanitized request
        F-->>P: model response
        P-->>C: response
    end
```

## 4. Compaction Debate 产物链路

```mermaid
flowchart LR
    handoff["Codex handoff-summary request"]
    history["_flatten_history<br/>剥离原始单次压缩指令"]

    subgraph Roles["并行本地草稿"]
        archaeologist["archaeologist<br/>保留路径 / 决策 / 错误等硬事实"]
        narrator["narrator<br/>重建工作故事线"]
        enumerator["enumerator<br/>列出 artifacts / TODO / risks"]
    end

    judge["judge<br/>合并为结构化 handoff"]
    parse["parse_judge_output<br/>markdown + JSON sidecar"]
    save["save_compaction<br/>compaction-N.md / .json / latest.md"]
    payload["build_responses_api_payload<br/>伪装为完成的 Responses 输出"]
    fallback["fallback_concat<br/>judge 失败时合并幸存草稿"]

    handoff --> history --> Roles --> judge --> parse --> save --> payload
    judge -.失败.-> fallback --> parse
```

## 5. MCP 与上下文治理拓扑

```mermaid
flowchart TB
    codex["Codex CLI"]
    proxy["tinyctx-proxy<br/>只处理模型 HTTP 路由"]

    subgraph DirectMCP["Codex 直连 MCP 层"]
        graphify["graphify<br/>repo graph / multimodal index"]
        serena["serena<br/>find_symbol / references / replace_body"]
        ctxmode["context-mode<br/>sandbox execution / searchable output"]
        mem0["mem0<br/>project memory"]
        caveman["caveman-shrink<br/>tool/output shrink"]
    end

    subgraph TinyctxLocal["tinyctx 本地增强"]
        scout["tinyctx-scout<br/>repo scan + compression ranking"]
        interest["tinyctx-interest<br/>compression PageRank"]
        keypin["tinyctx-keypin<br/>rollout key file detection"]
        trace["tinyctx-trace / stats<br/>routing observability"]
        recall["tinyctx-recall<br/>latest compaction retrieval"]
    end

    codex --> proxy
    codex -.MCP calls.-> DirectMCP
    proxy --> TinyctxLocal
    scout --> interest
    recall --> proxy
```

## 6. 持久化与可观测性数据流

```mermaid
flowchart TD
    request["每次模型请求"]
    trace["RequestTrace<br/>route / reason / tokens / tool trims / translated_calls"]
    logs["tinyctx-trace<br/>watch routing and upstream errors"]

    compaction["compaction summary"]
    markdown["compaction-N.md<br/>human-readable handoff"]
    json["compaction-N.json<br/>compartments / facts / open_questions"]
    latest["latest.md<br/>latest recall pointer"]

    scout["scout cache<br/>manifest / repo hash / graph ranking"]
    memory["optional mem0<br/>long-lived project memory"]

    request --> trace --> logs
    compaction --> markdown
    compaction --> json
    markdown --> latest
    json --> latest
    scout --> latest
    latest --> memory
```

