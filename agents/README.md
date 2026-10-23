# agents — 多 Agent 结构与协同

九个包：**六个执行 Agent** + 两条不检索路径 + 一个路由层。

## 总体协同

```mermaid
flowchart TD
    U[用户提问] --> S[supervisor 路由层]
    S -->|能力自述| H[help]
    S -->|私有资料| Q[qa]
    S -->|要实时事实| R[research]
    S -->|方法/生成| D[direct]
    S -->|URL + 采集| C[crawler]
    S -->|采集数据| A[analyst]
    S -->|指标| M[metrics]
    S -->|破坏性| O[ops]

    Q -->|证据不足| R
    A -->|带 URL| C
    A -->|数据不足| C

    R --> W[MCP web_search :8765]
    C --> W
    A --> T[MCP ops :8766]
    O --> T
    M --> P[(Postgres 只读视图)]
    Q --> V[(Milvus 向量库)]
    O --> G[执行网关<br/>审批/快照/审计]

    H -.不检索.-> H
    D -.不检索.-> D

    S -.记录纠正.-> F[(route_feedback)]
```

## 一次跨 Agent 请求的样子

```mermaid
sequenceDiagram
    participant U as 用户
    participant S as 路由层
    participant A as analyst
    participant C as crawler
    participant M as MCP 服务
    participant DB as Postgres

    U->>S: 抓取这个页面的行情然后画 K 线
    S->>A: 识别为采集分析
    A->>C: 检测到 URL，先抓
    C->>M: crawl(entry_url)
    M->>DB: 写入 pages / records
    M-->>A: task_id
    A->>M: task_records(task_id)
    A->>A: 预处理 + 沙箱执行 + 绘图
    A-->>U: 结论 + 图表 + 代码
```

## 各 Agent 文档

| Agent | 职责 | 文档 |
|---|---|---|
| `qa` | 内部知识问答 | [qa/README.md](qa/README.md) |
| `research` | 联网研究 | [research/README.md](research/README.md) |
| `crawler` | 网页采集 | [crawler/README.md](crawler/README.md) |
| `analyst` | 数据分析 | [analyst/README.md](analyst/README.md) |
| `ops` | 运维操作 | [ops/README.md](ops/README.md) |
| `ops/metrics` | 系统指标 | [ops/metrics/README.md](ops/metrics/README.md) |
| `direct` | 直接回答（不检索） | [direct/README.md](direct/README.md) |
| `help` | 能力说明（不检索） | [help/README.md](help/README.md) |
| `supervisor` | 路由与协同 | [supervisor/README.md](supervisor/README.md) |
