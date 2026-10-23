# research — 联网研究

内部知识答不出时检索公开资料，答案带来源编号；支持深度/快速两档。

## 流程

```mermaid
flowchart TD
    A[问题] --> B{缓存命中?}
    B -->|是| Z[返回缓存]
    B -->|否| C[规划: 拆子查询]
    C --> D[MCP web_search<br/>:8765]
    D --> E{有结果?}
    E -->|否| F[落回直接回答<br/>并说明无外部证据]
    E -->|是| G[证据筛选 + 时间过滤]
    G --> H[合成答案 + 来源编号]
    H --> I[写缓存]
    I --> Z
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `graph.py` | 图：cache_lookup → plan → search → synthesize |
| `planner.py` | 子查询规划（deep / fast） |
| `collector.py` | 调 MCP 检索并归一化结果 |
| `evidence.py` | 证据筛选与引用编号 |
| `time_filter.py` | 时间范围过滤 |
| `mcp_client.py` | MCP 客户端（web-search / ops 两个命名连接） |
