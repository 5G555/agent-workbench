# direct — 直接回答（不检索）

方法、写作、解释、闲聊类问题，不查任何数据，直接让模型答。

## 流程

```mermaid
flowchart TD
    A[问题 + 最近对话] --> B[模型输出<br/>answer + uncertain]
    B --> C{有拿不准的事实点?}
    C -->|有| D[答案后附说明<br/>"这部分可能需要我查一下"]
    C -->|无| E[直接返回]
    D --> F[返回]
    E --> F
```

## 设计要点

- **不接触任何检索组件**：`tests/test_routing_layer.py` 里有静态断言，
  这个包里不允许出现 `mcp_client` / `web_search_server` 等字样。
- **answer-then-flag**：不让模型判断"要不要联网"（它校准很差），
  而是让它答完标出不确定的事实点，把不确定性显式交给用户。

## 关键文件

| 文件 | 作用 |
|---|---|
| `nodes.py` | 结构化输出 `{answer, uncertain}` |
| `graph.py` | 单节点图 |
