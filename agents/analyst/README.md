# analyst — 采集数据分析

对爬回来的结构化记录做 pandas 计算与绘图；两条路径：
自然语言（模型写代码）与手动模式（用户定口径，代码按模板生成）。

## 流程

```mermaid
flowchart TD
    A[问题] --> B{带 URL?}
    B -->|是| C[先抓取<br/>MCP crawl]
    B -->|否| D{结果缓存命中?}
    C --> D
    D -->|是| Z[返回缓存]
    D -->|否| E[explore: 取记录<br/>MCP task_records]
    E --> F[预处理: 转数值/转时间/去重]
    F --> G[代码: 模型生成 或 模板生成]
    G --> H[沙箱执行<br/>子进程 + 25s 超时]
    H --> I{执行成功?}
    I -->|否| J[回传报错重写一次]
    J --> G
    I -->|是| K[narrative: 解读 + 图表]
    K --> L[写缓存] --> Z
```

## 手动模式（另一入口）

```mermaid
flowchart LR
    A[选任务/字段/聚合/图表] --> B[build_code 模板生成]
    B --> C[同一沙箱执行]
    C --> D[图表 + 代码 + 输出]
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `graph.py` | dispatch → explore → code → narrative |
| `frame.py` | 预处理（`_num` / `_dt` 列生成） |
| `_sandbox.py` | 子进程沙箱（pd/np/plt + kline 助手） |
| `code_runner.py` | 超时、静态筛查、import 白名单 |
| `charts.py` | 图表类型识别（K线/柱状/散点/饼图…） |
| `manual.py` | 手动模式：参数 → 确定性代码 |
| `result_cache.py` | 结果缓存（含数据指纹） |
