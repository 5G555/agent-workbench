# supervisor — 路由层（多 Agent 协调的入口）

决定一个问题交给谁。不是分类器，而是**闸门 + 证据判定 + 置信度兜底**。

## 路由流程

```mermaid
flowchart TD
    A[用户问题] --> B{能力自述?}
    B -->|是| H[help]
    B -->|否| C{私有资料?}
    C -->|是| Q[qa]
    C -->|否| D{URL + 采集动作?}
    D -->|是| CR[crawler]
    D -->|否| E{破坏性动作?}
    E -->|否| F{系统指标?}
    F -->|是| M[metrics]
    F -->|否| G{采集数据分析?}
    G -->|是| AN[analyst]
    G -->|否| I{证据判定}
    E -->|是| O[ops]
    I -->|must 要实时事实| R[research]
    I -->|never 方法/生成| DI[direct]
    I -->|gray| L[模型判断]
    L --> N{置信度 ≥ 阈值?}
    N -->|是| R2[按模型判断]
    N -->|否| DI2[direct]
```

## 为什么不是分类器

本地 BERT 三分类把"你可以做什么"判成 `research`（置信度 1.00）——
**高置信度不等于正确**，分布外照样给满分。所以：

1. 用户说得很具体的（URL、破坏性动作、指标词）用规则锁死
2. 需要外部事实的按证据判定走 research
3. 方法/生成类直接 direct
4. 灰区交给模型，但**低于阈值一律回 direct**（误检索代价远大于误直接答）

## 关键文件

| 文件 | 作用 |
|---|---|
| `router.py` | 闸门顺序 + 证据判定 + 置信度兜底 |
| `evidence.py` | 三档证据判定（must / never / gray） |
| `local_router.py` | 本地 BERT（现仅兜底） |
| `privacy.py` | 隐私闸门（敏感问题不外发） |
| `feedback.py` | 路由纠正记录（调阈值的依据） |
