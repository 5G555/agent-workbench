# Agent Workbench — 多 Agent 智能问答与数据平台

![CI](https://github.com/5G555/agent-workbench/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

一个可运行的多 Agent 系统：内部知识问答、联网研究、网页采集、数据分析、运维操作
由一个路由层统一分派，全部走同一套权限、缓存与审计。

不是"一个聊天框套一层检索"，而是**六个可执行 Agent + 三层护栏**：

- 路由层决定"这件事该谁做"，并保证**不需要检索的问题不触发检索**
- 每个 Agent 有独立的图（LangGraph），可单独测试、可替换
- 破坏性操作必须经过策略 → 审批 → 快照 → 审计，失败自动回滚

## 能力一览

| 入口 | Agent | 能做什么 |
|---|---|---|
| 💬 对话 | `qa` | 产品/业务问题：FAQ 精确匹配 + 向量检索 + 精排，答不出自动升级联网 |
| 💬 对话 | `research` | 联网研究，答案带来源编号 |
| 💬 对话 | `direct` | 方法/写作/解释类问题直接回答，不检索 |
| 💬 对话 | `help` | "你能做什么"——基于真实能力清单与当前数据作答 |
| 🕷️ 爬虫 | `crawler` | 抓正文、抽表格/列表/段落，落库、导出文件 |
| 📊 分析 | `analyst` | 对采集数据做 pandas 计算与绘图（含手动模式：自己定口径） |
| 🛠️ 运维 | `ops` | 采集任务、导出清理、缓存清理、回收站、审批与审计 |
| 🛠️ 运维 | `metrics` | 用自然语言问系统指标（自动转 SQL，只读视图） |

## 架构

```
                      ┌────────────── 路由层 ──────────────┐
用户提问 ──▶ 闸门（确定性）──▶ 证据判定 ──▶ 模型（灰度区）──▶ direct 兜底
             │ 能力/私有/URL/破坏性/指标/采集分析
             ▼
      ┌─────────────── Agent 层（LangGraph）───────────────┐
      │  qa   research   crawler   analyst   ops   metrics │
      └───────────────────────┬───────────────────────────┘
                              ▼
      数据层：Postgres（业务/审计） · Milvus（向量） · Redis（缓存）
      外部：MCP 服务（web_search / ops 工具）
```

### 三个值得说明的设计

**1. 路由是闸门 + 证据判定，不是分类器**

本地 BERT 三分类曾把"你可以做什么"判成 `research`（置信度 1.00），联网搜回一堆
无关内容。现在的顺序是：

```
能力自述 → 私有资料 → URL采集 → 破坏性动作 → 系统指标 → 采集分析
        → 证据 must（要实时事实）→ 证据 never（方法/生成）→ 模型判灰区
```

闸门用规则锁死明确情况；灰区交给模型，但**置信度低于阈值就回 direct**（误检索的
代价远大于误直接回答）。`help` / `direct` 两条路径有静态断言保证不引用任何检索组件。

**2. 检索范围由视图/ACL 强制，不靠提示词**

分析视图（`ana_*`）里写死 `current_setting('app.tenant_id')`，未设置时返回 0 行；
向量检索走 `RetrievalScope(tenant, department, role)` 拼 filter 表达式。
模型忘写租户条件也拿不到别家数据。

**3. 破坏性操作是"网关"而不是"函数调用"**

```
策略判定 → 幂等 → 审批票据 → 预算 → 快照 → 意图审计 → 执行 → 校验 → 回滚/结果审计
```

备份失败就一个都不删（Fail-Closed）；审批票据绑定 `sha256(tool|args|tenant)`，
换参数重放无效；回收站保留 7 天。

## 快速开始

### 1. 起基础设施（Postgres / Milvus / etcd / MinIO / Redis）

```bash
docker compose up -d
docker ps --filter name=myagent        # 四个容器应为 healthy
```

### 2. 装依赖

```bash
conda create -n my-agent python=3.13 -y
conda activate my-agent
pip install -r requirements.txt
```

### 3. 配置 `.env`

最少需要这三项（其余有默认值）：

```ini
DB_PASSWORD=你的数据库密码
DEEPSEEK_API_KEY=你的 DeepSeek Key
JWT_SECRET_KEY=随机长字符串
```

模型路径需指向本地权重（BGE-M3 与 reranker，约 4.3 GB，**不在仓库里**）：

```ini
bge_m3_model_path=D:/AI-Models/bge-m3
reranker_model_path=D:/AI-Models/reranker/bge-reranker-large
```

### 4. 建表 + 起服务

```bash
alembic upgrade head                     # 空库可重建（已验证 19 表 8 视图）
python -m uvicorn main:app --host 127.0.0.1 --port 8000
python -m mcp_servers.web_search_server  # :8765
python -m mcp_servers.ops_server         # :8766
```

打开 http://127.0.0.1:8000 即可（首次使用需先导入 FAQ 与文档，见 `scripts/`）。

## 目录结构

```
agents/           按领域分包，每个 Agent 自带 graph/nodes/state
  qa/             FAQ、检索、缓存、答案合成
  research/       规划、检索、证据、时间过滤
  crawler/        抓取、抽取、落库、导出
  analyst/        预处理、沙箱执行、图表类型、手动模式
  ops/            策略、网关、审批、审计、回收站、缓存巡检
    metrics/      系统指标（SQL 视图 + 守卫）
  direct/ help/   不检索的两条路径
  supervisor/     路由（闸门 + 证据判定 + 反馈回路）
core/             与业务无关：db、日志、重试、安全、编排、追踪
api/v1/           HTTP 接口（auth / ask / qa / research / crawler / ops / analyst）
mcp_servers/      对模型暴露的工具（web_search / ops）
migrations/       Alembic（baseline 由 pg_dump 生成）
scripts/          导入、评估、导出、清理
static/           单文件前端（四个分区）
tests/            332 条测试
```

## 开发

```bash
pip install -r requirements-dev.txt   # ruff / pyright / pytest
ruff check .          # 代码规范
pyright               # 类型检查
pytest -q             # 332 passed
python -m scripts.eval_route          # 路由评测：41 条，分类目匹配度
```

CI（GitHub Actions）会跑这三项，见 `.github/workflows/ci.yml`。

## 仓库里有什么、没有什么

**包含**：全部源码、测试、迁移、前端、CI 配置、Dockerfile。

**不包含**（`.gitignore` 挡住，需要自己准备）：

| 内容 | 原因 |
|---|---|
| `.env` | 数据库密码、API Key、JWT 密钥 |
| `data/` | 抓取数据、导出文件、回收站 |
| `var/` | 运行日志、图表、临时脚本 |
| 模型权重 | BGE-M3 与 reranker 约 4.3 GB，用环境变量指向本地路径 |

clone 下来跑不起来，通常就是缺这三类东西中的一个。

## 已知限制

诚实列出，避免误解：

- **无监控告警**：有 `/health`（含连接池状态），但没有 Prometheus 指标与慢查询日志
- **DB 迁移只完成一部分**：新代码走 SQLAlchemy，旧模块仍是 asyncpg 原生池
- **`deep` 模式对 JS 渲染页无效**：需要 playwright，当前只做了 HTTP 抓取
- **多轮追问有第二道关**：追问能改写（"那包年呢" → "包年怎么收费"），
  但"月付"与"按人头 300 元/月"这类同义表达检索层还认不出
- **审批无通知**：管理员需手动刷新面板查看待审批
- **`qa` 首字延迟 8~9 秒**：生成前的意图分类与策略选择串行

更细的待办与修法记录在 `docs/待办.md`。

## License

[MIT](LICENSE)
