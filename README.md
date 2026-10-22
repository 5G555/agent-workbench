# Myagent

AI Agent 项目骨架。环境与依赖已就绪，业务代码从零开始写。

## 环境

| 项 | 值 |
|---|---|
| 项目路径 | `D:\AI实战\Myagent` |
| conda 环境 | `my-agent`（Python 3.13.15，位于 C 盘） |
| 环境位置 | `C:\Users\Administrator\anaconda3\envs\my-agent` |
| 依赖管理 | pip，声明在 `requirements.txt` / `requirements-dev.txt` |

> 该环境与 `C:\AI-Projects\my-agent` 共用同一套包。**两个项目只要依赖一致就没问题**；一旦 `Myagent` 需要额外的包（比如 torch、pgvector），建议 `conda create -n myagent python=3.13` 单独开一个，避免互相污染。

## 快速开始

```powershell
# 开一个新窗口（旧窗口不认新 profile）
conda activate my-agent
cd D:\AI实战\Myagent

# 依赖已装好，这一步可跳过；换机时才需要
python -m pip install -r requirements-dev.txt

# 自检
pytest
ruff check .
pyright

# 配置密钥
Copy-Item .env.example .env
notepad .env
```

不激活也能跑：

```powershell
& "C:\Users\Administrator\anaconda3\envs\my-agent\python.exe" -m pytest
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 运行时依赖（Web、LLM SDK、LangGraph、MCP、存储、追踪） |
| `requirements-dev.txt` | `-r requirements.txt` + pytest / ruff / pyright / ipython |
| `requirements.lock.txt` | 精确版本快照（114 行），换机复现：`pip install -r requirements.lock.txt` |
| `pyproject.toml` | 仅工具配置（ruff / pytest / pyright），不参与打包 |
| `.env.example` | 环境变量模板 → 复制成 `.env` |
| `app/__init__.py` | 包骨架，空文件 |
| `tests/test_env.py` | 5 项环境自检：版本、导入、Pydantic v2、异步 |

## 已装版本（2026-10-21）

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| fastapi | 0.156.8 | langgraph | 1.2.16 |
| uvicorn | 0.54.0 | langchain-core | 1.6.12 |
| pydantic | 2.14.1 | mcp | 1.30.0 |
| openai | 3.35.1 | langfuse | 5.0.4 |
| anthropic | 1.17.0 | sqlalchemy | 2.1.5 |
| httpx | 0.28.1 | pytest | 9.1.1 |
| ruff | 0.17.1 | pyright | 1.1.414 |

## 依赖选型要点

- **`mcp>=1.2,<2`**：MCP 2.0 是破坏性升级（`FastMCP` → `MCPServer`、wire type 改 snake_case、`httpx` → `httpx2`）。确认需要 2.0 特性后再放开上限。
- **`langfuse` 5.x**：别照抄 2.x 教程，用 `from langfuse import observe, get_client`。
- **`openai` 3.x 依赖 `httpx2`**：`httpx` 与 `httpx2` 并存是上游设计，不是冲突。
- **没装 `torch`**：本骨架不含本地推理，需要 BGE 嵌入时再装（cp313 轮子齐全）。

## 目录结构（对齐 EduAgent 的分层）

```
Myagent/
├─ config.py             # 唯一配置中心（待写）
├─ main.py               # FastAPI 入口（待写）
├─ core/                 # 工程内核，所有 Agent 共用
│  ├─ state.py           # 主图状态契约 SupervisorState
│  └─ orchestrator.py    # 主 Agent：决策 + 路由 + 聚合（原 supervisor.py）
├─ agents/               # 每个子 Agent 一个包，四件套
│  └─ <name>/
│     ├─ state.py        #   TypedDict
│     ├─ nodes.py        #   节点函数
│     ├─ prompts.py      #   提示词
│     └─ graph.py        #   build_xxx_graph()
├─ api/
│  └─ v1/                # 每个 Agent 一个 router
└─ tests/
```

约定：主 Agent 不放在 `agents/` 里，它是编排器；`agents/` 只装子 Agent；跨 Agent 复用的东西进 `core/`。

## 常用命令

```powershell
pytest                    # asyncio_mode=auto，异步用例无需 marker
pytest --cov=core         # 覆盖率
ruff check . --fix
ruff format .
pyright
```

## 两个 Agent 与 MCP

| Agent | 数据来源 | 入口 |
|---|---|---|
| `qa` | 内部知识库（Milvus + FAQ + Redis 缓存） | `POST /api/v1/qa/ask` |
| `research` | 公网（经 MCP 调 web_search 工具） | `POST /api/v1/research/ask` |
''| `crawler` | 指定网页（抓正文 → 结构化记录 → 落库） | `POST /api/v1/crawler/run` |
| `ops` | 管理采集任务（查任务 / 导出 / 清理，经 MCP 工具） | `POST /api/v1/ask`（自动路由） |'

爬取数据全部存在 PostgreSQL（不落盘）：

| 表 | 内容 |
|---|---|
| `crawl_tasks` | 任务目标、状态、页数/记录数、停止原因 |
| `crawl_pages` | 正文（`content`）+ **原始 HTML（`html`，上限 30 万字符）** |
| `crawl_records` | 结构化记录（JSON 文本） |

'原始 HTML 一定要存：抽取规则出错时（比如字段名取错、丢字段），
没有原文就只能重新访问站点，而站点内容可能已经变了。

需要文件时再导出（数据默认只在库里）：

'```powershell
python -m scripts.export_crawl --list        # 看最近任务
python -m scripts.export_crawl --task 5      # 导出到 data/crawl/5/

# 导出目录会越攒越多，清一下（默认只统计，加 --apply 才删）
python -m scripts.clean_crawl --list
python -m scripts.clean_crawl --keep 3 --apply         # 只留最近 3 个任务
python -m scripts.clean_crawl --older-than 7 --apply   # 删 7 天前的
python -m scripts.clean_crawl --task 3 --apply         # 删指定任务
python -m scripts.clean_crawl --older-than 30 --db --apply   # 连库里的老任务一起删
```'

```
data/crawl/5/
    manifest.json            任务元信息 + 文件清单
    records.jsonl            结构化记录，一行一条
    pages/0013-标题.html      原始 HTML
    pages/0013-标题.txt       清洗后的正文（头部带来源 URL）
```''

**为什么 qa 不用 MCP**：它的检索是同进程内的函数调用，上 MCP 只会多一次网络往返和一个故障点。
**为什么 research 用 MCP**：web 搜索是外部服务，且要被别的客户端复用，协议化才有价值。

启动 MCP 服务（独立进程，research agent 依赖它）：

```powershell
python -m mcp_servers.web_search_server
# 监听 127.0.0.1:8765，端点为 http://127.0.0.1:8765/mcp
# 暴露的工具：web_search(query, max_results) / fetch_page(url, max_chars)
```

`.env` 里对应的地址：

```
WEB_SEARCH_MCP_URL=http://127.0.0.1:8765/mcp
```

验证工具能连上：

```powershell
python -c "import asyncio; from langchain_mcp_adapters.client import MultiServerMCPClient as M; print(asyncio.run(M({'ws':{'url':'http://127.0.0.1:8765/mcp','transport':'streamable_http'}}).get_tools()))"
```

网页前端顶部有模式切换：**自动（推荐）** / **知识库问答** / **联网研究**。三者共用同一张会话表，切换不丢上下文。

### 按数据性质路由（而不是"两个助手并列"）

联网搜索不是另一个问答系统，而是 RAG 的一个**检索增强组件**。两者的差别只在数据性质：

| 维度 | 联网搜索 | 内部 RAG |
|---|---|---|
| 本质 | RAG 的一种特定实现 / 增强组件 | 通用的"检索 + 生成"框架 |
| 典型数据源 | 公开互联网（实时、动态） | 私有知识库（静态、可控） |
| 核心贡献 | 解决时效性与公开信息覆盖 | 解决私有知识利用与信息整合 |
| **权限边界** | **无法处理私有数据的访问权限** | **可且必须嵌入企业权限体系** |

`POST /api/v1/ask` 据此分派：

```
问题 → supervisor（按数据性质判断）
        ├─ 内部 / 敏感 / 需权限 ──→ qa（自建 RAG，数据不出域）
        │        └─ 查不到 → 权限闸门 ─┬─ 私有表述 → 不外发，提示用户
        │                              └─ 可公开   → 补公网证据，统一作答
        └─ 公开 / 时效性 ─────────→ research（联网 + 引用）
```

三个关键实现：

1. **权限闸门**（`agents/supervisor/privacy.py`）：升级前先判断问题是否含私有表述（"我们公司""工号""内部规定"等）。命中就不外发——**联网搜索最大的风险不是搜到什么，而是把问题本身交出去了**。
2. **统一证据结构**（`agents/research/evidence.py`）：内部文档和公网结果用同一个 `Evidence` 表示，只靠 `source_type` 区分。作答只有一份提示词、一次生成，公网资料标注为"仅供参考"，与内部口径冲突时以内部为准。
3. **路由结论优先**：supervisor 判成内部问题后，qa 跳过二次分类（`force_professional`），避免两个模型判断打架。

升级的判据是精排分数：`RETRIEVAL_MIN_SCORE`（默认 0.30）以下的片段直接丢弃；丢空即 `no_result`。实测相关问题 top1 ≥0.80、无关问题 ≤0.01，阈值有足够余量。

**改了检索逻辑记得 bump 缓存版本**，否则旧答案还会被命中：

```powershell
python -c "import asyncio; from agents.qa.cache import bump_cache_version; print(asyncio.run(bump_cache_version()))"

# 旧版本的键不会自动删，攒多了跑一下清理（先统计，确认后加 --apply）
python -m scripts.clean_cache
python -m scripts.clean_cache --apply
```

## 多租户与细粒度权限

权限有三个维度，都来自 JWT（注册时可指定，生产应由管理员开通）：

| 维度 | 举例 | 作用 |
|---|---|---|
| `tenant_id` | `tenant_default` / `tenant_demo` | 公司级隔离，跨租户完全不可见 |
| `department` | `一号部门` / `二号部门` / `*` | 部门级；文档 `*` 表示全公司可见 |
| `role` | `user` / `admin` | 角色级；文档 `min_role` 表示最低角色要求 |

**检索条件的生成规则**（`RetrievalScope.filter_expr`）：

```
管理员（role=admin）：  tenant_id == "t1"
普通用户（一号部门）：  tenant_id == "t1"
                        and (department == "一号部门" or department == "*")
                        and min_role in ["user"]
普通用户（无部门 *）：  tenant_id == "t1" and department == "*" and min_role in ["user"]
```

注意 `*` 的两种含义：**文档上**表示全公司可见，**用户上**表示未分配部门（那就只能看公共文档）。

隔离渗透到五层：

| 层 | 隔离方式 |
|---|---|
| Milvus | chunk 带 `tenant_id` / `department` / `min_role`，检索表达式永远包含它们 |
| 主键 | `md5(租户\|部门\|角色\|来源\|内容)`——只用内容哈希会让相同句子跨部门互相覆盖 |
| FAQ | 表带 `tenant_id` + `department`，按「租户 × 部门」建多份 BM25 索引（详见下） |
| 答案缓存 | 键含完整范围：`qa:ans:v{ver}:s{scope_hash}:{query_hash}` |
| 提示词 / 来源列表 | 知识库主题与 `/sources` 都按同一范围过滤 |

**FAQ 的索引范围**：BM25 必须先有候选集再算分（不能算完再过滤，否则会命中看不到的条目），
所以按范围各建一份索引：

```
__all__     租户全部条目     （管理员）
__public__  只有 department='*' （未分配部门的用户）
一号部门     该部门 + '*' 的条目 （该部门用户）
```

同一句问题若在公共和部门各有一条，索引里**按问题去重、部门专属优先**——
不去重的话两条近似文本会互相稀释 BM25 置信度（实测从 0.9 掉到 0.46，直接判未命中）。


**灌数据要指定归属**：

```powershell
python -m scripts.ingest --reset --tenant tenant_default --department '*' --min-role user
python -m scripts.ingest data\tenant_demo_kb --tenant tenant_demo
python -m scripts.ingest data\dept_one_kb --tenant tenant_default --department 一号部门
python -m scripts.ingest data\admin_kb --tenant tenant_default --min-role admin
python -m scripts.ingest_faq --tenant tenant_default                      # 全公司公共
python -m scripts.ingest_faq data\dept_one_faq.csv --department 一号部门   # 部门专属
```

验证（仓库内带了三套演示数据：公共 / 一号部门 / 管理员）：

```
carol（一号部门）问「2026 年市场费用预算是多少」→ 命中 dept_one_预算与报销
dave （二号部门）问同一句                     → no_result（看不到）
root （admin） 问「新增租户需要先做什么」      → 命中 admin_操作手册
carol（一号部门）问同一句                     → no_result（角色不够）
```

实测可见条数：管理员 75 / 一号部门 65 / 二号部门 56 / 无部门 56 / 演示租户 8。

> 部门名目前用「一号部门 / 二号部门」占位，接入真实组织架构后替换即可
> （重灌数据，或 `UPDATE users SET department='真实部门'` 后重灌该部门文档）。
>
> FAQ 已支持部门级：公共条目 `department='*'` 对所有部门可见，部门专属条目只有本部门能命中。
> 实测 `carol`（一号部门）与 `dave`（二号部门）问同一句「你们怎么收费」，
> 前者拿到部门结算价，后者拿到公共报价。

验证隔离（仓库里带了 `tenant_demo` 的演示数据）：

```
alice（tenant_default）问「你们怎么收费」→ 按坐席数加对话量双维度计费…
bob  （tenant_demo）  问同一句        → 演示租户按年包干，每年 9800 元…
```

三个注意点：

- **改了 Milvus schema 必须 `--reset` 重建集合**；`CREATE TABLE IF NOT EXISTS` 不会改已有表结构，FAQ 表用的是 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 原地迁移。
- **注册接口接受 `tenant_id` 只为本地验证**，生产环境应由管理员开通租户。
- FAQ 命中阈值改成"相对均匀分布提升度"，与库大小无关（大库 0.93 / 小库 0.79 / 无关 0.00），默认 0.70。

## 排错

- **`conda : 无法识别`**：执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`（本机已设好），然后**重开终端**。
- **`pyright` 报 Import 无法解析**：conda 环境没有 `pyvenv.cfg`，`pyproject.toml` 里用 `venvPath` + `venv` 显式指定，改名要同步改。
- **`pip install` 装到别处**：确认 `(Get-Command python).Source` 指向 `...\envs\my-agent\python.exe`。
- **PyCharm 在项目里建了 `venv/`**：那是基于 conda 环境又套一层的空壳，已删除。解释器请直接选 `...\envs\my-agent\python.exe`，不要让它再建 venv。

## 下一步

1. `app/config.py`（Settings）+ `app/main.py`（`/health` 先跑通）
2. 落地第一个 Agent：LangGraph 状态图 / OpenAI Agents SDK / PydanticAI 选一
3. 工具走 MCP server，别把工具逻辑写死在 prompt 里
4. 第一天就挂 tracing（Langfuse）与 eval（promptfoo / deepeval）
