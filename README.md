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

网页前端顶部有模式切换：**知识库问答** / **联网研究**。两者共用同一张会话表，切换不丢上下文。


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
