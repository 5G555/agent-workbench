"""运维 Agent：让模型自己挑 MCP 工具来查任务、导出、清理。

为什么单独做一个 Agent 而不是塞进对话：
    这些操作是"动作"不是"问答"——要选工具、填参数、看执行结果。
    典型形态是一个工具调用循环：模型决定调哪个工具 → 拿到结果 → 再决定下一步。

工具从哪来：
    mcp_servers/crawler_server.py 通过 MCP 暴露（crawl / list_tasks /
    task_records / export_files / clean_exports）。模型只看到 schema，
    不需要知道背后是 Postgres 还是文件系统。

安全：
    清理工具默认 dry-run。系统提示明确要求"用户没明确说要删，就不要传 apply=true"，
    并且第一次清理必须先 dry-run 把结果报给用户。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agents.research.mcp_client import get_mcp_client
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6

OPS_SYSTEM = """你是爬虫任务的运维助手，通过工具管理"网页采集"系统。

当前租户：{tenant_id}

可用能力：
- list_tasks          看最近的采集任务
- task_records        看某个任务抽到的结构化记录
- crawl               抓取一个网址（会访问外网，属于有副作用的操作）
- export_files        把某任务导出成文件（HTML / 正文 / JSONL）
- clean_exports       清理导出目录

硬性要求：
1. 清理（clean_exports）默认只做 dry-run：只有用户明确说"删掉/清掉/执行"时才传 apply=true。
2. 用户没给任务 ID 时，先用 list_tasks 查，再从结果里推断，不要瞎编 ID。
3. 抓取（crawl）前必须确认入口 URL；没有 URL 就先问用户，不要猜。
4. 回答要具体：给出任务号、页数、记录数、目录路径这些实际数值。
5. 用中文，简洁，不要复述工具的原始 JSON。"""


class OpsState(TypedDict, total=False):
    """工具调用循环的状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    tenant_id: str
    department: str
    role: str
    session_id: str
    original_query: str
    structured_output: dict


_tools_cache: list[Any] | None = None


async def get_ops_tools() -> list[Any]:
    """从 MCP 服务取工具，进程内缓存。"""
    global _tools_cache
    if _tools_cache is None:
        client = await get_mcp_client("crawler")
        _tools_cache = list(await client.get_tools())
        logger.info("ops.tools_ready", tools=[t.name for t in _tools_cache])
    return _tools_cache


async def agent_node(state: OpsState) -> dict[str, Any]:
    """模型节点：决定调哪个工具，或者给出最终答复。"""
    tools = await get_ops_tools()
    llm = LLMFactory.get_llm("crawler", temperature=0)
    bound = llm.bind_tools(tools)

    system = SystemMessage(
        content=OPS_SYSTEM.format(tenant_id=state.get("tenant_id") or "tenant_default")
    )
    messages = [system, *(state.get("messages") or [])]
    response = await bound.ainvoke(messages)
    return {"messages": [response]}


def route_after_agent(state: OpsState) -> str:
    """没有工具调用就是最终答复；调太多次就收尾，避免死循环。"""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return "tools"
    if not getattr(last, "tool_calls", None):
        return END
    rounds = sum(1 for m in messages if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
    if rounds >= MAX_TOOL_ROUNDS:
        logger.warning("ops.too_many_rounds", rounds=rounds)
        return "finalize"
    return "tools"


async def finalize_node(state: OpsState) -> dict[str, Any]:
    """轮次用尽时的兜底收尾：把已拿到的结果结构化返回。"""
    messages = state.get("messages") or []
    text = "已达到工具调用上限，先把目前查到的结果给你：\n"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text += str(msg.content)
            break
    return {
        "messages": [AIMessage(content=text)],
        "structured_output": {"answer_mode": "ops", "truncated": True},
    }


def build_ops_graph(checkpointer: Any = None):
    from langgraph.checkpoint.memory import InMemorySaver

    builder = StateGraph(OpsState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([]))
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize", END: END}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer or InMemorySaver())
    # ToolNode 需要编译时才知道工具列表：这里在首次使用时注入
    return _LazyToolGraph(graph)


class _LazyToolGraph:
    """包一层：第一次 ainvoke 时才去 MCP 取工具并重建图。

    这样 `get_orchestrator()` 的懒加载不会被 MCP 服务是否启动卡住——
    搜索服务没起时，其它 Agent 仍然可用。
    """

    def __init__(self, default_graph: Any) -> None:
        self._graph = default_graph
        self._ready: Any = None

    async def ainvoke(self, state: dict, config: dict | None = None):
        if self._ready is None:
            self._ready = await self._build()
        return await self._ready.ainvoke(state, config=config)

    async def _build(self) -> Any:
        from langgraph.checkpoint.memory import InMemorySaver

        tools = await get_ops_tools()
        builder = StateGraph(OpsState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", route_after_agent, {"tools": "tools", "finalize": "finalize", END: END}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        logger.info("ops.graph_ready", tools=[t.name for t in tools])
        return builder.compile(checkpointer=InMemorySaver())
