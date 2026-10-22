# Orchestrator：Agent 编排服务
# 职责：接收 API 层请求 → 意图路由 → 单 Agent 直达 / 多 Agent 串联 Pipeline

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from core.exceptions import PipelineError
from core.logger import get_logger
from core.retry import with_retry
from core.streaming import TokenSink

logger = get_logger(__name__)


class ExecutionMode(StrEnum):
    """执行模式：决定一个请求怎么跑。"""

    SINGLE = "single"  # 单 Agent 直达（最常见）
    PIPELINE = "pipeline"  # 多 Agent 串联（如求职全链路）
    CLARIFY = "clarify"  # 澄清对话（意图不明，需追问）


class AgentType(StrEnum):
    """Agent 类型标识（继承 str，可直接当字符串用）。"""

    QA = "qa"  # 智能问答（内部知识库）
    RESEARCH = "research"  # 联网研究（MCP 工具 + 多步检索）
    CRAWLER = "crawler"  # 网页采集（抓正文落库，供数据分析使用）
    OPS = "ops"  # 运维（通过 MCP 工具查任务 / 导出 / 清理）


class AgentRequest(BaseModel):
    """所有 Agent 请求的统一入参 Schema"""

    user_id: str = Field(..., description="发起请求的使用者 ID")
    tenant_id: str = Field(default="tenant_default", description="租户 ID")
    session_id: str = Field(..., description="会话 ID，用于 thread_id 拼接")
    agent_type: AgentType = Field(..., description="目标 Agent 类型")
    user_message: str = Field(..., description="用户输入的原始文本")
    context: dict[str, Any] = Field(
        default_factory=dict, description="附加上下文（文件路径/历史数据等）"
    )
    pipeline_mode: bool = Field(default=False, description="是否强制走串联 Pipeline")

    @property
    def thread_id(self) -> str:
        """LangGraph Checkpointer 使用的线程 ID（与各 Agent 的 build_thread_id 格式一致）"""
        return f"user_{self.user_id}_session_{self.session_id}"


class AgentResponse(BaseModel):
    """所有 Agent 响应的统一出参 Schema"""

    success: bool = Field(..., description="执行是否成功")
    agent_type: AgentType = Field(..., description="实际执行的 Agent 类型")
    content: str = Field(default="", description="主要文本响应内容")
    structured: dict[str, Any] | None = Field(default=None, description="结构化数据（评分/报告等）")
    fallback_used: bool = Field(default=False, description="是否触发了降级处理")
    # 降级原因：retry_exhausted / deep_to_fast / agent_unavailable / system_fallback
    degrade_reason: str = Field(default="", description="降级原因，空串表示没降级")
    error_msg: str | None = Field(default=None, description="失败时的错误信息")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class PipelineResult(BaseModel):
    """多 Agent 串联 Pipeline 的聚合结果"""

    steps: list[AgentResponse] = Field(default_factory=list, description="各步骤结果列表")
    combined: dict[str, Any] = Field(default_factory=dict, description="聚合后的最终数据")


class Orchestrator:
    """
    Agent 编排服务。

    职责：
        - 单 Agent 模式：直接调用对应 Agent 的 LangGraph 图
        - 串联 Pipeline 模式：按顺序调用多个 Agent，前序输出注入后序输入
        - 澄清对话模式：返回结构化追问，等待用户补充信息后重新路由

    所有 Agent 图在首次使用时懒加载，避免启动时全量加载占用资源。
    """

    def __init__(self):
        # Agent 图注册表（懒加载，key=AgentType，value=编译后的 LangGraph）
        self._agent_graphs: dict[AgentType, Any] = {}

        # 串联 Pipeline 定义（触发条件 → Agent 执行顺序）
        # 目前只有 qa，等接入更多 Agent 后再往里加
        self._pipelines: dict[str, list[AgentType]] = {}

        logger.info("orchestrator.initialized")

    def _get_agent_graph(self, agent_type: AgentType) -> Any:
        """
        懒加载 Agent 的 LangGraph 编译图。

        读取/写入：self._agent_graphs（图缓存字典）
        首次访问某 Agent 时才 import 并 build 它的图，之后复用缓存。
        """
        if agent_type not in self._agent_graphs:  # 缓存里没有 → 首次加载
            if agent_type == AgentType.QA:
                from agents.qa.graph import build_qa_graph

                self._agent_graphs[agent_type] = build_qa_graph()

            elif agent_type == AgentType.RESEARCH:
                from agents.research.graph import build_research_graph

                self._agent_graphs[agent_type] = build_research_graph()

            elif agent_type == AgentType.CRAWLER:
                from agents.crawler.graph import build_crawler_graph

                self._agent_graphs[agent_type] = build_crawler_graph()

            elif agent_type == AgentType.OPS:
                from agents.ops.graph import build_ops_graph

                self._agent_graphs[agent_type] = build_ops_graph()

            else:
                raise ValueError(f"未知 AgentType: {agent_type}")

            logger.info(
                "orchestrator.agent_graph_loaded",
                agent_type=agent_type.value,
            )

        return self._agent_graphs[agent_type]  # 返回（缓存的）编译图

    async def _escalate_to_web(
        self,
        request: AgentRequest,
        base: AgentResponse,
        on_token: TokenSink | None = None,
    ) -> AgentResponse:
        """把公网证据并进统一作答，而不是让第二个 Agent 另答一遍。

        三道关：
            1. 权限闸门：涉及内部/敏感信息的问题不往外发（联网搜索处理不了私有权限）
            2. 脱敏：通过闸门后仍把"我们/你们"这类表述中性化
            3. 统一作答：公网结果只是补充依据，标注为"公网资料，仅供参考"
        """
        from agents.qa.compose import compose_answer
        from agents.research.collector import collect_web
        from agents.research.evidence import EvidenceSet
        from agents.supervisor.privacy import is_private, sanitize

        private, reason = is_private(request.user_message)
        if private:
            logger.info("orchestrator.web_blocked", reason=reason)
            base.content = (
                f"{base.content}\n\n"
                "（这个问题涉及内部信息，出于数据安全考虑不会发送到公网检索。"
                "请补充内部资料，或联系管理员开通权限。）"
            )
            base.metadata["web_blocked"] = reason
            return base

        safe_query = sanitize(request.user_message)
        outcome = await collect_web(safe_query, tenant_id=request.tenant_id)
        if not outcome.usable:
            # 分清楚"服务出问题"和"确实没资料"：前者要提示稍后重试，别让用户以为知识库缺失
            logger.info("orchestrator.web_unusable", query=safe_query, status=outcome.status.value)
            message = outcome.user_message()
            if message:
                base.content = f"{base.content}\\n\\n{message}"
            base.metadata["web_status"] = outcome.status.value
            return base

        evidence = EvidenceSet()
        evidence.extend(outcome.items)
        answer, cited = await compose_answer(request.user_message, evidence, on_token=on_token)

        return AgentResponse(
            success=True,
            agent_type=base.agent_type,
            content=answer,
            structured={
                "answer_mode": "web_enhanced",
                "sources": evidence.sources_labeled(),
                "citations": cited,
                "kb_available": False,
            },
            metadata={
                **base.metadata,
                "escalated_from": "qa",
                "web_items": len(outcome.items),
                "web_status": outcome.status.value,
            },
        )

    async def handle_auto(self, request: AgentRequest) -> AgentResponse:
        """自动模式：按数据性质路由，qa 查不到时用公网证据补充同一份答案。

        路由维度（对应"数据性质"）：
            公开、时效性、外部事实  → research
            内部、敏感、需权限控制  → qa（自建 RAG，数据不出域）
        qa 查不到时的升级不是"换一个 Agent 重答"，而是给统一作答补一层公网证据。
        """
        from agents.supervisor.router import needs_escalation, route

        decision = await route(request.user_message, request.tenant_id)
        route_meta = {
            "agent": decision.agent,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

        # 需要确认：不跑任何 Agent，直接把问题抛回给用户
        if decision.need_confirm:
            logger.info("orchestrator.need_confirm", agent=decision.agent, q=request.user_message)
            return AgentResponse(
                success=True,
                agent_type=AgentType(decision.agent),
                content=decision.confirm_question,
                structured={
                    "answer_mode": "confirm",
                    "confirm": {
                        "agent": decision.agent,
                        "entry_url": decision.entry_url,
                        "goal": decision.goal or request.user_message,
                        "question": decision.confirm_question,
                    },
                },
                metadata={"route": route_meta},
            )

        target = AgentType(decision.agent)
        routed_request = request.model_copy(update={"agent_type": target})
        if target == AgentType.QA:
            # 让 qa 内部跳过二次分类，避免两个模型判断冲突
            routed_request.context["force_professional"] = True
        elif target == AgentType.CRAWLER:
            # 路由阶段已经解析出入参，直接注入，避免爬虫再问一次
            routed_request.context["entry_url"] = decision.entry_url
            routed_request.context["goal"] = decision.goal or request.user_message

        result = await self._run_single_agent(routed_request)
        result.metadata["route"] = route_meta

        answer_mode = (result.structured or {}).get("answer_mode")
        if target == AgentType.QA and needs_escalation(answer_mode):
            logger.info("orchestrator.escalating", from_agent="qa", to="web evidence")
            return await self._escalate_to_web(request, result)

        return result

    async def handle(self, request: AgentRequest) -> AgentResponse:
        """
        统一请求处理入口。

        读取 request：pipeline_mode（决定单 Agent 还是 Pipeline）、agent_type、student_id
        返回：AgentResponse（统一响应格式，无论成功失败都返回它，不向上抛异常）

        流程：
            1. 判断执行模式（单 Agent / Pipeline）
            2. 分发到对应执行方法
            3. 返回统一格式的 AgentResponse
        """
        logger.info(
            "orchestrator.handle_start",
            agent_type=request.agent_type.value,
            user_id=request.user_id,
            pipeline_mode=request.pipeline_mode,
        )

        try:
            if request.pipeline_mode:  # 强制串联模式
                result = await self._run_pipeline(request)  # 跑多 Agent
                return self._aggregate_pipeline(result, request)  # 聚合成统一响应
            else:  # 默认：单 Agent 直达
                return await self._run_single_agent(request)

        except Exception as e:  # 兜底：任何异常都转成失败响应，不抛给上层
            logger.error(
                "orchestrator.handle_failed",
                agent_type=request.agent_type.value,
                error=str(e),
                exc_info=True,
            )
            return AgentResponse(
                success=False,
                agent_type=request.agent_type,
                content="系统处理请求时遇到问题，请稍后再试。",  # 给用户的友好兜底文案
                error_msg=str(e),
            )

    def _build_state(self, request: AgentRequest) -> dict[str, Any]:
        """构建图的初始 State。

        context 里的 history（[{role, content}, ...]）会被转成消息，放在本轮用户消息之前，
        这样多轮对话的上下文能传给模型；其余 context 字段平铺进 State（如 source 过滤）。
        """
        messages: list[Any] = []
        for item in request.context.get("history") or []:
            role = str(item.get("role", "")).lower()
            content = str(item.get("content", ""))
            if not content:
                continue
            messages.append(
                HumanMessage(content=content) if role == "user" else AIMessage(content=content)
            )
        messages.append(HumanMessage(content=request.user_message))

        extra = {k: v for k, v in request.context.items() if k != "history"}
        return {
            "messages": messages,
            # 显式带上本轮问题：checkpointer 会恢复上一轮的 original_query，
            # 不覆盖就会拿旧问题去查缓存/检索
            "original_query": request.user_message,
            # 爬虫 Agent 的输入：抓什么、抓几个、是否指定 URL
            "query": request.user_message,
            "entry_url": request.context.get("entry_url") or "",
            "max_pages": request.context.get("max_pages"),
            "goal": request.context.get("goal") or request.user_message,
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
            # 细粒度权限：部门与角色随请求带进图，检索时用于拼过滤条件
            "department": request.context.get("department") or "*",
            "role": request.context.get("role") or "user",
            "session_id": request.session_id,
            **extra,
        }

    @staticmethod
    def _thread_id(request: AgentRequest) -> str:
        """每次请求用独立 thread，避免 checkpointer 把上一轮状态带进来。

        多轮上下文由 context["history"]（Postgres）提供，不依赖 checkpoint；
        等以后要做"断点续跑/人工介入"时，再改成按 session 复用 thread。
        """
        return f"{request.thread_id}:{uuid.uuid4().hex[:8]}"

    async def stream(self, request: AgentRequest) -> AsyncIterator[dict[str, Any]]:
        """流式执行：先推节点进度，再推生成的 token。

        Yields:
            {"type": "node",  "name": "classify", ...}
            {"type": "token", "data": "..."}
            {"type": "done",  "answer": "...", "sources": [...]}
        """
        graph = self._get_agent_graph(request.agent_type)
        state = self._build_state(request)
        config = {"configurable": {"thread_id": self._thread_id(request)}}

        async for mode, chunk in graph.astream(state, config, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield {"type": "token", "data": str(chunk)}
                continue
            for node, update in chunk.items():
                yield {"type": "node", "name": node, "data": update or {}}
                if isinstance(update, dict) and update.get("answer"):
                    yield {
                        "type": "done",
                        "answer": update.get("answer"),
                        "sources": update.get("sources") or [],
                        "mode": update.get("answer_mode"),
                    }

    async def stream_auto(self, request: AgentRequest) -> AsyncIterator[dict[str, Any]]:
        """自动模式的流式执行：先推路由结果，再流式跑对应 Agent，必要时升级。

        事件序列：
            {"type": "route", ...}           选了哪个 Agent、理由
            ...被选 Agent 的 node / token 事件...
            {"type": "escalate", ...}        qa 没查到，转 research（仅此情况）
            ...research 的 node / token 事件...
        """
        from agents.supervisor.router import needs_escalation, route

        decision = await route(request.user_message, request.tenant_id)
        yield {
            "type": "route",
            "agent": decision.agent,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

        # 需要确认时只推一个 confirm 事件，不跑 Agent
        if decision.need_confirm:
            yield {
                "type": "confirm",
                "agent": decision.agent,
                "entry_url": decision.entry_url,
                "goal": decision.goal or request.user_message,
                "question": decision.confirm_question,
            }
            yield {"type": "final", "answer": decision.confirm_question, "mode": "confirm"}
            return

        target = AgentType(decision.agent)
        routed_request = request.model_copy(update={"agent_type": target})
        if target == AgentType.QA:
            routed_request.context["force_professional"] = True
        elif target == AgentType.CRAWLER:
            routed_request.context["entry_url"] = decision.entry_url
            routed_request.context["goal"] = decision.goal or request.user_message

        final_mode = ""
        final_answer = ""
        async for event in self.stream(routed_request):
            if event["type"] == "done":
                final_mode = str(event.get("mode") or "")
                final_answer = str(event.get("answer") or "")
            yield event

        if target == AgentType.QA and needs_escalation(final_mode):
            logger.info("orchestrator.stream_escalating", to="web evidence")
            yield {"type": "escalate", "from": "qa", "to": "web", "reason": "知识库无结果"}

            # 流式里也要走同一套闸门与统一作答，避免和 handle_auto 行为不一致
            base = AgentResponse(
                success=True,
                agent_type=AgentType.QA,
                content=final_answer,
                structured={"answer_mode": final_mode},
            )
            async for event in self._stream_escalation(request, base):
                yield event
        elif final_answer:
            yield {"type": "final", "answer": final_answer, "mode": final_mode}

    async def _stream_escalation(
        self, request: AgentRequest, base: AgentResponse
    ) -> AsyncIterator[dict[str, Any]]:
        """边生成边把 compose 的 token 转发出去。

        compose 在图之外执行，拿不到 LangGraph 的 stream writer，
        所以用"回调写队列 + 主协程读队列"的方式把它接回事件流。
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def on_token(text: str) -> None:
            await queue.put(("token", text))

        async def produce() -> None:
            try:
                result = await self._escalate_to_web(request, base, on_token=on_token)
                await queue.put(("result", result))
            except Exception as exc:  # 兜底：别让升级失败把流挂住
                logger.error("orchestrator.escalation_failed", error=str(exc))
                await queue.put(("error", str(exc)))
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                kind, payload = item
                if kind == "token":
                    yield {"type": "token", "data": payload}
                elif kind == "result":
                    structured = payload.structured or {}
                    yield {
                        "type": "done",
                        "answer": payload.content,
                        "sources": structured.get("sources") or [],
                        "mode": structured.get("answer_mode") or "web_enhanced",
                    }
                    yield {
                        "type": "final",
                        "answer": payload.content,
                        "mode": structured.get("answer_mode") or "web_enhanced",
                    }
                elif kind == "error":
                    yield {"type": "error", "message": payload}
        finally:
            await task

    async def _run_single_agent(self, request: AgentRequest) -> AgentResponse:
        """
        单 Agent 直达模式：直接调用目标 Agent 的 LangGraph 图。

        读取 request：agent_type / user_message / user_id / tenant_id / session_id / context
        返回：AgentResponse（从 Agent 图最终 State 提取 content + structured + fallback_used）
        使用 with_retry 包装，自动处理重试和降级。
        """
        graph = self._get_agent_graph(request.agent_type)  # 懒加载取出目标 Agent 图

        # 构建 LangGraph 输入 State（统一三件套 + context 展开）
        initial_state = self._build_state(request)

        config = {  # LangGraph 运行配置
            "configurable": {
                "thread_id": self._thread_id(request),  # 每次请求独立 thread
            }
        }

        @with_retry(
            agent_type=request.agent_type.value
        )  # 给本次调用套上三层兜底（重试→降级→系统兜底）
        async def _invoke(state: dict, config: dict) -> Any:
            # 参数化是为了让降级层能"换参数重跑"（如 research 的 deep→fast）
            return await graph.ainvoke(state, config=config)

        result_state = await _invoke(initial_state, config)  # 执行（失败时自动兜底）

        # 降级/系统兜底时返回的是普通 dict（不是图状态），单独处理：
        # 它没有 messages 字段，硬取会 KeyError / IndexError
        if isinstance(result_state, dict) and result_state.get("fallback_used"):
            reason = str(result_state.get("degrade_reason") or "")
            return AgentResponse(
                success=True,  # 兜底也算"给到了响应"
                agent_type=request.agent_type,
                content=result_state.get("content", ""),
                structured=result_state.get("structured_output"),
                fallback_used=True,
                degrade_reason=reason,
                metadata={
                    "degrade_reason": reason,
                    "system_fallback": bool(result_state.get("system_fallback")),
                },
            )

        messages = result_state.get("messages") or []
        if not messages:  # Agent 没产出任何消息
            return AgentResponse(
                success=False,
                agent_type=request.agent_type,
                error_msg="Agent 未返回任何消息",
            )

        # 从最终 State 提取响应内容：取最后一条消息的文本
        last_message = messages[-1]
        content = (
            str(last_message.text) if hasattr(last_message, "text") else str(last_message.content)
        )

        metadata: dict[str, Any] = {}
        # research 图会把联网检索状态写进 state，透出来让调用方分得清"搜失败"和"没搜到"
        if result_state.get("search_status"):
            metadata["web_status"] = str(result_state["search_status"])
        if result_state.get("time_range"):
            metadata["time_range"] = str(result_state["time_range"])

        return AgentResponse(
            success=True,
            agent_type=request.agent_type,
            content=content,  # 主文本响应
            structured=result_state.get("structured_output"),  # 结构化数据（报告/评分，若有）
            fallback_used=result_state.get("fallback_used", False),  # 是否走了降级
            metadata=metadata,
        )

    async def _run_pipeline(self, request: AgentRequest) -> PipelineResult:
        """
        多 Agent 串联 Pipeline 模式。

        读取 request：context["pipeline_key"]（选哪条 Pipeline）、student_id 等
        返回：PipelineResult（steps=各步 AgentResponse 列表）

        前序 Agent 的 structured_output 自动注入后序 Agent 的 context，
        已完成步骤的结果持久化保存，失败不丢弃前序成果。
        """
        # 根据 context 中的 pipeline_key 选择对应 Pipeline（默认求职全链路）
        pipeline_key = request.context.get("pipeline_key", "job_preparation")

        if pipeline_key not in self._pipelines:  # 未知 key → 抛 PipelineError（来自 3.3）
            raise PipelineError(f"未知 Pipeline 类型: {pipeline_key}")

        agent_sequence = self._pipelines[pipeline_key]  # 取出该 Pipeline 的 Agent 执行顺序
        result = PipelineResult()  # 准备收集各步结果

        # 初始上下文：从请求的 context 拷一份，后续逐步往里塞前序结果
        current_context = dict(request.context)

        for idx, agent_type in enumerate(agent_sequence):  # 按序执行每个 Agent
            # 为这一步构造独立的 AgentRequest（session_id 加 _step{idx} 后缀，避免检查点串台）
            step_request = AgentRequest(
                user_id=request.user_id,
                tenant_id=request.tenant_id,
                session_id=f"{request.session_id}_step{idx}",
                agent_type=agent_type,
                user_message=request.user_message,
                context=current_context,  # 带上累积的上下文（含前序结果）
                pipeline_mode=False,  # 单步内不再触发 Pipeline，防止递归
            )

            logger.info(
                "orchestrator.pipeline_step_start",
                step=idx + 1,
                total=len(agent_sequence),
                agent_type=agent_type.value,
            )

            step_response = await self._run_single_agent(step_request)  # 复用单 Agent 直达跑这一步
            result.steps.append(step_response)  # 记录本步结果

            if not step_response.success:  # 本步失败 → 保留已完成成果，终止后续
                logger.warning(
                    "orchestrator.pipeline_step_failed",
                    step=idx + 1,
                    agent_type=agent_type.value,
                    error=step_response.error_msg,
                )
                break  # 不再跑后面的步骤（但前面的结果已保存）

            # ★ 上下文传递：把本步的结构化输出注入累积上下文，供下一步使用
            if step_response.structured:
                current_context[f"{agent_type.value}_result"] = step_response.structured
                logger.info(
                    "orchestrator.pipeline_context_passed",
                    from_agent=agent_type.value,
                    keys=list(step_response.structured.keys()),
                )

        return result  # 返回各步聚合前的原始结果

    def _aggregate_pipeline(
        self,
        pipeline_result: PipelineResult,  # 各步结果
        request: AgentRequest,  # 原始请求
    ) -> AgentResponse:
        """
        聚合 Pipeline 各步骤结果，返回统一 AgentResponse。

        读取 pipeline_result.steps（各步 AgentResponse）
        返回 AgentResponse：content=各步文本拼接，structured=combined（含每步明细）
        combined 字段包含所有步骤的结构化数据。
        """
        combined = {}  # 汇总每步的结构化明细
        all_contents = []  # 收集每步的文本，最后拼接
        any_success = False  # 只要有一步成功，整体就算部分成功

        for idx, step in enumerate(pipeline_result.steps):
            step_key = f"step_{idx + 1}"  # step_1 / step_2 ...
            combined[step_key] = {  # 记录这一步的元信息
                "agent_type": step.agent_type.value,
                "success": step.success,
                "structured": step.structured,
            }
            if step.success:
                any_success = True
                if step.content:
                    all_contents.append(step.content)  # 成功且有文本 → 收进待拼接列表

        return AgentResponse(
            success=any_success,  # 任一步成功即 True
            agent_type=request.agent_type,
            content="\n\n---\n\n".join(all_contents),  # 各步文本用分隔线拼成一段
            structured=combined,  # 结构化里带每步明细
            fallback_used=any(s.fallback_used for s in pipeline_result.steps),  # 任一步降级即标记
        )


# ──────────────────────────────────────────────────────────────
# 模块级单例（应用生命周期内复用）
# ──────────────────────────────────────────────────────────────
_orchestrator_instance: Orchestrator | None = None  # 全局唯一实例（初始为空）


def get_orchestrator() -> Orchestrator:
    """获取 Orchestrator 单例（FastAPI 依赖注入使用）"""
    global _orchestrator_instance
    if _orchestrator_instance is None:  # 第一次调用才创建
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance  # 之后都返回同一个实例
