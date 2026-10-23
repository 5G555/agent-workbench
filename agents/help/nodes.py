"""能力问答：回答"你能做什么"这类关于系统自身的问题。

两个决定：
1. **路由是确定性的**：这类问题不能交给检索。实测本地意图模型把它判成
   research（置信度 1.00），联网搜回来一堆"职业规划""英语短语"，完全跑偏。
2. **答案是模型生成的**：能力介绍本质是"把已知事实说清楚"，这是大模型擅长的事。
   写死文案看似稳，实际会过期，也答不了"你能分析股票吗"这种追问。

模型拿到的是真实能力清单 + 当前租户数据，所以它说的是"现在真有的"。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.help.state import HelpState
from core.db import lease_connection
from core.llm_factory import LLMFactory
from core.logger import get_logger

logger = get_logger(__name__)

# 这些问法都指向"系统的能力边界"
CAPABILITY_PATTERNS = (
    "你可以做什么",
    "你能做什么",
    "你能干什么",
    "你能帮我做什么",
    "你会什么",
    "你有什么功能",
    "有哪些功能",
    "有什么能力",
    "支持哪些",
    "支持什么",
    "使用说明",
    "你会干嘛",
    "你是谁",
    "介绍一下你自己",
    "自我介绍",
    "帮助",
    "help",
)

# "怎么用" 单独匹配太宽（也可能问某个业务事项），要求整句就是这个意思
USAGE_PATTERN = re.compile(r"(怎么用|如何用|怎么使用|如何使用)(你|系统|这个系统|平台)?\s*[？?。]?$")


def is_capability_intent(text: str) -> bool:
    """是不是在问"你能做什么"。"""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if any(word in lowered for word in CAPABILITY_PATTERNS):
        return True
    return bool(USAGE_PATTERN.search(lowered))


SYSTEM_PROMPT = """你是这套系统的助手，正在向用户介绍你自己能做什么。

【真实能力】（以此为准，不要编造未上线的功能）
1. 内部知识问答：产品与业务问题先查内部知识库（FAQ + 向量检索），答不到才联网。
2. 联网研究：内部答不出时检索公开资料，答案带来源编号。
3. 网页采集：给一个网址，抓正文、抽表格/列表/段落，落库后可导出成文件。
4. 数据分析：对采集到的数据做预处理（转数值/转时间/去重），写 pandas 代码计算，
   并出图（折线 / K 线 / 柱状 / 散点 / 饼图 / 直方图 / 双轴），支持多个任务对比。
5. 运维操作：查系统指标（自然语言转 SQL）、看采集任务、清理导出目录、清理缓存、
   回收站恢复、审批与审计；破坏性操作需要管理员审批。

【表达要求】
- 用中文，分点作答，不要长篇大论。
- 每类给一句具体例子，让用户知道怎么说。
- 只描述上面列出的能力；用户问到没做的功能（例如"能发邮件吗"），直说没有。
- 不要提"路由""意图识别""分类模型"这类内部实现。"""

FALLBACK_ANSWER = (
    "我可以做这几类事：\n"
    "· 内部知识问答：产品/业务问题先查知识库\n"
    "· 联网研究：内部答不出时检索，答案带来源\n"
    "· 网页采集：给网址，抽表格/正文，可导出\n"
    "· 数据分析：pandas 计算 + 绘图，支持多任务对比\n"
    "· 运维操作：查指标、清缓存、回收站、审批审计"
)


async def _tenant_snapshot(tenant_id: str) -> dict[str, int]:
    """当前租户的数据现状，让回答落在真实数据上。"""
    snapshot = {"tasks": 0, "records": 0, "sessions": 0}
    conn = await lease_connection()
    try:
        snapshot["tasks"] = int(
            await conn.fetchval("SELECT count(*) FROM crawl_tasks WHERE tenant_id = $1", tenant_id)
            or 0
        )
        snapshot["records"] = int(
            await conn.fetchval(
                "SELECT count(*) FROM crawl_records r "
                "JOIN crawl_tasks t ON t.id = r.task_id WHERE t.tenant_id = $1",
                tenant_id,
            )
            or 0
        )
        snapshot["sessions"] = int(
            await conn.fetchval(
                "SELECT count(*) FROM qa_sessions s "
                "JOIN users u ON u.id::text = s.user_id WHERE u.tenant_id = $1",
                tenant_id,
            )
            or 0
        )
    except Exception as exc:  # 数据取不到也不影响能力说明
        logger.warning("help.snapshot_failed", error=str(exc)[:120])
    finally:
        await conn.close()
    return snapshot


async def capability_node(state: HelpState) -> dict[str, Any]:
    """让模型结合真实数据介绍能力；模型不可用时退回固定清单。"""
    tenant_id = str(state.get("tenant_id") or "tenant_default")
    role = str(state.get("role") or "user")
    question = str(state.get("question") or "").strip()

    if not question:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                question = str(message.content).strip()
                break

    snapshot = await _tenant_snapshot(tenant_id)
    context = (
        f"当前租户：{tenant_id}\n"
        f"用户角色：{'管理员' if role == 'admin' else '普通用户'}\n"
        f"该租户现有采集任务 {snapshot['tasks']} 个、结构化记录 {snapshot['records']} 条、"
        f"问答会话 {snapshot['sessions']} 个。\n"
        f"用户的问题：{question or '你可以做什么'}"
    )

    answer = ""
    try:
        llm = LLMFactory.get_llm("qa", temperature=0.3, streaming=True)
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]
        )
        answer = str(response.content).strip()
    except Exception as exc:
        logger.warning("help.llm_failed", error=str(exc)[:120])

    if not answer:
        answer = FALLBACK_ANSWER

    if role == "admin":
        answer += "\n\n（你是管理员：破坏性操作仍需审批，单人管理员租户可自批并留痕。）"

    logger.info("help.answered", tenant=tenant_id, length=len(answer))
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "answer_mode": "help",
        "structured_output": {
            "answer_mode": "help",
            "tenant_id": tenant_id,
            **snapshot,
        },
    }
