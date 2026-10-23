"""Supervisor：判断一个问题该交给哪个 Agent，并决定要不要先问用户确认。

三个去处：
    qa        内部知识库（私有、需权限）
    research  公开互联网（时效、外部事实）
'    crawler   主动采集指定网页（抓数据落库，代价最大）
    ops       管理已采集的任务（查任务 / 导出文件 / 清理导出目录）'

为什么要"先确认"：
    qa 和 research 都是只读、秒级、无副作用；而 crawler 会真的去访问外部站点、
    占用带宽、往库里写数据，还可能因为入口给错而白跑一轮。
    所以凡是"要动手做什么"的意图，先确认一次比直接执行更合适。

确认规则（代码判断，不交给模型）：
    1. 提到抓取/爬取/采集，且能识别出 URL  → 直接执行（意图明确，动手即所得）
    2. 提到抓取但没给 URL                  → 需要确认（否则不知道抓哪儿）
    3. "我想分析竞品价格"这类隐含采集的意图 → 需要确认（代价大，先对齐）
模型只负责判断"这是不是采集意图"，确认与否由这里的规则决定，便于测试。
"""

from __future__ import annotations

import re
from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.help.nodes import is_capability_intent
from agents.qa.nodes import kb_topics
from agents.supervisor.evidence import (
    MUST,
    NEVER,
    has_live_signal,
    has_method_signal,
    is_private_scope,
    needs_evidence,
)
from config import get_settings
from core.llm_factory import LLMFactory
from core.logger import get_logger

# 路由可选的 Agent（direct = 不调工具，直接让模型答）
AgentName = Literal["qa", "research", "crawler", "ops", "analyst", "metrics", "help", "direct"]

logger = get_logger(__name__)

ROUTE_SYSTEM = "你是一个请求路由器，只输出 JSON，不解释。"

ROUTE_USER = """系统里有三个 Agent，按"要做的事"分派：

A. qa（内部 RAG）：资料范围是——{topics}
   适用于：内部资料、企业私有信息、需要权限控制、必须给官方口径的问题。

B. research（联网搜索）：去公开互联网检索并给带引用的答案。
   适用于：公开可查、有时效性、外部事实（框架文档、行业动态、新闻、产品对比）。

C. crawler（网页采集）：主动抓取指定页面并落库，供后续分析。
   适用于：明确要求"抓取/爬取/采集某个网站的数据"、"把这些页面存下来"。

判断顺序：
1. 要动手采集网页数据 → crawler
2. 要查本企业内部资料 → qa
3. 要查公开信息 → research
4. 拿不准时选 qa（本地答错还能升级到联网）

用户问题：{query}

只返回 JSON：
{{"agent": "qa" 或 "research" 或 "crawler", "confidence": 0~1, "reason": "一句话理由"}}"""

# 采集意图的关键词
CRAWL_KEYWORDS = (
    "抓取",
    "爬取",
    "采集",
    "爬虫",
    "抓下来",
    "抓包",
    "存下来",
    "入库",
    "crawl",
    "scrape",
    "spider",
)
# 隐含采集：想拿"别人的页面数据"做对比分析，但没明说要抓
IMPLICIT_CRAWL_HINTS = ("竞品", "同行", "对标", "他们网站", "官网数据")

# 运维意图关键词：查任务、导出、清理这些"动作"不值得过一遍模型
OPS_KEYWORDS = (
    "回收站",
    "清理",
    "清掉",
    "删除导出",
    "删掉导出",
    "导出",
    "任务列表",
    "最近的爬取",
    "爬取任务",
    "采集任务",
    "抓取任务",
    "导出目录",
    "任务记录",
    "缓存",
    "有哪些任务",
)

# 系统指标意图：问的是"系统自己跑得怎么样"（会话、命中率、采集量）
METRICS_KEYWORDS = (
    "会话数",
    "消息数",
    "命中率",
    "满意度",
    "点踩",
    "点赞",
    "反馈",
    "多少会话",
    "多少消息",
    "多少个会话",
    "多少条消息",
    "任务数",
    "采集量",
    "增长率",
    "活跃",
    "用量",
    "成本",
)

# 数据分析意图：问的是"多少个/占比/趋势/排名/分布"这类量化问题
ANALYST_KEYWORDS = (
    "统计",
    "占比",
    "趋势",
    "排名",
    "分布",
    "多少个",
    "多少",
    "多少次",
    "多少条",
    "命中率",
    "满意度",
    "平均",
    "总数",
    "增长率",
    "top",
    "排行",
    "排前",
    "前五",
    "前十",
    "次数",
    # 图表类型：说出"K 线/柱状图"通常就是要在采集数据上出图
    "k 线",
    "k线",
    "图表",
    "柱状图",
    "散点图",
    "饼图",
    "直方图",
    "双轴",
    "箱线图",
)

# 问"多少钱"是问价格，不是分析
PRICE_HINTS = ("钱", "价格", "费用", "收费", "报价", "贵")

# "分析的对象是我们采集到的数据"这类说法
CRAWL_DATA_HINTS = (
    "抓取到",
    "采集到",
    "抓到的",
    "爬取到",
    "采集的数据",
    "抓取的数据",
    "这批数据",
    "这些数据",
    "数据里",
    "数据中",
)

URL_PATTERN = re.compile(r"https?://[^\s，。；、）)\"']+")


def is_metrics_intent(text: str) -> bool:
    """是不是在问系统自身的运行指标（归运维的可观测性）。"""
    lowered = (text or "").lower()
    if not lowered:
        return False
    # 带 URL 的是去抓，不算指标；
    # 但含"任务数/采集量"这类指标词时（哪怕提到采集）仍是指标，
    # 否则"采集任务数是多少"会被运维动作抢走
    if detect_url(text):
        return False
    return any(k in lowered for k in METRICS_KEYWORDS)


def is_analyst_intent(text: str) -> bool:
    """是不是在问数据统计/分析。

    和运维的区别：运维是删/清/导出，分析是多少/占比/趋势。
    和采集的区别：采集要 URL，分析不要。
    """
    lowered = (text or "").lower()
    if not lowered or not any(k in lowered for k in ANALYST_KEYWORDS):
        return False
    # "多少钱"是问价格，归 qa
    if any(h in lowered for h in PRICE_HINTS):
        return False
    # 带 URL 的是去抓，不是分析（抓完再分析由 crawler/chain 处理）
    if detect_url(text):
        return False
    # 实时/易变事实（天气、股价…）不归分析，除非明确在问采集到的数据
    if has_live_signal(text) and not any(h in lowered for h in CRAWL_DATA_HINTS):
        return False
    # "K 线怎么画"是问方法，不是让我分析数据
    if has_method_signal(text) and not any(h in lowered for h in CRAWL_DATA_HINTS):
        return False
    # 提到抓取到的数据这类说法时仍归分析；只是抓取一下归 crawler
    return not mentions_crawl(text) or any(h in lowered for h in CRAWL_DATA_HINTS)


def is_ops_intent(text: str) -> bool:
    """是不是在让我管理采集任务（而不是提问或采集新页面）。

    排除法很重要："抓取任务列表"是运维，"抓取这个网页"是采集。
    所以带 URL 的采集请求归 crawler，不归运维。
    """
    lowered = (text or "").lower()
    if not lowered:
        return False
    if not any(keyword in lowered for keyword in OPS_KEYWORDS):
        return False
    return not detect_url(text)


def detect_url(text: str) -> str:
    """从自然语言里抽出第一个 URL；没有返回空串。"""
    match = URL_PATTERN.search(text or "")
    return match.group(0).rstrip(".,;") if match else ""


def mentions_crawl(text: str) -> bool:
    """是否明确提到采集动作。"""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in CRAWL_KEYWORDS)


def implies_crawl(text: str) -> bool:
    """是否隐含采集意图（想要别人的页面数据但没明说）。"""
    lowered = text or ""
    return any(hint in lowered for hint in IMPLICIT_CRAWL_HINTS)


class RouteDecision(BaseModel):
    """路由结果。"""

    agent: AgentName = Field(default="qa")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="")


class ResolvedRoute(RouteDecision):
    """路由结果。no_retrieval=True 表示这条路径不接触任何检索组件。"""

    no_retrieval: bool = False
    """在模型判断之上补齐"要不要确认"与采集参数。"""

    need_confirm: bool = False
    confirm_question: str = ""
    entry_url: str = ""
    goal: str = ""


def apply_confidence_guard(agent: str, confidence: float, threshold: float) -> tuple[str, str]:
    """灰区兜底：置信度不够就回到 direct。

    返回 (最终 agent, 原因)。这层单独抽出来是为了能单测——
    "低置信度该走哪"是路由里最容易写错又不显眼的地方。
    """
    if agent in {"help", "direct"}:
        return agent, "无需检索"
    if confidence < threshold:
        return "direct", f"置信度 {confidence:.2f} 低于阈值 {threshold:.2f}，改直接回答"
    return agent, ""


def resolve_confirmation(decision: RouteDecision, query: str) -> ResolvedRoute:
    """按规则决定是否需要用户确认，并抽出采集参数。

    纯函数：不调模型、不碰 IO，方便单测。
    """
    resolved = ResolvedRoute(**decision.model_dump())
    url = detect_url(query)

    if decision.agent != "crawler":
        # 非采集意图不确认；但"隐含采集"的请求值得问一句
        if implies_crawl(query) and not mentions_crawl(query):
            resolved.need_confirm = True
            resolved.confirm_question = (
                "你是想让我去抓取相关网站的数据做分析吗？如果是，请把入口网址发我。"
            )
        return resolved

    resolved.entry_url = url
    resolved.goal = query.strip()

    if not url:
        resolved.need_confirm = True
        resolved.confirm_question = (
            "我需要一个入口网址才能开始抓取（例如 https://example.com/list）。"
            "把网址发我，我就按你的目标去抓。"
        )
        return resolved

    # 有 URL 且明确说要抓 → 意图明确，直接执行
    if mentions_crawl(query):
        return resolved

    resolved.need_confirm = True
    resolved.confirm_question = f"确认要去抓取 {url} 吗？我会抓取若干页面并把结构化记录存进数据库。"
    return resolved


async def route(query: str, tenant_id: str | None = None) -> ResolvedRoute:
    """按闸门顺序给问题选一个 Agent。

    设计原则（改自旧版的一锤定音）：
        1. 用户说得很具体的（URL、破坏性动作、指标词）用规则锁死，规则比模型准
        2. 需要外部证据的按 evidence 判定走 research
        3. 方法/生成/解释类直接让模型答，不检索
        4. 灰度交给模型，但**默认 direct**——误检索的代价远大于误直接答
    """
    if not query.strip():
        return ResolvedRoute(agent="qa", confidence=1.0, reason="空问题")

    # 1. 能力自述：问的是本系统，不能交给联网检索
    if is_capability_intent(query):
        return _decide(query, "help", "能力咨询")

    # 2. 私有资料（"我们/公司/报价"）：查内部知识库，不是外部事实
    if is_private_scope(query) and not mentions_crawl(query):
        return _decide(query, "qa", "私有资料")

    # 3. 带 URL 且明确要采集
    if detect_url(query) and mentions_crawl(query):
        return _decide(query, "crawler", "URL + 采集动作", need_confirm=True)

    # 4. 系统指标（先看是不是在问指标：指标问题带"任务/采集"字样时别被运维抢走）
    if is_metrics_intent(query):
        return _decide(query, "metrics", "系统指标")

    # 5. 运维动作（要动手删/清/导出）
    if is_ops_intent(query):
        return _decide(query, "ops", "运维动作")

    # 6. 采集数据分析
    if is_analyst_intent(query):
        return _decide(query, "analyst", "采集数据分析")

    # 7/8. 证据判定（明确要查 / 明确不用查）
    evidence = needs_evidence(query)
    if evidence == MUST:
        return _decide(query, "research", "需要实时外部事实")
    if evidence == NEVER:
        return _decide(query, "direct", "方法/生成/解释类，无需检索")

    # 9. 灰度：交给模型判断"需不需要工具"，判不准就 direct
    decision = await _route_llm(query, tenant_id)
    if decision is None:
        return _decide(query, "direct", "路由失败，默认直接回答")

    settings = get_settings()
    final_agent, guard_reason = apply_confidence_guard(
        decision.agent, decision.confidence, settings.route_confidence_threshold
    )
    if guard_reason and final_agent != decision.agent:
        decision.agent = cast(AgentName, final_agent)
        decision.reason = guard_reason

    resolved = resolve_confirmation(decision, query)
    logger.info(
        "supervisor.routed",
        query=query,
        agent=resolved.agent,
        confidence=round(resolved.confidence, 2),
        need_confirm=resolved.need_confirm,
        evidence=evidence,
        reason=resolved.reason,
    )
    return resolved


def _decide(
    query: str,
    agent: str,
    reason: str,
    *,
    need_confirm: bool = False,
) -> ResolvedRoute:
    """统一出口：打日志 + 组装 ResolvedRoute，避免每个闸门重复写。"""
    resolved = ResolvedRoute(
        agent=cast(AgentName, agent),
        confidence=1.0,
        reason=reason,
        need_confirm=need_confirm,
        no_retrieval=agent in {"help", "direct"},
    )
    logger.info(
        "supervisor.routed",
        query=query,
        agent=agent,
        need_confirm=need_confirm,
        reason=reason,
    )
    return resolved


async def _route_local(query: str) -> RouteDecision | None:
    """本地模型判断；置信度不够返回 None 交给大模型。"""
    from agents.supervisor.local_router import get_intent_router, should_use_local

    settings = get_settings()
    router = get_intent_router(settings)
    if router is None:
        return None

    try:
        agent, confidence = await router.apredict(query)
    except Exception as exc:
        logger.warning("supervisor.local_route_failed", error=str(exc))
        return None

    if not should_use_local(confidence, settings.intent_confidence_threshold):
        logger.info(
            "supervisor.local_low_confidence",
            query=query,
            agent=agent,
            confidence=round(confidence, 3),
        )
        return None

    return RouteDecision(
        agent=cast(AgentName, agent),
        confidence=confidence,
        reason="本地意图模型",
    )


async def _route_llm(query: str, tenant_id: str | None = None) -> RouteDecision:
    """大模型判断：意图模糊、或本地模型不可用时的兜底。"""
    try:
        topics = await kb_topics(tenant_id)
        llm = LLMFactory.get_structured_llm("qa", RouteDecision)
        decision = await llm.ainvoke(
            [
                SystemMessage(content=ROUTE_SYSTEM),
                HumanMessage(content=ROUTE_USER.format(topics=topics, query=query)),
            ]
        )
        if not isinstance(decision, RouteDecision):
            raise ValueError("路由输出类型不符")
        decision.reason = f"{decision.reason}（大模型）"
        return decision
    except Exception as exc:
        logger.warning("supervisor.route_failed", error=str(exc))
        return RouteDecision(agent="qa", confidence=0.0, reason=f"路由失败，默认 qa：{exc}")


def needs_escalation(answer_mode: str | None, sources: list[str] | None = None) -> bool:
    """qa 的答案是否"不够用"，需要转 research。

    只在 qa 真的查过知识库但没查到（no_result）时升级；faq/rag 已有依据不升级。
    """
    return answer_mode == "no_result"
