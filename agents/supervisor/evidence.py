"""判断一个问题"需不需要外部证据"。

为什么单独抽一层：
    路由最容易错的就是这一环——模型对自己"知不知道"的校准很差，
    问它"你需要联网吗"，它常答"不需要"，然后编一个过期数字。
    所以先用确定性规则把明显的情况拿走，只把灰度留给模型。

三档结论：
    must  必须查（实时数据、易变实体属性）
    never 不必查（方法论、生成类、解释类、自指类）
    gray  判不了（交给 answer-then-flag：先答，标出拿不准的事实点）

判定顺序：方法/生成/解释类优先——
    "怎么查看股价"是方法，不该因为它含"股价"就去搜。
"""

from __future__ import annotations

import re

MUST = "must"
NEVER = "never"
GRAY = "gray"

# 闲聊词：命中直接归 direct，哪怕含"今天"这种时点词
CHITCHAT_WORDS = (
    "心情",
    "开心",
    "难过",
    "累了",
    "好累",
    "无聊",
    "哈哈",
    "嘿嘿",
    "在吗",
    "你好",
    "早上好",
    "晚上好",
    "谢谢你",
    "多谢",
    "辛苦了",
    "讲个笑话",
    "陪我聊",
    "聊天",
)

# 实时/时点类词：命中基本就该查
LIVE_WORDS = (
    "天气",
    "气温",
    "股价",
    "行情",
    "汇率",
    "比分",
    "赛果",
    "新闻",
    "热搜",
    "实时",
    "今天",
    "昨天",
    "刚刚",
    "最新",
    "当前",
    "现任",
    "目前",
    "今年",
    "本月",
    "上个月",
    "截至",
    "多少钱",
    "几个人",
)

# 方法论/生成/解释/自指：命中就不该查
METHOD_WORDS = (
    "怎么",
    "如何",
    "为什么",
    "为何",
    "区别",
    "差异",
    "优缺点",
    "原理",
    "步骤",
    "思路",
    "建议",
    "帮我画",
    "画个",
    "画一张",
    "画一幅",
    "帮我写",
    "帮我改",
    "帮我润色",
    "帮我总结",
    "翻译",
    "润色",
    "改写",
    "总结一下",
    "起个名",
    "列个清单",
    "是什么",
    "什么是",
    "什么意思",
    "举个例子",
    "通俗解释",
    "解释一下",
    "你能做什么",
    "你是谁",
    "使用说明",
    "怎么用",
    "如何用",
)

# 选型/对比类："怎么选"看起来像方法，实际需要外部横向信息
COMPARISON_WORDS = (
    "怎么选",
    "如何选",
    "选哪个",
    "哪个好",
    "哪家好",
    "推荐哪个",
    "对比一下",
    "有什么推荐",
)

# 易变实体的属性问法：X 的 CEO / 价格 / 市值 是多少
ATTRIBUTE_PATTERN = re.compile(
    r"(?:"
    + "|".join(["ceo", "价格", "市值", "库存", "版本号", "负责人", "法定代表人", "状态"])
    + r")"
    r"\s*(?:是|为|多少|谁|什么|几)",
    re.I,
)

# 指向本系统私有知识（不是外部事实）
PRIVATE_SCOPE_WORDS = (
    "我们",
    "你们",
    "贵公司",
    "贵方",
    "咱们",
    "本公司",
    "公司",
    "贵司",
    "内部",
    "产品",
    "报价",
    "收费",
    "费用",
    "价格表",
    "发票",
    "退款",
    "售后",
    "合同",
    "内部政策",
    "公司制度",
    "内部流程",
    "审批流程",
    "报销流程",
    "内部规范",
    "上面",
    "刚才",
    "这个系统",
)


def has_live_signal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in LIVE_WORDS)


# "怎么样 / 怎样" 是问评价，不是问方法——先摘掉再判，
# 否则 "今天天气怎么样" 会因为含 "怎么" 被判成方法类
EVALUATION_PATTERN = re.compile(r"(怎么样|怎样|咋样)")


def has_method_signal(text: str) -> bool:
    lowered = EVALUATION_PATTERN.sub("", (text or "").lower())
    return any(word in lowered for word in METHOD_WORDS)


def has_attribute_signal(text: str) -> bool:
    return bool(ATTRIBUTE_PATTERN.search(text or ""))


def is_private_scope(text: str) -> bool:
    """是不是在问"我们自己的资料"——这类要查内部知识库，不是外部事实。"""
    lowered = (text or "").lower()
    return any(word in lowered for word in PRIVATE_SCOPE_WORDS)


def needs_evidence(text: str) -> str:
    """返回 must / never / gray。

    顺序很重要：方法类优先于实时类。
        "怎么查看股价"  → never（教方法，不用查）
        "AAPL 股价多少" → must（要真数据）
    """
    question = (text or "").strip()
    if not question:
        return GRAY

    if any(word in question.lower() for word in CHITCHAT_WORDS):
        return NEVER
    if any(word in question.lower() for word in COMPARISON_WORDS):
        return MUST
    if has_method_signal(question):
        return NEVER
    if has_live_signal(question) or has_attribute_signal(question):
        return MUST
    return GRAY
