"""路由闸门的测试（只测确定性部分，不触发模型调用）。"""

from __future__ import annotations

import pytest

from agents.supervisor.router import route


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # 能力自述：不能去搜
        ("你可以做什么", "help"),
        ("你能干什么", "help"),
        ("你是谁", "help"),
        # 私有资料：查内部知识库
        ("你们怎么收费", "qa"),
        ("公司内部流程怎么走", "qa"),
        # 明确的采集动作
        ("抓取 https://example.com 的表格", "crawler"),
        # 破坏性动作
        ("把旧版本缓存清掉", "ops"),
        # 系统指标
        ("一共有多少会话", "metrics"),
        # 采集数据分析
        ("抓取到的数据里作者分布如何", "analyst"),
        # 实时事实
        ("今天天气怎么样", "research"),
        ("AAPL 现在多少钱", "research"),
        ("2026 年向量数据库怎么选", "research"),
        # 方法/生成类：不检索
        ("K 线怎么画", "direct"),
        ("帮我写一封道歉邮件", "direct"),
        ("什么是向量检索", "direct"),
        ("怎么查看股价", "direct"),
    ],
)
async def test_route_gates(question: str, expected: str) -> None:
    resolved = await route(question)
    assert resolved.agent == expected, f"{question} → {resolved.agent}（期望 {expected}）"
