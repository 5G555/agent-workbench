"""前置路由层的行为测试。

三件事必须成立：
1. 不检索的路径真的不碰检索组件（help / direct 模块里不能出现 MCP 检索客户端）
2. 灰区低置信度必须回到 direct
3. 阈值兜底函数本身要正确
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.supervisor.router import apply_confidence_guard, route

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("agent", ["help", "direct"])
def test_no_retrieval_paths_do_not_touch_search(agent: str) -> None:
    """help / direct 的实现里不允许出现检索客户端。

    这是"路由层在检索插件之前生效"的静态保证——
    比运行时断言更可靠：代码里没这个 import，就永远不可能发网络请求。
    """
    module_dir = ROOT / "agents" / agent
    forbidden = ("mcp_client", "web_search_server", "research.collector", "research.nodes")
    for path in module_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} 引用了检索组件 {token}"


async def test_capability_question_marks_no_retrieval() -> None:
    resolved = await route("你可以做什么")
    assert resolved.agent == "help"
    assert resolved.no_retrieval is True


async def test_method_question_marks_no_retrieval() -> None:
    resolved = await route("K 线怎么画")
    assert resolved.agent == "direct"
    assert resolved.no_retrieval is True


async def test_research_route_allows_retrieval() -> None:
    resolved = await route("今天天气怎么样")
    assert resolved.agent == "research"
    assert resolved.no_retrieval is False


@pytest.mark.parametrize(
    ("agent", "confidence", "expected"),
    [
        ("research", 0.95, "research"),  # 高置信度保留
        ("research", 0.5, "direct"),  # 低置信度回 direct
        ("qa", 0.3, "direct"),
        ("help", 0.2, "help"),  # 不检索的路径不受阈值影响
        ("direct", 0.1, "direct"),
    ],
)
def test_confidence_guard(agent: str, confidence: float, expected: str) -> None:
    final, reason = apply_confidence_guard(agent, confidence, 0.6)
    assert final == expected
    if expected != agent:
        assert "低于阈值" in reason
