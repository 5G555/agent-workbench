"""运维意图识别 + 爬虫 MCP 工具注册的测试。"""

from __future__ import annotations

import pytest

from agents.supervisor.router import is_ops_intent


@pytest.mark.parametrize(
    "query",
    [
        "列出最近的爬取任务",
        "把任务 3 的导出清掉",
        "导出任务 5 的文件",
        "清理一下导出目录",
        "有哪些任务记录",
        "最近采集任务都成功了吗",
    ],
)
def test_ops_intent_detected(query):
    assert is_ops_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "帮我抓取 https://example.com/list 的数据",  # 带 URL → 采集，不是运维
        "你们怎么收费",
        "Milvus 2.4 支持哪些索引类型",
        "什么是 RAG",
    ],
)
def test_non_ops_intent(query):
    assert is_ops_intent(query) is False


def test_empty_query_is_not_ops():
    assert is_ops_intent("") is False
    assert is_ops_intent("   ") is False


def test_crawler_mcp_server_registers_expected_tools():
    """工具名是模型能看到的契约，改名会直接影响调用，固定住。"""
    from mcp_servers import crawler_server

    names = {tool.name for tool in crawler_server.mcp._tool_manager.list_tools()}
    assert names == {
        "crawl",
        "list_tasks",
        "task_records",
        "export_files",
        "clean_exports",
    }


def test_clean_tool_is_dry_run_by_default():
    """清理工具默认必须不删（apply 默认 False），这是安全底线。"""
    import inspect

    from mcp_servers import crawler_server

    signature = inspect.signature(crawler_server.clean_exports)
    # FastMCP 的 @mcp.tool() 保留原函数签名
    assert signature.parameters["apply"].default is False
