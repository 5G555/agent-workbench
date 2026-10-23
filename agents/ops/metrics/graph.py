"""系统指标问答的图。

START → metric（生成 SQL → 守卫 → 执行 → 解读）→ END

只有一步：指标问题不需要多路径编排，
失败自愈在节点内部完成（重试一次），不做成图上的循环。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.ops.metrics.nodes import metric_node
from agents.ops.metrics.state import MetricState


def build_metrics_graph(checkpointer: Any = None):
    builder = StateGraph(MetricState)
    builder.add_node("metric", metric_node)
    builder.add_edge(START, "metric")
    builder.add_edge("metric", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
