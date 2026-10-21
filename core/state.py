"""共享契约：主图与子 Agent 之间唯一的数据交换结构。

只允许依赖 pydantic / langgraph，不 import 项目内任何模块（防止 import 成环）。
当前只有字段声明，没有逻辑；字段名和类型都可以改。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import AnyMessage

AgentName = Literal["researcher", "coder", "FINISH"]
"""路由目标。FINISH 不是子 Agent，是"收工"这个动作。"""


class SupervisorState(TypedDict, total=False):
    """主图的状态。所有节点只读它、返回它的增量。"""

    messages: list[AnyMessage]  # 对话历史
    next_agent: AgentName  # 下一步派给谁（与 SupervisorDecision.next_agent 同名）
    task_instruction: str  # 派给子 Agent 的任务描述
    results: list[dict]  # 子 Agent 回执（是否改成累加 reducer 待定）
    errors: list[str]  # 异常回灌
    steps: int  # 轮次计数，防死循环
