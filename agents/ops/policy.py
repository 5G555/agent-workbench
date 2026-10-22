"""确定性安全策略：判断一次工具调用能不能放行。

为什么不能让模型自己判断"该不该删"：
    模型的推理不可靠，上下文偏差、幻觉、提示注入都可能让它在参数里塞进
    一条越界删除。只要"是否该删"由模型决定，安全边界就是空的。

所以这里是一条纯函数链：
    AI 提议（工具名 + 参数）→ 本模块分级 → 执行网关（幂等/预算/快照/回滚）

规则是**声明式、表驱动、不可被模型说服**的：模型只能选择"提出请求"，
能不能执行由这张表决定。改表等于改安全边界，评审时只需要看这一处。

四级风险：
    LOW       只读，或写文件但可重新生成（导出、查任务）
    MEDIUM    单点、可逆的破坏（删一个任务的导出、清旧版本缓存）
    HIGH      批量破坏（按保留数/天数批量删导出）
    CRITICAL  不可逆或越界（未知工具、可疑参数、清库、purge）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(StrEnum):
    ALLOW = "allow"  # 直接执行
    SOFTEN = "soften"  # 降级：把参数改成更安全的形式（如 apply→false）
    REQUIRE_APPROVAL = "require_approval"  # 需要人工审批票据
    BLOCK = "block"  # 直接阻断（Fail-Closed）


@dataclass
class Verdict:
    """策略判定结果。"""

    decision: Decision
    risk: Risk
    reasons: list[str] = field(default_factory=list)
    # 降级后的参数（decision=SOFTEN 时使用）
    softened: dict[str, Any] | None = None
    # 需要几个审批（当前实现只支持 1，保留字段以便将来做双人复核）
    approvals_needed: int = 0

    @property
    def allowed(self) -> bool:
        return self.decision in (Decision.ALLOW, Decision.SOFTEN)


# ── 危险参数特征 ────────────────────────────────────────────
# 这些不是"业务参数"，出现任何一个都说明请求跑偏了
DANGEROUS_PATTERNS = [
    (re.compile(r"\brm\s+-rf\b", re.I), "包含 rm -rf"),
    (re.compile(r"\bdrop\s+(table|database)\b", re.I), "包含 DROP TABLE/DATABASE"),
    (re.compile(r"\btruncate\s+table\b", re.I), "包含 TRUNCATE TABLE"),
    (re.compile(r"\bdelete\s+from\b", re.I), "包含 DELETE FROM"),
    (re.compile(r"(^|[/\\])\.\.([/\\]|$)"), "包含路径穿越 .."),
    (re.compile(r"^(file|ftp)://", re.I), "包含非 http(s) 协议"),
    (re.compile(r"\b(169\.254\.169\.254|metadata\.google\.internal)\b", re.I), "指向云元数据地址"),
    (re.compile(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)", re.I), "指向本机地址"),
]

# 只读工具：无条件允许
READ_ONLY_TOOLS = {"list_tasks", "task_records", "cache_stats", "list_trash"}

# 可重新生成、失败也无害的写操作
REGENERABLE_TOOLS = {"export_files", "restore_exports", "restore_cache"}

# 会消耗外部资源的写操作（有确认流程，但不必审批）
BOUNDED_WRITE_TOOLS = {"crawl"}

# 破坏性工具：必须过策略表
DESTRUCTIVE_TOOLS = {"clean_exports", "clean_cache", "purge_trash"}


def _scan_args(args: dict[str, Any]) -> list[str]:
    """扫描参数里的危险特征。"""
    hits: list[str] = []
    for key, value in (args or {}).items():
        if not isinstance(value, str):
            continue
        for pattern, reason in DANGEROUS_PATTERNS:
            if pattern.search(value):
                hits.append(f"{key} {reason}")
    return hits


def _count_targets(args: dict[str, Any]) -> int:
    """估算这次要动多少资源（用于预算判断）。"""
    if args.get("task_id") is not None:
        return 1
    # 批量清理：keep=N 时至少动 (总数-N) 个，这里按上限保守估计
    return max(1, int(args.get("keep") or 0) + 1) if args.get("keep") else 2


def evaluate(tool: str, args: dict[str, Any] | None = None) -> Verdict:
    """判定一次调用。纯函数：同样的输入永远得到同样的结论。"""
    arguments = args or {}

    # ① 未知工具一律阻断（Fail-Closed）
    known = READ_ONLY_TOOLS | REGENERABLE_TOOLS | BOUNDED_WRITE_TOOLS | DESTRUCTIVE_TOOLS
    if tool not in known:
        return Verdict(
            decision=Decision.BLOCK,
            risk=Risk.CRITICAL,
            reasons=[f"未知工具 {tool}，默认拒绝"],
        )

    # ② 参数里出现危险特征 → 阻断
    hits = _scan_args(arguments)
    if hits:
        return Verdict(
            decision=Decision.BLOCK,
            risk=Risk.CRITICAL,
            reasons=hits,
        )

    # ③ 只读 / 可重生成 → 放行
    if tool in READ_ONLY_TOOLS or tool in REGENERABLE_TOOLS:
        return Verdict(decision=Decision.ALLOW, risk=Risk.LOW)

    if tool in BOUNDED_WRITE_TOOLS:
        return Verdict(decision=Decision.ALLOW, risk=Risk.MEDIUM, reasons=["有副作用但可重做"])

    # ④ 破坏性工具：没开 apply 就是干跑，直接放行
    if not arguments.get("apply"):
        return Verdict(
            decision=Decision.ALLOW,
            risk=Risk.LOW,
            reasons=["dry-run，不改变状态"],
        )

    # ⑤ 开了 apply：按影响面分级
    if tool == "purge_trash":
        return Verdict(
            decision=Decision.REQUIRE_APPROVAL,
            risk=Risk.HIGH,
            reasons=["清空回收站是不可逆操作"],
            approvals_needed=1,
        )

    if tool == "clean_cache":
        return Verdict(
            decision=Decision.REQUIRE_APPROVAL,
            risk=Risk.MEDIUM,
            reasons=["清理缓存键（会先备份到回收区）"],
            approvals_needed=1,
        )

    # clean_exports
    if arguments.get("task_id") is not None:
        return Verdict(
            decision=Decision.REQUIRE_APPROVAL,
            risk=Risk.MEDIUM,
            reasons=[f"删除任务 {arguments['task_id']} 的导出（会先进回收站）"],
            approvals_needed=1,
        )

    if arguments.get("keep") is not None or arguments.get("older_than_days") is not None:
        count = _count_targets(arguments)
        return Verdict(
            decision=Decision.REQUIRE_APPROVAL,
            risk=Risk.HIGH,
            reasons=[f"批量清理，预计涉及 {count}+ 个目录"],
            approvals_needed=1,
        )

    # apply=true 但没给范围 → 等价于"全删"，必须阻断
    return Verdict(
        decision=Decision.BLOCK,
        risk=Risk.CRITICAL,
        reasons=["apply=true 但未指定范围（task_id / keep / older_than_days），拒绝执行"],
    )


def describe(verdict: Verdict) -> dict[str, Any]:
    """把判定结果转成可返回给模型/前端的结构。"""
    return {
        "decision": verdict.decision.value,
        "risk": verdict.risk.value,
        "reasons": verdict.reasons,
        "approvals_needed": verdict.approvals_needed,
    }
