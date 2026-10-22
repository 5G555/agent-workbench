"""执行网关：所有破坏性操作必经的最后一关。

职责顺序（任何一步不过就停，绝不放行）：
    1. 策略判定    policy.evaluate → ALLOW / SOFTEN / REQUIRE_APPROVAL / BLOCK
    2. 审批校验    高风险操作必须带票据，且票据指纹要匹配
    3. 幂等检查    同一个幂等键只执行一次，重放直接返回上次结果
    4. 预算熔断    单位时间内删除次数有上限，超了直接停
    5. 状态快照    执行前落一份"要动哪些东西"的清单，失败可回滚
    6. 执行
    7. 事后验证    验证不通过就回滚，并如实报告失败（不重试！）

为什么这里"不重试"：
    爬虫超时重试是合理的；删除重试可能造成重复删除或竞态。
    删除场景的正确做法是：失败即停、如实报告、靠快照回滚。

Fail-Closed：
    策略引擎、Redis（幂等/预算/审批）、快照写入，任何一项不可用都不放行。
    "护栏坏了就先裸奔"是这类系统最常见的事故来源。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.ops import approval
from agents.ops.policy import Decision, Verdict, describe, evaluate
from agents.qa.cache import get_redis
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

IDEM_PREFIX = "ops:idem"
BUDGET_PREFIX = "ops:budget"
IDEM_TTL_SECONDS = 86400
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "var" / "ops_snapshots"


@dataclass
class ExecutionResult:
    """网关返回。status 是给调用方（模型/界面）看的唯一信号。"""

    status: str  # executed / replayed / blocked / needs_approval / failed / rolled_back
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    approval: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "ok": self.ok,
            "verdict": self.verdict,
            "message": self.message,
            "data": self.data,
        }
        if self.approval:
            payload["approval"] = self.approval
        return payload


def _idem_key(key: str) -> str:
    return f"{IDEM_PREFIX}:{key}"


def _budget_key(tenant_id: str) -> str:
    hour = datetime.now(UTC).strftime("%Y%m%d%H")
    return f"{BUDGET_PREFIX}:{tenant_id}:{hour}"


IDEM_WINDOW_SECONDS = 300  # 默认幂等窗口 5 分钟


def default_idempotency_key(tool: str, args: dict[str, Any], tenant_id: str) -> str:
    """没给幂等键时的兜底：参数指纹 + 时间窗口。

    为什么加时间窗口：
        只用参数指纹的话，"清理缓存"这类操作 24 小时内只能执行一次，
        之后哪怕又出现了新的残留，也会被判成重放而什么都不做。

    窗口取 5 分钟：
        网络超时后的重发通常在一分钟内，落在同一个窗口 → 正确去重；
        过一会儿再执行同类操作 → 视为新请求，正常执行。
    需要严格"一次且仅一次"的场景，调用方应自己传 idempotency_key。
    """
    bucket = int(datetime.now(UTC).timestamp()) // IDEM_WINDOW_SECONDS
    base = approval.fingerprint(tool, args, tenant_id)
    return f"{base}-b{bucket}"


async def _check_idempotency(key: str) -> dict[str, Any] | None:
    client = await get_redis()
    if client is None:
        return None
    raw = await client.get(_idem_key(key))
    return json.loads(raw) if raw else None


async def _store_idempotency(key: str, result: dict[str, Any]) -> None:
    client = await get_redis()
    if client is None:
        return
    await client.set(
        _idem_key(key), json.dumps(result, ensure_ascii=False, default=str), ex=IDEM_TTL_SECONDS
    )


async def _consume_budget(tenant_id: str, amount: int) -> tuple[bool, int, int]:
    """扣预算，返回 (是否允许, 剩余, 上限)。Redis 不可用 → 不允许。"""
    settings = get_settings()
    limit = int(settings.ops_delete_budget_per_hour or 0)
    if limit <= 0:
        return True, -1, 0  # 配置成 0 表示不限

    client = await get_redis()
    if client is None:
        return False, 0, limit

    key = _budget_key(tenant_id)
    used = int(await client.incrby(key, max(1, amount)))
    if used == max(1, amount):
        await client.expire(key, 3600)
    if used > limit:
        await client.decrby(key, max(1, amount))
        return False, max(0, limit - (used - amount)), limit
    return True, max(0, limit - used), limit


def _write_snapshot(
    tool: str, args: dict[str, Any], tenant_id: str, victims: list[Any]
) -> str | None:
    """执行前落快照，返回快照文件路径；失败返回 None（调用方应阻断）。"""
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        payload = {
            "snapshot_id": snapshot_id,
            "tool": tool,
            "args": args,
            "tenant_id": tenant_id,
            "victims": victims,
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = SNAPSHOT_DIR / f"{snapshot_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except Exception as exc:
        logger.error("gateway.snapshot_failed", error=str(exc))
        return None


async def execute(
    tool: str,
    args: dict[str, Any],
    *,
    tenant_id: str,
    user_id: str = "",
    idempotency_key: str | None = None,
    approval_token: str | None = None,
    runner: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    victims: Callable[[dict[str, Any]], Awaitable[list[Any]]] | None = None,
    verify: Callable[[dict[str, Any], dict[str, Any]], Awaitable[bool]] | None = None,
    rollback: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]] | None = None,
    impact: str = "",
    reversible: str = "",
) -> ExecutionResult:
    """走完整条安全链执行一次工具调用。"""
    verdict: Verdict = evaluate(tool, args)
    meta = describe(verdict)

    # ① 阻断
    if verdict.decision is Decision.BLOCK:
        logger.warning("gateway.blocked", tool=tool, reasons=verdict.reasons, tenant=tenant_id)
        return ExecutionResult(
            status="blocked",
            ok=False,
            verdict=meta,
            message="；".join(verdict.reasons) or "策略拒绝",
        )

    # ② 幂等检查（**放在审批之前**）
    # 理由：网络超时后客户端重发同一请求时，票据已经被消费掉了。
    # 若先校验票据，重发会被判"票据无效"，用户看到失败，但操作其实已成功——
    # 这正是幂等要解决的场景。同参数请求直接返回首次结果，安全且不会重复删。
    key = idempotency_key or default_idempotency_key(tool, args, tenant_id)
    client = await get_redis()
    if client is None and verdict.risk != "low":
        return ExecutionResult(
            status="blocked",
            ok=False,
            verdict=meta,
            message="幂等/预算存储不可用，拒绝执行（Fail-Closed）",
        )
    previous = await _check_idempotency(key)
    if previous is not None:
        logger.info("gateway.replayed", tool=tool, key=key)
        return ExecutionResult(
            status="replayed",
            ok=bool(previous.get("ok")),
            verdict=meta,
            data=previous,
            message="该请求已执行过，返回首次结果（幂等重放）",
        )

    # ③ 审批
    ticket_info: dict[str, Any] | None = None
    if verdict.decision is Decision.REQUIRE_APPROVAL:
        if not approval_token:
            ticket = await approval.create_ticket(
                tool, args, tenant_id, user_id or "unknown", impact=impact, reversible=reversible
            )
            if ticket is None:
                return ExecutionResult(
                    status="blocked",
                    ok=False,
                    verdict=meta,
                    message="审批存储不可用，拒绝执行（Fail-Closed）",
                )
            return ExecutionResult(
                status="needs_approval",
                ok=False,
                verdict=meta,
                message="该操作需要人工审批后才能执行",
                approval={**ticket.as_dict(), "impact": impact, "reversible": reversible},
            )

        passed, reason = await approval.verify_ticket(approval_token, tool, args, tenant_id)
        if not passed:
            logger.warning("gateway.approval_rejected", tool=tool, reason=reason)
            return ExecutionResult(
                status="blocked", ok=False, verdict=meta, message=f"审批校验失败：{reason}"
            )
        ticket_info = {"approval_id": approval_token}
        await approval.consume_ticket(approval_token)

    # ④ 预算
    amount = 1
    allowed, remaining, limit = await _consume_budget(tenant_id, amount)
    if not allowed:
        return ExecutionResult(
            status="blocked",
            ok=False,
            verdict=meta,
            message=f"删除预算已用尽（{limit} 次/小时），本次不执行",
        )

    # ⑤ 快照
    victims_list: list[Any] = []
    if victims is not None:
        try:
            victims_list = await victims(args)
        except Exception as exc:
            logger.error("gateway.victims_failed", error=str(exc))
            return ExecutionResult(
                status="blocked", ok=False, verdict=meta, message=f"无法确定影响范围：{exc}"
            )
    snapshot_path = _write_snapshot(tool, args, tenant_id, victims_list)
    if snapshot_path is None:
        return ExecutionResult(
            status="blocked",
            ok=False,
            verdict=meta,
            message="快照写入失败，拒绝执行（Fail-Closed）",
        )

    # ⑥ 执行（不重试）
    if runner is None:
        return ExecutionResult(status="blocked", ok=False, verdict=meta, message="未提供执行函数")
    try:
        data = await runner(args)
    except Exception as exc:
        logger.error("gateway.run_failed", tool=tool, error=str(exc))
        if rollback is not None:
            try:
                await rollback(args, {"error": str(exc)})
                return ExecutionResult(
                    status="rolled_back",
                    ok=False,
                    verdict=meta,
                    message=f"执行失败，已回滚：{exc}",
                    data={"snapshot": snapshot_path},
                )
            except Exception as rb_exc:
                logger.error("gateway.rollback_failed", error=str(rb_exc))
        return ExecutionResult(status="failed", ok=False, verdict=meta, message=f"执行失败：{exc}")

    # ⑦ 事后验证
    verified = True
    if verify is not None:
        try:
            verified = await verify(args, data)
        except Exception as exc:
            verified = False
            logger.error("gateway.verify_error", error=str(exc))
    if not verified:
        if rollback is not None:
            try:
                await rollback(args, data)
                return ExecutionResult(
                    status="rolled_back",
                    ok=False,
                    verdict=meta,
                    message="执行结果未通过验证，已回滚",
                    data={"snapshot": snapshot_path, **data},
                )
            except Exception as rb_exc:
                logger.error("gateway.rollback_failed", error=str(rb_exc))
        return ExecutionResult(
            status="failed",
            ok=False,
            verdict=meta,
            message="执行结果未通过验证（回滚未成功）",
            data=data,
        )

    result = {
        **data,
        "snapshot": snapshot_path,
        "budget_remaining": remaining,
        "approval": ticket_info,
    }
    await _store_idempotency(key, {"ok": True, **result})
    logger.info("gateway.executed", tool=tool, tenant=tenant_id, key=key)
    return ExecutionResult(status="executed", ok=True, verdict=meta, data=result, message="已执行")
