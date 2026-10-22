"""运维审计表：把每一次破坏性操作的决定与结果落库。

为什么要独立一张表，而不是只写日志：
    日志会被轮转、会被忽略、也不方便按租户查。审计要回答的是几个具体问题：
        - 谁在什么时候发起了什么操作？（intent）
        - 策略当时怎么判的？风险几级？为什么放行/阻断？
        - 有没有审批？谁批的？是不是自审（self_approved）？
        - 执行结果是什么？成功、回滚、还是失败？快照在哪？

两阶段写入（这一点很关键）：
    执行前先写一条 status='executing' 的 intent，执行后再更新为终态。
    这样即使进程在执行中途崩了，也能从表里看到"有一条操作没走完"，
    而不是事后完全无痕。intent 写失败就直接拒绝执行（Fail-Closed）。

表放在 Postgres（与业务同库），不用 Redis：
    审计要长期保留、要能 SQL 查询、要能在 Redis 整个丢的时候还在。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import Settings, get_settings
from core.db import lease_connection
from core.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ops_audit (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64)  NOT NULL DEFAULT 'tenant_default',
    requester       VARCHAR(64)  NOT NULL DEFAULT '',
    tool            VARCHAR(64)  NOT NULL DEFAULT '',
    args            TEXT         NOT NULL DEFAULT '{}',
    risk            VARCHAR(16)  NOT NULL DEFAULT '',
    decision        VARCHAR(24)  NOT NULL DEFAULT '',
    reasons         TEXT         NOT NULL DEFAULT '',
    approval_id     VARCHAR(64)  NOT NULL DEFAULT '',
    approver        VARCHAR(64)  NOT NULL DEFAULT '',
    self_approved   BOOLEAN      NOT NULL DEFAULT false,
    snapshot        VARCHAR(1024) NOT NULL DEFAULT '',
    status          VARCHAR(24)  NOT NULL DEFAULT 'executing',
    result          TEXT         NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ops_audit_tenant_time
    ON ops_audit (tenant_id, id DESC);
CREATE INDEX IF NOT EXISTS ix_ops_audit_tool ON ops_audit (tool, id DESC);
"""

# 这些状态表示"操作已经结束"，可以写 finished_at
TERMINAL_STATUSES = {
    "blocked",
    "needs_approval",
    "executed",
    "replayed",
    "failed",
    "rolled_back",
}


@dataclass
class AuditIntent:
    """执行前记录的操作意图。"""

    tenant_id: str
    requester: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    risk: str = ""
    decision: str = ""
    reasons: list[str] = field(default_factory=list)
    approval_id: str = ""
    approver: str = ""
    self_approved: bool = False
    snapshot: str = ""


def summarize_result(data: dict[str, Any] | None) -> str:
    """把执行结果压成一行摘要（审计表不存全量 JSON，太大了）。"""
    if not data:
        return ""
    keys = ("deleted", "purged", "freed", "backup_id", "dirs", "version", "trash_ids")
    picked = {k: data[k] for k in keys if k in data}
    if not picked:
        return json.dumps(data, ensure_ascii=False)[:500]
    return json.dumps(picked, ensure_ascii=False)[:500]


class AuditStore:
    """ops_audit 的写入与查询。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ready = False

    async def _connect(self):
        return await lease_connection()

    async def ensure_table(self) -> None:
        if self._ready:
            return
        conn = await self._connect()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            self._ready = True
        finally:
            await conn.close()

    async def record_intent(self, intent: AuditIntent) -> int:
        """执行前写入意图，返回审计 ID。失败抛异常（调用方据此拒绝执行）。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            audit_id = await conn.fetchval(
                """
                INSERT INTO ops_audit
                    (tenant_id, requester, tool, args, risk, decision, reasons,
                     approval_id, approver, self_approved, snapshot, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'executing')
                RETURNING id
                """,
                intent.tenant_id,
                intent.requester,
                intent.tool,
                json.dumps(intent.args, ensure_ascii=False, default=str),
                intent.risk,
                intent.decision,
                "；".join(intent.reasons)[:1000],
                intent.approval_id,
                intent.approver,
                intent.self_approved,
                intent.snapshot[:1000],
            )
        finally:
            await conn.close()
        return int(audit_id or 0)

    async def record_outcome(
        self, audit_id: int, status: str, result: dict[str, Any] | None = None
    ) -> None:
        """写终态。审计写失败只记日志，不影响已经完成的业务结果。"""
        if not audit_id:
            return
        conn = await self._connect()
        try:
            await conn.execute(
                """
                UPDATE ops_audit
                SET status = $2, result = $3,
                    finished_at = CASE WHEN $4 THEN now() ELSE finished_at END
                WHERE id = $1
                """,
                audit_id,
                status[:24],
                summarize_result(result),
                status in TERMINAL_STATUSES,
            )
        finally:
            await conn.close()

    async def record_one_shot(
        self, intent: AuditIntent, status: str, result: dict[str, Any] | None = None
    ) -> None:
        """记录没有实际执行的事件（被阻断、需要审批、幂等重放）。"""
        audit_id = await self.record_intent(intent)
        await self.record_outcome(audit_id, status, result)

    async def list_entries(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        await self.ensure_table()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT id, requester, tool, risk, decision, reasons, approval_id,
                       approver, self_approved, status, result, created_at, finished_at
                FROM ops_audit
                WHERE tenant_id = $1
                ORDER BY id DESC LIMIT $2
                """,
                tenant_id,
                limit,
            )
        finally:
            await conn.close()
        return [
            {
                "id": int(r["id"]),
                "requester": str(r["requester"]),
                "tool": str(r["tool"]),
                "risk": str(r["risk"]),
                "decision": str(r["decision"]),
                "reasons": str(r["reasons"]),
                "approval_id": str(r["approval_id"]),
                "approver": str(r["approver"]),
                "self_approved": bool(r["self_approved"]),
                "status": str(r["status"]),
                "result": str(r["result"]),
                "created_at": r["created_at"].isoformat(),
                "finished_at": r["finished_at"].isoformat() if r["finished_at"] else "",
            }
            for r in rows
        ]

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        """按状态汇总，用于面板顶部显示。"""
        await self.ensure_table()
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                "SELECT status, count(*) AS n FROM ops_audit WHERE tenant_id = $1 GROUP BY status",
                tenant_id,
            )
            self_approved = await conn.fetchval(
                "SELECT count(*) FROM ops_audit WHERE tenant_id = $1 AND self_approved",
                tenant_id,
            )
            unfinished = await conn.fetchval(
                "SELECT count(*) FROM ops_audit WHERE tenant_id = $1 AND status = 'executing'",
                tenant_id,
            )
        finally:
            await conn.close()
        by_status = {str(r["status"]): int(r["n"]) for r in rows}
        return {
            "by_status": by_status,
            "total": sum(by_status.values()),
            "self_approved": int(self_approved or 0),
            "unfinished": int(unfinished or 0),
        }


_store: AuditStore | None = None


def get_audit_store() -> AuditStore:
    global _store
    if _store is None:
        _store = AuditStore()
    return _store
