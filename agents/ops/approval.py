"""删除操作的审批票据（HITL）。

设计要点：
    1. 审批**绑定证据**，不是一句"同意"
       票据里存工具名 + 参数 + 影响范围 + 可逆性说明，
       指纹是 sha256(tool|args|tenant)，参数一变就失效——
       防止"批的是删任务 3，实际执行删全部"。
    2. 一次性
       确认后只能消费一次（status: pending → confirmed → used）。
    3. 有有效期
       默认 15 分钟，过期作废，避免长期挂在系统里被翻出来用。
    4. 审批人不能是发起人自己（防止 Agent 用一个账号自批自干）。
       —— 这一条在小团队里可能过严，配置项 ops_allow_self_approval 可关。

存储用 Redis：票据是短生命周期、可丢的数据，不需要进 Postgres。
Redis 不可用时一律拒绝发放/校验（Fail-Closed）。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agents.qa.cache import get_redis
from config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "ops:approval"
DEFAULT_TTL_SECONDS = 900  # 15 分钟


@dataclass
class Ticket:
    """一张审批票据。"""

    approval_id: str
    tool: str
    args: dict[str, Any]
    tenant_id: str
    requester: str
    fingerprint: str
    status: str = "pending"  # pending / confirmed / used
    operator: str = ""
    created_at: str = ""
    expires_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "tool": self.tool,
            "args": self.args,
            "tenant_id": self.tenant_id,
            "requester": self.requester,
            # 指纹必须存下来：执行时要拿它和当前参数比对，否则"批 A 删 B"就成立
            "fingerprint": self.fingerprint,
            "status": self.status,
            "operator": self.operator,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


def fingerprint(tool: str, args: dict[str, Any], tenant_id: str) -> str:
    """把"批准的具体内容"压成一个指纹。参数排序保证稳定。"""
    payload = json.dumps(
        {"tool": tool, "args": args, "tenant": tenant_id},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _key(approval_id: str) -> str:
    return f"{KEY_PREFIX}:{approval_id}"


async def create_ticket(
    tool: str,
    args: dict[str, Any],
    tenant_id: str,
    requester: str,
    *,
    impact: str = "",
    reversible: str = "",
) -> Ticket | None:
    """创建待审批票据。返回 None 表示存储不可用（调用方应拒绝执行）。"""
    client = await get_redis()
    if client is None:
        logger.warning("approval.store_unavailable")
        return None

    settings = get_settings()
    ttl = int(settings.ops_approval_ttl_seconds or DEFAULT_TTL_SECONDS)
    now = datetime.now(UTC)
    ticket = Ticket(
        approval_id=uuid.uuid4().hex[:16],
        tool=tool,
        args=args,
        tenant_id=tenant_id,
        requester=requester,
        fingerprint=fingerprint(tool, args, tenant_id),
        created_at=now.isoformat(),
        expires_at=datetime.fromtimestamp(now.timestamp() + ttl, tz=UTC).isoformat(),
    )
    payload = {**ticket.as_dict(), "impact": impact, "reversible": reversible}
    await client.set(_key(ticket.approval_id), json.dumps(payload, ensure_ascii=False), ex=ttl)
    logger.info("approval.created", approval=ticket.approval_id, tool=tool, tenant=tenant_id)
    return ticket


async def confirm_ticket(
    approval_id: str, operator: str, allow_self: bool = False
) -> tuple[bool, str, dict | None]:
    """人工确认。返回 (成功, 说明, 票据)。

    allow_self：是否允许发起人自己批准。
        调用方按"本租户管理员数量"决定：
            只有 1 个管理员 → 允许（否则这个租户永远删不了东西，会把功能锁死）
            2 个及以上    → 不允许，要求换一个人批（真正的双人复核）
        自批会在票据里记 self_approved=True 并打 warning，便于事后审计。
    """
    client = await get_redis()
    if client is None:
        return False, "审批存储不可用，拒绝执行", None

    raw = await client.get(_key(approval_id))
    if not raw:
        return False, "票据不存在或已过期", None

    data = json.loads(raw)
    if data.get("status") != "pending":
        return False, f"票据状态是 {data.get('status')}，不能再次确认", data

    settings = get_settings()
    self_approval = operator == data.get("requester")
    if self_approval and not (allow_self or settings.ops_allow_self_approval):
        return False, "不能自己批准自己的删除请求（本租户还有其他管理员）", data

    data["status"] = "confirmed"
    data["operator"] = operator
    data["confirmed_at"] = datetime.now(UTC).isoformat()
    if self_approval:
        # 自批必须留痕：审计时能一眼看出"这次没有第二个人看过"
        data["self_approved"] = True
        logger.warning("approval.self_approved", approval=approval_id, operator=operator)
    ttl = int(settings.ops_approval_ttl_seconds or DEFAULT_TTL_SECONDS)
    await client.set(_key(approval_id), json.dumps(data, ensure_ascii=False), ex=ttl)
    logger.info("approval.confirmed", approval=approval_id, operator=operator)
    return True, "已确认", data


async def verify_ticket(
    approval_id: str, tool: str, args: dict[str, Any], tenant_id: str
) -> tuple[bool, str]:
    """执行前校验票据：状态、指纹、租户三者都要对上。"""
    client = await get_redis()
    if client is None:
        return False, "审批存储不可用，拒绝执行（Fail-Closed）"

    raw = await client.get(_key(approval_id))
    if not raw:
        return False, "票据不存在或已过期"

    data = json.loads(raw)
    if data.get("status") != "confirmed":
        return False, f"票据状态是 {data.get('status')}，需要先人工确认"
    if data.get("tenant_id") != tenant_id:
        return False, "票据不属于当前租户"
    if data.get("fingerprint") != fingerprint(tool, args, tenant_id):
        return False, "票据与当前参数不匹配（可能被篡改或参数已变）"
    return True, "ok"


async def consume_ticket(approval_id: str) -> None:
    """标记为已使用（一次性）。"""
    client = await get_redis()
    if client is None:
        return
    raw = await client.get(_key(approval_id))
    if not raw:
        return
    data = json.loads(raw)
    data["status"] = "used"
    data["used_at"] = datetime.now(UTC).isoformat()
    await client.set(_key(approval_id), json.dumps(data, ensure_ascii=False), ex=300)


async def get_ticket(approval_id: str) -> dict | None:
    client = await get_redis()
    if client is None:
        return None
    raw = await client.get(_key(approval_id))
    return json.loads(raw) if raw else None


async def list_tickets(status: str | None = None) -> list[dict[str, Any]]:
    """列出票据（默认全部），按创建时间倒序。

    为什么用 SCAN 而不是维护一个索引集合：
        票据数量很少（一次删除一条，过期自动消失），SCAN 足够；
        维护 Set 还要处理"过期了要从集合里删"，反而多一处不一致。
    """
    client: Any = await get_redis()
    if client is None:
        return []

    tickets: list[dict[str, Any]] = []
    async for raw in client.scan_iter(match=f"{KEY_PREFIX}:*", count=200):
        key = raw if isinstance(raw, str) else raw.decode()
        payload = await client.get(key)
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if status and data.get("status") != status:
            continue
        tickets.append(data)

    tickets.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return tickets


async def reject_ticket(approval_id: str, operator: str) -> tuple[bool, str, dict | None]:
    """拒绝一张票据：标记 rejected 并让它彻底失效。"""
    client = await get_redis()
    if client is None:
        return False, "审批存储不可用", None

    raw = await client.get(_key(approval_id))
    if not raw:
        return False, "票据不存在或已过期", None

    data = json.loads(raw)
    if data.get("status") != "pending":
        return False, f"票据状态是 {data.get('status')}，不能拒绝", data

    data["status"] = "rejected"
    data["operator"] = operator
    data["rejected_at"] = datetime.now(UTC).isoformat()
    # 拒绝后只留 5 分钟可查，随即消失
    await client.set(_key(approval_id), json.dumps(data, ensure_ascii=False), ex=300)
    logger.info("approval.rejected", approval=approval_id, operator=operator)
    return True, "已拒绝", data
