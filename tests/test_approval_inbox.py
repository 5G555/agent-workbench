"""审批收件箱的测试：列表、过滤、拒绝后失效。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.ops import approval


class FakeRedis:
    """够用的内存 Redis：只需要 get/set/scan_iter。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def scan_iter(self, match: str | None = None, count: int = 100):
        prefix = (match or "").rstrip("*")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key


@pytest.fixture()
def fake_redis(monkeypatch):
    client = FakeRedis()

    async def _get():
        return client

    monkeypatch.setattr(approval, "get_redis", _get)
    return client


async def _make(tool: str = "clean_cache", requester: str = "1") -> str:
    ticket = await approval.create_ticket(
        tool, {"apply": True}, "tenant_default", requester, impact="清理缓存", reversible="可恢复"
    )
    assert ticket is not None
    return ticket.approval_id


async def test_list_pending_only(fake_redis):
    first = await _make()
    second = await _make()
    await approval.confirm_ticket(second, "admin")

    pending = await approval.list_tickets("pending")
    ids = {t["approval_id"] for t in pending}
    assert ids == {first}

    all_tickets = await approval.list_tickets()
    assert {t["approval_id"] for t in all_tickets} == {first, second}


async def test_list_carries_evidence(fake_redis):
    await _make()
    tickets = await approval.list_tickets("pending")
    assert tickets
    ticket = tickets[0]
    # 审批人必须能看到"批准的是什么"，否则就是盲签
    assert ticket["tool"] == "clean_cache"
    assert ticket["args"] == {"apply": True}
    assert ticket["impact"] == "清理缓存"
    assert ticket["reversible"] == "可恢复"
    assert ticket["requester"] == "1"
    assert ticket["fingerprint"]


async def test_reject_makes_ticket_unusable(fake_redis):
    approval_id = await _make()
    ok, message, _ = await approval.reject_ticket(approval_id, "admin")
    assert ok is True
    assert message == "已拒绝"

    passed, reason = await approval.verify_ticket(
        approval_id, "clean_cache", {"apply": True}, "tenant_default"
    )
    assert passed is False
    assert "rejected" in reason


async def test_cannot_reject_twice(fake_redis):
    approval_id = await _make()
    await approval.reject_ticket(approval_id, "admin")
    ok, message, _ = await approval.reject_ticket(approval_id, "admin")
    assert ok is False
    assert "rejected" in message


async def test_cannot_confirm_after_reject(fake_redis):
    approval_id = await _make()
    await approval.reject_ticket(approval_id, "admin")
    ok, message, _ = await approval.confirm_ticket(approval_id, "admin")
    assert ok is False
    assert "rejected" in message


async def test_reject_missing_ticket(fake_redis):
    ok, message, _ = await approval.reject_ticket("nope", "admin")
    assert ok is False
    assert "不存在" in message


async def test_ticket_payload_is_json_serializable(fake_redis):
    """收件箱直接把票据透给前端，必须能 JSON 序列化。"""
    await _make()
    tickets = await approval.list_tickets()
    json.dumps(tickets, ensure_ascii=False)
