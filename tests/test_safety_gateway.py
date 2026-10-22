"""执行网关的测试：审批、幂等、预算、快照、回滚、Fail-Closed。

用内存版假 Redis，避免测试依赖真实服务。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.ops import approval, gateway

TENANT = "tenant_default"


class FakeRedis:
    """够用的内存 Redis：只实现网关和票据用到的那几个命令。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)

    async def incrby(self, key: str, amount: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + amount
        return self.counters[key]

    async def decrby(self, key: str, amount: int) -> int:
        self.counters[key] = self.counters.get(key, 0) - amount
        return self.counters[key]

    async def expire(self, key: str, seconds: int) -> None:
        return None

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def mset(self, mapping: dict[str, str]) -> None:
        self.store.update(mapping)


class FakeAudit:
    """测试用审计：只记内存，不连数据库。"""

    def __init__(self) -> None:
        self.intents: list[tuple] = []
        self.outcomes: list[tuple] = []

    async def record_intent(self, intent):
        self.intents.append((intent.tool, intent.tenant_id, intent.self_approved))
        return len(self.intents)

    async def record_outcome(self, audit_id, status, result=None):
        self.outcomes.append((audit_id, status))

    async def record_one_shot(self, intent, status, result=None):
        self.intents.append((intent.tool, intent.tenant_id, intent.self_approved))
        self.outcomes.append((len(self.intents), status))


@pytest.fixture(autouse=True)
def fake_audit(monkeypatch):
    store = FakeAudit()

    def _get():
        return store

    monkeypatch.setattr(gateway.audit_mod, "get_audit_store", _get)
    return store


@pytest.fixture()
def fake_redis(monkeypatch):
    client = FakeRedis()

    async def _get():
        return client

    monkeypatch.setattr(gateway, "get_redis", _get)
    monkeypatch.setattr(approval, "get_redis", _get)
    return client


@pytest.fixture()
def no_redis(monkeypatch):
    async def _get():
        return None

    monkeypatch.setattr(gateway, "get_redis", _get)
    monkeypatch.setattr(approval, "get_redis", _get)


@pytest.fixture(autouse=True)
def temp_snapshot_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "SNAPSHOT_DIR", tmp_path / "snapshots")
    return tmp_path / "snapshots"


def _approval_id(result: gateway.ExecutionResult) -> str:
    """取出票据号；类型上 approval 可能为 None，这里显式断言。"""
    assert result.approval is not None
    return str(result.approval["approval_id"])


async def _runner(args: dict[str, Any]) -> dict[str, Any]:
    return {"deleted": [3], "freed": 100}


# ── 策略阻断 ────────────────────────────────────────────────
async def test_blocked_by_policy(fake_redis):
    result = await gateway.execute("shell_exec", {}, tenant_id=TENANT, runner=_runner)
    assert result.status == "blocked"
    assert result.ok is False
    assert "未知工具" in result.message


async def test_apply_without_scope_blocked(fake_redis):
    result = await gateway.execute(
        "clean_exports", {"apply": True}, tenant_id=TENANT, runner=_runner
    )
    assert result.status == "blocked"


# ── 审批 ────────────────────────────────────────────────────
async def test_high_risk_returns_needs_approval(fake_redis):
    result = await gateway.execute(
        "clean_exports", {"keep": 2, "apply": True}, tenant_id=TENANT, runner=_runner
    )
    assert result.status == "needs_approval"
    assert result.approval is not None
    approval_id = _approval_id(result)

    # 票据确实落库了，状态是 pending
    stored = json.loads(fake_redis.store[f"ops:approval:{approval_id}"])
    assert stored["status"] == "pending"
    assert stored["fingerprint"]


async def test_approved_call_executes(fake_redis):
    args = {"task_id": 3, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    assert first.status == "needs_approval"
    approval_id = _approval_id(first)

    ok, _, _ = await approval.confirm_ticket(approval_id, "admin-1")
    assert ok

    second = await gateway.execute(
        "clean_exports", args, tenant_id=TENANT, runner=_runner, approval_token=approval_id
    )
    assert second.status == "executed"
    assert second.ok is True


async def test_ticket_bound_to_arguments(fake_redis):
    """批的是删任务 3，不能拿去删别的。"""
    args = {"task_id": 3, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    approval_id = _approval_id(first)
    await approval.confirm_ticket(approval_id, "admin-1")

    tampered = {"task_id": 99, "apply": True}
    result = await gateway.execute(
        "clean_exports", tampered, tenant_id=TENANT, runner=_runner, approval_token=approval_id
    )
    assert result.status == "blocked"
    assert "不匹配" in result.message


async def test_self_approval_rejected(fake_redis):
    first = await gateway.execute(
        "clean_exports",
        {"task_id": 3, "apply": True},
        tenant_id=TENANT,
        user_id="alice",
        runner=_runner,
    )
    ok, message, _ = await approval.confirm_ticket(_approval_id(first), "alice")
    assert ok is False
    assert "自己" in message


async def test_ticket_is_single_use(fake_redis):
    args = {"task_id": 4, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    approval_id = _approval_id(first)
    await approval.confirm_ticket(approval_id, "admin-1")

    await gateway.execute(
        "clean_exports", args, tenant_id=TENANT, runner=_runner, approval_token=approval_id
    )
    again = await gateway.execute(
        "clean_exports",
        {"task_id": 5, "apply": True},
        tenant_id=TENANT,
        runner=_runner,
        approval_token=approval_id,
    )
    assert again.status == "blocked"


# ── 幂等 ────────────────────────────────────────────────────
async def test_idempotent_replay(fake_redis):
    args = {"task_id": 6, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    approval_id = _approval_id(first)
    await approval.confirm_ticket(approval_id, "admin-1")
    executed = await gateway.execute(
        "clean_exports", args, tenant_id=TENANT, runner=_runner, approval_token=approval_id
    )
    assert executed.status == "executed"

    # 同样的参数再来一次：返回首次结果，不重复执行
    replay = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    assert replay.status == "replayed"
    assert replay.data.get("deleted") == [3]


# ── 预算 ────────────────────────────────────────────────────
async def test_budget_exhausted_blocks(fake_redis, monkeypatch):
    class Stub:
        ops_delete_budget_per_hour = 1
        ops_approval_ttl_seconds = 900
        ops_allow_self_approval = True

    monkeypatch.setattr(gateway, "get_settings", lambda: Stub())

    args = {"task_id": 7, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    await approval.confirm_ticket(_approval_id(first), "admin-1")
    await gateway.execute(
        "clean_exports",
        args,
        tenant_id=TENANT,
        runner=_runner,
        approval_token=_approval_id(first),
    )

    # 第二次不同参数 → 预算已用尽（1 次/小时）
    second = await gateway.execute(
        "clean_exports", {"task_id": 8, "apply": True}, tenant_id=TENANT, runner=_runner
    )
    if second.status == "needs_approval":
        await approval.confirm_ticket(_approval_id(second), "admin-1")
        second = await gateway.execute(
            "clean_exports",
            {"task_id": 8, "apply": True},
            tenant_id=TENANT,
            runner=_runner,
            approval_token=_approval_id(second),
        )
    assert second.status == "blocked"
    assert "预算" in second.message


# ── Fail-Closed ────────────────────────────────────────────
async def test_fail_closed_when_redis_missing(no_redis):
    result = await gateway.execute(
        "clean_exports", {"task_id": 1, "apply": True}, tenant_id=TENANT, runner=_runner
    )
    assert result.status == "blocked"
    assert "Fail-Closed" in result.message


async def test_snapshot_written_before_run(fake_redis, temp_snapshot_dir):
    args = {"task_id": 9, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    await approval.confirm_ticket(_approval_id(first), "admin-1")
    result = await gateway.execute(
        "clean_exports",
        args,
        tenant_id=TENANT,
        runner=_runner,
        approval_token=_approval_id(first),
        victims=lambda _a: _victims(),
    )
    assert result.status == "executed"
    snapshots = list(temp_snapshot_dir.glob("*.json"))
    assert len(snapshots) == 1
    payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert payload["tool"] == "clean_exports"
    assert payload["tenant_id"] == TENANT


async def _victims() -> list[Any]:
    return [9]


# ── 失败 / 回滚：不重试 ────────────────────────────────────
async def test_runner_failure_rolls_back(fake_redis):
    rolled: list[str] = []

    async def failing(_args):
        raise RuntimeError("移动失败")

    async def rollback(_args, _data):
        rolled.append("done")

    args = {"task_id": 10, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=failing)
    await approval.confirm_ticket(_approval_id(first), "admin-1")
    result = await gateway.execute(
        "clean_exports",
        args,
        tenant_id=TENANT,
        runner=failing,
        rollback=rollback,
        approval_token=_approval_id(first),
    )
    assert result.status == "rolled_back"
    assert rolled == ["done"]


async def test_verify_failure_rolls_back(fake_redis):
    rolled: list[str] = []

    async def verify_false(_args, _data):
        return False

    async def rollback(_args, _data):
        rolled.append("done")

    args = {"task_id": 11, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=_runner)
    await approval.confirm_ticket(_approval_id(first), "admin-1")
    result = await gateway.execute(
        "clean_exports",
        args,
        tenant_id=TENANT,
        runner=_runner,
        verify=verify_false,
        rollback=rollback,
        approval_token=_approval_id(first),
    )
    assert result.status == "rolled_back"
    assert rolled == ["done"]


async def test_runner_is_not_retried(fake_redis):
    """删除失败不重试：retry 会造成重复删除或竞态。"""
    calls = {"n": 0}

    async def failing(_args):
        calls["n"] += 1
        raise RuntimeError("boom")

    args = {"task_id": 12, "apply": True}
    first = await gateway.execute("clean_exports", args, tenant_id=TENANT, runner=failing)
    await approval.confirm_ticket(_approval_id(first), "admin-1")
    await gateway.execute(
        "clean_exports",
        args,
        tenant_id=TENANT,
        runner=failing,
        approval_token=_approval_id(first),
    )
    assert calls["n"] == 1
