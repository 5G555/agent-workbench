"""分析结果缓存清理的测试（用假 Redis，不依赖真服务）。"""

from __future__ import annotations

from typing import Any

import pytest

from agents.ops import cache_maintenance as maint


class FakeRedis:
    """最小可用的 Redis 替身：只实现清理路径用到的命令。"""

    def __init__(self, keys: dict[str, str]) -> None:
        self.data = dict(keys)
        self.hashes: dict[str, dict[str, str]] = {}

    async def scan_iter(self, match: str, count: int = 100) -> Any:
        prefix = match.rstrip("*")
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                removed += 1
        return removed

    async def hset(self, name: str, field: str, value: str) -> int:
        self.hashes.setdefault(name, {})[field] = value
        return 1

    async def expire(self, name: str, seconds: int) -> bool:
        return True

    async def hgetall(self, name: str) -> dict[str, str]:
        return self.hashes.get(name, {})

    async def mset(self, mapping: dict[str, str]) -> bool:
        self.data.update(mapping)
        return True


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    client = FakeRedis(
        {
            "analyst:cache:tenant_default:aaa": "x" * 100,
            "analyst:cache:tenant_default:bbb": "y" * 100,
            "analyst:cache:tenant_demo:ccc": "z" * 100,
            "qa:ans:v37:s1:q1": "answer",
            "qa:ans:v36:old:q0": "stale-answer",  # 旧版本残留
        }
    )

    async def _get_redis() -> FakeRedis:
        return client

    async def _version() -> int:
        return 37

    monkeypatch.setattr(maint, "get_redis", _get_redis)
    monkeypatch.setattr(maint, "cache_version", _version)
    return client


async def test_scan_analyst_all_tenants(fake_redis: FakeRedis) -> None:
    result = await maint.scan_analyst(None)
    assert len(result.stale) == 3
    assert all(key.startswith("analyst:cache:") for key in result.stale)


async def test_scan_analyst_single_tenant(fake_redis: FakeRedis) -> None:
    result = await maint.scan_analyst("tenant_demo")
    assert result.stale == ["analyst:cache:tenant_demo:ccc"]


async def test_clean_analyst_only_keeps_qa(fake_redis: FakeRedis) -> None:
    report = await maint.clean(apply=True, kinds=["analyst"])
    assert report.deleted == 3
    assert report.backup_id  # 删除前有备份
    assert "qa:ans:v37:s1:q1" in fake_redis.data  # 问答缓存没被动
    assert not any(k.startswith("analyst:cache:") for k in fake_redis.data)


async def test_clean_qa_only_keeps_analyst(fake_redis: FakeRedis) -> None:
    report = await maint.clean(apply=True, kinds=["qa"])
    assert report.deleted == 1  # 只删旧版本那条
    assert "qa:ans:v37:s1:q1" in fake_redis.data
    assert sum(1 for k in fake_redis.data if k.startswith("analyst:cache:")) == 3


async def test_clean_dry_run_deletes_nothing(fake_redis: FakeRedis) -> None:
    report = await maint.clean(apply=False, kinds=["analyst"])
    assert report.deleted == 0
    assert report.stale_total == 3
    assert len([k for k in fake_redis.data if k.startswith("analyst:cache:")]) == 3


async def test_clean_fail_closed_when_backup_fails(
    fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """备份失败时一个都不能删。"""

    async def _broken_hset(name: str, field: str, value: str) -> int:
        raise RuntimeError("redis 写失败")

    monkeypatch.setattr(fake_redis, "hset", _broken_hset)

    report = await maint.clean(apply=True, kinds=["analyst"])
    assert report.deleted == 0
    assert len([k for k in fake_redis.data if k.startswith("analyst:cache:")]) == 3


async def test_overview_counts_analyst(fake_redis: FakeRedis) -> None:
    summary = await maint.overview()
    assert summary["analyst"] == 3
    assert summary["stale"] == 1  # qa 旧版本
