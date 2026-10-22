"""联网额度控制的纯逻辑测试。"""

from __future__ import annotations

from datetime import UTC, datetime

from core.quota import KEY_PREFIX, quota_key


def test_quota_key_is_tenant_and_day_scoped():
    key = quota_key("tenant_default", "2026-10-22")
    assert key == f"{KEY_PREFIX}:tenant_default:2026-10-22"


def test_different_tenants_do_not_share_quota():
    a = quota_key("tenant_a", "2026-10-22")
    b = quota_key("tenant_b", "2026-10-22")
    assert a != b


def test_quota_resets_next_day():
    today = quota_key("t1", "2026-10-22")
    tomorrow = quota_key("t1", "2026-10-23")
    assert today != tomorrow, "跨天必须是不同的键，否则额度不会恢复"


def test_default_day_is_today_utc():
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    assert quota_key("t1").endswith(today)
