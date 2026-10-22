"""FAQ 部门级隔离的纯逻辑测试。"""

from __future__ import annotations

from agents.qa.faq import PUBLIC, SCOPE_ALL, SCOPE_PUBLIC, resolve_scope


def test_admin_sees_all_scopes():
    assert resolve_scope("一号部门", "admin") == SCOPE_ALL
    assert resolve_scope("*", "admin") == SCOPE_ALL
    assert resolve_scope(None, "admin") == SCOPE_ALL


def test_user_with_department_uses_that_scope():
    assert resolve_scope("一号部门", "user") == "一号部门"
    assert resolve_scope(" 二号部门 ", "user") == "二号部门"


def test_user_without_department_sees_public_only():
    assert resolve_scope("*", "user") == SCOPE_PUBLIC
    assert resolve_scope("", "user") == SCOPE_PUBLIC
    assert resolve_scope(None, "user") == SCOPE_PUBLIC
    assert resolve_scope(None, None) == SCOPE_PUBLIC


def test_public_constant_is_star():
    assert PUBLIC == "*"
    # 三个范围键互不相同，避免索引串用
    assert len({SCOPE_ALL, SCOPE_PUBLIC, "一号部门"}) == 3
