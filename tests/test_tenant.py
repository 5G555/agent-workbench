"""多租户隔离的纯逻辑测试。"""

from __future__ import annotations

from agents.qa.cache import answer_key
from agents.qa.faq import bm25_confidence
from agents.qa.retrieval.vector_store import RetrievalScope, roles_for


# ── 置信度归一化（与库大小无关）──────────────────────────
def test_confidence_is_size_independent():
    """同样"完全命中"，大库和小库都应该给出高置信度。"""
    big = [5.2, 1.0, 0.2] + [0.0] * 9  # 12 条
    small = [2.79, 0.0, 0.0, 0.0]  # 4 条
    _, big_conf = bm25_confidence(big)
    _, small_conf = bm25_confidence(small)
    assert big_conf > 0.85
    assert small_conf > 0.70  # 旧实现这里只有 0.845，被 0.85 阈值挡掉
    assert abs(big_conf - small_conf) < 0.2


def test_confidence_zero_when_all_equal():
    _, conf = bm25_confidence([0.0, 0.0, 0.0, 0.0])
    assert conf == 0.0


def test_confidence_low_for_weak_overlap():
    _, conf = bm25_confidence([1.0, 0.9, 0.9, 0.9])
    assert conf < 0.1


def test_confidence_handles_empty():
    assert bm25_confidence([]) == (0, 0.0)


# ── 缓存键 ────────────────────────────────────────────────
def test_cache_key_scoped_by_tenant():
    a = answer_key(1, "你们怎么收费", "tenant_a")
    b = answer_key(1, "你们怎么收费", "tenant_b")
    assert a != b, "不同租户的相同问题不能共用缓存"


def test_cache_key_scoped_by_department_and_role():
    """同一租户内，不同部门 / 不同角色也不能共用缓存。"""
    base = answer_key(1, "q", "t1", "一号部门", "user")
    assert base != answer_key(1, "q", "t1", "二号部门", "user")
    assert base != answer_key(1, "q", "t1", "一号部门", "admin")


def test_cache_key_stable_for_same_tenant():
    assert answer_key(1, "你们怎么收费", "t1") == answer_key(1, " 你们 怎么收费 ", "t1")


def test_cache_key_without_tenant_keeps_old_shape():
    """不传租户时保持旧格式，脚本/测试仍可用。"""
    assert ":t" not in answer_key(1, "q")


# ── Milvus 过滤表达式 ─────────────────────────────────────
def test_filter_always_contains_tenant():
    scope = RetrievalScope(tenant_id="tenant_a", department="*", role="admin")
    assert scope.filter_expr() == 'tenant_id == "tenant_a"'


def test_filter_combines_tenant_and_source():
    scope = RetrievalScope(tenant_id="tenant_a", role="admin")
    expr = scope.filter_expr("01_产品概览")
    assert 'tenant_id == "tenant_a"' in expr
    assert 'source == "01_产品概览"' in expr
    assert " and " in expr


def test_filter_without_tenant_keeps_role_condition():
    """租户为空（脚本场景）时不再拼租户条件，但角色限制仍然生效。"""
    expr = RetrievalScope(tenant_id="").filter_expr()
    assert "tenant_id" not in expr
    assert 'min_role in ["user"]' in expr


# ── 细粒度权限：部门与角色 ────────────────────────────────
def test_user_sees_own_department_and_public():
    scope = RetrievalScope(tenant_id="t1", department="二号部门", role="user")
    expr = scope.filter_expr()
    assert '(department == "二号部门" or department == "*")' in expr
    assert 'min_role in ["user"]' in expr


def test_user_without_department_only_sees_public():
    """用户部门为 * 表示未分配部门，只能看全公司文档，不能因此跳过部门限制。"""
    scope = RetrievalScope(tenant_id="t1", department="*", role="user")
    expr = scope.filter_expr()
    assert 'department == "*"' in expr
    assert "or department" not in expr
    assert 'min_role in ["user"]' in expr


def test_admin_bypasses_department_and_role():
    scope = RetrievalScope(tenant_id="t1", department="二号部门", role="admin")
    expr = scope.filter_expr()
    assert "department" not in expr, "管理员不该被部门限制"
    assert "min_role" not in expr, "管理员能看所有角色级别的文档"


def test_roles_for_hierarchy():
    assert roles_for("user") == ["user"]
    assert set(roles_for("admin")) == {"user", "admin"}
    assert roles_for("unknown") == ["user"], "未知角色按最低权限处理"
