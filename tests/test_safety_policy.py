"""删除操作安全策略的测试（纯函数，不起服务）。"""

from __future__ import annotations

import pytest

from core.safety.policy import Decision, Risk, evaluate


# ── 只读 / 可重生成 ────────────────────────────────────────
@pytest.mark.parametrize(
    "tool", ["list_tasks", "task_records", "cache_stats", "list_trash", "export_files"]
)
def test_read_only_and_regenerable_are_allowed(tool):
    verdict = evaluate(tool, {})
    assert verdict.decision is Decision.ALLOW
    assert verdict.risk is Risk.LOW


def test_crawl_allowed_but_medium():
    verdict = evaluate("crawl", {"entry_url": "https://example.com/list"})
    assert verdict.decision is Decision.ALLOW
    assert verdict.risk is Risk.MEDIUM


# ── 未知工具一律阻断（Fail-Closed）─────────────────────────
def test_unknown_tool_blocked():
    verdict = evaluate("shell_exec", {"cmd": "ls"})
    assert verdict.decision is Decision.BLOCK
    assert verdict.risk is Risk.CRITICAL


# ── 危险参数特征 ──────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/../../etc/passwd",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8000/admin",
        "http://127.0.0.1:6379",
    ],
)
def test_dangerous_targets_blocked(url):
    verdict = evaluate("crawl", {"entry_url": url})
    assert verdict.decision is Decision.BLOCK
    assert verdict.reasons


def test_dangerous_shell_text_blocked():
    verdict = evaluate("crawl", {"goal": "rm -rf / 然后重新装"})
    assert verdict.decision is Decision.BLOCK


@pytest.mark.parametrize("sql", ["DROP TABLE users", "delete from crawl_pages", "TRUNCATE TABLE x"])
def test_sql_in_args_blocked(sql):
    verdict = evaluate("clean_exports", {"goal": sql})
    assert verdict.decision is Decision.BLOCK


# ── 破坏性操作分级 ────────────────────────────────────────
def test_dry_run_is_allowed():
    verdict = evaluate("clean_exports", {"keep": 3, "apply": False})
    assert verdict.decision is Decision.ALLOW
    assert verdict.risk is Risk.LOW


def test_single_task_delete_needs_approval_medium():
    verdict = evaluate("clean_exports", {"task_id": 3, "apply": True})
    assert verdict.decision is Decision.REQUIRE_APPROVAL
    assert verdict.risk is Risk.MEDIUM
    assert verdict.approvals_needed == 1


def test_batch_delete_needs_approval_high():
    verdict = evaluate("clean_exports", {"keep": 2, "apply": True})
    assert verdict.decision is Decision.REQUIRE_APPROVAL
    assert verdict.risk is Risk.HIGH


def test_cache_clean_needs_approval():
    verdict = evaluate("clean_cache", {"apply": True})
    assert verdict.decision is Decision.REQUIRE_APPROVAL
    assert verdict.risk is Risk.MEDIUM


def test_purge_needs_approval_high():
    verdict = evaluate("purge_trash", {"older_than_days": 30, "apply": True})
    assert verdict.decision is Decision.REQUIRE_APPROVAL
    assert verdict.risk is Risk.HIGH
    assert "不可逆" in "".join(verdict.reasons)


def test_apply_without_scope_is_blocked():
    """apply=true 但没给范围 ≈ 全删，必须阻断。"""
    verdict = evaluate("clean_exports", {"apply": True})
    assert verdict.decision is Decision.BLOCK


# ── 确定性：同样输入永远同样结论 ──────────────────────────
def test_evaluation_is_deterministic():
    args = {"task_id": 7, "apply": True}
    first = evaluate("clean_exports", args)
    for _ in range(5):
        got = evaluate("clean_exports", args)
        assert got.decision == first.decision
        assert got.risk == first.risk
