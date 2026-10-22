"""SQL 守卫与探索统计的测试（纯逻辑，不连数据库）。"""

from __future__ import annotations

import pytest

from agents.analyst.nodes import field_distribution
from agents.analyst.sql_guard import MAX_ROWS, referenced_views, validate


# ── 放行的正常查询 ────────────────────────────────────────────
def test_simple_select_passes():
    result = validate("SELECT count(*) AS n FROM ana_qa_session")
    assert result.ok is True
    assert "limit" in result.sql.lower()  # 没写 LIMIT 会兜底补上


def test_existing_limit_not_duplicated():
    result = validate("SELECT task_id FROM ana_crawl_task LIMIT 5")
    assert result.ok is True
    assert result.sql.lower().count("limit") == 1


def test_cte_passes():
    sql = (
        "WITH t AS (SELECT task_id, records_extracted FROM ana_crawl_task) "
        "SELECT sum(records_extracted) AS total FROM t"
    )
    assert validate(sql).ok is True


def test_json_field_access_passes():
    sql = "SELECT data->>'author' AS author, count(*) AS n FROM ana_crawl_record GROUP BY 1"
    assert validate(sql).ok is True


# ── 拒绝的危险查询 ────────────────────────────────────────────
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM ana_qa_turn",  # SELECT *
        "DELETE FROM ana_qa_turn",  # DML
        "DROP VIEW ana_qa_turn",  # DDL
        "SELECT 1; DROP TABLE users",  # 多语句
        "SELECT count(*) FROM users",  # 绕过视图查基表
        "SELECT pg_read_file('/etc/passwd')",  # 系统函数
        "SELECT pg_sleep(10) FROM ana_qa_turn",  # 拖库
        "",  # 空
    ],
)
def test_dangerous_sql_rejected(sql):
    result = validate(sql)
    assert result.ok is False
    assert result.reason


def test_comment_cannot_hide_keyword():
    """用注释把关键字藏起来也要被拦。"""
    result = validate("SELECT count(*) FROM ana_qa_turn /* ; drop table users */")
    assert result.ok is False or "drop" not in result.sql.lower()


def test_referenced_views_extracts_names():
    sql = "SELECT a.task_id FROM ana_crawl_task a JOIN ana_crawl_page p ON p.task_id = a.task_id"
    assert set(referenced_views(sql)) == {"ana_crawl_task", "ana_crawl_page"}


def test_max_rows_is_bounded():
    assert 0 < MAX_ROWS <= 1000


# ── 探索统计 ──────────────────────────────────────────────────
def test_field_distribution_counts_values():
    records = [
        {"data": {"author": "A", "tag": "x"}},
        {"data": {"author": "A", "tag": "y"}},
        {"data": {"author": "B", "tag": "x"}},
        {"data": {"author": "", "tag": "x"}},  # 空值不计
    ]
    stats = {item["field"]: item for item in field_distribution(records)}
    assert stats["author"]["distinct"] == 2
    assert stats["author"]["top"][0] == {"value": "A", "count": 2}
    assert stats["tag"]["distinct"] == 2


def test_field_distribution_handles_dirty_data():
    records = [
        {"data": "not-a-dict"},  # 脏数据
        {"data": None},
        {},  # 没有 data 字段
        {"data": {"k": "v"}},
    ]
    stats = field_distribution(records)
    assert [item["field"] for item in stats] == ["k"]


def test_field_distribution_empty():
    assert field_distribution([]) == []
