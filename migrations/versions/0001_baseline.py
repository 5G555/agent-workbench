"""baseline：既有表结构的起点

现有 10 张表（users / qa_sessions / qa_messages / qa_feedback / faq_entries /
crawl_tasks / crawl_pages / crawl_records / ops_audit / route_feedback）
历史上由各模块的 CREATE TABLE IF NOT EXISTS 建立。

这里不再重复建表，只把当前库标记为 head（alembic stamp head），
之后的表结构变更一律通过 revision 叠加。

revision = 0001_baseline
"""

from __future__ import annotations

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """空：既有表已存在。"""


def downgrade() -> None:
    """空：baseline 不撤销既有表。"""
