"""数据性质路由与权限边界的测试（纯逻辑，不联网）。"""

from __future__ import annotations

from agents.research.evidence import Evidence, EvidenceSet
from agents.supervisor.privacy import is_private, sanitize


# ── 权限闸门 ───────────────────────────────────────────────
def test_private_questions_are_detected():
    blocked = [
        "我们公司年假有几天",
        "我司的合同模板在哪",
        "你们的后台怎么登录",
        "我的订单号是 12345，能查下吗",
        "内部规定里报销要走什么流程",
        "员工号是 A001 的员工权限是什么",
    ]
    for query in blocked:
        is_p, _ = is_private(query)
        assert is_p, f"应判定为私有：{query}"


def test_public_questions_pass_the_gate():
    allowed = [
        "Milvus 支持哪些索引类型",
        "LangGraph 1.x 有什么变化",
        "什么是 RAG",
        "BGE-M3 和 bge-reranker 的区别",
    ]
    for query in allowed:
        is_p, _ = is_private(query)
        assert not is_p, f"不该被拦：{query}"


def test_gate_reports_reason():
    blocked, reason = is_private("我们公司的报销流程")
    assert blocked and reason


def test_sanitize_neutralizes_owner_words():
    assert "我们" not in sanitize("我们公司支持私有化部署吗")
    assert sanitize("你们的服务时间是什么") == "服务时间是什么"


# ── 统一证据结构 ───────────────────────────────────────────
def test_evidence_labels_source_type():
    kb = Evidence.from_kb("计费按坐席数", "02_计费与套餐", score=0.9)
    web = Evidence.from_web("计费说明", "https://x.com/a", "第三方说明", score=0.4)
    kb_line = kb.as_line(1)
    web_line = web.as_line(2)

    assert "内部文档" in kb_line
    assert "公网资料" in web_line
    assert "https://x.com/a" in web_line
    assert "链接：无" in kb_line


def test_evidence_set_separates_types():
    ev = EvidenceSet()
    ev.add(Evidence.from_kb("a", "doc1"))
    ev.add(Evidence.from_web("b", "http://b", "b"))
    assert ev.has_kb and ev.has_web
    assert len(ev.of_type("kb")) == 1
    assert len(ev.of_type("web")) == 1


def test_sources_labeled_marks_internal_vs_public():
    ev = EvidenceSet()
    ev.add(Evidence.from_kb("a", "03_售后与SLA"))
    ev.add(Evidence.from_web("某博客", "http://blog", "b"))
    labels = ev.sources_labeled()
    assert labels[0].startswith("[内部]")
    assert labels[1].startswith("[公网]")


def test_cited_by_maps_numbers_to_evidence():
    ev = EvidenceSet()
    ev.add(Evidence.from_kb("a", "doc1"))
    ev.add(Evidence.from_web("t", "http://b", "b"))
    answer = "第一条依据 [1]，第二条依据 [2]。"
    cited = ev.cited_by(answer)
    assert len(cited) == 2
    assert cited[1].url == "http://b"


def test_empty_evidence_set():
    ev = EvidenceSet()
    assert ev.empty
    assert ev.as_context() == ""
