"""缓存巡检模块的纯逻辑测试（不连 Redis）。"""

from __future__ import annotations

from core.cache_maintenance import (
    PATTERNS,
    CleanupReport,
    PatternScan,
    key_version,
)


def test_key_version_parses_all_shapes():
    assert key_version("qa:ans:v34:sabc:def") == 34
    assert key_version("qa:ans:research:deep:v34:sabc") == 34
    assert key_version("faq:questions:v34") == 34
    assert key_version("faq:tokens:v7") == 7


def test_key_version_ignores_non_data_keys():
    assert key_version("qa:cache:version") is None
    assert key_version("something:else") is None


def test_patterns_cover_three_kinds():
    assert len(PATTERNS) == 3
    assert any("qa:ans" in p for p in PATTERNS)
    assert any("faq:questions" in p for p in PATTERNS)
    assert any("faq:tokens" in p for p in PATTERNS)


def test_pattern_label_strips_wildcard():
    scan = PatternScan(pattern="qa:ans:*")
    assert scan.label == "qa:ans:"


def test_report_totals():
    report = CleanupReport(
        version=35,
        scans=[
            PatternScan(pattern="qa:ans:*", stale=["a", "b"], kept=3),
            PatternScan(pattern="faq:questions:*", stale=[], kept=1),
        ],
    )
    assert report.stale_total == 2
    assert report.kept_total == 4
