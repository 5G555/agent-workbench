"""缓存清理脚本的纯逻辑测试。"""

from __future__ import annotations

from scripts.clean_cache import key_version


def test_versions_from_plain_qa_keys():
    assert key_version("qa:ans:v27:s1a2b3:c4d5e6") == 27
    assert key_version("qa:ans:v1:abc") == 1


def test_versions_from_namespaced_keys():
    # research 的键带命名空间和深度
    assert key_version("qa:ans:research:deep:v34:s1a2b:c3d4") == 34
    assert key_version("qa:ans:research:fast:v7:abc") == 7


def test_versions_from_faq_index_keys():
    assert key_version("faq:questions:v12") == 12
    assert key_version("faq:tokens:v9") == 9


def test_key_without_version_is_ignored():
    """版本号解析不出来时返回 None，调用方会跳过（宁可不删）。"""
    assert key_version("qa:ans:version") is None
    assert key_version("some:other:key") is None


def test_version_key_itself_is_not_mistaken():
    """qa:cache:version 这类键不该被当成某个版本的数据键。"""
    assert key_version("qa:cache:version") is None
