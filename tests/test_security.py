"""鉴权基础件测试：密码哈希与 JWT。全是纯函数，不碰网络和数据库。"""

from __future__ import annotations

import pytest

from core.exceptions import AuthenticationError
from core.security import create_access_token, decode_token, hash_password, verify_password


def test_hash_is_not_plaintext_and_verifies():
    stored = hash_password("s3cret-pwd")
    assert "s3cret-pwd" not in stored
    assert stored.startswith("scrypt$")
    assert verify_password("s3cret-pwd", stored) is True


def test_verify_rejects_wrong_password():
    stored = hash_password("s3cret-pwd")
    assert verify_password("wrong", stored) is False
    assert verify_password("", stored) is False


def test_same_password_gets_different_hash():
    """每次哈希都要用新盐，否则相同密码在库里长得一样。"""
    assert hash_password("same") != hash_password("same")


def test_token_roundtrip():
    token, expires_in = create_access_token(user_id="42", role="admin", tenant_id="t1")
    assert expires_in > 0
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"
    assert payload["tenant_id"] == "t1"


def test_expired_token_rejected():
    token, _ = create_access_token(user_id="42", expires_minutes=-1)
    with pytest.raises(AuthenticationError):
        decode_token(token)


def test_tampered_token_rejected():
    token, _ = create_access_token(user_id="42")
    broken = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
    with pytest.raises(AuthenticationError):
        decode_token(broken)
