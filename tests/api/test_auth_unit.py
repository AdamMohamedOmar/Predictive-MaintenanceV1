"""Unit tests for src/api/auth.py primitives (no DB required)."""

from __future__ import annotations

import pytest
import jwt as pyjwt

from src.api.auth import decode_token, hash_password, make_token, verify_password


def test_hash_is_not_plaintext():
    hashed = hash_password("hunter2")
    assert hashed != "hunter2"
    assert len(hashed) > 20   # bcrypt output is 60 chars


def test_verify_password_roundtrip():
    pw = "correct-horse-battery-staple"
    hashed = hash_password(pw)
    assert verify_password(pw, hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("right")
    assert verify_password("wrong", hashed) is False


def test_token_roundtrips_to_user_id():
    token = make_token(42)
    assert decode_token(token) == 42


def test_token_is_string():
    assert isinstance(make_token(1), str)


def test_tampered_token_raises():
    token = make_token(99)
    # Flip the last character to corrupt the signature
    corrupted = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(pyjwt.PyJWTError):
        decode_token(corrupted)


def test_different_users_get_different_tokens():
    assert make_token(1) != make_token(2)
