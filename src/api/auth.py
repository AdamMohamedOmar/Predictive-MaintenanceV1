"""Auth helpers: password hashing, JWT token issue/verify, current_user dep.

This is mock auth — real enough for a demo (bcrypt hashes, signed tokens),
not production-grade (no refresh tokens, no revocation, no email verify).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from src.api.config import JWT_ALGO, JWT_EXPIRY_DAYS, JWT_SECRET
from src.api.db import get_db


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return a bcrypt hash of the plaintext password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Return True if password matches the stored hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────

def make_token(user_id: int) -> str:
    """Issue a JWT containing the user ID, expiring in JWT_EXPIRY_DAYS days."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(tz=timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> int:
    """Decode a JWT and return the user_id (int).

    Raises jwt.PyJWTError on invalid/expired tokens.
    """
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    return int(payload["sub"])


# ── FastAPI dependency ────────────────────────────────────────────────────────

def current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    """FastAPI dependency that extracts and validates the Bearer token.

    Returns the ORM User object for the authenticated caller.
    Raises HTTP 401 on missing, invalid, or expired tokens.
    """
    from src.api.models import User

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise credentials_exc

    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_id = decode_token(token)
    except jwt.PyJWTError:
        raise credentials_exc

    user = db.get(User, user_id)
    if user is None:
        raise credentials_exc

    return user
