"""Auth router — /api/auth/signup and /api/auth/login."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.auth import hash_password, make_token, verify_password
from src.api.db import get_db
from src.api.models import User
from src.api.schemas import TokenOut, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def signup(body: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account and return a session token."""
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        created_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenOut(
        access_token=make_token(user.id),
        user=UserOut.model_validate(user),
    )


@router.post("/login", response_model=TokenOut)
def login(body: UserCreate, db: Session = Depends(get_db)):
    """Authenticate an existing user and return a session token."""
    user = db.query(User).filter_by(username=body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    return TokenOut(
        access_token=make_token(user.id),
        user=UserOut.model_validate(user),
    )
