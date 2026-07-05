"""SQLAlchemy engine, session, Base, and FastAPI dependency for the PM web app."""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.api.config import DATA_APP_DIR, DB_URL


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def _make_engine(db_url: str = DB_URL):
    # check_same_thread=False is required for SQLite when FastAPI's async
    # request-handling calls the sync session from multiple threads.
    return create_engine(db_url, connect_args={"check_same_thread": False})


_engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def init_db(db_url: str | None = None) -> None:
    """Create all tables if they don't exist.

    Accepts an optional db_url so tests can point at a temp file.
    """
    # Import models so their __tablename__ registrations are visible to Base.
    import src.api.models  # noqa: F401  (side-effect: registers ORM models)

    if db_url and db_url != DB_URL:
        engine = _make_engine(db_url)
    else:
        DATA_APP_DIR.mkdir(parents=True, exist_ok=True)
        engine = _engine

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
