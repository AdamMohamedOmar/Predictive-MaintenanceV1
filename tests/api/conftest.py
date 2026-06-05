"""Shared fixtures for API tests.

Provides a TestClient wired to an in-memory SQLite database so tests
never touch the real data/app/app.db.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """FastAPI TestClient backed by a temp SQLite file."""
    import src.api.config as cfg
    import src.api.db as db_module

    # Point config at a temp location
    tmp_db = tmp_path / "test_app.db"
    monkeypatch.setattr(cfg, "DB_URL", f"sqlite:///{tmp_db}")
    monkeypatch.setattr(cfg, "DATA_APP_DIR", tmp_path)

    # Rebuild the engine against the temp DB
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine(
        f"sqlite:///{tmp_db}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(db_module, "_engine", test_engine)
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=test_engine),
    )

    from src.api.main import create_app
    app = create_app()   # startup event runs init_db() with monkeypatched config

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
