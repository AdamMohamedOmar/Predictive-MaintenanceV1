"""Tests for src/api/db.py — SQLite init and session dependency."""

from __future__ import annotations


from sqlalchemy import inspect, text


def test_init_db_creates_tables(tmp_path):
    """init_db() must create all three tables on a fresh SQLite file."""
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"

    from src.api.db import init_db, _make_engine

    init_db(db_url)

    engine = _make_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"users", "cars", "recordings"}.issubset(tables), (
        f"Expected users/cars/recordings, got: {tables}"
    )


def test_get_db_yields_session(tmp_path, monkeypatch):
    """get_db() must yield a working session that can execute a simple query."""
    import src.api.config as cfg
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(cfg, "DB_URL", f"sqlite:///{db_file}")
    monkeypatch.setattr(cfg, "DATA_APP_DIR", tmp_path)

    from src.api.db import init_db, get_db
    init_db()

    gen = get_db()
    db = next(gen)
    result = db.execute(text("SELECT 1")).scalar()
    assert result == 1
    try:
        next(gen)
    except StopIteration:
        pass  # expected: generator closes the session
