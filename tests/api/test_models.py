"""Tests for src/api/models.py and src/api/schemas.py."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.db import Base
from src.api.models import Car, Recording, User
from src.api.schemas import CarOut, RecordingOut, UserOut


@pytest.fixture()
def db(tmp_path):
    """Temp SQLite session for model tests."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_user_car_recording_roundtrip(db):
    """Create a User + Car + Recording; query back; round-trip through schemas."""
    now = datetime.utcnow().isoformat()

    user = User(username="adam", password_hash="hashed", created_at=now)
    db.add(user)
    db.flush()

    car = Car(user_id=user.id, make="Skoda", model="Roomster",
              year=2007, engine_metering="maf", created_at=now)
    db.add(car)
    db.flush()

    rec = Recording(
        car_id=car.id, kind="csv", original_filename="drive.csv",
        label_summary=json.dumps({"healthy": 40, "fuel_system": 5}),
        anomaly_mean=0.42, created_at=now,
    )
    db.add(rec)
    db.commit()

    # Query back
    fetched_user = db.query(User).filter_by(username="adam").one()
    fetched_car = db.query(Car).filter_by(user_id=fetched_user.id).one()
    fetched_rec = db.query(Recording).filter_by(car_id=fetched_car.id).one()

    assert fetched_user.username == "adam"
    assert fetched_car.make == "Skoda"
    assert fetched_rec.anomaly_mean == pytest.approx(0.42)

    # Pydantic round-trip
    user_out = UserOut.model_validate(fetched_user)
    assert user_out.id == fetched_user.id

    car_out = CarOut.model_validate(fetched_car)
    assert car_out.engine_metering == "maf"
    assert car_out.baseline_normalizer_path is None

    rec_out = RecordingOut.model_validate(fetched_rec)
    assert rec_out.kind == "csv"
    assert rec_out.anomaly_mean == pytest.approx(0.42)
    assert rec_out.recall is None   # not set
