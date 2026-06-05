"""SQLAlchemy ORM models for the PM web app.

Three tables: users → cars → recordings (all owned/scoped per-user).
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import mapped_column

from src.api.db import Base


class User(Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True)
    username = mapped_column(String, unique=True, nullable=False)
    password_hash = mapped_column(String, nullable=False)
    created_at = mapped_column(String)


class Car(Base):
    __tablename__ = "cars"

    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    make = mapped_column(String)
    model = mapped_column(String)
    year = mapped_column(Integer)
    # 'speed_density' | 'maf' | 'unknown' — drives caveat visibility in the UI
    engine_metering = mapped_column(String, default="unknown")
    # Nullable until a healthy baseline drive is uploaded for this car
    baseline_normalizer_path = mapped_column(String, nullable=True)
    created_at = mapped_column(String)


class Recording(Base):
    __tablename__ = "recordings"

    id = mapped_column(Integer, primary_key=True)
    car_id = mapped_column(Integer, ForeignKey("cars.id"), nullable=False)
    kind = mapped_column(String)                    # 'csv' | 'live'
    original_filename = mapped_column(String)
    adapted_csv_path = mapped_column(String, nullable=True)
    result_json_path = mapped_column(String, nullable=True)
    label_summary = mapped_column(String, nullable=True)    # JSON string {label: count}
    anomaly_mean = mapped_column(Float, nullable=True)
    recall = mapped_column(Float, nullable=True)            # fault-interval recall
    fault_from_s = mapped_column(Integer, nullable=True)
    fault_to_s = mapped_column(Integer, nullable=True)
    created_at = mapped_column(String)
