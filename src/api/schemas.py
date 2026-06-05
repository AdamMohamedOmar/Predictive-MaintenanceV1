"""Pydantic v2 request/response schemas for the PM web app API.

*Out schemas use `from_attributes=True` so they can be constructed directly
from SQLAlchemy ORM model instances with `model_validate(orm_obj)`.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: Optional[str] = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Cars (garage) ─────────────────────────────────────────────────────────────

class CarCreate(BaseModel):
    make: str
    model: str
    year: int
    engine_metering: str = "unknown"    # 'speed_density' | 'maf' | 'unknown'


class CarOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    make: str
    model: str
    year: int
    engine_metering: str
    baseline_normalizer_path: Optional[str] = None
    created_at: Optional[str] = None


# ── Recordings ────────────────────────────────────────────────────────────────

class RecordingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    car_id: int
    kind: str
    original_filename: Optional[str] = None
    adapted_csv_path: Optional[str] = None
    result_json_path: Optional[str] = None
    label_summary: Optional[str] = None   # JSON string; client parses
    anomaly_mean: Optional[float] = None
    recall: Optional[float] = None
    fault_from_s: Optional[int] = None
    fault_to_s: Optional[int] = None
    created_at: Optional[str] = None


class RecordingDetail(BaseModel):
    """Full result payload returned by GET /api/recordings/{id}."""
    recording: RecordingOut
    result: Optional[dict[str, Any]] = None   # full evaluate_real_fault output
    inspect: Optional[dict[str, Any]] = None  # inspect_recording report


# ── Baseline ──────────────────────────────────────────────────────────────────

class BaselineOut(BaseModel):
    mode: str = "baseline"
    normalizer_path: str
    n_windows: Optional[int] = None
    message: str = "Baseline captured successfully."
