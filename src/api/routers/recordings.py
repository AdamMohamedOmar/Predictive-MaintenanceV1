"""Recordings router — CSV upload, baseline capture, scoring, retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

import pandas as pd

from src.api.auth import current_user
from src.api.config import DATA_APP_DIR
from src.api.db import get_db
from src.api.models import Car, Recording, User
from src.api.routers.cars import _own_car
from src.api.schemas import BaselineOut, RecordingDetail, RecordingOut
from src.config import USEFUL_PIDS

router = APIRouter(tags=["recordings"])


def _car_dir(user_id: int, car_id: int) -> Path:
    return DATA_APP_DIR / "users" / str(user_id) / "cars" / str(car_id)



@router.post(
    "/cars/{car_id}/recordings",
    status_code=status.HTTP_201_CREATED,
)
async def upload_recording(
    car_id: int,
    file: UploadFile = File(...),
    is_baseline: bool = Form(False),
    fault_from_s: Optional[int] = Form(None),
    fault_to_s: Optional[int] = Form(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Upload a CSV recording for a car.

    Set is_baseline=true on a healthy drive to capture the per-car normalizer.
    Set fault_from_s / fault_to_s to compute recall over the fault interval.
    Returns a BaselineOut (baseline mode) or RecordingOut (score mode).
    """
    car: Car = _own_car(car_id, user, db)
    car_dir = _car_dir(user.id, car.id)
    car_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw upload
    raw_path = car_dir / f"raw_{file.filename}"
    content = await file.read()
    raw_path.write_bytes(content)

    from src.api.service import process_upload

    normalizer_path = Path(car.baseline_normalizer_path) if car.baseline_normalizer_path else None
    vehicle_name = f"{car.make} {car.model} {car.year}".strip()

    try:
        result = process_upload(
            raw_csv=raw_path,
            out_dir=car_dir,
            normalizer_path=normalizer_path,
            is_baseline=is_baseline,
            vehicle_name=vehicle_name,
        )
    except ValueError as exc:
        # Guard check failed (cold/idle/too-short for baseline, or adapt error)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    if result["mode"] == "baseline":
        # Update the car's normalizer path in the DB
        car.baseline_normalizer_path = result["normalizer_path"]
        db.commit()
        return BaselineOut(
            normalizer_path=result["normalizer_path"],
            n_windows=result.get("n_windows"),
            message=(
                f"Baseline captured: {result.get('n_windows', '?')} windows. "
                f"Future recordings for this car will use this normalizer."
            ),
        )

    # Score mode — persist a Recording row
    score = result["result"]
    summary = score.get("summary", {})
    label_summary = json.dumps(summary.get("label_counts", {}))
    anomaly_mean = None
    windows = score.get("windows", [])
    if windows:
        anomaly_mean = sum(w.get("anomaly_score", 0.0) for w in windows) / len(windows)

    recall = None
    recall_detail = None
    if fault_from_s is not None and fault_to_s is not None:
        from src.eval.real_fault_eval import compute_fault_recall

        recall_detail = compute_fault_recall(windows, fault_from_s, fault_to_s)
        recall = recall_detail["recall"]

    rec = Recording(
        car_id=car.id,
        kind="csv",
        original_filename=file.filename,
        adapted_csv_path=result.get("adapted_csv"),
        result_json_path=result.get("result_json"),
        label_summary=label_summary,
        anomaly_mean=anomaly_mean,
        recall=recall,
        fault_from_s=fault_from_s,
        fault_to_s=fault_to_s,
        created_at=now,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return RecordingOut.model_validate(rec)


@router.get("/recordings/{recording_id}", response_model=RecordingDetail)
def get_recording(
    recording_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Fetch full result for a recording (owner-checked).

    Returns the recording metadata + the full result JSON (all windows with
    label, confidence, severities, forecasts, anomaly_score, top_shap).
    """
    rec = db.get(Recording, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found.")

    # Owner check via car
    car = db.get(Car, rec.car_id)
    if car is None or car.user_id != user.id:
        raise HTTPException(status_code=404, detail="Recording not found.")

    result = None
    if rec.result_json_path and Path(rec.result_json_path).exists():
        result = json.loads(Path(rec.result_json_path).read_text())

    inspect = None   # stored separately if needed; for now omit from detail

    return RecordingDetail(
        recording=RecordingOut.model_validate(rec),
        result=result,
        inspect=inspect,
    )


@router.get("/recordings/{recording_id}/rows")
def get_recording_rows(
    recording_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Return downsampled PID rows for post-hoc SensorTimeline rendering."""
    rec = db.get(Recording, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found.")
    car = db.get(Car, rec.car_id)
    if car is None or car.user_id != user.id:
        raise HTTPException(status_code=404, detail="Recording not found.")
    if not rec.adapted_csv_path or not Path(rec.adapted_csv_path).exists():
        raise HTTPException(status_code=404, detail="No adapted CSV for this recording.")

    df = pd.read_csv(rec.adapted_csv_path)
    stride = max(1, len(df) // 1200)  # cap payload ~1200 points for charting
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    rows = []
    for i in range(0, len(df), stride):
        r: dict = {"elapsed_s": int(i)}
        for p in pid_cols:
            v = df[p].iloc[i]
            r[p] = None if pd.isna(v) else float(v)
        rows.append(r)
    return {"rows": rows, "stride_s": stride, "n_total": len(df)}


@router.get("/cars/{car_id}/recordings", response_model=list[RecordingOut])
def list_recordings(
    car_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List all recordings for a car (owner-checked)."""
    car = _own_car(car_id, user, db)
    return db.query(Recording).filter_by(car_id=car.id).order_by(Recording.id.desc()).all()
