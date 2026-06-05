"""Smoke tests for the FastAPI app skeleton (Task 0.5).

Tests:
  - GET /api/health returns {"status": "ok"}
  - POST /api/_smoke scores the ahmed adapted CSV (skipped if models absent)
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_AHMED_CSV = _REPO / "data" / "real_faults" / "ahmed" / "ahmed_drive_20260602.csv"
_AHMED_NORM = _REPO / "models" / "ahmed_normalizer.pkl"
_MODELS_PRESENT = (_REPO / "models" / "xgb_classifier_v1.pkl").exists()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(not _MODELS_PRESENT, reason="trained models not present")
@pytest.mark.skipif(not _AHMED_CSV.exists(), reason="ahmed CSV not present")
def test_smoke_scores_ahmed_csv(client):
    """The smoke endpoint must score the ahmed recording and return a label summary."""
    params = {"csv_path": str(_AHMED_CSV)}
    if _AHMED_NORM.exists():
        params["normalizer_path"] = str(_AHMED_NORM)

    resp = client.post("/api/_smoke", params=params)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "label_counts" in body
    assert body["fault_window_count"] >= 0   # can be 0 for a healthy recording
    assert sum(body["label_counts"].values()) > 0
