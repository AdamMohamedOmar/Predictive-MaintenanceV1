"""Tests for src/api/service.process_upload (Task 2.1).

The score path is skipped when trained models are absent.
The baseline path is tested with both a warm/moving CSV (should succeed)
and a cold/idle CSV (should raise ValueError).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import USEFUL_PIDS

_REPO = Path(__file__).resolve().parents[2]
_MODELS_PRESENT = (_REPO / "models" / "xgb_classifier_v1.pkl").exists()
_AHMED_CSV = _REPO / "data" / "real_faults" / "ahmed" / "ahmed_drive_20260602.csv"


def _make_warm_csv(dest: Path, n: int = 350) -> Path:
    rng = np.random.default_rng(7)
    data = {p: rng.uniform(10, 50, n) for p in USEFUL_PIDS}
    data["VEHICLE_SPEED"] = rng.uniform(20, 80, n)
    data["COOLANT_TEMPERATURE"] = np.full(n, 90.0)
    data["ENGINE_RPM"] = rng.uniform(800, 2500, n)
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _make_cold_csv(dest: Path, n: int = 350) -> Path:
    data = {p: np.zeros(n) for p in USEFUL_PIDS}
    data["ENGINE_RPM"] = np.full(n, 800.0)
    data["COOLANT_TEMPERATURE"] = np.full(n, 30.0)
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


@pytest.mark.skipif(not _MODELS_PRESENT, reason="trained models not present")
@pytest.mark.skipif(not _AHMED_CSV.exists(), reason="ahmed CSV not present")
def test_process_upload_score_mode(tmp_path):
    """Score mode returns mode='score' with result windows and inspect report."""
    from src.api.service import process_upload

    out = tmp_path / "car1"
    result = process_upload(
        raw_csv=_AHMED_CSV,
        out_dir=out,
        normalizer_path=None,
        is_baseline=False,
        vehicle_name="ahmed_test",
    )

    assert result["mode"] == "score"
    assert "inspect" in result
    assert result["inspect"]["metering_type"] != ""
    assert result["result"]["n_windows"] > 0
    assert (out / "adapted.csv").exists()
    assert (out / "result.json").exists()


def test_process_upload_baseline_warm(tmp_path):
    """Baseline mode on a warm/moving CSV saves a normalizer .pkl."""
    from src.api.service import process_upload

    csv = _make_warm_csv(tmp_path / "warm.csv")
    out = tmp_path / "car1"
    result = process_upload(
        raw_csv=csv,
        out_dir=out,
        normalizer_path=None,
        is_baseline=True,
        vehicle_name="test_car",
    )

    assert result["mode"] == "baseline"
    assert Path(result["normalizer_path"]).exists()
    assert result["n_windows"] is not None and result["n_windows"] >= 20


def test_process_upload_baseline_cold_raises(tmp_path):
    """Baseline mode on a cold/idle CSV raises ValueError (guard check)."""
    from src.api.service import process_upload

    csv = _make_cold_csv(tmp_path / "cold.csv")
    out = tmp_path / "car1"
    with pytest.raises(ValueError, match=""):
        process_upload(
            raw_csv=csv,
            out_dir=out,
            normalizer_path=None,
            is_baseline=True,
        )
