"""Tests for scripts/capture_baseline_from_csv.py (P0-1).

Verifies that a warm, moving adapted CSV produces a valid BaselineNormalizer
and that an idle/cold CSV triggers the existing guard checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import USEFUL_PIDS
from src.features.normalizer import BaselineNormalizer


def _make_warm_driving_csv(dest: Path, n_rows: int = 350) -> Path:
    """Synthetic 1 Hz adapted CSV: warm engine (coolant >= 75 C), moving."""
    rng = np.random.default_rng(42)
    n = n_rows
    data = {
        "ENGINE_RPM": rng.uniform(800, 2500, n),
        "VEHICLE_SPEED": rng.uniform(20, 80, n),       # mean > 15 km/h guard
        "THROTTLE": rng.uniform(5, 60, n),
        "ENGINE_LOAD": rng.uniform(20, 70, n),
        "COOLANT_TEMPERATURE": 90.0 + rng.normal(0, 0.3, n),    # >= 75 C, ±0.3 °C engine jitter
        "LONG_TERM_FUEL_TRIM_BANK_1": rng.uniform(-3, 3, n),
        "SHORT_TERM_FUEL_TRIM_BANK_1": rng.uniform(-2, 2, n),
        "INTAKE_MANIFOLD_PRESSURE": rng.uniform(30, 80, n),
        "ACCELERATOR_PEDAL_POSITION_D": rng.uniform(10, 60, n),
        "ACCELERATOR_PEDAL_POSITION_E": rng.uniform(10, 60, n),
        "COMMANDED_THROTTLE_ACTUATOR": rng.uniform(5, 60, n),
        "INTAKE_AIR_TEMPERATURE": rng.uniform(20, 40, n),
        "TIMING_ADVANCE": rng.uniform(8, 25, n),
        "CONTROL_MODULE_VOLTAGE": rng.uniform(13.5, 14.5, n),
    }
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _make_idle_cold_csv(dest: Path, n_rows: int = 350) -> Path:
    """Synthetic CSV that must fail guards: speed = 0, coolant < 75 C."""
    n = n_rows
    data = {
        "ENGINE_RPM": np.full(n, 800.0),
        "VEHICLE_SPEED": np.zeros(n),       # mean speed = 0 -> guard fails
        "THROTTLE": np.zeros(n),
        "ENGINE_LOAD": np.full(n, 20.0),
        "COOLANT_TEMPERATURE": np.full(n, 30.0),    # cold -> guard also fails
        "LONG_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
        "SHORT_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
        "INTAKE_MANIFOLD_PRESSURE": np.full(n, 35.0),
        "ACCELERATOR_PEDAL_POSITION_D": np.zeros(n),
        "ACCELERATOR_PEDAL_POSITION_E": np.zeros(n),
        "COMMANDED_THROTTLE_ACTUATOR": np.zeros(n),
        "INTAKE_AIR_TEMPERATURE": np.full(n, 20.0),
        "TIMING_ADVANCE": np.full(n, 5.0),
        "CONTROL_MODULE_VOLTAGE": np.full(n, 14.0),
    }
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


class TestCaptureBaselineFromCsv:
    def test_warm_drive_produces_saved_normalizer(self, tmp_path):
        """A warm, moving, sufficiently long CSV produces a normalizer that saves
        and re-loads as a valid BaselineNormalizer."""
        from scripts.capture_baseline_from_csv import capture_baseline_from_csv

        csv = _make_warm_driving_csv(tmp_path / "warm_drive.csv", n_rows=350)
        out = tmp_path / "models" / "test_normalizer.pkl"

        result = capture_baseline_from_csv(csv, vehicle_name="Test Car 2024", out_path=out)

        assert result == out
        assert out.exists(), "normalizer .pkl was not written"

        # Re-load and verify it is a functioning normalizer
        norm = BaselineNormalizer.load(out)
        assert norm is not None

        # Sidecar JSON must exist and carry vehicle metadata
        meta_path = out.with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["vehicle"] == "Test Car 2024"
        assert meta["n_windows"] >= 20

    def test_cold_idle_csv_raises_value_error(self, tmp_path):
        """An idle/cold CSV triggers the guard and raises ValueError.
        Nothing should be written on failure."""
        from scripts.capture_baseline_from_csv import capture_baseline_from_csv

        csv = _make_idle_cold_csv(tmp_path / "idle_cold.csv", n_rows=350)
        out = tmp_path / "models" / "bad_normalizer.pkl"

        with pytest.raises(ValueError):
            capture_baseline_from_csv(csv, vehicle_name="Bad Session", out_path=out)

        # Guard failed -> no artifacts should have been created
        assert not out.exists()

    def test_default_out_path_uses_vehicle_slug(self, tmp_path, monkeypatch):
        """When --out is omitted, the output goes to models/<slug>_normalizer.pkl."""
        import src.config as cfg
        from scripts.capture_baseline_from_csv import capture_baseline_from_csv

        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")

        csv = _make_warm_driving_csv(tmp_path / "drive.csv", n_rows=350)
        out = capture_baseline_from_csv(csv, vehicle_name="My Test Vehicle")

        assert out.name == "my_test_vehicle_normalizer.pkl"
        assert out.exists()
