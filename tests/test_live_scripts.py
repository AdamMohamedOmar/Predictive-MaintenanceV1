"""Tests for the pure (hardware-free) logic in the live CLI scripts."""

import math

import numpy as np
import pandas as pd
import pytest

from src.config import USEFUL_PIDS, WINDOW_LENGTH_S
from src.features.normalizer import BaselineNormalizer
from scripts.live_discover import MIN_POLL_HZ, MIN_SUPPORTED_PIDS, evaluate
from scripts.live_baseline_capture import (
    _MIN_COOLANT_TEMP,
    _MIN_MEAN_SPEED,
    _MIN_WINDOWS,
    process_captured_rows,
)


# ── live_discover: evaluate() ─────────────────────────────────────────────────

def test_evaluate_go_when_all_good():
    go, reasons = evaluate(n_supported=14, actual_hz=1.2)
    assert go is True
    assert reasons == []


def test_evaluate_fail_too_few_pids():
    go, reasons = evaluate(n_supported=10, actual_hz=1.2)
    assert go is False
    assert any("PID" in r or "pid" in r.lower() or str(10) in r for r in reasons)


def test_evaluate_fail_poll_rate_too_low():
    go, reasons = evaluate(n_supported=14, actual_hz=0.5)
    assert go is False
    assert any("Hz" in r or "poll" in r.lower() for r in reasons)


def test_evaluate_fail_both_issues():
    go, reasons = evaluate(n_supported=8, actual_hz=0.3)
    assert go is False
    assert len(reasons) == 2


def test_evaluate_exactly_at_thresholds():
    """Edge values exactly at the thresholds should pass."""
    go, reasons = evaluate(n_supported=MIN_SUPPORTED_PIDS, actual_hz=MIN_POLL_HZ)
    assert go is True


def test_evaluate_one_below_threshold():
    go, _ = evaluate(n_supported=MIN_SUPPORTED_PIDS - 1, actual_hz=MIN_POLL_HZ)
    assert go is False


# ── live_baseline_capture: process_captured_rows() ────────────────────────────

def _make_rows(
    n: int = 400,
    coolant: float = 85.0,
    speed: float = 40.0,
    rpm: float = 2000.0,
) -> list[dict]:
    """Synthesise n healthy rows suitable for baseline capture."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        row = {
            "ENGINE_RPM": rpm + rng.normal(0, 20),
            "VEHICLE_SPEED": speed + rng.normal(0, 3),
            "THROTTLE": 25.0 + rng.normal(0, 2),
            "ENGINE_LOAD": 30.0 + rng.normal(0, 2),
            "COOLANT_TEMPERATURE": coolant + rng.normal(0, 0.5),
            "LONG_TERM_FUEL_TRIM_BANK_1": 0.5 + rng.normal(0, 0.3),
            "SHORT_TERM_FUEL_TRIM_BANK_1": 0.3 + rng.normal(0, 0.5),
            "INTAKE_MANIFOLD_PRESSURE": 55.0 + rng.normal(0, 3),
            "ACCELERATOR_PEDAL_POSITION_D": 22.0 + rng.normal(0, 2),
            "ACCELERATOR_PEDAL_POSITION_E": 20.0 + rng.normal(0, 2),
            "COMMANDED_THROTTLE_ACTUATOR": 25.0 + rng.normal(0, 2),
            "INTAKE_AIR_TEMPERATURE": 28.0 + rng.normal(0, 1),
            "TIMING_ADVANCE": 14.0 + rng.normal(0, 1),
            "CONTROL_MODULE_VOLTAGE": 14.2 + rng.normal(0, 0.05),
        }
        rows.append(row)
    return rows


def test_process_returns_normalizer_and_metadata():
    rows = _make_rows(400)
    norm, meta = process_captured_rows(rows, vehicle_name="TestCar")
    assert isinstance(norm, BaselineNormalizer)
    assert norm.is_fitted
    assert isinstance(meta, dict)


def test_process_metadata_fields():
    rows = _make_rows(400)
    _, meta = process_captured_rows(rows, vehicle_name="TestCar")
    for key in ("vehicle", "capture_date", "n_rows", "n_windows",
                "supported_pids", "missing_pids", "feature_means", "feature_stds"):
        assert key in meta, f"metadata missing key: {key}"


def test_process_vehicle_name_stored():
    rows = _make_rows(400)
    _, meta = process_captured_rows(rows, vehicle_name="Skoda Roomster 2007")
    assert meta["vehicle"] == "Skoda Roomster 2007"


def test_process_n_rows_correct():
    rows = _make_rows(400)
    _, meta = process_captured_rows(rows)
    assert meta["n_rows"] == 400


def test_process_produces_valid_normalizer():
    """The fitted normaliser must not produce NaN on healthy rows."""
    rows = _make_rows(400)
    norm, _ = process_captured_rows(rows)
    # Build a test feature row and z-score it
    import pandas as pd
    from src.features.extractor import feature_names
    feat_names = feature_names()
    # Create a dummy feature row with realistic values
    feat_row = pd.DataFrame([{f: 1.0 for f in feat_names}])
    feat_row["label"] = "healthy"
    transformed = norm.transform(feat_row)
    from src.features.normalizer import normalised_feature_names
    z_cols = normalised_feature_names()
    z_values = transformed[z_cols].values[0]
    assert not any(math.isnan(v) for v in z_values), "NaN in z-scored output"


# ── Guard: too few rows ───────────────────────────────────────────────────────

def test_guard_too_few_rows():
    rows = _make_rows(n=30)  # less than WINDOW_LENGTH_S (60)
    with pytest.raises(ValueError, match="rows"):
        process_captured_rows(rows)


# ── Guard: cold engine ────────────────────────────────────────────────────────

def test_guard_cold_engine_raises():
    rows = _make_rows(n=400, coolant=40.0)  # never reaches 75°C
    with pytest.raises(ValueError, match=str(int(_MIN_COOLANT_TEMP))):
        process_captured_rows(rows)


def test_guard_warm_engine_passes():
    rows = _make_rows(n=400, coolant=88.0)
    norm, _ = process_captured_rows(rows)  # should not raise
    assert norm.is_fitted


# ── Guard: mostly idle ────────────────────────────────────────────────────────

def test_guard_idle_only_raises():
    rows = _make_rows(n=400, speed=2.0, rpm=800.0)  # idling
    with pytest.raises(ValueError, match="speed"):
        process_captured_rows(rows)


def test_guard_moving_passes():
    rows = _make_rows(n=400, speed=50.0)
    norm, _ = process_captured_rows(rows)
    assert norm.is_fitted


# ── Guard: too few windows ────────────────────────────────────────────────────

def test_guard_too_few_windows_raises():
    # 70 rows → only 1 window → below _MIN_WINDOWS
    rows = _make_rows(n=70, coolant=88.0, speed=40.0)
    with pytest.raises(ValueError, match="window"):
        process_captured_rows(rows)


def test_guard_enough_windows_passes():
    # _MIN_WINDOWS windows need at least WINDOW_LENGTH_S + (_MIN_WINDOWS - 1) * WINDOW_STRIDE_S rows
    from src.config import WINDOW_STRIDE_S
    min_rows = WINDOW_LENGTH_S + (_MIN_WINDOWS - 1) * WINDOW_STRIDE_S + 10
    rows = _make_rows(n=min_rows, coolant=88.0, speed=40.0)
    norm, _ = process_captured_rows(rows)
    assert norm.is_fitted


# ── NaN handling (unsupported PIDs) ──────────────────────────────────────────

def test_nan_pids_dont_crash():
    """Rows with some NaN PIDs (unsupported ECU) should produce a valid normaliser."""
    rows = _make_rows(n=400, coolant=88.0, speed=40.0)
    # Simulate two PIDs being unsupported (NaN for all rows)
    for row in rows:
        row["ACCELERATOR_PEDAL_POSITION_E"] = float("nan")
        row["COMMANDED_THROTTLE_ACTUATOR"] = float("nan")

    # Pass the explicit supported_pids list (what LiveObdSource.supported_pids returns)
    supported = [p for p in USEFUL_PIDS
                 if p not in ("ACCELERATOR_PEDAL_POSITION_E", "COMMANDED_THROTTLE_ACTUATOR")]
    norm, meta = process_captured_rows(rows, supported_pids=supported)
    assert norm.is_fitted
    assert "ACCELERATOR_PEDAL_POSITION_E" not in meta["supported_pids"]
    assert "ACCELERATOR_PEDAL_POSITION_E" in meta["missing_pids"]
