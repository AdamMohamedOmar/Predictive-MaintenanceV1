"""T3.1 regression: InferenceEngine must resample live rows to exactly 1 Hz.

Two cases:
  1. Slow adapter (0.3 Hz): 18 raw rows spanning 60 s should produce ~60
     1-second buffer ticks so the 60-row window covers 60 real seconds.
  2. Fast adapter (2 Hz): every other row is dropped so elapsed_s only
     advances once per real second.
"""

import pytest
from src.dashboard.inference import InferenceEngine


def _make_row(rpm: float = 900.0) -> dict:
    """Minimal sensor row with no __t (CSV-mode).  Add __t for live-mode tests."""
    return {
        "ENGINE_RPM": rpm,
        "VEHICLE_SPEED": 30.0,
        "THROTTLE": 15.0,
        "ENGINE_LOAD": 25.0,
        "COOLANT_TEMPERATURE": 90.0,
        "LONG_TERM_FUEL_TRIM_BANK_1": 0.0,
        "SHORT_TERM_FUEL_TRIM_BANK_1": 0.0,
        "INTAKE_MANIFOLD_PRESSURE": 40.0,
        "ACCELERATOR_PEDAL_POSITION_D": 15.0,
        "ACCELERATOR_PEDAL_POSITION_E": 15.0,
        "COMMANDED_THROTTLE_ACTUATOR": 15.0,
        "INTAKE_AIR_TEMPERATURE": 25.0,
        "TIMING_ADVANCE": 15.0,
        "CONTROL_MODULE_VOLTAGE": 14.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Return an InferenceEngine with model loading mocked out."""
    import numpy as np
    from unittest.mock import MagicMock
    from src.dashboard import inference as inf_module

    # --- mock load_xgb_model ---
    fake_clf = MagicMock()
    fake_norm = MagicMock()
    fake_norm.feature_means = np.zeros(83)
    monkeypatch.setattr(inf_module, "load_xgb_model", lambda p: (fake_clf, fake_norm))

    # --- mock SHAPExplainer ---
    fake_explainer = MagicMock()
    fake_explainer.explain_window.return_value = {
        "predicted_label": "healthy",
        "probabilities": {"healthy": 1.0},
        "top_features": [],
    }
    monkeypatch.setattr(inf_module, "SHAPExplainer", lambda clf: fake_explainer)

    # --- mock FaultForecaster ---
    fake_forecaster = MagicMock()
    fake_forecaster.predict_all.return_value = {}
    monkeypatch.setattr(
        inf_module.FaultForecaster, "load", staticmethod(lambda p: fake_forecaster)
    )

    eng = InferenceEngine(models_dir=tmp_path)
    return eng


def test_slow_adapter_fills_buffer_with_held_rows(engine):
    """At 0.3 Hz, each raw row should replicate into ~3 buffer ticks.

    18 raw rows at 3.333 s intervals span t=0 to t=56.67 s (17 intervals ×
    3.333 s).  The resampler should fill ~57 one-second slots — one per real
    second — rather than just 18 slots (one per raw row).
    """
    for i in range(18):
        row = _make_row()
        row["__t"] = float(i) * 3.333
        engine.update(row)

    # 18 raw rows × ~3.3 ticks each ≈ 57 buffer entries
    # (17 intervals × 3.333 s = 56.67 s elapsed)
    # Confirm 3× amplification vs raw-row counting — key correctness property.
    assert 55 <= engine._elapsed_s <= 60, (
        f"Expected ~57 buffer ticks from 18 rows at 0.3 Hz, got {engine._elapsed_s}"
    )
    assert engine._elapsed_s > 18 * 2, (
        f"Resampler not working — elapsed_s={engine._elapsed_s} is too close to "
        f"raw row count (18). Each row should produce ~3 buffer ticks at 0.3 Hz."
    )


def test_fast_adapter_drops_extra_rows(engine):
    """At 2 Hz, every other row must be dropped so elapsed_s == real seconds.

    200 raw rows at 0.5 s intervals span 100 s.  The resampler should
    advance elapsed_s to ~100 (one buffer tick per real second).
    """
    for i in range(200):
        row = _make_row()
        row["__t"] = float(i) * 0.5
        engine.update(row)

    assert 98 <= engine._elapsed_s <= 102, (
        f"Expected ~100 buffer ticks from 200 rows at 2 Hz, got {engine._elapsed_s}"
    )


def test_csv_path_unchanged(engine):
    """Rows without __t must bypass the resampler entirely (CSV replay).

    100 rows with no __t key must yield elapsed_s == 100.
    """
    for _ in range(100):
        engine.update(_make_row())

    assert engine._elapsed_s == 100


def test_reset_restarts_resampler_clock(engine):
    """After reset(), the next live row anchors a fresh 1-Hz clock."""
    for i in range(10):
        row = _make_row()
        row["__t"] = float(i) * 3.0
        engine.update(row)

    engine.reset()
    assert engine._next_sample_t is None
    assert engine._elapsed_s == 0

    # Feed one row after reset — should start fresh
    row = _make_row()
    row["__t"] = 1000.0  # far-future timestamp — no carry-over from old session
    engine.update(row)
    assert engine._elapsed_s == 1
