"""Tests for InferenceEngine — the per-row ML pipeline.

These tests use the real saved models (xgb_classifier_v1.pkl and
forecaster_v1.pkl) so they require a built model directory.  All tests
are skipped if the model files don't exist (fresh clone before first build).

The synthetic row fixture feeds constant 'healthy' sensor values so the
classifier will consistently predict 'healthy' once the buffer warms up.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from src.dashboard.inference import DashboardState, InferenceEngine, _initial_state
from src.models.classifier import ALL_LABELS
from src.models.forecaster import FAULT_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

_MODELS_EXIST = (
    (MODELS_DIR / "xgb_classifier_v1.pkl").exists()
    and (MODELS_DIR / "forecaster_v1.pkl").exists()
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _healthy_row() -> dict[str, float]:
    """A plausible warm-engine healthy row (post cold-start)."""
    return {
        "ENGINE_RPM": 900.0,
        "VEHICLE_SPEED": 0.0,
        "THROTTLE": 5.0,
        "ENGINE_LOAD": 22.0,
        "COOLANT_TEMPERATURE": 88.0,
        "LONG_TERM_FUEL_TRIM_BANK_1": 0.5,
        "SHORT_TERM_FUEL_TRIM_BANK_1": 0.3,
        "INTAKE_MANIFOLD_PRESSURE": 34.0,
        "ACCELERATOR_PEDAL_POSITION_D": 5.0,
        "ACCELERATOR_PEDAL_POSITION_E": 4.0,
        "COMMANDED_THROTTLE_ACTUATOR": 5.0,
        "INTAKE_AIR_TEMPERATURE": 28.0,
        "TIMING_ADVANCE": 13.5,
        "CONTROL_MODULE_VOLTAGE": 14.2,
    }


def _feed(engine: InferenceEngine, row: dict, n: int) -> DashboardState:
    """Feed n identical rows to the engine, return the final state."""
    state = None
    for _ in range(n):
        state = engine.update(row)
    return state


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_state_not_buffer_ready():
    state = _initial_state()
    assert state.buffer_ready is False


def test_initial_state_warming_up():
    state = _initial_state()
    assert state.classifier_label == "warming_up"


def test_initial_state_all_severities_zero():
    state = _initial_state()
    for ft in FAULT_TYPES:
        assert state.severities[ft] == 0.0


def test_initial_state_no_alerts():
    state = _initial_state()
    assert state.stable_alert.active is False
    assert state.rule_alerts == []


# ── DashboardState fields ─────────────────────────────────────────────────────

def test_dashboard_state_has_all_fields():
    state = _initial_state()
    assert hasattr(state, "elapsed_s")
    assert hasattr(state, "latest_row")
    assert hasattr(state, "buffer_ready")
    assert hasattr(state, "classifier_label")
    assert hasattr(state, "classifier_confidence")
    assert hasattr(state, "all_class_probs")
    assert hasattr(state, "severities")
    assert hasattr(state, "forecasts")
    assert hasattr(state, "stable_alert")
    assert hasattr(state, "rule_alerts")
    assert hasattr(state, "top_features")


def test_all_class_probs_covers_all_labels():
    state = _initial_state()
    for label in ALL_LABELS:
        assert label in state.all_class_probs


def test_severities_keys_are_fault_types():
    state = _initial_state()
    assert set(state.severities.keys()) == set(FAULT_TYPES)


def test_forecasts_keys_are_fault_types():
    state = _initial_state()
    assert set(state.forecasts.keys()) == set(FAULT_TYPES)


# ── Model-dependent tests ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_engine_loads_without_error():
    engine = InferenceEngine(MODELS_DIR)
    assert engine is not None


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_single_row_update_returns_state():
    engine = InferenceEngine(MODELS_DIR)
    state = engine.update(_healthy_row())
    assert isinstance(state, DashboardState)


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_buffer_not_ready_before_60_rows():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 59)
    assert state.buffer_ready is False
    assert state.classifier_label == "warming_up"


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_buffer_ready_after_60_rows():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 60)
    assert state.buffer_ready is True


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_classifier_label_is_valid_class_after_warmup():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    assert state.classifier_label in ALL_LABELS


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_elapsed_s_increments_per_row():
    engine = InferenceEngine(MODELS_DIR)
    for i in range(1, 11):
        state = engine.update(_healthy_row())
        assert state.elapsed_s == i


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_latest_row_matches_last_fed():
    engine = InferenceEngine(MODELS_DIR)
    row = _healthy_row()
    row["ENGINE_RPM"] = 1234.0
    state = engine.update(row)
    assert state.latest_row["ENGINE_RPM"] == 1234.0


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_classifier_confidence_in_unit_interval():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    assert 0.0 <= state.classifier_confidence <= 1.0


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_probs_sum_to_one():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    total = sum(state.all_class_probs.values())
    assert abs(total - 1.0) < 1e-4


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_severities_in_unit_interval():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    for ft, sev in state.severities.items():
        assert 0.0 <= sev <= 1.0, f"{ft} severity {sev} out of [0, 1]"


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_forecasts_in_unit_interval():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    for ft, fc in state.forecasts.items():
        assert 0.0 <= fc <= 1.0, f"{ft} forecast {fc} out of [0, 1]"


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_top_features_list_after_warmup():
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _healthy_row(), 70)
    # After a full window, top_features should be a list of (name, value) pairs
    assert isinstance(state.top_features, list)
    assert len(state.top_features) > 0
    name, value = state.top_features[0]
    assert isinstance(name, str)
    assert isinstance(value, float)


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_reset_clears_buffer():
    engine = InferenceEngine(MODELS_DIR)
    _feed(engine, _healthy_row(), 70)
    engine.reset()
    state = engine.update(_healthy_row())
    assert state.buffer_ready is False
    assert state.elapsed_s == 1


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_reset_clears_elapsed_s():
    engine = InferenceEngine(MODELS_DIR)
    _feed(engine, _healthy_row(), 30)
    engine.reset()
    state = engine.update(_healthy_row())
    assert state.elapsed_s == 1


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_current_state_property_matches_last_update():
    engine = InferenceEngine(MODELS_DIR)
    returned = _feed(engine, _healthy_row(), 50)
    assert engine.current_state is returned


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_cold_start_row_does_not_crash():
    """Cold-start rows (low coolant) should not raise any exception."""
    engine = InferenceEngine(MODELS_DIR)
    cold_row = _healthy_row()
    cold_row["COOLANT_TEMPERATURE"] = 40.0
    cold_row["ENGINE_RPM"] = 1100.0
    state = _feed(engine, cold_row, 70)
    assert isinstance(state, DashboardState)


# ── NaN robustness (A1 + A2 — live demo with unsupported PIDs) ────────────────

def _nan_row() -> dict[str, float]:
    """Healthy row with 3 PIDs missing (simulates unsupported ECU sensors)."""
    row = _healthy_row()
    row["COOLANT_TEMPERATURE"] = float("nan")
    row["INTAKE_AIR_TEMPERATURE"] = float("nan")
    row["TIMING_ADVANCE"] = float("nan")
    return row


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_nan_pids_do_not_crash_inference():
    """NaN PIDs (unsupported ECU) must not raise and must return a valid state."""
    engine = InferenceEngine(MODELS_DIR)
    # Feed enough rows to trigger at least one full inference window (≥70 rows)
    state = _feed(engine, _nan_row(), 70)
    assert isinstance(state, DashboardState)


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_nan_pids_produce_no_nan_outputs():
    """All numeric outputs must be finite after NaN-fill — no NaN leaking out."""
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _nan_row(), 70)

    assert not math.isnan(state.classifier_confidence), "classifier_confidence is NaN"
    for ft, sev in state.severities.items():
        assert not math.isnan(sev), f"severities[{ft}] is NaN"
    for ft, fc in state.forecasts.items():
        assert not math.isnan(fc), f"forecasts[{ft}] is NaN"


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_nan_pids_classifier_label_is_valid():
    """Classifier label must be a recognised class even when PIDs are missing."""
    engine = InferenceEngine(MODELS_DIR)
    state = _feed(engine, _nan_row(), 70)
    valid_labels = set(ALL_LABELS) | {"warming_up"}
    assert state.classifier_label in valid_labels


@pytest.mark.skipif(not _MODELS_EXIST, reason="models not built yet")
def test_nan_coolant_does_not_trigger_false_thermostat_alert():
    """Missing COOLANT_TEMPERATURE PID must not cause a false thermostat alert."""
    engine = InferenceEngine(MODELS_DIR)
    # Feed well past the 480-second thermostat timeout with NaN coolant
    state = _feed(engine, _nan_row(), 500)
    rule_names = [ra.rule for ra in state.rule_alerts]
    assert "thermostat_stuck_open" not in rule_names, (
        "thermostat_stuck_open fired despite NaN coolant being filled with 90°C default"
    )
