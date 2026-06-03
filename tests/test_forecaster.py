"""Tests for severity formulas, forecast dataset builder, and FaultForecaster."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import feature_names
from src.features.severity import compute_severity, compute_baselines
from src.features.forecast_dataset import FAULT_TYPES, _HORIZON_STEPS
from src.models.forecaster import FaultForecaster, forecast_session_split, train_all_forecasters

REPO_ROOT = Path(__file__).resolve().parents[1]
CAROBD_DIR = REPO_ROOT / "data" / "raw" / "carOBD"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

_FEAT_COLS = feature_names()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _healthy_features(n: int = 50) -> dict[str, float]:
    """Feature dict representing a healthy warm-up cruise."""
    rng = np.random.default_rng(3)
    return {col: float(rng.uniform(0.1, 0.9)) for col in _FEAT_COLS} | {
        "INTAKE_MANIFOLD_PRESSURE__mean": 40.0,
        "SHORT_TERM_FUEL_TRIM_BANK_1__mean": 0.0,
        "LONG_TERM_FUEL_TRIM_BANK_1__mean": 0.5,
        "COOLANT_TEMPERATURE__mean": 90.0,
        "THROTTLE_TO_PEDAL_RATIO": 1.0,
        # Ensure severity gates pass for fully-warm, closed-loop operating condition.
        # THROTTLE__mean > 15 keeps TPS gate open.
        # REGIME__COLD_START=0 + REGIME__WARMUP=0 + FUEL_LOOP_ACTIVE=1 ensures
        # coolant severity is not suppressed (simulates warm, closed-loop engine).
        "THROTTLE__mean": 20.0,
        "REGIME__COLD_START": 0.0,
        "REGIME__WARMUP": 0.0,
        "FUEL_LOOP_ACTIVE": 1.0,
    }


def _fault_features(fault_type: str) -> dict[str, float]:
    """Feature dict representing a fully-developed fault."""
    f = _healthy_features()
    if fault_type == "air_system":
        # STFT+LTFT combined response at full ramp with 13 kPa magnitude:
        #   STFT: 0.8×13=10.4% above baseline 0.0; LTFT: 0.32×13=4.16% above baseline 0.5
        #   Combined delta = (10.4+4.66)−(0.0+0.5) = 14.56 → severity = 14.56/14.56 = 1.0
        f["SHORT_TERM_FUEL_TRIM_BANK_1__mean"] = 10.4  # +10.4 % over baseline of 0.0
        f["LONG_TERM_FUEL_TRIM_BANK_1__mean"] = 4.66   # +4.16 % over baseline of 0.5
    elif fault_type == "fuel_system":
        f["LONG_TERM_FUEL_TRIM_BANK_1__mean"] = 18.5  # +18 % over baseline of 0.5
    elif fault_type == "coolant_temp_sensor":
        f["COOLANT_TEMPERATURE__mean"] = 42.0
        f["COOLANT_WARMUP_RATE"] = 0.0  # stuck sensor: cold AND flat (P1-1 discriminator)
    elif fault_type == "throttle_position_sensor":
        # ratio=1.35: delta=0.35, post-deadband=(0.35-0.20)/0.15=1.0 severity
        # THROTTLE__mean=25.0 ensures the low-throttle gate (15%) does not suppress.
        # THROTTLE_CMD_ACTUAL_DELTA=15.0: cmd_term=clip(15/10, 0, 1)=1.0 so blended
        # severity = 0.5*1.0 + 0.5*1.0 = 1.0 as expected.
        f["THROTTLE_TO_PEDAL_RATIO"] = 1.35
        f["THROTTLE__mean"] = 25.0
        f["THROTTLE_CMD_ACTUAL_DELTA"] = 15.0
    return f


def _healthy_baselines() -> dict[str, float]:
    return {
        "INTAKE_MANIFOLD_PRESSURE__mean": 40.0,
        "SHORT_TERM_FUEL_TRIM_BANK_1__mean": 0.0,
        "LONG_TERM_FUEL_TRIM_BANK_1__mean": 0.5,
        "THROTTLE_TO_PEDAL_RATIO": 1.0,
    }


# ─── compute_severity ─────────────────────────────────────────────────────────

def test_severity_healthy_is_zero_for_all_faults():
    feats = _healthy_features()
    bases = _healthy_baselines()
    for fault in FAULT_TYPES:
        sev = compute_severity(feats, fault, bases)
        assert sev == pytest.approx(0.0, abs=0.05), f"{fault}: expected ~0, got {sev}"


def test_severity_full_fault_is_one():
    bases = _healthy_baselines()
    for fault in FAULT_TYPES:
        feats = _fault_features(fault)
        sev = compute_severity(feats, fault, bases)
        assert sev == pytest.approx(1.0, abs=0.05), f"{fault}: expected ~1.0, got {sev}"


def test_severity_clamped_above_one():
    feats = _healthy_features()
    feats["COOLANT_TEMPERATURE__mean"] = 0.0  # extreme cold — would be >1 without clamp
    sev = compute_severity(feats, "coolant_temp_sensor", _healthy_baselines())
    assert sev <= 1.0


def test_severity_clamped_below_zero():
    feats = _healthy_features()
    feats["SHORT_TERM_FUEL_TRIM_BANK_1__mean"] = -5.0  # below baseline — would be <0
    sev = compute_severity(feats, "air_system", _healthy_baselines())
    assert sev >= 0.0


def test_severity_monotone_with_fuel_trim_increase():
    bases = _healthy_baselines()
    feats = _healthy_features()
    prev = 0.0
    for stft_val in [0.0, 2.0, 4.0, 6.0, 8.0]:
        feats["SHORT_TERM_FUEL_TRIM_BANK_1__mean"] = stft_val
        sev = compute_severity(feats, "air_system", bases)
        assert sev >= prev - 1e-9
        prev = sev


def test_coolant_severity_zero_during_healthy_warmup():
    """A cold engine that is actively WARMING (≥0.5 °C/min) is healthy, not a fault.

    P1-1: the discriminator is warm-up DYNAMICS, not absolute temperature. A
    40 °C engine climbing 1.5 °C/min is a normal cold-start, scored 0.
    """
    feats = _healthy_features()
    feats["COOLANT_TEMPERATURE__mean"] = 40.0
    feats["COOLANT_WARMUP_RATE"] = 1.5  # climbing normally
    sev = compute_severity(feats, "coolant_temp_sensor", _healthy_baselines())
    assert sev == 0.0


def test_coolant_severity_fires_even_in_open_loop():
    """P1-1 regression: a stuck-cold sensor is open-loop, but must still register.

    The old FUEL_LOOP_ACTIVE gate wrongly suppressed coolant severity whenever
    the ECU was open-loop — which a stuck-cold engine always is. That gate was
    removed; a cold AND flat reading now scores regardless of loop state.
    """
    feats = _healthy_features()
    feats["COOLANT_TEMPERATURE__mean"] = 42.0
    feats["COOLANT_WARMUP_RATE"] = 0.0  # stuck (flat)
    feats["FUEL_LOOP_ACTIVE"] = 0.0      # open loop — must NOT suppress anymore
    sev = compute_severity(feats, "coolant_temp_sensor", _healthy_baselines())
    assert sev > 0.5


def test_coolant_severity_zero_during_warmup():
    """Engine warming through 60 °C at a healthy rate is not a coolant fault."""
    feats = _healthy_features()
    feats["COOLANT_TEMPERATURE__mean"] = 60.0  # legitimately warming up
    feats["COOLANT_WARMUP_RATE"] = 1.0
    sev = compute_severity(feats, "coolant_temp_sensor", _healthy_baselines())
    assert sev == 0.0


def test_tps_severity_zero_at_low_throttle():
    """Idle/coast windows must not register a TPS fault."""
    feats = _healthy_features()
    feats["THROTTLE__mean"] = 5.0       # below the 15% gate
    feats["THROTTLE_TO_PEDAL_RATIO"] = 1.5  # would normally be a fault
    sev = compute_severity(feats, "throttle_position_sensor", _healthy_baselines())
    assert sev == 0.0


def test_air_system_severity_zero_when_loop_inactive():
    """Open-loop ECU → STFT is frozen; air_system severity must be 0."""
    feats = _healthy_features()
    feats["SHORT_TERM_FUEL_TRIM_BANK_1__mean"] = 10.4  # would normally be fault
    feats["LONG_TERM_FUEL_TRIM_BANK_1__mean"] = 4.66
    feats["FUEL_LOOP_ACTIVE"] = 0.0
    sev = compute_severity(feats, "air_system", _healthy_baselines())
    assert sev == 0.0


def test_fuel_system_severity_zero_when_loop_inactive():
    """Open-loop ECU → LTFT is frozen; fuel_system severity must be 0."""
    feats = _healthy_features()
    feats["LONG_TERM_FUEL_TRIM_BANK_1__mean"] = 18.5  # would normally be fault
    feats["FUEL_LOOP_ACTIVE"] = 0.0
    sev = compute_severity(feats, "fuel_system", _healthy_baselines())
    assert sev == 0.0


def test_tps_severity_deadband_suppresses_small_delta():
    """Natural ratio variance below the deadband must yield severity 0."""
    feats = _healthy_features()
    feats["THROTTLE__mean"] = 25.0
    feats["THROTTLE_TO_PEDAL_RATIO"] = 1.08  # 0.08 above baseline (within 0.10 deadband)
    sev = compute_severity(feats, "throttle_position_sensor", _healthy_baselines())
    assert sev == 0.0


def test_severity_unknown_fault_raises():
    with pytest.raises(ValueError):
        compute_severity(_healthy_features(), "engine_explodes", _healthy_baselines())


def test_compute_baselines_from_dataframe():
    rng = np.random.default_rng(5)
    n = 100
    df = pd.DataFrame({
        "INTAKE_MANIFOLD_PRESSURE__mean": rng.uniform(38, 42, n),
        "SHORT_TERM_FUEL_TRIM_BANK_1__mean": rng.uniform(-1, 1, n),
        "LONG_TERM_FUEL_TRIM_BANK_1__mean": rng.uniform(-2, 2, n),
        "THROTTLE_TO_PEDAL_RATIO": rng.uniform(0.95, 1.05, n),
    })
    bases = compute_baselines(df)
    assert abs(bases["INTAKE_MANIFOLD_PRESSURE__mean"] - 40.0) < 2.0
    assert "SHORT_TERM_FUEL_TRIM_BANK_1__mean" in bases
    assert "LONG_TERM_FUEL_TRIM_BANK_1__mean" in bases
    assert "THROTTLE_TO_PEDAL_RATIO" in bases


# ─── Horizon constant ─────────────────────────────────────────────────────────

def test_horizon_steps_equals_six():
    # FORECAST_HORIZON_S=60, WINDOW_STRIDE_S=10 → 6 steps
    assert _HORIZON_STEPS == 6


# ─── forecast_session_split ───────────────────────────────────────────────────

def _make_forecast_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sessions = ["drive1", "live5", "live6", "live12"]
    rows = []
    for i in range(n):
        row = {col: float(rng.uniform(0, 1)) for col in _FEAT_COLS}
        row["severity_target"] = float(rng.uniform(0, 1))
        row["session_id"] = sessions[i % len(sessions)]
        rows.append(row)
    return pd.DataFrame(rows)


def test_forecast_session_split_no_overlap():
    df = _make_forecast_df()
    train_df, test_df = forecast_session_split(df, held_out={"live12"})
    assert len(set(train_df["session_id"]) & set(test_df["session_id"])) == 0


def test_forecast_session_split_covers_all():
    df = _make_forecast_df()
    train_df, test_df = forecast_session_split(df, held_out={"live12"})
    assert len(train_df) + len(test_df) == len(df)


# ─── FaultForecaster.predict ──────────────────────────────────────────────────

def _make_tiny_forecaster() -> tuple[FaultForecaster, dict]:
    """Build a FaultForecaster trained on minimal synthetic data."""
    from src.features.normalizer import BaselineNormalizer

    rng = np.random.default_rng(9)
    n = 120
    sessions = ["s1", "s2", "s3"]

    def _make_ds(fault: str) -> pd.DataFrame:
        rows = []
        for i in range(n):
            row = {col: float(rng.uniform(0, 1)) for col in _FEAT_COLS}
            row["severity_target"] = float(i / n)  # monotone ramp
            row["session_id"] = sessions[i % len(sessions)]
            row["label"] = "healthy" if i < n // 2 else fault
            rows.append(row)
        return pd.DataFrame(rows)

    datasets = {f: _make_ds(f) for f in FAULT_TYPES}

    # Fit normaliser on healthy rows
    combined = pd.concat(datasets.values(), ignore_index=True)
    norm = BaselineNormalizer()
    norm.fit(combined)

    forecaster = train_all_forecasters(datasets, norm, held_out={"s3"}, n_estimators=10, random_seed=0)
    baselines = {
        "INTAKE_MANIFOLD_PRESSURE__mean": 0.5,
        "SHORT_TERM_FUEL_TRIM_BANK_1__mean": 0.5,
        "LONG_TERM_FUEL_TRIM_BANK_1__mean": 0.5,
        "THROTTLE_TO_PEDAL_RATIO": 1.0,
    }
    return forecaster, baselines


def test_forecaster_predict_returns_float_in_unit_interval():
    forecaster, _ = _make_tiny_forecaster()
    feats = _healthy_features()
    for fault in FAULT_TYPES:
        pred = forecaster.predict(fault, feats)
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 1.0


def test_forecaster_predict_unknown_fault_raises():
    forecaster, _ = _make_tiny_forecaster()
    with pytest.raises(ValueError):
        forecaster.predict("engine_explodes", _healthy_features())


def test_forecaster_summary_has_all_faults():
    forecaster, _ = _make_tiny_forecaster()
    summary = forecaster.summary()
    assert set(summary["fault"]) == set(FAULT_TYPES)


def test_forecaster_save_load_roundtrip(tmp_path):
    forecaster, _ = _make_tiny_forecaster()
    forecaster.save(models_dir=tmp_path, results_dir=tmp_path)
    loaded = FaultForecaster.load(models_dir=tmp_path)
    feats = _healthy_features()
    for fault in FAULT_TYPES:
        p1 = forecaster.predict(fault, feats)
        p2 = loaded.predict(fault, feats)
        assert abs(p1 - p2) < 1e-6


def test_feature_means_returns_copy_not_reference():
    """Mutating the returned array must not alter the scaler's internal state."""
    from src.features.normalizer import BaselineNormalizer

    rng = np.random.default_rng(11)
    n = 40
    df = pd.DataFrame(
        {col: rng.uniform(0.0, 1.0, n) for col in _FEAT_COLS} | {"label": "healthy"}
    )
    norm = BaselineNormalizer().fit(df)

    means_a = norm.feature_means
    means_a[:] = 999.0  # mutate the returned array

    means_b = norm.feature_means
    assert not np.any(means_b == 999.0), "feature_means returned a mutable reference to scaler internals"


def test_normalizer_save_load_roundtrip_checks_feature_order(tmp_path):
    """Saving with v2 format and loading detects a feature-order mismatch."""
    import importlib
    from src.features.normalizer import BaselineNormalizer, _FEAT_COLS

    rng = np.random.default_rng(13)
    n = 40
    df = pd.DataFrame(
        {col: rng.uniform(0.0, 1.0, n) for col in _FEAT_COLS} | {"label": "healthy"}
    )
    norm = BaselineNormalizer().fit(df)

    save_path = tmp_path / "norm_test.pkl"
    norm.save(save_path)

    # Normal load must succeed
    loaded = BaselineNormalizer.load(save_path)
    assert loaded.is_fitted

    # Tamper with the saved feature order and reload — must raise
    import pickle
    with open(save_path, "rb") as f:
        bundle = pickle.load(f)
    bundle["feature_order"] = bundle["feature_order"][:-1]  # remove last feature
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)

    with pytest.raises(RuntimeError, match="feature order mismatch"):
        BaselineNormalizer.load(save_path)


def test_predict_all_returns_fault_types_order():
    """predict_all dict keys must follow FAULT_TYPES order (not thread-completion order)."""
    forecaster, _ = _make_tiny_forecaster()
    result = forecaster.predict_all(_healthy_features())
    assert list(result.keys()) == FAULT_TYPES


def test_predict_all_isolates_single_model_failure():
    """One broken model must not zero all four predictions."""
    import unittest.mock as mock

    forecaster, _ = _make_tiny_forecaster()
    feats = _healthy_features()

    # Break the air_system model by replacing predict() with a side-effect
    with mock.patch.object(
        forecaster._models["air_system"], "predict", side_effect=RuntimeError("broken")
    ):
        result = forecaster.predict_all(feats)

    # Broken model yields 0.0 for its own fault
    assert result["air_system"] == 0.0
    # All other models must still return valid floats in [0, 1]
    for ft in FAULT_TYPES:
        if ft != "air_system":
            assert 0.0 <= result[ft] <= 1.0


# ─── Integration on real data ─────────────────────────────────────────────────

@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_and_train_real_forecasters(tmp_path):
    """Full end-to-end: build datasets, train 4 forecasters, check MAE targets."""
    from src.features.forecast_dataset import build_all_forecast_datasets
    from src.features.dataset_builder import load_dataset
    from src.features.severity import compute_baselines
    from src.features.normalizer import BaselineNormalizer
    from src.models.classifier import session_split

    # Baselines from the classifier training split
    ds = load_dataset()
    train_df, _ = session_split(ds)
    healthy_train = train_df[train_df["label"] == "healthy"]
    baselines = compute_baselines(healthy_train)

    # Build forecast datasets
    forecast_ds = build_all_forecast_datasets(
        baselines, carobd_dir=CAROBD_DIR, output_dir=tmp_path
    )

    # Normaliser from classifier training split
    norm = BaselineNormalizer().fit(train_df)

    # Train in parallel
    forecaster = train_all_forecasters(forecast_ds, norm, n_estimators=50, random_seed=42)

    summary = forecaster.summary()
    print("\n", summary.to_string(index=False))

    # Per-fault commit targets for this integration test (50 estimators).
    # NOTE: limits are intentionally looser than production (300 estimators).
    # Production rebuild_all achieves TPS=22.3%, air=9.7%, fuel=9.3%, coolant=1.0%.
    # TPS limit widened 25→35: the new _TPS_DEADBAND=0.20 changes the severity
    # target distribution (many ramp windows score 0.0 until ratio exceeds deadband),
    # which makes regression harder at 50 trees; 300-tree production model hits 22.3%.
    _FAULT_MAE_LIMITS = {
        "air_system": 20.0,       # loosened: T5.6 idle-weight reduces uniformity of
                                  # signal at high load; 50-tree test model is noisier.
                                  # Production 300-tree model targets ≤15%.
        "fuel_system": 15.0,
        "coolant_temp_sensor": 15.0,
        "throttle_position_sensor": 35.0,  # loosened per Decision log after deadband change
    }
    for _, row in summary.iterrows():
        limit = _FAULT_MAE_LIMITS.get(row["fault"], 15.0)
        assert row["MAE % of range"] <= limit, (
            f"{row['fault']}: MAE {row['MAE % of range']:.1f}% exceeds {limit:.0f}% target"
        )
