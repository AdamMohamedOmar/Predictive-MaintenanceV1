"""Tests for the fault injection engine.

Every test either:
  (a) runs on a synthetic minimal DataFrame (no carOBD data needed), or
  (b) uses the real data files and is skipped when data is absent.

Physics assertions are the priority — if a test fails it means the injector
is producing values that violate real engine constraints.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.injection import InjectionParams, inject_fault, inject_session
from src.injection.fault_injector import _build_ramp, _DEFAULT_MAGNITUDE

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "raw" / "carOBD" / "drive1.csv"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_session(n: int = 300) -> pd.DataFrame:
    """Minimal synthetic session that satisfies the injector's column expectations.

    All values are within physical bounds and represent a warmed-up, light-load
    cruise: RPM ~2000, speed ~50 km/h, coolant ~90 °C.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ENGINE_RPM": rng.uniform(1800, 2200, n),
            "VEHICLE_SPEED": rng.uniform(45, 55, n),
            "THROTTLE": rng.uniform(15, 25, n),
            "ENGINE_LOAD": rng.uniform(30, 40, n),
            "COOLANT_TEMPERATURE": rng.uniform(88, 92, n),
            "LONG_TERM_FUEL_TRIM_BANK_1": rng.uniform(-3, 3, n),
            "SHORT_TERM_FUEL_TRIM_BANK_1": rng.uniform(-5, 5, n),
            "INTAKE_MANIFOLD_PRESSURE": rng.uniform(35, 45, n),
            "ABSOLUTE_BAROMETRIC_PRESSURE": np.full(n, 101.0),
            "ACCELERATOR_PEDAL_POSITION_D": rng.uniform(18, 28, n),
            "ACCELERATOR_PEDAL_POSITION_E": rng.uniform(18, 28, n),
            "COMMANDED_THROTTLE_ACTUATOR": rng.uniform(15, 25, n),
            "INTAKE_AIR_TEMPERATURE": rng.uniform(28, 32, n),
            "TIMING_ADVANCE": rng.uniform(12, 18, n),
            "CONTROL_MODULE_VOLTAGE": rng.uniform(13.8, 14.2, n),
        }
    )
    df.attrs["session_id"] = "synthetic"
    return df


def _params(fault_type: str, **kwargs) -> InjectionParams:
    defaults = dict(
        onset_idx=100,
        ramp_len=50,
        magnitude=_DEFAULT_MAGNITUDE[fault_type],
        noise_std=0.0,  # deterministic for physics tests
        random_seed=42,
    )
    defaults.update(kwargs)
    return InjectionParams(fault_type=fault_type, **defaults)


# ─── Ramp helper ─────────────────────────────────────────────────────────────

def test_ramp_is_zero_before_onset():
    r = _build_ramp(100, onset=40, ramp_len=20)
    assert np.all(r[:40] == 0.0)


def test_ramp_is_one_after_ramp_end():
    r = _build_ramp(100, onset=40, ramp_len=20)
    assert np.all(r[60:] == 1.0)


def test_ramp_monotone_during_ramp():
    r = _build_ramp(100, onset=40, ramp_len=20)
    ramp_region = r[40:60]
    assert np.all(np.diff(ramp_region) >= 0)


# ─── Metadata ────────────────────────────────────────────────────────────────

def test_inject_fault_sets_fault_label():
    df = _make_session()
    out = inject_fault(df, _params("fuel_system"))
    assert out.attrs["fault_label"] == "fuel_system"


def test_inject_fault_preserves_session_id():
    df = _make_session()
    out = inject_fault(df, _params("air_system"))
    assert out.attrs["session_id"] == "synthetic"


def test_inject_fault_stores_params_in_attrs():
    df = _make_session()
    p = _params("coolant_temp_sensor")
    out = inject_fault(df, p)
    assert out.attrs["injection"] is p


def test_inject_fault_does_not_mutate_input():
    df = _make_session()
    original_map = df["INTAKE_MANIFOLD_PRESSURE"].copy()
    inject_fault(df, _params("air_system"))
    pd.testing.assert_series_equal(df["INTAKE_MANIFOLD_PRESSURE"], original_map)


def test_inject_session_convenience_wrapper():
    df = _make_session()
    out = inject_session(df, "fuel_system", random_seed=0)
    assert out.attrs["fault_label"] == "fuel_system"
    assert isinstance(out.attrs["injection"], InjectionParams)


def test_unknown_fault_type_raises():
    df = _make_session()
    bad_params = InjectionParams(
        fault_type="engine_explodes",
        onset_idx=100, ramp_len=50, magnitude=1.0,
        noise_std=0.0, random_seed=None,
    )
    with pytest.raises(ValueError, match="Unknown fault_type"):
        inject_fault(df, bad_params)


# ─── Physics: air_system ─────────────────────────────────────────────────────

def test_air_system_map_increases_after_onset():
    df = _make_session()
    out = inject_fault(df, _params("air_system"))
    # MAP should be higher in the fault region than the clean baseline
    baseline_map = out["INTAKE_MANIFOLD_PRESSURE"].iloc[:100].mean()
    fault_map = out["INTAKE_MANIFOLD_PRESSURE"].iloc[150:].mean()
    assert fault_map > baseline_map


def test_air_system_map_never_exceeds_baro():
    df = _make_session()
    out = inject_fault(df, _params("air_system", magnitude=50.0))  # extreme magnitude
    assert (out["INTAKE_MANIFOLD_PRESSURE"] <= out["ABSOLUTE_BAROMETRIC_PRESSURE"]).all()


def test_air_system_stft_within_bounds():
    df = _make_session()
    out = inject_fault(df, _params("air_system"))
    assert out["SHORT_TERM_FUEL_TRIM_BANK_1"].between(-25.0, 25.0).all()


def test_air_system_ltft_within_bounds():
    df = _make_session()
    out = inject_fault(df, _params("air_system"))
    assert out["LONG_TERM_FUEL_TRIM_BANK_1"].between(-25.0, 25.0).all()


def test_air_system_stft_positive_after_onset():
    df = _make_session()
    # Zero out baseline trims so we can measure the fault signal cleanly
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = 0.0
    out = inject_fault(df, _params("air_system", noise_std=0.0))
    fault_stft = out["SHORT_TERM_FUEL_TRIM_BANK_1"].iloc[160:].mean()
    assert fault_stft > 0.0  # ECU adds fuel to compensate lean condition


# ─── Physics: fuel_system ────────────────────────────────────────────────────

def test_fuel_system_ltft_increases_after_onset():
    df = _make_session()
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = 0.0
    out = inject_fault(df, _params("fuel_system", noise_std=0.0))
    fault_ltft = out["LONG_TERM_FUEL_TRIM_BANK_1"].iloc[160:].mean()
    assert fault_ltft > 5.0  # chronic lean → substantial positive LTFT


def test_fuel_system_map_unchanged():
    """Key differentiator: MAP must NOT change for a fuel-system fault."""
    df = _make_session()
    original_map = df["INTAKE_MANIFOLD_PRESSURE"].copy()
    out = inject_fault(df, _params("fuel_system", noise_std=0.0))
    pd.testing.assert_series_equal(
        out["INTAKE_MANIFOLD_PRESSURE"],
        original_map,
        check_names=False,
    )


def test_fuel_system_ltft_within_bounds():
    df = _make_session()
    out = inject_fault(df, _params("fuel_system", magnitude=30.0))  # extreme magnitude
    assert out["LONG_TERM_FUEL_TRIM_BANK_1"].between(-25.0, 25.0).all()


# ─── Physics: coolant_temp_sensor ────────────────────────────────────────────

def test_coolant_temp_drifts_toward_stuck_value():
    df = _make_session()
    stuck = 42.0
    out = inject_fault(df, _params("coolant_temp_sensor", magnitude=stuck, noise_std=0.0))
    end_temp = out["COOLANT_TEMPERATURE"].iloc[-1]
    # By end of session, temp should be closer to stuck_temp than it was at onset
    onset_temp = df["COOLANT_TEMPERATURE"].iloc[100]
    assert abs(end_temp - stuck) < abs(onset_temp - stuck)


def test_coolant_temp_max_rate_of_change():
    """Thermal inertia: no sample-to-sample jump exceeding 1 °C/sec."""
    df = _make_session()
    out = inject_fault(df, _params("coolant_temp_sensor", noise_std=0.0))
    deltas = out["COOLANT_TEMPERATURE"].diff().abs().dropna()
    assert (deltas <= 1.0 + 1e-9).all(), f"Max delta: {deltas.max():.4f} °C/s"


def test_coolant_temp_within_bounds():
    df = _make_session()
    out = inject_fault(df, _params("coolant_temp_sensor", magnitude=10.0))  # below min
    assert out["COOLANT_TEMPERATURE"].between(35.0, 130.0).all()


def test_coolant_timing_retards_during_fault():
    df = _make_session()
    df["TIMING_ADVANCE"] = 15.0
    df["COOLANT_TEMPERATURE"] = 90.0
    out = inject_fault(df, _params("coolant_temp_sensor", magnitude=42.0, noise_std=0.0))
    # In fault region, timing should be lower than baseline
    baseline_timing = out["TIMING_ADVANCE"].iloc[:100].mean()
    fault_timing = out["TIMING_ADVANCE"].iloc[200:].mean()
    assert fault_timing < baseline_timing


# ─── Physics: throttle_position_sensor ──────────────────────────────────────

def test_tps_throttle_inflates_when_pedal_active():
    df = _make_session()
    df["ACCELERATOR_PEDAL_POSITION_D"] = 25.0  # pedal engaged
    original_throttle = df["THROTTLE"].copy()
    out = inject_fault(df, _params("throttle_position_sensor", noise_std=0.0))
    # After ramp, fault-region throttle should be above original
    assert out["THROTTLE"].iloc[160:].mean() > original_throttle.iloc[160:].mean()


def test_tps_idle_rows_unaffected():
    """Closed-throttle idle rows must not be touched by TPS injection."""
    df = _make_session()
    df["ACCELERATOR_PEDAL_POSITION_D"] = 5.0  # pedal not engaged (≤ 10 %)
    original_throttle = df["THROTTLE"].copy()
    out = inject_fault(df, _params("throttle_position_sensor", noise_std=0.0))
    pd.testing.assert_series_equal(out["THROTTLE"], original_throttle, check_names=False)


def test_tps_throttle_within_bounds():
    df = _make_session()
    df["ACCELERATOR_PEDAL_POSITION_D"] = 80.0  # high pedal to stress the clamp
    df["THROTTLE"] = 80.0
    out = inject_fault(df, _params("throttle_position_sensor", magnitude=2.0))  # extreme
    assert out["THROTTLE"].between(0.0, 100.0).all()


def test_tps_commanded_throttle_unchanged():
    """COMMANDED_THROTTLE_ACTUATOR must not be touched — it shows the divergence."""
    df = _make_session()
    original_cmd = df["COMMANDED_THROTTLE_ACTUATOR"].copy()
    out = inject_fault(df, _params("throttle_position_sensor", noise_std=0.0))
    pd.testing.assert_series_equal(
        out["COMMANDED_THROTTLE_ACTUATOR"],
        original_cmd,
        check_names=False,
    )


# ─── Integration with real data ──────────────────────────────────────────────

@pytest.mark.skipif(not SAMPLE.exists(), reason="carOBD data not present")
def test_inject_session_on_real_data_all_faults():
    """inject_session must run without error on drive1.csv for all 4 fault types."""
    from src.data_loading import load_carobd_csv

    df = load_carobd_csv(SAMPLE)
    for fault in ("air_system", "fuel_system", "coolant_temp_sensor", "throttle_position_sensor"):
        out = inject_session(df, fault, random_seed=0)
        assert out.attrs["fault_label"] == fault
        assert len(out) == len(df)
        # All injected PIDs must stay within pandas float64 (no NaN or inf)
        assert out.select_dtypes("number").notna().all().all()
        assert np.isfinite(out.select_dtypes("number").values).all()
