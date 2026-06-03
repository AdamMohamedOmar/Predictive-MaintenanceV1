"""P0-1 acceptance: speed-density vacuum-leak physics.

carOBD is a MAP-based (speed-density) engine — no MAF PID. On such engines a
vacuum leak's robust signature is mechanical (raised idle RPM, slightly raised
idle MAP), NOT a large fuel-trim swing. Any trim response is idle-only and
washes out off-idle. These tests pin that behaviour so a future edit can't
quietly reintroduce the MAF-style large-fuel-trim model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.injection import InjectionParams, inject_fault
from src.injection.fault_injector import _DEFAULT_MAGNITUDE


def _idle_session(n: int = 300) -> pd.DataFrame:
    """Warmed-up engine sitting at idle: low RPM, stationary, low load."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "ENGINE_RPM": rng.uniform(820, 880, n),
            "VEHICLE_SPEED": np.zeros(n),
            "THROTTLE": rng.uniform(3, 6, n),
            "ENGINE_LOAD": rng.uniform(12, 18, n),
            "COOLANT_TEMPERATURE": rng.uniform(88, 92, n),
            "LONG_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
            "SHORT_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
            "INTAKE_MANIFOLD_PRESSURE": rng.uniform(28, 34, n),
            "ABSOLUTE_BAROMETRIC_PRESSURE": np.full(n, 101.0),
            "ACCELERATOR_PEDAL_POSITION_D": rng.uniform(2, 5, n),
            "ACCELERATOR_PEDAL_POSITION_E": rng.uniform(2, 5, n),
            "COMMANDED_THROTTLE_ACTUATOR": rng.uniform(3, 6, n),
            "INTAKE_AIR_TEMPERATURE": rng.uniform(28, 32, n),
            "TIMING_ADVANCE": rng.uniform(10, 14, n),
            "CONTROL_MODULE_VOLTAGE": rng.uniform(13.8, 14.2, n),
        }
    )


def _highway_session(n: int = 300) -> pd.DataFrame:
    """Off-idle cruise under genuine load: high RPM, moving, load > 50 %."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "ENGINE_RPM": rng.uniform(2400, 2800, n),
            "VEHICLE_SPEED": rng.uniform(75, 95, n),
            "THROTTLE": rng.uniform(40, 55, n),
            "ENGINE_LOAD": rng.uniform(60, 72, n),
            "COOLANT_TEMPERATURE": rng.uniform(88, 92, n),
            "LONG_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
            "SHORT_TERM_FUEL_TRIM_BANK_1": np.zeros(n),
            "INTAKE_MANIFOLD_PRESSURE": rng.uniform(70, 85, n),
            "ABSOLUTE_BAROMETRIC_PRESSURE": np.full(n, 101.0),
            "ACCELERATOR_PEDAL_POSITION_D": rng.uniform(40, 55, n),
            "ACCELERATOR_PEDAL_POSITION_E": rng.uniform(40, 55, n),
            "COMMANDED_THROTTLE_ACTUATOR": rng.uniform(40, 55, n),
            "INTAKE_AIR_TEMPERATURE": rng.uniform(28, 32, n),
            "TIMING_ADVANCE": rng.uniform(20, 28, n),
            "CONTROL_MODULE_VOLTAGE": rng.uniform(13.8, 14.2, n),
        }
    )


def _params(**kw) -> InjectionParams:
    d = dict(
        fault_type="air_system",
        onset_idx=100,
        ramp_len=50,
        magnitude=_DEFAULT_MAGNITUDE["air_system"],
        noise_std=0.0,
        random_seed=42,
    )
    d.update(kw)
    return InjectionParams(**d)


# ─── PRIMARY signature: idle RPM + MAP rise ──────────────────────────────────


def test_idle_rpm_rises_post_onset():
    df = _idle_session()
    out = inject_fault(df, _params())
    pre = out["ENGINE_RPM"].iloc[:100].mean()
    post = out["ENGINE_RPM"].iloc[160:].mean()
    assert post > pre + 20.0, (
        f"Idle RPM should rise on a speed-density vacuum leak — "
        f"pre={pre:.0f}, post={post:.0f} rpm."
    )


def test_idle_map_rises_post_onset():
    df = _idle_session()
    out = inject_fault(df, _params())
    pre = out["INTAKE_MANIFOLD_PRESSURE"].iloc[:100].mean()
    post = out["INTAKE_MANIFOLD_PRESSURE"].iloc[160:].mean()
    assert post > pre, f"Idle MAP should rise — pre={pre:.2f}, post={post:.2f} kPa."


def test_idle_rpm_respects_ceiling():
    """Even at an extreme leak the idle-up is clamped to a sane ceiling."""
    df = _idle_session()
    out = inject_fault(df, _params(magnitude=100.0))
    assert out["ENGINE_RPM"].max() <= 1500.0 + 1e-6


# ─── Trim is idle-localized: washes out off-idle ─────────────────────────────


def test_offidle_stft_near_zero():
    """At load > 50 %, the fuel-trim response must be negligible (idle-localized)."""
    df = _highway_session()
    out = inject_fault(df, _params())
    pre = out["SHORT_TERM_FUEL_TRIM_BANK_1"].iloc[:100].mean()
    post = out["SHORT_TERM_FUEL_TRIM_BANK_1"].iloc[160:].mean()
    assert abs(post - pre) < 1.0, (
        f"Off-idle STFT shift should wash out (<1%) on a speed-density engine — "
        f"got Δ={post - pre:.3f}%."
    )


def test_offidle_rpm_unchanged():
    """The idle-up must not fire off-idle (RPM already well above idle)."""
    df = _highway_session()
    original_rpm = df["ENGINE_RPM"].copy()
    out = inject_fault(df, _params())
    # No idle rows → RPM identical to input.
    pd.testing.assert_series_equal(out["ENGINE_RPM"], original_rpm, check_names=False)


# ─── Physical clamp preserved ────────────────────────────────────────────────


def test_map_never_exceeds_baro_extreme():
    df = _idle_session()
    out = inject_fault(df, _params(magnitude=200.0))
    assert (out["INTAKE_MANIFOLD_PRESSURE"] <= out["ABSOLUTE_BAROMETRIC_PRESSURE"]).all()
