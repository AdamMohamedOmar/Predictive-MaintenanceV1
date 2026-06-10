"""Guardrails: process_captured_rows must reject captures that cannot be a
real engine (the mock-baseline incident: constant 90.0 coolant, std=0 stats
poisoned every later z-score, and a healthy Yaris read 64/64 air_system)."""

import numpy as np
import pytest

from scripts.live_baseline_capture import process_captured_rows


def _mk_rows(n: int = 400, *, coolant: float | None = None,
             stft_amp: float = 2.0, seed: int = 0) -> list[dict]:
    """Physically plausible warmed-up drive: ~50 km/h, coolant climbing to 92°C
    at 0.08°C/s (< 1°C/s thermal inertia), trims inside ±25%."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "ENGINE_RPM": 1900 + 250 * np.sin(i / 30) + rng.normal(0, 40),
            "VEHICLE_SPEED": 50 + 8 * np.sin(i / 45) + rng.normal(0, 1.5),
            "THROTTLE": 22 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "ENGINE_LOAD": 35 + 5 * np.sin(i / 35) + rng.normal(0, 1),
            "COOLANT_TEMPERATURE": (coolant if coolant is not None
                                    else min(92.0, 76.0 + i * 0.08) + rng.normal(0, 0.2)),
            "LONG_TERM_FUEL_TRIM_BANK_1": 1.5 + rng.normal(0, 0.5),
            "SHORT_TERM_FUEL_TRIM_BANK_1": rng.normal(0, stft_amp),
            "INTAKE_MANIFOLD_PRESSURE": 55 + 8 * np.sin(i / 40) + rng.normal(0, 1),
            "ACCELERATOR_PEDAL_POSITION_D": 24 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "ACCELERATOR_PEDAL_POSITION_E": 24 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "COMMANDED_THROTTLE_ACTUATOR": 22 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "INTAKE_AIR_TEMPERATURE": 32 + rng.normal(0, 0.5),
            "TIMING_ADVANCE": 16 + 3 * np.sin(i / 20) + rng.normal(0, 1),
            "CONTROL_MODULE_VOLTAGE": 14.0 + rng.normal(0, 0.05),
        })
    return rows


def test_accepts_healthy_varied_capture():
    norm, meta = process_captured_rows(_mk_rows(), vehicle_name="fixture")
    assert meta["n_windows"] >= 20


def test_rejects_coolant_frozen_at_fallback():
    """Constant 90.0 is the NaN-fallback constant — the mock-capture signature."""
    with pytest.raises(ValueError, match="90.0"):
        process_captured_rows(_mk_rows(coolant=90.0), vehicle_name="mock")


def test_rejects_any_constant_present_pid():
    rows = _mk_rows()
    for r in rows:
        r["INTAKE_MANIFOLD_PRESSURE"] = 55.0
    with pytest.raises(ValueError, match="INTAKE_MANIFOLD_PRESSURE"):
        process_captured_rows(rows, vehicle_name="mock")


def test_rejects_open_loop_never_reached():
    """|STFT| sigma=0.2% stays under the 0.5% FUEL_LOOP_ACTIVE threshold in
    extractor.py — closed loop never detected, baseline must be refused."""
    with pytest.raises(ValueError, match="closed-loop"):
        process_captured_rows(_mk_rows(stft_amp=0.2), vehicle_name="openloop")


def test_absent_pid_is_exempt_from_variance_guard():
    """An unsupported PID arrives as NaN every row (LiveObdSource contract).
    That is legitimate (2007 Skoda may lack pedal PIDs) — must NOT reject."""
    rows = _mk_rows()
    for r in rows:
        r["ACCELERATOR_PEDAL_POSITION_D"] = float("nan")
    norm, meta = process_captured_rows(rows, vehicle_name="no-pedal")
    assert meta["n_windows"] >= 20
