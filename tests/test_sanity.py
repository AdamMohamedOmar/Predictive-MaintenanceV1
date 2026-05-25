"""Tests for the per-row physics sanity check."""

import math

import pytest

from src.dashboard.sanity import check_row


def _good_row() -> dict:
    return {
        "ENGINE_RPM": 1500.0,
        "VEHICLE_SPEED": 50.0,
        "THROTTLE": 20.0,
        "ENGINE_LOAD": 40.0,
        "COOLANT_TEMPERATURE": 90.0,
        "LONG_TERM_FUEL_TRIM_BANK_1": 1.0,
        "SHORT_TERM_FUEL_TRIM_BANK_1": -1.0,
        "INTAKE_MANIFOLD_PRESSURE": 50.0,
        "ACCELERATOR_PEDAL_POSITION_D": 22.0,
        "ACCELERATOR_PEDAL_POSITION_E": 22.0,
        "COMMANDED_THROTTLE_ACTUATOR": 20.0,
        "INTAKE_AIR_TEMPERATURE": 25.0,
        "TIMING_ADVANCE": 15.0,
        "CONTROL_MODULE_VOLTAGE": 14.0,
    }


def test_good_row_passes():
    assert check_row(_good_row()).ok is True


def test_out_of_bounds_fuel_trim_fails():
    row = _good_row()
    row["LONG_TERM_FUEL_TRIM_BANK_1"] = 75.0  # OBD max is ±25
    v = check_row(row)
    assert v.ok is False
    assert any("LONG_TERM_FUEL_TRIM_BANK_1" in m for m in v.violations)


def test_rpm_zero_while_moving_fails():
    row = _good_row()
    row["ENGINE_RPM"] = 0.0
    row["VEHICLE_SPEED"] = 50.0
    v = check_row(row)
    assert v.ok is False
    assert any("ENGINE_RPM" in m and "VEHICLE_SPEED" in m for m in v.violations)


def test_nan_values_are_skipped_not_flagged():
    """NaN means PID unsupported by the ECU — not a violation."""
    row = _good_row()
    row["TIMING_ADVANCE"] = float("nan")
    assert check_row(row).ok is True


def test_garbage_row_collects_all_violations():
    """Multiple simultaneous violations must all be reported."""
    bad = {
        "ENGINE_RPM": 0.0,
        "VEHICLE_SPEED": 50.0,
        "LONG_TERM_FUEL_TRIM_BANK_1": 75.0,
        "COOLANT_TEMPERATURE": 250.0,
    }
    v = check_row(bad)
    assert v.ok is False
    assert len(v.violations) >= 3
