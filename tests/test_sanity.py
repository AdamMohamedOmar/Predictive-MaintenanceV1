"""Tests for the per-row physics sanity check."""

import math

import pytest

from src.dashboard.sanity import check_row, RowSanityChecker


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


# ─── T5.3 new rules ───────────────────────────────────────────────────────────

def test_map_above_baro_fails():
    """MAP > BARO + 5 kPa is impossible on a naturally-aspirated engine."""
    row = _good_row()
    row["INTAKE_MANIFOLD_PRESSURE"] = 120.0
    row["ABSOLUTE_BAROMETRIC_PRESSURE"] = 101.0  # MAP exceeds baro + 5
    v = check_row(row)
    assert v.ok is False
    assert any("MAP" in m and "BARO" in m for m in v.violations)


def test_map_at_baro_passes():
    """MAP ≈ BARO is legal at full open-throttle (throttle plate fully open)."""
    row = _good_row()
    row["INTAKE_MANIFOLD_PRESSURE"] = 100.0
    row["ABSOLUTE_BAROMETRIC_PRESSURE"] = 101.0
    assert check_row(row).ok is True


def test_throttle_vs_commanded_delta_over_30_fails():
    """A 31 % gap between throttle and commanded = hardware fault."""
    row = _good_row()
    row["THROTTLE"] = 60.0
    row["COMMANDED_THROTTLE_ACTUATOR"] = 20.0  # delta = 40 %
    v = check_row(row)
    assert v.ok is False
    assert any("THROTTLE" in m and "COMMANDED" in m for m in v.violations)


def test_throttle_vs_commanded_small_delta_passes():
    """Normal TPS wear (< 30 %) must NOT trigger the sanity rule."""
    row = _good_row()
    row["THROTTLE"] = 25.0
    row["COMMANDED_THROTTLE_ACTUATOR"] = 20.0  # delta = 5 %
    assert check_row(row).ok is True


def test_row_sanity_checker_flags_coolant_spike():
    """RowSanityChecker detects a 10 °C/s coolant jump (sensor glitch)."""
    checker = RowSanityChecker()
    row1 = _good_row()
    row1["COOLANT_TEMPERATURE"] = 90.0
    checker.check(row1, now=0.0)

    row2 = _good_row()
    row2["COOLANT_TEMPERATURE"] = 100.0  # +10 °C in 1 s = 10 °C/s > 1 °C/s limit
    v = checker.check(row2, now=1.0)
    assert v.ok is False
    assert any("Coolant" in m for m in v.violations)


def test_row_sanity_checker_accepts_normal_warmup():
    """Normal warmup (< 1 °C/s) must not trigger the coolant-rate rule."""
    checker = RowSanityChecker()
    row1 = _good_row()
    row1["COOLANT_TEMPERATURE"] = 70.0
    checker.check(row1, now=0.0)

    row2 = _good_row()
    row2["COOLANT_TEMPERATURE"] = 70.5  # +0.5 °C in 1 s = well within 1 °C/s
    v = checker.check(row2, now=1.0)
    assert v.ok is True


def test_row_sanity_checker_reset_clears_state():
    """After reset(), the checker forgets the previous row and can start fresh."""
    checker = RowSanityChecker()
    row = _good_row()
    row["COOLANT_TEMPERATURE"] = 90.0
    checker.check(row, now=0.0)

    checker.reset()

    # A "spike" on the first row after reset must NOT flag — no previous state
    row2 = _good_row()
    row2["COOLANT_TEMPERATURE"] = 999.0  # extreme value, but no previous row to compare to
    v = checker.check(row2, now=1.0)
    # Only the out-of-bounds range check may flag here — NOT the coolant rate check
    coolant_rate_msgs = [m for m in v.violations if "Coolant" in m and "°C/s" in m]
    assert coolant_rate_msgs == []
