"""Tests for the python-OBD ↔ canonical PID translation layer."""

import math

import obd
import pytest
from obd import OBDResponse
from obd import Unit as OBDUnit

from src.config import USEFUL_PIDS
from src.live.pid_map import PID_MAP, available_commands, to_float


# ── PID_MAP coverage ─────────────────────────────────────────────────────────

def test_pid_map_covers_all_useful_pids():
    """Every PID in USEFUL_PIDS must have an entry in PID_MAP.

    If a command resolves to None (version mismatch) it is filtered at import
    time, so the resolved map should still cover all 14 PIDs for obd 0.7.2.
    """
    for pid in USEFUL_PIDS:
        assert pid in PID_MAP, f"{pid} missing from PID_MAP"


def test_pid_map_values_are_obd_commands():
    for pid, cmd in PID_MAP.items():
        assert isinstance(cmd, obd.OBDCommand), f"{pid} → {cmd!r} is not an OBDCommand"


def test_available_commands_matches_pid_map_keys():
    assert set(available_commands()) == set(PID_MAP.keys())


def test_commanded_throttle_actuator_present():
    """python-obd 0.7.2 exposes this as THROTTLE_ACTUATOR; our map remaps it."""
    assert "COMMANDED_THROTTLE_ACTUATOR" in PID_MAP


# ── to_float: null / missing response ────────────────────────────────────────

def test_to_float_null_response_returns_nan():
    r = OBDResponse()  # is_null() == True, value == None
    result = to_float(r)
    assert math.isnan(result)


def test_to_float_none_value_returns_nan():
    r = OBDResponse()
    r.value = None
    r.messages = ["mock"]  # prevent is_null short-circuit
    # is_null() still returns True when value is None
    assert math.isnan(to_float(r))


# ── to_float: pint Quantity values ───────────────────────────────────────────

def _make_response(magnitude: float, unit) -> OBDResponse:
    """Helper: build an OBDResponse that looks like a real ECU response."""
    r = OBDResponse()
    r.value = OBDUnit.Quantity(magnitude, unit)
    r.messages = ["mock"]  # prevents is_null() from returning True
    return r


def test_to_float_rpm():
    r = _make_response(850.0, OBDUnit.rpm)
    assert to_float(r) == pytest.approx(850.0)


def test_to_float_speed_kph():
    r = _make_response(60.0, OBDUnit.kph)
    assert to_float(r) == pytest.approx(60.0)


def test_to_float_celsius():
    r = _make_response(90.0, OBDUnit.celsius)
    assert to_float(r) == pytest.approx(90.0)


def test_to_float_kilopascal():
    r = _make_response(35.0, OBDUnit.kilopascal)
    assert to_float(r) == pytest.approx(35.0)


def test_to_float_percent():
    r = _make_response(2.5, OBDUnit.percent)
    assert to_float(r) == pytest.approx(2.5)


def test_to_float_volt():
    r = _make_response(14.2, OBDUnit.volt)
    assert to_float(r) == pytest.approx(14.2)


# ── to_float: plain Python numeric values (some ECUs return these) ────────────

def test_to_float_plain_int():
    r = OBDResponse()
    r.value = 42
    r.messages = ["mock"]
    assert to_float(r) == pytest.approx(42.0)


def test_to_float_plain_float():
    r = OBDResponse()
    r.value = 3.14
    r.messages = ["mock"]
    assert to_float(r) == pytest.approx(3.14)


def test_to_float_non_numeric_returns_nan():
    r = OBDResponse()
    r.value = "not_a_number"
    r.messages = ["mock"]
    assert math.isnan(to_float(r))


def test_to_float_zero_is_valid():
    """Zero is a legal sensor value (e.g. idle throttle = 0%)."""
    r = _make_response(0.0, OBDUnit.percent)
    result = to_float(r)
    assert result == pytest.approx(0.0)
    assert not math.isnan(result)
