"""Tests for CsvStreamer — the row-by-row CSV playback component."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dashboard.streamer import CsvStreamer

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE1 = REPO_ROOT / "data" / "raw" / "carOBD" / "drive1.csv"


# ── Synthetic CSV fixture ─────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, n_rows: int = 10) -> Path:
    """Write a minimal well-formed carOBD CSV with n_rows of data."""
    # Column names must match _RENAME_MAP in data_loading.py
    cols = {
        "ENGINE_RUN_TINE ()": 0,
        "ENGINE_RPM ()": 800.0,
        "VEHICLE_SPEED ()": 0.0,
        "THROTTLE ()": 5.0,
        "ENGINE_LOAD ()": 20.0,
        "COOLANT_TEMPERATURE ()": 80.0,
        "LONG_TERM_FUEL_TRIM_BANK_1 ()": 0.0,
        "SHORT_TERM_FUEL_TRIM_BANK_1 ()": 0.0,
        "INTAKE_MANIFOLD_PRESSURE ()": 35.0,
        "FUEL_TANK ()": 50.0,
        "ABSOLUTE_THROTTLE_B ()": 5.0,
        "PEDAL_D ()": 5.0,
        "PEDAL_E ()": 4.0,
        "COMMANDED_THROTTLE_ACTUATOR ()": 5.0,
        "FUEL_AIR_COMMANDED_EQUIV_RATIO ()": 0.0,
        "ABSOLUTE_BAROMETRIC_PRESSURE ()": 100.0,
        "RELATIVE_THROTTLE_POSITION ()": 5.0,
        "INTAKE_AIR_TEMP ()": 25.0,
        "TIMING_ADVANCE ()": 12.0,
        "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()": 400.0,
        "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()": 380.0,
        "CONTROL_MODULE_VOLTAGE ()": 14.2,
        "COMMANDED_EVAPORATIVE_PURGE ()": 0.0,
        "TIME_RUN_WITH_MIL_ON ()": 0,
        "TIME_SINCE_TROUBLE_CODES_CLEARED ()": 100,
        "DISTANCE_TRAVELED_WITH_MIL_ON ()": 0,
        "WARM_UPS_SINCE_CODES_CLEARED ()": 255,
    }
    df = pd.DataFrame([cols] * n_rows)
    # Give each row a unique RPM so we can verify ordering
    df["ENGINE_RPM ()"] = [float(800 + i) for i in range(n_rows)]
    path = tmp_path / "test_session.csv"
    df.to_csv(path, index=False)
    return path


# ── Construction ──────────────────────────────────────────────────────────────

def test_total_equals_csv_row_count(tmp_path):
    path = _make_csv(tmp_path, n_rows=15)
    s = CsvStreamer(path)
    assert s.total == 15


def test_initial_remaining_equals_total(tmp_path):
    path = _make_csv(tmp_path, n_rows=10)
    s = CsvStreamer(path)
    assert s.remaining == s.total


def test_speed_stored(tmp_path):
    path = _make_csv(tmp_path)
    s = CsvStreamer(path, speed=5.0)
    assert s.speed == 5.0


def test_session_id_is_stem(tmp_path):
    path = _make_csv(tmp_path)
    s = CsvStreamer(path)
    assert s.session_id == "test_session"


# ── Streaming ─────────────────────────────────────────────────────────────────

def test_next_row_returns_dict(tmp_path):
    path = _make_csv(tmp_path)
    s = CsvStreamer(path)
    row = s.next_row()
    assert isinstance(row, dict)


def test_next_row_contains_engine_rpm(tmp_path):
    path = _make_csv(tmp_path)
    s = CsvStreamer(path)
    row = s.next_row()
    assert "ENGINE_RPM" in row


def test_rows_yield_in_order(tmp_path):
    """Each successive row must have a higher ENGINE_RPM (see _make_csv fixture)."""
    path = _make_csv(tmp_path, n_rows=5)
    s = CsvStreamer(path)
    rpms = [s.next_row()["ENGINE_RPM"] for _ in range(5)]
    assert rpms == sorted(rpms)


def test_remaining_decrements_on_next_row(tmp_path):
    path = _make_csv(tmp_path, n_rows=10)
    s = CsvStreamer(path)
    for expected_remaining in range(9, -1, -1):
        s.next_row()
        assert s.remaining == expected_remaining


def test_elapsed_s_increments(tmp_path):
    path = _make_csv(tmp_path, n_rows=5)
    s = CsvStreamer(path)
    for i in range(1, 6):
        s.next_row()
        assert s.elapsed_s == i


def test_returns_none_when_exhausted(tmp_path):
    path = _make_csv(tmp_path, n_rows=3)
    s = CsvStreamer(path)
    for _ in range(3):
        s.next_row()
    assert s.next_row() is None


def test_exhausted_flag(tmp_path):
    path = _make_csv(tmp_path, n_rows=2)
    s = CsvStreamer(path)
    assert not s.exhausted
    s.next_row()
    assert not s.exhausted
    s.next_row()
    assert s.exhausted


# ── reset() ──────────────────────────────────────────────────────────────────

def test_reset_restores_remaining(tmp_path):
    path = _make_csv(tmp_path, n_rows=10)
    s = CsvStreamer(path)
    for _ in range(7):
        s.next_row()
    s.reset()
    assert s.remaining == 10


def test_reset_restores_elapsed_s(tmp_path):
    path = _make_csv(tmp_path, n_rows=10)
    s = CsvStreamer(path)
    for _ in range(7):
        s.next_row()
    s.reset()
    assert s.elapsed_s == 0


def test_reset_allows_full_replay(tmp_path):
    path = _make_csv(tmp_path, n_rows=5)
    s = CsvStreamer(path)
    first_pass = [s.next_row()["ENGINE_RPM"] for _ in range(5)]
    s.reset()
    second_pass = [s.next_row()["ENGINE_RPM"] for _ in range(5)]
    assert first_pass == second_pass


# ── peek() ───────────────────────────────────────────────────────────────────

def test_peek_does_not_advance_pointer(tmp_path):
    path = _make_csv(tmp_path, n_rows=5)
    s = CsvStreamer(path)
    before = s.remaining
    s.peek()
    assert s.remaining == before


def test_peek_matches_next_row(tmp_path):
    path = _make_csv(tmp_path, n_rows=5)
    s = CsvStreamer(path)
    peeked = s.peek()
    actual = s.next_row()
    assert peeked == actual


def test_peek_returns_none_when_exhausted(tmp_path):
    path = _make_csv(tmp_path, n_rows=1)
    s = CsvStreamer(path)
    s.next_row()
    assert s.peek() is None


# ── Integration: real drive1.csv ──────────────────────────────────────────────

@pytest.mark.skipif(not DRIVE1.exists(), reason="drive1.csv not available")
def test_real_file_streams_all_rows():
    s = CsvStreamer(DRIVE1)
    count = 0
    while s.next_row() is not None:
        count += 1
    assert count == s.total
    assert s.remaining == 0


@pytest.mark.skipif(not DRIVE1.exists(), reason="drive1.csv not available")
def test_real_file_session_id():
    s = CsvStreamer(DRIVE1)
    assert s.session_id == "drive1"
