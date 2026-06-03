"""Tests for the Torque/Car-Scanner ELM327 CSV adapter (scripts.adapt_torque_csv).

A real app export is wide, sparse (round-robin polling), has app-specific
column names with unit suffixes, and decoy columns with garbage values. The
adapter must select the right source per PID, reject out-of-range garbage,
forward-fill the sparse readings, and emit a dense 1 Hz clean-column frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import USEFUL_PIDS
from scripts.adapt_torque_csv import adapt_torque_csv


def _make_torque_fixture(path) -> None:
    """A tiny Torque-style export: ~6 Hz, sparse, with a garbage decoy column."""
    n = 120  # ~20 s at 6 Hz
    base = pd.Timestamp("2026-06-02 16:39:00.000")
    times = [(base + pd.Timedelta(seconds=i / 6.0)).strftime("%H:%M:%S.%f")[:-3] for i in range(n)]
    df = pd.DataFrame({"time": times})

    # Real source columns — sparse: a value every few rows, NaN elsewhere.
    def sparse(values_at):
        col = np.full(n, np.nan)
        for idx, val in values_at.items():
            col[idx] = val
        return col

    df["Engine RPM (rpm)"] = sparse({i: 900 + (i % 5) * 30 for i in range(0, n, 6)})
    # Decoy garbage column for RPM — must be ignored (out of range).
    df["Av Engine Speed of All Cyl (rpm)"] = sparse({i: 51199.2 for i in range(0, n, 6)})
    df["Vehicle speed (km/h)"] = sparse({i: 0.0 for i in range(0, n, 6)})
    df["Throttle position (%)"] = sparse({i: 20.0 for i in range(0, n, 7)})
    df["Calculated Load_7E0 (%)"] = sparse({i: 45.0 for i in range(0, n, 7)})
    df["Coolant Temperature_7E0 (℃)"] = sparse({i: 95.0 for i in range(0, n, 7)})
    # Decoy garbage coolant (out of range) — ignored.
    df["Coolant Temp (℃)"] = sparse({i: 253.0 for i in range(0, n, 6)})
    df["Long term fuel % trim - Bank 1 (%)"] = sparse({i: -8.0 for i in range(0, n, 10)})
    df["Short term fuel % trim - Bank 1 (%)"] = sparse({i: 1.0 for i in range(0, n, 8)})
    df["MAP (kPa)"] = sparse({i: 104.0 for i in range(0, n, 7)})
    df["Absolute pedal position D (%)"] = sparse({i: 15.0 for i in range(0, n, 30)})
    df["Absolute pedal position E (%)"] = sparse({i: 15.0 for i in range(0, n, 30)})
    df["Commanded throttle actuator (%)"] = sparse({i: 20.0 for i in range(0, n, 30)})
    df["Intake Air Temperature_7E0 (℃)"] = sparse({i: 40.0 for i in range(0, n, 7)})
    df["Timing advance (°)"] = sparse({i: 10.0 for i in range(0, n, 10)})
    df["OBD Module Voltage (V)"] = sparse({i: 14.0 for i in range(0, n, 8)})
    # An irrelevant always-empty column (the export is full of these).
    df["ATF temperature v.10 (℃)"] = np.nan

    df.to_csv(path, index=False, encoding="utf-8")


def test_adapter_emits_all_14_pids_at_1hz(tmp_path):
    src = tmp_path / "torque.csv"
    _make_torque_fixture(src)
    clean, report = adapt_torque_csv(src)

    # All 14 PIDs present, dense (no NaN after fill), ~1 row/second.
    assert list(clean.columns) == list(USEFUL_PIDS)
    assert clean.notna().all().all()
    assert 18 <= len(clean) <= 22  # ~20 s recording → ~20 1 Hz rows
    assert report["n_clean_rows_1hz"] == len(clean)


def test_adapter_rejects_garbage_decoys(tmp_path):
    src = tmp_path / "torque.csv"
    _make_torque_fixture(src)
    clean, report = adapt_torque_csv(src)

    # RPM came from the real column (~900), NOT the 51199 decoy.
    assert clean["ENGINE_RPM"].between(850, 1100).all()
    assert report["pids"]["ENGINE_RPM"]["source_column"] == "Engine RPM (rpm)"
    # Coolant came from the _7E0 column (~95), NOT the 253 garbage decoy.
    assert clean["COOLANT_TEMPERATURE"].between(90, 100).all()
    assert report["pids"]["COOLANT_TEMPERATURE"]["source_column"].startswith("Coolant Temperature_7E0")


def test_adapter_forward_fills_sparse_readings(tmp_path):
    src = tmp_path / "torque.csv"
    _make_torque_fixture(src)
    clean, _ = adapt_torque_csv(src)
    # Pedal D had a single reading but must be carried to every 1 Hz row.
    assert (clean["ACCELERATOR_PEDAL_POSITION_D"] == 15.0).all()


# ── P1-1 robustness tests ─────────────────────────────────────────────────────

def _make_iso_time_fixture(path) -> None:
    """Fixture with an ISO-8601 datetime time column (F4a: crashes the old adapter)."""
    n = 120
    base = pd.Timestamp("2026-06-02 16:39:00")
    times = [(base + pd.Timedelta(seconds=i / 6.0)).isoformat() for i in range(n)]
    df = pd.DataFrame({"time": times})

    def sparse(values_at):
        col = np.full(n, np.nan)
        for idx, val in values_at.items():
            col[idx] = val
        return col

    df["Engine RPM (rpm)"] = sparse({i: 900 for i in range(0, n, 6)})
    df["Vehicle Speed (km/h)"] = sparse({i: 50.0 + (i % 10) for i in range(0, n, 6)})
    df["Throttle position (%)"] = sparse({i: 20.0 for i in range(0, n, 7)})
    df["Calculated Load_7E0 (%)"] = sparse({i: 45.0 for i in range(0, n, 7)})
    df["Coolant Temperature_7E0 (℃)"] = sparse({i: 90.0 for i in range(0, n, 7)})
    df["Long term fuel % trim - Bank 1 (%)"] = sparse({i: -2.0 for i in range(0, n, 10)})
    df["Short term fuel % trim - Bank 1 (%)"] = sparse({i: 1.0 for i in range(0, n, 8)})
    df["MAP (kPa)"] = sparse({i: 50.0 for i in range(0, n, 7)})
    df["Absolute pedal position D (%)"] = sparse({i: 15.0 for i in range(0, n, 30)})
    df["Absolute pedal position E (%)"] = sparse({i: 15.0 for i in range(0, n, 30)})
    df["Commanded throttle actuator (%)"] = sparse({i: 20.0 for i in range(0, n, 30)})
    df["Intake Air Temperature_7E0 (℃)"] = sparse({i: 35.0 for i in range(0, n, 7)})
    df["Timing advance (°)"] = sparse({i: 12.0 for i in range(0, n, 10)})
    df["OBD Module Voltage (V)"] = sparse({i: 14.0 for i in range(0, n, 8)})
    df.to_csv(path, index=False, encoding="utf-8")


def _make_no_millis_time_fixture(path) -> None:
    """Fixture with HH:MM:SS time (no milliseconds — F4a variant)."""
    n = 120
    base = pd.Timestamp("2026-06-02 16:39:00")
    times = [(base + pd.Timedelta(seconds=i)).strftime("%H:%M:%S") for i in range(n)]
    df = pd.DataFrame({"time": times})

    def sparse(values_at):
        col = np.full(n, np.nan)
        for idx, val in values_at.items():
            col[idx] = val
        return col

    df["Engine RPM (rpm)"] = sparse({i: 900 for i in range(0, n, 1)})
    df["Vehicle Speed (km/h)"] = sparse({i: 50.0 for i in range(0, n, 1)})
    df["Throttle position (%)"] = sparse({i: 20.0 for i in range(0, n, 1)})
    df["Calculated Load_7E0 (%)"] = sparse({i: 45.0 for i in range(0, n, 1)})
    df["Coolant Temperature_7E0 (℃)"] = sparse({i: 90.0 for i in range(0, n, 1)})
    df["Long term fuel % trim - Bank 1 (%)"] = sparse({i: -2.0 for i in range(0, n, 1)})
    df["Short term fuel % trim - Bank 1 (%)"] = sparse({i: 1.0 for i in range(0, n, 1)})
    df["MAP (kPa)"] = sparse({i: 50.0 for i in range(0, n, 1)})
    df["Absolute pedal position D (%)"] = sparse({i: 15.0 for i in range(0, n, 1)})
    df["Absolute pedal position E (%)"] = sparse({i: 15.0 for i in range(0, n, 1)})
    df["Commanded throttle actuator (%)"] = sparse({i: 20.0 for i in range(0, n, 1)})
    df["Intake Air Temperature_7E0 (℃)"] = sparse({i: 35.0 for i in range(0, n, 1)})
    df["Timing advance (°)"] = sparse({i: 12.0 for i in range(0, n, 1)})
    df["OBD Module Voltage (V)"] = sparse({i: 14.0 for i in range(0, n, 1)})
    df.to_csv(path, index=False, encoding="utf-8")


def _make_moving_car_stuck_speed_fixture(path) -> None:
    """Fixture that reproduces F4b: two speed columns -- stuck-at-0 lowercase 's'
    and a real-motion uppercase 'S'. The old adapter picked the stuck one."""
    n = 120
    base = pd.Timestamp("2026-06-02 16:39:00")
    times = [(base + pd.Timedelta(seconds=i / 6.0)).strftime("%H:%M:%S.%f")[:-3] for i in range(n)]
    df = pd.DataFrame({"time": times})

    def sparse(values_at):
        col = np.full(n, np.nan)
        for idx, val in values_at.items():
            col[idx] = val
        return col

    # Stuck-at-0 decoy (lowercase s) — this was incorrectly preferred
    df["Vehicle speed (km/h)"] = sparse({i: 0.0 for i in range(0, n, 1)})
    # Real motion column (uppercase S)
    df["Vehicle Speed (km/h)"] = sparse({i: 50.0 + (i % 20) for i in range(0, n, 6)})

    df["Engine RPM (rpm)"] = sparse({i: 1500 + (i % 5) * 30 for i in range(0, n, 6)})
    df["Throttle position (%)"] = sparse({i: 25.0 for i in range(0, n, 7)})
    df["Calculated Load_7E0 (%)"] = sparse({i: 45.0 for i in range(0, n, 7)})
    df["Coolant Temperature_7E0 (℃)"] = sparse({i: 90.0 for i in range(0, n, 7)})
    df["Long term fuel % trim - Bank 1 (%)"] = sparse({i: -2.0 for i in range(0, n, 10)})
    df["Short term fuel % trim - Bank 1 (%)"] = sparse({i: 1.0 for i in range(0, n, 8)})
    df["MAP (kPa)"] = sparse({i: 55.0 for i in range(0, n, 7)})
    df["Absolute pedal position D (%)"] = sparse({i: 20.0 for i in range(0, n, 30)})
    df["Absolute pedal position E (%)"] = sparse({i: 20.0 for i in range(0, n, 30)})
    df["Commanded throttle actuator (%)"] = sparse({i: 25.0 for i in range(0, n, 30)})
    df["Intake Air Temperature_7E0 (℃)"] = sparse({i: 35.0 for i in range(0, n, 7)})
    df["Timing advance (°)"] = sparse({i: 12.0 for i in range(0, n, 10)})
    df["OBD Module Voltage (V)"] = sparse({i: 14.0 for i in range(0, n, 8)})
    df.to_csv(path, index=False, encoding="utf-8")


def test_adapter_parses_iso_datetime_time_column(tmp_path):
    """An ISO-8601 time column must parse without crashing (F4a fix).
    The old adapter hard-crashed with ValueError on NaT → int conversion."""
    src = tmp_path / "iso_time.csv"
    _make_iso_time_fixture(src)
    # This CRASHED on the old adapter. Must succeed now.
    clean, report = adapt_torque_csv(src)
    assert len(clean) > 0
    assert report["n_clean_rows_1hz"] > 0


def test_adapter_parses_hhmm_ss_no_millis(tmp_path):
    """An HH:MM:SS time column (no milliseconds) must parse without crashing."""
    src = tmp_path / "no_millis.csv"
    _make_no_millis_time_fixture(src)
    clean, report = adapt_torque_csv(src)
    assert len(clean) > 0
    assert report["n_clean_rows_1hz"] > 0


def test_adapter_rejects_stuck_speed_column_for_moving_car(tmp_path):
    """The all-zero-but-densely-polled 'Vehicle speed (km/h)' decoy must NOT be
    selected when a varying 'Vehicle Speed (km/h)' column is present (F4b fix)."""
    src = tmp_path / "moving_car.csv"
    _make_moving_car_stuck_speed_fixture(src)
    clean, report = adapt_torque_csv(src)

    # Must pick the real-motion column, not the stuck-at-0 one
    assert clean["VEHICLE_SPEED"].mean() > 10.0, (
        f"VEHICLE_SPEED mean is {clean['VEHICLE_SPEED'].mean():.1f} -- "
        "adapter picked the stuck-at-0 decoy instead of the real-motion column"
    )
    assert report["pids"]["VEHICLE_SPEED"]["source_column"] == "Vehicle Speed (km/h)"


def test_adapter_mapping_override_pins_column(tmp_path):
    """--mapping forces a specific column even when a higher-priority candidate exists."""
    import json as _json

    src = tmp_path / "torque.csv"
    _make_torque_fixture(src)

    # The fixture has "Engine RPM (rpm)" as the natural pick.
    # Force it to use "Av Engine Speed of All Cyl (rpm)" via mapping.
    # That decoy is all 51199 -- out of valid range -- so the result for ENGINE_RPM
    # should be NaN (no in-range readings from that forced column).
    mapping = {"ENGINE_RPM": "Av Engine Speed of All Cyl (rpm)"}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(_json.dumps(mapping))

    clean, report = adapt_torque_csv(src, mapping=mapping)
    # The forced column is all out-of-range (51199) → source_column should be None
    assert report["pids"]["ENGINE_RPM"]["source_column"] is None, (
        "Mapping override should have forced the garbage decoy column, "
        "which has no in-range readings."
    )
