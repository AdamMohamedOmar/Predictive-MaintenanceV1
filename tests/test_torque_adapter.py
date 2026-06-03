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
