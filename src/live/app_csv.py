"""Core adapter: ELM327 app-export DataFrame -> 14-PID, 1 Hz canonical frame.

Shared by scripts/adapt_app_csv.py (CLI) and the dashboard CsvStreamer, so a
raw app export loaded straight into the dashboard gets the SAME rename +
resample treatment as the offline scoring path. Without this, the dashboard
silently dropped PEDAL_D / PEDAL_E / INTAKE_AIR_TEMP (non-canonical names) and
played 0.34 Hz rows as if they were 1 Hz.

The app export uses ONE fixed schema across cars; only which PIDs the ECU
populates varies. Detection: an app-format frame carries a ``timestamp_ms``
column (carOBD and pre-adapted demo files do not).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import USEFUL_PIDS

# App names -> canonical training names.
APP_RENAME = {
    "PEDAL_D": "ACCELERATOR_PEDAL_POSITION_D",
    "PEDAL_E": "ACCELERATOR_PEDAL_POSITION_E",
    "INTAKE_AIR_TEMP": "INTAKE_AIR_TEMPERATURE",
}

# Generous physical bounds — null out impossible readings (same spirit as the
# loader's guard). NOT calibration limits.
BOUNDS = {
    "ENGINE_RPM": (0, 8000),
    "VEHICLE_SPEED": (0, 250),
    "THROTTLE": (0, 100),
    "ENGINE_LOAD": (0, 100),
    "COOLANT_TEMPERATURE": (-40, 130),
    "LONG_TERM_FUEL_TRIM_BANK_1": (-100, 100),
    "SHORT_TERM_FUEL_TRIM_BANK_1": (-100, 100),
    "INTAKE_MANIFOLD_PRESSURE": (0, 300),
    "ACCELERATOR_PEDAL_POSITION_D": (0, 100),
    "ACCELERATOR_PEDAL_POSITION_E": (0, 100),
    "COMMANDED_THROTTLE_ACTUATOR": (0, 100),
    "INTAKE_AIR_TEMPERATURE": (-40, 130),
    "TIMING_ADVANCE": (-64, 64),
    "CONTROL_MODULE_VOLTAGE": (0, 18),
}


def is_app_format(df: pd.DataFrame) -> bool:
    """True for raw ELM327 app exports (identified by the timestamp_ms column)."""
    return "timestamp_ms" in df.columns


def adapt_app_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Adapt a raw app-export frame.

    Returns (clean 1 Hz DataFrame over USEFUL_PIDS, list of (pid, reason) missing).
    A missing PID stays all-NaN — never filled — so downstream Untested handling
    works. Hold-last resampling places ~0.34 Hz sweeps onto a 1 Hz grid without
    fabricating trends.
    """
    df = df.rename(columns=APP_RENAME)

    if "timestamp_ms" not in df.columns:
        raise ValueError("Not an app-format frame: no timestamp_ms column.")

    t = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    sec = ((t - t.iloc[0]) / 1000.0).round()

    out = pd.DataFrame()
    missing: list[tuple[str, str]] = []
    for pid in USEFUL_PIDS:
        if pid not in df.columns:
            out[pid] = np.nan
            missing.append((pid, "absent"))
            continue
        s = pd.to_numeric(df[pid], errors="coerce")
        lo, hi = BOUNDS.get(pid, (-np.inf, np.inf))
        s = s.where((s >= lo) & (s <= hi))
        if s.notna().sum() == 0:
            missing.append((pid, "empty"))
        out[pid] = s.to_numpy()

    out["__sec"] = sec.to_numpy()

    # Hold-last within the poll (a fully-absent PID stays NaN — never filled).
    out[USEFUL_PIDS] = out[USEFUL_PIDS].ffill().bfill()

    # 1 Hz grid: last reading per integer second, then hold-last across gaps.
    per_sec = out.groupby("__sec")[USEFUL_PIDS].last()
    lo_s, hi_s = int(per_sec.index.min()), int(per_sec.index.max())
    grid = per_sec.reindex(range(lo_s, hi_s + 1)).ffill().bfill()

    clean = grid.reset_index(drop=True)[USEFUL_PIDS]
    return clean, missing