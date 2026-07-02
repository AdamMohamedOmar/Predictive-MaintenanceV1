"""Per-fault PID requirements and evaluability for cross-vehicle scoring.

Different ECUs expose different PIDs. When a car does not report a PID that a
fault's PRIMARY signal depends on, that fault cannot be honestly scored on that
car: the model still emits a class (XGBoost routes NaN features to a default
split), but the number is meaningless. This module marks such faults as
"Untested" instead of letting a fake score through.

It also reports which USEFUL_PIDS are unavailable at all, because a missing PID
feeds features used across MANY classes (e.g. INTAKE_MANIFOLD_PRESSURE feeds the
MAP_PER_THROTTLE ratio and MAP statistics). Its absence therefore reduces
confidence in EVERY score, not only the fault that names it as primary.

A PID is "available" iff its column exists AND has at least one non-NaN value —
so a present-but-entirely-empty column (the Yaris MAP case) counts as missing.
"""

from __future__ import annotations

import pandas as pd

from src.config import USEFUL_PIDS

# A fault is evaluable only if ALL of its required PIDs are available. "Required"
# means the PID(s) carrying the fault's primary discriminative signal, per the
# injector's fault definitions in src/injection/fault_injector.py.
FAULT_REQUIRED_PIDS: dict[str, set[str]] = {
    "healthy": set(),  # baseline — always evaluable
    "cold_start": {"COOLANT_TEMPERATURE"},
    "air_system": {"INTAKE_MANIFOLD_PRESSURE"},  # speed-density signature dies without MAP
    "fuel_system": {"LONG_TERM_FUEL_TRIM_BANK_1"},  # LTFT is the primary fuel signal
    "coolant_temp_sensor": {"COOLANT_TEMPERATURE"},
    "throttle_position_sensor": {"THROTTLE", "COMMANDED_THROTTLE_ACTUATOR"},  # divergence needs both
}


def available_pids(df: pd.DataFrame) -> set[str]:
    """USEFUL_PIDS present in *df* AND carrying at least one non-NaN value."""
    return {p for p in USEFUL_PIDS if p in df.columns and df[p].notna().any()}


def missing_pids(df: pd.DataFrame) -> list[str]:
    """USEFUL_PIDS absent or entirely NaN in this recording (contract order)."""
    avail = available_pids(df)
    return [p for p in USEFUL_PIDS if p not in avail]


def untested_faults(avail: set[str]) -> dict[str, list[str]]:
    """Map each fault that CANNOT be evaluated -> the required PIDs it lacks."""
    out: dict[str, list[str]] = {}
    for fault, required in FAULT_REQUIRED_PIDS.items():
        lacking = sorted(required - avail)
        if lacking:
            out[fault] = lacking
    return out


def evaluable_faults(avail: set[str]) -> list[str]:
    """Faults whose required PIDs are all available."""
    return [fault for fault, req in FAULT_REQUIRED_PIDS.items() if req <= avail]