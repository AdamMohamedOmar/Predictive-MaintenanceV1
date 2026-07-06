"""Translation layer between python-OBD command objects and our canonical PID names.

Why a separate module?
-----------------------
python-OBD uses its own command registry (obd.commands.RPM, etc.) and returns
pint-style Quantity objects with units attached.  The rest of our codebase
expects plain floats with canonical column names (ENGINE_RPM, VEHICLE_SPEED…).
This module owns the entire mapping so neither the streamer nor the feature
extractor has to know about python-OBD internals.

Unit alignment
--------------
python-OBD returns values in the units the OBD-II standard specifies.  These
happen to match what the carOBD CSVs contain (the CSVs were recorded by the same
OBD stack), so .magnitude extraction is always unit-safe:

  RPM         → revolutions_per_minute  (e.g. 800.0)
  SPEED       → kph                     (e.g. 50.0)
  COOLANT_TEMP → degree_Celsius         (e.g. 90.0)
  INTAKE_PRESSURE → kilopascal          (e.g. 35.0)
  fuel trims  → percent                 (e.g. +2.5)
  throttle    → percent                 (0–100)
  voltage     → volt                    (e.g. 14.2)
  timing      → degree                  (e.g. 12.5)

Note on COMMANDED_THROTTLE_ACTUATOR
-------------------------------------
python-obd 0.7.2 exposes this as THROTTLE_ACTUATOR (without the "COMMANDED_"
prefix).  The value is the same OBD-II PID 0x4C; only the Python attribute name
differs from what we call it in our canonical PID set.
"""

from __future__ import annotations

import math

import obd
from obd import OBDResponse

# ── Canonical PID name → python-OBD command object ───────────────────────────
#
# Use getattr() for every lookup so a missing command (version mismatch) maps
# to None rather than raising AttributeError.  The streamer skips None entries
# and fills those PIDs with NaN.

PID_MAP: dict[str, obd.OBDCommand] = {
    name: cmd
    for name, cmd in {
        "ENGINE_RPM":                    getattr(obd.commands, "RPM", None),
        "VEHICLE_SPEED":                 getattr(obd.commands, "SPEED", None),
        "THROTTLE":                      getattr(obd.commands, "THROTTLE_POS", None),
        "ENGINE_LOAD":                   getattr(obd.commands, "ENGINE_LOAD", None),
        "COOLANT_TEMPERATURE":           getattr(obd.commands, "COOLANT_TEMP", None),
        "LONG_TERM_FUEL_TRIM_BANK_1":    getattr(obd.commands, "LONG_FUEL_TRIM_1", None),
        "SHORT_TERM_FUEL_TRIM_BANK_1":   getattr(obd.commands, "SHORT_FUEL_TRIM_1", None),
        "INTAKE_MANIFOLD_PRESSURE":      getattr(obd.commands, "INTAKE_PRESSURE", None),
        "ACCELERATOR_PEDAL_POSITION_D":  getattr(obd.commands, "ACCELERATOR_POS_D", None),
        "ACCELERATOR_PEDAL_POSITION_E":  getattr(obd.commands, "ACCELERATOR_POS_E", None),
        # python-obd 0.7.2 uses THROTTLE_ACTUATOR for OBD-II PID 0x4C
        "COMMANDED_THROTTLE_ACTUATOR":   getattr(obd.commands, "THROTTLE_ACTUATOR", None),
        "INTAKE_AIR_TEMPERATURE":        getattr(obd.commands, "INTAKE_TEMP", None),
        "TIMING_ADVANCE":                getattr(obd.commands, "TIMING_ADVANCE", None),
        "CONTROL_MODULE_VOLTAGE":        getattr(obd.commands, "CONTROL_MODULE_VOLTAGE", None),
    }.items()
    if cmd is not None  # filter out missing commands at import time
}


def to_float(response: OBDResponse) -> float:
    """Extract a plain Python float from an OBDResponse.

    Returns ``float('nan')`` for null responses, missing values, or any
    non-numeric response type (e.g. status bit-fields that some ECUs return
    for mode-01 PIDs that are nominally numeric).

    The caller (LiveObdSource) propagates NaN through to the feature extractor,
    which will produce NaN-tainted features for that window — the correct
    signal that a PID was unavailable, rather than silently substituting zero.
    """
    if response.is_null() or response.value is None:
        return float("nan")

    val = response.value

    # python-OBD Quantity objects (obd.Unit.Quantity) have .magnitude
    if hasattr(val, "magnitude"):
        m = val.magnitude
        # Guard against pint returning a numpy scalar that math.isnan rejects
        try:
            return float(m)
        except (TypeError, ValueError):
            return float("nan")

    # Some commands return plain Python numbers directly
    try:
        result = float(val)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def available_commands() -> list[str]:
    """Return canonical PID names whose python-OBD commands resolved successfully."""
    return list(PID_MAP.keys())
