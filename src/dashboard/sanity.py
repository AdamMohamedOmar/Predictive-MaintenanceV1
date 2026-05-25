"""Per-row physics sanity check for ingested OBD-II data.

A glitchy ELM327 adapter (especially cheap Bluetooth clones) occasionally
delivers physically impossible rows: ENGINE_RPM=0 while VEHICLE_SPEED=50 km/h,
fuel trims at ±75%, MAP exceeding barometric, etc.  Running classification
on such rows produces confident-sounding false alarms.

This module returns a quality verdict that the dashboard uses to display
"SENSOR DATA INVALID — ADAPTER FAULT?" instead of running inference.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# (lower, upper) physical bounds for each PID.
# Slightly wider than the injector clamps so legitimately healthy edge values
# (cold IAT, high-altitude MAP) don't false-flag.
_BOUNDS: dict[str, tuple[float, float]] = {
    "ENGINE_RPM":                    (0,    8000),
    "VEHICLE_SPEED":                 (0,     250),
    "THROTTLE":                      (0,     100),
    "ENGINE_LOAD":                   (0,     100),
    "COOLANT_TEMPERATURE":           (-40,   135),
    "LONG_TERM_FUEL_TRIM_BANK_1":    (-30,    30),  # OBD ±25% + small margin
    "SHORT_TERM_FUEL_TRIM_BANK_1":   (-30,    30),
    "INTAKE_MANIFOLD_PRESSURE":      (0,     250),  # 250 covers turbo edge case
    "ACCELERATOR_PEDAL_POSITION_D":  (0,     100),
    "ACCELERATOR_PEDAL_POSITION_E":  (0,     100),
    "COMMANDED_THROTTLE_ACTUATOR":   (0,     100),
    "INTAKE_AIR_TEMPERATURE":        (-40,   120),
    "TIMING_ADVANCE":                (-30,    60),
    "CONTROL_MODULE_VOLTAGE":        (6,      18),
}


@dataclass
class QualityVerdict:
    ok: bool
    violations: list[str] = field(default_factory=list)  # human-readable strings for the dashboard


def check_row(row: dict[str, float]) -> QualityVerdict:
    """Validate one row against physical bounds and cross-PID sanity rules.

    NaN values are SKIPPED (an ECU not exposing a PID is not a violation).
    Returns ok=True if no violations; ok=False with a list of one-line
    explanations otherwise.  All violations are also logged at WARNING level.
    """
    violations: list[str] = []

    # Per-PID range checks
    for pid, val in row.items():
        if pid not in _BOUNDS:
            continue
        if val is None:
            continue
        if isinstance(val, float) and math.isnan(val):
            continue
        lo, hi = _BOUNDS[pid]
        if val < lo or val > hi:
            violations.append(f"{pid}={val:.1f} outside [{lo}, {hi}]")

    # Cross-PID rule: engine off while car is moving = impossible
    rpm = row.get("ENGINE_RPM")
    speed = row.get("VEHICLE_SPEED")
    if rpm is not None and speed is not None:
        rpm_ok = not (isinstance(rpm, float) and math.isnan(rpm))
        spd_ok = not (isinstance(speed, float) and math.isnan(speed))
        if rpm_ok and spd_ok and rpm < 200 and speed > 5:
            violations.append(
                f"ENGINE_RPM={rpm:.0f} but VEHICLE_SPEED={speed:.0f} — "
                f"physically impossible (engine off while moving)"
            )

    if violations:
        log.warning("Sanity violations in OBD row: %s", violations)

    return QualityVerdict(ok=len(violations) == 0, violations=violations)
