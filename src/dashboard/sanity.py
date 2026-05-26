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

    # Cross-PID rule: MAP cannot exceed barometric on a naturally-aspirated engine.
    # +5 kPa tolerance for sensor noise and altitude variation.
    map_val = row.get("INTAKE_MANIFOLD_PRESSURE")
    baro = row.get("ABSOLUTE_BAROMETRIC_PRESSURE", 105.0)  # Skoda at sea-level fallback
    if map_val is not None and not (isinstance(map_val, float) and math.isnan(map_val)):
        if map_val > baro + 5:
            violations.append(f"MAP={map_val:.1f} > BARO={baro:.1f}+5 (NA engine)")

    # Cross-PID rule: throttle actual vs commanded > 30 % = hardware fault, not wear.
    # (Slow TPS potentiometer wear is handled by the ML classifier, not sanity checks.)
    thr = row.get("THROTTLE")
    cmd = row.get("COMMANDED_THROTTLE_ACTUATOR")
    if thr is not None and cmd is not None:
        both_ok = not any(isinstance(v, float) and math.isnan(v) for v in (thr, cmd))
        if both_ok and abs(thr - cmd) > 30:
            violations.append(
                f"THROTTLE={thr:.0f} but COMMANDED_THROTTLE_ACTUATOR={cmd:.0f} "
                f"(delta {abs(thr - cmd):.0f}% > 30% — hardware fault)"
            )

    if violations:
        log.warning("Sanity violations in OBD row: %s", violations)

    return QualityVerdict(ok=len(violations) == 0, violations=violations)


class RowSanityChecker:
    """Stateful wrapper around check_row() that adds cross-row coolant-rate check.

    Coolant temperature cannot change faster than ~1 °C/s due to thermal inertia.
    A jump of > 5 °C/s is a sensor glitch, not a real thermal event.  The stateless
    check_row() cannot detect this because it only sees one row at a time — this class
    remembers the previous row's coolant value and timestamp.

    Usage
    -----
        checker = RowSanityChecker()
        verdict = checker.check(row, now=time.monotonic())
    """

    def __init__(self) -> None:
        self._prev_coolant: float | None = None
        self._prev_t: float | None = None

    def check(self, row: dict[str, float], now: float) -> QualityVerdict:
        """Run stateless + coolant-rate check; return combined verdict."""
        verdict = check_row(row)

        c = row.get("COOLANT_TEMPERATURE")
        if (
            c is not None
            and not (isinstance(c, float) and math.isnan(c))
            and self._prev_coolant is not None
            and self._prev_t is not None
        ):
            dt = max(now - self._prev_t, 1e-3)
            rate = abs(c - self._prev_coolant) / dt
            if rate > 1.0:  # > 1 °C/s violates thermal inertia — CLAUDE.md physics rule
                msg = (
                    f"Coolant jumped {c - self._prev_coolant:+.1f}°C in {dt:.1f}s "
                    f"({rate:.1f}°C/s > 1.0 limit)"
                )
                verdict.violations.append(msg)
                verdict.ok = False
                log.warning("Sanity violation (coolant rate): %s", msg)

        self._prev_coolant = c
        self._prev_t = now
        return verdict

    def reset(self) -> None:
        """Clear previous-row state — call between sessions."""
        self._prev_coolant = None
        self._prev_t = None
