"""Rule-based cold-start diagnostics that run parallel to the ML classifier.

Why rules instead of ML here?
------------------------------
The ML classifier is great at recognising PATTERNS it was trained on.
But three cold-start fault types are fundamentally TRAJECTORY faults —
they only make sense when you look at the entire warm-up arc, not a single
60-second window:

  Thermostat stuck open:   coolant never reaches operating temperature
  ECT sensor frozen:       coolant reading is flat despite time passing
  IAC valve fault:         idle RPM stays elevated after warm-up should be done

These require comparing "what the engine has been doing for the last N
minutes" to a physical expectation.  That is what a rule engine does well.

Architecture
------------
ColdStartChecker maintains a rolling buffer of per-second rows from the
live session.  After a warm-up period (configurable), it evaluates three
deterministic checks and emits ColdStartAlert objects.

The dashboard feeds rows into it with update() and reads alerts from
the .alerts property.  Once the engine is fully warm (all checks cleared),
the checker goes dormant — it calls check() is a no-op after warmup.

Rule parameters are chosen conservatively to minimise false positives.
Each rule has a documented physical basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Physical thresholds ──────────────────────────────────────────────────────

# Thermostat: coolant should reach 75°C within this many seconds of engine start.
# A stuck-open thermostat keeps the coolant circulating too fast to hold heat.
# Real-world: Etios warms up in 3–5 minutes; we allow 8 minutes to be generous.
_WARMUP_TARGET_TEMP = 75.0           # °C — operating-temp threshold
_WARMUP_TIMEOUT_S = 480              # seconds before we flag slow warm-up

# ECT sensor frozen: if coolant has been running ≥ 90 s but the std of its
# readings is tiny, the sensor is stuck.  A warming engine always shows
# some variance even if it just oscillates ±1°C.
_FROZEN_SENSOR_MIN_S = 90            # don't check until this many seconds in
_FROZEN_SENSOR_MAX_STD = 0.5         # °C std — below this over 90 s → frozen

# IAC valve: after the engine is warm (coolant ≥ _WARMUP_TARGET_TEMP), idle RPM
# should have settled.  A stuck-high IAC keeps RPM elevated.
_IAC_WARM_MIN_S = 120                # don't check until warm + 2 min settled
_IAC_HIGH_RPM_THRESHOLD = 1100.0     # rpm — normal warm idle is 700–900
_IAC_IDLE_SPEED_MAX = 3.0            # km/h — what counts as idle


@dataclass
class ColdStartAlert:
    """One diagnostic finding from the cold-start rule engine."""
    rule: str           # e.g. "thermostat_stuck_open"
    description: str    # human-readable explanation
    confidence: float   # 0–1; deterministic rules use 0.85–0.95
    triggered_at_s: int # seconds since session start when alert fired


class ColdStartChecker:
    """Stateful cold-start rule engine.  Feed rows one-by-one via update().

    Parameters
    ----------
    warmup_timeout_s : int
        Override the thermostat-timeout threshold (seconds).
    frozen_sensor_min_s : int
        Override how many seconds of data to collect before checking frozen ECT.
    iac_warm_min_s : int
        Override how long after reaching warmup temp before checking IAC.
    """

    def __init__(
        self,
        warmup_timeout_s: int = _WARMUP_TIMEOUT_S,
        frozen_sensor_min_s: int = _FROZEN_SENSOR_MIN_S,
        iac_warm_min_s: int = _IAC_WARM_MIN_S,
    ) -> None:
        self._warmup_timeout_s = warmup_timeout_s
        self._frozen_min_s = frozen_sensor_min_s
        self._iac_warm_min_s = iac_warm_min_s

        # Rolling buffer — list of (coolant, rpm, speed) tuples, one per second
        self._coolant_buf: list[float] = []
        self._rpm_buf: list[float] = []
        self._speed_buf: list[float] = []

        self._elapsed_s: int = 0
        self._warm_since_s: Optional[int] = None  # when coolant first hit target
        self._dormant: bool = False               # True once all checks have cleared

        self._fired: set[str] = set()             # rules that already fired (no repeats)
        self._alerts: list[ColdStartAlert] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, coolant: float, rpm: float, speed: float) -> list[ColdStartAlert]:
        """Ingest one second of sensor data and return any new alerts.

        Parameters
        ----------
        coolant : float   COOLANT_TEMPERATURE in °C
        rpm : float       ENGINE_RPM
        speed : float     VEHICLE_SPEED in km/h

        Returns
        -------
        list[ColdStartAlert]
            New alerts since the last update.  Empty list if nothing new.
        """
        if self._dormant:
            return []

        self._coolant_buf.append(coolant)
        self._rpm_buf.append(rpm)
        self._speed_buf.append(speed)
        self._elapsed_s += 1

        # Track when engine first reached operating temperature
        if self._warm_since_s is None and coolant >= _WARMUP_TARGET_TEMP:
            self._warm_since_s = self._elapsed_s

        new_alerts = self._evaluate()
        self._alerts.extend(new_alerts)

        # Go dormant once warm AND all possible checks have been evaluated
        if self._warm_since_s is not None:
            seconds_since_warm = self._elapsed_s - self._warm_since_s
            if seconds_since_warm >= self._iac_warm_min_s:
                self._dormant = True

        return new_alerts

    def update_from_window(self, window: pd.DataFrame) -> list[ColdStartAlert]:
        """Convenience wrapper: feed all rows in a 60-row window at once.

        Returns the combined list of new alerts from the entire window.
        """
        all_new: list[ColdStartAlert] = []
        for _, row in window.iterrows():
            new = self.update(
                coolant=float(row["COOLANT_TEMPERATURE"]),
                rpm=float(row["ENGINE_RPM"]),
                speed=float(row["VEHICLE_SPEED"]),
            )
            all_new.extend(new)
        return all_new

    @property
    def alerts(self) -> list[ColdStartAlert]:
        """All alerts fired so far in this session."""
        return list(self._alerts)

    @property
    def is_dormant(self) -> bool:
        """True once the engine is warm and all checks have been evaluated."""
        return self._dormant

    def reset(self) -> None:
        """Clear all state — call between sessions."""
        self._coolant_buf.clear()
        self._rpm_buf.clear()
        self._speed_buf.clear()
        self._elapsed_s = 0
        self._warm_since_s = None
        self._dormant = False
        self._fired.clear()
        self._alerts.clear()

    # ── Internal rule evaluations ─────────────────────────────────────────────

    def _evaluate(self) -> list[ColdStartAlert]:
        new: list[ColdStartAlert] = []

        alert = self._check_thermostat()
        if alert:
            new.append(alert)

        alert = self._check_frozen_sensor()
        if alert:
            new.append(alert)

        alert = self._check_iac_valve()
        if alert:
            new.append(alert)

        return new

    def _check_thermostat(self) -> Optional[ColdStartAlert]:
        """Coolant never reached operating temperature within timeout."""
        rule = "thermostat_stuck_open"
        if rule in self._fired:
            return None
        if self._warm_since_s is not None:
            return None  # engine did warm up — rule cleared
        if self._elapsed_s < self._warmup_timeout_s:
            return None  # haven't timed out yet

        self._fired.add(rule)
        return ColdStartAlert(
            rule=rule,
            description=(
                f"Coolant never reached {_WARMUP_TARGET_TEMP}°C after "
                f"{self._elapsed_s}s. Thermostat may be stuck open, "
                f"preventing heat build-up."
            ),
            confidence=0.90,
            triggered_at_s=self._elapsed_s,
        )

    def _check_frozen_sensor(self) -> Optional[ColdStartAlert]:
        """ECT sensor reading is completely flat during warm-up — sensor is stuck.

        Scope limitation
        ----------------
        This check only fires *during the warm-up phase* (before coolant first
        reaches _WARMUP_TARGET_TEMP).  Two failure modes are therefore invisible:

        1. Sensor stuck at warm operating temp from key-on (e.g. reads 90 °C
           even when the engine is cold).  The coolant would appear to "warm up"
           instantly, skipping this rule entirely.

        2. Sensor that was healthy during warm-up but freezes afterward.  The
           checker goes dormant once warm, so a post-warmup freeze is undetected.

        Both cases would require a continuous plausibility check against IAT
        and engine-off soak time, which is out of scope for this rule engine.
        """
        rule = "ect_sensor_frozen"
        if rule in self._fired:
            return None
        # A stable reading at operating temperature is completely normal — a healthy
        # warm engine sits at ~90 °C ± 0.2 °C.  Only flag a frozen sensor while the
        # engine is still supposed to be warming up (i.e. before it first reached
        # the warmup target).  After that point, a flat signal is expected.
        if self._warm_since_s is not None:
            return None
        if self._elapsed_s < self._frozen_min_s:
            return None

        coolant_arr = np.array(self._coolant_buf[-self._frozen_min_s:])
        std = float(np.std(coolant_arr, ddof=0))
        if std >= _FROZEN_SENSOR_MAX_STD:
            return None  # sensor is moving normally

        self._fired.add(rule)
        return ColdStartAlert(
            rule=rule,
            description=(
                f"COOLANT_TEMPERATURE std = {std:.2f}°C over the last "
                f"{self._frozen_min_s}s.  A real engine always shows some "
                f"thermal variation; a flat signal indicates a stuck sensor."
            ),
            confidence=0.95,
            triggered_at_s=self._elapsed_s,
        )

    def _check_iac_valve(self) -> Optional[ColdStartAlert]:
        """Idle RPM is still elevated well after engine reached operating temp."""
        rule = "iac_valve_stuck_open"
        if rule in self._fired:
            return None
        if self._warm_since_s is None:
            return None  # not warm yet
        seconds_since_warm = self._elapsed_s - self._warm_since_s
        if seconds_since_warm < self._iac_warm_min_s:
            return None  # give the IAC time to close

        # Only evaluate at-idle rows
        recent_n = min(60, len(self._rpm_buf))
        rpm_arr = np.array(self._rpm_buf[-recent_n:])
        spd_arr = np.array(self._speed_buf[-recent_n:])
        idle_rpm = rpm_arr[spd_arr < _IAC_IDLE_SPEED_MAX]

        if len(idle_rpm) < 5:
            return None  # not enough idle samples

        mean_idle_rpm = float(np.mean(idle_rpm))
        if mean_idle_rpm <= _IAC_HIGH_RPM_THRESHOLD:
            return None

        self._fired.add(rule)
        return ColdStartAlert(
            rule=rule,
            description=(
                f"Mean idle RPM = {mean_idle_rpm:.0f} after {seconds_since_warm}s "
                f"of warm engine.  Normal warm idle is 700–900 RPM; elevated idle "
                f"suggests the IAC valve is stuck open or the throttle body is dirty."
            ),
            confidence=0.85,
            triggered_at_s=self._elapsed_s,
        )
