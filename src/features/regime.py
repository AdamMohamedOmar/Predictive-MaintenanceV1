"""Engine operating regime detector for 60-second OBD-II windows.

Why this exists
---------------
The same sensor reading means different things in different operating states.
COOLANT_TEMPERATURE = 50°C is normal at cold-start, alarming after 20 minutes
of driving.  STFT = 0 is correct in open-loop (cold-start), suspicious in
closed-loop (warmed-up cruise).

By detecting the regime and one-hot encoding it as 5 extra features, the
classifier and feature extractor can condition their judgments on context
without us having to train a separate model per regime.

Regimes (mutually exclusive, priority-ordered top-to-bottom)
------------------------------------------------------------
cold_start  coolant < 55°C                     open-loop ECU, idle-up active
warmup      55°C ≤ coolant < 75°C             transitioning to closed-loop
idle        coolant ≥ 75°C AND speed < 3 km/h  stationary, closed-loop
accel       coolant ≥ 75°C AND pedal std > 8%  dynamic driving
cruise      everything else                     steady closed-loop driving
"""

from __future__ import annotations

import pandas as pd

# Coolant and speed thresholds (single-value, priority-ordered)
_COLD_START_MAX = 55.0   # °C — above this, regime moves out of cold_start
_WARMUP_MAX = 75.0       # °C — above this, regime moves out of warmup
_IDLE_SPEED_MAX = 3.0    # km/h — below this (and warm) = idle
_ACCEL_PEDAL_STD = 8.0   # % std of pedal position — above this = accel

REGIMES = ["cold_start", "warmup", "idle", "accel", "cruise"]


def detect_regime(window: pd.DataFrame) -> str:
    """Return the operating regime for a 60-second window.

    Parameters
    ----------
    window : pd.DataFrame
        A 60-row window from sliding_windows(), with 1-Hz OBD-II rows.

    Returns
    -------
    str
        One of: "cold_start", "warmup", "idle", "accel", "cruise".
    """
    coolant_mean = float(window["COOLANT_TEMPERATURE"].mean()) if "COOLANT_TEMPERATURE" in window.columns else 90.0
    speed_mean = float(window["VEHICLE_SPEED"].mean()) if "VEHICLE_SPEED" in window.columns else 0.0
    pedal_std = float(window["ACCELERATOR_PEDAL_POSITION_D"].std()) if "ACCELERATOR_PEDAL_POSITION_D" in window.columns else 0.0

    if coolant_mean < _COLD_START_MAX:
        return "cold_start"
    if coolant_mean < _WARMUP_MAX:
        return "warmup"
    if speed_mean < _IDLE_SPEED_MAX:
        return "idle"
    if pedal_std > _ACCEL_PEDAL_STD:
        return "accel"
    return "cruise"


def regime_one_hot(regime: str) -> dict[str, float]:
    """Return a dict of 5 binary features for a regime string.

    Feature names are "REGIME__{name}" (uppercase) to match the
    existing feature naming convention in extractor.py.

    Parameters
    ----------
    regime : str
        One of the 5 regime strings from detect_regime().

    Returns
    -------
    dict[str, float]
        5 keys, exactly one is 1.0, the rest are 0.0.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}. Valid: {REGIMES}")
    return {f"REGIME__{r.upper()}": 1.0 if r == regime else 0.0 for r in REGIMES}


def regime_feature_names() -> list[str]:
    """Return the 5 one-hot regime feature column names, in stable order."""
    return [f"REGIME__{r.upper()}" for r in REGIMES]
