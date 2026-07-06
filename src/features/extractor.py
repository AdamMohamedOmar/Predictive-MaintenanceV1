"""Per-window feature extraction for OBD-II windows.

For each of the 14 working PIDs we compute 5 statistical aggregates:
  mean, std, min, max, delta (last row minus first row)

That gives 70 base features (5 × 14 = 70).

Cross-PID ratio features (4):
  THROTTLE_TO_PEDAL_RATIO, MAP_PER_THROTTLE, FUEL_TRIM_DIVERGENCE,
  THROTTLE_CMD_ACTUAL_DELTA

Trajectory features (4) — capture *how fast things are changing*, not
just the current snapshot.  These are the key to distinguishing a cold
engine that is warming up normally from one whose thermostat is stuck:

  COOLANT_WARMUP_RATE   °C/min slope of coolant over the window.
                        Healthy warm-up: ~1–2 °C/min.
                        Stuck thermostat: < 0.3 °C/min.

  FUEL_LOOP_ACTIVE      1.0 if |STFT| > 0.5% for ≥ 10 rows in the window,
                        else 0.0.  ECU enters closed-loop (fuel trim active)
                        only after coolant > ~60°C; if it fires too early,
                        or never fires despite a warm engine, that is notable.

  RPM_IDLE_DRIFT        std(RPM) across rows where VEHICLE_SPEED < 2 km/h.
                        Healthy idle: very stable RPM.
                        IAC valve fault / injector clog: noisy idle.

  TIMING_VS_TEMP        TIMING_ADVANCE__mean − expected_timing(coolant_mean).
                        Cold ECU retards timing; warm ECU advances it.
                        A negative deviation means the ECU is behaving as if
                        the engine is colder than the coolant reads.

Regime one-hot features (5):
  REGIME__COLD_START, REGIME__WARMUP, REGIME__IDLE, REGIME__ACCEL,
  REGIME__CRUISE — exactly one is 1.0 per window.

Total: 70 + 4 + 4 + 5 = 83 features per window.

Output is a plain Python dict of {feature_name: float}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import USEFUL_PIDS
from src.features.regime import detect_regime, regime_one_hot

# Small epsilon for safe division in ratio features
_EPS = 1e-3


def extract_features(window: pd.DataFrame, sample_hz: float = 1.0) -> dict[str, float]:
    """Return a flat feature dict for one 60-row OBD-II window.

    Parameters
    ----------
    window : pd.DataFrame
        A single window slice produced by ``sliding_windows``.
        Must contain all columns in ``USEFUL_PIDS``.
    sample_hz : float
        Actual ECU poll rate for this session.  Training data is 1 Hz;
        live ELM327 on older ECUs may deliver 0.1–0.5 Hz.  Rate-of-change
        features (COOLANT_WARMUP_RATE, FUEL_LOOP_ACTIVE) are computed in
        physical time units, so they must know the real sample rate.
        Pass 1.0 for training/CSV replay (the carOBD dataset default).

    Returns
    -------
    dict[str, float]
        83 named features ready to be stacked into a training matrix.
    """
    features: dict[str, float] = {}

    # Tolerate a reduced-PID ECU: if a PID column is absent, inject a NaN
    # column so the five stats for that PID become NaN.  Every downstream
    # caller (BaselineNormalizer, InferenceEngine) already NaN-fills those
    # slots with the healthy-baseline mean, so the model still runs.
    if missing := [p for p in USEFUL_PIDS if p not in window.columns]:
        window = window.copy()
        for p in missing:
            window[p] = float("nan")

    for pid in USEFUL_PIDS:
        col = window[pid].to_numpy(dtype=float)
        features[f"{pid}__mean"] = float(np.mean(col))
        features[f"{pid}__std"] = float(np.std(col, ddof=0))
        features[f"{pid}__min"] = float(np.min(col))
        features[f"{pid}__max"] = float(np.max(col))
        features[f"{pid}__delta"] = float(col[-1] - col[0])

    # Cross-PID ratio features
    map_mean      = features["INTAKE_MANIFOLD_PRESSURE__mean"]
    ltft_mean     = features["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
    stft_mean     = features["SHORT_TERM_FUEL_TRIM_BANK_1__mean"]

    # THROTTLE_TO_PEDAL_RATIO: only meaningful when the pedal is engaged.
    # Using mean(throttle) / mean(pedal) inflates to 5000+ at idle (pedal ≈ 0),
    # which drowns the TPS fault signal in mixed windows.  Median of per-row
    # ratios from active-throttle rows only — matches the injector's pedal > 10 %
    # guard exactly.
    pedal_arr    = window["ACCELERATOR_PEDAL_POSITION_D"].to_numpy(dtype=float)
    throttle_arr = window["THROTTLE"].to_numpy(dtype=float)
    active_mask  = pedal_arr > 10.0
    if active_mask.any():
        features["THROTTLE_TO_PEDAL_RATIO"] = float(
            np.median(throttle_arr[active_mask] / (pedal_arr[active_mask] + _EPS))
        )
    else:
        features["THROTTLE_TO_PEDAL_RATIO"] = 1.0  # no pedal input this window → neutral

    # MAP_PER_THROTTLE: only meaningful when the throttle plate is open.
    # At closed throttle MAP / throttle → ∞ (no physical signal, pure noise).
    map_arr   = window["INTAKE_MANIFOLD_PRESSURE"].to_numpy(dtype=float)
    open_mask = throttle_arr > 5.0
    if open_mask.any():
        features["MAP_PER_THROTTLE"] = float(
            np.mean(map_arr[open_mask] / (throttle_arr[open_mask] + _EPS))
        )
    else:
        # All rows at closed throttle — use MAP / 5 % as a stable fallback
        features["MAP_PER_THROTTLE"] = float(map_mean / 5.0)

    features["FUEL_TRIM_DIVERGENCE"] = ltft_mean - stft_mean

    # THROTTLE_CMD_ACTUAL_DELTA: mean(actual − commanded) at open-throttle rows.
    # A TPS potentiometer fault causes THROTTLE to over-read vs COMMANDED_THROTTLE_ACTUATOR.
    # This is a second, independent TPS fault signal alongside THROTTLE_TO_PEDAL_RATIO
    # — combining both in severity gives better SNR than either alone.
    commanded_arr = window["COMMANDED_THROTTLE_ACTUATOR"].to_numpy(dtype=float)
    cmd_open_mask = commanded_arr > 5.0
    if cmd_open_mask.any():
        features["THROTTLE_CMD_ACTUAL_DELTA"] = float(
            np.mean(throttle_arr[cmd_open_mask] - commanded_arr[cmd_open_mask])
        )
    else:
        features["THROTTLE_CMD_ACTUAL_DELTA"] = 0.0

    # Trajectory features — rate-of-change signals for cold-start diagnostics
    coolant = window["COOLANT_TEMPERATURE"].to_numpy(dtype=float)
    n = len(coolant)
    # Row indices → real seconds.  At 1 Hz row i = i seconds; at 0.3 Hz row i = i/0.3 s.
    # Without this correction, a 0.3 Hz stream reads warmup rate 3.3× too high.
    t_sec = np.arange(n, dtype=float) / max(sample_hz, 1e-3)
    slope_per_sec = float(np.polyfit(t_sec, coolant, 1)[0])
    features["COOLANT_WARMUP_RATE"] = slope_per_sec * 60.0  # convert °C/s → °C/min

    stft = window["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float)
    active_rows = int(np.sum(np.abs(stft) > 0.5))
    # Threshold is 10 *seconds* of closed-loop activity, not 10 rows.
    # At 0.3 Hz, 10 rows = 33 s — too strict, flag would never fire on Skoda.
    threshold_rows = max(3, int(round(10.0 * sample_hz)))
    features["FUEL_LOOP_ACTIVE"] = 1.0 if active_rows >= threshold_rows else 0.0

    speed = window["VEHICLE_SPEED"].to_numpy(dtype=float)
    idle_mask = speed < 2.0
    rpm = window["ENGINE_RPM"].to_numpy(dtype=float)
    idle_rpm = rpm[idle_mask]
    features["RPM_IDLE_DRIFT"] = float(np.std(idle_rpm, ddof=0)) if len(idle_rpm) > 1 else 0.0

    coolant_mean = features["COOLANT_TEMPERATURE__mean"]
    # Expected timing advance at operating temp ≈ 15°, drops ~0.2°/°C below 90°C
    expected_timing = 15.0 - max(0.0, 90.0 - coolant_mean) * 0.2
    features["TIMING_VS_TEMP"] = features["TIMING_ADVANCE__mean"] - expected_timing

    # Regime one-hot features
    regime = detect_regime(window)
    features.update(regime_one_hot(regime))

    return features


def feature_names() -> list[str]:
    """Return the ordered list of feature names produced by ``extract_features``.

    Use this to reconstruct column names when loading a saved feature matrix.
    """
    from src.features.regime import regime_feature_names
    names: list[str] = []
    for pid in USEFUL_PIDS:
        for stat in ("mean", "std", "min", "max", "delta"):
            names.append(f"{pid}__{stat}")
    names += [
        "THROTTLE_TO_PEDAL_RATIO",
        "MAP_PER_THROTTLE",
        "FUEL_TRIM_DIVERGENCE",
        "THROTTLE_CMD_ACTUAL_DELTA",
        "COOLANT_WARMUP_RATE",
        "FUEL_LOOP_ACTIVE",
        "RPM_IDLE_DRIFT",
        "TIMING_VS_TEMP",
    ]
    names += regime_feature_names()
    return names
