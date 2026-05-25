"""PID-based fault severity in [0, 1] — usable at training AND Skoda inference.

Each formula maps current sensor readings to a scalar that is 0 during
healthy operation and 1 at full fault development. The formulas are
physics-grounded (derived from the injection magnitudes in CLAUDE.md) and
require four vehicle-specific baselines:
  - SHORT_TERM_FUEL_TRIM_BANK_1__mean    (healthy STFT mean, %)
  - LONG_TERM_FUEL_TRIM_BANK_1__mean     (healthy LTFT mean, %)
  - THROTTLE_TO_PEDAL_RATIO              (healthy active-throttle median ratio)

The coolant formula uses a fixed baseline (90 °C operating temp) which is
universal across petrol engines.

At Skoda inference time: collect 3–5 minutes of normal driving, compute
STFT, LTFT, and throttle-to-pedal-ratio means from those healthy windows,
pass as baselines here.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-3  # safe divisor for ratio features

# Denominator (full-scale delta) for each fault — matches injection magnitudes
#
# Air system: severity is based on the ECU's lean-correction fuel trim response,
# NOT on MAP directly. MAP varies with throttle position (±15 kPa in normal driving)
# making it too noisy for absolute-value severity. STFT + LTFT are the ECU's
# measured response to unmetered air — they accumulate monotonically and are
# independent of the driving regime.
#   STFT response: 0.8 × 13 kPa = 10.4 % at full ramp
#   LTFT response: 0.32 × 13 kPa = 4.16 % at full ramp
#   Combined scale: 10.4 + 4.16 = 14.56 %
_AIR_SYSTEM_SCALE = 14.56        # % combined fuel trim response at full ramp
                                 # = (STFT coeff 0.8 + LTFT coeff 0.32) × 13 kPa magnitude

_FUEL_SYSTEM_SCALE = 18.0        # % LTFT bias at full ramp
_COOLANT_NORMAL_TEMP = 90.0      # °C — petrol engine normal operating temp
_COOLANT_SCALE = 48.0            # °C deficit at full fault (90 − 42 = 48)
_TPS_SCALE = 0.15                # range above deadband mapped to [0, 1] (0.35 delta − 0.20 deadband)
_TPS_DEADBAND = 0.20             # ratio band treated as natural healthy variance — no fault
                                 # empirically covers cross-session TPS ratio scatter (live12 had 0.19 Δ)
_TPS_MIN_THROTTLE_MEAN = 15.0    # below this throttle mean, ratio is unstable (idle/coast)

# Gate: suppress fuel-trim and coolant severity when the ECU has not yet entered
# closed-loop operation.  STFT/LTFT are frozen in open loop — their values carry
# no diagnostic information.  Set False only during diagnostic testing.
_CLOSED_LOOP_REQUIRED = True


def compute_severity(
    features: dict[str, float],
    fault_type: str,
    baselines: dict[str, float],
) -> float:
    """Return fault severity in [0.0, 1.0] from a window feature dict.

    Parameters
    ----------
    features : dict
        Output of ``extract_features`` for one 60-row window.
    fault_type : str
        One of the 4 fault strings.
    baselines : dict
        Must contain at minimum:
          "SHORT_TERM_FUEL_TRIM_BANK_1__mean"   (for air_system)
          "LONG_TERM_FUEL_TRIM_BANK_1__mean"    (for air_system and fuel_system)
          "THROTTLE_TO_PEDAL_RATIO"              (for throttle_position_sensor)
        Coolant uses a universal fixed baseline (90 °C).

    Returns
    -------
    float in [0.0, 1.0]
    """
    if fault_type == "air_system":
        # STFT/LTFT are frozen in open-loop (cold start, DFCO) — severity computed
        # on frozen trims is garbage.  Gate on FUEL_LOOP_ACTIVE before reading them.
        if _CLOSED_LOOP_REQUIRED and features.get("FUEL_LOOP_ACTIVE", 1.0) < 0.5:
            return 0.0
        # Use ECU fuel trim response — MAP varies ±15 kPa with throttle (SNR < 1:1)
        # STFT+LTFT are the ECU's measured lean correction; they accumulate reliably
        stft_mean = features["SHORT_TERM_FUEL_TRIM_BANK_1__mean"]
        ltft_mean = features["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
        stft_base = baselines["SHORT_TERM_FUEL_TRIM_BANK_1__mean"]
        ltft_base = baselines["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
        combined = (stft_mean + ltft_mean) - (stft_base + ltft_base)
        return float(np.clip(combined / _AIR_SYSTEM_SCALE, 0.0, 1.0))

    if fault_type == "fuel_system":
        # Same open-loop gate — LTFT doesn't integrate during open-loop operation
        if _CLOSED_LOOP_REQUIRED and features.get("FUEL_LOOP_ACTIVE", 1.0) < 0.5:
            return 0.0
        ltft_mean = features["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
        ltft_base = baselines["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
        return float(np.clip((ltft_mean - ltft_base) / _FUEL_SYSTEM_SCALE, 0.0, 1.0))

    if fault_type == "coolant_temp_sensor":
        # A cold engine is NOT a coolant sensor fault. Suppress severity until
        # the ECU is in closed loop and the engine is fully warm (≥ 75 °C).
        # REGIME__COLD_START covers < 55 °C; REGIME__WARMUP covers 55–75 °C.
        # A stuck sensor in this temperature range is indistinguishable from
        # a legitimately warming engine — verification shows live5 has Δsev=0.66
        # at 300 s when coolant is ~58 °C but rising normally.
        if features.get("REGIME__COLD_START", 0.0) >= 0.5:
            return 0.0
        if features.get("REGIME__WARMUP", 0.0) >= 0.5:
            return 0.0
        if _CLOSED_LOOP_REQUIRED and features.get("FUEL_LOOP_ACTIVE", 1.0) < 0.5:
            return 0.0
        cool_mean = features["COOLANT_TEMPERATURE__mean"]
        return float(np.clip((_COOLANT_NORMAL_TEMP - cool_mean) / _COOLANT_SCALE, 0.0, 1.0))

    if fault_type == "throttle_position_sensor":
        # Gate 1: idle/coast windows produce unstable ratios because the
        # active-throttle median is computed from very few samples.
        # Require a meaningful average throttle position in the window.
        if features.get("THROTTLE__mean", 0.0) < _TPS_MIN_THROTTLE_MEAN:
            return 0.0
        ratio = features["THROTTLE_TO_PEDAL_RATIO"]
        ratio_base = baselines["THROTTLE_TO_PEDAL_RATIO"]
        # Gate 2: deadband — natural healthy variance can exceed ±0.05 across
        # cruise vs accel regimes. Only treat positive drift (TPS over-reads)
        # as a fault; negative drift is a different fault mode not modeled.
        delta = ratio - ratio_base
        if delta < _TPS_DEADBAND:
            return 0.0
        # Map the post-deadband range [DEADBAND, DEADBAND+SCALE] → [0, 1]
        return float(np.clip((delta - _TPS_DEADBAND) / _TPS_SCALE, 0.0, 1.0))

    raise ValueError(f"Unknown fault_type: {fault_type!r}")


def compute_baselines(healthy_feature_df) -> dict[str, float]:
    """Compute the two session-dependent baselines from healthy window rows.

    Pass healthy windows from either the training dataset (for offline
    training) or from the first few minutes of a new vehicle's drive
    (for live Skoda inference).

    Parameters
    ----------
    healthy_feature_df : pd.DataFrame
        Rows where label == "healthy", with the standard feature columns.

    Returns
    -------
    dict with keys:
      "INTAKE_MANIFOLD_PRESSURE__mean",
      "SHORT_TERM_FUEL_TRIM_BANK_1__mean",
      "LONG_TERM_FUEL_TRIM_BANK_1__mean",
      "THROTTLE_TO_PEDAL_RATIO".
    """
    # For the TPS ratio baseline, exclude idle-only windows.  The extractor
    # returns a fallback of 1.0 when no row in the window had pedal > 10%,
    # so averaging ALL windows biases the baseline toward 1.0 regardless of
    # the vehicle's actual throttle-to-pedal characteristic.
    # Guard: if "THROTTLE__mean" is absent (legacy test call-sites that only
    # pass the 4 baseline columns), fall back to using all rows.
    if "THROTTLE__mean" in healthy_feature_df.columns:
        active_mask = healthy_feature_df["THROTTLE__mean"] > 10.0
        tps_rows = healthy_feature_df.loc[active_mask]
        if tps_rows.empty:
            tps_rows = healthy_feature_df  # no active-throttle windows → use all
    else:
        tps_rows = healthy_feature_df

    return {
        "INTAKE_MANIFOLD_PRESSURE__mean": float(
            healthy_feature_df["INTAKE_MANIFOLD_PRESSURE__mean"].mean()
        ),
        "SHORT_TERM_FUEL_TRIM_BANK_1__mean": float(
            healthy_feature_df["SHORT_TERM_FUEL_TRIM_BANK_1__mean"].mean()
        ),
        "LONG_TERM_FUEL_TRIM_BANK_1__mean": float(
            healthy_feature_df["LONG_TERM_FUEL_TRIM_BANK_1__mean"].mean()
        ),
        "THROTTLE_TO_PEDAL_RATIO": float(tps_rows["THROTTLE_TO_PEDAL_RATIO"].mean()),
    }
