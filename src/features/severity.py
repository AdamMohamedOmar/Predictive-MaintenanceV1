"""PID-based fault severity in [0, 1] — usable at training AND Skoda inference.

Each formula maps current sensor readings to a scalar that is 0 during
healthy operation and 1 at full fault development.

Scales are derived from EXTERNAL diagnostic thresholds (OBD-II fuel-trim
limits, ECT plausibility), NOT from the injector's own coefficients (P0-2).
The previous version set ``_AIR_SYSTEM_SCALE = (0.8 + 0.32) × 13`` — the
algebraic inverse of the injector — so the supervised target was a
deterministic function of the data-generator and the held-out F1 measured
generator-recovery, not fault detection. Each scale below now cites a
real-world diagnostic line so the same severity definition is meaningful on
the Skoda's real faults.  The formulas require these vehicle-specific
baselines:
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

# ── Severity scales — sourced to external diagnostic thresholds (P0-2) ──────
#
# Air system (speed-density vacuum leak): on a MAP-based engine the leak's
# fuel-trim signature is SMALL and idle-only (the ECU self-compensates from
# the directly-measured MAP — see fault_injector._inject_air_system).  We
# therefore scale the combined idle fuel-trim deviation by the OBD-II
# "lean-trim concern" line: a sustained total fuel trim beyond ~10 % is the
# widely-cited threshold at which a lean condition is diagnostically notable
# (OBD-II fuel-trim diagnostic guides).  This is a DIAGNOSTIC threshold, not
# the injector's coefficient.  Severity is idle-gated because off-idle the
# leak is undetectable.
_AIR_SYSTEM_SCALE = 10.0         # % combined idle fuel-trim deviation at "clearly lean" (OBD lean-trim watch line)
_AIR_IDLE_LOAD_MAX = 40.0        # % calculated load — above this a speed-density leak washes out

# Fuel system: LTFT is the ECU's learned lean correction.  ±10 % is the watch
# line and ±20–25 % is where a real ECU sets a lean DTC (e.g. P0171).  We scale
# to the 20 % "problem" line (Fleetrabbit / standard OBD-II fuel-trim guidance),
# NOT the 18 % injection magnitude the old code used.
_FUEL_SYSTEM_SCALE = 20.0        # % LTFT deviation at the diagnostic "problem" line (±20 %)

_COOLANT_NORMAL_TEMP = 90.0      # °C — petrol engine normal operating temp (thermostat setpoint)
# A coolant reading ≥ 40 °C below the thermostat setpoint, on an engine that
# has run long enough to be warm, is an unambiguous sensor fault — the engine
# physically cannot run that cold once warmed.  Scale to that 40 °C deficit
# (a plausibility threshold), NOT the injector's (90 − 42 = 48) inverse.
_COOLANT_SCALE = 40.0            # °C deficit at which a stuck-cold reading is unambiguously faulty
# P1-1: distinguish a stuck-cold ECT from a legitimately warming engine by
# warm-up DYNAMICS, not absolute temperature.  A real warm-up climbs ~1–2 °C/min;
# a stuck sensor sits near 0 °C/min.  Gating coolant severity on "coolant <55 °C"
# (the old behaviour) made the fault's own symptom — a permanently-cold reading —
# trigger the gate that nullified it (coolant severity_target_mean was 0.0028).
_COOLANT_STUCK_MAX = 75.0        # °C — a reading at/above this can't be a stuck-COLD sensor
_WARMUP_RATE_HEALTHY = 0.5       # °C/min — at/above this the engine is actively warming (not stuck)
# TPS scales are threshold-based, not injector-derived: the deadband is the
# observed healthy cross-session ratio scatter, and full severity is reached at
# a ~0.35 throttle-vs-pedal over-read (a 35 % divergence is an unambiguous TPS
# correlation fault — cf. P2135 correlation-error diagnostics).
_TPS_SCALE = 0.15                # maps (deadband → 0.35 over-read) onto [0, 1]
_TPS_DEADBAND = 0.20             # healthy cross-session ratio scatter (live12 had 0.19 Δ) — no fault below this
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
        # Speed-density physics (P0-1): a vacuum leak's fuel-trim signature is
        # idle-only and washes out off-idle.  Gate on low calculated load so we
        # only grade severity where the leak is actually observable.
        if features.get("ENGINE_LOAD__mean", 100.0) > _AIR_IDLE_LOAD_MAX:
            return 0.0
        # Small idle lean-correction the ECU still applies (STFT leads, LTFT
        # holds a marginal offset).  Scaled to the OBD lean-trim watch line.
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
        # P1-1: gate on warm-up DYNAMICS, not absolute temperature.  The old
        # regime/fuel-loop gates all keyed on "coolant is cold" — which a
        # stuck-cold sensor reports forever, so the fault gated away its own
        # severity (committed coolant target_mean was 0.0028 ≈ constant zero).
        cool_mean = features["COOLANT_TEMPERATURE__mean"]
        # A warm reading cannot be a stuck-COLD sensor — no fault.
        if cool_mean >= _COOLANT_STUCK_MAX:
            return 0.0
        # Coolant is cold.  If it is CLIMBING at a healthy rate the engine is
        # legitimately warming up — not a fault.  Only a cold AND flat reading
        # (the sensor stuck while the engine should be heating) is the fault.
        # COOLANT_WARMUP_RATE is °C/min over the window (see extractor.py).
        # Note: a perfect run-time signal (ENGINE_RUN_TIME) would further
        # disambiguate the rare "genuinely cold engine, barely warming" window;
        # warm-up rate is the discriminator the extractor exposes today.
        warmup_rate = features.get("COOLANT_WARMUP_RATE", 0.0)
        if warmup_rate >= _WARMUP_RATE_HEALTHY:
            return 0.0
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
        # Term 1: throttle-to-pedal ratio drift (primary TPS indicator)
        ratio_term = np.clip((delta - _TPS_DEADBAND) / _TPS_SCALE, 0.0, 1.0)

        # Term 2: commanded-vs-actual throttle gap (secondary TPS indicator).
        # THROTTLE_CMD_ACTUAL_DELTA > 10 % gap → full severity contribution.
        # This term is independent of the pedal sensor, so the two terms are
        # complementary: one is noisy when the car is coasting, the other is not.
        cmd_delta = features.get("THROTTLE_CMD_ACTUAL_DELTA", 0.0)
        cmd_term = float(np.clip(cmd_delta / 10.0, 0.0, 1.0))

        return float(np.clip(0.5 * ratio_term + 0.5 * cmd_term, 0.0, 1.0))

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
