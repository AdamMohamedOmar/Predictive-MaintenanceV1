"""Physics-constrained fault injection for OBD-II time-series data.

Ramp mode: each fault degrades linearly from zero to full magnitude over
`ramp_len` seconds, then holds at full magnitude. This models gradual wear
or progressive sensor degradation — the academically-useful mode for a
predictive maintenance paper.

Step mode (sudden onset) is reserved for a later iteration.

Usage
-----
    from src.injection.fault_injector import inject_fault, inject_session, InjectionParams

    df_clean = load_carobd_csv("data/raw/carOBD/drive1.csv")
    df_faulty = inject_session(df_clean, "fuel_system", random_seed=42)
    print(df_faulty.attrs["fault_label"])   # "fuel_system"
    print(df_faulty.attrs["injection"])     # InjectionParams(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

FaultType = Literal[
    "air_system",
    "fuel_system",
    "coolant_temp_sensor",
    "throttle_position_sensor",
]

# ─── Physical bounds (OBD-II standard + engine physics) ──────────────────────
# Changing these values must be justified by a spec reference or audit finding.
_STFT_MAX = 25.0          # % (OBD-II standard range ±25 %)
_LTFT_MAX = 25.0          # %
_THROTTLE_MIN = 0.0       # %
_THROTTLE_MAX = 100.0     # %
_COOLANT_MIN = 35.0       # °C
_COOLANT_MAX = 130.0      # °C
_COOLANT_MAX_DELTA = 1.0  # °C per sample (thermal inertia — cannot spike instantly)
_TIMING_MIN = -10.0       # ° BTDC
_TIMING_MAX = 40.0        # ° BTDC
_BARO_FALLBACK = 101.0    # kPa — used when ABSOLUTE_BAROMETRIC_PRESSURE absent

# Sensible defaults for each fault at "moderate" severity
_DEFAULT_MAGNITUDE: dict[str, float] = {
    "air_system": 13.0,               # kPa-equivalent leak size (drives idle RPM + MAP, see _inject_air_system)
    "fuel_system": 18.0,              # % LTFT bias at full ramp
    "coolant_temp_sensor": 42.0,      # °C stuck-sensor target temperature
    "throttle_position_sensor": 1.35, # multiplier on THROTTLE at full ramp — widened by _TPS_DEADBAND so full-ramp severity stays = 1.0
}

# ── Speed-density vacuum-leak constants (P0-1) ───────────────────────────────
# carOBD reports MAP + ENGINE_LOAD + BARO and NO MAF PID → this is a
# speed-density (MAP-based) engine.  On such engines the MAP sensor measures
# post-leak manifold pressure directly, so the ECU re-computes airflow and
# largely self-compensates fuel.  The robust, reliable signatures of a vacuum
# leak are therefore mechanical, not fuel-trim:
#   1. raised idle RPM (the leak acts like a partly-open throttle blade),
#   2. slightly elevated MAP at idle / low load (less vacuum),
#   3. only a small, idle-only fuel-trim bump that washes out off-idle.
# Sources: Vehicle Service Pros "Fuel Trim for Diagnostics"; ScannerDanner
# ("MAP-only engines tend to be very tolerant of vacuum leaks … often the
# only symptom is a slightly raised idle speed"); ASE Fuel-Injection Diagnosis.
_AIR_RPM_PER_KPA = 15.0      # idle RPM rise per unit leak magnitude at full ramp (13 → ~195 rpm)
_AIR_IDLE_RPM_CEILING = 1500.0  # a vacuum-leak idle-up rarely exceeds this; clamp for sanity
_AIR_IDLE_RPM_MAX = 1200.0   # rows above this RPM are not "idle" for the bump
_AIR_IDLE_SPEED_MAX = 2.0    # km/h — rows above this are not "idle"
_AIR_MAP_COEFF = 0.5         # MAP delta per leak magnitude (reduced from 1.0; idle-gated)
_AIR_STFT_COEFF = 0.15       # small idle-only STFT bump (cut hard from 0.8)
_AIR_LTFT_COEFF = 0.05       # LTFT marginal on speed-density (greatly shrunk from 0.32)
_AIR_LOAD_COEFF = 0.3        # calculated-load rise per leak magnitude (idle-gated)

# ── Stuck-cold ECT rich-bias constants (P1-2) ────────────────────────────────
# A stuck-cold ECT makes the ECU believe the engine never warmed up, so it
# holds cold-enrichment fuelling.  The rich mixture shows, once closed-loop is
# reached, as NEGATIVE fuel trims — the single most OBD-detectable consequence,
# and one the old injector (timing-retard only) never produced.  Without it a
# real stuck-ECT on the Skoda would present with a signature the model never saw.
_COOLANT_RICH_LTFT = -8.0       # % LTFT at full fault (chronic rich bias)
_COOLANT_RICH_STFT_PEAK = -4.0  # % STFT peak during the transient before handoff

# ── STFT→LTFT steady-state handoff (P0-3) ────────────────────────────────────
# Real adaptive fuel control: STFT is the fast corrector that LEADS during a
# developing fault, then bleeds back toward ~0 (still oscillating) as LTFT
# integrates and HOLDS the persistent offset.  A developed fault therefore
# shows high LTFT and near-zero MEAN STFT.  Modelling both elevated together
# (the old behaviour) trains FUEL_TRIM_DIVERGENCE on a relationship the real
# ECU never reproduces.  Source: Foxwell "Understanding LTFT Bank 1".
_HANDOFF_TAU = 60.0          # rows (≈ seconds at 1 Hz) for STFT to decay post-ramp


def _steady_state_trim(
    ramp: np.ndarray,
    stft_peak: np.ndarray,
    ltft_full: np.ndarray,
    handoff_tau: float = _HANDOFF_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a trim correction into a leading STFT and a lagging/holding LTFT.

    During the ramp STFT tracks the ramp (leads) while LTFT lags at half-weight.
    Once the fault is "developed" (ramp has reached 1.0) STFT decays toward 0
    over ``handoff_tau`` rows while LTFT integrates up to the full offset and
    holds it.

    Parameters
    ----------
    ramp : np.ndarray
        The 0→1 ramp from ``_build_ramp`` (0 pre-onset, 1 post-ramp).
    stft_peak, ltft_full : np.ndarray
        Per-row peak STFT delta and full LTFT offset (may already be
        idle-weighted by the caller).
    handoff_tau : float
        Rows over which STFT bleeds to ~0 after the ramp completes.

    Returns
    -------
    (stft_mean_delta, ltft_delta) : tuple[np.ndarray, np.ndarray]
        Mean (noise-free) trim deltas to add to the baseline trims.  The
        caller adds oscillation noise to STFT separately so it keeps
        wandering around its decaying mean.
    """
    n = len(ramp)
    # First row where the fault is fully developed (ramp == 1.0).
    if np.any(ramp >= 1.0):
        dev_start = int(np.argmax(ramp >= 1.0))
    else:
        dev_start = n  # ramp never completes → no developed region
    t_since_dev = np.maximum(0.0, np.arange(n) - dev_start)
    p = np.clip(t_since_dev / handoff_tau, 0.0, 1.0)  # 0 at ramp end → 1 after tau

    stft_mean = stft_peak * ramp * (1.0 - p)          # leads, then bleeds to 0
    ltft = ltft_full * (0.5 * ramp + 0.5 * p)          # lags, then climbs to full
    return stft_mean, ltft


@dataclass
class InjectionParams:
    """Complete, reproducible specification of one fault injection.

    Stored in the output DataFrame's ``attrs["injection"]`` so any result
    can be traced back to exactly how it was produced.
    """

    fault_type: FaultType
    onset_idx: int       # row index where ramp begins
    ramp_len: int        # rows over which fault grows from 0 → full magnitude
    magnitude: float     # fault-specific unit: kPa / % / °C-target / ratio
    noise_std: float     # Gaussian σ multiplier added to injected signal (0 = deterministic)
    random_seed: int | None


# ─── Public API ──────────────────────────────────────────────────────────────

def inject_fault(df: pd.DataFrame, params: InjectionParams) -> pd.DataFrame:
    """Return a copy of *df* with one fault injected according to *params*.

    The original DataFrame is never modified. The returned copy carries two
    extra ``attrs`` keys:
    - ``"fault_label"``  — the fault_type string (for labelling training data)
    - ``"injection"``    — the InjectionParams used (for reproducibility)

    All injected values are clamped to physical bounds before writing.
    """
    out = df.copy()
    n = len(out)
    rng = np.random.default_rng(params.random_seed)
    ramp = _build_ramp(n, params.onset_idx, params.ramp_len)

    def noise(scale: float) -> np.ndarray:
        if params.noise_std == 0.0 or scale == 0.0:
            return np.zeros(n)
        return rng.normal(0.0, params.noise_std * scale, size=n)

    dispatch: dict[str, Callable] = {
        "air_system": _inject_air_system,
        "fuel_system": _inject_fuel_system,
        "coolant_temp_sensor": _inject_coolant_temp,
        "throttle_position_sensor": _inject_tps,
    }
    if params.fault_type not in dispatch:
        raise ValueError(
            f"Unknown fault_type {params.fault_type!r}. "
            f"Valid options: {list(dispatch)}"
        )

    dispatch[params.fault_type](out, ramp, params.magnitude, noise)
    out.attrs = {**df.attrs, "injection": params, "fault_label": params.fault_type}
    return out


def inject_session(
    df: pd.DataFrame,
    fault_type: FaultType,
    *,
    onset_fraction: float = 0.40,
    ramp_fraction: float = 0.15,
    magnitude: float | None = None,
    noise_std: float = 0.3,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: inject a fault using session-relative fractions.

    Translates ``onset_fraction`` and ``ramp_fraction`` into row indices so
    callers don't need to know the session length in advance.

    Parameters
    ----------
    df : pd.DataFrame
        Clean session from ``load_carobd_csv``.
    fault_type : FaultType
        One of the 4 supported fault strings.
    onset_fraction : float
        Fraction of session rows before fault onset (default 0.40 = 40 %).
        A pre-fault baseline period helps the classifier distinguish healthy
        from early-fault windows.
    ramp_fraction : float
        Fraction of session rows used for the ramp-up (default 0.15 = 15 %).
    magnitude : float or None
        Fault-specific magnitude. Falls back to ``_DEFAULT_MAGNITUDE`` if None.
    noise_std : float
        Gaussian σ multiplier for injected signals. 0 = deterministic.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Injected session with ``attrs["fault_label"]`` and ``attrs["injection"]``.
    """
    n = len(df)
    onset = int(n * onset_fraction)
    ramp_len = max(1, int(n * ramp_fraction))
    mag = magnitude if magnitude is not None else _DEFAULT_MAGNITUDE[fault_type]

    params = InjectionParams(
        fault_type=fault_type,
        onset_idx=onset,
        ramp_len=ramp_len,
        magnitude=mag,
        noise_std=noise_std,
        random_seed=random_seed,
    )
    return inject_fault(df, params)


# ─── Ramp helper ─────────────────────────────────────────────────────────────

def _build_ramp(n_rows: int, onset: int, ramp_len: int) -> np.ndarray:
    """0.0 before onset, linearly 0→1 over ramp_len rows, 1.0 afterwards."""
    t = np.zeros(n_rows, dtype=float)
    ramp_end = min(onset + ramp_len, n_rows)
    if onset < n_rows:
        steps = ramp_end - onset
        t[onset:ramp_end] = np.linspace(0.0, 1.0, steps, endpoint=False)
    if ramp_end < n_rows:
        t[ramp_end:] = 1.0
    return t


# ─── Per-fault injectors ──────────────────────────────────────────────────────

def _inject_air_system(
    df: pd.DataFrame,
    ramp: np.ndarray,
    magnitude_kpa: float,
    noise: Callable[[float], np.ndarray],
) -> None:
    """Speed-density vacuum leak.

    carOBD is a MAP-based (speed-density) engine — there is no MAF PID.  On
    such engines a vacuum leak does NOT produce a large fuel-trim swing: the
    MAP sensor measures the post-leak manifold pressure directly, so the ECU
    re-computes airflow from the higher MAP and largely self-compensates fuel.
    The robust, observable signature is mechanical:

      PRIMARY   ENGINE_RPM rises at idle (leak == partly-open throttle blade)
      PRIMARY   INTAKE_MANIFOLD_PRESSURE rises slightly at idle (less vacuum)
      secondary ENGINE_LOAD rises slightly (higher calculated load)
      small     a tiny, idle-only positive STFT bump that washes out off-idle
                and hands off to a marginal LTFT (P0-3 handoff)

    Off-idle (load high, RPM/speed up) every effect decays toward zero — a
    vacuum leak is a fraction of total airflow at idle but negligible at WOT.

    Clamps: MAP ≤ barometric (naturally-aspirated); idle RPM ≤ ceiling;
    STFT/LTFT within OBD-II ±25 %.
    """
    baro = (
        df["ABSOLUTE_BAROMETRIC_PRESSURE"].to_numpy(dtype=float)
        if "ABSOLUTE_BAROMETRIC_PRESSURE" in df.columns
        else np.full(len(df), _BARO_FALLBACK)
    )

    # Sharper idle-weight than the old 1−load/60: the effect genuinely vanishes
    # off-idle (clip to 0, not 0.3) because on a speed-density engine the leak's
    # contribution to airflow is negligible once the throttle is open.
    load = df["ENGINE_LOAD"].to_numpy(dtype=float)
    idle_weight = np.clip(1.0 - load / 40.0, 0.0, 1.0)

    # ── PRIMARY 1: idle RPM bump ────────────────────────────────────────────
    # Only at genuine idle (low RPM AND ~stationary); the leak pulls the idle
    # speed up like a cracked-open throttle.  Clamp to a sane idle ceiling.
    rpm = df["ENGINE_RPM"].to_numpy(dtype=float)
    speed = (
        df["VEHICLE_SPEED"].to_numpy(dtype=float)
        if "VEHICLE_SPEED" in df.columns
        else np.zeros(len(df))
    )
    idle_mask = (rpm < _AIR_IDLE_RPM_MAX) & (speed < _AIR_IDLE_SPEED_MAX)
    rpm_bump = ramp * idle_weight * magnitude_kpa * _AIR_RPM_PER_KPA + noise(2.0)
    rpm_new = np.where(idle_mask, np.minimum(rpm + rpm_bump, _AIR_IDLE_RPM_CEILING), rpm)
    df["ENGINE_RPM"] = np.clip(rpm_new, a_min=0.0, a_max=None)

    # ── PRIMARY 2: slightly elevated MAP at idle ────────────────────────────
    map_delta = ramp * magnitude_kpa * idle_weight * _AIR_MAP_COEFF + noise(0.3) * idle_weight
    df["INTAKE_MANIFOLD_PRESSURE"] = np.clip(
        df["INTAKE_MANIFOLD_PRESSURE"].to_numpy(dtype=float) + map_delta,
        a_min=0.0,
        a_max=baro,
    )

    # ── secondary: calculated load rises a little at idle ───────────────────
    load_delta = ramp * magnitude_kpa * idle_weight * _AIR_LOAD_COEFF + noise(0.2) * idle_weight
    df["ENGINE_LOAD"] = np.clip(load + load_delta, 0.0, 100.0)

    # ── small idle-only fuel-trim bump with STFT→LTFT handoff (P0-3) ─────────
    # Tiny coefficients, idle-weighted so the bump washes out off-idle.  The
    # handoff makes the developed-state STFT decay toward 0 while a marginal
    # LTFT holds — matching real speed-density behaviour where any trim is
    # small and transient.
    stft_peak = magnitude_kpa * idle_weight * _AIR_STFT_COEFF
    ltft_full = magnitude_kpa * idle_weight * _AIR_LTFT_COEFF
    stft_mean, ltft_delta = _steady_state_trim(ramp, stft_peak, ltft_full)
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_mean + noise(0.3),
        -_STFT_MAX,
        _STFT_MAX,
    )
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_delta + noise(0.05),
        -_LTFT_MAX,
        _LTFT_MAX,
    )


def _inject_fuel_system(
    df: pd.DataFrame,
    ramp: np.ndarray,
    magnitude_pct: float,
    noise: Callable[[float], np.ndarray],
) -> None:
    """Injector clog / low fuel rail pressure.

    Less fuel per spray cycle → chronic lean condition → ECU compensates with
    a rising LTFT. MAP is intentionally left unchanged — that unchanged MAP is
    the key feature distinguishing this fault from an air-system fault.

    STFT→LTFT handoff (P0-3): during the ramp STFT leads (rises first) while
    LTFT lags; once the fault is developed, STFT bleeds back toward 0 (still
    oscillating on noise) and LTFT holds the chronic offset.  A developed
    fuel-system fault therefore shows high LTFT and near-zero MEAN STFT — the
    relationship a real adaptive ECU produces.
    """
    # STFT leads at ~half the offset during the transient, then decays to 0;
    # LTFT lags then climbs to the full offset and holds it.
    stft_peak = np.full(len(ramp), magnitude_pct * 0.5)
    ltft_full = np.full(len(ramp), magnitude_pct)
    stft_mean, ltft_delta = _steady_state_trim(ramp, stft_peak, ltft_full)

    df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_delta + noise(0.2),
        -_LTFT_MAX,
        _LTFT_MAX,
    )
    # STFT keeps its oscillation noise around the decaying mean.
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_mean + noise(1.0),
        -_STFT_MAX,
        _STFT_MAX,
    )
    # MAP deliberately unchanged (differentiator from air_system)

    # ENGINE_LOAD deliberately unchanged: the PID is normalised AIRFLOW, and a
    # clogged injector cuts fuel, not air.  (A real driver compensating for
    # lost torque would RAISE it — driver behaviour is outside replayed data.)

    # ENGINE_RPM: injector clog causes idle roughness — small low-frequency jitter
    # only at low RPM (< 1200 rpm) where cylinder-dropout effects are felt.
    rpm = df["ENGINE_RPM"].to_numpy(dtype=float)
    idle_mask = rpm < 1200.0
    if idle_mask.any():
        rpm_jitter = noise(15.0) * ramp  # up to ±4.5 rpm σ at full ramp
        df["ENGINE_RPM"] = np.clip(
            rpm + np.where(idle_mask, rpm_jitter, 0.0),
            a_min=0.0,
            a_max=None,
        )


def _inject_coolant_temp(
    df: pd.DataFrame,
    ramp: np.ndarray,
    stuck_temp_c: float,
    noise: Callable[[float], np.ndarray],
) -> None:
    """Stuck / biased ECT sensor.

    The sensor reading drifts toward a stuck (too-cold) value. The ECU thinks
    the engine never reached operating temperature and retards ignition timing.

    The 1 °C/sec max-change constraint is enforced sample-by-sample after
    computing the target trajectory, preserving thermal inertia realism.

    Rich-bias signature (P1-2): the ECU holds cold-enrichment fuelling because
    it believes the engine is cold, so once closed-loop is reached the fuel
    trims go NEGATIVE (rich).  This is the dominant OBD-detectable consequence
    and is modelled here with the same STFT→LTFT handoff as the fuel-system
    fault, alongside the retarded timing.
    """
    actual = df["COOLANT_TEMPERATURE"].to_numpy(dtype=float)

    # Target: drift from current reading toward stuck_temp_c
    raw_target = actual + ramp * (stuck_temp_c - actual) + noise(0.3)

    # Enforce thermal inertia: no more than 1 °C change per second
    target = raw_target.copy()
    for i in range(1, len(target)):
        delta = target[i] - target[i - 1]
        target[i] = target[i - 1] + np.clip(delta, -_COOLANT_MAX_DELTA, _COOLANT_MAX_DELTA)

    df["COOLANT_TEMPERATURE"] = np.clip(target, _COOLANT_MIN, _COOLANT_MAX)

    # TIMING_ADVANCE: ECU retards timing proportional to perceived cold deficit.
    # Normal operating temp ≈ 90 °C; max retard ≈ 5° at full fault.
    normal_temp = 90.0
    temp_deficit = np.maximum(0.0, normal_temp - df["COOLANT_TEMPERATURE"].to_numpy(dtype=float))
    timing_retard = ramp * np.minimum(temp_deficit * 0.1, 5.0)
    df["TIMING_ADVANCE"] = np.clip(
        df["TIMING_ADVANCE"].to_numpy(dtype=float) - timing_retard + noise(0.1),
        _TIMING_MIN,
        _TIMING_MAX,
    )

    # Rich fuel-trim bias (P1-2): negative STFT/LTFT from chronic cold-enrichment,
    # with the same STFT-leads-then-hands-off-to-LTFT dynamics as a fuel fault.
    stft_peak = np.full(len(ramp), _COOLANT_RICH_STFT_PEAK)
    ltft_full = np.full(len(ramp), _COOLANT_RICH_LTFT)
    stft_mean, ltft_delta = _steady_state_trim(ramp, stft_peak, ltft_full)
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_mean + noise(0.5),
        -_STFT_MAX,
        _STFT_MAX,
    )
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_delta + noise(0.1),
        -_LTFT_MAX,
        _LTFT_MAX,
    )


def _inject_tps(
    df: pd.DataFrame,
    ramp: np.ndarray,
    drift_factor: float,
    noise: Callable[[float], np.ndarray],
) -> None:
    """TPS potentiometer wear.

    The THROTTLE reading drifts upward relative to the actual pedal position.
    Only active when ACCELERATOR_PEDAL_POSITION_D > 10 % (idle rows are clean
    — the potentiometer is not loaded at rest).

    COMMANDED_THROTTLE_ACTUATOR is left unchanged so the commanded-vs-reported
    divergence is a visible second feature for the classifier.
    """
    pedal = df["ACCELERATOR_PEDAL_POSITION_D"].to_numpy(dtype=float)
    throttle = df["THROTTLE"].to_numpy(dtype=float)

    effective_factor = 1.0 + ramp * (drift_factor - 1.0)
    injected = throttle * effective_factor + noise(0.2)

    # Only apply drift where pedal is engaged; leave idle rows unaffected
    pedal_active = pedal > 10.0
    df["THROTTLE"] = np.clip(
        np.where(pedal_active, injected, throttle),
        _THROTTLE_MIN,
        _THROTTLE_MAX,
    )
