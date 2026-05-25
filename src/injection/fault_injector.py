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
    "air_system": 13.0,               # kPa MAP offset at full ramp
    "fuel_system": 18.0,              # % LTFT bias at full ramp
    "coolant_temp_sensor": 42.0,      # °C stuck-sensor target temperature
    "throttle_position_sensor": 1.35, # multiplier on THROTTLE at full ramp — widened by _TPS_DEADBAND so full-ramp severity stays = 1.0
}


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
    """Vacuum leak / MAF drift.

    Extra air bypasses the MAF sensor → MAP reads high for a given throttle
    angle → ECU senses lean exhaust → STFT climbs positive → LTFT slowly
    integrates the STFT offset.

    MAP clamp: naturally-aspirated engine cannot exceed barometric pressure.
    STFT/LTFT clamp: OBD-II standard ±25 %.
    """
    baro = (
        df["ABSOLUTE_BAROMETRIC_PRESSURE"].to_numpy(dtype=float)
        if "ABSOLUTE_BAROMETRIC_PRESSURE" in df.columns
        else np.full(len(df), _BARO_FALLBACK)
    )

    map_delta = ramp * magnitude_kpa + noise(0.3)
    df["INTAKE_MANIFOLD_PRESSURE"] = np.clip(
        df["INTAKE_MANIFOLD_PRESSURE"].to_numpy(dtype=float) + map_delta,
        a_min=0.0,
        a_max=baro,
    )

    # STFT lean-correction: approximately 0.8× the effective air-excess signal
    stft_delta = ramp * magnitude_kpa * 0.8 + noise(0.5)
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_delta,
        -_STFT_MAX,
        _STFT_MAX,
    )

    # LTFT: slow integrator — reaches ~40 % of STFT magnitude at full ramp
    ltft_delta = ramp * magnitude_kpa * 0.32 + noise(0.1)
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_delta,
        -_LTFT_MAX,
        _LTFT_MAX,
    )

    # ENGINE_LOAD: extra unmetered air means the engine is doing more work than
    # the ECU believes — load reading rises slightly (~30 % of MAP delta in %)
    load_delta = ramp * magnitude_kpa * 0.3 + noise(0.2)
    df["ENGINE_LOAD"] = np.clip(
        df["ENGINE_LOAD"].to_numpy(dtype=float) + load_delta,
        0.0,
        100.0,
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

    STFT oscillates with positive mean (~30 % of LTFT delta) as the ECU hunts
    around the trim offset.
    """
    ltft_delta = ramp * magnitude_pct + noise(0.2)
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_delta,
        -_LTFT_MAX,
        _LTFT_MAX,
    )

    # STFT: positive-biased oscillation around the LTFT offset
    stft_bias = ramp * magnitude_pct * 0.3
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
        df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_bias + noise(1.0),
        -_STFT_MAX,
        _STFT_MAX,
    )
    # MAP deliberately unchanged (differentiator from air_system)

    # ENGINE_LOAD: lean misfires mean less useful work per cycle — load drops
    # slightly at full ramp (~1.5 % load reduction per % LTFT bias).
    load_delta = ramp * magnitude_pct * 0.08  # ~1.4 % drop at full 18 % LTFT fault
    df["ENGINE_LOAD"] = np.clip(
        df["ENGINE_LOAD"].to_numpy(dtype=float) - load_delta,
        0.0,
        100.0,
    )

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
