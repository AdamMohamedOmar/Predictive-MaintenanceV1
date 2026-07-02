"""Adapt an ELM327 app-export CSV (any car) to the 14-PID, 1 Hz clean format.

The app export uses ONE fixed schema across cars; only which PIDs the ECU
populates varies. It differs from the carOBD training schema in three ways this
adapter reconciles (car-agnostically):

  * 3 columns use app names -> canonical names
        PEDAL_D          -> ACCELERATOR_PEDAL_POSITION_D
        PEDAL_E          -> ACCELERATOR_PEDAL_POSITION_E
        INTAKE_AIR_TEMP  -> INTAKE_AIR_TEMPERATURE
  * INTAKE_MANIFOLD_PRESSURE may be EMPTY (MAF-based cars) -> kept as NaN.
    Every MAP-derived feature is therefore NaN and the air_system fault is NOT
    evaluable on this vehicle. This is reported, never silently zero-filled.
  * effective row rate is ~0.34 Hz (one full 24-PID sweep every ~3 s) -> the
    rows are placed on a 1 Hz grid via HOLD-LAST (no interpolation, so no
    fabricated trends). A real value is held until the next sweep arrives.

Output: a clean CSV with exactly the 14 USEFUL_PIDS columns at 1 Hz, ready for
    python -m scripts.score_recording <out.csv> --pre-adapted ...

WARNING (sampling): 0.34 Hz cannot represent the ~1 Hz closed-loop fuel-trim
oscillation the model trained on — it is aliased away. Hold-last is the least-
fabricating option, but window statistics (esp. STFT std) will not match the
training distribution, so any false-positive rate from this data OVERSTATES the
true rate. Report it with that caveat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from src.config import USEFUL_PIDS  # noqa: E402

log = logging.getLogger(__name__)

_YARIS_RENAME = {
    "PEDAL_D": "ACCELERATOR_PEDAL_POSITION_D",
    "PEDAL_E": "ACCELERATOR_PEDAL_POSITION_E",
    "INTAKE_AIR_TEMP": "INTAKE_AIR_TEMPERATURE",
}

# Generous physical bounds — null out impossible readings (same spirit as the
# loader's guard). NOT calibration limits.
_BOUNDS = {
    "ENGINE_RPM": (0, 8000),
    "VEHICLE_SPEED": (0, 250),
    "THROTTLE": (0, 100),
    "ENGINE_LOAD": (0, 100),
    "COOLANT_TEMPERATURE": (-40, 130),
    "LONG_TERM_FUEL_TRIM_BANK_1": (-100, 100),
    "SHORT_TERM_FUEL_TRIM_BANK_1": (-100, 100),
    "INTAKE_MANIFOLD_PRESSURE": (0, 300),
    "ACCELERATOR_PEDAL_POSITION_D": (0, 100),
    "ACCELERATOR_PEDAL_POSITION_E": (0, 100),
    "COMMANDED_THROTTLE_ACTUATOR": (0, 100),
    "INTAKE_AIR_TEMPERATURE": (-40, 130),
    "TIMING_ADVANCE": (-64, 64),
    "CONTROL_MODULE_VOLTAGE": (0, 18),
}


def adapt_app_csv(path: Path | str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Return (clean 1 Hz DataFrame over USEFUL_PIDS, list of (pid, reason) missing)."""
    df = pd.read_csv(path)
    df = df.rename(columns=_YARIS_RENAME)

    if "timestamp_ms" not in df.columns:
        raise ValueError(f"{path}: no timestamp_ms column; cannot resample to 1 Hz.")

    t = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    sec = ((t - t.iloc[0]) / 1000.0).round()

    out = pd.DataFrame()
    missing: list[tuple[str, str]] = []
    for pid in USEFUL_PIDS:
        if pid not in df.columns:
            out[pid] = np.nan
            missing.append((pid, "absent"))
            continue
        s = pd.to_numeric(df[pid], errors="coerce")
        lo, hi = _BOUNDS.get(pid, (-np.inf, np.inf))
        s = s.where((s >= lo) & (s <= hi))
        if s.notna().sum() == 0:
            missing.append((pid, "empty"))
        out[pid] = s.to_numpy()

    out["__sec"] = sec.to_numpy()

    # Hold-last within the poll (a fully-absent PID stays NaN — never filled).
    out[USEFUL_PIDS] = out[USEFUL_PIDS].ffill().bfill()

    # 1 Hz grid: last reading per integer second, then hold-last across gaps.
    per_sec = out.groupby("__sec")[USEFUL_PIDS].last()
    lo_s, hi_s = int(per_sec.index.min()), int(per_sec.index.max())
    grid = per_sec.reindex(range(lo_s, hi_s + 1)).ffill().bfill()

    clean = grid.reset_index(drop=True)[USEFUL_PIDS]
    return clean, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapt an ELM327 app CSV (any car) to 14-PID 1 Hz.")
    ap.add_argument("csv", help="Path to the raw ELM327 app CSV export (any car).")
    ap.add_argument("--out", required=True, help="Output path for the clean 1 Hz CSV.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    clean, missing = adapt_app_csv(Path(args.csv))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    log.info("Wrote %d rows @1Hz x %d PIDs -> %s", len(clean), clean.shape[1], out)

    for pid, why in missing:
        log.warning(
            "  MISSING PID: %-30s (%s) -> all-NaN; features/faults depending on it "
            "are NOT evaluable on this vehicle.", pid, why
        )
    if not missing:
        log.info("  All 14 USEFUL_PIDS present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())