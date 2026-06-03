"""Adapt a Torque / Car-Scanner style ELM327 CSV export into our 14-PID,
clean-column, 1 Hz format so a real-vehicle recording can be fed to the model.

Why an adapter is needed
------------------------
These app exports are nothing like the carOBD training format:
  * ~230 columns, one per PID the app *could* read, most empty per row;
  * round-robin polling at ~6 Hz, so each PID is only sparsely populated
    (each individual PID effectively updates well under 1 Hz);
  * app-specific column names with unit suffixes ("Engine RPM (rpm)");
  * several decoy columns for the same PID, some of which are garbage
    (e.g. "Av Engine Speed of All Cyl (rpm)" = 51199, "Coolant Temp" = 253).

This adapter:
  1. selects the best candidate column for each of our 14 USEFUL_PIDS,
  2. range-filters physically-impossible readings to NaN,
  3. forward-fills each PID (hold-last, the correct semantics for a
     round-robin poll),
  4. resamples to a 1 Hz grid (the rate the whole pipeline assumes),
  5. writes a clean-column CSV the harness/CsvStreamer reads directly.

Run:
    python -m scripts.adapt_torque_csv "C:/path/to/export.csv" --out data/real_faults/ahmed/ahmed_drive.csv

IMPORTANT cross-vehicle caveat: the model is trained on the speed-density
Toyota Etios. A MAF-based car reports MAP as barometric pressure (~101 kPa,
no idle vacuum), so the MAP feature is meaningless here and the Etios baseline
will read this car's normal as anomalous. For a real assessment, capture a
healthy baseline on THIS car and re-fit the normalizer (normalizer_override) —
see docs/REAL_FAULT_COLLECTION.md and scripts/live_baseline_capture.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.config import USEFUL_PIDS

# For each clean PID: candidate source columns matched by PREFIX (so we avoid
# embedding the ℃ / ° unicode unit glyphs), in priority order. The first
# candidate that exists AND yields in-range values wins.
_CANDIDATE_PREFIXES: dict[str, list[str]] = {
    "ENGINE_RPM": ["Engine RPM (rpm)"],
    "VEHICLE_SPEED": ["Vehicle speed (km/h)", "Vehicle Speed (km/h)"],
    "THROTTLE": ["Throttle position (%)", "Throttle Sensor Position (%)"],
    "ENGINE_LOAD": ["Calculated Load_7E0 (%)", "Calculated engine load value (%)"],
    "COOLANT_TEMPERATURE": ["Coolant Temperature_7E0", "Engine coolant temperature"],
    "LONG_TERM_FUEL_TRIM_BANK_1": ["Long term fuel % trim - Bank 1 (%)", "Long FT B1S1 (%)"],
    "SHORT_TERM_FUEL_TRIM_BANK_1": ["Short term fuel % trim - Bank 1 (%)", "Short FT B1S1 (%)"],
    "INTAKE_MANIFOLD_PRESSURE": ["MAP (kPa)", "Manifold Air Pressure_7E0"],
    "ACCELERATOR_PEDAL_POSITION_D": ["Absolute pedal position D (%)"],
    "ACCELERATOR_PEDAL_POSITION_E": ["Absolute pedal position E (%)"],
    "COMMANDED_THROTTLE_ACTUATOR": ["Commanded throttle actuator (%)"],
    "INTAKE_AIR_TEMPERATURE": ["Intake Air Temperature_7E0", "Intake air temperature"],
    "TIMING_ADVANCE": ["Timing advance ("],
    "CONTROL_MODULE_VOLTAGE": ["OBD Module Voltage (V)", "Control module voltage (V)", "+BM Voltage (V)"],
}

# Physical plausibility window per PID. Readings outside → NaN (rejected before
# fill), which kills the garbage decoy columns (RPM=51199, coolant=253, etc.).
_VALID_RANGE: dict[str, tuple[float, float]] = {
    "ENGINE_RPM": (200.0, 8000.0),
    "VEHICLE_SPEED": (0.0, 220.0),
    "THROTTLE": (0.0, 100.0),
    "ENGINE_LOAD": (0.0, 100.0),
    "COOLANT_TEMPERATURE": (-40.0, 150.0),
    "LONG_TERM_FUEL_TRIM_BANK_1": (-30.0, 30.0),
    "SHORT_TERM_FUEL_TRIM_BANK_1": (-30.0, 30.0),
    "INTAKE_MANIFOLD_PRESSURE": (0.0, 115.0),
    "ACCELERATOR_PEDAL_POSITION_D": (0.0, 100.0),
    "ACCELERATOR_PEDAL_POSITION_E": (0.0, 100.0),
    "COMMANDED_THROTTLE_ACTUATOR": (0.0, 100.0),
    "INTAKE_AIR_TEMPERATURE": (-40.0, 90.0),
    "TIMING_ADVANCE": (-30.0, 60.0),
    "CONTROL_MODULE_VOLTAGE": (6.0, 18.0),
}


def _find_column(df: pd.DataFrame, prefixes: list[str], lo: float, hi: float) -> "pd.Series | None":
    """Return the first candidate column (by prefix) that has in-range data."""
    for pref in prefixes:
        matches = [c for c in df.columns if c.startswith(pref)]
        for col in matches:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.where((s >= lo) & (s <= hi))  # out-of-range → NaN
            if s.notna().sum() > 0:
                return s
    return None


def adapt_torque_csv(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Convert a Torque-style export to a 1 Hz, clean-column DataFrame.

    Returns (clean_df, report) where report records, per PID, which source
    column was used and how many raw readings it had (provenance for the thesis).
    """
    path = Path(path)
    raw = pd.read_csv(path)
    if "time" not in raw.columns:
        raise ValueError("Expected a 'time' column (HH:MM:SS.mmm) in the export.")

    t = pd.to_datetime(raw["time"], format="%H:%M:%S.%f", errors="coerce")
    elapsed = (t - t.iloc[0]).dt.total_seconds().to_numpy()

    report: dict = {"source": str(path), "n_raw_rows": int(len(raw)),
                    "duration_s": float(np.nanmax(elapsed)), "pids": {}}

    # Resolve each PID to a sparse Series aligned to the raw row index.
    cols: dict[str, pd.Series] = {}
    for pid in USEFUL_PIDS:
        lo, hi = _VALID_RANGE[pid]
        s = _find_column(raw, _CANDIDATE_PREFIXES[pid], lo, hi)
        if s is None:
            cols[pid] = pd.Series(np.nan, index=raw.index)
            report["pids"][pid] = {"source_column": None, "n_readings": 0}
        else:
            cols[pid] = s
            report["pids"][pid] = {"source_column": s.name, "n_readings": int(s.notna().sum())}

    sparse = pd.DataFrame(cols)
    sparse["__elapsed"] = elapsed
    sparse = sparse.dropna(subset=["__elapsed"]).sort_values("__elapsed")

    # Forward-fill each PID along time (hold-last for a round-robin poll), then
    # back-fill the head so early seconds before a PID's first reading are seeded.
    sparse[USEFUL_PIDS] = sparse[USEFUL_PIDS].ffill().bfill()

    # Resample to a 1 Hz grid: for each integer second take the last reading.
    sparse["__sec"] = np.floor(sparse["__elapsed"]).astype(int)
    per_sec = sparse.groupby("__sec")[USEFUL_PIDS].last()
    full = per_sec.reindex(range(int(per_sec.index.max()) + 1)).ffill().bfill()

    clean = full.reset_index(drop=True)[USEFUL_PIDS]
    report["n_clean_rows_1hz"] = int(len(clean))
    return clean, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapt a Torque ELM327 CSV to model input.")
    ap.add_argument("csv", help="Path to the Torque/Car-Scanner export CSV.")
    ap.add_argument("--out", default=None, help="Output clean-column CSV path.")
    args = ap.parse_args()

    clean, report = adapt_torque_csv(args.csv)

    out = Path(args.out) if args.out else _REPO / "data" / "real_faults" / "ahmed" / "ahmed_drive.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)

    log.info("Adapted %s", args.csv)
    log.info("  raw: %d rows / %.0f s  →  clean: %d rows @ 1 Hz",
             report["n_raw_rows"], report["duration_s"], report["n_clean_rows_1hz"])
    log.info("  per-PID source columns (raw reading counts):")
    for pid, info in report["pids"].items():
        src = info["source_column"]
        log.info("    %-30s ← %s  (%d readings)", pid, src if src else "** MISSING **", info["n_readings"])
    log.info("  Written: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
