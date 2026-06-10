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

Column selection notes
----------------------
The adapter uses a priority-ordered prefix list for each PID and picks the
first candidate that (a) has in-range data and (b) is not stuck at a constant
value for PIDs that should vary (speed, RPM, throttle, load).  This prevents
the known F4b bug where a stuck-at-0 "Vehicle speed (km/h)" column is preferred
over the real-motion "Vehicle Speed (km/h)" column.

If the auto-selection is wrong, pass --mapping mapping.json to pin columns:
    {"VEHICLE_SPEED": "Vehicle Speed (km/h)", "ENGINE_RPM": "Engine RPM (rpm)"}

Time parsing
------------
The adapter tries multiple time formats (HH:MM:SS.ffffff, HH:MM:SS, ISO-8601,
numeric epoch) before falling back to row-index / --rate-hz.  This prevents
the known F4a hard-crash on non-Torque exports.

IMPORTANT cross-vehicle caveat: the model is trained on the speed-density
Toyota Etios. A MAF-based car reports MAP as barometric pressure (~101 kPa,
no idle vacuum), so the MAP feature is meaningless here and the Etios baseline
will read this car's normal as anomalous. For a real assessment, capture a
healthy baseline on THIS car and re-fit the normalizer (normalizer_override) --
see docs/REAL_FAULT_COLLECTION.md and scripts/live_baseline_capture.py.
"""

from __future__ import annotations

import argparse
import json
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
# candidate that exists AND yields in-range values AND (for vary-expected PIDs)
# is not stuck at a constant value wins.
# Broadened to cover Skoda/Car-Scanner variants that omit "_7E0" ECU suffixes.
_CANDIDATE_PREFIXES: dict[str, list[str]] = {
    "ENGINE_RPM": ["Engine RPM (rpm)", "Engine speed (rpm)"],
    "VEHICLE_SPEED": [
        "Vehicle Speed (km/h)",   # capital S variant first (more likely to be real motion)
        "Vehicle speed (km/h)",   # lowercase s — often the stuck-at-0 decoy in Torque
        "GPS Speed (Kilometers/hour)",
        "Speed (km/h)",
    ],
    "THROTTLE": [
        "Throttle position (%)",
        "Throttle Sensor Position (%)",
        "Throttle Position (Manifold) (%)",
    ],
    "ENGINE_LOAD": [
        "Calculated Load_7E0 (%)",
        "Calculated engine load value (%)",
        "Engine load (%)",
        "Engine Load (%)",
    ],
    "COOLANT_TEMPERATURE": [
        "Coolant Temperature_7E0",
        "Engine coolant temperature",
        "Coolant temp",
        "ECT (",
    ],
    "LONG_TERM_FUEL_TRIM_BANK_1": [
        "Long term fuel % trim - Bank 1 (%)",
        "Long FT B1S1 (%)",
        "Long Term Fuel Trim Bank 1 (%)",
    ],
    "SHORT_TERM_FUEL_TRIM_BANK_1": [
        "Short term fuel % trim - Bank 1 (%)",
        "Short FT B1S1 (%)",
        "Short Term Fuel Trim Bank 1 (%)",
    ],
    "INTAKE_MANIFOLD_PRESSURE": [
        "MAP (kPa)",
        "Manifold Air Pressure_7E0",
        "Intake Manifold Pressure (kPa)",
        "Manifold pressure (kPa)",
    ],
    "ACCELERATOR_PEDAL_POSITION_D": [
        "Absolute pedal position D (%)",
        "Accelerator PedalPosition D (%)",
    ],
    "ACCELERATOR_PEDAL_POSITION_E": [
        "Absolute pedal position E (%)",
        "Accelerator PedalPosition E (%)",
    ],
    "COMMANDED_THROTTLE_ACTUATOR": [
        "Commanded throttle actuator (%)",
        "Throttle Actuator Control (%)",
    ],
    "INTAKE_AIR_TEMPERATURE": [
        "Intake Air Temperature_7E0",
        "Intake air temperature",
        "Intake Air Temperature (",
        "Air intake temperature (",
    ],
    "TIMING_ADVANCE": ["Timing advance ("],
    "CONTROL_MODULE_VOLTAGE": [
        "OBD Module Voltage (V)",
        "Control module voltage (V)",
        "+BM Voltage (V)",
        "Voltage (V)",
    ],
}

# PIDs that physically must vary during a moving drive.  A candidate column
# that is stuck at a single constant value for these PIDs is a decoy and must
# be rejected regardless of how many readings it has.
_MUST_VARY_PIDS: frozenset[str] = frozenset({
    "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
})

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
    "INTAKE_AIR_TEMPERATURE": (-40.0, 120.0),
    "TIMING_ADVANCE": (-30.0, 60.0),
    "CONTROL_MODULE_VOLTAGE": (6.0, 18.0),
}


def _parse_elapsed(raw: pd.DataFrame, rate_hz: float = 1.0) -> np.ndarray:
    """Convert the 'time' column to elapsed seconds (length == len(raw)).

    Fallback chain (F4a fix):
      1. HH:MM:SS.ffffff  (Torque/Car-Scanner default)
      2. HH:MM:SS         (millis absent)
      3. ISO-8601 / any format (pd.to_datetime auto-inference)
      4. Numeric epoch (seconds or ms since midnight)
      5. Row-index / rate_hz  (last resort -- emits a warning)
    """
    if "time" not in raw.columns:
        log.warning("No 'time' column found -- using row-index / %.1f Hz.", rate_hz)
        return np.arange(len(raw), dtype=float) / rate_hz

    # Strategies 1-3: datetime string parsing
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", None):
        kwargs: dict = {"errors": "coerce"}
        if fmt is not None:
            kwargs["format"] = fmt
        t = pd.to_datetime(raw["time"], **kwargs)
        valid = t.notna()
        if valid.sum() > len(raw) * 0.5:
            t0 = t[valid].iloc[0]
            return (t - t0).dt.total_seconds().to_numpy().astype(float)

    # Strategy 4: numeric epoch (seconds or milliseconds)
    s = pd.to_numeric(raw["time"], errors="coerce")
    valid = s.notna()
    if valid.sum() > len(raw) * 0.5:
        vals = s[valid].values.astype(float)
        if vals.mean() > 1e6:   # milliseconds since midnight
            vals = vals / 1000.0
        t0 = vals[0]
        result = np.full(len(raw), np.nan, dtype=float)
        result[valid.values] = vals - t0
        return result

    # Strategy 5: row-index fallback
    log.warning(
        "Could not parse 'time' column (tried HH:MM:SS.f, HH:MM:SS, ISO, numeric). "
        "Falling back to row-index / %.1f Hz.  Pass --rate-hz to correct this.",
        rate_hz,
    )
    return np.arange(len(raw), dtype=float) / rate_hz


def _find_column(
    df: pd.DataFrame,
    pid: str,
    prefixes: list[str],
    lo: float,
    hi: float,
    mapping: "dict[str, str] | None" = None,
) -> "pd.Series | None":
    """Return the best candidate column for a PID (F4b variance-aware fix).

    Selection order:
      1. Explicit --mapping entry for this PID (always wins if column present).
      2. Auto two-pass:
           Pass 1 -- collect all in-range candidates in priority order.
           Pass 2 -- for PIDs in _MUST_VARY_PIDS, if multiple candidates exist,
                     prefer the one with non-zero variance (i.e. the real-motion
                     column, not a stuck-at-0 decoy).  If only ONE candidate
                     exists, accept it regardless of variance (it may be a
                     legitimately parked/constant session).
    """
    # Explicit mapping override
    if mapping and pid in mapping:
        col_name = mapping[pid]
        if col_name in df.columns:
            s = pd.to_numeric(df[col_name], errors="coerce")
            s = s.where((s >= lo) & (s <= hi))
            return s if s.notna().sum() > 0 else None
        log.warning(
            "Mapping column '%s' not found in CSV for PID %s -- falling back to auto.",
            col_name, pid,
        )

    # Pass 1: collect all in-range candidates in priority order
    candidates: list[pd.Series] = []
    for pref in prefixes:
        matches = [c for c in df.columns if c.startswith(pref)]
        for col in matches:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.where((s >= lo) & (s <= hi))
            if s.notna().sum() > 0:
                candidates.append(s)

    if not candidates:
        return None

    # Pass 2: for vary-expected PIDs with multiple candidates, prefer varying ones
    if pid in _MUST_VARY_PIDS and len(candidates) > 1:
        varying = [s for s in candidates if s.dropna().std() > 0.0]
        if varying:
            # At least one non-stuck candidate exists; pick the first (highest priority)
            return varying[0]
        # All candidates are stuck (e.g. genuinely parked car) -- fall through
        log.debug(
            "All %d candidates for PID %s are zero-variance -- car may be parked.",
            len(candidates), pid,
        )

    return candidates[0]


def adapt_torque_csv(
    path: Path | str,
    mapping: "dict[str, str] | None" = None,
    rate_hz: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Convert a Torque-style export to a 1 Hz, clean-column DataFrame.

    Parameters
    ----------
    path : Path or str
        The raw Torque/Car-Scanner export CSV.
    mapping : dict[str, str] or None
        Optional explicit column override: {USEFUL_PID: source_column_name}.
        Bypasses the auto-heuristic for any PID listed here.  Load from a
        mapping.json file and pass here, or use the ``--mapping`` CLI flag.
    rate_hz : float
        Approximate recording rate used ONLY as a last-resort time fallback
        when no parseable time column is present.

    Returns (clean_df, report) where report records, per PID, which source
    column was used and how many raw readings it had (provenance for the thesis).
    """
    path = Path(path)
    raw = pd.read_csv(path)

    elapsed = _parse_elapsed(raw, rate_hz=rate_hz)
    if np.all(np.isnan(elapsed)):
        raise ValueError(
            f"Could not derive elapsed time from {path}. "
            "Check that the file has a 'time' column or pass --rate-hz."
        )

    report: dict = {
        "source": str(path),
        "n_raw_rows": int(len(raw)),
        "duration_s": float(np.nanmax(elapsed)),
        "pids": {},
    }

    # Resolve each PID to a sparse Series aligned to the raw row index.
    cols: dict[str, pd.Series] = {}
    for pid in USEFUL_PIDS:
        lo, hi = _VALID_RANGE[pid]
        s = _find_column(raw, pid, _CANDIDATE_PREFIXES[pid], lo, hi, mapping=mapping)
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
    ap.add_argument(
        "--mapping",
        default=None,
        metavar="JSON",
        help=(
            "Path to a JSON file with explicit column overrides: "
            '{"VEHICLE_SPEED": "Vehicle Speed (km/h)", ...}. '
            "Use this to pin the correct column when the auto-heuristic picks wrong."
        ),
    )
    ap.add_argument(
        "--rate-hz",
        type=float,
        default=1.0,
        metavar="N",
        help=(
            "Recording rate in Hz (default: 1.0). Used ONLY as a last-resort time "
            "fallback when the time column cannot be parsed by any format."
        ),
    )
    args = ap.parse_args()

    mapping: "dict[str, str] | None" = None
    if args.mapping:
        mapping_path = Path(args.mapping)
        if not mapping_path.exists():
            log.error("Mapping file not found: %s", mapping_path)
            return 1
        with open(mapping_path) as f:
            mapping = json.load(f)
        log.info("Loaded column mapping from %s: %d overrides", mapping_path, len(mapping))

    clean, report = adapt_torque_csv(args.csv, mapping=mapping, rate_hz=args.rate_hz)

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
