"""Inspect a raw or adapted OBD-II recording before a Skoda test drive.

Run this on a 30-second test recording (engine warm, idling) to confirm:
  1. Metering type  -- is this a MAF car or a speed-density (MAP) car?
  2. PID coverage   -- which of the 14 USEFUL_PIDS are present and how dense?
  3. Drive character -- what fraction of the session is driving vs. idling?

This information shapes what you can expect from the model:
  - Speed-density cars (MAP): vacuum-leak shows as raised idle MAP + trim.
    The Etios training data is speed-density; the injected air_system fault
    mirrors this signature.
  - MAF cars: MAP reads barometric (~101 kPa) regardless of load; a vacuum
    leak downstream of the MAF shows as large positive fuel trims (lean).
    The model may label this fuel_system or air_system -- both are "detected".
    The cross-architecture mismatch is a known limitation (see CHARTER R12).

Usage
-----
    python -m scripts.inspect_recording <path_to_csv>

Works on both:
  - raw Torque/Car-Scanner exports (wide, sparse, app column names)
  - clean-column adapted CSVs (14-PID 1 Hz format from adapt_torque_csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.config import USEFUL_PIDS

# MAP is considered "pinned barometric" (MAF engine) if its std/mean ratio is
# below this fraction (tight cluster around ~101 kPa).
_MAP_VARIANCE_THRESHOLD = 0.05  # 5 % coefficient of variation

# A vehicle is "moving" when speed > this threshold (km/h).
_MOVING_SPEED_KMH = 5.0

# MAP centroid for a naturally-aspirated engine at idle vacuum vs. barometric.
_BARO_KPA_APPROX = 101.0


def _detect_metering_type(df: pd.DataFrame) -> tuple[str, str]:
    """Return (verdict, explanation) based on MAP column behaviour.

    Returns one of: 'MAF-based', 'speed-density (MAP)', 'unknown'.
    """
    # Check the raw export for an explicit MAF column
    maf_col = next(
        (c for c in df.columns if "maf" in c.lower() and "g" in c.lower()), None
    )
    if maf_col is not None:
        series = pd.to_numeric(df[maf_col], errors="coerce").dropna()
        if series.notna().sum() > 0 and series.mean() > 0.5:
            return (
                "MAF-based",
                f"MAF column '{maf_col}' present with {series.notna().sum()} readings "
                f"(mean {series.mean():.1f} g/s).",
            )

    # Fall back to MAP constancy test
    map_col = next(
        (c for c in df.columns if "map" in c.lower() or "manifold" in c.lower()
         or c == "INTAKE_MANIFOLD_PRESSURE"),
        None,
    )
    if map_col is None:
        return "unknown", "No MAP column found in the recording."

    series = pd.to_numeric(df[map_col], errors="coerce").dropna()
    if series.empty:
        return "unknown", f"MAP column '{map_col}' has no valid readings."

    mean_map = series.mean()
    cv = series.std() / (mean_map + 1e-6)  # coefficient of variation

    if cv < _MAP_VARIANCE_THRESHOLD:
        # MAP is nearly constant -- typical of a MAF engine where MAP = baro
        if mean_map > 90.0:
            return (
                "MAF-based (likely)",
                f"MAP column '{map_col}' is near-constant at "
                f"{mean_map:.1f} kPa (CV={cv:.3f}) -- barometric pressure, "
                f"not intake vacuum. MAP carries no idle-vacuum signal here.",
            )
        else:
            return (
                "unknown",
                f"MAP column '{map_col}' is near-constant at "
                f"{mean_map:.1f} kPa but below barometric -- unusual.",
            )
    else:
        return (
            "speed-density (MAP) (likely)",
            f"MAP column '{map_col}' varies (mean={mean_map:.1f} kPa, "
            f"CV={cv:.3f}) -- typical of a speed-density engine where MAP "
            f"tracks intake vacuum.",
        )


def _pid_fill_rates(df: pd.DataFrame) -> dict[str, dict]:
    """Return per-PID fill rate for USEFUL_PIDS only, from a clean-column CSV."""
    out = {}
    for pid in USEFUL_PIDS:
        if pid not in df.columns:
            out[pid] = {"present": False, "fill_pct": 0.0, "mean": None, "std": None}
        else:
            s = pd.to_numeric(df[pid], errors="coerce")
            n_valid = int(s.notna().sum())
            n_total = len(s)
            out[pid] = {
                "present": True,
                "fill_pct": 100.0 * n_valid / n_total if n_total else 0.0,
                "mean": float(s.mean()) if n_valid else None,
                "std": float(s.std()) if n_valid > 1 else None,
            }
    return out


def _drive_fraction(df: pd.DataFrame) -> tuple[float, str]:
    """Return fraction of rows where vehicle is moving (VEHICLE_SPEED > threshold)."""
    speed_col = next(
        (c for c in df.columns
         if c == "VEHICLE_SPEED" or "vehicle speed" in c.lower()),
        None,
    )
    if speed_col is None:
        return float("nan"), "VEHICLE_SPEED not found"

    s = pd.to_numeric(df[speed_col], errors="coerce").dropna()
    if s.empty:
        return 0.0, "no valid speed readings"

    moving = (s > _MOVING_SPEED_KMH).mean()
    return float(moving), speed_col


def inspect_recording(csv_path: Path | str) -> dict:
    """Analyse a recording and return an inspection report dict.

    Parameters
    ----------
    csv_path : Path or str
        Path to either a raw Torque/Car-Scanner export or a clean-column CSV.

    Returns
    -------
    dict with keys: source, n_rows, duration_s, metering_type, metering_detail,
    drive_fraction, pid_coverage, missing_pids, warnings.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    n_rows = len(df)

    # Estimate duration: prefer a time column, else assume 1 Hz.
    duration_s: float = float(n_rows)
    for time_col in ("time", "Time", "timestamp", "Timestamp"):
        if time_col in df.columns:
            t = pd.to_datetime(df[time_col], errors="coerce").dropna()
            if len(t) >= 2:
                duration_s = float((t.iloc[-1] - t.iloc[0]).total_seconds())
            break

    metering_type, metering_detail = _detect_metering_type(df)
    drive_frac, speed_src = _drive_fraction(df)

    # PID coverage (on a clean-column CSV this is definitive; on a raw export
    # it will show 0 fill for all because column names don't match USEFUL_PIDS)
    is_clean_column = all(p in df.columns for p in USEFUL_PIDS[:4])
    pid_info = _pid_fill_rates(df) if is_clean_column else {}

    missing_pids = [p for p in USEFUL_PIDS if p not in df.columns] if is_clean_column else []
    required_pids_for_model = [
        "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
        "COOLANT_TEMPERATURE", "LONG_TERM_FUEL_TRIM_BANK_1",
        "SHORT_TERM_FUEL_TRIM_BANK_1", "INTAKE_MANIFOLD_PRESSURE",
    ]
    missing_critical = [p for p in required_pids_for_model if p in missing_pids]

    warnings: list[str] = []
    if "MAF" in metering_type:
        warnings.append(
            "MAF ENGINE DETECTED: MAP will read ~barometric (no idle vacuum). "
            "The air_system injector was trained on a speed-density (MAP) engine. "
            "A real vacuum leak on this car will most likely appear as positive "
            "LTFT/STFT (lean condition) -- the model may label it fuel_system "
            "or air_system. Both are acceptable detection per REAL_FAULT_COLLECTION.md S8."
        )
    if missing_critical:
        warnings.append(
            f"MISSING CRITICAL PIDs: {missing_critical}. "
            "Change your app's PID selection to include these before the drive."
        )
    if not np.isnan(drive_frac) and drive_frac < 0.05:
        warnings.append(
            "MOSTLY IDLE: drive fraction is < 5 %. "
            "The baseline drive must include real road driving (mean speed > 15 km/h)."
        )

    return {
        "source": str(csv_path),
        "n_rows": n_rows,
        "duration_s": duration_s,
        "is_clean_column_format": is_clean_column,
        "metering_type": metering_type,
        "metering_detail": metering_detail,
        "drive_fraction": drive_frac,
        "speed_source_column": speed_src,
        "pid_coverage": pid_info,
        "missing_pids": missing_pids,
        "missing_critical_pids": missing_critical,
        "warnings": warnings,
    }


def _print_report(report: dict) -> None:
    print()
    print("=" * 60)
    print("  RECORDING INSPECTION REPORT")
    print("=" * 60)
    print(f"  Source    : {report['source']}")
    print(f"  Rows      : {report['n_rows']}  |  Duration: {report['duration_s']:.0f} s")
    print(f"  Format    : {'clean-column (14 PID)' if report['is_clean_column_format'] else 'raw app export'}")
    print()
    print(f"  Metering type : {report['metering_type']}")
    print(f"  Detail        : {report['metering_detail']}")
    print()
    drive_pct = report['drive_fraction']
    if not isinstance(drive_pct, float) or not np.isnan(drive_pct):
        print(f"  Drive fraction: {drive_pct * 100:.1f}%  (speed > 5 km/h)")
    print()

    if report["is_clean_column_format"]:
        print("  PID coverage (clean-column format):")
        for pid, info in report["pid_coverage"].items():
            if info["present"]:
                mean_str = f"{info['mean']:.2f}" if info["mean"] is not None else "n/a"
                std_str  = f"{info['std']:.2f}" if info["std"]  is not None else "n/a"
                print(f"    {pid:<40} fill={info['fill_pct']:5.1f}%  mean={mean_str}  std={std_str}")
            else:
                print(f"    {pid:<40} MISSING")
        if report["missing_pids"]:
            print(f"\n  Missing PIDs: {report['missing_pids']}")
        if report["missing_critical_pids"]:
            print(f"\n  *** CRITICAL PIDs missing: {report['missing_critical_pids']} ***")
    else:
        print("  (Raw export -- run adapt_torque_csv.py first for detailed PID coverage)")

    if report["warnings"]:
        print()
        print("  WARNINGS:")
        for w in report["warnings"]:
            print(f"  ! {w}")
    else:
        print()
        print("  No warnings.")

    print()
    print("=" * 60)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an OBD-II recording: metering type, PID coverage, "
            "and drive character. Run before a Skoda test drive."
        )
    )
    parser.add_argument("csv", help="Path to the OBD-II CSV (raw or clean-column).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[FAIL] File not found: {csv_path}", file=sys.stderr)
        return 1

    report = inspect_recording(csv_path)
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
