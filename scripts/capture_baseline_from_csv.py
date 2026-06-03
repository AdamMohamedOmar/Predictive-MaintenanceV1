"""Fit a per-vehicle BaselineNormalizer from an already-adapted 1 Hz CSV.

Use this after running adapt_torque_csv.py on a healthy-drive recording.
The normalizer is required for cross-vehicle scoring: it re-centres the
XGBoost classifier's features on THIS vehicle's own healthy distribution.

Usage
-----
    python -m scripts.capture_baseline_from_csv \\
        --csv data/real_faults/ahmed/ahmed_drive_20260602.csv \\
        --vehicle "Toyota Corolla 2012"

    python -m scripts.capture_baseline_from_csv \\
        --csv data/real_faults/skoda/skoda_baseline_20260610.csv \\
        --vehicle "Skoda Roomster 2007 1.6" \\
        --out models/skoda_normalizer.pkl

Notes
-----
* The input CSV must be in clean-column 1 Hz format (as produced by
  adapt_torque_csv.py) — column names are the 14 USEFUL_PIDS.
* The same three guard checks as live capture apply:
    - Coolant must reach >= 75 degrees C at some point.
    - Mean vehicle speed must be >= 15 km/h.
    - At least 20 valid windows must be produced.
  An idle/cold or too-short session will raise ValueError with a clear message.
* The output .pkl suffix matches the pattern the dashboard's normalizer picker
  expects: models/<anything>_normalizer.pkl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.config import MODELS_DIR, USEFUL_PIDS
from scripts.live_baseline_capture import process_captured_rows


def capture_baseline_from_csv(
    csv_path: Path | str,
    vehicle_name: str = "vehicle",
    out_path: Path | str | None = None,
) -> Path:
    """Fit and save a BaselineNormalizer from a clean-column 1 Hz CSV.

    Parameters
    ----------
    csv_path : Path or str
        Path to a clean-column, 1 Hz OBD-II CSV (output of adapt_torque_csv).
    vehicle_name : str
        Free-text label stored in the sidecar JSON.
    out_path : Path or str or None
        Where to write the .pkl. Defaults to
        ``models/<vehicle_slug>_normalizer.pkl``.

    Returns
    -------
    Path
        The path to the saved normalizer .pkl.

    Raises
    ------
    ValueError
        If any of the three guard conditions fail (coolant, speed, windows).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Only pass columns in USEFUL_PIDS — extras (row-index artifacts, etc.)
    # would confuse the guard checks in process_captured_rows.
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    rows: list[dict] = df[pid_cols].to_dict(orient="records")

    norm, meta = process_captured_rows(
        rows,
        vehicle_name=vehicle_name,
        supported_pids=pid_cols,
        poll_hz=1.0,  # adapted CSVs are always 1 Hz after resampling
    )

    if out_path is None:
        slug = vehicle_name.lower().replace(" ", "_")
        out_path = MODELS_DIR / f"{slug}_normalizer.pkl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm.save(out_path)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a per-vehicle BaselineNormalizer from an adapted 1 Hz OBD-II CSV. "
            "Run adapt_torque_csv.py first to convert a raw Torque/Car-Scanner export."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the clean-column 1 Hz OBD-II CSV (from adapt_torque_csv).",
    )
    parser.add_argument(
        "--vehicle",
        default="vehicle",
        help='Free-text vehicle label, e.g. "Skoda Roomster 2007 1.4". Default: "vehicle".',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pkl path. Default: models/<vehicle_slug>_normalizer.pkl",
    )
    args = parser.parse_args()

    try:
        out = capture_baseline_from_csv(
            csv_path=args.csv,
            vehicle_name=args.vehicle,
            out_path=args.out,
        )
    except ValueError as exc:
        print(f"[FAIL] Guard check failed:\n  {exc}", file=sys.stderr)
        print(
            "\n  The baseline recording must be a real drive (warm engine, "
            "varied speed/throttle, some road km). An idle-only or cold-start "
            "session will not pass the guards.",
            file=sys.stderr,
        )
        return 1

    print(f"[OK] Normalizer saved : {out}")
    print(f"     Metadata         : {out.with_suffix('.json')}")
    print(
        f"\n  Next steps:\n"
        f"  1. Score a recording:\n"
        f"       python -m scripts.eval_real_fault <csv> --normalizer {out}\n"
        f"  2. Or launch the dashboard and select this normalizer in the sidebar."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
