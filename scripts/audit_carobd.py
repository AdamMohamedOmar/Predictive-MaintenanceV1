"""
Audit carOBD CSVs against physical bounds for our signature PIDs.

Produces two pieces of evidence used elsewhere in the project:
  1. A printed per-file table of which files pass/fail bounds checks
  2. The list of files that pass all checks (the "usable" subset)

Run from the repo root:
    python scripts/audit_carobd.py

The output of this script is the basis for the USABLE_CAROBD_FILES set in
src/data_loading.py and the audit findings documented in docs/DATA_NOTES.md.

If the carOBD data is updated or replaced, re-run this script and update
USABLE_CAROBD_FILES accordingly.
"""

from pathlib import Path
import sys

import pandas as pd

# Make src/ importable when this script is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_CAROBD_DIR


# Physical bounds for each PID. A value outside these bounds means the column
# does NOT actually contain that PID — typically a firmware-encoding artefact.
# Bounds are deliberately generous; they're catching gross violations, not
# tight calibration errors.
BOUNDS = [
    # (raw column name, low, high, human-readable description)
    ("VEHICLE_SPEED ()", 0, 200, "vehicle speed (km/h)"),
    ("COOLANT_TEMPERATURE ()", -10, 130, "coolant temperature (°C)"),
    ("TIMING_ADVANCE ()", -64, 64, "timing advance (degrees)"),
    ("SHORT_TERM_FUEL_TRIM_BANK_1 ()", -100, 100, "short-term fuel trim (%)"),
]


def audit_file(path: Path) -> dict:
    """Return per-file pass/fail info for each bounded PID."""
    df = pd.read_csv(path)
    result = {"file": path.name, "n_rows": len(df)}
    for col, lo, hi, _ in BOUNDS:
        s = df[col].dropna()
        if len(s) == 0:
            result[col] = "EMPTY"
        elif s.min() >= lo and s.max() <= hi:
            result[col] = "OK"
        else:
            result[col] = f"OUT [{s.min():.1f}, {s.max():.1f}]"
    return result


def main() -> int:
    if not DATA_CAROBD_DIR.exists():
        print(f"!! carOBD data directory not found: {DATA_CAROBD_DIR}", file=sys.stderr)
        return 1

    csvs = sorted(DATA_CAROBD_DIR.glob("*.csv"))
    if not csvs:
        print(f"!! No CSVs found in {DATA_CAROBD_DIR}", file=sys.stderr)
        return 1

    print(f"Auditing {len(csvs)} files in {DATA_CAROBD_DIR}\n")

    rows = [audit_file(p) for p in csvs]
    df = pd.DataFrame(rows)

    # Build a per-file pattern: e.g. "OK|OK|BAD|BAD" for the 4 checks
    bound_cols = [b[0] for b in BOUNDS]
    df["pattern"] = df[bound_cols].apply(
        lambda r: "|".join("OK" if v == "OK" else "BAD" for v in r),
        axis=1,
    )

    print("Group sizes by pass/fail pattern:")
    print(df["pattern"].value_counts().to_string())
    print()

    print("Files passing all bounds checks (the usable subset):")
    usable = df[df["pattern"] == "OK|OK|OK|OK"]["file"].tolist()
    for f in usable:
        print(f"  {f}")
    print(f"\nTotal usable files: {len(usable)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
