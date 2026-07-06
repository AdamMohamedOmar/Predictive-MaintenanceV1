"""
Audit carOBD CSVs against physical bounds for our signature PIDs.

Produces two pieces of evidence used elsewhere in the project:
  1. A printed per-file table of which files pass/fail bounds checks
  2. The list of files that pass all checks (the "usable" subset)

Run from the repo root:
    python scripts/audit_carobd.py

The output of this script is the basis for the audit findings documented in
docs/DATA_NOTES.md. The former USABLE_CAROBD_FILES whitelist was removed after
the trailing-comma parse fix (index_col=False): all 129 files now pass, and
src.data_loading.list_usable_files() validates files dynamically instead.

If the carOBD data is updated or replaced, re-run this script to confirm every
file still parses aligned and within physical bounds.
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
    df = pd.read_csv(path, index_col=False)
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


def audit_catalyst_temp(usable_files: list[Path]) -> dict:
    """Report CATALYST_TEMPERATURE variance across the usable files (P2-2).

    Cat temp is currently DROPPED from USEFUL_PIDS. If it carries real
    combustion/exhaust signal it could corroborate the coolant (rich-running)
    and a future O2 fault. This reports whether it varies usefully so the
    add-to-USEFUL_PIDS decision is evidence-based.
    """
    col = "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()"
    stats = {"pid": col, "per_file": {}, "varies_usefully": False}
    stds = []
    for p in usable_files:
        df = pd.read_csv(p, index_col=False)
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        std = float(s.std())
        stds.append(std)
        stats["per_file"][p.name] = {
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": std,
        }
    if stds:
        stats["median_std"] = float(pd.Series(stds).median())
        # "Varies usefully" = typical within-file σ above a few °C (not a flat
        # sentinel). 88 °C-class spread → clearly informative.
        stats["varies_usefully"] = stats["median_std"] > 5.0
    return stats


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

    # ── P2-2: catalyst-temperature variance audit ───────────────────────────
    usable_paths = [DATA_CAROBD_DIR / f for f in usable]
    cat = audit_catalyst_temp(usable_paths)
    print("\nCATALYST_TEMPERATURE_BANK1_SENSOR1 variance (P2-2 audit):")
    if cat.get("median_std") is not None:
        print(f"  median within-file σ = {cat['median_std']:.1f} °C")
        print(f"  varies usefully: {cat['varies_usefully']}")
        print(
            "  NOTE: it varies, so it likely carries combustion/exhaust signal. "
            "Adding it to USEFUL_PIDS is DEFERRED — it would change the 83-feature "
            "contract every model/normalizer/test depends on, so it belongs in a "
            "dedicated feature-expansion change, not bundled with the physics fixes."
        )
    else:
        print("  column absent or empty in usable files")

    return 0


if __name__ == "__main__":
    sys.exit(main())