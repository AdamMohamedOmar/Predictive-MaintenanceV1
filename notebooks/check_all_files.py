"""
Full PID audit across all carOBD CSVs.

Outputs a per-file, per-PID summary table that becomes evidence for the
charter amendment proposal regarding FUEL_AIR_COMMANDED_EQUIV_RATIO.

Run from the repo root:
    python notebooks/check_all_files.py
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "carOBD"

# Use raw column names (not the cleaned ones) — this script is the audit
# that justifies the loader's _UNUSABLE_PIDS list, so it must run before
# the loader does any filtering.
SIGNATURE_RAW = [
    "FUEL_AIR_COMMANDED_EQUIV_RATIO ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "COOLANT_TEMPERATURE ()",
    "TIMING_ADVANCE ()",
    "THROTTLE ()",
    "PEDAL_D ()",
]


def audit_file(path: Path) -> dict:
    df = pd.read_csv(path)
    result = {"file": path.name, "n_rows": len(df)}
    for col in SIGNATURE_RAW:
        if col not in df.columns:
            result[col] = {"n_unique": "MISSING"}
        else:
            s = df[col]
            result[col] = {
                "n_unique": int(s.nunique()),
                "min": float(s.min()),
                "max": float(s.max()),
                "std": float(s.std()) if s.nunique() > 1 else 0.0,
            }
    return result


def main():
    csvs = sorted(DATA_DIR.glob("*.csv"))
    print(f"Auditing {len(csvs)} files in {DATA_DIR}\n")

    all_results = [audit_file(p) for p in csvs]

    # One column at a time, for readability
    for col in SIGNATURE_RAW:
        print(f"\n=== {col} ===")
        rows = []
        for r in all_results:
            stats = r[col]
            if stats["n_unique"] == "MISSING":
                rows.append(
                    {
                        "file": r["file"],
                        "n_unique": "MISSING",
                        "min": "-",
                        "max": "-",
                        "std": "-",
                    }
                )
            else:
                rows.append(
                    {
                        "file": r["file"],
                        "n_unique": stats["n_unique"],
                        "min": f"{stats['min']:.3f}",
                        "max": f"{stats['max']:.3f}",
                        "std": f"{stats['std']:.3f}",
                    }
                )
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
