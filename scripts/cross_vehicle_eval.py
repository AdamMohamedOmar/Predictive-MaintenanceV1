"""Cross-vehicle paired-fault evaluation skeleton.

When the team eventually has BOTH an Etios fault recording AND a Skoda
recording of the same fault type, this script produces a paired
detection-rate report so the thesis can answer the cross-vehicle
generalisation question directly:

    given the same physical fault, does the pipeline trained on Etios
    data detect it on the Skoda at a comparable rate?

Until that day, the script gracefully reports "no data yet" — that's
the default path, since both `data/real_faults/skoda/` and
`data/real_faults/etios/` ship empty (`.gitkeep` only).

Usage
-----
    python -m scripts.cross_vehicle_eval \\
        --fault-type air_system \\
        [--etios-fault data/real_faults/etios/etios_vacuumleak_20260605_run1.csv] \\
        [--skoda-fault data/real_faults/skoda/skoda_vacuumleak_20260605_run1.csv] \\
        [--out results/cross_vehicle_eval/air_system_v1.json]

Behaviour
---------
* No CSV paths given: writes a stub JSON with `status: "no_data_yet"` on
  both vehicles. Exits 0.
* Only one CSV: runs the harness on that side, marks the other side as
  `"no_data_yet"`, and emits a single-vehicle JSON. Exits 0.
* Both CSVs: runs the harness on each side, computes per-vehicle fault
  fractions, and writes a paired JSON. Exits 0.

The script is intentionally permissive on the no-data path so it can be
scheduled as a cron / loop check that waits for data to land.

This script does NOT decide whether a "successful detection" has
occurred — that's a downstream interpretation against the recording's
sibling metadata (mods-in / mods-out). The harness emits per-window
predictions; this script aggregates them per-vehicle.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.eval.real_fault_eval import evaluate_real_fault

VALID_FAULT_TYPES = (
    "air_system",
    "fuel_system",
    "coolant_temp_sensor",
    "throttle_position_sensor",
)


def evaluate_one_vehicle(csv_path: Optional[Path]) -> dict:
    """Return either an evaluation dict or a `no_data_yet` stub."""
    if csv_path is None:
        return {"csv_path": None, "status": "no_data_yet"}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"csv_path": str(csv_path), "status": "path_not_found"}

    result = evaluate_real_fault(csv_path)
    return {
        "csv_path": str(csv_path),
        "status": "evaluated",
        "n_windows": result["n_windows"],
        "label_counts": result["summary"]["label_counts"],
        "fault_window_count": result["summary"]["fault_window_count"],
        "fault_fraction": result["summary"]["fault_fraction"],
    }


def cross_vehicle_report(
    fault_type: str,
    etios_csv: Optional[Path],
    skoda_csv: Optional[Path],
) -> dict:
    """Build the paired report dict.

    The note field always carries the honest framing — even when both
    sides evaluate cleanly, the recall interpretation is downstream.
    """
    if fault_type not in VALID_FAULT_TYPES:
        raise ValueError(
            f"Unknown fault_type {fault_type!r}. Valid: {list(VALID_FAULT_TYPES)}"
        )

    etios = evaluate_one_vehicle(etios_csv)
    skoda = evaluate_one_vehicle(skoda_csv)

    paired_delta: Optional[float]
    if etios["status"] == "evaluated" and skoda["status"] == "evaluated":
        paired_delta = float(skoda["fault_fraction"] - etios["fault_fraction"])
    else:
        paired_delta = None

    return {
        "fault_type": fault_type,
        "vehicles": {"etios": etios, "skoda": skoda},
        "paired_skoda_minus_etios_fault_fraction": paired_delta,
        "note": (
            "Per-vehicle fault_fraction is a coarse aggregate. The real "
            "detection-rate computation slices each recording into "
            "pre / fault / post intervals using the sibling metadata JSON "
            "(mods_in_place_from_s, mods_removed_at_s) and counts only "
            "windows inside the fault interval. See "
            "docs/REAL_FAULT_COLLECTION.md §10 for the headline metric "
            "definition."
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-vehicle paired-fault evaluation. Run with either or both "
            "vehicle CSVs; missing sides report `no_data_yet` without "
            "failing the run."
        )
    )
    parser.add_argument(
        "--fault-type", required=True, choices=VALID_FAULT_TYPES,
        help="Which fault class is being evaluated.",
    )
    parser.add_argument(
        "--etios-fault", default=None,
        help="Path to the Etios CSV (optional).",
    )
    parser.add_argument(
        "--skoda-fault", default=None,
        help="Path to the Skoda CSV (optional).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path. Defaults to "
             "results/cross_vehicle_eval/<fault_type>_v1.json.",
    )
    args = parser.parse_args(argv)

    etios_path = Path(args.etios_fault) if args.etios_fault else None
    skoda_path = Path(args.skoda_fault) if args.skoda_fault else None

    log.info("Cross-vehicle evaluation for fault_type=%s", args.fault_type)
    if etios_path is None and skoda_path is None:
        log.info("  (no CSV paths given — writing stub)")
    else:
        log.info("  etios: %s", etios_path or "—")
        log.info("  skoda: %s", skoda_path or "—")

    report = cross_vehicle_report(args.fault_type, etios_path, skoda_path)

    out_path = (
        Path(args.out)
        if args.out
        else _REPO / "results" / "cross_vehicle_eval" / f"{args.fault_type}_v1.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    for vehicle, v in report["vehicles"].items():
        if v["status"] == "evaluated":
            log.info(
                "  %s: %d windows · %.0f%% non-healthy",
                vehicle,
                v["n_windows"],
                v["fault_fraction"] * 100.0,
            )
        else:
            log.info("  %s: %s", vehicle, v["status"])

    if report["paired_skoda_minus_etios_fault_fraction"] is not None:
        log.info(
            "  paired Δ (Skoda − Etios fault fraction): %+.3f",
            report["paired_skoda_minus_etios_fault_fraction"],
        )

    log.info("  Written: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
