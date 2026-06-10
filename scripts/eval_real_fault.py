"""CLI: run the real-fault evaluation harness on one CSV and write JSON.

Usage
-----
    python -m scripts.eval_real_fault data/real_faults/mock/mock_lean_fault.csv

Or, against a Skoda recording once one lands:
    python -m scripts.eval_real_fault data/real_faults/skoda/skoda_vacuumleak_20260605_run1.csv

Output goes to results/real_fault_eval/<stem>_v1.json.

This script is the production entry point. The test in
``tests/test_real_fault_harness_plumbing.py`` exercises the underlying
function directly without writing files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.dashboard.inference import InferenceEngine
from src.eval.real_fault_eval import evaluate_real_fault


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the per-window inference harness on one CSV. "
            "Produces per-stride classifier predictions; does NOT claim "
            "to validate real-fault detection. See data/real_faults/README.md."
        )
    )
    parser.add_argument("csv", help="Path to the OBD-II CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to results/real_fault_eval/<stem>_v1.json.",
    )
    parser.add_argument(
        "--normalizer",
        default=None,
        metavar="PKL",
        help=(
            "Path to a per-vehicle normalizer .pkl (from capture_baseline_from_csv "
            "or live_baseline_capture). When given, the classifier is re-centred on "
            "THIS vehicle's healthy distribution instead of the Etios baseline. "
            "Required for a meaningful cross-vehicle score."
        ),
    )
    parser.add_argument(
        "--fault-from",
        type=int,
        default=None,
        metavar="S",
        help="Fault-interval start (seconds since recording start; mods-in mark).",
    )
    parser.add_argument(
        "--fault-to",
        type=int,
        default=None,
        metavar="S",
        help="Fault-interval end (seconds since recording start; mods-out mark).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 1

    # Build the engine here so the normalizer override can be passed cleanly.
    # evaluate_real_fault accepts engine= and skips its own construction when given.
    engine_kwargs: dict = {}
    if args.normalizer:
        norm_path = Path(args.normalizer)
        if not norm_path.exists():
            log.error("Normalizer not found: %s", norm_path)
            return 1
        engine_kwargs["engine"] = InferenceEngine(normalizer_override=norm_path)
        log.info("Using normalizer override: %s", norm_path)

    if (args.fault_from is None) != (args.fault_to is None):
        log.error("--fault-from and --fault-to must be given together.")
        return 1

    log.info("Evaluating %s …", csv_path)
    result = evaluate_real_fault(csv_path, **engine_kwargs)

    if args.fault_from is not None:
        from src.eval.real_fault_eval import compute_fault_recall

        recall_block = compute_fault_recall(
            result["windows"], args.fault_from, args.fault_to
        )
        result["fault_interval"] = {
            "from_s": args.fault_from,
            "to_s": args.fault_to,
            **recall_block,
        }

    out_path = (
        Path(args.out)
        if args.out
        else _REPO / "results" / "real_fault_eval" / f"{csv_path.stem}_v1.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = result["summary"]
    log.info(
        "  %d windows · %d non-healthy labels (%.0f%%)",
        result["n_windows"],
        summary["fault_window_count"],
        summary["fault_fraction"] * 100.0,
    )
    log.info("  Label counts: %s", summary["label_counts"])
    if "fault_interval" in result:
        fi = result["fault_interval"]
        log.info(
            "  §10 vacuum-leak recall: %.3f  (%d/%d windows; by-label %d, anomaly-only %d)",
            fi["recall"], fi["n_detected"], fi["n_fault_windows"],
            fi["detected_by_label"], fi["detected_by_anomaly_only"],
        )
        log.info("  Pass criterion: recall >= 0.60 (REAL_FAULT_COLLECTION.md §10)")
    log.info("  Written: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
