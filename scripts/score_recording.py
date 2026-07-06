"""One-command Skoda recording pipeline.

Chains three steps into a single command:
  1. Adapt -- convert a raw Torque/Car-Scanner export to a clean 1 Hz CSV.
  2. Baseline -- (re-)build the per-vehicle normalizer from the ADAPTED CSV,
                 OR use an existing normalizer via --normalizer.
  3. Score  -- run the inference harness and print a human-readable verdict.

Usage
-----
Full pipeline (adapt + build baseline + score in one go):
    python -m scripts.score_recording raw_export.csv \\
        --vehicle "Skoda Roomster 2007 1.6" \\
        --out-dir results/skoda_run1

Score against a PREVIOUSLY built normalizer (skip baseline rebuild):
    python -m scripts.score_recording raw_export.csv \\
        --normalizer models/skoda_normalizer.pkl \\
        --out-dir results/skoda_run2

Score a PRE-ADAPTED clean-column CSV (skip adaptation step):
    python -m scripts.score_recording adapted.csv \\
        --pre-adapted \\
        --normalizer models/skoda_normalizer.pkl

Field procedure for the Skoda day
----------------------------------
Session 1 (baseline drive -- 5-8 min, warm engine, real road km):
    python -m scripts.score_recording session1_baseline.csv \\
        --vehicle "Skoda Roomster 2007 1.6" \\
        --out-dir results/skoda_baseline

Session 2+ (validation / fault recording):
    python -m scripts.score_recording session2_fault.csv \\
        --normalizer results/skoda_baseline/normalizer.pkl \\
        --out-dir results/skoda_fault_run1

The script prints:
  - Metering-type verdict (MAF vs speed-density)
  - Per-label window counts and fraction
  - Anomaly score mean and peak
  - Top 3 SHAP features (if models are loaded)
  - Recall estimate over a marked fault interval (if --fault-from and --fault-to given)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)



def _adapt(raw_csv: Path, mapping_path: Path | None, rate_hz: float, out_dir: Path) -> Path:
    """Run adapt_torque_csv on raw_csv, write adapted CSV to out_dir."""
    from scripts.adapt_torque_csv import adapt_torque_csv

    mapping = None
    if mapping_path:
        with open(mapping_path) as f:
            mapping = json.load(f)

    log.info("Step 1/3  Adapting %s ...", raw_csv.name)
    clean, report = adapt_torque_csv(raw_csv, mapping=mapping, rate_hz=rate_hz)

    adapted = out_dir / f"{raw_csv.stem}_adapted.csv"
    clean.to_csv(adapted, index=False)
    report_path = out_dir / f"{raw_csv.stem}_adapt_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("  Adapted: %d rows -> %d 1 Hz rows  |  written: %s",
             report["n_raw_rows"], report["n_clean_rows_1hz"], adapted)
    return adapted


def _build_baseline(adapted: Path, vehicle: str, out_dir: Path) -> Path:
    """Fit a normalizer from an adapted CSV, save to out_dir/normalizer.pkl."""
    from scripts.capture_baseline_from_csv import capture_baseline_from_csv

    norm_path = out_dir / "normalizer.pkl"
    log.info("Step 2/3  Building %s baseline normalizer ...", vehicle)
    try:
        capture_baseline_from_csv(adapted, vehicle_name=vehicle, out_path=norm_path)
        log.info("  Normalizer saved: %s", norm_path)
    except ValueError as exc:
        log.error(
            "  [FAIL] Baseline guard failed: %s\n"
            "  The recording must be a REAL DRIVE (warm engine, mean speed > 15 km/h,\n"
            "  at least 4 min of data). Re-run on a proper baseline session, or\n"
            "  pass --normalizer to use an existing one.", exc
        )
        return None
    return norm_path


def _score(
    adapted: Path,
    norm_path: Path | None,
    out_dir: Path,
    fault_from: int | None,
    fault_to: int | None,
) -> dict:
    """Run the eval harness and return the result dict."""
    from src.dashboard.inference import InferenceEngine
    from src.eval.real_fault_eval import evaluate_real_fault

    log.info("Step 3/3  Scoring %s ...", adapted.name)
    engine_kwargs: dict = {}
    if norm_path:
        engine_kwargs["engine"] = InferenceEngine(normalizer_override=norm_path)
        log.info("  Using normalizer: %s", norm_path)

    result = evaluate_real_fault(adapted, **engine_kwargs)

    # Cross-vehicle PID availability: a fault whose PRIMARY PID is absent on this
    # car cannot be honestly scored (XGBoost still emits a class for NaN
    # features), so mark it Untested instead of trusting the number. Also record
    # PIDs that are missing entirely, since their features are NaN across ALL
    # classes and reduce confidence in every score.
    import pandas as pd
    from src.eval.pid_availability import available_pids, missing_pids, untested_faults

    _avail = available_pids(pd.read_csv(adapted))
    result["untested_faults"] = untested_faults(_avail)
    result["missing_pids"] = missing_pids(pd.read_csv(adapted))

    result_path = out_dir / f"{adapted.stem}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("  Full result written: %s", result_path)

    # Optional recall over the fault interval
    if fault_from is not None and fault_to is not None:
        fault_windows = [
            w for w in result["windows"]
            if fault_from <= w["elapsed_s"] <= fault_to
        ]
        detected = sum(1 for w in fault_windows if w["label"] != "healthy")
        recall = detected / len(fault_windows) if fault_windows else float("nan")
        result["fault_interval_recall"] = {
            "from_s": fault_from,
            "to_s": fault_to,
            "n_windows": len(fault_windows),
            "n_detected": detected,
            "recall": recall,
        }

    return result


def _print_verdict(result: dict) -> None:
    summary = result["summary"]
    print()
    print("=" * 60)
    print("  SCORING VERDICT")
    print("=" * 60)
    print(f"  Source  : {result['csv_path']}")
    print(f"  Windows : {result['n_windows']}  ({result['n_rows']} rows)")
    print()
    untested = result.get("untested_faults", {})
    missing = result.get("missing_pids", [])

    print("  Label distribution:")
    for label, count in sorted(summary["label_counts"].items(),
                               key=lambda x: -x[1]):
        pct = 100.0 * count / result["n_windows"] if result["n_windows"] else 0
        bar = "#" * int(pct / 5)
        tag = "  [UNTESTED - unreliable]" if label in untested else ""
        print(f"    {label:<35} {count:4d} windows  ({pct:5.1f}%)  {bar}{tag}")
    print()

    if missing:
        print(f"  Missing PIDs on this vehicle (all-NaN): {', '.join(missing)}")
        print("    -> every feature derived from them is NaN; ALL scores carry "
              "reduced confidence.")
        print()
    if untested:
        print("  UNTESTED faults (primary PID unavailable - score is NOT meaningful):")
        for fault, pids in untested.items():
            print(f"    {fault:<35} needs {', '.join(pids)}")
        print("    Windows assigned to an Untested class above are unreliable, not "
              "true detections.")
        print()

    anomaly_scores = [w["anomaly_score"] for w in result["windows"]]
    if anomaly_scores:
        print(f"  Anomaly score -- mean: {sum(anomaly_scores)/len(anomaly_scores):.3f}  "
              f"peak: {max(anomaly_scores):.3f}")
    print()

    # Fault interval recall if computed
    fi = result.get("fault_interval_recall")
    if fi:
        print(f"  Fault interval ({fi['from_s']}s -- {fi['to_s']}s):")
        print(f"    Windows in interval : {fi['n_windows']}")
        print(f"    Detected (non-healthy): {fi['n_detected']}")
        if isinstance(fi["recall"], float) and not (fi["recall"] != fi["recall"]):
            print(f"    Recall              : {fi['recall']:.2f}  "
                  f"({'PASS >=0.60' if fi['recall'] >= 0.60 else 'BELOW target 0.60'})")
        print()

    # Verdict computed ONLY over evaluable windows. Windows classified as an
    # Untested class (primary PID absent) are set aside, never counted as faults
    # -- otherwise a MAF car with no MAP reads as "FAULT DETECTED" on air_system.
    evaluable_counts = {k: v for k, v in summary["label_counts"].items() if k not in untested}
    untested_count = sum(v for k, v in summary["label_counts"].items() if k in untested)
    n_eval = sum(evaluable_counts.values())
    fault_eval = sum(c for lbl, c in evaluable_counts.items()
                     if lbl not in ("healthy", "cold_start"))

    if untested_count:
        print(f"  Untested (set aside, not counted as faults): {untested_count} windows")
        print()

    min_eval = max(10, int(0.2 * result["n_windows"]))
    if n_eval < min_eval:
        verdict = (f"INSUFFICIENT EVALUABLE DATA "
                   f"({n_eval}/{result['n_windows']} windows evaluable; rest Untested)")
    elif fault_eval / n_eval >= 0.40:
        verdict = "FAULT DETECTED (>= 40% of EVALUABLE windows)"
    elif (n_eval - fault_eval) / n_eval >= 0.70:
        verdict = "LOOKS HEALTHY (>= 70% of EVALUABLE windows)"
    else:
        verdict = "MIXED / AMBIGUOUS (of EVALUABLE windows)"
    print(f"  Verdict: {verdict}")
    print()
    print("=" * 60)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command Skoda recording pipeline: adapt -> baseline -> score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="Path to the raw Torque/Car-Scanner export (or adapted CSV if --pre-adapted).")
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output directory. Default: results/score_recording/<csv_stem>/",
    )
    parser.add_argument(
        "--vehicle", default="vehicle",
        help='Free-text vehicle label (used in baseline metadata). Default: "vehicle".',
    )
    parser.add_argument(
        "--normalizer", default=None, metavar="PKL",
        help=(
            "Path to an existing per-vehicle normalizer .pkl. When given, the baseline "
            "rebuild step is skipped. Use this for recordings AFTER the baseline drive."
        ),
    )
    parser.add_argument(
        "--pre-adapted", action="store_true",
        help="Skip the adapt step (csv is already a clean-column 1 Hz CSV).",
    )
    parser.add_argument(
        "--mapping", default=None, metavar="JSON",
        help="Column-mapping JSON for the adapter (passed through to adapt_torque_csv).",
    )
    parser.add_argument(
        "--rate-hz", type=float, default=1.0, metavar="N",
        help="Recording rate (Hz) used as a time-parse fallback. Default: 1.0.",
    )
    parser.add_argument(
        "--fault-from", type=int, default=None, metavar="S",
        help="Start second of the fault interval (for recall computation).",
    )
    parser.add_argument(
        "--fault-to", type=int, default=None, metavar="S",
        help="End second of the fault interval (for recall computation).",
    )
    args = parser.parse_args()

    raw_csv = Path(args.csv)
    if not raw_csv.exists():
        log.error("File not found: %s", raw_csv)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO / "results" / "score_recording" / raw_csv.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: adapt (unless already adapted)
    if args.pre_adapted:
        adapted = raw_csv
        log.info("Step 1/3  Skipped (--pre-adapted).")
    else:
        mapping_path = Path(args.mapping) if args.mapping else None
        adapted = _adapt(raw_csv, mapping_path, args.rate_hz, out_dir)

    # Step 2: baseline (unless a normalizer is supplied)
    norm_path: Path | None = None
    if args.normalizer:
        norm_path = Path(args.normalizer)
        if not norm_path.exists():
            log.error("Normalizer not found: %s", norm_path)
            return 1
        log.info("Step 2/3  Skipped (--normalizer provided: %s).", norm_path)
    else:
        norm_path = _build_baseline(adapted, args.vehicle, out_dir)
        if norm_path is None:
            return 1  # guard failed; error already printed

    # Step 3: score
    result = _score(
        adapted, norm_path, out_dir,
        fault_from=args.fault_from, fault_to=args.fault_to,
    )

    _print_verdict(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())