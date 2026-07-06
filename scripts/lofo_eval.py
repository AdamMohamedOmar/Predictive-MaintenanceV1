"""Leave-One-Family-Out (LOFO) cross-validation for the XGBoost classifier.

Why LOFO instead of a fixed holdout or plain LOSO
-------------------------------------------------
carOBD sessions cluster into 5 families by recording context:
    drive (highway), live (work->home commute), long (long trips),
    idle (parked), ufpe (low-speed campus).
Within a family, sessions are near-duplicate trips. So:
  * A fixed 2-session holdout (drive1, live12) tests on sessions whose siblings
    are in training -> inflated, and at 129 sessions it's only ~1.2% of the data.
  * Even plain LOSO leaks: hold out live12 and its ~35 commute siblings remain in
    training and teach the model the answer.
LOFO holds out an ENTIRE family, so the model is tested on a driving context it
never saw. That is the honest generalization claim for a cross-vehicle /
cross-context predictive-maintenance system.

Run (from repo root, after build_dataset has produced dataset_v1.parquet):
    python scripts/lofo_eval.py

Writes results/lofo_eval.json and prints a per-family table.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR
from src.features.dataset_builder import load_dataset
from src.models import xgb_classifier

logging.basicConfig(level=logging.WARNING)


def family_of(session_id: str) -> str:
    """drive1 -> drive, live12 -> live, ufpe3 -> ufpe (strip trailing digits)."""
    return re.sub(r"\d+$", "", str(session_id))


def main() -> int:
    df = load_dataset().copy()
    df["family"] = df["session_id"].map(family_of)
    families = sorted(df["family"].unique())
    print(f"Families found: {families}")
    print(f"Total windows: {len(df):,} | sessions: {df['session_id'].nunique()}\n")

    per_family: dict[str, dict] = {}
    for held in families:
        train_df = df[df["family"] != held]
        test_df = df[df["family"] == held]

        n_classes_test = test_df["label_id"].nunique()
        if n_classes_test < 2:
            print(f"  hold out {held:6s}: SKIPPED (only {n_classes_test} class in test)")
            continue

        clf, norm = xgb_classifier.train(train_df)
        res = xgb_classifier.evaluate(clf, norm, test_df)
        per_family[held] = {
            "macro_f1": res["macro_f1"],
            "n_test_windows": int(len(test_df)),
            "n_test_sessions": int(test_df["session_id"].nunique()),
            "per_class_f1": {k: round(v["f1"], 3) for k, v in res["per_class"].items()},
        }
        print(
            f"  hold out {held:6s}: macro-F1={res['macro_f1']:.3f}  "
            f"({len(test_df):,} windows, {test_df['session_id'].nunique()} sessions)"
        )

    if not per_family:
        print("No evaluable folds.")
        return 1

    f1s = [v["macro_f1"] for v in per_family.values()]
    worst = min(per_family, key=lambda k: per_family[k]["macro_f1"])
    summary = {
        "metric": "leave_one_family_out",
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "min_macro_f1": float(np.min(f1s)),
        "worst_family": worst,
        "per_family": per_family,
    }
    print(
        f"\nLOFO mean macro-F1: {summary['mean_macro_f1']:.3f} "
        f"(std {summary['std_macro_f1']:.3f}) | "
        f"worst: {summary['min_macro_f1']:.3f} on '{worst}'"
    )
    print("Report the WORST fold, not the mean — it is the honest floor.")

    out = Path(RESULTS_DIR) / "lofo_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())