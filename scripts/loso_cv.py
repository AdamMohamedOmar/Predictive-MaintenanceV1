"""Leave-one-session-out cross-validation for the XGBoost classifier.

Trains 9 models, each holding out exactly one session.  Reports mean +/- std
macro-F1 instead of a single point estimate.  This is the honest version of
the headline 0.96 number — a single held-out test set gives no error bars.

Run with:
    python -m scripts.loso_cv

Output:
    results/loso_cv_results.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.dataset_builder import load_dataset
from src.models.xgb_classifier import train as train_clf, evaluate as eval_clf

ds = load_dataset()
sessions = sorted(ds["session_id"].unique())

print(f"Running leave-one-session-out CV over {len(sessions)} sessions…\n")

f1_scores: list[float] = []
for held in sessions:
    train_df = ds[ds["session_id"] != held]
    test_df  = ds[ds["session_id"] == held]

    clf, norm_trained = train_clf(train_df, n_estimators=300, random_seed=42)
    res = eval_clf(clf, norm_trained, test_df)
    f1 = res["macro_f1"]
    f1_scores.append(f1)
    print(f"  Held out {held}: F1 = {f1:.4f}  ({len(test_df)} test windows)")

mean_f1 = statistics.mean(f1_scores)
std_f1  = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
min_f1  = min(f1_scores)
max_f1  = max(f1_scores)

print(f"\n{'='*50}")
print(f"Mean F1:  {mean_f1:.4f}")
print(f"Std  F1:  {std_f1:.4f}")
print(f"Min  F1:  {min_f1:.4f}")
print(f"Max  F1:  {max_f1:.4f}")

out_path = REPO_ROOT / "results" / "loso_cv_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "per_session": dict(zip(sessions, f1_scores)),
    "mean_f1": mean_f1,
    "std_f1": std_f1,
    "min_f1": min_f1,
    "max_f1": max_f1,
}, indent=2))

print(f"\nSaved: {out_path}")
