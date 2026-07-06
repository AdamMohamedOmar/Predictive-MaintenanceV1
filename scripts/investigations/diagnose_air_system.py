"""Diagnose air_system false negatives (classified as healthy).

Loads the saved test split, finds air_system windows the XGBoost model
missed, and prints a feature comparison so we can see why they look healthy.

Usage
-----
    python -m scripts.investigations.diagnose_air_system
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from src.features.dataset_builder import load_dataset
from src.features.normalizer import normalised_feature_names
from src.models.classifier import session_split
from src.models.xgb_classifier import load_model

# ── Load data & model ─────────────────────────────────────────────────────────

print("Loading dataset and model…")
ds = load_dataset()
_, test_df = session_split(ds)
clf, norm = load_model()

feat_cols = normalised_feature_names()
test_norm = norm.transform(test_df)
X_test = test_norm[feat_cols].to_numpy(dtype=float)
y_true = test_df["label"].to_numpy()
y_pred_id = clf.predict(X_test)

from src.models.classifier import ALL_LABELS
y_pred = [ALL_LABELS[i] for i in y_pred_id]

test_df = test_df.copy()
test_df["predicted"] = y_pred

# ── Isolate air_system rows ───────────────────────────────────────────────────

air = test_df[test_df["label"] == "air_system"]
air_correct = air[air["predicted"] == "air_system"]
air_missed  = air[air["predicted"] != "air_system"]

print(f"\n{'='*60}")
print(f"Air-system test windows : {len(air)}")
print(f"  Correctly classified  : {len(air_correct)}  ({len(air_correct)/len(air):.1%})")
print(f"  Missed (false neg)    : {len(air_missed)}  ({len(air_missed)/len(air):.1%})")
print(f"\nMissed -> predicted as:")
print(air_missed["predicted"].value_counts().to_string())

# ── Regime breakdown ──────────────────────────────────────────────────────────

regime_cols = [c for c in test_df.columns if c.startswith("REGIME__")]
if regime_cols:
    print(f"\n{'='*60}")
    print("Regime breakdown — MISSED windows:")
    for rc in regime_cols:
        n = int(air_missed[rc].sum())
        if n:
            print(f"  {rc:<30} {n:>4}  ({n/len(air_missed):.1%})")
    print("\nRegime breakdown — CORRECT windows:")
    for rc in regime_cols:
        n = int(air_correct[rc].sum())
        if n:
            print(f"  {rc:<30} {n:>4}  ({n/len(air_correct):.1%})")

# ── Feature comparison: key air_system signals ────────────────────────────────

key_feats = [
    "INTAKE_MANIFOLD_PRESSURE__mean",
    "INTAKE_MANIFOLD_PRESSURE__delta",
    "SHORT_TERM_FUEL_TRIM_BANK_1__mean",
    "SHORT_TERM_FUEL_TRIM_BANK_1__max",
    "LONG_TERM_FUEL_TRIM_BANK_1__mean",
    "LONG_TERM_FUEL_TRIM_BANK_1__delta",
    "ENGINE_LOAD__mean",
    "MAP_PER_THROTTLE",
    "FUEL_TRIM_DIVERGENCE",
]

avail = [f for f in key_feats if f in test_df.columns]
print(f"\n{'='*60}")
print(f"{'Feature':<45}  {'MISSED mean':>12}  {'CORRECT mean':>13}  {'HEALTHY mean':>13}")
print("-" * 87)

healthy_test = test_df[test_df["label"] == "healthy"]
for feat in avail:
    m_missed   = air_missed[feat].mean()
    m_correct  = air_correct[feat].mean()
    m_healthy  = healthy_test[feat].mean()
    print(f"{feat:<45}  {m_missed:>12.3f}  {m_correct:>13.3f}  {m_healthy:>13.3f}")

# ── Distribution of MAP and STFT in missed vs correct ────────────────────────

print(f"\n{'='*60}")
print("STFT mean distribution (key discriminator):")
for label, grp in [("Missed", air_missed), ("Correct", air_correct)]:
    vals = grp["SHORT_TERM_FUEL_TRIM_BANK_1__mean"]
    print(f"  {label:8s}  min={vals.min():.2f}  p25={vals.quantile(0.25):.2f}  "
          f"median={vals.median():.2f}  p75={vals.quantile(0.75):.2f}  max={vals.max():.2f}")

print(f"\nLTFT mean distribution:")
for label, grp in [("Missed", air_missed), ("Correct", air_correct)]:
    vals = grp["LONG_TERM_FUEL_TRIM_BANK_1__mean"]
    print(f"  {label:8s}  min={vals.min():.2f}  p25={vals.quantile(0.25):.2f}  "
          f"median={vals.median():.2f}  p75={vals.quantile(0.75):.2f}  max={vals.max():.2f}")

print(f"\nMAP mean distribution:")
for label, grp in [("Missed", air_missed), ("Correct", air_correct)]:
    vals = grp["INTAKE_MANIFOLD_PRESSURE__mean"]
    print(f"  {label:8s}  min={vals.min():.2f}  p25={vals.quantile(0.25):.2f}  "
          f"median={vals.median():.2f}  p75={vals.quantile(0.75):.2f}  max={vals.max():.2f}")

print(f"\n{'='*60}")
print("Done.")
