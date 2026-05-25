"""Simulate a Skoda baseline shift to test cross-vehicle generalisation.

Skoda Roomster 2007 likely has different baseline calibrations than Etios 2014:
  - Slightly higher idle throttle reading (different ECU calibration)
  - Different baseline LTFT (factory mixture-richness setting)
  - Different ECT sensor curve (slightly different operating temp)
  - Different barometric pressure if at different altitude

We simulate this by adding plausible offsets to the test set and re-classifying."""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score

from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, ALL_LABELS

ds = load_dataset()
train_df, test_df = session_split(ds)

# Load saved classifier
with open("models/xgb_classifier_v1.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
norm_loaded = bundle["normalizer"]
_FEAT_COLS = normalised_feature_names()

def evaluate(test_df, label):
    X = norm_loaded.transform(test_df)[_FEAT_COLS].to_numpy(dtype=float)
    y = test_df["label"].map({l: i for i, l in enumerate(ALL_LABELS)}).to_numpy()
    pred = clf.predict(X)
    f1 = f1_score(y, pred, average="macro")
    print(f"  {label}: macro-F1 = {f1:.4f}")
    return f1

print("Cross-vehicle baseline-shift simulation")
print("=" * 60)
evaluate(test_df, "Etios test (no shift)         ")

# Shift 1: +5% LTFT bias (Skoda runs slightly leaner from factory)
shifted = test_df.copy()
for col in [c for c in shifted.columns if "LONG_TERM_FUEL_TRIM" in c and "__" in c and "delta" not in c]:
    shifted[col] = shifted[col] + 5.0
evaluate(shifted, "Skoda-LTFT +5% bias           ")

# Shift 2: +3 kPa MAP (altitude difference)
shifted = test_df.copy()
for col in [c for c in shifted.columns if "INTAKE_MANIFOLD_PRESSURE" in c and "delta" not in c]:
    shifted[col] = shifted[col] + 3.0
evaluate(shifted, "Skoda-MAP +3 kPa (altitude)   ")

# Shift 3: Different idle throttle calibration
shifted = test_df.copy()
for col in [c for c in shifted.columns if "THROTTLE__" in c and "delta" not in c and "__z" not in c]:
    shifted[col] = shifted[col] + 2.0   # +2% throttle offset
evaluate(shifted, "Skoda-THROTTLE +2% (idle cal) ")

# Shift 4: ALL combined (worst-case Skoda mismatch)
shifted = test_df.copy()
for col in [c for c in shifted.columns if "LONG_TERM_FUEL_TRIM" in c and "delta" not in c]:
    shifted[col] = shifted[col] + 5.0
for col in [c for c in shifted.columns if "INTAKE_MANIFOLD_PRESSURE" in c and "delta" not in c]:
    shifted[col] = shifted[col] + 3.0
for col in [c for c in shifted.columns if "THROTTLE__" in c and "delta" not in c and "__z" not in c]:
    shifted[col] = shifted[col] + 2.0
evaluate(shifted, "Skoda-COMBINED (worst case)   ")
