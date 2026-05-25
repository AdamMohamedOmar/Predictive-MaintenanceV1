"""Test the REAL Skoda deployment path: refit normaliser on Skoda healthy, then predict.

This is what the Skoda workflow looks like:
  1. Driver runs the car for 3-5 min healthy → we collect feature rows
  2. Refit the BaselineNormalizer on those healthy rows
  3. New driving rows are transformed with the refitted normaliser
  4. Classifier (trained on Etios) makes predictions

Question: does the classifier's reliance on RAW absolute features break this?
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, classification_report

from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.extractor import feature_names
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, ALL_LABELS

ds = load_dataset()
train_df, test_df = session_split(ds)

with open("models/xgb_classifier_v1.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
_FEAT_COLS = normalised_feature_names()
_RAW_COLS = feature_names()

def _shift_skoda(df):
    """Apply realistic Skoda baseline shifts to a feature dataframe."""
    out = df.copy()
    for col in _RAW_COLS:
        if "LONG_TERM_FUEL_TRIM" in col and "delta" not in col:
            out[col] = out[col] + 5.0
        elif "INTAKE_MANIFOLD_PRESSURE" in col and "delta" not in col:
            out[col] = out[col] + 3.0
        elif col.startswith("THROTTLE__") and "delta" not in col:
            out[col] = out[col] + 2.0
    return out

# Simulate Skoda: shift test_df features, refit normalizer on shifted healthy
test_skoda = _shift_skoda(test_df)
train_skoda = _shift_skoda(train_df)

# Scenario A: REFIT normalizer on the Skoda-shifted healthy training data
# (mimics the actual deployment workflow)
healthy_skoda = train_skoda[train_skoda["label"] == "healthy"]
norm_refitted = BaselineNormalizer().fit(train_skoda)

X_skoda = norm_refitted.transform(test_skoda)[_FEAT_COLS].to_numpy(dtype=float)
y_test = test_df["label"].map({l: i for i, l in enumerate(ALL_LABELS)}).to_numpy()
pred = clf.predict(X_skoda)
print("Skoda deployment with REFITTED normaliser (proper workflow):")
print(f"  Macro-F1 = {f1_score(y_test, pred, average='macro'):.4f}")
print(classification_report(y_test, pred, target_names=ALL_LABELS, zero_division=0))

# Scenario B: KEEP Etios normalizer (a deployment mistake)
norm_etios = bundle["normalizer"]
X_skoda_etios = norm_etios.transform(test_skoda)[_FEAT_COLS].to_numpy(dtype=float)
pred_etios = clf.predict(X_skoda_etios)
print(f"\nSkoda deployment with ETIOS normaliser (mistake): macro-F1 = {f1_score(y_test, pred_etios, average='macro'):.4f}")
