"""Probe classifier and forecaster for evidence of injection-signature memorisation."""
import json
import pandas as pd
import pickle
from pathlib import Path

from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split

ds = load_dataset()
train_df, test_df = session_split(ds)
norm = BaselineNormalizer().fit(train_df)
_FEAT_COLS = normalised_feature_names()

# Load saved XGBoost classifier
with open("models/xgb_classifier_v1.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]

print("=" * 70)
print("1. FEATURE IMPORTANCES — what is the classifier actually using?")
print("=" * 70)
imp = pd.DataFrame({"feature": _FEAT_COLS, "importance": clf.feature_importances_})
imp["is_z_score"] = imp["feature"].str.endswith("__z")
print(imp.nlargest(15, "importance").to_string(index=False))
print()
total_imp_raw = imp.loc[~imp["is_z_score"], "importance"].sum()
total_imp_z   = imp.loc[ imp["is_z_score"], "importance"].sum()
print(f"  * Cumulative importance — RAW features:    {total_imp_raw:.3f}  (vehicle-specific!)")
print(f"  * Cumulative importance — Z-SCORED features: {total_imp_z:.3f}  (vehicle-agnostic)")
print(f"  * RAW share: {total_imp_raw / (total_imp_raw + total_imp_z) * 100:.1f}%")
print("  * VERDICT: classifier learned Etios-specific thresholds; will not transfer cleanly to Skoda."
      if total_imp_raw > 0.3 else "  * OK")

print()
print("=" * 70)
print("2. CONFUSION MATRIX — what is it getting wrong?")
print("=" * 70)
from src.models.classifier import ALL_LABELS
X_test = norm.transform(test_df)[_FEAT_COLS].to_numpy(dtype=float)
y_test = test_df["label"].map({l: i for i, l in enumerate(ALL_LABELS)}).to_numpy()
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS))

print()
print("=" * 70)
print("3. CLASS SEPARABILITY — how separable is each fault on its #1 feature?")
print("=" * 70)
# For each fault class, show its top feature's distribution overlap with healthy
for fault in ["air_system", "fuel_system", "coolant_temp_sensor", "throttle_position_sensor"]:
    norm_df = norm.transform(train_df).reset_index(drop=True)
    train_reset = train_df.reset_index(drop=True)
    healthy_idx = train_reset["label"] == "healthy"
    fault_idx = train_reset["label"] == fault
    # Pick the feature with highest |mean difference| between healthy and fault
    h_mean = norm_df.loc[healthy_idx, _FEAT_COLS].mean()
    f_mean = norm_df.loc[fault_idx, _FEAT_COLS].mean()
    diff = (f_mean - h_mean).abs().sort_values(ascending=False)
    top_feat = diff.index[0]
    h_vals = norm_df.loc[healthy_idx, top_feat]
    f_vals = norm_df.loc[fault_idx, top_feat]
    overlap = max(0, min(h_vals.max(), f_vals.max()) - max(h_vals.min(), f_vals.min()))
    overlap_pct = overlap / max(h_vals.max() - h_vals.min(), 1e-6) * 100
    print(f"  {fault:30s} top feat: {top_feat:45s} "
          f"healthy={h_vals.mean():+.2f}±{h_vals.std():.2f}  "
          f"fault={f_vals.mean():+.2f}±{f_vals.std():.2f}  "
          f"range_overlap={overlap_pct:.0f}%")

print()
print("=" * 70)
print("4. POST-RAMP ONLY? — are classifier's correctly-labelled faults all easy ones?")
print("=" * 70)
# Look at correctly-predicted faults: are they tightly clustered (high severity = post-ramp easy)
# or evenly distributed?
meta_path = Path("data/synthetic/dataset_v1_meta.json")
if meta_path.exists():
    with open(meta_path) as f:
        print(json.dumps(json.load(f), indent=2))
