"""Test if classifier detects EARLY-RAMP faults (which it never trained on)."""
from pathlib import Path
import pandas as pd
import pickle

from src.config import DATA_CAROBD_DIR, RANDOM_SEED
from src.data_loading import list_usable_files, load_carobd_csv
from src.features.extractor import extract_features
from src.features.normalizer import normalised_feature_names
from src.features.windowing import sliding_windows
from src.injection import inject_session
from src.models.classifier import ALL_LABELS

_FEAT_COLS = normalised_feature_names()

# Load classifier
with open("models/xgb_classifier_v1.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
norm = bundle["normalizer"]

# Take a held-out session, inject each fault, slice into 3 regions:
#   pre-onset (should be healthy)
#   early ramp (~30% through ramp — fault barely visible — CLASSIFIER NEVER SAW THIS)
#   late ramp (~70% through ramp — fault halfway — ALSO NEVER SEEN)
#   post-ramp (full fault — what classifier was trained on)

path = next(p for p in list_usable_files(Path(DATA_CAROBD_DIR)) if p.name == "drive1.csv")
df_clean = load_carobd_csv(path)
session_id = df_clean.attrs["session_id"]
print(f"Session: {session_id}, {len(df_clean)} rows")

for fault in ["air_system", "fuel_system", "coolant_temp_sensor", "throttle_position_sensor"]:
    df_faulty = inject_session(df_clean, fault, onset_fraction=0.40, ramp_fraction=0.15,
                                noise_std=0.3, random_seed=RANDOM_SEED)
    params = df_faulty.attrs["injection"]
    onset = params.onset_idx
    ramp_end = onset + params.ramp_len

    early_ramp_start = onset + int(0.2 * params.ramp_len)
    early_ramp_end   = onset + int(0.4 * params.ramp_len)
    mid_ramp_start   = onset + int(0.4 * params.ramp_len)
    mid_ramp_end     = onset + int(0.7 * params.ramp_len)

    regions = {
        "pre-onset (healthy)": df_faulty.iloc[:onset].reset_index(drop=True),
        "early ramp (20-40%)": df_faulty.iloc[early_ramp_start:early_ramp_end].reset_index(drop=True),
        "mid ramp   (40-70%)": df_faulty.iloc[mid_ramp_start:mid_ramp_end].reset_index(drop=True),
        "post-ramp  (100%)  ": df_faulty.iloc[ramp_end:].reset_index(drop=True),
    }

    print(f"\n=== Fault: {fault} ===")
    for region_name, region in regions.items():
        if len(region) < 60:
            continue
        rows = [extract_features(w) for w, _ in sliding_windows(region, fault)]
        if not rows:
            continue
        feats = pd.DataFrame(rows)
        feats["label"] = fault  # not actually used for prediction
        X = norm.transform(feats)[_FEAT_COLS].to_numpy(dtype=float)
        pred_idx = clf.predict(X)
        pred_label = [ALL_LABELS[i] for i in pred_idx]
        # Distribution: how many predicted as fault vs healthy
        n = len(pred_label)
        n_healthy = sum(p == "healthy" for p in pred_label)
        n_correct_fault = sum(p == fault for p in pred_label)
        n_other_fault = n - n_healthy - n_correct_fault
        print(f"  {region_name}  n={n:3d}  pred-healthy={n_healthy:3d}  "
              f"pred-{fault}={n_correct_fault:3d}  pred-other-fault={n_other_fault:3d}")
