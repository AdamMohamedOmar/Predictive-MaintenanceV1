"""Investigate TPS per-session severity to identify distribution shift."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, _HELD_OUT_SESSIONS

ds = load_dataset()
train_df, _ = session_split(ds)
norm = BaselineNormalizer().fit(train_df)
_FEAT_COLS = normalised_feature_names()

tps = pd.read_parquet("data/synthetic/forecast_throttle_position_sensor_v1.parquet")

print("Per-session severity stats:")
for sess, grp in tps.groupby("session_id"):
    flag = "[TEST]" if sess in _HELD_OUT_SESSIONS else "[train]"
    print(f"  {flag} {sess}: n={len(grp):4d} sev_mean={grp['severity_target'].mean():.3f} std={grp['severity_target'].std():.3f}")

print()
print("Key features by session (THROTTLE_TO_PEDAL_RATIO):")
for sess, grp in tps.groupby("session_id"):
    flag = "[TEST]" if sess in _HELD_OUT_SESSIONS else "[train]"
    ratio = grp["THROTTLE_TO_PEDAL_RATIO"]
    print(f"  {flag} {sess}: mean={ratio.mean():.3f} std={ratio.std():.3f} min={ratio.min():.3f}")
