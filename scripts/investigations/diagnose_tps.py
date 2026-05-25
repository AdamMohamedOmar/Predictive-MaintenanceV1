"""Diagnose TPS forecaster performance."""
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
train_tps = tps[~tps["session_id"].isin(_HELD_OUT_SESSIONS)]
test_tps = tps[tps["session_id"].isin(_HELD_OUT_SESSIONS)]

print(f"Train size: {len(train_tps)}, Test size: {len(test_tps)}")
print(f"Test sessions: {test_tps['session_id'].unique()}")
print(f"Test  severity: mean={test_tps['severity_target'].mean():.3f}, std={test_tps['severity_target'].std():.3f}")
print(f"Train severity: mean={train_tps['severity_target'].mean():.3f}, std={train_tps['severity_target'].std():.3f}")

X_train = norm.transform(train_tps)[_FEAT_COLS].to_numpy(dtype=float)
y_train = train_tps["severity_target"].to_numpy(dtype=float)
X_test = norm.transform(test_tps)[_FEAT_COLS].to_numpy(dtype=float)
y_test = test_tps["severity_target"].to_numpy(dtype=float)

# More capacity
reg = xgb.XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
reg.fit(X_train, y_train)

y_pred_train = reg.predict(X_train).clip(0, 1)
y_pred_test = reg.predict(X_test).clip(0, 1)
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train)*100:.1f}%")
print(f"Test  MAE: {mean_absolute_error(y_test, y_pred_test)*100:.1f}%")

# What does test distribution look like?
print(f"\nTest prediction stats: mean={y_pred_test.mean():.3f}, std={y_pred_test.std():.3f}")
print(f"Test target   stats:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")
