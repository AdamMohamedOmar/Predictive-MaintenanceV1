"""Check TPS per-session MAE."""
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

X_train = norm.transform(train_tps)[_FEAT_COLS].to_numpy(dtype=float)
y_train = train_tps["severity_target"].to_numpy(dtype=float)

reg = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)
reg.fit(X_train, y_train)

for sess in _HELD_OUT_SESSIONS:
    sess_data = test_tps[test_tps["session_id"] == sess]
    if len(sess_data) == 0:
        continue
    X_s = norm.transform(sess_data)[_FEAT_COLS].to_numpy(dtype=float)
    y_s = sess_data["severity_target"].to_numpy(dtype=float)
    pred = reg.predict(X_s).clip(0, 1)
    mae = mean_absolute_error(y_s, pred) * 100
    naive_mae = mean_absolute_error(y_s, np.ones_like(y_s) * y_train.mean()) * 100
    print(f"{sess}: n={len(sess_data)}, MAE={mae:.1f}%, naive_MAE(predict_train_mean)={naive_mae:.1f}%")
    print(f"  target: mean={y_s.mean():.3f}, std={y_s.std():.3f}")
    print(f"  pred:   mean={pred.mean():.3f}, std={pred.std():.3f}")
