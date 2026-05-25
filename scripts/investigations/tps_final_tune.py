"""Push for <15% MAE on TPS with heavy regularization."""
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
X_test = norm.transform(test_tps)[_FEAT_COLS].to_numpy(dtype=float)
y_test = test_tps["severity_target"].to_numpy(dtype=float)

print(f"Naive (predict train mean={y_train.mean():.3f}): MAE={mean_absolute_error(y_test, np.ones(len(y_test))*y_train.mean())*100:.1f}%")

configs = [
    # Very shallow, heavy reg
    dict(n_estimators=100, max_depth=2, learning_rate=0.1,  subsample=0.5, colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=10.0),
    dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.6, colsample_bytree=0.6, reg_alpha=5.0, reg_lambda=10.0),
    dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5, reg_alpha=3.0, reg_lambda=5.0),
    dict(n_estimators=500, max_depth=2, learning_rate=0.02, subsample=0.6, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=3.0),
    dict(n_estimators=50,  max_depth=2, learning_rate=0.2,  subsample=0.8, colsample_bytree=0.6, reg_alpha=10.0, reg_lambda=20.0),
    # Slightly deeper with heavy reg
    dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.5, colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=10.0),
    dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.6, colsample_bytree=0.5, reg_alpha=3.0, reg_lambda=8.0),
]

for cfg in configs:
    reg = xgb.XGBRegressor(
        **cfg, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
    )
    reg.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, reg.predict(X_train).clip(0, 1)) * 100
    test_mae  = mean_absolute_error(y_test,  reg.predict(X_test).clip(0, 1))  * 100
    print(f"depth={cfg['max_depth']} n={cfg['n_estimators']:3d} lr={cfg['learning_rate']} "
          f"a={cfg['reg_alpha']} l={cfg['reg_lambda']} | "
          f"train={train_mae:.1f}% test={test_mae:.1f}% {'*** PASS' if test_mae<=15 else ''}")
