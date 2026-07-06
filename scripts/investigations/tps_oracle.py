"""Oracle test: what MAE is possible for TPS if we used perfect ramp-stage knowledge?
We rebuild the TPS forecast dataset but inject the oracle ramp_stage as severity target."""
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from src.config import DATA_CAROBD_DIR, FORECAST_HORIZON_S, WINDOW_STRIDE_S, RANDOM_SEED
from src.data_loading import list_usable_files, load_carobd_csv
from src.features.extractor import extract_features, feature_names
from src.features.windowing import sliding_windows
from src.injection import inject_session
from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, _HELD_OUT_SESSIONS

FAULT_TYPE = "throttle_position_sensor"
_HORIZON_STEPS = FORECAST_HORIZON_S // WINDOW_STRIDE_S

# Build oracle dataset with RAMP-STAGE severity
carobd_dir = Path(DATA_CAROBD_DIR)
usable = list_usable_files(carobd_dir)

all_rows = []
for file_idx, path in enumerate(usable):
    session_seed = RANDOM_SEED + file_idx
    df_clean = load_carobd_csv(path)
    session_id = df_clean.attrs["session_id"]

    df_faulty = inject_session(df_clean, FAULT_TYPE, onset_fraction=0.40,
                                ramp_fraction=0.15, noise_std=0.3, random_seed=session_seed)
    params = df_faulty.attrs["injection"]
    fault_region = df_faulty.iloc[params.onset_idx:].reset_index(drop=True)
    n_fault_rows = len(fault_region)

    all_windows_feats = [extract_features(w) for w, _ in sliding_windows(fault_region, FAULT_TYPE)]
    n = len(all_windows_feats)

    # Oracle: ramp_stage at window i = position in ramp, clamped to [0,1]
    # ramp_len in rows = params.ramp_len
    ramp_len = params.ramp_len  # number of rows in the ramp

    def ramp_stage(row_in_fault: float) -> float:
        # Approximate centre row of window
        return float(np.clip(row_in_fault / ramp_len, 0.0, 1.0))

    for i in range(n - _HORIZON_STEPS):
        future_window_start = (i + _HORIZON_STEPS) * WINDOW_STRIDE_S  # approximate row idx
        target = ramp_stage(future_window_start)
        row = dict(all_windows_feats[i])
        row["severity_target"] = target
        row["session_id"] = session_id
        all_rows.append(row)

feat_cols = feature_names()
oracle_df = pd.DataFrame(all_rows)[feat_cols + ["severity_target", "session_id"]]
print(f"Oracle dataset: {len(oracle_df)} samples")
print(f"Target stats: mean={oracle_df['severity_target'].mean():.3f}, std={oracle_df['severity_target'].std():.3f}")

# Train and evaluate
ds = load_dataset()
train_df, _ = session_split(ds)
norm = BaselineNormalizer().fit(train_df)
_FEAT_COLS = normalised_feature_names()

train_oracle = oracle_df[~oracle_df["session_id"].isin(_HELD_OUT_SESSIONS)]
test_oracle = oracle_df[oracle_df["session_id"].isin(_HELD_OUT_SESSIONS)]

X_train = norm.transform(train_oracle)[_FEAT_COLS].to_numpy(dtype=float)
y_train = train_oracle["severity_target"].to_numpy(dtype=float)
X_test = norm.transform(test_oracle)[_FEAT_COLS].to_numpy(dtype=float)
y_test = test_oracle["severity_target"].to_numpy(dtype=float)

reg = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)
reg.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, reg.predict(X_train).clip(0,1)) * 100
test_mae  = mean_absolute_error(y_test,  reg.predict(X_test).clip(0,1))  * 100
print(f"Oracle: train_MAE={train_mae:.1f}%  test_MAE={test_mae:.1f}%")
