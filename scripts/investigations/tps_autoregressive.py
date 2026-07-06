"""Test: does current-window severity predict future severity for TPS?"""
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features.normalizer import BaselineNormalizer
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, _HELD_OUT_SESSIONS
from src.features.severity import compute_baselines

ds = load_dataset()
train_df, _ = session_split(ds)
norm = BaselineNormalizer().fit(train_df)
healthy_train = train_df[train_df["label"] == "healthy"]
baselines = compute_baselines(healthy_train)

tps = pd.read_parquet("data/synthetic/forecast_throttle_position_sensor_v1.parquet")
test_tps = tps[tps["session_id"].isin(_HELD_OUT_SESSIONS)]

# Compute current-window TPS severity using same formula
ratio_base = baselines["THROTTLE_TO_PEDAL_RATIO"]
current_sev = ((tps["THROTTLE_TO_PEDAL_RATIO"] - ratio_base) / 0.25).clip(0, 1)
tps["current_severity"] = current_sev

test_tps = tps[tps["session_id"].isin(_HELD_OUT_SESSIONS)]
print("Test: predict future=current (naive AR baseline)")
naive_mae = mean_absolute_error(test_tps["severity_target"], test_tps["current_severity"]) * 100
print(f"  MAE = {naive_mae:.1f}%")

print()
print("Correlation current_sev vs future target:", tps["current_severity"].corr(tps["severity_target"]))

# What's mean severity trajectory within sessions?
print()
print("Drive1 severity progression (window_idx vs sev):")
d1 = tps[tps["session_id"] == "drive1"].reset_index(drop=True)
for i in range(0, min(len(d1), 30), 5):
    print(f"  window {i:3d}: current={d1.loc[i,'current_severity']:.3f}, target={d1.loc[i,'severity_target']:.3f}")
