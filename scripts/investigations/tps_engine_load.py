"""Test THROTTLE/ENGINE_LOAD as TPS severity proxy."""
import pandas as pd
import numpy as np
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, _HELD_OUT_SESSIONS

ds = load_dataset()
train_df, _ = session_split(ds)
_EPS = 1e-3

healthy = train_df[train_df["label"] == "healthy"]
# Check healthy THROTTLE__mean / ENGINE_LOAD__mean
h_ratio = healthy["THROTTLE__mean"] / (healthy["ENGINE_LOAD__mean"] + _EPS)
print("Healthy THROTTLE__mean/ENGINE_LOAD__mean:")
print(f"  mean={h_ratio.mean():.4f}, std={h_ratio.std():.4f}, min={h_ratio.min():.4f}, max={h_ratio.max():.4f}")

# Now TPS forecast dataset
tps = pd.read_parquet("data/synthetic/forecast_throttle_position_sensor_v1.parquet")
baseline = float(h_ratio.mean())
tps_ratio = tps["THROTTLE__mean"] / (tps["ENGINE_LOAD__mean"] + _EPS)
tps_sev = ((tps_ratio - baseline) / 0.25).clip(0, 1)
print(f"\nTPS THROTTLE/ENGINE_LOAD (current window):")
print(f"  mean={tps_ratio.mean():.4f}, std={tps_ratio.std():.4f}")
print(f"  severity: mean={tps_sev.mean():.4f}, std={tps_sev.std():.4f}")
print(f"  Correlation with severity_target: {tps_sev.corr(tps['severity_target']):.4f}")

# Compare with original THROTTLE_TO_PEDAL_RATIO
ratio_base = 1.008586
t2p = tps["THROTTLE_TO_PEDAL_RATIO"]
t2p_sev = ((t2p - ratio_base) / 0.25).clip(0, 1)
print(f"\nOriginal THROTTLE_TO_PEDAL_RATIO (current window):")
print(f"  Correlation with severity_target: {t2p_sev.corr(tps['severity_target']):.4f}")

# Per-session breakdown
print("\nPer-session THROTTLE/ENGINE_LOAD target vs current_sev:")
for sess, grp in tps.groupby("session_id"):
    r = grp["THROTTLE__mean"] / (grp["ENGINE_LOAD__mean"] + _EPS)
    s = ((r - baseline) / 0.25).clip(0, 1)
    flag = "[TEST]" if sess in _HELD_OUT_SESSIONS else "[train]"
    print(f"  {flag} {sess}: curr_sev={s.mean():.3f}±{s.std():.3f}  target={grp['severity_target'].mean():.3f}±{grp['severity_target'].std():.3f}")
