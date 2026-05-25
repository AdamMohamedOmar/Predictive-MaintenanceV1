"""Check max-based TPS ratio quality."""
import pandas as pd
import numpy as np
from src.features.dataset_builder import load_dataset
from src.models.classifier import session_split, _HELD_OUT_SESSIONS

ds = load_dataset()
train_df, _ = session_split(ds)

# Check healthy THROTTLE__max / COMMANDED__max
healthy = train_df[train_df["label"] == "healthy"]
_EPS = 1e-3
healthy_ratio = healthy["THROTTLE__max"] / (healthy["COMMANDED_THROTTLE_ACTUATOR__max"] + _EPS)
print("Healthy THROTTLE__max/COMMANDED__max:")
print(f"  mean={healthy_ratio.mean():.4f}, std={healthy_ratio.std():.4f}, min={healthy_ratio.min():.4f}, max={healthy_ratio.max():.4f}")

# Now check the TPS forecast dataset
tps = pd.read_parquet("data/synthetic/forecast_throttle_position_sensor_v1.parquet")

# Compute max-based severity for current and future
baseline = float(healthy_ratio.mean())
tps_ratio = tps["THROTTLE__max"] / (tps["COMMANDED_THROTTLE_ACTUATOR__max"] + _EPS)
tps_sev = ((tps_ratio - baseline) / 0.25).clip(0, 1)
print(f"\nTPS forecast max-ratio (current window):")
print(f"  mean={tps_ratio.mean():.4f}, std={tps_ratio.std():.4f}")
print(f"  severity: mean={tps_sev.mean():.4f}, std={tps_sev.std():.4f}")
print(f"  Correlation severity_current vs severity_target: {tps_sev.corr(tps['severity_target']):.4f}")

# Check per-session breakdown
print("\nPer-session: max_ratio_mean  vs  target_mean")
for sess, grp in tps.groupby("session_id"):
    r = grp["THROTTLE__max"] / (grp["COMMANDED_THROTTLE_ACTUATOR__max"] + _EPS)
    flag = "[TEST]" if sess in _HELD_OUT_SESSIONS else "[train]"
    print(f"  {flag} {sess}: max_ratio={r.mean():.3f}±{r.std():.3f}  target={grp['severity_target'].mean():.3f}±{grp['severity_target'].std():.3f}")
