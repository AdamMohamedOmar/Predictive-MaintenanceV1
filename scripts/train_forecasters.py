"""Rebuild forecast datasets and retrain all 4 forecasters."""
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.features.dataset_builder import load_dataset
from src.features.severity import compute_baselines
from src.features.forecast_dataset import build_all_forecast_datasets
from src.features.normalizer import BaselineNormalizer
from src.models.classifier import session_split
from src.models.forecaster import train_all_forecasters

ds = load_dataset()
train_df, test_df = session_split(ds)
healthy_train = train_df[train_df["label"] == "healthy"]
baselines = compute_baselines(healthy_train)
print("Baselines:", {k: f"{v:.4f}" for k, v in baselines.items()})

forecast_ds = build_all_forecast_datasets(baselines)
print("Dataset sizes:", {k: len(v) for k, v in forecast_ds.items()})

norm = BaselineNormalizer().fit(train_df)
forecaster = train_all_forecasters(forecast_ds, norm, n_estimators=300, random_seed=42)

for fault, r in forecaster.results.items():
    status = "OK (commit)" if r["meets_commit_target"] else "MISS"
    stretch = "OK (stretch)" if r["meets_stretch_target"] else ""
    print(f"  {fault}: MAE={r['mae_pct_of_range']:.1f}%  {status} {stretch}")

forecaster.save()
print("Saved.")
