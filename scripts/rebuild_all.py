"""Full rebuild: classifier dataset → normalizer → XGBoost classifier → forecasters."""
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 1. Build classifier dataset
from src.features.dataset_builder import build_dataset, load_dataset
print("=== Building classifier dataset ===")
ds = build_dataset()
print(f"Dataset: {len(ds)} windows, {ds['label'].value_counts().to_dict()}")

# 2. Session split and normalizer
from src.models.classifier import session_split
from src.features.normalizer import BaselineNormalizer

train_df, test_df = session_split(ds)
norm = BaselineNormalizer().fit(train_df)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# 3. Retrain XGBoost classifier
from src.models.xgb_classifier import train as train_xgb, evaluate as eval_xgb, save_model as save_xgb
from src.features.normalizer import normalised_feature_names

print("\n=== Training XGBoost classifier ===")
clf, norm_clf = train_xgb(train_df, n_estimators=300)
results = eval_xgb(clf, norm_clf, test_df)
print(f"Macro-F1: {results['macro_f1']:.4f}")
print(f"Per-class F1: {results['per_class']}")
save_xgb(clf, norm_clf, results)

# 4. Build forecast datasets
from src.features.severity import compute_baselines
from src.features.forecast_dataset import build_all_forecast_datasets

print("\n=== Building forecast datasets ===")
healthy_train = train_df[train_df["label"] == "healthy"]
baselines = compute_baselines(healthy_train)
print("Baselines:", {k: f"{v:.4f}" for k, v in baselines.items()})
forecast_ds = build_all_forecast_datasets(baselines)
print("Dataset sizes:", {k: len(v) for k, v in forecast_ds.items()})

# 5. Retrain forecasters
from src.models.forecaster import train_all_forecasters

print("\n=== Training forecasters ===")
forecaster = train_all_forecasters(forecast_ds, norm, n_estimators=300, random_seed=42)
for fault, r in forecaster.results.items():
    status = "OK" if r["meets_commit_target"] else "MISS"
    print(f"  {fault}: MAE={r['mae_pct_of_range']:.1f}%  {status}")
forecaster.save()
print("\nDone.")
