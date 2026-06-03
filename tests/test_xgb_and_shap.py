"""Tests for BaselineNormalizer, XGBoost classifier, and SHAP explainer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import feature_names
from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.features.dataset_builder import LABEL_TO_ID, FAULT_TYPES
from src.models.classifier import ALL_LABELS, session_split
from src.models import xgb_classifier
from src.models.explainer import SHAPExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "synthetic" / "dataset_v1.parquet"

_FEAT_COLS = feature_names()


# ─── Shared synthetic dataset ─────────────────────────────────────────────────

def _make_dataset(n_per_class: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    sessions = ["s1", "s2", "s3"]
    for i, label in enumerate(ALL_LABELS):
        for j in range(n_per_class):
            row = {col: rng.uniform(0, 1) + i * 10.0 for col in _FEAT_COLS}
            row["label"] = label
            row["label_id"] = LABEL_TO_ID[label]
            row["session_id"] = sessions[j % len(sessions)]
            row["fault_type"] = label
            rows.append(row)
    return pd.DataFrame(rows)


# ─── BaselineNormalizer ───────────────────────────────────────────────────────

def test_normalizer_adds_z_columns():
    ds = _make_dataset()
    norm = BaselineNormalizer()
    out = norm.fit_transform(ds)
    for col in _FEAT_COLS:
        assert f"{col}__z" in out.columns


def test_normalizer_feature_count():
    assert len(normalised_feature_names()) == 83  # 83 z-scored only (78 continuous + 5 regime passthrough)


def test_normalizer_healthy_mean_z_near_zero():
    """After fitting on healthy windows, healthy z-scores should average ≈ 0.

    Regime one-hot __z columns are binary passthrough (not z-scored), so their
    means are whatever fraction of healthy windows are in each regime — excluded
    from this check.  Only continuous features are expected to average near 0.
    """
    from src.features.regime import regime_feature_names
    regime_z = {f"{c}__z" for c in regime_feature_names()}

    ds = _make_dataset()
    norm = BaselineNormalizer()
    out = norm.fit_transform(ds)
    continuous_z_cols = [f"{c}__z" for c in _FEAT_COLS if f"{c}__z" not in regime_z]
    healthy_z = out[out["label"] == "healthy"][continuous_z_cols]
    assert healthy_z.mean().abs().max() < 0.1


def test_normalizer_fault_z_clearly_nonzero():
    """Fault windows should have z-scores far from zero (signal should be visible)."""
    ds = _make_dataset()
    norm = BaselineNormalizer()
    out = norm.fit_transform(ds)
    fault_z = out[out["label"] != "healthy"][[f"{c}__z" for c in _FEAT_COLS]]
    assert fault_z.mean().abs().max() > 1.0


def test_normalizer_transform_without_fit_raises():
    norm = BaselineNormalizer()
    ds = _make_dataset()
    with pytest.raises(RuntimeError):
        norm.transform(ds)


def test_normalizer_no_healthy_rows_raises():
    ds = _make_dataset()
    fault_only = ds[ds["label"] != "healthy"]
    norm = BaselineNormalizer()
    with pytest.raises(ValueError):
        norm.fit(fault_only)


def test_normalizer_save_load_roundtrip(tmp_path):
    ds = _make_dataset()
    norm = BaselineNormalizer()
    norm.fit(ds)
    norm.save(tmp_path / "norm.pkl")
    norm2 = BaselineNormalizer.load(tmp_path / "norm.pkl")
    out1 = norm.transform(ds)[[f"{_FEAT_COLS[0]}__z"]]
    out2 = norm2.transform(ds)[[f"{_FEAT_COLS[0]}__z"]]
    pd.testing.assert_frame_equal(out1, out2)


def test_normalizer_does_not_mutate_input():
    ds = _make_dataset()
    before = ds[_FEAT_COLS[0]].copy()
    norm = BaselineNormalizer()
    norm.fit_transform(ds)
    pd.testing.assert_series_equal(ds[_FEAT_COLS[0]], before)


# ─── XGBoost classifier ───────────────────────────────────────────────────────

def test_xgb_train_returns_model_and_norm():
    ds = _make_dataset()
    train_df, _ = session_split(ds, held_out={"s3"})
    clf, norm = xgb_classifier.train(train_df, n_estimators=10, random_seed=0)
    import xgboost as xgb
    assert isinstance(clf, xgb.XGBClassifier)
    assert norm.is_fitted


def test_xgb_evaluate_returns_macro_f1():
    ds = _make_dataset(n_per_class=80)
    train_df, test_df = session_split(ds, held_out={"s3"})
    clf, norm = xgb_classifier.train(train_df, n_estimators=20, random_seed=0)
    results = xgb_classifier.evaluate(clf, norm, test_df)
    assert "macro_f1" in results
    assert 0.0 <= results["macro_f1"] <= 1.0


def test_xgb_separable_data_near_perfect():
    ds = _make_dataset(n_per_class=100)
    train_df, test_df = session_split(ds, held_out={"s3"})
    clf, norm = xgb_classifier.train(train_df, n_estimators=50, random_seed=0)
    results = xgb_classifier.evaluate(clf, norm, test_df)
    assert results["macro_f1"] > 0.90


def test_xgb_save_load_roundtrip(tmp_path):
    ds = _make_dataset(n_per_class=60)
    train_df, test_df = session_split(ds, held_out={"s3"})
    clf, norm = xgb_classifier.train(train_df, n_estimators=10, random_seed=0)
    results = xgb_classifier.evaluate(clf, norm, test_df)
    xgb_classifier.save_model(clf, norm, results, models_dir=tmp_path, results_dir=tmp_path)
    clf2, norm2 = xgb_classifier.load_model(tmp_path)
    results2 = xgb_classifier.evaluate(clf2, norm2, test_df)
    assert abs(results["macro_f1"] - results2["macro_f1"]) < 1e-6


# ─── SHAP explainer ───────────────────────────────────────────────────────────

def _make_fitted_xgb(n_per_class: int = 60):
    ds = _make_dataset(n_per_class)
    train_df, test_df = session_split(ds, held_out={"s3"})
    clf, norm = xgb_classifier.train(train_df, n_estimators=20, random_seed=0)
    return clf, norm, test_df


def test_shap_explain_window_returns_predicted_label():
    clf, norm, test_df = _make_fitted_xgb()
    explainer = SHAPExplainer(clf)
    row = test_df.iloc[:1][_FEAT_COLS]
    result = explainer.explain_window(row, norm)
    assert result["predicted_label"] in ALL_LABELS


def test_shap_explain_window_probabilities_sum_to_one():
    clf, norm, test_df = _make_fitted_xgb()
    explainer = SHAPExplainer(clf)
    row = test_df.iloc[:1][_FEAT_COLS]
    result = explainer.explain_window(row, norm)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-5


def test_shap_explain_window_top_features_count():
    clf, norm, test_df = _make_fitted_xgb()
    explainer = SHAPExplainer(clf)
    row = test_df.iloc[:1][_FEAT_COLS]
    result = explainer.explain_window(row, norm, top_n=7)
    assert len(result["top_features"]) == 7


def test_shap_explain_window_top_features_sorted_by_magnitude():
    clf, norm, test_df = _make_fitted_xgb()
    explainer = SHAPExplainer(clf)
    row = test_df.iloc[:1][_FEAT_COLS]
    result = explainer.explain_window(row, norm)
    magnitudes = [abs(sv) for _, sv in result["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_global_summary_returns_dataframe():
    clf, norm, test_df = _make_fitted_xgb()
    explainer = SHAPExplainer(clf)
    X_norm = norm.transform(test_df)[normalised_feature_names()].to_numpy(dtype=float)
    summary = explainer.global_summary(X_norm, top_n=10)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 10
    assert "feature" in summary.columns and "mean_abs_shap" in summary.columns


def test_shap_class_summary_for_each_fault(tmp_path):
    clf, norm, test_df = _make_fitted_xgb(n_per_class=80)
    explainer = SHAPExplainer(clf)
    X_norm = norm.transform(test_df)[normalised_feature_names()].to_numpy(dtype=float)
    for fault in FAULT_TYPES:
        df = explainer.class_summary(X_norm, fault, top_n=5)
        assert len(df) == 5


# ─── Integration on real dataset ─────────────────────────────────────────────

@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not built yet")
def test_real_xgb_trains_without_error():
    ds = pd.read_parquet(DATASET_PATH)
    train_df, test_df = session_split(ds)
    clf, norm = xgb_classifier.train(train_df, n_estimators=50, random_seed=42)
    results = xgb_classifier.evaluate(clf, norm, test_df)
    # Corrected-physics floor (not the old inflated 0.80+). Fixing the
    # speed-density air physics, the STFT→LTFT handoff, and the severity
    # decoupling — plus a jittered severity continuum that includes mild,
    # genuinely-hard faults — honestly lowered the synthetic fixed-holdout
    # macro-F1 to ≈ 0.80 at 300 trees (≈ 0.77 at the 50 trees used here).
    # We guard the charter's Week-4 minimum of 0.70; the headline numbers and
    # the per-class / worst-LOSO-fold story live in the README.
    assert results["macro_f1"] >= 0.70
