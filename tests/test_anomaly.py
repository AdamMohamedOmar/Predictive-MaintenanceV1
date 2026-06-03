"""Tests for AnomalyDetector.

Sanity scope only — verifies the model fits, scores in the documented
range, and separates synthetic healthy from synthetic anomalous on a
controlled fixture. **NOT a real-fault detection validation** — see
docs/REAL_FAULT_COLLECTION.md (Step 5) for that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import feature_names
from src.features.normalizer import BaselineNormalizer
from src.features.regime import regime_feature_names
from src.models.anomaly import AnomalyDetector


# ─── Synthetic fixture ──────────────────────────────────────────────────────


def _make_dataset(n_healthy: int = 400, n_fault: int = 80, seed: int = 42) -> pd.DataFrame:
    """Tiny in-memory dataset for unit tests.

    Healthy rows: N(0, 1) per continuous feature, regime one-hot = CRUISE.
    Fault rows: same baseline + a strong (+5σ) bias on LTFT features so
    they're unambiguously out-of-distribution.
    """
    rng = np.random.default_rng(seed)
    cols = feature_names()
    regime_cols = regime_feature_names()
    continuous = [c for c in cols if c not in set(regime_cols)]

    def _row_block(n: int, label: str) -> pd.DataFrame:
        data = {c: rng.normal(0.0, 1.0, n) for c in continuous}
        for rcol in regime_cols:
            data[rcol] = 0.0
        df = pd.DataFrame(data)
        df[regime_cols[4]] = 1.0  # CRUISE regime for every row
        df["label"] = label
        return df

    healthy = _row_block(n_healthy, "healthy")
    fault = _row_block(n_fault, "fuel_system")
    # Push LTFT features +5σ above healthy mean — clear anomalies
    for col in (
        "LONG_TERM_FUEL_TRIM_BANK_1__mean",
        "LONG_TERM_FUEL_TRIM_BANK_1__max",
        "LONG_TERM_FUEL_TRIM_BANK_1__min",
    ):
        if col in fault.columns:
            fault[col] = fault[col] + 5.0

    return pd.concat([healthy, fault], ignore_index=True)


# ─── Plumbing tests ─────────────────────────────────────────────────────────


def test_score_before_fit_raises():
    det = AnomalyDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.score({"X": 1.0}, BaselineNormalizer())


def test_score_batch_before_fit_raises():
    det = AnomalyDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.score_batch(pd.DataFrame([{"X": 1.0}]), BaselineNormalizer())


def test_fit_with_no_healthy_rows_raises():
    ds = _make_dataset()
    fault_only = ds[ds["label"] != "healthy"]
    norm = BaselineNormalizer()
    # The normalizer itself rejects no-healthy input — same guard
    with pytest.raises(ValueError):
        norm.fit(fault_only)


def test_is_fitted_flag():
    ds = _make_dataset()
    norm = BaselineNormalizer().fit(ds)
    det = AnomalyDetector(n_estimators=20, random_seed=0)
    assert det.is_fitted is False
    det.fit(ds, norm)
    assert det.is_fitted is True


def test_score_in_unit_range():
    ds = _make_dataset()
    norm = BaselineNormalizer().fit(ds)
    det = AnomalyDetector(n_estimators=50, random_seed=0).fit(ds, norm)
    for _, row in ds.head(20).iterrows():
        feat_dict = {c: float(row[c]) for c in feature_names()}
        s = det.score(feat_dict, norm)
        assert 0.0 <= s <= 1.0


# ─── Sanity (separation on synthetic anomalies) ────────────────────────────


def test_fpr_budget_limits_false_alarms():
    """P1-4: with a 1 % budget, ≤ ~1 % of fresh healthy windows reach the alarm
    ceiling. The old p95 calibration maxed out ~5 % of healthy by design.
    """
    train = _make_dataset(n_healthy=2000, n_fault=0, seed=1)
    norm = BaselineNormalizer().fit(train)
    det = AnomalyDetector(n_estimators=120, random_seed=0, fpr_budget=0.01).fit(train, norm)

    fresh = _make_dataset(n_healthy=2000, n_fault=0, seed=999)  # same dist, unseen draw
    scores = det.score_batch(fresh, norm)
    fpr = float((scores >= 0.99).mean())
    assert fpr <= 0.03, (
        f"Healthy false-positive rate {fpr:.3f} exceeds the 1 % budget (+margin). "
        f"The FPR-budget calibration is not holding."
    )


def test_fault_scores_separate_from_healthy():
    """Synthetic anomalies score higher than synthetic healthy windows.

    Sanity check that the fit-and-score pipeline produces a usable signal.
    This is NOT real-fault validation. The fault windows are constructed
    by hand-biasing the features the classifier trains on — the same
    self-consistency loop documented in the project root README.
    """
    ds = _make_dataset(n_healthy=500, n_fault=100)
    norm = BaselineNormalizer().fit(ds)
    det = AnomalyDetector(n_estimators=100, random_seed=0).fit(ds, norm)

    healthy_df = ds[ds["label"] == "healthy"]
    fault_df = ds[ds["label"] != "healthy"]

    healthy_scores = det.score_batch(healthy_df, norm)
    fault_scores = det.score_batch(fault_df, norm)

    assert fault_scores.mean() > healthy_scores.mean() + 0.15, (
        f"Detector failed sanity separation. "
        f"Healthy mean={healthy_scores.mean():.3f}, fault mean={fault_scores.mean():.3f}. "
        f"This is a unit-test sanity check, not real-fault validation."
    )


# ─── Persistence ────────────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    ds = _make_dataset()
    norm = BaselineNormalizer().fit(ds)
    det = AnomalyDetector(n_estimators=50, random_seed=0).fit(ds, norm)

    path = tmp_path / "anomaly.pkl"
    det.save(path)
    det2 = AnomalyDetector.load(path)
    assert det2.is_fitted

    feat_dict = {c: float(ds.iloc[0][c]) for c in feature_names()}
    s1 = det.score(feat_dict, norm)
    s2 = det2.score(feat_dict, norm)
    assert abs(s1 - s2) < 1e-9


def test_save_unfitted_raises(tmp_path):
    det = AnomalyDetector()
    with pytest.raises(RuntimeError, match="unfitted"):
        det.save(tmp_path / "x.pkl")
