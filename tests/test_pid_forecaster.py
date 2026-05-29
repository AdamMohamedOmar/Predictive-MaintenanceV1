"""Tests for the PID-target forecaster.

Sanity scope only — verifies the dataset builder, the per-PID training
loop, the predict/residual API, and save/load round-trip. **Not a real-
fault detection validation.** Whether residuals separate real-fault
from healthy windows is validated against Skoda data per
docs/REAL_FAULT_COLLECTION.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import feature_names
from src.features.normalizer import BaselineNormalizer
from src.features.pid_forecast_dataset import TARGET_PIDS
from src.features.regime import regime_feature_names
from src.models.pid_forecaster import (
    PIDForecaster,
    _zscore_target,
    forecast_session_split,
    train_all_pid_forecasters,
)


# ─── Synthetic fixture (tiny, controllable) ──────────────────────────────────


def _make_synthetic_pairs(n_per_session: int = 60, n_sessions: int = 4) -> pd.DataFrame:
    """Build a tiny PID-forecast dataset for unit tests.

    Each row carries the 83 base features plus 4 target_* columns. The
    targets are derived from the input features so a regressor with any
    capacity can fit perfectly on the training portion.
    """
    rng = np.random.default_rng(0)
    cols = feature_names()
    regime_cols = regime_feature_names()
    continuous = [c for c in cols if c not in set(regime_cols)]

    rows: list[dict] = []
    for s in range(n_sessions):
        session_id = f"sess_{s}"
        for _ in range(n_per_session):
            row: dict = {c: float(rng.normal(0.0, 1.0)) for c in continuous}
            for rc in regime_cols:
                row[rc] = 0.0
            row[regime_cols[4]] = 1.0  # CRUISE
            # Targets are simple linear functions of the current PID values
            # (+ small noise) so the model has a learnable signal.
            for pid in TARGET_PIDS:
                row[f"target_{pid}"] = row[pid] + float(rng.normal(0.0, 0.1))
            row["session_id"] = session_id
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Plumbing / API ─────────────────────────────────────────────────────────


def test_target_pids_subset_of_features():
    """Every TARGET_PID must be one of the 83 base features."""
    feats = set(feature_names())
    for pid in TARGET_PIDS:
        assert pid in feats, f"{pid} is not a base feature — z-scoring will fail."


def test_zscore_target_uses_normalizer_stats():
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=3)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)

    pid = TARGET_PIDS[0]
    raw = ds[f"target_{pid}"].to_numpy(dtype=float)
    z = _zscore_target(raw, norm, pid)
    # Roundtrip via the same stats reproduces the raw value (up to float eps)
    from src.features.normalizer import _CONTINUOUS_COLS
    idx = _CONTINUOUS_COLS.index(pid)
    mean = norm._scaler.mean_[idx]
    scale = norm._scaler.scale_[idx]
    raw_back = z * scale + mean
    assert np.allclose(raw_back, raw, atol=1e-9)


def test_session_split_holds_out_correct_sessions():
    ds = _make_synthetic_pairs(n_per_session=20, n_sessions=4)
    train, test = forecast_session_split(ds, held_out={"sess_0", "sess_3"})
    assert set(train["session_id"].unique()) == {"sess_1", "sess_2"}
    assert set(test["session_id"].unique()) == {"sess_0", "sess_3"}


# ─── Training (small synthetic dataset) ────────────────────────────────────


def test_train_all_pid_forecasters_returns_bundle():
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)

    forecaster = train_all_pid_forecasters(
        ds,
        norm,
        held_out={"sess_3"},
        n_estimators=30,
        random_seed=0,
    )
    assert isinstance(forecaster, PIDForecaster)
    assert set(forecaster._models.keys()) == set(TARGET_PIDS)
    # results dict has one entry per target
    assert set(forecaster.results.keys()) == set(TARGET_PIDS)
    for pid in TARGET_PIDS:
        r = forecaster.results[pid]
        assert "mae_z" in r and "mae_persistence_baseline_z" in r


def test_predict_pid_values_returns_one_per_target():
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)
    forecaster = train_all_pid_forecasters(
        ds, norm, held_out={"sess_3"}, n_estimators=30, random_seed=0
    )
    feat_dict = {c: float(ds.iloc[0][c]) for c in feature_names()}
    preds = forecaster.predict_pid_values(feat_dict)
    assert set(preds.keys()) == set(TARGET_PIDS)
    assert all(isinstance(v, float) for v in preds.values())


def test_residuals_returns_per_pid_plus_aggregate():
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)
    forecaster = train_all_pid_forecasters(
        ds, norm, held_out={"sess_3"}, n_estimators=30, random_seed=0
    )
    feat_dict = {c: float(ds.iloc[0][c]) for c in feature_names()}
    actual_future = {c: float(ds.iloc[1][c]) for c in feature_names()}
    residuals = forecaster.residuals(feat_dict, actual_future)

    for pid in TARGET_PIDS:
        assert pid in residuals
        assert residuals[pid] >= 0.0
    assert "_aggregate" in residuals
    assert residuals["_aggregate"] >= 0.0


def test_summary_dataframe_has_one_row_per_pid():
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)
    forecaster = train_all_pid_forecasters(
        ds, norm, held_out={"sess_3"}, n_estimators=30, random_seed=0
    )
    summary = forecaster.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(TARGET_PIDS)
    assert {"pid", "MAE_z", "MAE_persistence_baseline_z", "beats_persistence"} <= set(
        summary.columns
    )


# ─── Persistence ────────────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)
    forecaster = train_all_pid_forecasters(
        ds, norm, held_out={"sess_3"}, n_estimators=30, random_seed=0
    )

    forecaster.save(models_dir=tmp_path, results_dir=tmp_path)
    loaded = PIDForecaster.load(models_dir=tmp_path)

    # Predictions match exactly after round-trip.
    feat_dict = {c: float(ds.iloc[0][c]) for c in feature_names()}
    p1 = forecaster.predict_pid_values(feat_dict)
    p2 = loaded.predict_pid_values(feat_dict)
    for pid in TARGET_PIDS:
        assert abs(p1[pid] - p2[pid]) < 1e-9


def test_load_rejects_mismatched_target_list(tmp_path, monkeypatch):
    """If the saved target list doesn't match the current codebase, load fails loud."""
    ds = _make_synthetic_pairs(n_per_session=80, n_sessions=4)
    ds["label"] = "healthy"
    norm = BaselineNormalizer().fit(ds)
    forecaster = train_all_pid_forecasters(
        ds, norm, held_out={"sess_3"}, n_estimators=30, random_seed=0
    )
    forecaster.save(models_dir=tmp_path, results_dir=tmp_path)

    # Monkey-patch the in-memory TARGET_PIDS list to simulate code drift
    monkeypatch.setattr(
        "src.models.pid_forecaster.TARGET_PIDS",
        TARGET_PIDS + ["BOGUS_NEW_PID__mean"],
    )
    with pytest.raises(RuntimeError, match="targets"):
        PIDForecaster.load(models_dir=tmp_path)
