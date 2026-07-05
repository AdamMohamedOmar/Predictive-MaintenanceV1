"""Tests for the Random Forest classifier module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.dataset_builder import LABEL_TO_ID
from src.features.extractor import feature_names
from src.models.classifier import (
    session_split,
    train,
    evaluate,
    top_features,
    _HELD_OUT_SESSIONS,
    ALL_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "synthetic" / "dataset_v1.parquet"


# ─── Synthetic dataset fixture ────────────────────────────────────────────────

def _make_dataset(n_per_class: int = 50) -> pd.DataFrame:
    """Minimal synthetic dataset with all 5 classes and 3 fake sessions."""
    rng = np.random.default_rng(99)
    rows = []
    feat_cols = feature_names()
    sessions = ["sess_a", "sess_b", "sess_c"]
    labels = ALL_LABELS

    for i, label in enumerate(labels):
        for j in range(n_per_class):
            row = {col: rng.uniform(0, 1) for col in feat_cols}
            # Give each class a distinct mean so the RF has signal to learn
            for col in feat_cols:
                row[col] += i * 5.0
            row["label"] = label
            row["label_id"] = LABEL_TO_ID[label]
            row["session_id"] = sessions[j % len(sessions)]
            row["fault_type"] = label
            rows.append(row)

    return pd.DataFrame(rows)


# ─── session_split ────────────────────────────────────────────────────────────

def test_session_split_no_overlap():
    ds = _make_dataset()
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    assert len(set(train_df["session_id"]) & set(test_df["session_id"])) == 0


def test_session_split_held_out_in_test():
    ds = _make_dataset()
    _, test_df = session_split(ds, held_out={"sess_b", "sess_c"})
    assert set(test_df["session_id"]) == {"sess_b", "sess_c"}


def test_session_split_covers_all_rows():
    ds = _make_dataset()
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    assert len(train_df) + len(test_df) == len(ds)


# ─── train ────────────────────────────────────────────────────────────────────

def test_train_returns_fitted_rf():
    ds = _make_dataset()
    train_df, _ = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=10, random_seed=0)
    assert isinstance(clf, RandomForestClassifier)
    assert hasattr(clf, "feature_importances_")


def test_train_n_estimators_respected():
    ds = _make_dataset()
    train_df, _ = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=17, random_seed=0)
    assert len(clf.estimators_) == 17


# ─── evaluate ────────────────────────────────────────────────────────────────

def test_evaluate_returns_macro_f1():
    ds = _make_dataset(n_per_class=80)
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=20, random_seed=0)
    results = evaluate(clf, test_df)
    assert "macro_f1" in results
    assert 0.0 <= results["macro_f1"] <= 1.0


def test_evaluate_confusion_matrix_shape():
    ds = _make_dataset(n_per_class=80)
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=20, random_seed=0)
    results = evaluate(clf, test_df)
    cm = results["confusion_matrix"]
    n = len(cm)
    assert n == len(cm[0])  # square; size equals number of ALL_LABELS (6 with cold_start)


def test_evaluate_per_class_covers_all_labels():
    ds = _make_dataset(n_per_class=80)
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=20, random_seed=0)
    results = evaluate(clf, test_df)
    for label in ALL_LABELS:
        assert label in results["per_class"]


def test_evaluate_separable_data_near_perfect():
    """When classes are well-separated, RF should score near 1.0."""
    ds = _make_dataset(n_per_class=100)
    train_df, test_df = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=50, random_seed=0)
    results = evaluate(clf, test_df)
    assert results["macro_f1"] > 0.90


# ─── top_features ────────────────────────────────────────────────────────────

def test_top_features_returns_dataframe():
    ds = _make_dataset(n_per_class=50)
    train_df, _ = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=10, random_seed=0)
    df = top_features(clf, n=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "feature" in df.columns and "importance" in df.columns


def test_top_features_sorted_descending():
    ds = _make_dataset(n_per_class=50)
    train_df, _ = session_split(ds, held_out={"sess_c"})
    clf = train(train_df, n_estimators=10, random_seed=0)
    df = top_features(clf, n=10)
    assert (df["importance"].diff().dropna() <= 0).all()


# ─── Integration on real dataset ─────────────────────────────────────────────

@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not built yet")
def test_real_dataset_session_split_no_leakage():
    ds = pd.read_parquet(DATASET_PATH)
    train_df, test_df = session_split(ds)
    train_sessions = set(train_df["session_id"])
    test_sessions = set(test_df["session_id"])
    assert train_sessions.isdisjoint(test_sessions)
    assert _HELD_OUT_SESSIONS.issubset(test_sessions)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not built yet")
def test_real_dataset_all_classes_in_train():
    ds = pd.read_parquet(DATASET_PATH)
    train_df, _ = session_split(ds)
    assert set(train_df["label"].unique()) == set(ALL_LABELS)
