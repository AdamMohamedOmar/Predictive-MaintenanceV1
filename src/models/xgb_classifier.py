"""XGBoost fault classifier — Week 4 model.

Why XGBoost after Random Forest?
---------------------------------
RF grows trees independently and averages them — every tree is equally
weighted. XGBoost grows trees sequentially: each new tree focuses on the
windows the previous trees got wrong (gradient boosting). On tabular
sensor data, boosting typically squeezes out another 2–5 % F1 over
bagging — worth having as the model we ship.

More importantly for this project: XGBoost + TreeExplainer is the
gold-standard combination for SHAP explanations. The SHAP values from
XGBoost are computed exactly (not sampled), so the explanation
"LTFT pushed this prediction toward fuel_system by 0.43" is precise
rather than approximate.

This module mirrors the RF module's API: session_split lives in
classifier.py and is shared by both models.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.features.dataset_builder import FAULT_TYPES, LABEL_TO_ID
from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.models.classifier import ALL_LABELS

log = logging.getLogger(__name__)


def train(
    train_df: pd.DataFrame,
    *,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_seed: int = RANDOM_SEED,
) -> tuple[xgb.XGBClassifier, BaselineNormalizer]:
    """Fit a normaliser then an XGBoost classifier on *train_df*.

    Returns both the fitted model and the fitted normaliser so they
    can be saved together and used as a unit during inference.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split from ``session_split``.

    Returns
    -------
    (XGBClassifier, BaselineNormalizer)
    """
    norm = BaselineNormalizer()
    train_norm = norm.fit_transform(train_df)

    feat_cols = normalised_feature_names()
    X = train_norm[feat_cols].to_numpy(dtype=float)
    y = train_norm["label_id"].to_numpy(dtype=int)

    # sample_weight mirrors class_weight='balanced' from sklearn
    sample_weights = compute_sample_weight("balanced", y)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        num_class=len(ALL_LABELS),
        eval_metric="mlogloss",
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X, y, sample_weight=sample_weights)
    log.info(
        "Trained XGB: %d trees, depth %d, lr %.3f, %d train samples, %d features",
        n_estimators,
        max_depth,
        learning_rate,
        len(X),
        len(feat_cols),
    )
    return clf, norm


def evaluate(
    clf: xgb.XGBClassifier,
    norm: BaselineNormalizer,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate XGBoost on the test split.

    Parameters
    ----------
    clf : XGBClassifier
    norm : BaselineNormalizer
        Must be the same normaliser used during training.
    test_df : pd.DataFrame

    Returns
    -------
    dict with keys: macro_f1, per_class, confusion_matrix, test_sessions
    """
    test_norm = norm.transform(test_df)
    feat_cols = normalised_feature_names()
    X_test = test_norm[feat_cols].to_numpy(dtype=float)
    y_true = test_df["label_id"].to_numpy(dtype=int)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ALL_LABELS))),
        target_names=ALL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_LABELS))))
    macro_f1 = report["macro avg"]["f1-score"]
    log.info("XGB test macro-F1: %.4f", macro_f1)

    return {
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in ALL_LABELS
        },
        "confusion_matrix": cm.tolist(),
        "label_order": ALL_LABELS,
        "test_sessions": sorted(test_df["session_id"].unique().tolist()),
    }


def save_model(
    clf: xgb.XGBClassifier,
    norm: BaselineNormalizer,
    results: dict,
    models_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Save model, normaliser, and results to disk."""
    models_dir = Path(models_dir or MODELS_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgb_classifier_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "normalizer": norm}, f)

    results_path = results_dir / "xgb_classifier_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("XGB model saved to %s", model_path)
    return model_path


def load_model(models_dir: Path | None = None) -> tuple[xgb.XGBClassifier, BaselineNormalizer]:
    """Load a previously saved XGBoost model and normaliser."""
    models_dir = Path(models_dir or MODELS_DIR)
    path = models_dir / "xgb_classifier_v1.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved XGB model at {path}.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["normalizer"]
