"""Random Forest fault classifier — Week 3 baseline.

Session-level train/test split
--------------------------------
We split by session_id, NOT by row or window index. This is the most
important implementation detail in the entire pipeline.

Why it matters: windows from the same session overlap heavily (60-row
windows, 10-row stride → adjacent windows share 50 of 60 rows). If we
split randomly by row, the test set leaks training data. The model would
memorise the session rather than learn the fault signature.

Splitting by session means the model has never seen ANY windows from the
test sessions — it must generalise to a new trip, which is exactly what
it needs to do on the Skoda Roomster in production.

With 9 sessions, we use 7 for training and 2 for testing (≈ 78/22 split).
The 2 held-out sessions are chosen to include the one "drive" session
(drive1.csv) so the test set contains the more varied driving regime.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.features.dataset_builder import FAULT_TYPES, LABEL_TO_ID
from src.features.extractor import feature_names

log = logging.getLogger(__name__)

# The "drive" session is the only highway-driving sample; always hold it out
# so the test set isn't just commute-mode windows.
_HELD_OUT_SESSIONS = {"drive1", "live12"}

ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
# cold_start is appended last so existing fault-class indices (0-4) are unchanged
ALL_LABELS = ["healthy"] + FAULT_TYPES + ["cold_start"]


def session_split(
    dataset: pd.DataFrame,
    held_out: set[str] = _HELD_OUT_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/test by session_id.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full feature dataset from ``build_dataset``.
    held_out : set[str]
        Session IDs to reserve for testing.

    Returns
    -------
    (train_df, test_df)
    """
    test_mask = dataset["session_id"].isin(held_out)
    return dataset[~test_mask].copy(), dataset[test_mask].copy()


def train(
    train_df: pd.DataFrame,
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_seed: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """Fit a Random Forest on the training split.

    class_weight='balanced' compensates for the healthy class being ~2×
    larger than each fault class without requiring manual resampling.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training rows from ``session_split``.
    n_estimators : int
        Number of trees. 300 is enough to stabilise variance on this dataset
        without making inference too slow for the dashboard.
    max_depth : int or None
        None = grow until pure leaves (default). Set a value if overfitting
        appears in the Week 4 XGBoost comparison.
    random_seed : int

    Returns
    -------
    Fitted RandomForestClassifier.
    """
    feat_cols = feature_names()
    X = train_df[feat_cols].to_numpy(dtype=float)
    y = train_df["label_id"].to_numpy(dtype=int)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    log.info(
        "Trained RF: %d trees, %d train samples, %d features",
        n_estimators,
        len(X),
        len(feat_cols),
    )
    return clf


def evaluate(
    clf: RandomForestClassifier,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate the classifier on the test split.

    Returns a results dict containing:
      - macro_f1       : the headline metric for the charter target (≥ 0.80)
      - classification_report : per-class precision/recall/F1
      - confusion_matrix : raw counts as a nested list

    Parameters
    ----------
    clf : RandomForestClassifier
    test_df : pd.DataFrame
        Test rows from ``session_split``.

    Returns
    -------
    dict with keys: macro_f1, per_class, confusion_matrix, test_sessions
    """
    feat_cols = feature_names()
    X_test = test_df[feat_cols].to_numpy(dtype=float)
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
    log.info("Test macro-F1: %.4f", macro_f1)

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


def save_model(clf: RandomForestClassifier, results: dict, models_dir: Path | None = None, results_dir: Path | None = None) -> Path:
    """Persist the fitted model and evaluation results to disk.

    Returns the path of the saved model file.
    """
    models_dir = Path(models_dir or MODELS_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "rf_classifier_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    results_path = results_dir / "rf_classifier_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Model saved to %s", model_path)
    log.info("Results saved to %s", results_path)
    return model_path


def load_model(models_dir: Path | None = None) -> RandomForestClassifier:
    """Load a previously saved Random Forest model."""
    models_dir = Path(models_dir or MODELS_DIR)
    model_path = models_dir / "rf_classifier_v1.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Run train() first."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def top_features(clf: RandomForestClassifier, n: int = 20) -> pd.DataFrame:
    """Return the top-n most important features by mean decrease in impurity.

    This is a quick sanity check: the cross-PID ratio features
    (THROTTLE_TO_PEDAL_RATIO, MAP_PER_THROTTLE, FUEL_TRIM_DIVERGENCE)
    should rank near the top if the injection signatures are working.
    """
    names = feature_names()
    importances = clf.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.nlargest(n, "importance").reset_index(drop=True)
