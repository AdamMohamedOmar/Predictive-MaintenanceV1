"""SHAP-based explainability for the XGBoost fault classifier.

What SHAP does (the mechanic's version)
-----------------------------------------
Imagine the classifier is a jury of 300 decision trees. SHAP asks: "For
this specific prediction, how much did each feature shift the verdict?"
Features that pushed toward the predicted class get positive SHAP values;
features that argued against it get negative values. The sum of all SHAP
values equals the log-odds of the prediction.

For our paper: we can show "Window #142 was classified as fuel_system
because LONG_TERM_FUEL_TRIM_BANK_1__z = +3.8σ (SHAP +0.91), while
MAP_PER_THROTTLE was normal (SHAP ≈ 0.0)." That distinguishes our
approach from a black-box model.

We use ``shap.TreeExplainer`` which computes exact SHAP values for
tree-based models (no sampling approximation needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.models.classifier import ALL_LABELS

_FEAT_COLS = normalised_feature_names()


class SHAPExplainer:
    """Wraps shap.TreeExplainer for the XGBoost fault classifier.

    Usage
    -----
        explainer = SHAPExplainer(clf)

        # Explain a single window (one row DataFrame or 1-D array):
        result = explainer.explain_window(window_features, norm)
        print(result["predicted_label"])   # "fuel_system"
        print(result["top_features"])      # list of (feature, shap_value) pairs

        # Global feature importance across a whole split:
        summary = explainer.global_summary(X_norm, top_n=15)
    """

    def __init__(self, clf: xgb.XGBClassifier) -> None:
        self._clf = clf
        self._explainer = shap.TreeExplainer(clf)

    def explain_window(
        self,
        window_features: pd.DataFrame | np.ndarray,
        norm: BaselineNormalizer,
        top_n: int = 10,
    ) -> dict:
        """Return a human-readable explanation for one window.

        Parameters
        ----------
        window_features : pd.DataFrame (1 row) or 1-D ndarray
            The raw (un-normalised) feature vector for one 60-row window,
            as produced by ``extract_features``.
        norm : BaselineNormalizer
            The normaliser from the XGBoost training bundle.
        top_n : int
            How many top features to include in the explanation.

        Returns
        -------
        dict with keys:
          predicted_label  — the fault class name
          predicted_id     — the integer class id
          probabilities    — {label: probability} for all 6 classes
          top_features     — list of (feature_name, shap_value) sorted by |shap|
          shap_values      — full array of shape (n_features,) for predicted class
        """
        if isinstance(window_features, np.ndarray):
            window_features = pd.DataFrame([window_features], columns=_FEAT_COLS[:len(window_features)])

        # Normalise the single window using the training baseline
        row_norm = norm.transform(window_features)
        X = row_norm[_FEAT_COLS].to_numpy(dtype=float)  # shape (1, n_feats)

        # Probabilities from the original XGBoost model
        proba = self._clf.predict_proba(X)[0]  # shape (n_classes,)
        pred_id = int(np.argmax(proba))
        pred_label = ALL_LABELS[pred_id]

        # SHAP values — normalise to (n_samples, n_features, n_classes)
        sv_for_pred = self._shap_for_class(X, pred_id)[0]  # shape (n_features,)

        indices = np.argsort(np.abs(sv_for_pred))[::-1][:top_n]
        top = [(str(_FEAT_COLS[i]), float(sv_for_pred[i])) for i in indices]

        return {
            "predicted_label": pred_label,
            "predicted_id": pred_id,
            "probabilities": {label: float(proba[i]) for i, label in enumerate(ALL_LABELS)},
            "top_features": top,
            "shap_values": sv_for_pred,
        }

    def global_summary(
        self,
        X_norm: np.ndarray,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Mean absolute SHAP across all samples and classes.

        This is the "global" feature importance: which features
        matter most across the whole test set, regardless of class.

        Parameters
        ----------
        X_norm : np.ndarray
            Normalised feature matrix, shape (n_samples, n_features).
        top_n : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame with columns [feature, mean_abs_shap], sorted descending.
        """
        # Stack to (n_samples, n_features, n_classes), take mean abs across both axes
        # mean absolute SHAP across all classes and samples → (n_features,)
        stacked = self._shap_3d(X_norm)  # (n_samples, n_features, n_classes)
        mean_abs = np.abs(stacked).mean(axis=(0, 2))

        df = pd.DataFrame({"feature": _FEAT_COLS, "mean_abs_shap": mean_abs})
        return df.nlargest(top_n, "mean_abs_shap").reset_index(drop=True)

    def class_summary(
        self,
        X_norm: np.ndarray,
        class_label: str,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """Mean absolute SHAP for one specific fault class.

        Use this to answer: "which features are most responsible for
        the model deciding something IS a fuel_system fault?"
        """
        class_id = ALL_LABELS.index(class_label)
        sv = np.abs(self._shap_for_class(X_norm, class_id)).mean(axis=0)  # (n_features,)
        df = pd.DataFrame({"feature": _FEAT_COLS, "mean_abs_shap": sv})
        return df.nlargest(top_n, "mean_abs_shap").reset_index(drop=True)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _shap_3d(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values as (n_samples, n_features, n_classes) regardless
        of whether TreeExplainer returns a list or a 3D array."""
        raw = self._explainer.shap_values(X)
        if isinstance(raw, list):
            # list of n_classes arrays each (n_samples, n_features)
            return np.stack(raw, axis=2)
        # already (n_samples, n_features, n_classes)
        return raw

    def _shap_for_class(self, X: np.ndarray, class_id: int) -> np.ndarray:
        """Return SHAP values for one class: shape (n_samples, n_features)."""
        return self._shap_3d(X)[:, :, class_id]
