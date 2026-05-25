"""Baseline normalisation for vehicle-agnostic fault classification.

Problem with absolute features
--------------------------------
Our RF scored 1.0 on synthetic data because it learned thresholds like
"LTFT__max > 15 % → fuel_system". That threshold is Etios-specific.
The Skoda Roomster may have a healthy LTFT baseline of +6 % (slightly
lean calibration from the factory). A model trained on absolute values
would misclassify that as a fuel-system fault before any injection starts.

Fix: z-score normalisation relative to healthy-window statistics
-----------------------------------------------------------------
We fit a StandardScaler on the healthy windows of the training set.
For each window in train/test/inference, we also output the z-scored
version of every feature: z = (x - μ_healthy) / σ_healthy.

At inference on a new vehicle (Skoda), collect 3–5 minutes of normal
driving first to re-fit the scaler. The fault classifier then reads
"LTFT is 4 σ above this vehicle's normal" — a vehicle-agnostic signal.

Output columns
--------------
For each of the 82 base features we add a ``{feature}__z`` column.
The classifier and forecasters use ONLY the 82 z-scored columns so that
absolute vehicle-specific baselines do not leak into the model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.extractor import feature_names

_FEAT_COLS = feature_names()


class BaselineNormalizer:
    """Fit on healthy training windows; transform all splits.

    Usage
    -----
        norm = BaselineNormalizer()
        norm.fit(train_df)                        # uses healthy rows only
        train_norm = norm.transform(train_df)     # adds __z columns
        test_norm  = norm.transform(test_df)
        norm.save(path)                           # persist for Skoda inference
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame, healthy_label: str = "healthy") -> "BaselineNormalizer":
        """Fit the scaler on healthy windows in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training feature matrix with a ``label`` column.
        healthy_label : str
            The label string that identifies healthy windows.
        """
        healthy = df[df["label"] == healthy_label][_FEAT_COLS]
        if len(healthy) == 0:
            raise ValueError("No healthy windows found in training data to fit normaliser.")
        self._scaler = StandardScaler()
        self._scaler.fit(healthy.to_numpy(dtype=float))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with z-scored columns added.

        Z-scored columns are named ``{feature}__z`` for each base feature.
        The original absolute-value columns are preserved unchanged.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before transform().")
        X = df[_FEAT_COLS].to_numpy(dtype=float)
        X_z = self._scaler.transform(X)
        z_cols = {f"{col}__z": X_z[:, i] for i, col in enumerate(_FEAT_COLS)}
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(z_cols)], axis=1)

    def fit_transform(self, df: pd.DataFrame, healthy_label: str = "healthy") -> pd.DataFrame:
        return self.fit(df, healthy_label).transform(df)

    def save(self, path: Path | str) -> None:
        # Store feature_order alongside the scaler so a future load can detect
        # if the codebase has added/removed/reordered features since training.
        # Old pickles (raw StandardScaler) are still accepted by load() below.
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "feature_order": list(_FEAT_COLS)}, f)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineNormalizer":
        norm = cls()
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            # New format (v2): includes feature_order for compatibility check.
            saved_order = bundle.get("feature_order")
            if saved_order is not None and saved_order != list(_FEAT_COLS):
                raise RuntimeError(
                    f"Normalizer feature order mismatch: saved {len(saved_order)} features "
                    f"but current codebase has {len(_FEAT_COLS)}. "
                    f"Retrain the normalizer with the current feature set."
                )
            norm._scaler = bundle["scaler"]
        else:
            # Legacy format (v1): raw StandardScaler — accept without version check.
            # This keeps old Skoda baseline pickles working after a code update.
            norm._scaler = bundle
        return norm

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    @property
    def feature_means(self) -> "np.ndarray":
        """Healthy-window mean for each of the 82 base features.

        This is the StandardScaler's ``mean_`` vector, exposed as a public
        property so callers don't need to reach into the private ``_scaler``
        attribute.  Used by InferenceEngine to derive vehicle-specific baselines
        for the physics-based severity formulas.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before accessing feature_means.")
        # Return a copy so callers cannot mutate the scaler's internal state.
        return self._scaler.mean_.copy()


def normalised_feature_names() -> list[str]:
    """82 z-scored feature names only (raw absolute features excluded).

    Raw features are dropped so the classifier never sees vehicle-specific
    absolute baselines — only deviations from each vehicle's own healthy
    distribution.  This is the key to cross-vehicle generalisation.
    """
    return [f"{c}__z" for c in _FEAT_COLS]
