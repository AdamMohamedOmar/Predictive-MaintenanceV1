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
We fit a StandardScaler on the CONTINUOUS healthy-window features.
For each window in train/test/inference, we also output the z-scored
version of every continuous feature: z = (x - μ_healthy) / σ_healthy.

At inference on a new vehicle (Skoda), collect 3–5 minutes of normal
driving first to re-fit the scaler. The fault classifier then reads
"LTFT is 4 σ above this vehicle's normal" — a vehicle-agnostic signal.

Regime one-hots excluded from StandardScaler
---------------------------------------------
The five REGIME__* columns are binary {0, 1}.  Z-scoring them creates
near-constant columns on the training vehicle (e.g., REGIME__CRUISE ≈ 0.8
on the Etios highway data) that become large-magnitude spikes on the Skoda
(where cruise share may differ).  The model treats those spikes as anomalies
even during healthy driving — a soft cross-vehicle transfer failure.
Fix: copy regime columns verbatim into their ``__z`` slots; StandardScaler
is fitted on the remaining 78 continuous features only.

Output columns
--------------
For each of the 83 base features we add a ``{feature}__z`` column.
The classifier and forecasters use ONLY the 83 z-scored columns so that
absolute vehicle-specific baselines do not leak into the model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.extractor import feature_names
from src.features.regime import regime_feature_names

_FEAT_COLS = feature_names()

# Regime one-hots are binary {0, 1} — excluded from StandardScaler to prevent
# cross-vehicle distribution shifts from being mistaken for fault signals.
_REGIME_COLS: set[str] = set(regime_feature_names())
_CONTINUOUS_COLS: list[str] = [c for c in _FEAT_COLS if c not in _REGIME_COLS]
_REGIME_COL_LIST: list[str] = [c for c in _FEAT_COLS if c in _REGIME_COLS]


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
        # Healthy-window means of regime one-hot flags, stored at fit() time
        # so that feature_means() can reconstruct the full 83-element vector.
        self._regime_means: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, healthy_label: str = "healthy") -> "BaselineNormalizer":
        """Fit the scaler on healthy windows in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training feature matrix with a ``label`` column.
        healthy_label : str
            The label string that identifies healthy windows.
        """
        healthy = df[df["label"] == healthy_label]
        if len(healthy) == 0:
            raise ValueError("No healthy windows found in training data to fit normaliser.")
        # Fit on continuous features only — regime one-hots are binary {0, 1}
        # and z-scoring them creates distribution-shift artefacts on the Skoda.
        self._scaler = StandardScaler()
        self._scaler.fit(healthy[_CONTINUOUS_COLS].to_numpy(dtype=float))
        self._regime_means = healthy[_REGIME_COL_LIST].mean().to_numpy(dtype=float)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with z-scored columns added.

        Z-scored columns are named ``{feature}__z`` for each base feature.
        Continuous features are z-scored; regime one-hots are copied verbatim
        into their __z slots (binary encoding; z-scoring creates false spikes).
        The original absolute-value columns are preserved unchanged.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before transform().")

        if self._scaler.n_features_in_ == len(_FEAT_COLS):
            # Legacy scaler fitted on all features (pre-T5.1 artefact) — preserve
            # original behaviour so old pickles keep working until the retrain.
            X = df[_FEAT_COLS].to_numpy(dtype=float)
            X_z = self._scaler.transform(X)
            z_cols = {f"{col}__z": X_z[:, i] for i, col in enumerate(_FEAT_COLS)}
        else:
            # Post-T5.1: z-score continuous features; copy regime flags as-is.
            X_cont = df[_CONTINUOUS_COLS].to_numpy(dtype=float)
            X_z_cont = self._scaler.transform(X_cont)
            z_cols = {f"{col}__z": X_z_cont[:, i] for i, col in enumerate(_CONTINUOUS_COLS)}
            for rcol in _REGIME_COL_LIST:
                # Regime one-hots pass through verbatim — getattr guards against
                # old pickles that were serialised before this column existed.
                z_cols[f"{rcol}__z"] = df[rcol].to_numpy(dtype=float)

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(z_cols)], axis=1)

    def fit_transform(self, df: pd.DataFrame, healthy_label: str = "healthy") -> pd.DataFrame:
        return self.fit(df, healthy_label).transform(df)

    def save(self, path: Path | str) -> None:
        # Store feature_order alongside the scaler so a future load can detect
        # if the codebase has added/removed/reordered features since training.
        # Old pickles (raw StandardScaler) are still accepted by load() below.
        bundle: dict = {"scaler": self._scaler, "feature_order": list(_FEAT_COLS)}
        if self._regime_means is not None:
            bundle["regime_means"] = self._regime_means
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineNormalizer":
        norm = cls()
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            # New format (v2+): includes feature_order for compatibility check.
            saved_order = bundle.get("feature_order")
            if saved_order is not None and saved_order != list(_FEAT_COLS):
                raise RuntimeError(
                    f"Normalizer feature order mismatch: saved {len(saved_order)} features "
                    f"but current codebase has {len(_FEAT_COLS)}. "
                    f"Retrain the normalizer with the current feature set."
                )
            norm._scaler = bundle["scaler"]
            norm._regime_means = bundle.get("regime_means")  # None for pre-T5.1 artefacts
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
        """Healthy-window mean for each of the 83 base features, in canonical order.

        For continuous features this is the StandardScaler's ``mean_`` vector.
        For regime one-hot flags this is the observed fraction of healthy windows
        in each regime (stored at fit() time).

        Exposed as a property so InferenceEngine can derive vehicle-specific
        baselines for the physics-based severity formulas without reaching into
        the private ``_scaler`` attribute.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before accessing feature_means.")

        if self._scaler.n_features_in_ == len(_FEAT_COLS):
            # Legacy scaler fitted on all features — return means directly.
            return self._scaler.mean_.copy()

        # Post-T5.1: reconstruct full 83-element vector in canonical feature order.
        # Continuous features → from the fitted scaler.
        # Regime flags → observed healthy-window means (or 0.5 fallback for
        # artefacts loaded from disk before _regime_means was stored).
        # getattr guard: old pickles (serialised as bundle["normalizer"] inside
        # xgb_classifier_v1.pkl) bypass __init__ and lack _regime_means entirely.
        stored = getattr(self, "_regime_means", None)
        regime_fallback = (
            stored if stored is not None else np.full(len(_REGIME_COL_LIST), 0.5)
        )
        cont_map = dict(zip(_CONTINUOUS_COLS, self._scaler.mean_))
        regime_map = dict(zip(_REGIME_COL_LIST, regime_fallback))
        all_means = {**cont_map, **regime_map}
        return np.array([all_means[f] for f in _FEAT_COLS])


def normalised_feature_names() -> list[str]:
    """83 z-scored feature names only (raw absolute features excluded).

    Raw features are dropped so the classifier never sees vehicle-specific
    absolute baselines — only deviations from each vehicle's own healthy
    distribution.  This is the key to cross-vehicle generalisation.
    """
    return [f"{c}__z" for c in _FEAT_COLS]
