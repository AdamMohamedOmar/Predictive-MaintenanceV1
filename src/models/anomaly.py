"""One-class anomaly detector — IsolationForest on healthy z-scored windows.

Why a one-class detector
------------------------
The team has ~5 hours of healthy carOBD data and zero real-fault labels.
A six-class supervised classifier addresses "which of the known faults
is this" — but it can only be trained against synthetic injector outputs,
producing the self-consistency-floor problem documented in the project
root README. A one-class detector asks a different, logically cleaner
question: "does this window look like the healthy data we trained on?"

At fit time only healthy windows are used; no fault labels enter the
model. At inference, any window that lies outside the learned healthy
manifold scores high — regardless of which physical fault drove it
out of distribution. The detector is therefore a model-agnostic fault
sentinel, complementing the classifier's per-class identification.

Score range
-----------
``score()`` returns a value in [0, 1]. The mapping is calibrated at
fit time using the 5th and 95th percentiles of raw IsolationForest
``decision_function`` scores on the healthy training set:
  - score = 0.0 ↔ raw at or below the healthy 5th percentile (very
    much like healthy data)
  - score = 1.0 ↔ raw at or above the healthy 95th percentile (already
    unusual *for healthy data*); real anomalies push well beyond
  - in between: linear interpolation

The [0, 1] range matches the existing severity-gauge convention so
the dashboard can render the anomaly score with the same widget.

Not a real-fault validation
---------------------------
A working separation between healthy and injector-fault windows
in unit tests is a sanity check, not a real-fault detection metric.
Real-fault validation is performed against data collected per
docs/REAL_FAULT_COLLECTION.md (Step 5 of the honest-framing series).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RANDOM_SEED
from src.features.normalizer import BaselineNormalizer, normalised_feature_names


# A curated fault-bearing feature subset (P1-4): the dimensions that actually
# move under the four faults — trims, MAP, warm-up rate, idle RPM drift, TPS.
# Most of the 83 z-features are healthy-invariant and only dilute these few
# discriminative axes inside the IsolationForest.  Pass as ``feature_subset``
# to score on these alone.
FAULT_BEARING_FEATURES: list[str] = [
    "LONG_TERM_FUEL_TRIM_BANK_1__mean",
    "SHORT_TERM_FUEL_TRIM_BANK_1__mean",
    "FUEL_TRIM_DIVERGENCE",
    "INTAKE_MANIFOLD_PRESSURE__mean",
    "MAP_PER_THROTTLE",
    "COOLANT_TEMPERATURE__mean",
    "COOLANT_WARMUP_RATE",
    "RPM_IDLE_DRIFT",
    "ENGINE_RPM__mean",
    "THROTTLE_TO_PEDAL_RATIO",
    "THROTTLE_CMD_ACTUAL_DELTA",
    "TIMING_VS_TEMP",
]


class AnomalyDetector:
    """IsolationForest with an FPR-budget-calibrated [0, 1] score map.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees. 200 gives stable scores at small cost.
    contamination : float or "auto"
        Expected outlier fraction in the training data. "auto" lets sklearn
        pick a sensible threshold. We don't use the binary outlier flag —
        only the continuous decision_function — so this mainly affects
        sklearn's internal offset, not our normalized score.
    random_seed : int
        Tree-construction RNG seed.
    fpr_budget : float
        Design false-positive rate (P1-4). ``score = 1.0`` is mapped to the
        ``(1 − fpr_budget)`` healthy percentile, so by construction only this
        fraction of healthy windows max out the score. The old code mapped the
        95th percentile to 1.0, guaranteeing ~5 % healthy false alarms; the
        IsolationForest score distribution is non-Gaussian, so a percentile
        budget (not a σ rule) is the correct calibration.
    feature_subset : list[str] or None
        Base feature names to score on. None → all 83. Pass
        ``FAULT_BEARING_FEATURES`` to concentrate on discriminative axes.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: Union[float, str] = "auto",
        random_seed: int = RANDOM_SEED,
        fpr_budget: float = 0.01,
        feature_subset: list[str] | None = None,
    ) -> None:
        self._model: IsolationForest | None = None
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._random_seed = random_seed
        self._fpr_budget = fpr_budget
        self._feature_subset = feature_subset
        self._score_lo: float | None = None
        self._score_hi: float | None = None

    def _z_cols(self) -> list[str]:
        """Resolve the z-scored columns to score on (subset or all)."""
        if self._feature_subset is None:
            return normalised_feature_names()
        return [f"{c}__z" for c in self._feature_subset]

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        norm: BaselineNormalizer,
        *,
        healthy_label: str = "healthy",
    ) -> "AnomalyDetector":
        """Fit on the healthy windows in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe with a ``label`` column. Rows where
            ``label == healthy_label`` train the model; others are ignored.
        norm : BaselineNormalizer
            Already-fitted normalizer. Its ``transform`` is used to
            z-score features before fitting.
        """
        healthy_df = df[df["label"] == healthy_label]
        if len(healthy_df) == 0:
            raise ValueError(
                "No healthy windows in training data — cannot fit anomaly detector."
            )

        norm_df = norm.transform(healthy_df)
        z_cols = self._z_cols()
        X = norm_df[z_cols].to_numpy(dtype=float)

        # Held-out calibration split: 80 % fit, 20 % calibrate. Scoring the
        # training points themselves gives near-zero raw values by construction
        # (the trees were grown on them), so percentiles taken there don't
        # represent the true healthy-score distribution. The held-out 20 %
        # is what an unseen healthy window from another session looks like.
        rng = np.random.default_rng(self._random_seed)
        perm = rng.permutation(len(X))
        split = max(1, int(0.8 * len(X)))
        fit_idx = perm[:split]
        cal_idx = perm[split:] if len(perm) > split else perm[:split]

        # n_jobs=1 (not -1): joblib's CPU-counting via psutil is broken on
        # some Windows venvs (AttributeError: module 'psutil' has no
        # attribute 'Process'). IsolationForest finishes in < 1 s anyway.
        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_seed,
            n_jobs=1,
        )
        self._model.fit(X[fit_idx])

        # Negate decision_function so higher = more anomalous, then calibrate
        # from a FALSE-ALARM BUDGET (P1-4): anchor 0.0 at the healthy median
        # (clearly normal) and 1.0 at the (1 − fpr_budget) healthy percentile,
        # so only ~fpr_budget of healthy windows reach the alarm ceiling.
        raw_cal = -self._model.decision_function(X[cal_idx])
        self._score_lo = float(np.percentile(raw_cal, 50))
        self._score_hi = float(np.percentile(raw_cal, 100.0 * (1.0 - self._fpr_budget)))
        return self

    # ── Score (single & batch) ───────────────────────────────────────────

    def score(self, features: dict[str, float], norm: BaselineNormalizer) -> float:
        """Return anomaly score in [0, 1] for one window's features.

        Parameters
        ----------
        features : dict
            Raw (un-z-scored) features from ``extract_features``.
        norm : BaselineNormalizer
            Fitted normalizer used for z-scoring.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        row_df = pd.DataFrame([features])
        return float(self.score_batch(row_df, norm)[0])

    def score_batch(
        self,
        features_df: pd.DataFrame,
        norm: BaselineNormalizer,
    ) -> np.ndarray:
        """Vectorized scoring — used for evaluation and dashboards.

        Parameters
        ----------
        features_df : pd.DataFrame
            Raw feature rows (one per window). Must contain all 83
            base feature columns expected by ``BaselineNormalizer``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before score_batch().")
        if self._score_lo is None or self._score_hi is None:
            raise RuntimeError("Detector calibration missing — refit.")

        norm_df = norm.transform(features_df)
        X = norm_df[self._z_cols()].to_numpy(dtype=float)
        raw = -self._model.decision_function(X)
        span = self._score_hi - self._score_lo
        if span <= 0:
            # Degenerate calibration (all percentiles equal) — neutral score.
            return np.full(len(features_df), 0.5)
        return np.clip((raw - self._score_lo) / span, 0.0, 1.0)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the trained detector to disk.

        Stores the fitted IsolationForest, calibration percentiles, and
        the constructor arguments so ``load()`` can rebuild an identical
        detector.
        """
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted AnomalyDetector.")
        bundle: dict = {
            "model": self._model,
            "score_lo": self._score_lo,
            "score_hi": self._score_hi,
            "n_estimators": self._n_estimators,
            "contamination": self._contamination,
            "random_seed": self._random_seed,
            "fpr_budget": self._fpr_budget,
            "feature_subset": self._feature_subset,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path: Path | str) -> "AnomalyDetector":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        det = cls(
            n_estimators=bundle["n_estimators"],
            contamination=bundle["contamination"],
            random_seed=bundle["random_seed"],
            fpr_budget=bundle.get("fpr_budget", 0.01),
            feature_subset=bundle.get("feature_subset"),
        )
        det._model = bundle["model"]
        det._score_lo = bundle["score_lo"]
        det._score_hi = bundle["score_hi"]
        return det

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
