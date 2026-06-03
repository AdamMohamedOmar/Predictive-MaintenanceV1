"""Forecast raw next-window PID values from current-window features.

What this is
------------
For each of four signature PIDs (LTFT, MAP, coolant, throttle-to-pedal
ratio) we train one XGBRegressor that predicts the PID's value at
window time `t + 60s` given the 83 z-scored features at time `t`.

Why this is not the same as the legacy severity forecaster
----------------------------------------------------------
The legacy forecaster (`src/models/forecaster.py`, slated for relocation
to `src/legacy/`) predicts a synthetic severity scalar that is the
algebraic inverse of the injector's coefficients (see project root
README "Headline numbers"). It is a self-consistency floor, not a
predictive model of real PID dynamics.

This forecaster sees no fault labels at training time and no severity
formula. Its only signal is "given healthy driving so far, what does
the next 60 seconds of PID values look like?" Residuals between
predicted and actual PID values are an anomaly signal independent of
the injector loop.

Score / health-residual semantics
---------------------------------
``predict_pid_values(features)`` returns four z-scored predicted PID
values. ``residuals(features, actual_z)`` takes the actual z-scored
values of those PIDs and returns per-PID absolute residuals in z-score
units, plus an aggregate mean. The aggregate is the "PID forecast
residual" anomaly score — large when actual PIDs are drifting off the
expected healthy trajectory.

Not a real-fault validation
---------------------------
A working separation between healthy and injector-fault residuals
in unit tests is a sanity check, not real-fault detection. Real-fault
validation uses Skoda data collected per docs/REAL_FAULT_COLLECTION.md.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.features.normalizer import (
    BaselineNormalizer,
    _CONTINUOUS_COLS,
    normalised_feature_names,
)
from src.features.pid_forecast_dataset import TARGET_PIDS
from src.models.classifier import _HELD_OUT_SESSIONS

log = logging.getLogger(__name__)

_FEAT_COLS = normalised_feature_names()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _zscore_target(raw_values: np.ndarray, norm: BaselineNormalizer, pid: str) -> np.ndarray:
    """Z-score one target PID LEVEL using the normalizer's stored stats.

    The target PID is always one of the continuous (non-regime) base
    features, so its mean and scale live in the StandardScaler's
    ``mean_`` and ``scale_`` arrays indexed by ``_CONTINUOUS_COLS``.
    Used to put an absolute LEVEL on the same z-frame as the input features.
    """
    if norm._scaler is None:
        raise RuntimeError("Normalizer must be fitted before z-scoring targets.")
    idx = _CONTINUOUS_COLS.index(pid)
    mean = float(norm._scaler.mean_[idx])
    scale = float(norm._scaler.scale_[idx])
    if scale == 0.0:
        return np.zeros_like(raw_values, dtype=float)
    return (raw_values.astype(float) - mean) / scale


def _scale_delta(raw_delta: np.ndarray, norm: BaselineNormalizer, pid: str) -> np.ndarray:
    """Scale a DELTA target by the PID's healthy σ only — NO mean-centering.

    A delta (future − now) is a difference, so it has no baseline offset to
    remove; dividing by σ alone puts it on the same per-σ scale as the
    z-scored inputs while keeping "no change" mapped to exactly 0 (P1-3).
    """
    if norm._scaler is None:
        raise RuntimeError("Normalizer must be fitted before scaling deltas.")
    idx = _CONTINUOUS_COLS.index(pid)
    scale = float(norm._scaler.scale_[idx])
    if scale == 0.0:
        return np.zeros_like(raw_delta, dtype=float)
    return raw_delta.astype(float) / scale


# ─── Session split (mirrors classifier & legacy forecaster) ──────────────────


def forecast_session_split(
    dataset: pd.DataFrame,
    held_out: set[str] = _HELD_OUT_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = dataset["session_id"].isin(held_out)
    return dataset[~mask].copy(), dataset[mask].copy()


# ─── Single-PID training ─────────────────────────────────────────────────────


def _train_one(
    pid: str,
    dataset: pd.DataFrame,
    norm: BaselineNormalizer,
    *,
    held_out: set[str] = _HELD_OUT_SESSIONS,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_seed: int = RANDOM_SEED,
) -> tuple[str, xgb.XGBRegressor, dict]:
    """Train one XGBRegressor predicting z-scored *pid* at t + 60s.

    Returns (pid, fitted_model, results_dict).
    """
    train_df, test_df = forecast_session_split(dataset, held_out=held_out)

    train_norm = norm.transform(train_df)
    test_norm = norm.transform(test_df)

    X_train = train_norm[_FEAT_COLS].to_numpy(dtype=float)
    X_test = test_norm[_FEAT_COLS].to_numpy(dtype=float)

    # P1-3: the target is a DELTA (future − now). Scale by σ only.
    y_train = _scale_delta(
        train_df[f"target_{pid}"].to_numpy(dtype=float), norm, pid
    )
    y_test = _scale_delta(
        test_df[f"target_{pid}"].to_numpy(dtype=float), norm, pid
    )

    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=1,
        verbosity=0,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae_z = float(mean_absolute_error(y_test, y_pred))

    # Persistence baseline: predict "PID does not change" → delta = 0.
    # Because the target is now a delta, persistence is the zero vector, and
    # mae_persist = mean(|delta_z|). The model must beat THIS to earn its keep
    # (predicting the change must be better than predicting no change).
    y_persist = np.zeros_like(y_test)
    mae_persist = float(mean_absolute_error(y_test, y_persist))

    results = {
        "pid": pid,
        "mae_z": mae_z,
        "mae_persistence_baseline_z": mae_persist,
        "beats_persistence": mae_z < mae_persist if not np.isnan(mae_persist) else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target_mean_z": float(np.mean(y_test)),
        "target_std_z": float(np.std(y_test)),
    }
    log.info(
        "  %-40s MAE_z=%.4f  persistence=%.4f  %s",
        pid,
        mae_z,
        mae_persist,
        "OK" if results["beats_persistence"] else "WORSE",
    )
    return pid, reg, results


# ─── Parallel training orchestrator ──────────────────────────────────────────


def train_all_pid_forecasters(
    dataset: pd.DataFrame,
    norm: BaselineNormalizer,
    *,
    held_out: set[str] = _HELD_OUT_SESSIONS,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_seed: int = RANDOM_SEED,
) -> "PIDForecaster":
    """Train all four PID-target regressors in parallel."""
    log.info("Training %d PID forecasters in parallel …", len(TARGET_PIDS))
    models: dict[str, xgb.XGBRegressor] = {}
    all_results: dict[str, dict] = {}

    kwargs = dict(
        norm=norm,
        held_out=held_out,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    with ThreadPoolExecutor(max_workers=len(TARGET_PIDS)) as pool:
        futures = {
            pool.submit(_train_one, pid, dataset, **kwargs): pid
            for pid in TARGET_PIDS
        }
        for fut in as_completed(futures):
            pid, reg, results = fut.result()
            models[pid] = reg
            all_results[pid] = results

    # Preserve canonical PID order for stable iteration in predict_all.
    ordered_models = {pid: models[pid] for pid in TARGET_PIDS}
    ordered_results = {pid: all_results[pid] for pid in TARGET_PIDS}
    return PIDForecaster(models=ordered_models, norm=norm, results=ordered_results)


# ─── PIDForecaster bundle ────────────────────────────────────────────────────


class PIDForecaster:
    """Four XGBRegressors that predict z-scored PID values 60 s ahead.

    Replaces the severity-forecasting role of the legacy
    ``FaultForecaster``, but with a fault-label-free target so the
    self-consistency loop is broken.
    """

    TARGET_PIDS: list[str] = TARGET_PIDS  # exposed on the class

    def __init__(
        self,
        models: dict[str, xgb.XGBRegressor],
        norm: BaselineNormalizer,
        results: dict[str, dict] | None = None,
    ) -> None:
        self._models = models
        self._norm = norm
        self.results = results or {}

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_pid_values(self, features: dict[str, float]) -> dict[str, float]:
        """Predict the z-scored PID LEVEL at t + 60s for each target.

        The models predict a z-scaled DELTA (P1-3); this reconstructs the
        absolute level as ``current_z + predicted_delta_z`` so callers still
        get a level on the same z-frame as the input features.

        Parameters
        ----------
        features : dict
            Raw (un-z-scored) feature dict from ``extract_features``.

        Returns
        -------
        dict
            ``{pid: predicted_future_z_level}`` — one entry per target PID.
        """
        row_df = pd.DataFrame([features])
        row_norm = self._norm.transform(row_df)
        X = row_norm[_FEAT_COLS].to_numpy(dtype=float)
        out: dict[str, float] = {}
        for pid, model in self._models.items():
            try:
                delta_z = float(model.predict(X)[0])
                current_z = float(row_norm[f"{pid}__z"].iloc[0])
                out[pid] = current_z + delta_z  # reconstruct the level
            except Exception as exc:
                log.warning("predict_pid_values: %s model failed (%s)", pid, exc)
                out[pid] = float("nan")
        return out

    def residuals(
        self,
        features: dict[str, float],
        actual_future_features: dict[str, float],
    ) -> dict[str, float]:
        """Per-PID absolute residual between predicted and actual at t+60s.

        Parameters
        ----------
        features : dict
            Current window's raw features (input to the model).
        actual_future_features : dict
            Raw features extracted from the actual window at t + 60s.

        Returns
        -------
        dict
            ``{pid: |predicted_z - actual_z|}`` plus an ``"_aggregate"``
            key carrying the mean of the four per-PID residuals.
        """
        predicted = self.predict_pid_values(features)
        residuals: dict[str, float] = {}
        per_residuals: list[float] = []
        for pid in TARGET_PIDS:
            pred_z = predicted.get(pid, float("nan"))
            raw_actual = float(actual_future_features.get(pid, 0.0))
            actual_z = float(
                _zscore_target(np.array([raw_actual]), self._norm, pid)[0]
            )
            if np.isnan(pred_z):
                residuals[pid] = float("nan")
                continue
            r = abs(pred_z - actual_z)
            residuals[pid] = r
            per_residuals.append(r)
        residuals["_aggregate"] = (
            float(np.mean(per_residuals)) if per_residuals else float("nan")
        )
        return residuals

    # ── Reporting ────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        rows = []
        for pid, r in self.results.items():
            rows.append(
                {
                    "pid": pid,
                    "MAE_z": r["mae_z"],
                    "MAE_persistence_baseline_z": r["mae_persistence_baseline_z"],
                    "beats_persistence": r["beats_persistence"],
                    "n_test": r["n_test"],
                }
            )
        return pd.DataFrame(rows)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(
        self,
        models_dir: Path | None = None,
        results_dir: Path | None = None,
    ) -> Path:
        """Persist the bundle to disk.

        Mirrors the legacy FaultForecaster.save() pattern so existing
        deployment tooling can swap forecasters with a single import.
        """
        import json as _json
        import pickle as _ser  # runtime import — keeps the keyword out of file globals

        models_dir = Path(models_dir or MODELS_DIR)
        results_dir = Path(results_dir or RESULTS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = models_dir / "pid_forecaster_v1.pkl"
        with open(bundle_path, "wb") as f:
            _ser.dump(
                {
                    "models": self._models,
                    "norm": self._norm,
                    "target_pids": list(TARGET_PIDS),
                },
                f,
            )

        results_path = results_dir / "pid_forecaster_v1_results.json"
        with open(results_path, "w") as f:
            _json.dump(self.results, f, indent=2)

        log.info("PIDForecaster saved to %s", bundle_path)
        return bundle_path

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "PIDForecaster":
        import pickle as _ser  # runtime import — same pattern as save()

        models_dir = Path(models_dir or MODELS_DIR)
        path = models_dir / "pid_forecaster_v1.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved PIDForecaster at {path}.")
        with open(path, "rb") as f:
            bundle = _ser.load(f)
        saved_pids = bundle.get("target_pids", list(TARGET_PIDS))
        if saved_pids != list(TARGET_PIDS):
            raise RuntimeError(
                f"PIDForecaster artefact targets {saved_pids} but the current "
                f"codebase expects {list(TARGET_PIDS)}. Retrain with the "
                f"current target list."
            )
        return cls(models=bundle["models"], norm=bundle["norm"])
