"""Build the 60-second-ahead PID-value forecasting dataset.

Why this exists separately from forecast_dataset.py
---------------------------------------------------
The legacy forecast dataset (`src/features/forecast_dataset.py`) targets
the *severity scalar* computed by `src/features/severity.py` — which is
the algebraic inverse of the injector's own coefficients (see project
root README "Headline numbers"). That makes the forecaster a self-
consistency floor, not a predictive model of physical reality.

This dataset targets *raw next-window PID values*. The forecaster sees
no fault labels at training time and no derived severity. It only learns
"given the last 60 seconds of healthy driving, what will the LTFT / MAP /
coolant / TPS ratio be 60 seconds from now?" — a clean signal of normal
trajectory dynamics.

At inference, the *residual* between predicted and actual PID values is
an anomaly signal: a real fault perturbs the PID off its expected
trajectory, no matter which physical mechanism caused it.

Inputs
------
* Healthy windows from the usable carOBD files (no injection).
* For each window at time `t`, the target is the same window's PID values
  at time `t + FORECAST_HORIZON_S` (60 s later).

Output columns
--------------
* All 83 base feature columns (same as the classifier dataset).
* ``target_{PID}`` columns — one per target PID, the **DELTA** (future − now)
  in raw units (P1-3). Forecasting the change rather than the absolute level
  cancels the per-session/per-vehicle baseline offset, so the model is not
  doomed to lose to persistence under the exact baseline shift this project
  exists to handle.
* ``session_id`` — for the session-level split.

The forecaster scales the delta target by each PID's healthy σ (from the same
BaselineNormalizer that z-scores the inputs) and reconstructs the absolute
level at inference as ``current + predicted_delta``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_CAROBD_DIR,
    DATA_SYNTHETIC_DIR,
    FORECAST_HORIZON_S,
    WINDOW_STRIDE_S,
)
from src.data_loading import list_usable_files, load_carobd_csv
from src.features.extractor import extract_features, feature_names
from src.features.windowing import sliding_windows

log = logging.getLogger(__name__)

# Targets are continuous base features — chosen as the four primary
# signature PIDs across the fault taxonomy (see CHARTER §6). They're
# verified continuous (not regime one-hots) at module-import time so
# z-scoring via BaselineNormalizer works without special-casing.
TARGET_PIDS: list[str] = [
    "LONG_TERM_FUEL_TRIM_BANK_1__mean",
    "INTAKE_MANIFOLD_PRESSURE__mean",
    "COOLANT_TEMPERATURE__mean",
    "THROTTLE_TO_PEDAL_RATIO",
]

_HORIZON_STEPS = FORECAST_HORIZON_S // WINDOW_STRIDE_S  # 60 / 10 = 6


def build_pid_forecast_dataset(
    carobd_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the PID-forecast dataset from healthy carOBD sessions.

    Parameters
    ----------
    carobd_dir : Path or None
        Directory containing the usable carOBD CSVs. Defaults to the
        config-level location.
    output_dir : Path or None
        Where to save ``pid_forecast_v1.parquet``. Defaults to the
        synthetic-data dir (gitignored; regenerable).

    Returns
    -------
    pd.DataFrame
        Columns: all 83 base features + 4 ``target_*`` columns +
        ``session_id``.
    """
    carobd_dir = Path(carobd_dir or DATA_CAROBD_DIR)
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = list_usable_files(carobd_dir)
    if not usable:
        raise FileNotFoundError(f"No usable carOBD files in {carobd_dir}")

    log.info("Building PID forecast dataset from %d healthy sessions …", len(usable))
    all_rows: list[dict] = []

    for path in usable:
        df_clean = load_carobd_csv(path)
        session_id = df_clean.attrs["session_id"]

        # No injection — feed the clean session straight through the
        # window/feature pipeline. The "healthy" label is a stand-in for
        # the windowing function; it's not used downstream.
        feats_seq: list[dict] = [
            extract_features(window)
            for window, _ in sliding_windows(df_clean, "healthy")
        ]

        for i in range(len(feats_seq) - _HORIZON_STEPS):
            now = feats_seq[i]
            future = feats_seq[i + _HORIZON_STEPS]

            row = dict(now)
            # P1-3: target is the DELTA (future − now), not the absolute level.
            # Forecasting the absolute z-value made the model regress toward the
            # training mean on sessions whose baseline is offset, losing to a
            # naive persistence baseline by 6× on LTFT. Predicting the change
            # cancels the per-session/per-vehicle baseline offset — exactly the
            # shift this project must handle. The level is reconstructed at
            # inference as current + predicted_delta.
            for pid in TARGET_PIDS:
                row[f"target_{pid}"] = float(future[pid]) - float(now[pid])
            row["session_id"] = session_id
            all_rows.append(row)

        log.info("  %s: %d pairs", session_id, len(feats_seq) - _HORIZON_STEPS)

    feat_cols = feature_names()
    target_cols = [f"target_{pid}" for pid in TARGET_PIDS]
    keep_cols = feat_cols + target_cols + ["session_id"]
    dataset = pd.DataFrame(all_rows)[keep_cols]

    out_path = output_dir / "pid_forecast_v1.parquet"
    dataset.to_parquet(out_path, index=False)
    log.info(
        "PID forecast dataset: %d pairs across %d sessions saved to %s",
        len(dataset),
        dataset["session_id"].nunique(),
        out_path,
    )
    return dataset


def load_pid_forecast_dataset(output_dir: Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    path = output_dir / "pid_forecast_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"PID forecast dataset not found at {path}. "
            f"Run build_pid_forecast_dataset() first."
        )
    return pd.read_parquet(path)
