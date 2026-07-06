"""Builds the 60-second-ahead severity forecasting datasets.

One dataset per fault type. Each sample is:
  input  — the 82 z-scored features of window at time t
  target — PID-based severity of the window at time t + FORECAST_HORIZON_S

Pairing logic
-------------
Windows slide at WINDOW_STRIDE_S = 10 s. FORECAST_HORIZON_S = 60 s.
So the target is window[i + 6].severity (6 strides × 10 s = 60 s ahead).

Unlike the classifier dataset, the forecaster uses ALL windows in the
injected session (pre-onset, ramp, and post-ramp) so the model sees
the full severity progression curve and can learn to extrapolate it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_CAROBD_DIR,
    DATA_SYNTHETIC_DIR,
    FORECAST_HORIZON_S,
    INJECTION_NOISE_STD,
    RANDOM_SEED,
    WINDOW_STRIDE_S,
)
from src.data_loading import list_usable_files, load_carobd_csv
from src.features.extractor import extract_features, feature_names
from src.features.severity import compute_severity
from src.features.windowing import sliding_windows
from src.injection import inject_session

log = logging.getLogger(__name__)

FAULT_TYPES = [
    "air_system",
    "fuel_system",
    "coolant_temp_sensor",
    "throttle_position_sensor",
]

_HORIZON_STEPS = FORECAST_HORIZON_S // WINDOW_STRIDE_S  # 60 / 10 = 6


def build_forecast_dataset(
    fault_type: str,
    baselines: dict[str, float],
    carobd_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    random_seed: int = RANDOM_SEED,
    noise_std: float = INJECTION_NOISE_STD,
) -> pd.DataFrame:
    """Build and save the forecasting dataset for one fault type.

    Parameters
    ----------
    fault_type : str
        One of the 4 supported fault strings.
    baselines : dict
        Healthy-window baselines from ``compute_baselines``.
    carobd_dir, output_dir : Path or None
        Override defaults from config.
    random_seed, noise_std : forwarded to the injector.

    Returns
    -------
    pd.DataFrame with columns:
        <feature_name> × 73, severity_target (float), session_id (str)
    """
    carobd_dir = Path(carobd_dir or DATA_CAROBD_DIR)
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = list_usable_files(carobd_dir)
    if not usable:
        raise FileNotFoundError(f"No usable carOBD files in {carobd_dir}")

    log.info("Building forecast dataset: %s (%d sessions)", fault_type, len(usable))
    all_rows: list[dict] = []

    for file_idx, path in enumerate(usable):
        session_seed = random_seed + file_idx
        df_clean = load_carobd_csv(path)
        session_id = df_clean.attrs["session_id"]

        df_faulty = inject_session(
            df_clean,
            fault_type,
            onset_fraction=0.40,
            ramp_fraction=0.15,
            noise_std=noise_std,
            random_seed=session_seed,
        )
        params = df_faulty.attrs["injection"]

        # Only use rows from onset onwards — pre-onset windows can't predict
        # an imminent fault from healthy sensors (no precursor signal exists).
        # The forecaster is called after the classifier detects a fault.
        fault_region = df_faulty.iloc[params.onset_idx:].reset_index(drop=True)

        # Extract features for every window in the fault region
        all_windows = [
            (extract_features(w), label)
            for w, label in sliding_windows(fault_region, fault_type)
        ]
        n = len(all_windows)

        for i in range(n - _HORIZON_STEPS):
            feat_now, _ = all_windows[i]
            feat_future, _ = all_windows[i + _HORIZON_STEPS]

            # TPS fault only manifests when the throttle is active (pedal > 10%).
            # Skip samples where the future window is at idle — the ratio formula
            # gives undefined/noisy severity at near-zero throttle, producing
            # unpredictable targets that inflate MAE without adding signal.
            if fault_type == "throttle_position_sensor":
                if feat_future["THROTTLE__mean"] < 10.0 or feat_now["THROTTLE__mean"] < 10.0:
                    continue

            severity_target = compute_severity(feat_future, fault_type, baselines)

            row = dict(feat_now)
            row["severity_target"] = severity_target
            row["session_id"] = session_id
            all_rows.append(row)

    feat_cols = feature_names()
    meta_cols = ["severity_target", "session_id"]
    dataset = pd.DataFrame(all_rows)[feat_cols + meta_cols]

    out_path = output_dir / f"forecast_{fault_type}_v1.parquet"
    dataset.to_parquet(out_path, index=False)
    log.info(
        "  %s: %d samples saved to %s",
        fault_type,
        len(dataset),
        out_path,
    )
    return dataset


def build_all_forecast_datasets(
    baselines: dict[str, float],
    carobd_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    random_seed: int = RANDOM_SEED,
    noise_std: float = INJECTION_NOISE_STD,
) -> dict[str, pd.DataFrame]:
    """Build forecasting datasets for all 4 fault types sequentially.

    Returns a dict mapping fault_type → DataFrame.
    Parallel training (not building) is handled in forecaster.py.
    """
    carobd_dir = Path(carobd_dir or DATA_CAROBD_DIR)
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)

    return {
        fault: build_forecast_dataset(
            fault,
            baselines,
            carobd_dir=carobd_dir,
            output_dir=output_dir,
            random_seed=random_seed,
            noise_std=noise_std,
        )
        for fault in FAULT_TYPES
    }


def load_forecast_dataset(fault_type: str, output_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously built forecast dataset from disk."""
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    path = output_dir / f"forecast_{fault_type}_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found at {path}. "
            f"Run build_forecast_dataset('{fault_type}', ...) first."
        )
    return pd.read_parquet(path)
