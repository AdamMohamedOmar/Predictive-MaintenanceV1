"""Orchestrates injection → windowing → extraction to produce a training dataset.

Labelling strategy
------------------
For each of the 9 clean carOBD sessions we produce two kinds of rows:

  Healthy windows
      Windowed directly from the clean (un-injected) session.

  Fault windows
      For each of the 4 fault types:
        1. Inject the fault with ramp mode (onset at 40 %, ramp over 15 %).
        2. Skip the ramp transition (rows onset_idx → onset_idx + ramp_len).
           These rows are labelled ambiguously; excluding them gives the
           classifier cleaner fault boundaries.
        3. Window the fully-developed fault region (post-ramp rows only).

The pre-onset (healthy-baseline) rows from injected sessions are NOT
re-windowed — they are identical to the clean session and would just
duplicate healthy examples.

Output
------
A single parquet file at data/synthetic/dataset_v1.parquet with columns:
  <feature_name_0>, …, <feature_name_72>, label (str), label_id (int),
  session_id (str), fault_type (str)

A companion metadata JSON is saved alongside the parquet file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_CAROBD_DIR, DATA_SYNTHETIC_DIR, INJECTION_NOISE_STD, RANDOM_SEED
from src.data_loading import list_usable_files, load_carobd_csv
from src.features.extractor import extract_features, feature_names
from src.features.regime import detect_regime
from src.features.windowing import sliding_windows
from src.injection import inject_session
from src.injection.fault_injector import _DEFAULT_MAGNITUDE

log = logging.getLogger(__name__)

FAULT_TYPES = [
    "air_system",
    "fuel_system",
    "coolant_temp_sensor",
    "throttle_position_sensor",
]

LABEL_TO_ID: dict[str, int] = {
    "healthy": 0,
    "air_system": 1,
    "fuel_system": 2,
    "coolant_temp_sensor": 3,
    "throttle_position_sensor": 4,
    "cold_start": 5,
}

# Injection configuration — matches CLAUDE.md defaults
_ONSET_FRAC = 0.40
_RAMP_FRAC  = 0.15
_NOISE_STD  = INJECTION_NOISE_STD  # centralised in config.py


def build_dataset(
    carobd_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    random_seed: int = RANDOM_SEED,
    noise_std: float = _NOISE_STD,
) -> pd.DataFrame:
    """Build the labelled feature dataset from carOBD files and save it.

    Parameters
    ----------
    carobd_dir : Path or None
        Directory containing carOBD CSV files. Defaults to DATA_CAROBD_DIR.
    output_dir : Path or None
        Where to write the parquet + metadata files. Defaults to DATA_SYNTHETIC_DIR.
    random_seed : int
        Master seed; per-session seeds are derived from this.
    noise_std : float
        Gaussian noise level passed to the fault injector.

    Returns
    -------
    pd.DataFrame
        The complete labelled feature matrix (also saved to disk).
    """
    carobd_dir = Path(carobd_dir or DATA_CAROBD_DIR)
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    usable_files = list_usable_files(carobd_dir)
    if not usable_files:
        raise FileNotFoundError(f"No usable carOBD files found in {carobd_dir}")

    log.info("Building dataset from %d sessions × 5 classes", len(usable_files))

    all_rows: list[dict] = []

    for file_idx, path in enumerate(usable_files):
        session_seed = random_seed + file_idx
        df_clean = load_carobd_csv(path)
        session_id = df_clean.attrs["session_id"]
        log.info("  Processing %s (%d rows)", session_id, len(df_clean))

        # ── Healthy / cold_start windows from the clean session ──────────
        # Regime determines the label: cold-start windows get their own class
        # so the classifier can distinguish "engine warming up normally" from
        # "stuck-cold ECT sensor fault".
        for window, _ in sliding_windows(df_clean, "healthy"):
            regime = detect_regime(window)
            label = "cold_start" if regime == "cold_start" else "healthy"
            row = extract_features(window)
            row["label"] = label
            row["label_id"] = LABEL_TO_ID[label]
            row["session_id"] = session_id
            row["fault_type"] = label
            all_rows.append(row)

        # ── Fault windows (one pass per fault type) ───────────────────────
        for fault in FAULT_TYPES:
            df_faulty = inject_session(
                df_clean,
                fault,
                onset_fraction=_ONSET_FRAC,
                ramp_fraction=_RAMP_FRAC,
                noise_std=noise_std,
                random_seed=session_seed,
            )
            params = df_faulty.attrs["injection"]
            # Skip only the first 25% of the ramp (the most ambiguous transition
            # region) so the classifier trains on both early-developing and
            # fully-developed faults.  Excluding the entire ramp caused 0%
            # recall on early/mid-ramp windows at inference time.
            fault_start = params.onset_idx + max(1, params.ramp_len // 4)

            fault_region = df_faulty.iloc[fault_start:].reset_index(drop=True)
            if len(fault_region) == 0:
                log.warning(
                    "    %s/%s: fault region is empty after ramp — skipping",
                    session_id,
                    fault,
                )
                continue

            for window, label in sliding_windows(fault_region, fault):
                row = extract_features(window)
                row["label"] = label
                row["label_id"] = LABEL_TO_ID[label]
                row["session_id"] = session_id
                row["fault_type"] = fault
                all_rows.append(row)

    dataset = pd.DataFrame(all_rows)

    # Ensure feature columns come first, metadata columns last
    feat_cols = feature_names()
    meta_cols = ["label", "label_id", "session_id", "fault_type"]
    dataset = dataset[feat_cols + meta_cols]

    out_path = output_dir / "dataset_v1.parquet"
    dataset.to_parquet(out_path, index=False)
    log.info("Saved %d rows × %d features to %s", len(dataset), len(feat_cols), out_path)

    _save_metadata(output_dir, dataset, usable_files, random_seed, noise_std)

    return dataset


def _save_metadata(
    output_dir: Path,
    dataset: pd.DataFrame,
    usable_files: list[Path],
    random_seed: int,
    noise_std: float,
) -> None:
    class_counts = dataset.groupby("label")["label"].count().to_dict()
    meta = {
        "n_samples": len(dataset),
        "n_features": len(feature_names()),
        "sessions": [p.name for p in usable_files],
        "fault_types": FAULT_TYPES,
        "label_to_id": LABEL_TO_ID,
        "class_counts": class_counts,
        "injection": {
            "mode": "ramp",
            "onset_fraction": _ONSET_FRAC,
            "ramp_fraction": _RAMP_FRAC,
            "noise_std": noise_std,
            "magnitudes": _DEFAULT_MAGNITUDE,
        },
        "random_seed": random_seed,
    }
    meta_path = output_dir / "dataset_v1_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved to %s", meta_path)


def load_dataset(output_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously built dataset from disk."""
    output_dir = Path(output_dir or DATA_SYNTHETIC_DIR)
    path = output_dir / "dataset_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run build_dataset() first."
        )
    return pd.read_parquet(path)
