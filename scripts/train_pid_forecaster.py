"""Train the PID-target forecaster on healthy carOBD sessions.

Workflow
--------
1. Build (or load) the PID forecast dataset (`data/synthetic/pid_forecast_v1.parquet`).
2. Session-level train/test split (same held-out sessions as the classifier).
3. Fit a fresh BaselineNormalizer on the training split.
4. Train one XGBRegressor per target PID in parallel.
5. Compare each model's MAE_z against a "PID stays at current value"
   persistence baseline — beating it confirms the forecaster learned
   real trajectory dynamics rather than the identity mapping.
6. Save bundle + results JSON.

Output
------
  models/pid_forecaster_v1.pkl
  results/pid_forecaster_v1_results.json

Not a real-fault validation
---------------------------
The metrics here only show that a model trained on healthy trajectories
generalises across held-out healthy sessions better than persistence.
Whether predicted-vs-actual residuals separate healthy from real-fault
windows is validated against Skoda data per docs/REAL_FAULT_COLLECTION.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.features.normalizer import BaselineNormalizer
from src.features.pid_forecast_dataset import (
    TARGET_PIDS,
    build_pid_forecast_dataset,
    load_pid_forecast_dataset,
)
from src.models.pid_forecaster import (
    forecast_session_split,
    train_all_pid_forecasters,
)


def main() -> int:
    try:
        ds = load_pid_forecast_dataset()
        log.info("Loaded existing PID forecast dataset (%d pairs)", len(ds))
    except FileNotFoundError:
        log.info("PID forecast dataset not found — building from carOBD …")
        ds = build_pid_forecast_dataset()

    log.info("Dataset: %d pairs across %d sessions",
             len(ds), ds["session_id"].nunique())

    train_df, test_df = forecast_session_split(ds)
    log.info("Train pairs=%d  test pairs=%d", len(train_df), len(test_df))

    log.info("Fitting BaselineNormalizer on training split …")
    # The dataset's "label" column doesn't exist (this is a healthy-only
    # build) — synthesise one so the normalizer's healthy-only fit path
    # treats every row as fittable.
    train_df_with_label = train_df.copy()
    train_df_with_label["label"] = "healthy"
    norm = BaselineNormalizer().fit(train_df_with_label)

    forecaster = train_all_pid_forecasters(
        ds, norm, n_estimators=300, random_seed=42
    )
    forecaster.save()

    log.info("")
    log.info("Summary:")
    for pid in TARGET_PIDS:
        r = forecaster.results[pid]
        verdict = "BEATS persistence" if r["beats_persistence"] else "worse than persistence"
        log.info(
            "  %-40s MAE_z=%.4f  (persistence=%.4f) %s",
            pid,
            r["mae_z"],
            r["mae_persistence_baseline_z"],
            verdict,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
