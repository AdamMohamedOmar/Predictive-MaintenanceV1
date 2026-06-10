"""Per-window inference harness for non-injector-generated CSV data.

This module is a thin loop around `InferenceEngine.update()` that records
one entry per stride window into a structured dict. It is **not** a
fault-detection benchmark — it produces predictions; whether those
predictions constitute "detection" is a downstream question against
per-run metadata (mods-in / mods-out timestamps).

Two intended callers:
  1. tests/test_real_fault_harness_plumbing.py — smoke test against the
     hand-edited mock fixture in data/real_faults/mock/.
  2. scripts/eval_real_fault.py — CLI for the Skoda recordings that will
     land per docs/REAL_FAULT_COLLECTION.md.

Why this is separate from CsvStreamer
-------------------------------------
CsvStreamer is the dashboard's row-by-row playback with a configurable
speed multiplier. The eval harness has no playback concept — it consumes
every row as fast as possible and records per-stride predictions. It also
accepts both the raw carOBD column format ("ENGINE_RPM ()") and the
clean-name format used by demo / mock / Skoda recordings, so it can read
the fixture in data/real_faults/mock/ without depending on the dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import MODELS_DIR, USEFUL_PIDS, WINDOW_STRIDE_S
from src.dashboard.inference import InferenceEngine
from src.data_loading import load_carobd_csv

log = logging.getLogger(__name__)

# Labels that are NOT fault detections — mirrors StableAlerter._NON_FAULT_LABELS.
# cold_start is a normal regime; warming_up is the pre-buffer placeholder.
_NON_FAULT_LABELS: frozenset[str] = frozenset({"healthy", "cold_start", "warming_up"})


def _summarise_labels(label_counts: dict[str, int], n_windows: int) -> dict:
    fault_count = sum(c for lbl, c in label_counts.items() if lbl not in _NON_FAULT_LABELS)
    return {
        "fault_window_count": fault_count,
        "non_fault_window_count": n_windows - fault_count,
        "fault_fraction": (fault_count / n_windows) if n_windows else 0.0,
        "label_counts": label_counts,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    """Load a CSV in either raw carOBD format or clean-column format.

    Raw carOBD: column names like "ENGINE_RPM ()" — load_carobd_csv handles
    the rename map and drops the constant PIDs.
    Clean-column: demo / mock / future Skoda recordings written with clean
    PID names already.
    """
    try:
        return load_carobd_csv(path)
    except ValueError:
        df = pd.read_csv(path)
        df.attrs["session_id"] = path.stem
        df.attrs["source_file"] = path.name
        return df


def evaluate_real_fault(
    csv_path: Path | str,
    *,
    models_dir: Optional[Path] = None,
    engine: Optional[InferenceEngine] = None,
) -> dict:
    """Stream a CSV through the InferenceEngine and record per-stride results.

    Parameters
    ----------
    csv_path : Path or str
        Path to an OBD-II CSV. Either raw carOBD format or clean-column
        format (see ``_read_csv``).
    models_dir : Path or None
        Directory holding ``xgb_classifier_v1.pkl`` and
        ``forecaster_v1.pkl``. Defaults to ``src.config.MODELS_DIR``.
        Ignored if ``engine`` is provided.
    engine : InferenceEngine or None
        Pre-loaded engine. If provided, ``engine.reset()`` is called and
        ``models_dir`` is ignored. The test suite uses this to avoid
        rebuilding the SHAP TreeExplainer between tests.

    Returns
    -------
    dict
        ``csv_path`` (str), ``n_rows`` (int), ``n_windows`` (int),
        ``windows`` (list of {elapsed_s, label, confidence, anomaly_score,
        all_probs, severities, forecasts}),
        ``summary`` (label_counts, fault_window_count, fault_fraction).

    Note
    ----
    Whether any specific label-flag in the returned ``windows`` constitutes
    detection of a real fault depends on the data, not on this harness.
    For the mock fixture, label-flags are by-construction expected to fire
    because the fixture biases the same PIDs the injector biases — see
    data/real_faults/README.md.
    """
    csv_path = Path(csv_path)
    df = _read_csv(csv_path)

    # Tolerate a real adapter that doesn't expose all 14 PIDs: backfill any
    # missing PID column with NaN so feature extraction still runs. The engine
    # NaN-fills those features with the healthy baseline ("no signal → nominal")
    # and counts them toward its degraded-PID warning.
    import numpy as np

    for pid in USEFUL_PIDS:
        if pid not in df.columns:
            df[pid] = np.nan
    rows = df[list(USEFUL_PIDS)].to_dict(orient="records")

    if engine is None:
        engine = InferenceEngine(models_dir=models_dir)
    else:
        engine.reset()

    windows: list[dict] = []
    seen_elapsed: set[int] = set()

    for row in rows:
        row_clean = {k: float(v) for k, v in row.items()}
        state = engine.update(row_clean)

        # One record per stride boundary after buffer is ready. The engine
        # only re-runs classification every WINDOW_STRIDE_S rows, so
        # recording every row would produce duplicates of the same window.
        if (
            state.buffer_ready
            and state.elapsed_s % WINDOW_STRIDE_S == 0
            and state.elapsed_s not in seen_elapsed
        ):
            seen_elapsed.add(state.elapsed_s)
            windows.append(
                {
                    "elapsed_s": int(state.elapsed_s),
                    "label": str(state.classifier_label),
                    "confidence": float(state.classifier_confidence),
                    "anomaly_score": float(getattr(state, "anomaly_score", 0.0)),
                    "all_probs": {
                        str(k): float(v) for k, v in state.all_class_probs.items()
                    },
                    # Current severity [0, 1] per fault type (physics formula)
                    "severities": {
                        str(k): float(v) for k, v in state.severities.items()
                    },
                    # 60-second-ahead forecasted severity [0, 1] per fault type
                    "forecasts": {
                        str(k): float(v) for k, v in state.forecasts.items()
                    },
                    # Top SHAP features driving this prediction [[name, value], ...]
                    "top_shap": [
                        [str(name), float(val)]
                        for name, val in (state.top_features or [])
                    ],
                }
            )

    label_counts: dict[str, int] = {}
    for w in windows:
        label_counts[w["label"]] = label_counts.get(w["label"], 0) + 1

    return {
        "csv_path": str(csv_path),
        "n_rows": len(rows),
        "n_windows": len(windows),
        "windows": windows,
        "summary": _summarise_labels(label_counts, len(windows)),
    }


# §10 headline metric (docs/REAL_FAULT_COLLECTION.md): a vacuum leak may present
# through the trim route (fuel_system) or the mechanical route (air_system) —
# both count.  cold_start / coolant / TPS labels do NOT detect a vacuum leak.
_VACUUM_LEAK_DETECTION_LABELS: frozenset[str] = frozenset({"fuel_system", "air_system"})
_ANOMALY_DETECTION_THRESHOLD = 0.85


def compute_fault_recall(
    windows: list[dict],
    fault_from_s: int,
    fault_to_s: int,
    *,
    detection_labels: frozenset[str] = _VACUUM_LEAK_DETECTION_LABELS,
    anomaly_threshold: float = _ANOMALY_DETECTION_THRESHOLD,
) -> dict:
    """Vacuum-leak recall over the fault interval, exactly as defined in §10.

    Parameters
    ----------
    windows : list of dict
        Per-stride window records from ``evaluate_real_fault`` (each must
        carry ``elapsed_s``, ``label``, ``anomaly_score``).
    fault_from_s, fault_to_s : int
        Fault interval (mods-in / mods-out timestamps), seconds since start.
    detection_labels : frozenset
        Labels that constitute a detection.  Default = §10's set.
    anomaly_threshold : float
        Anomaly-route OR-branch threshold.  Default = §10's 0.85.

    Returns
    -------
    dict with recall, n_fault_windows, n_detected, detected_by_label,
    detected_by_anomaly_only.
    """
    in_interval = [w for w in windows if fault_from_s <= w["elapsed_s"] <= fault_to_s]
    if not in_interval:
        return {
            "recall": 0.0,
            "n_fault_windows": 0,
            "n_detected": 0,
            "detected_by_label": 0,
            "detected_by_anomaly_only": 0,
        }
    by_label = sum(1 for w in in_interval if w["label"] in detection_labels)
    detected = sum(
        1
        for w in in_interval
        if w["label"] in detection_labels
        or float(w.get("anomaly_score", 0.0)) >= anomaly_threshold
    )
    return {
        "recall": detected / len(in_interval),
        "n_fault_windows": len(in_interval),
        "n_detected": detected,
        "detected_by_label": by_label,
        "detected_by_anomaly_only": detected - by_label,
    }
