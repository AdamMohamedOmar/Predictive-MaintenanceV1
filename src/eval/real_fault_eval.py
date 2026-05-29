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
        ``windows`` (list of {elapsed_s, label, confidence, all_probs}),
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

    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    rows = df[pid_cols].to_dict(orient="records")

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
                    "all_probs": {
                        str(k): float(v) for k, v in state.all_class_probs.items()
                    },
                }
            )

    label_counts: dict[str, int] = {}
    for w in windows:
        label_counts[w["label"]] = label_counts.get(w["label"], 0) + 1
    fault_count = sum(c for lbl, c in label_counts.items() if lbl != "healthy")

    return {
        "csv_path": str(csv_path),
        "n_rows": len(rows),
        "n_windows": len(windows),
        "windows": windows,
        "summary": {
            "fault_window_count": fault_count,
            "fault_fraction": (fault_count / len(windows)) if windows else 0.0,
            "label_counts": label_counts,
        },
    }
