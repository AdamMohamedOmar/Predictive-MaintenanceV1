"""Orchestration layer — the ONLY place the API touches the ML core.

Routers call functions here; they never import from src.dashboard or
src.eval directly. This keeps routers dumb and testable with stubs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


def process_upload(
    raw_csv: Path,
    out_dir: Path,
    normalizer_path: Optional[Path],
    is_baseline: bool,
    vehicle_name: str = "vehicle",
) -> dict:
    """Full upload pipeline: inspect → adapt → baseline-capture OR score.

    Parameters
    ----------
    raw_csv : Path
        The raw file saved from the multipart upload. May be either a raw
        Torque/Car-Scanner export OR an already-adapted clean-column CSV.
    out_dir : Path
        Per-car directory where adapted.csv, result.json, normalizer.pkl land.
    normalizer_path : Path or None
        Car's existing baseline normalizer. Used for scoring; ignored for baseline.
    is_baseline : bool
        True → run capture_baseline_from_csv and save normalizer.pkl.
        False → adapt and score with the existing normalizer.
    vehicle_name : str
        Free-text label stored in the baseline sidecar JSON.

    Returns
    -------
    dict with keys:
        mode        : 'baseline' | 'score'
        inspect     : dict from inspect_recording (metering, PID coverage, warnings)
        — if mode=='baseline' —
        normalizer_path : str  path to saved .pkl
        n_windows       : int
        — if mode=='score' —
        result          : dict  full evaluate_real_fault output
        adapted_csv     : str   path to adapted clean-column CSV
        result_json     : str   path to result.json

    Raises
    ------
    ValueError
        If is_baseline=True and the guard checks fail (cold/idle/too-short).
    RuntimeError
        If adaptation or scoring fails for unexpected reasons.
    """
    from scripts.adapt_torque_csv import adapt_torque_csv
    from scripts.capture_baseline_from_csv import capture_baseline_from_csv
    from scripts.inspect_recording import inspect_recording

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: inspect (works on both raw and clean-column formats)
    inspect = inspect_recording(raw_csv)

    # Step 2: adapt — skip if already clean-column (inspect flags it)
    if inspect.get("is_clean_column_format"):
        adapted = out_dir / "adapted.csv"
        shutil.copy2(raw_csv, adapted)
    else:
        clean_df, _report = adapt_torque_csv(raw_csv)
        adapted = out_dir / "adapted.csv"
        clean_df.to_csv(adapted, index=False)

    # Step 3a: baseline capture
    if is_baseline:
        norm_path = out_dir / "normalizer.pkl"
        # Raises ValueError if guards fail (cold/idle/too-short)
        capture_baseline_from_csv(adapted, vehicle_name=vehicle_name, out_path=norm_path)
        meta_path = norm_path.with_suffix(".json")
        n_windows = None
        if meta_path.exists():
            try:
                n_windows = json.loads(meta_path.read_text()).get("n_windows")
            except Exception:
                pass
        return {
            "mode": "baseline",
            "inspect": inspect,
            "normalizer_path": str(norm_path),
            "n_windows": n_windows,
        }

    # Step 3b: score
    result = score_adapted_csv(adapted, normalizer_path)
    result_json = out_dir / "result.json"
    result_json.write_text(json.dumps(result, indent=2))
    return {
        "mode": "score",
        "inspect": inspect,
        "result": result,
        "adapted_csv": str(adapted),
        "result_json": str(result_json),
    }


def score_adapted_csv(adapted_csv: Path, normalizer_path: Optional[Path]) -> dict:
    """Score an already-adapted clean-column CSV. Wraps evaluate_real_fault.

    Parameters
    ----------
    adapted_csv : Path
        Clean-column 1 Hz CSV produced by adapt_torque_csv.
    normalizer_path : Path or None
        Per-vehicle baseline normalizer. When None, uses the Etios default.
    """
    from src.dashboard.inference import InferenceEngine
    from src.eval.real_fault_eval import evaluate_real_fault

    engine: Optional[InferenceEngine] = None
    if normalizer_path and Path(normalizer_path).exists():
        engine = InferenceEngine(normalizer_override=Path(normalizer_path))

    kwargs = {"engine": engine} if engine else {}
    return evaluate_real_fault(adapted_csv, **kwargs)
