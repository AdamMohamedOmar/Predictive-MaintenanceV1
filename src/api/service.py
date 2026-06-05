"""Orchestration layer — the ONLY place the API touches the ML core.

Routers call functions here; they never import from src.dashboard or
src.eval directly. This keeps routers dumb and testable with stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


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
