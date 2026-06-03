"""Train the IsolationForest anomaly detector on healthy windows.

Workflow
--------
1. Load `data/synthetic/dataset_v1.parquet` (must already exist — produced
   by `scripts/rebuild_all.py`).
2. Apply the same session-level train/test split as the classifier so the
   detector and the classifier evaluate on the same held-out sessions.
3. Fit the BaselineNormalizer on the training split.
4. Fit AnomalyDetector on the healthy training windows.
5. Sanity-evaluate on the held-out test split. Healthy windows should
   score near 0; injector-fault windows should score noticeably higher.
   This is **not** real-fault validation — the fault windows come from
   the same self-consistency loop as the classifier's training labels.
   It only proves the detector wires up.
6. Save model artefact and results JSON.

Output
------
  models/isolation_forest_v1.pkl
  results/anomaly_v1_results.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.config import DATA_SYNTHETIC_DIR, MODELS_DIR, RESULTS_DIR, USEFUL_PIDS, WINDOW_LENGTH_S, WINDOW_STRIDE_S
from src.features.dataset_builder import load_dataset
from src.features.extractor import extract_features
from src.features.normalizer import BaselineNormalizer
from src.features.windowing import sliding_windows
from src.models.anomaly import AnomalyDetector, FAULT_BEARING_FEATURES
from src.models.classifier import session_split

_FPR_BUDGET = 0.01  # design false-positive rate (P1-4)


def _bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, *, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """95 % bootstrap CI for AUC (P1-4): a point estimate on a few hundred
    windows has wide error bars; report the interval instead."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # a resample with one class has no defined AUC
        aucs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _evaluate_on_mock(
    mock_path: Path,
    detector: AnomalyDetector,
    norm: BaselineNormalizer,
) -> dict:
    """Score the Step-2 mock fixture window-by-window.

    Same logical loop as the synthetic dataset (the fixture biases the
    same PIDs the injector biases). A clean separation here proves the
    end-to-end wiring (CSV → features → norm → detector) works.
    """
    df = pd.read_csv(mock_path)
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    df = df[pid_cols]

    # Build per-stride windows the same way dataset_builder does.
    feats: list[dict] = []
    centres: list[int] = []  # row index at the end of each window
    for i in range(WINDOW_LENGTH_S, len(df) + 1, WINDOW_STRIDE_S):
        window = df.iloc[i - WINDOW_LENGTH_S:i]
        feats.append(extract_features(window))
        centres.append(i)

    feats_df = pd.DataFrame(feats)
    scores = detector.score_batch(feats_df, norm)

    # Pre-bias = windows ending before row 260 (full ramp by 300);
    # post-bias = windows ending at row 260+.
    centres_arr = np.array(centres)
    pre_mask = centres_arr < 260
    post_mask = ~pre_mask

    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        # Shouldn't happen given the fixture size, but guard for safety.
        return {
            "n_windows": int(len(scores)),
            "auc": None,
            "note": "insufficient pre- or post-bias windows for AUC",
        }

    y_true = np.where(post_mask, 1.0, 0.0)
    auc = float(roc_auc_score(y_true, scores))

    return {
        "n_windows": int(len(scores)),
        "n_pre_bias": int(pre_mask.sum()),
        "n_post_bias": int(post_mask.sum()),
        "pre_bias_mean_score": float(scores[pre_mask].mean()),
        "post_bias_mean_score": float(scores[post_mask].mean()),
        "auc": auc,
        "note": (
            "Plumbing AUC, not detection AUC. The mock fixture biases the "
            "same PIDs the injector biases (see data/real_faults/README.md). "
            "Real-fault AUC will be computed against Skoda data per "
            "docs/REAL_FAULT_COLLECTION.md."
        ),
    }


def main() -> int:
    dataset_path = DATA_SYNTHETIC_DIR / "dataset_v1.parquet"
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        log.error("Run `python -m scripts.rebuild_all` first.")
        return 1

    log.info("Loading dataset from %s …", dataset_path)
    ds = load_dataset()
    log.info(
        "  %d windows total · labels: %s",
        len(ds),
        ds["label"].value_counts().to_dict(),
    )

    train_df, test_df = session_split(ds)
    log.info("  train=%d  test=%d", len(train_df), len(test_df))

    log.info("Fitting BaselineNormalizer on training split …")
    norm = BaselineNormalizer().fit(train_df)

    log.info("Fitting IsolationForest on healthy training windows …")
    # P1-4: FPR-budget calibration + fault-bearing feature subset (the 83
    # z-features dilute the few discriminative axes inside the IsolationForest).
    detector = AnomalyDetector(
        n_estimators=200,
        random_seed=42,
        fpr_budget=_FPR_BUDGET,
        feature_subset=FAULT_BEARING_FEATURES,
    ).fit(train_df, norm)
    log.info("  fitted on %d healthy windows", (train_df["label"] == "healthy").sum())

    # ── Sanity evaluation on held-out test split ─────────────────────────
    healthy_test = test_df[test_df["label"] == "healthy"]
    fault_test = test_df[test_df["label"] != "healthy"]

    healthy_scores = detector.score_batch(healthy_test, norm)
    fault_scores = detector.score_batch(fault_test, norm)

    # Per-class breakdown
    per_class: dict[str, dict] = {}
    for lbl in sorted(test_df["label"].unique()):
        sub = test_df[test_df["label"] == lbl]
        scores = detector.score_batch(sub, norm)
        per_class[lbl] = {
            "n": int(len(sub)),
            "mean_score": float(scores.mean()),
            "p50_score": float(np.percentile(scores, 50)),
            "p95_score": float(np.percentile(scores, 95)),
        }

    # AUC-ROC: healthy = 0, any-fault = 1. Held-out session split.
    y_true = np.concatenate(
        [np.zeros(len(healthy_test)), np.ones(len(fault_test))]
    )
    y_score = np.concatenate([healthy_scores, fault_scores])
    test_auc = float(roc_auc_score(y_true, y_score))
    auc_lo, auc_hi = _bootstrap_auc_ci(y_true, y_score)

    # Realised healthy false-positive rate at the alarm ceiling (P1-4 budget).
    test_healthy_fpr = float((healthy_scores >= 0.99).mean())

    # ── Step-2 mock fixture (plumbing-only — same loop, different costume) ──
    mock_path = _REPO / "data" / "real_faults" / "mock" / "mock_lean_fault.csv"
    mock_metrics: dict | None = None
    if mock_path.exists():
        mock_metrics = _evaluate_on_mock(mock_path, detector, norm)
        log.info("Mock fixture AUC (pre-bias vs post-bias): %.3f", mock_metrics["auc"])

    results = {
        "n_train_healthy": int((train_df["label"] == "healthy").sum()),
        "n_test_healthy": int(len(healthy_test)),
        "n_test_fault": int(len(fault_test)),
        "test_healthy_mean_score": float(healthy_scores.mean()),
        "test_healthy_p95_score": float(np.percentile(healthy_scores, 95)),
        "test_fault_mean_score": float(fault_scores.mean()),
        "test_fault_p5_score": float(np.percentile(fault_scores, 5)),
        "score_separation": float(fault_scores.mean() - healthy_scores.mean()),
        "test_auc_healthy_vs_any_fault": test_auc,
        "test_auc_95ci": [auc_lo, auc_hi],
        "fpr_budget": _FPR_BUDGET,
        "test_healthy_fpr_at_0.99": test_healthy_fpr,
        "feature_subset": FAULT_BEARING_FEATURES,
        "per_class": per_class,
        "mock_fixture": mock_metrics,
        "note": (
            "Not real-fault validation. The fault windows come from the same "
            "injector that the classifier trains on; a clean separation here "
            "proves the detector wires up, not that it detects real faults. "
            "Real-fault validation pending data per docs/REAL_FAULT_COLLECTION.md."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "isolation_forest_v1.pkl"
    detector.save(model_path)
    log.info("Saved model → %s", model_path)

    results_path = RESULTS_DIR / "anomaly_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results → %s", results_path)

    log.info("")
    log.info("Summary (sanity, not real-fault detection):")
    log.info("  Test healthy score mean: %.3f", results["test_healthy_mean_score"])
    log.info("  Test fault   score mean: %.3f", results["test_fault_mean_score"])
    log.info("  Separation: %.3f", results["score_separation"])
    log.info("  Test AUC (healthy vs any-fault): %.3f  95%% CI [%.3f, %.3f]", test_auc, auc_lo, auc_hi)
    log.info("  Healthy FPR at score≥0.99: %.3f  (budget %.2f)", test_healthy_fpr, _FPR_BUDGET)
    log.info("  Per class:")
    for lbl, r in per_class.items():
        log.info("    %-26s n=%4d  mean=%.3f  p95=%.3f", lbl, r["n"], r["mean_score"], r["p95_score"])
    if mock_metrics:
        log.info("  Mock fixture: pre=%.3f → post=%.3f  AUC=%.3f",
                 mock_metrics["pre_bias_mean_score"],
                 mock_metrics["post_bias_mean_score"],
                 mock_metrics["auc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
