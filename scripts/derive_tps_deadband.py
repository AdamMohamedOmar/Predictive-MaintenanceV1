"""Derive the TPS deadband from TRAIN sessions only.

_TPS_DEADBAND was set citing live12's healthy ratio scatter — but live12 is a
held-out TEST session, so the threshold was tuned on test data.  This script
recomputes the healthy cross-session ratio scatter using train sessions only,
which is the defensible derivation for the thesis.

Run:
    python -m scripts.derive_tps_deadband
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.features.dataset_builder import load_dataset
from src.models.classifier import _HELD_OUT_SESSIONS


def main() -> int:
    ds = load_dataset()
    train = ds[~ds["session_id"].isin(_HELD_OUT_SESSIONS)]
    healthy_active = train[
        (train["label"] == "healthy") & (train["THROTTLE__mean"] > 10.0)
    ]
    per_session = healthy_active.groupby("session_id")["THROTTLE_TO_PEDAL_RATIO"].mean()

    print("Per-session healthy active-throttle ratio means (TRAIN only):")
    print(per_session.round(4).to_string())
    scatter = float(per_session.max() - per_session.min())
    print(f"\nmax cross-session scatter: {scatter:.4f}")
    print(f"suggested _TPS_DEADBAND  : {scatter + 0.02:.2f}  (scatter + 0.02 margin)")
    print("\nDecision rule:")
    print("  suggestion <  0.20 -> update severity.py constant, rerun rebuild_all + loso_cv")
    print("  suggestion >= 0.20 -> keep 0.20, but fix the comment to cite this derivation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
