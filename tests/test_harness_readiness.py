"""P0-0: prove the real-fault harness is ready for a dropped-in recording.

This does NOT prove real-fault detection (no real data exists yet). It proves
that a single OBD-II CSV with sibling mods-in/mods-out metadata runs through
`src/eval/real_fault_eval.py` AND `scripts/cross_vehicle_eval.py` to a JSON
without error — so the day a real Skoda vacuum-leak recording lands, the
pipeline already works. The classifier AND the anomaly detector must both load
and score the CSV without erroring on the column set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, USEFUL_PIDS

_MODELS_PRESENT = (MODELS_DIR / "xgb_classifier_v1.pkl").exists() and (
    MODELS_DIR / "forecaster_v1.pkl"
).exists()
_RAW_DRIVE1 = Path(__file__).resolve().parents[1] / "data" / "raw" / "carOBD" / "drive1.csv"


def _make_pretend_real_csv(dest_dir: Path) -> Path:
    """A clean-column 'pretend-real' recording: 5 min healthy, then a biased
    interval, then healthy again — with a sibling metadata JSON."""
    from src.data_loading import load_carobd_csv

    df = load_carobd_csv(_RAW_DRIVE1).head(900).copy()
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    df = df[pid_cols].reset_index(drop=True)

    rng = np.random.default_rng(7)
    # Fault interval rows 300..600: push trims + drop a couple PIDs to mimic a
    # real, slightly-messy recording (the harness must tolerate this).
    lo, hi = 300, 600
    df.loc[lo:hi, "LONG_TERM_FUEL_TRIM_BANK_1"] += 10.0 + rng.normal(0, 0.5, hi - lo + 1)
    df.loc[lo:hi, "SHORT_TERM_FUEL_TRIM_BANK_1"] += 5.0

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "skoda_pretendreal_20260601_run1.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "vehicle": "skoda_roomster_2007_1.6L",
        "fault_type": "fuel_system",
        "mods_in_place_from_s": lo,
        "mods_removed_at_s": hi,
        "note": "synthetic pretend-real fixture for harness readiness only",
    }
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return csv_path


@pytest.mark.skipif(not _MODELS_PRESENT, reason="trained models not present")
@pytest.mark.skipif(not _RAW_DRIVE1.exists(), reason="drive1.csv not present")
class TestHarnessReadiness:
    def test_real_fault_eval_runs_to_json(self, tmp_path):
        from src.eval.real_fault_eval import evaluate_real_fault

        csv_path = _make_pretend_real_csv(tmp_path / "skoda")
        result = evaluate_real_fault(csv_path)

        assert result["n_rows"] == 900
        assert result["n_windows"] > 0
        # Every window carries both a classifier label AND an anomaly score —
        # the two detectors the real recording will be judged by.
        for w in result["windows"]:
            assert "label" in w
            assert "anomaly_score" in w
            assert 0.0 <= w["anomaly_score"] <= 1.0
        # Serialises cleanly (the harness writes this to disk in production).
        json.dumps(result)

    def test_cross_vehicle_eval_runs_with_one_side(self, tmp_path):
        from scripts.cross_vehicle_eval import cross_vehicle_report

        csv_path = _make_pretend_real_csv(tmp_path / "skoda")
        report = cross_vehicle_report("fuel_system", etios_csv=None, skoda_csv=csv_path)

        assert report["vehicles"]["skoda"]["status"] == "evaluated"
        assert report["vehicles"]["etios"]["status"] == "no_data_yet"
        assert report["vehicles"]["skoda"]["n_windows"] > 0
        json.dumps(report)

    def test_harness_tolerates_a_missing_pid(self, tmp_path):
        """A real adapter may not expose all 14 PIDs — the harness must not crash."""
        from src.eval.real_fault_eval import evaluate_real_fault

        csv_path = _make_pretend_real_csv(tmp_path / "skoda")
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["CONTROL_MODULE_VOLTAGE"])  # drop one PID
        df.to_csv(csv_path, index=False)

        result = evaluate_real_fault(csv_path)  # must not raise
        assert result["n_windows"] > 0
