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


_AHMED_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "real_faults" / "ahmed" / "ahmed_drive_20260602.csv"
)


@pytest.mark.skipif(not _MODELS_PRESENT, reason="trained models not present")
@pytest.mark.skipif(not _RAW_DRIVE1.exists(), reason="drive1.csv not present")
class TestHarnessReadiness:
    def test_real_fault_eval_runs_to_json(self, tmp_path):
        from src.eval.real_fault_eval import evaluate_real_fault

        csv_path = _make_pretend_real_csv(tmp_path / "skoda")
        result = evaluate_real_fault(csv_path)

        assert result["n_rows"] == 900
        assert result["n_windows"] > 0
        # Every window carries the classifier label, anomaly score, severities
        # and 60-s-ahead forecasts so testers can see the full picture.
        for w in result["windows"]:
            assert "label" in w
            assert "anomaly_score" in w
            assert 0.0 <= w["anomaly_score"] <= 1.0
            # P1-2: severities and forecasts must be present and in [0, 1]
            assert "severities" in w, "severities key missing from window dict"
            assert "forecasts" in w, "forecasts key missing from window dict"
            for v in w["severities"].values():
                assert 0.0 <= v <= 1.0, f"severity out of [0,1]: {v}"
            for v in w["forecasts"].values():
                assert 0.0 <= v <= 1.0, f"forecast out of [0,1]: {v}"
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

    @pytest.mark.skipif(not _AHMED_CSV.exists(), reason="ahmed recording not present")
    def test_normalizer_override_changes_label_distribution(self, tmp_path):
        """Scoring with vs without a per-vehicle normalizer must produce different
        label distributions — proving the override is actually being applied."""
        from scripts.capture_baseline_from_csv import capture_baseline_from_csv
        from src.dashboard.inference import InferenceEngine
        from src.eval.real_fault_eval import evaluate_real_fault

        # Build a normalizer from the first half of Ahmed's recording (warm + moving)
        # The guard will pass because Ahmed's drive has coolant >= 75 C and speed > 0.
        # If the guard fails (too cold/idle) fall back to the pretend-real CSV so the
        # test still exercises the override plumbing.
        norm_path = tmp_path / "ahmed_normalizer.pkl"
        try:
            capture_baseline_from_csv(
                _AHMED_CSV, vehicle_name="ahmed_test", out_path=norm_path
            )
        except ValueError:
            # Ahmed's recording doesn't pass the speed guard (mostly parked).
            # Use the pretend-real CSV as a fallback baseline source.
            from tests.test_capture_baseline_from_csv import _make_warm_driving_csv
            warm_csv = _make_warm_driving_csv(tmp_path / "warm.csv")
            capture_baseline_from_csv(
                warm_csv, vehicle_name="ahmed_test", out_path=norm_path
            )

        csv_path = _make_pretend_real_csv(tmp_path / "skoda")

        # Score without override
        result_etios = evaluate_real_fault(csv_path)
        counts_etios = result_etios["summary"]["label_counts"]

        # Score with override — different normalizer → different z-scores → different labels
        engine_override = InferenceEngine(normalizer_override=norm_path)
        result_override = evaluate_real_fault(csv_path, engine=engine_override)
        counts_override = result_override["summary"]["label_counts"]

        # Both must produce valid JSON-serialisable output
        json.dumps(result_etios)
        json.dumps(result_override)

        # Both must score some windows
        assert result_etios["n_windows"] > 0
        assert result_override["n_windows"] > 0

        # At least one label count must differ — the override must have an effect
        assert counts_etios != counts_override, (
            "Normalizer override produced identical label distribution — "
            "the override is not being applied."
        )
