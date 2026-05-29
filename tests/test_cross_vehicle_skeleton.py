"""Tests for the cross-vehicle paired-fault evaluation skeleton.

Two execution paths are exercised:
  1. No real-vehicle data present (default state at this point in the
     project). The script must write a stub JSON with `no_data_yet`
     status on both vehicles and exit 0. This path needs no model
     artefacts and runs in CI from a fresh clone.
  2. Both CSVs supplied (using the Step-2 mock fixture as a stand-in
     for both vehicles). This path exercises the harness and therefore
     requires the trained model artefacts; it skips when they are not
     present.

Neither path validates real-fault detection. The skeleton's job is to
exist and produce structured output the moment real data lands.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.config import MODELS_DIR

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOCK_CSV = _REPO_ROOT / "data" / "real_faults" / "mock" / "mock_lean_fault.csv"
_MODELS_PRESENT = (MODELS_DIR / "xgb_classifier_v1.pkl").exists() and (
    MODELS_DIR / "forecaster_v1.pkl"
).exists()

# Import the script as a module so we can call main() directly without spawning
# a subprocess (faster, easier assertions).
_cross_vehicle = importlib.import_module("scripts.cross_vehicle_eval")


# ─── Path 1: no data (always runs) ───────────────────────────────────────────


def test_no_data_writes_stub_and_exits_zero(tmp_path):
    """Both vehicles missing → stub JSON with `no_data_yet` on both sides."""
    out = tmp_path / "report.json"
    rc = _cross_vehicle.main(
        ["--fault-type", "air_system", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["fault_type"] == "air_system"
    assert report["vehicles"]["etios"]["status"] == "no_data_yet"
    assert report["vehicles"]["skoda"]["status"] == "no_data_yet"
    assert report["paired_skoda_minus_etios_fault_fraction"] is None


def test_invalid_fault_type_rejected(tmp_path):
    """argparse should reject anything outside the 4-fault enum."""
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        _cross_vehicle.main(
            ["--fault-type", "bogus_class", "--out", str(out)]
        )


def test_nonexistent_csv_path_handled(tmp_path):
    """A path that doesn't exist on disk is reported, not crashed."""
    out = tmp_path / "report.json"
    rc = _cross_vehicle.main(
        [
            "--fault-type", "fuel_system",
            "--etios-fault", str(tmp_path / "missing.csv"),
            "--out", str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["vehicles"]["etios"]["status"] == "path_not_found"
    assert report["vehicles"]["skoda"]["status"] == "no_data_yet"


# ─── Path 2: paired mock (requires trained artefacts) ────────────────────────


@pytest.mark.skipif(
    not _MODELS_PRESENT,
    reason=(
        "Trained model artefacts not present in models/. Run "
        "`python -m scripts.rebuild_all` to build them."
    ),
)
@pytest.mark.skipif(
    not _MOCK_CSV.exists(),
    reason="Step-2 mock fixture missing — re-generate per data/real_faults/README.md.",
)
class TestPairedMock:
    """Use the Step-2 mock fixture as both vehicles' CSVs.

    NOT a cross-vehicle generalisation test — the same CSV on both
    sides means the paired delta should be ~0. This is purely a
    plumbing check that the paired-evaluation code path runs.
    """

    def test_paired_mock_produces_evaluated_both_sides(self, tmp_path):
        out = tmp_path / "report.json"
        rc = _cross_vehicle.main(
            [
                "--fault-type", "fuel_system",
                "--etios-fault", str(_MOCK_CSV),
                "--skoda-fault", str(_MOCK_CSV),
                "--out", str(out),
            ]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["vehicles"]["etios"]["status"] == "evaluated"
        assert report["vehicles"]["skoda"]["status"] == "evaluated"
        assert isinstance(report["vehicles"]["etios"]["n_windows"], int)
        assert report["vehicles"]["etios"]["n_windows"] > 0
        assert isinstance(
            report["paired_skoda_minus_etios_fault_fraction"], float
        )

    def test_paired_mock_delta_is_zero_on_identical_input(self, tmp_path):
        """Same CSV on both sides → fault fractions match exactly."""
        out = tmp_path / "report.json"
        rc = _cross_vehicle.main(
            [
                "--fault-type", "fuel_system",
                "--etios-fault", str(_MOCK_CSV),
                "--skoda-fault", str(_MOCK_CSV),
                "--out", str(out),
            ]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        delta = report["paired_skoda_minus_etios_fault_fraction"]
        # Identical inputs → identical predictions → delta exactly 0.
        assert abs(delta) < 1e-9


# ─── Module-level cross_vehicle_report API ──────────────────────────────────


def test_cross_vehicle_report_returns_documented_shape():
    """The pure-function API returns the documented dict shape."""
    report = _cross_vehicle.cross_vehicle_report(
        "throttle_position_sensor", None, None
    )
    assert set(report.keys()) >= {
        "fault_type",
        "vehicles",
        "paired_skoda_minus_etios_fault_fraction",
        "note",
    }
    assert set(report["vehicles"].keys()) == {"etios", "skoda"}
    assert report["vehicles"]["etios"]["status"] == "no_data_yet"
    assert report["vehicles"]["skoda"]["status"] == "no_data_yet"


def test_cross_vehicle_report_rejects_unknown_fault_type():
    with pytest.raises(ValueError, match="fault_type"):
        _cross_vehicle.cross_vehicle_report("not_a_fault", None, None)
