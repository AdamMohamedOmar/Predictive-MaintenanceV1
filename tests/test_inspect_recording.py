"""Tests for scripts/inspect_recording.py (P0-A).

Verifies metering-type detection and that the script runs without error on
the adapted Ahmed recording and on synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import USEFUL_PIDS

_AHMED_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "real_faults" / "ahmed" / "ahmed_drive_20260602.csv"
)


def _make_maf_csv(dest: Path, n: int = 120) -> Path:
    """Synthetic clean-column CSV: MAP pinned at baro (MAF-based engine)."""
    rng = np.random.default_rng(11)
    data = {p: rng.uniform(10, 50, n) for p in USEFUL_PIDS}
    # Overwrite MAP: constant at ~101 kPa (barometric) with tiny noise
    data["INTAKE_MANIFOLD_PRESSURE"] = 101.0 + rng.normal(0, 0.2, n)
    data["VEHICLE_SPEED"] = rng.uniform(20, 80, n)
    data["COOLANT_TEMPERATURE"] = np.full(n, 90.0)
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _make_speed_density_csv(dest: Path, n: int = 120) -> Path:
    """Synthetic clean-column CSV: MAP varies with load (speed-density engine)."""
    rng = np.random.default_rng(22)
    data = {p: rng.uniform(10, 50, n) for p in USEFUL_PIDS}
    # MAP oscillates between vacuum (30 kPa) and open-throttle (85 kPa)
    data["INTAKE_MANIFOLD_PRESSURE"] = rng.uniform(30, 85, n)
    data["VEHICLE_SPEED"] = rng.uniform(20, 80, n)
    df = pd.DataFrame(data)[list(USEFUL_PIDS)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


class TestInspectRecording:
    def test_maf_fixture_detected_as_maf(self, tmp_path):
        """A recording with MAP pinned at barometric must be identified as MAF."""
        from scripts.inspect_recording import inspect_recording

        csv = _make_maf_csv(tmp_path / "maf_drive.csv")
        report = inspect_recording(csv)

        assert "MAF" in report["metering_type"], (
            f"Expected MAF detection, got: {report['metering_type']}"
        )

    def test_speed_density_fixture_detected(self, tmp_path):
        """A recording with varying MAP must NOT be flagged as MAF-based."""
        from scripts.inspect_recording import inspect_recording

        csv = _make_speed_density_csv(tmp_path / "sd_drive.csv")
        report = inspect_recording(csv)

        assert "speed-density" in report["metering_type"].lower() or \
               "unknown" in report["metering_type"].lower(), (
            f"Expected speed-density or unknown, got: {report['metering_type']}"
        )
        assert "MAF" not in report["metering_type"] or "likely" not in report["metering_type"]

    def test_report_contains_required_keys(self, tmp_path):
        """Report must carry all expected keys."""
        from scripts.inspect_recording import inspect_recording

        csv = _make_maf_csv(tmp_path / "maf.csv")
        report = inspect_recording(csv)

        for key in (
            "source", "n_rows", "duration_s", "metering_type",
            "metering_detail", "drive_fraction", "pid_coverage",
            "missing_pids", "warnings",
        ):
            assert key in report, f"Missing key: {key}"

    def test_maf_warning_is_present(self, tmp_path):
        """A MAF recording must include the cross-architecture caveat warning."""
        from scripts.inspect_recording import inspect_recording

        csv = _make_maf_csv(tmp_path / "maf.csv")
        report = inspect_recording(csv)

        assert any("MAF" in w for w in report["warnings"]), (
            "Expected MAF caveat warning in report"
        )

    @pytest.mark.skipif(not _AHMED_CSV.exists(), reason="ahmed recording not present")
    def test_runs_on_ahmed_adapted_csv_without_error(self):
        """The inspector must run on Ahmed's adapted recording without raising."""
        from scripts.inspect_recording import inspect_recording

        report = inspect_recording(_AHMED_CSV)

        assert report["n_rows"] > 0
        assert report["metering_type"] != ""
        # Ahmed's car shows MAP pinned at baro -> should be flagged MAF
        assert "MAF" in report["metering_type"], (
            f"Ahmed's car should be MAF-based, got: {report['metering_type']}"
        )
