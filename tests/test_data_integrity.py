"""Data-integrity tests over the REAL carOBD files (not synthetic fixtures).

Why this file exists
--------------------
The pre-existing suite built fixtures with `to_csv(index=False)`, which never
emits a trailing comma, so it was structurally blind to the column-shift bug
that silently corrupted ~120 files (a trailing comma made bare `pd.read_csv`
promote column 0 to the index and shift every column left by one). These tests
validate every real file on disk against physical AND semantic invariants, so a
regression of that bug — or any new read error — fails the build instead of
slipping into training as scrambled "healthy" data.

These tests are skipped automatically if the carOBD data is not present.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_CAROBD_DIR, USEFUL_PIDS, WINDOW_LENGTH_S
from src.data_loading import load_carobd_csv

_CSV_FILES = sorted(DATA_CAROBD_DIR.glob("*.csv")) if DATA_CAROBD_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not _CSV_FILES, reason="carOBD data not present in data/raw/carOBD"
)

# Physical bounds (OBD-II spec + engine physics). Generous — catching shifts,
# not calibration error.
_BOUNDS = {
    "VEHICLE_SPEED": (0.0, 250.0),
    "ENGINE_RPM": (0.0, 8000.0),
    "COOLANT_TEMPERATURE": (-40.0, 130.0),
    "TIMING_ADVANCE": (-64.0, 64.0),
    "SHORT_TERM_FUEL_TRIM_BANK_1": (-100.0, 100.0),
    "LONG_TERM_FUEL_TRIM_BANK_1": (-100.0, 100.0),
    "CONTROL_MODULE_VOLTAGE": (0.0, 18.0),
    "ENGINE_LOAD": (0.0, 100.0),
}


@pytest.fixture(scope="module")
def loaded():
    """Load every file once (keeping unusable cols so counter checks can run)."""
    return {p.name: load_carobd_csv(p, drop_unusable=False) for p in _CSV_FILES}


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_every_file_loads_without_error(path):
    """The loader must ingest every file (raises on misalignment/out-of-bounds)."""
    df = load_carobd_csv(path)
    assert len(df) > 0


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_no_object_dtype_in_useful_pids(path):
    """Every working PID must be numeric — a stray non-numeric cell (e.g. the
    lone ' ' in live16 INTAKE_AIR_TEMPERATURE) would make the column object-typed
    and silently break describe()/feature extraction."""
    df = load_carobd_csv(path)
    non_numeric = [c for c in USEFUL_PIDS
                   if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    assert not non_numeric, f"{path.name}: non-numeric PID columns {non_numeric}"


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_signature_pids_within_physical_bounds(path):
    """Out-of-bounds timing/STFT is the canonical symptom of a column shift."""
    df = load_carobd_csv(path)
    for col, (lo, hi) in _BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s):
            assert s.min() >= lo and s.max() <= hi, (
                f"{path.name}: {col} range [{s.min():.1f}, {s.max():.1f}] "
                f"outside physical bounds [{lo}, {hi}] — possible column shift"
            )


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_minimum_length_for_one_window(path):
    """A session must contain at least one full window to be usable."""
    df = load_carobd_csv(path)
    assert len(df) >= WINDOW_LENGTH_S, (
        f"{path.name}: only {len(df)} rows, need >= {WINDOW_LENGTH_S}"
    )


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_engine_run_time_is_monotonic(path):
    """ENGINE_RUN_TIME is a clock: monotonic non-decreasing. If columns were
    shifted, this column would hold some other PID and break monotonicity — a
    semantic check that catches shifts even when values happen to be in-bounds."""
    df = load_carobd_csv(path, drop_unusable=False)
    if "ENGINE_RUN_TIME" not in df.columns:
        pytest.skip("ENGINE_RUN_TIME not present")
    s = df["ENGINE_RUN_TIME"].dropna()
    # allow occasional resets/dropouts; require overwhelmingly non-decreasing
    frac_non_decreasing = (s.diff().dropna() >= 0).mean()
    assert frac_non_decreasing > 0.98, (
        f"{path.name}: ENGINE_RUN_TIME only {frac_non_decreasing:.1%} non-decreasing "
        f"— column may be misaligned"
    )


@pytest.mark.parametrize("path", _CSV_FILES, ids=lambda p: p.name)
def test_rpm_speed_semantic_alignment(path):
    """Physical sanity: when the car is stopped (speed==0) the engine idles, so
    RPM should be low for the vast majority of stopped rows. If speed and RPM
    columns were swapped/shifted, this relationship collapses."""
    df = load_carobd_csv(path)
    stopped = df[df["VEHICLE_SPEED"] == 0]
    if len(stopped) < 30:
        pytest.skip("too few stopped rows to assess")
    frac_idle = (stopped["ENGINE_RPM"] < 2000).mean()
    assert frac_idle > 0.9, (
        f"{path.name}: at speed=0, only {frac_idle:.1%} of rows have RPM<2000 "
        f"— speed/RPM columns may be misaligned"
    )


def test_full_dataset_recovered(loaded):
    """After the parse fix the whole dataset should load — a guard against a
    regression that silently drops files back to the old 8-9 file set."""
    assert len(loaded) == len(_CSV_FILES)
    assert len(_CSV_FILES) >= 120, (
        f"Only {len(_CSV_FILES)} files present; expected the full ~129-file dataset"
    )