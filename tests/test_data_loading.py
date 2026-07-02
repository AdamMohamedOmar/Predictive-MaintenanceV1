"""Smoke test for the carOBD loader."""

from pathlib import Path
import pandas as pd
import pytest
from src.data_loading import load_carobd_csv


def _make_minimal_csv(tmp_path: Path, coolant_values: list[float]) -> Path:
    """Write a minimal carOBD-schema CSV with the given COOLANT_TEMPERATURE values."""
    n = len(coolant_values)
    data = {
        "ENGINE_RUN_TINE ()": range(n),
        "ENGINE_RPM ()": [800.0] * n,
        "VEHICLE_SPEED ()": [0.0] * n,
        "THROTTLE ()": [0.0] * n,
        "ENGINE_LOAD ()": [10.0] * n,
        "COOLANT_TEMPERATURE ()": coolant_values,
        "LONG_TERM_FUEL_TRIM_BANK_1 ()": [1.0] * n,
        "SHORT_TERM_FUEL_TRIM_BANK_1 ()": [0.0] * n,
        "INTAKE_MANIFOLD_PRESSURE ()": [30.0] * n,
        "FUEL_TANK ()": [50.0] * n,
        "ABSOLUTE_THROTTLE_B ()": [0.0] * n,
        "PEDAL_D ()": [0.0] * n,
        "PEDAL_E ()": [0.0] * n,
        "COMMANDED_THROTTLE_ACTUATOR ()": [0.0] * n,
        "FUEL_AIR_COMMANDED_EQUIV_RATIO ()": [0.0] * n,
        "ABSOLUTE_BAROMETRIC_PRESSURE ()": [101.0] * n,
        "RELATIVE_THROTTLE_POSITION ()": [0.0] * n,
        "INTAKE_AIR_TEMP ()": [25.0] * n,
        "TIMING_ADVANCE ()": [10.0] * n,
        "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()": [0.0] * n,
        "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()": [0.0] * n,
        "CONTROL_MODULE_VOLTAGE ()": [14.2] * n,
        "COMMANDED_EVAPORATIVE_PURGE ()": [0.0] * n,
        "TIME_RUN_WITH_MIL_ON ()": [0.0] * n,
        "TIME_SINCE_TROUBLE_CODES_CLEARED ()": [0.0] * n,
        "DISTANCE_TRAVELED_WITH_MIL_ON ()": [0.0] * n,
        "WARM_UPS_SINCE_CODES_CLEARED ()": [255.0] * n,
    }
    p = tmp_path / "test_session.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "raw" / "carOBD" / "drive1.csv"


@pytest.mark.skipif(not SAMPLE.exists(), reason="carOBD data not present")
def test_load_renames_typo():
    """The typo'd column ENGINE_RUN_TINE must be renamed to ENGINE_RUN_TIME."""
    df = load_carobd_csv(SAMPLE)
    assert "ENGINE_RUN_TIME" in df.columns
    assert "ENGINE_RUN_TINE" not in df.columns
    assert "ENGINE_RUN_TINE ()" not in df.columns


@pytest.mark.skipif(not SAMPLE.exists(), reason="carOBD data not present")
def test_load_drops_unusable():
    """Constant-valued PIDs are dropped by default."""
    df = load_carobd_csv(SAMPLE)
    assert "FUEL_AIR_COMMANDED_EQUIV_RATIO" not in df.columns


@pytest.mark.skipif(not SAMPLE.exists(), reason="carOBD data not present")
def test_load_keeps_unusable_when_asked():
    """Audit mode preserves all columns."""
    df = load_carobd_csv(SAMPLE, drop_unusable=False)
    assert "FUEL_AIR_COMMANDED_EQUIV_RATIO" in df.columns


@pytest.mark.skipif(not SAMPLE.exists(), reason="carOBD data not present")
def test_session_id_attached():
    """Session ID (filename stem) is attached for the Week 3 splitter."""
    df = load_carobd_csv(SAMPLE)
    assert df.attrs["session_id"] == "drive1"


def test_list_usable_files_returns_all_bounds_valid_files():
    """After the trailing-comma parse fix, every carOBD file aligns and passes
    the physical-bounds guard, so list_usable_files returns the full set — not a
    hardcoded 9-file whitelist. This test guards against a regression of the
    column-shift bug (which would make files fail the guard and drop out)."""
    from src.data_loading import list_usable_files, load_carobd_csv

    data_dir = REPO_ROOT / "data" / "raw" / "carOBD"
    if not data_dir.exists():
        pytest.skip("carOBD data not present")

    all_csvs = {p.name for p in data_dir.glob("*.csv")}
    found = {p.name for p in list_usable_files(data_dir)}
    # No file should be silently dropped as "unusable" any more.
    assert found == all_csvs, f"Files unexpectedly rejected: {all_csvs - found}"

    # And every returned file must actually load and pass the bounds guard.
    for p in list_usable_files(data_dir):
        load_carobd_csv(p)  # raises if misaligned / out of bounds


def test_trailing_comma_file_is_not_column_shifted(tmp_path):
    """Regression test for the column-shift bug: a carOBD-schema CSV whose rows
    carry a trailing comma must still load with COLUMNS ALIGNED, i.e. coolant
    reads as coolant (not a shifted neighbour) and timing stays in bounds."""
    coolant = [40.0, 70.0, 90.0]
    p = _make_minimal_csv(tmp_path, coolant)
    # Re-write the file with a trailing comma on every data row (the carOBD quirk).
    text = p.read_text().splitlines()
    header, *rows = text
    p.write_text("\n".join([header] + [r + "," for r in rows]) + "\n")

    df = load_carobd_csv(p)
    assert list(df["COOLANT_TEMPERATURE"]) == coolant  # aligned, not shifted
    assert df["TIMING_ADVANCE"].between(-64, 64).all()  # in bounds, not catalyst temps


def test_loader_keeps_cold_start_rows(tmp_path):
    """Cold-start rows must be preserved — regime detector handles them, not the loader."""
    coolant = [40.0, 55.0, 70.0, 90.0]
    p = _make_minimal_csv(tmp_path, coolant)
    df = load_carobd_csv(p)
    assert len(df) == 4
    assert 40.0 in df["COOLANT_TEMPERATURE"].values


def test_loader_keeps_mid_session_coolant_dip(tmp_path):
    """A mid-session dip below 70°C (e.g. traffic stop) must be preserved."""
    coolant = [90.0, 85.0, 60.0, 88.0, 90.0]
    p = _make_minimal_csv(tmp_path, coolant)
    df = load_carobd_csv(p)
    assert len(df) == 5
    assert 60.0 in df["COOLANT_TEMPERATURE"].values


def test_useful_pids_are_all_produced_by_loader():
    """Every PID in USEFUL_PIDS must be a column the loader produces.

    This test fails if anyone adds a PID name to USEFUL_PIDS that doesn't
    correspond to a real column in the loaded carOBD data — e.g. a PID
    that was dropped by _UNUSABLE_PIDS, or a name with a typo.
    """
    from src.config import USEFUL_PIDS, DATA_CAROBD_DIR
    from src.data_loading import load_carobd_csv, list_usable_files

    files = list_usable_files(DATA_CAROBD_DIR)
    if not files:
        pytest.skip("carOBD data not present")

    df = load_carobd_csv(files[0])
    missing = set(USEFUL_PIDS) - set(df.columns)
    assert not missing, (
        f"USEFUL_PIDS contains names not produced by the loader: {missing}. "
        f"Either add them back, or remove them from USEFUL_PIDS."
    )