"""Smoke test for the carOBD loader."""

from pathlib import Path
import pytest
from src.data_loading import load_carobd_csv

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


def test_list_usable_files_returns_known_set():
    """The usable-files helper must return exactly the 9 audited-clean files."""
    from src.data_loading import list_usable_files, USABLE_CAROBD_FILES

    data_dir = REPO_ROOT / "data" / "raw" / "carOBD"
    if not data_dir.exists():
        pytest.skip("carOBD data not present")

    found = {p.name for p in list_usable_files(data_dir)}
    expected = set(USABLE_CAROBD_FILES)
    # If new files appear in the data dir, this test won't fail wrongly:
    assert found == expected, (
        f"Mismatch: missing={expected - found}, extra={found - expected}"
    )


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
