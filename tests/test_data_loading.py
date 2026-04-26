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
