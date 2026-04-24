"""Smoke tests that verify the project's basic plumbing works.

These tests have no scientific meaning - they exist to confirm:
- pytest can discover and run tests
- src/ is importable as a package
- config.py loads without errors
- The paths it defines resolve to existing project directories

If any of these fail, the environment is broken and no other work can proceed.
"""

from pathlib import Path


def test_src_package_importable():
    """Importing src should not raise."""
    import src  # noqa: F401


def test_config_paths_resolve():
    """PROJECT_ROOT should resolve to a directory that actually contains the repo."""
    from src.config import PROJECT_ROOT

    assert PROJECT_ROOT.exists(), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"
    assert PROJECT_ROOT.is_dir(), f"PROJECT_ROOT is not a directory: {PROJECT_ROOT}"

    # The repo root should contain at least README.md and requirements.txt.
    # If these are missing, either the project is corrupted or PROJECT_ROOT
    # is resolving wrong.
    assert (PROJECT_ROOT / "README.md").exists(), "README.md missing from project root"
    assert (PROJECT_ROOT / "requirements.txt").exists(), "requirements.txt missing"


def test_config_constants_sane():
    """Windowing / forecast horizon constants should match charter values."""
    from src.config import (
        FORECAST_HORIZON_S,
        SAMPLE_RATE_HZ,
        WINDOW_LENGTH_S,
        WINDOW_STRIDE_S,
    )

    assert SAMPLE_RATE_HZ == 1, "carOBD is 1 Hz; do not change this without updating the dataset"
    assert WINDOW_LENGTH_S == 60, "Charter §7.2 locks window length at 60 s"
    assert WINDOW_STRIDE_S == 10, "Charter §7.2 locks window stride at 10 s"
    assert FORECAST_HORIZON_S == 60, "Charter §3.1 locks forecast horizon at 60 s"
