"""Central configuration for paths and project-wide constants.

Importing from here (rather than hardcoding strings) means:
- Paths work correctly on Windows, macOS, and Linux.
- Changing a folder name is a one-line edit, not a grep-and-replace.
- Tests can monkey-patch paths for isolation.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────
# PROJECT_ROOT resolves relative to this file, not to the current working
# directory. This means running `python scripts/foo.py` from anywhere works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CAROBD_DIR = DATA_RAW_DIR / "carOBD"
DATA_SYNTHETIC_DIR = DATA_DIR / "synthetic"
DATA_SKODA_DIR = DATA_DIR / "skoda_baseline"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ─── Sampling ─────────────────────────────────────────────────────────────
# carOBD is sampled at 1 Hz. This is a dataset property, not a choice.
SAMPLE_RATE_HZ = 1

# Window configuration (locked in charter §7.2).
WINDOW_LENGTH_S = 60
WINDOW_STRIDE_S = 10

# Forecaster prediction horizon (locked in charter §3.1).
FORECAST_HORIZON_S = 60


# ─── Reproducibility ──────────────────────────────────────────────────────
# Single shared seed. Change this only with reason, and re-run all experiments.
RANDOM_SEED = 42


# ─── PID selection ────────────────────────────────────────────────────────
# The 27 PIDs in carOBD include several that are near-constant or irrelevant
# (e.g. DISTANCE_TRAVELED_WITH_MIL_ON is always 0 on a healthy vehicle).
# The USEFUL_PIDS list below is the working set for feature extraction.
# This list is a PLACEHOLDER - confirmed during Week 1 data exploration
# and updated in docs/DATA_NOTES.md.
USEFUL_PIDS = [
    "ENGINE_RPM",
    "VEHICLE_SPEED",
    "THROTTLE",
    "ENGINE_LOAD",
    "COOLANT_TEMPERATURE",
    "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1",
    "INTAKE_MANIFOLD_PRESSURE",
    "ACCELERATOR_PEDAL_POSITION_D",
    "ACCELERATOR_PEDAL_POSITION_E",
    "COMMANDED_THROTTLE_ACTUATOR",
    "FUEL_AIR_COMMANDED_EQUIV_RATIO",
    "INTAKE_AIR_TEMPERATURE",
    "TIMING_ADVANCE",
    "CONTROL_MODULE_VOLTAGE",
]
