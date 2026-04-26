"""Loader and column normalizer for carOBD CSV files."""

from pathlib import Path
import pandas as pd

# Files in the carOBD dataset whose values pass physical-bounds checks for all
# 4 critical signature PIDs (vehicle speed, coolant temp, timing advance, STFT).
# The other 120 files in the dataset have values that violate physical bounds
# for timing and/or STFT — likely a firmware-version inconsistency in the
# original carOBD recordings. See docs/DATA_NOTES.md for full audit details.
USABLE_CAROBD_FILES = frozenset(
    {
        "drive1.csv",
        "live5.csv",
        "live6.csv",
        "live7.csv",
        "live8.csv",
        "live9.csv",
        "live10.csv",
        "live11.csv",
        "live12.csv",
    }
)

# Map raw carOBD CSV column names → cleaned, charter-aligned names.
# This is the single source of truth for column names in the project.
# All downstream code uses the cleaned names.
_RENAME_MAP = {
    "ENGINE_RUN_TINE ()": "ENGINE_RUN_TIME",  # typo fix
    "ENGINE_RPM ()": "ENGINE_RPM",
    "VEHICLE_SPEED ()": "VEHICLE_SPEED",
    "THROTTLE ()": "THROTTLE",
    "ENGINE_LOAD ()": "ENGINE_LOAD",
    "COOLANT_TEMPERATURE ()": "COOLANT_TEMPERATURE",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()": "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()": "SHORT_TERM_FUEL_TRIM_BANK_1",
    "INTAKE_MANIFOLD_PRESSURE ()": "INTAKE_MANIFOLD_PRESSURE",
    "FUEL_TANK ()": "FUEL_TANK_LEVEL_INPUT",
    "ABSOLUTE_THROTTLE_B ()": "ABSOLUTE_THROTTLE_B",
    "PEDAL_D ()": "ACCELERATOR_PEDAL_POSITION_D",
    "PEDAL_E ()": "ACCELERATOR_PEDAL_POSITION_E",
    "COMMANDED_THROTTLE_ACTUATOR ()": "COMMANDED_THROTTLE_ACTUATOR",
    "FUEL_AIR_COMMANDED_EQUIV_RATIO ()": "FUEL_AIR_COMMANDED_EQUIV_RATIO",
    "ABSOLUTE_BAROMETRIC_PRESSURE ()": "ABSOLUTE_BAROMETRIC_PRESSURE",
    "RELATIVE_THROTTLE_POSITION ()": "RELATIVE_THROTTLE_POSITION",
    "INTAKE_AIR_TEMP ()": "INTAKE_AIR_TEMPERATURE",
    "TIMING_ADVANCE ()": "TIMING_ADVANCE",
    "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()": "CATALYST_TEMPERATURE_BANK1_SENSOR1",
    "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()": "CATALYST_TEMPERATURE_BANK1_SENSOR2",
    "CONTROL_MODULE_VOLTAGE ()": "CONTROL_MODULE_VOLTAGE",
    "COMMANDED_EVAPORATIVE_PURGE ()": "COMMANDED_EVAPORATIVE_PURGE",
    "TIME_RUN_WITH_MIL_ON ()": "TIME_RUN_WITH_MIL_ON",
    "TIME_SINCE_TROUBLE_CODES_CLEARED ()": "TIME_SINCE_TROUBLE_CODES_CLEARED",
    "DISTANCE_TRAVELED_WITH_MIL_ON ()": "DISTANCE_TRAVELED_WITH_MIL_ON",
    "WARM_UPS_SINCE_CODES_CLEARED ()": "WARM_UPS_SINCE_CODES_CLEARED",
}

# PIDs known to be unusable in carOBD (constant or sentinel-only).
# Documented in docs/DATA_NOTES.md. Loaded callers can opt-out via drop_unusable=False.
_UNUSABLE_PIDS = {
    "FUEL_AIR_COMMANDED_EQUIV_RATIO",  # always 0 in carOBD recordings
    "TIME_RUN_WITH_MIL_ON",  # always 0
    "DISTANCE_TRAVELED_WITH_MIL_ON",  # always 0
    "WARM_UPS_SINCE_CODES_CLEARED",  # always 255 (OBD "no data" sentinel)
}


def load_carobd_csv(path: Path | str, drop_unusable: bool = True) -> pd.DataFrame:
    """Load a carOBD CSV, normalize column names, optionally drop unusable PIDs.

    The session_id (filename stem) is attached as a DataFrame attribute via
    `.attrs`, which is what the Week 3 session-level splitter will read.

    Parameters
    ----------
    path : Path or str
        Path to a carOBD CSV file (e.g. data/raw/carOBD/drive1.csv).
    drop_unusable : bool, default True
        If True, drops PIDs that are known constants in carOBD.
        Set False only when explicitly auditing data quality.

    Returns
    -------
    pd.DataFrame
        Columns are the cleaned PID names; row index is the integer second.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Verify schema. If a future CSV has an unexpected column, we want to know loudly.
    unknown = set(df.columns) - set(_RENAME_MAP.keys())
    if unknown:
        raise ValueError(f"{path.name}: unknown columns {unknown}. Update _RENAME_MAP.")

    df = df.rename(columns=_RENAME_MAP)

    if drop_unusable:
        df = df.drop(columns=[c for c in _UNUSABLE_PIDS if c in df.columns])

    df.attrs["session_id"] = path.stem
    df.attrs["source_file"] = path.name
    return df


def list_usable_files(data_dir: Path | str) -> list[Path]:
    """Return paths of carOBD CSVs that pass our physical-bounds audit.

    Use this everywhere instead of globbing the data directory directly.
    Centralizes the 'which files do we actually trust' decision in one place.

    Parameters
    ----------
    data_dir : Path or str
        Path to the directory containing the carOBD CSV files.

    Returns
    -------
    list[Path]
        Sorted list of usable file paths. Empty list if data_dir is empty.
    """
    data_dir = Path(data_dir)
    return sorted(p for p in data_dir.glob("*.csv") if p.name in USABLE_CAROBD_FILES)
