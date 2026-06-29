"""Loader and column normalizer for carOBD CSV files."""

from pathlib import Path
import pandas as pd

# NOTE: the old USABLE_CAROBD_FILES 9-file whitelist was removed. It encoded a
# parse bug, not a data-quality fact: 120 files were wrongly rejected because a
# trailing comma shifted their columns under a bare pd.read_csv (see
# load_carobd_csv). With index_col=False all 129 files align and pass the
# physical-bounds guard, so `list_usable_files` now validates each file
# dynamically instead of filtering against a fixed list. See docs/DATA_NOTES.md.

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


def load_carobd_csv(
    path: Path | str,
    drop_unusable: bool = True,
) -> pd.DataFrame:
    """Load a carOBD CSV, normalize column names, optionally drop unusable PIDs.

    Cold-start rows (low coolant temp) are intentionally KEPT.  The regime
    detector in src/features/regime.py labels them correctly, and the
    cold_start class in the classifier handles them at the model level.
    Trimming them would discard real diagnostic signals: slow warm-up
    (thermostat), IAC valve faults, and ECT sensor lies all show up first
    during cold-start.

    The session_id (filename stem) is attached as a DataFrame attribute via
    `.attrs`, which is what the session-level splitter reads.

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
    # index_col=False is load-bearing: ~120 carOBD files end each data row with a
    # trailing comma (28 fields vs 27 headers). Without this, pandas silently
    # promotes the first column to the index and shifts EVERY column left by one,
    # so COOLANT_TEMPERATURE reads fuel-trim values and TIMING_ADVANCE reads
    # catalyst temps. That misalignment — not sensor corruption — is what the
    # Week 1 audit mistook for "120 unusable files". See docs/DATA_NOTES.md.
    df = pd.read_csv(path, index_col=False)

    # Verify schema. If a future CSV has an unexpected column, we want to know loudly.
    unknown = set(df.columns) - set(_RENAME_MAP.keys())
    if unknown:
        raise ValueError(f"{path.name}: unknown columns {unknown}. Update _RENAME_MAP.")

    df = df.rename(columns=_RENAME_MAP)

    # Guard against silent column misalignment. A name-based schema check cannot
    # catch a value shift (the column NAMES stay valid), so assert that the
    # signature PIDs sit inside physically possible ranges. A misaligned file
    # fails here LOUDLY instead of entering training as scrambled "healthy" data.
    _assert_physical_bounds(df, path)

    if drop_unusable:
        df = df.drop(columns=[c for c in _UNUSABLE_PIDS if c in df.columns])

    df.attrs["session_id"] = path.stem
    df.attrs["source_file"] = path.name
    return df


# Physical bounds for signature PIDs (OBD-II spec + engine physics). Used to
# detect column misalignment, not to validate calibration — bounds are generous.
_PHYSICAL_BOUNDS = {
    "VEHICLE_SPEED": (0.0, 250.0),
    "COOLANT_TEMPERATURE": (-40.0, 130.0),
    "TIMING_ADVANCE": (-64.0, 64.0),
    "SHORT_TERM_FUEL_TRIM_BANK_1": (-100.0, 100.0),
}


def _assert_physical_bounds(df: pd.DataFrame, path: Path) -> None:
    """Raise if any signature PID falls outside physically possible bounds.

    The canonical symptom of the trailing-comma column shift is an out-of-range
    TIMING_ADVANCE or STFT, so this doubles as a misalignment detector.
    """
    for col, (lo, hi) in _PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) and (s.min() < lo or s.max() > hi):
            raise ValueError(
                f"{path.name}: {col} range [{s.min():.1f}, {s.max():.1f}] is outside "
                f"physical bounds [{lo}, {hi}]. Likely a column-shift / parse error — "
                f"check for a trailing delimiter and confirm index_col=False."
            )


def list_usable_files(data_dir: Path | str) -> list[Path]:
    """Return paths of carOBD CSVs that load and pass the physical-bounds guard.

    Use this everywhere instead of globbing the data directory directly.
    Centralizes the 'which files do we actually trust' decision in one place.

    After the trailing-comma parse fix (see load_carobd_csv), every carOBD file
    aligns and passes bounds, so this now validates each file rather than
    filtering against a hardcoded 9-file whitelist. A file that fails to load or
    breaches physical bounds is skipped with a warning, never silently included.

    Parameters
    ----------
    data_dir : Path or str
        Path to the directory containing the carOBD CSV files.

    Returns
    -------
    list[Path]
        Sorted list of usable file paths. Empty list if data_dir is empty.
    """
    import warnings

    data_dir = Path(data_dir)
    usable: list[Path] = []
    for p in sorted(data_dir.glob("*.csv")):
        try:
            load_carobd_csv(p)  # raises on misalignment / bounds breach
            usable.append(p)
        except (ValueError, pd.errors.ParserError) as exc:
            warnings.warn(f"Skipping {p.name}: {exc}")
    return usable