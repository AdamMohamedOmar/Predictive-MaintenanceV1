"""Generate fault-injected demo CSV files for dashboard playback.

Creates one CSV per fault type in data/demo/.  Each file is 10 minutes of
driving: 2 min healthy baseline then fault onset at the 2-minute mark,
ramping/stepping to full severity by 4 min and holding until end.

The dashboard's CsvStreamer can read these files directly — they use the
clean PID column names (not the raw "ENGINE_RPM ()" carOBD format).

Run:
    python -m scripts.generate_demo_data
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.config import USEFUL_PIDS, DATA_RAW_DIR
from src.data_loading import load_carobd_csv
from src.injection.fault_injector import inject_session

_DEMO_DIR = _REPO / "data" / "demo"
_DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Use drive1.csv as the healthy baseline — it's 45 min of highway driving,
# enough signal variation to make the demo visually interesting.
_SOURCE_FILE = DATA_RAW_DIR / "carOBD" / "drive1.csv"

_FAULT_CONFIGS = [
    # (fault_type, mode, onset_fraction, magnitude)
    # Magnitudes are fault-specific units — see fault_injector._DEFAULT_MAGNITUDE:
    #   air_system:               kPa MAP offset at full ramp  (default 13 kPa)
    #   fuel_system:              % LTFT bias at full ramp     (default 18 %)
    #   coolant_temp_sensor:      °C stuck-sensor target        (default 42 °C)
    #   throttle_position_sensor: THROTTLE multiplier           (default 1.35)
    ("air_system",               "ramp", 0.20, 13.0),
    ("fuel_system",              "ramp", 0.20, 18.0),
    ("coolant_temp_sensor",      "step", 0.20, 42.0),
    ("throttle_position_sensor", "ramp", 0.20,  1.35),
]


def main() -> None:
    if not _SOURCE_FILE.exists():
        log.error("Source file not found: %s", _SOURCE_FILE)
        log.error("Run the data loader first: python -m src.data_loading")
        sys.exit(1)

    df = load_carobd_csv(_SOURCE_FILE)
    # Use first 600 rows (~10 min) for a compact demo file
    df = df.head(600).copy()
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]

    for fault_type, mode, onset, magnitude in _FAULT_CONFIGS:
        log.info("Generating %s (%s, onset=%.0f%%)…", fault_type, mode, onset * 100)
        injected = inject_session(
            df,
            fault_type=fault_type,
            onset_fraction=onset,
            magnitude=magnitude,
            random_seed=42,
        )
        out_path = _DEMO_DIR / f"demo_{fault_type}.csv"
        injected[pid_cols].to_csv(out_path, index=False)
        log.info("  → %s  (%d rows)", out_path.name, len(injected))

    log.info("\nDone.  %d demo files in %s", len(_FAULT_CONFIGS), _DEMO_DIR)
    log.info("Restart the dashboard and select a [DEMO] file to see fault detection.")


if __name__ == "__main__":
    main()
