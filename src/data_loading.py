"""carOBD CSV loading utilities.

Status: STUB. Implementation scheduled for Week 1 Tuesday.
See docs/WEEK_01.md Tuesday section for the function contracts.
"""

from pathlib import Path

import pandas as pd


def load_carobd_csv(path: Path | str) -> pd.DataFrame:
    """Load one carOBD CSV file and return a DataFrame with a timestamp index.

    Args:
        path: Path to the CSV file (e.g. data/raw/carOBD/drive1.csv).

    Returns:
        DataFrame with one row per second of recording. Index is a numeric
        timestamp in seconds from the start of the recording. Columns are
        the PIDs present in the file.

    Raises:
        FileNotFoundError: if the file does not exist.

    # TODO (Week 1): Implement. Verify timestamps are monotonic and 1 Hz.
    """
    raise NotImplementedError("Week 1 Tuesday task. See docs/WEEK_01.md.")


def get_healthy_baseline(df: pd.DataFrame, pid_list: list[str]) -> dict:
    """Compute per-PID mean and std from a healthy recording.

    Args:
        df: A healthy-drive DataFrame loaded by load_carobd_csv.
        pid_list: Which PIDs to summarize (typically config.USEFUL_PIDS).

    Returns:
        Dict keyed by PID name, each value a dict with keys 'mean' and 'std'.

    # TODO (Week 1): Implement.
    """
    raise NotImplementedError("Week 1 Tuesday task. See docs/WEEK_01.md.")
