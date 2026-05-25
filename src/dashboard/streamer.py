"""Row-by-row CSV streamer for the live dashboard.

Responsibility
--------------
Read a carOBD CSV once at startup, then hand out one row at a time on
demand.  The dashboard loop calls ``next_row()`` on every Streamlit rerun;
the caller decides how many rows to advance per tick based on the speed
multiplier chosen in the sidebar.

Why not stream the file line-by-line?
--------------------------------------
We load the whole file upfront so that:
  1. ``remaining`` / ``total`` are known immediately (progress bar).
  2. ``reset()`` is instant — just reset an integer index, no re-read.
  3. The CSV parsing cost (rename map, column drops) is paid once.

Memory is trivial: a 5-minute session at 1 Hz is ≤ 300 rows × 14 PIDs.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import USEFUL_PIDS
from src.data_loading import load_carobd_csv


class CsvStreamer:
    """Streams one OBD-II row per tick from a carOBD CSV file.

    Parameters
    ----------
    path : Path or str
        Path to a carOBD CSV (one of the 9 usable files or a live capture).
    speed : float
        Playback multiplier.  ``speed=10.0`` means the dashboard advances
        10 simulated seconds per real second.  Stored as an attribute so the
        Streamlit loop can read it for rate-limiting; it does NOT affect what
        ``next_row()`` returns — the caller decides how many rows to advance.
    """

    def __init__(self, path: Path | str, speed: float = 1.0) -> None:
        path = Path(path)
        df = load_carobd_csv(path)

        # Keep only the 14 working PIDs so downstream code never sees stale
        # or unusable columns.  Clip missing PIDs gracefully in case a Skoda
        # file has a different working set.
        pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
        self._rows: list[dict[str, float]] = [
            {k: float(v) for k, v in row.items()}
            for row in df[pid_cols].to_dict(orient="records")
        ]

        self._idx: int = 0
        self.speed: float = float(speed)
        self.session_id: str = path.stem
        self._pid_cols: list[str] = pid_cols

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def next_row(self) -> Optional[dict[str, float]]:
        """Return the next row dict and advance the pointer, or None if exhausted."""
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def peek(self) -> Optional[dict[str, float]]:
        """Return the next row without advancing the pointer."""
        if self._idx >= len(self._rows):
            return None
        return self._rows[self._idx]

    def reset(self) -> None:
        """Reset to the beginning of the file (e.g. replay or new session)."""
        self._idx = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> int:
        """How many rows are left to stream."""
        return max(0, len(self._rows) - self._idx)

    @property
    def total(self) -> int:
        """Total number of rows in the file."""
        return len(self._rows)

    @property
    def elapsed_s(self) -> int:
        """Simulated seconds elapsed (== number of rows consumed so far)."""
        return self._idx

    @property
    def pid_cols(self) -> list[str]:
        """Ordered list of PID columns available in this file."""
        return list(self._pid_cols)

    @property
    def exhausted(self) -> bool:
        """True once all rows have been consumed."""
        return self._idx >= len(self._rows)
