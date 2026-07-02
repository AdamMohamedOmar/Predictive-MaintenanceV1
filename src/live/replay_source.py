"""Replay a recorded 1 Hz session CSV through the LiveObdSource interface.

Demo insurance: if the ELM327 or the car misbehaves on stage, the SAME
LiveSession UI keeps running from a recorded drive — identical rendering path,
identical frames. Enabled only when PM_ALLOW_REPLAY=1 (see routers/live.py).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import USEFUL_PIDS


class ReplayObdSource:
    """Drop-in for LiveObdSource: connect/start/stop/next_row/connected/
    missing_pids/measured_poll_hz. Paces rows at true 1 Hz unless
    realtime=False (tests drain instantly)."""

    def __init__(self, csv_path: Path | str, *, realtime: bool = True,
                 loop: bool = True) -> None:
        self.csv_path = Path(csv_path)
        self.realtime = realtime
        self.loop = loop
        self._df: Optional[pd.DataFrame] = None
        self._idx = 0
        self._t0: Optional[float] = None

    def connect(self, timeout: float = 0.0) -> bool:
        if not self.csv_path.exists():
            return False
        # index_col=False guards against the carOBD trailing-comma column shift
        # (see src/data_loading.py) — without it, replaying such a CSV silently
        # misaligns every column.
        df = pd.read_csv(self.csv_path, index_col=False)
        if not any(p in df.columns for p in USEFUL_PIDS):
            return False
        self._df = df
        return True

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._idx = 0

    def stop(self) -> None:
        self._t0 = None

    @property
    def connected(self) -> bool:
        return self._df is not None

    @property
    def missing_pids(self) -> list[str]:
        if self._df is None:
            return list(USEFUL_PIDS)
        return [p for p in USEFUL_PIDS
                if p not in self._df.columns or self._df[p].isna().all()]

    @property
    def measured_poll_hz(self) -> float:
        return 1.0 if (self._t0 is not None and self._idx > 0) else 0.0

    def next_row(self) -> Optional[dict[str, float]]:
        if self._df is None or self._t0 is None:
            return None
        if self._idx >= len(self._df):
            if not self.loop:
                return None
            self._idx = 0
            self._t0 = time.monotonic()
        if self.realtime and (time.monotonic() - self._t0) < self._idx:
            return None  # not yet time for the next 1 Hz row
        i = self._idx
        self._idx += 1
        out: dict[str, float] = {}
        for p in USEFUL_PIDS:
            if p in self._df.columns and pd.notna(self._df[p].iloc[i]):
                out[p] = float(self._df[p].iloc[i])
            else:
                out[p] = float("nan")
        return out