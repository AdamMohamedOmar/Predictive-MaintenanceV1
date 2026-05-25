"""Sliding-window segmentation for OBD-II time-series sessions.

Turns a raw time-series DataFrame into overlapping fixed-length windows
using the constants locked in config.py (60 s length, 10 s stride).

Design note: this module is label-agnostic. It knows nothing about fault
injection — that responsibility belongs to dataset_builder. The caller
passes a label string and every emitted window carries that same label.
This keeps the logic clean and testable in isolation.
"""

from __future__ import annotations

from typing import Generator

import pandas as pd

from src.config import WINDOW_LENGTH_S, WINDOW_STRIDE_S


def sliding_windows(
    df: pd.DataFrame,
    label: str,
    *,
    window_len: int = WINDOW_LENGTH_S,
    stride: int = WINDOW_STRIDE_S,
) -> Generator[tuple[pd.DataFrame, str], None, None]:
    """Yield (window_df, label) pairs by sliding a fixed-length window over *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series rows at 1 Hz (one row per second).
    label : str
        Class label to attach to every emitted window (e.g. "healthy",
        "fuel_system"). The caller is responsible for choosing the right
        label for the slice being windowed.
    window_len : int
        Window length in seconds / rows (default: WINDOW_LENGTH_S = 60).
    stride : int
        Step between window starts in seconds / rows (default: WINDOW_STRIDE_S = 10).

    Yields
    ------
    tuple[pd.DataFrame, str]
        (window_slice, label) — the slice is a view with reset integer index.
    """
    n = len(df)
    if n < window_len:
        return  # session too short for even one window; yield nothing

    start = 0
    while start + window_len <= n:
        window = df.iloc[start : start + window_len].reset_index(drop=True)
        yield window, label
        start += stride


def count_windows(n_rows: int, window_len: int = WINDOW_LENGTH_S, stride: int = WINDOW_STRIDE_S) -> int:
    """Return how many windows a session of *n_rows* will produce."""
    if n_rows < window_len:
        return 0
    return (n_rows - window_len) // stride + 1
