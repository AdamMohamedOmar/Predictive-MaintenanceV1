"""Regression guard for the single most important invariant in the pipeline:
train/test splits are by SESSION, never by row or window.

Why this test exists
--------------------
60-second windows slide with a 10-second stride, so adjacent windows from
the same session share 50 of their 60 rows. If a split lets windows from
one session land in both train and test, the test set leaks training data
and held-out scores inflate by 10-20 F1 points (CHARTER §5.3, R1).

`session_split` (classifier) and `forecast_session_split` (PID forecaster)
are the two functions that partition data. This test fails loudly if either
ever starts splitting by row — e.g. if someone swaps in
`train_test_split(dataset, ...)` without `groups=session_id`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.classifier import session_split
from src.models.pid_forecaster import forecast_session_split


def _toy_dataset(n_sessions: int = 6, rows_per_session: int = 40) -> pd.DataFrame:
    """A dataset where each session has many rows, so a row-wise split would
    necessarily scatter a session across both partitions."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_sessions):
        sid = f"sess_{s}"
        for _ in range(rows_per_session):
            rows.append(
                {
                    "feat_a": float(rng.normal()),
                    "feat_b": float(rng.normal()),
                    "session_id": sid,
                    "severity_target": float(rng.uniform()),  # for forecast split
                }
            )
    return pd.DataFrame(rows)


def _assert_session_partition(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """The invariant: no session_id appears on both sides, and every row of a
    session stays whole. A row-wise split violates both."""
    train_sessions = set(train["session_id"].unique())
    test_sessions = set(test["session_id"].unique())

    # 1. Disjoint session sets — the headline guarantee.
    overlap = train_sessions & test_sessions
    assert not overlap, (
        f"Session leakage: {overlap} appear in BOTH train and test. "
        f"The split is no longer by session — this inflates held-out scores."
    )

    # 2. Every session is whole (all its rows on one side). This is what a
    #    row-wise split would break even before set-overlap shows up.
    all_sessions = train_sessions | test_sessions
    for sid in all_sessions:
        in_train = (train["session_id"] == sid).sum()
        in_test = (test["session_id"] == sid).sum()
        assert in_train == 0 or in_test == 0, (
            f"Session {sid!r} is split across the boundary "
            f"({in_train} train rows, {in_test} test rows) — rows from one "
            f"recording must never straddle the split."
        )


def test_session_split_partitions_by_session():
    ds = _toy_dataset()
    train, test = session_split(ds, held_out={"sess_0", "sess_3"})
    _assert_session_partition(train, test)
    assert set(test["session_id"].unique()) == {"sess_0", "sess_3"}


def test_forecast_session_split_partitions_by_session():
    ds = _toy_dataset()
    train, test = forecast_session_split(ds, held_out={"sess_1", "sess_4"})
    _assert_session_partition(train, test)
    assert set(test["session_id"].unique()) == {"sess_1", "sess_4"}


def test_split_preserves_all_rows():
    """No row is dropped or duplicated by the split."""
    ds = _toy_dataset()
    train, test = session_split(ds, held_out={"sess_2"})
    assert len(train) + len(test) == len(ds)


def test_split_is_not_row_proportional():
    """Sanity that the split is structural, not a fixed-fraction row slice.

    Holding out 1 of 6 equal-sized sessions must yield exactly 1/6 of rows
    in test — a row-wise 50/50 or 80/20 splitter would produce a different,
    session-blind count and fail this.
    """
    ds = _toy_dataset(n_sessions=6, rows_per_session=40)
    _, test = session_split(ds, held_out={"sess_5"})
    assert len(test) == 40, (
        f"Expected exactly the 40 rows of the one held-out session, got "
        f"{len(test)} — the split is not honouring session boundaries."
    )
