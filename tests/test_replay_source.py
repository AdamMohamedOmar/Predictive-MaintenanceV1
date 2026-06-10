"""ReplayObdSource must be drop-in for LiveObdSource: same methods, same
row contract (all 14 USEFUL_PIDS keys, NaN for absent ones)."""

import math
from pathlib import Path

import pandas as pd

from src.config import USEFUL_PIDS
from src.live.replay_source import ReplayObdSource


def _demo_csv(tmp_path: Path, drop: str | None = None) -> Path:
    n = 5
    data = {p: [float(i) + 1.0 for i in range(n)] for p in USEFUL_PIDS}
    if drop:
        del data[drop]
    p = tmp_path / "session.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def test_drains_all_rows_instantly_when_not_realtime(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path), realtime=False, loop=False)
    assert src.connect()
    src.start()
    rows = []
    while (r := src.next_row()) is not None:
        rows.append(r)
    assert len(rows) == 5
    assert set(rows[0].keys()) == set(USEFUL_PIDS)


def test_missing_column_becomes_nan_and_missing_pid(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path, drop="ACCELERATOR_PEDAL_POSITION_D"),
                          realtime=False, loop=False)
    assert src.connect()
    src.start()
    row = src.next_row()
    assert math.isnan(row["ACCELERATOR_PEDAL_POSITION_D"])
    assert "ACCELERATOR_PEDAL_POSITION_D" in src.missing_pids


def test_connect_false_for_missing_file(tmp_path):
    src = ReplayObdSource(tmp_path / "nope.csv")
    assert src.connect() is False


def test_loop_mode_wraps_around(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path), realtime=False, loop=True)
    src.connect(); src.start()
    for _ in range(7):
        assert src.next_row() is not None  # 5 rows + wrap + 2 more
