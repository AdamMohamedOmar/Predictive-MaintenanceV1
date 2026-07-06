"""Tests for the cross-vehicle PID-availability / Untested-fault contract."""

import numpy as np
import pandas as pd

from src.config import USEFUL_PIDS
from src.eval.pid_availability import (
    FAULT_REQUIRED_PIDS,
    available_pids,
    evaluable_faults,
    missing_pids,
    untested_faults,
)


def _df(present: set[str], n: int = 5) -> pd.DataFrame:
    """Build a frame where `present` PIDs have data and the rest are all-NaN."""
    return pd.DataFrame(
        {p: ([1.0] * n if p in present else [np.nan] * n) for p in USEFUL_PIDS}
    )


def test_all_present_nothing_untested():
    avail = available_pids(_df(set(USEFUL_PIDS)))
    assert untested_faults(avail) == {}
    assert set(evaluable_faults(avail)) == set(FAULT_REQUIRED_PIDS)


def test_missing_map_untests_air_only():
    """The Yaris case: no MAP -> air_system Untested, nothing else."""
    df = _df(set(USEFUL_PIDS) - {"INTAKE_MANIFOLD_PRESSURE"})
    ut = untested_faults(available_pids(df))
    assert ut == {"air_system": ["INTAKE_MANIFOLD_PRESSURE"]}
    assert "INTAKE_MANIFOLD_PRESSURE" in missing_pids(df)


def test_present_but_all_nan_column_counts_as_missing():
    """A column that exists but is entirely NaN must be treated as missing."""
    df = _df(set(USEFUL_PIDS))
    df["INTAKE_MANIFOLD_PRESSURE"] = np.nan
    assert "air_system" in untested_faults(available_pids(df))
    assert "INTAKE_MANIFOLD_PRESSURE" in missing_pids(df)


def test_missing_coolant_untests_two_faults():
    df = _df(set(USEFUL_PIDS) - {"COOLANT_TEMPERATURE"})
    ut = untested_faults(available_pids(df))
    assert "cold_start" in ut
    assert "coolant_temp_sensor" in ut


def test_throttle_needs_pedals_and_both_throttle_channels():
    # missing COMMANDED_THROTTLE_ACTUATOR -> untested, and it names the lack
    df = _df(set(USEFUL_PIDS) - {"COMMANDED_THROTTLE_ACTUATOR"})
    ut = untested_faults(available_pids(df))
    assert ut.get("throttle_position_sensor") == ["COMMANDED_THROTTLE_ACTUATOR"]
    # missing BOTH pedal channels -> untested too (THROTTLE_TO_PEDAL_RATIO
    # would be fabricated -> phantom TPS fault)
    df2 = _df(set(USEFUL_PIDS) - {
        "ACCELERATOR_PEDAL_POSITION_D", "ACCELERATOR_PEDAL_POSITION_E"})
    ut2 = untested_faults(available_pids(df2))
    assert set(ut2.get("throttle_position_sensor", [])) == {
        "ACCELERATOR_PEDAL_POSITION_D", "ACCELERATOR_PEDAL_POSITION_E"}


def test_healthy_always_evaluable():
    assert "healthy" in evaluable_faults(available_pids(_df(set())))