"""Tests for LiveObdSource — the live ELM327 streaming class.

All tests use a mock OBD connection — no real adapter required.
The mock is patched at the ``src.live.obd_source`` module level so the
import resolution is correct regardless of test runner working directory.
"""

import math
import time
from unittest.mock import MagicMock, patch

import obd
import pytest
from obd import OBDResponse
from obd import Unit as OBDUnit
from obd.utils import OBDStatus

from src.config import USEFUL_PIDS
from src.live.obd_source import LiveObdSource


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_response(magnitude: float = 800.0) -> OBDResponse:
    """OBDResponse that looks like a live ECU reply (non-null, has value)."""
    r = OBDResponse()
    r.value = OBDUnit.Quantity(magnitude, OBDUnit.rpm)
    r.messages = ["mock"]
    return r


def _mock_obd_instance(all_supported: bool = True) -> MagicMock:
    """Return a mock obd.OBD instance that reports CAR_CONNECTED."""
    instance = MagicMock()
    instance.status.return_value = OBDStatus.CAR_CONNECTED
    instance.supports.return_value = all_supported
    instance.query.return_value = _make_response(800.0)
    instance.close.return_value = None
    return instance


@pytest.fixture
def mock_conn():
    """Patch obd.OBD constructor in the obd_source module namespace."""
    instance = _mock_obd_instance()
    with patch("src.live.obd_source.obd.OBD", return_value=instance) as MockOBD:
        yield MockOBD, instance


# ── Construction ──────────────────────────────────────────────────────────────

def test_initial_state_not_connected():
    src = LiveObdSource()
    assert src.connected is False


def test_initial_elapsed_zero():
    src = LiveObdSource()
    assert src.elapsed_s == 0


def test_session_id_starts_with_live():
    src = LiveObdSource()
    assert src.session_id.startswith("live_")


def test_initial_supported_pids_empty():
    src = LiveObdSource()
    assert src.supported_pids == []


# ── connect() ────────────────────────────────────────────────────────────────

def test_connect_returns_true_when_car_connected(mock_conn):
    _, instance = mock_conn
    src = LiveObdSource(port="COM3")
    result = src.connect()
    assert result is True
    assert src.connected is True


def test_connect_discovers_pids(mock_conn):
    """After connect(), supported_pids should be populated."""
    _, instance = mock_conn
    src = LiveObdSource()
    src.connect()
    # With supports=True for all, every PID in USEFUL_PIDS is supported
    assert len(src.supported_pids) == len(USEFUL_PIDS)
    assert set(src.supported_pids) == set(USEFUL_PIDS)


def test_connect_returns_false_when_not_car_connected():
    instance = _mock_obd_instance()
    instance.status.return_value = OBDStatus.ELM_CONNECTED  # adapter found, no ECU
    with patch("src.live.obd_source.obd.OBD", return_value=instance):
        src = LiveObdSource()
        result = src.connect()
    assert result is False
    assert src.connected is False


def test_connect_returns_false_on_exception():
    with patch("src.live.obd_source.obd.OBD", side_effect=Exception("port busy")):
        src = LiveObdSource(port="COM9")
        result = src.connect()
    assert result is False
    assert src.connected is False


def test_missing_pids_after_connect_with_partial_support():
    """If ECU supports only half our PIDs, missing_pids reports the rest."""
    instance = _mock_obd_instance()
    # Only USEFUL_PIDS[0..6] are supported
    supported_set = set(USEFUL_PIDS[:7])

    def _supports(cmd):
        # Reverse-lookup: find canonical name for this command
        from src.live.pid_map import PID_MAP
        for name, c in PID_MAP.items():
            if c == cmd:
                return name in supported_set
        return False

    instance.supports.side_effect = _supports

    with patch("src.live.obd_source.obd.OBD", return_value=instance):
        src = LiveObdSource()
        src.connect()

    assert len(src.supported_pids) == 7
    assert len(src.missing_pids) == len(USEFUL_PIDS) - 7


# ── start() / stop() ─────────────────────────────────────────────────────────

def test_start_before_connect_does_nothing():
    """start() without a successful connect() should be a no-op."""
    src = LiveObdSource()
    src.start()  # should not raise, should not spawn a thread
    assert src._thread is None


def test_start_spawns_thread(mock_conn):
    src = LiveObdSource()
    src.connect()
    src.start()
    assert src._thread is not None
    assert src._thread.is_alive()
    src.stop()


def test_stop_joins_thread(mock_conn):
    src = LiveObdSource()
    src.connect()
    src.start()
    src.stop()
    assert src._thread is None
    assert src.connected is False


def test_stop_without_start_does_not_raise():
    src = LiveObdSource()
    src.stop()  # should be idempotent


# ── next_row() — polling behavior ────────────────────────────────────────────

def test_next_row_returns_none_before_start(mock_conn):
    src = LiveObdSource()
    src.connect()
    # Not started yet — queue is empty
    assert src.next_row() is None


def test_next_row_returns_dict_after_polling(mock_conn):
    """After the poll thread runs at least one tick, next_row should have data."""
    src = LiveObdSource(sample_hz=50.0)  # fast poll for test speed
    src.connect()
    src.start()
    # Wait for at least one poll tick
    deadline = time.monotonic() + 2.0
    row = None
    while time.monotonic() < deadline:
        row = src.next_row()
        if row is not None:
            break
        time.sleep(0.02)
    src.stop()
    assert row is not None
    assert isinstance(row, dict)


def test_next_row_contains_all_useful_pids(mock_conn):
    """Every PID in USEFUL_PIDS must appear in the row (NaN for unsupported)."""
    src = LiveObdSource(sample_hz=50.0)
    src.connect()
    src.start()
    deadline = time.monotonic() + 2.0
    row = None
    while time.monotonic() < deadline:
        row = src.next_row()
        if row is not None:
            break
        time.sleep(0.02)
    src.stop()
    assert row is not None
    for pid in USEFUL_PIDS:
        assert pid in row, f"{pid} missing from live row"


def test_unsupported_pids_are_nan_not_zero():
    """PIDs the ECU doesn't support should be NaN, not 0.0."""
    instance = _mock_obd_instance()
    instance.supports.return_value = False  # ECU supports nothing

    with patch("src.live.obd_source.obd.OBD", return_value=instance):
        src = LiveObdSource(sample_hz=50.0)
        src.connect()
        src.start()
        deadline = time.monotonic() + 2.0
        row = None
        while time.monotonic() < deadline:
            row = src.next_row()
            if row is not None:
                break
            time.sleep(0.02)
        src.stop()

    assert row is not None
    for pid in USEFUL_PIDS:
        assert math.isnan(row[pid]), f"{pid} should be NaN for unsupported ECU"


# ── elapsed_s ────────────────────────────────────────────────────────────────

def test_elapsed_s_increments_with_ticks(mock_conn):
    src = LiveObdSource(sample_hz=50.0)
    src.connect()
    src.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and src.elapsed_s < 3:
        time.sleep(0.02)
    src.stop()
    assert src.elapsed_s >= 3


# ── reset() ──────────────────────────────────────────────────────────────────

def test_reset_clears_elapsed_s(mock_conn):
    src = LiveObdSource(sample_hz=50.0)
    src.connect()
    src.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and src.elapsed_s < 2:
        time.sleep(0.02)
    src.stop()
    assert src.elapsed_s >= 2
    src.reset()
    assert src.elapsed_s == 0


def test_reset_drains_queue(mock_conn):
    src = LiveObdSource(sample_hz=50.0)
    src.connect()
    src.start()
    # Let at least one row accumulate
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and src._queue.empty():
        time.sleep(0.02)
    src.stop()
    src.reset()
    assert src.next_row() is None


# ── T3.4: RPM liveness check on reconnect ────────────────────────────────────

def test_try_reconnect_returns_false_when_engine_not_running():
    """_try_reconnect() must refuse a connection where ENGINE_RPM < 50 (ACC mode).

    T3.4 fix: _verify_engine_running() is called from _try_reconnect() just as
    it is from connect(), so a post-Bluetooth-drop reconnect with the ignition
    in ACC is rejected before garbage rows reach the classifier.
    """
    dead_response = _make_response(0.0)  # RPM = 0 → engine off
    instance = _mock_obd_instance()
    instance.query.return_value = dead_response

    with patch("src.live.obd_source.obd.OBD", return_value=instance):
        src = LiveObdSource(port="COM3")
        # Prime _supported_pids so _verify_engine_running() finds ENGINE_RPM
        src._supported_pids = list(USEFUL_PIDS)
        src._conn = instance
        result = src._try_reconnect()

    assert result is False
    assert src.connected is False


def test_try_reconnect_returns_true_when_engine_running():
    """_try_reconnect() succeeds when ENGINE_RPM ≥ 50."""
    live_response = _make_response(850.0)  # RPM = 850 → engine running
    instance = _mock_obd_instance()
    instance.query.return_value = live_response

    with patch("src.live.obd_source.obd.OBD", return_value=instance):
        src = LiveObdSource(port="COM3")
        src._conn = instance
        result = src._try_reconnect()

    assert result is True
    assert src.connected is True
