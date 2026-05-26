"""Live OBD-II row source via ELM327 adapter.

Drop-in sibling to CsvStreamer
--------------------------------
Both classes expose the same interface:

    next_row() → dict[str, float] | None
    reset()
    elapsed_s (property)
    session_id (property)

The dashboard's main loop can branch on source type for things like progress
bars (undefined for an open-ended live stream), but the core
`engine.update(row)` call is identical.

Threading model
---------------
One background poll thread runs at `sample_hz` (default 1 Hz).  On each tick
it queries all ECU-supported PIDs, builds a row dict, and pushes it to a
queue of size 1.  If the dashboard is slow to consume (e.g. SHAP takes > 1 s),
the old row is discarded and only the latest is kept — the UI shows live data,
not a replay of backlogged rows.

The main thread calls next_row() which does a non-blocking queue pop.

Connection lifecycle
--------------------
  1. Instantiate:  src = LiveObdSource(port="COM3")
  2. Connect:      ok = src.connect()    # blocks up to timeout seconds
  3. Start poll:   src.start()           # spawns background thread
  4. Stream:       row = src.next_row()  # in Streamlit loop
  5. Stop:         src.stop()            # join thread, close port
  6. Reset:        src.reset()           # clear buffer between calibration passes

Auto-reconnect
--------------
If the adapter drops mid-drive (cable wiggle, Bluetooth hiccup), the poll
thread retries connection every 2 seconds.  The `connected` property reflects
current state; the dashboard renders a status indicator from it.

Missing PIDs
------------
If the ECU does not support a PID we request, that PID is filled with NaN.
NaN propagates through feature extraction to produce NaN-tainted z-scores,
which the classifier treats as an absent feature rather than a false zero.
If more than 2 of the 14 PIDs are absent, the dashboard should warn the user
(checked by the live_discover script before inference starts).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import obd
from obd.utils import OBDStatus

from src.config import USEFUL_PIDS
from src.live.pid_map import PID_MAP, to_float

log = logging.getLogger(__name__)

# Seconds to wait between reconnection attempts when the adapter drops
_RECONNECT_INTERVAL_S = 2.0

# Maximum seconds for initial connect() call before giving up
_CONNECT_TIMEOUT_S = 15.0


class LiveObdSource:
    """Streams one OBD-II row per tick from a live ELM327-connected ECU.

    Parameters
    ----------
    port : str or None
        Serial port for the ELM327 adapter (e.g. "COM3", "/dev/ttyUSB0").
        None lets python-OBD auto-detect — works when only one adapter is
        plugged in.
    sample_hz : float
        Target polling rate.  1.0 Hz matches the 1-Hz training data exactly.
        Going faster (e.g. 2.0) increases row count but may exceed what cheap
        ELM327 adapters can sustain across 14 PIDs — check with live_discover.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        sample_hz: float = 1.0,
    ) -> None:
        self._port = port
        self._sample_interval = 1.0 / max(0.1, sample_hz)

        self._conn: Optional[obd.OBD] = None
        self._supported_pids: list[str] = []  # subset of USEFUL_PIDS the ECU exposes

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._elapsed_s: int = 0
        self._connected: bool = False
        self._session_id: str = f"live_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"

        # Track actual measured poll rate for the discover script
        self._last_poll_duration_s: float = 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = _CONNECT_TIMEOUT_S) -> bool:
        """Attempt to connect to the ELM327 adapter and the car ECU.

        Parameters
        ----------
        timeout : float
            Seconds to wait for the ECU to respond before giving up.

        Returns
        -------
        bool  — True if python-OBD reports CAR_CONNECTED status.
        """
        log.info("Connecting to ELM327 (port=%s, timeout=%ss)…", self._port, timeout)
        try:
            self._conn = obd.OBD(
                portstr=self._port,
                fast=False,
                timeout=min(timeout, _CONNECT_TIMEOUT_S),
            )
        except Exception as exc:
            log.warning("OBD connection failed: %s", exc)
            self._connected = False
            return False

        if self._conn.status() == OBDStatus.CAR_CONNECTED:
            self._supported_pids = self._discover_pids()

            # Verify the engine is actually running before accepting the connection.
            # Cheap clones report CAR_CONNECTED in ACC-only key position.
            if not self._verify_engine_running():
                self._connected = False
                return False

            self._connected = True
            log.info(
                "Connected. %d / %d PIDs supported by ECU: %s",
                len(self._supported_pids),
                len(USEFUL_PIDS),
                self._supported_pids,
            )
            return True

        log.warning("ELM327 responded but no ECU detected (status=%s).", self._conn.status())
        self._connected = False
        return False

    def start(self) -> None:
        """Spawn the background poll thread.

        Call ``connect()`` first; ``start()`` does nothing if not connected.
        """
        if not self._connected:
            log.warning("start() called before connect() succeeded — ignoring.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="obd-poll",
            daemon=True,  # won't block process exit
        )
        self._thread.start()
        log.info("OBD poll thread started at %.1f Hz.", 1.0 / self._sample_interval)

    def stop(self) -> None:
        """Signal the poll thread to stop and close the serial port.

        Blocks until the thread exits (max ~2× sample interval).
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._sample_interval * 3))
            self._thread = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._connected = False
        log.info("OBD poll thread stopped.")

    # ------------------------------------------------------------------
    # Streaming API (mirrors CsvStreamer)
    # ------------------------------------------------------------------

    def next_row(self) -> Optional[dict[str, float]]:
        """Non-blocking pop of the latest polled row.

        Returns None if no fresh row is available yet (adapter busy,
        or not started).  This is NOT "exhausted" — a live source is
        open-ended.  The dashboard should treat None as "no new data
        this rerun" and re-render the previous state.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """Drain the queue and reset the elapsed counter.

        Call between baseline-capture passes or when replaying a segment.
        Does NOT stop the poll thread.
        """
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._elapsed_s = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True when the poll thread has an active ECU connection."""
        return self._connected

    @property
    def supported_pids(self) -> list[str]:
        """Canonical PID names the ECU confirmed support for."""
        return list(self._supported_pids)

    @property
    def missing_pids(self) -> list[str]:
        """PIDs in USEFUL_PIDS that the ECU did NOT support."""
        return [p for p in USEFUL_PIDS if p not in self._supported_pids]

    @property
    def elapsed_s(self) -> int:
        """Rows produced so far (proxy for elapsed seconds at 1 Hz)."""
        return self._elapsed_s

    @property
    def session_id(self) -> str:
        """Unique identifier for this live session (timestamp-based)."""
        return self._session_id

    @property
    def exhausted(self) -> bool:
        """Always False — a live source never runs out of rows."""
        return False

    @property
    def measured_poll_hz(self) -> float:
        """Actual measured poll rate from the last completed tick.

        Below 0.8 Hz indicates an overloaded or slow adapter — check with
        ``scripts/live_discover.py`` before relying on inference results.
        """
        if self._last_poll_duration_s <= 0:
            return 0.0
        return 1.0 / self._last_poll_duration_s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_pids(self) -> list[str]:
        """Ask the ECU which of our 14 PIDs it supports."""
        supported = []
        for pid_name, cmd in PID_MAP.items():
            try:
                if self._conn.supports(cmd):
                    supported.append(pid_name)
            except Exception:
                pass
        return supported

    def _verify_engine_running(self) -> bool:
        """Return True only if ENGINE_RPM is ≥ 50 (engine actually running).

        Guards against cheap ELM327 clones that report CAR_CONNECTED in
        ACC-only key position — the ECU responds to the bus but returns null
        or near-zero RPM because the engine is not running.  Shared by
        connect() and _try_reconnect() so both paths get the same check.

        Returns True when ENGINE_RPM is not in the supported PID list
        (cannot verify, so we allow the connection to proceed).
        """
        if "ENGINE_RPM" not in self._supported_pids:
            return True  # ECU doesn't expose RPM — can't verify either way
        import math
        try:
            resp = self._conn.query(PID_MAP["ENGINE_RPM"])
            rpm = to_float(resp)
            if math.isnan(rpm) or rpm < 50:
                log.warning(
                    "ENGINE_RPM=%s — ignition not in RUN mode or ECU not awake. "
                    "Start the engine before connecting.",
                    rpm,
                )
                return False
        except Exception as exc:
            log.warning("ENGINE_RPM liveness check failed: %s", exc)
            return False
        return True

    def _try_reconnect(self) -> bool:
        """Attempt reconnection from within the poll thread."""
        log.info("OBD poll: attempting reconnect…")
        try:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = obd.OBD(portstr=self._port, fast=False, timeout=5.0)
            if self._conn.status() == OBDStatus.CAR_CONNECTED:
                self._supported_pids = self._discover_pids()
                # Same RPM liveness guard as connect() — avoids feeding garbage
                # into the classifier after a Bluetooth drop in ACC-mode ECUs.
                if not self._verify_engine_running():
                    self._connected = False
                    return False
                self._connected = True
                log.info("OBD reconnected. Supported PIDs: %d", len(self._supported_pids))
                return True
        except Exception as exc:
            log.debug("Reconnect failed: %s", exc)
        self._connected = False
        return False

    def _poll_loop(self) -> None:
        """Background thread: poll all supported PIDs at sample_hz."""
        while not self._stop_event.is_set():
            t_start = time.monotonic()

            # Auto-reconnect if the connection dropped
            if not self._connected:
                if not self._try_reconnect():
                    self._stop_event.wait(timeout=_RECONNECT_INTERVAL_S)
                    continue

            # Build one row dict
            row: dict[str, float] = {}
            for pid_name in self._supported_pids:
                cmd = PID_MAP.get(pid_name)
                if cmd is None:
                    row[pid_name] = float("nan")
                    continue
                try:
                    response = self._conn.query(cmd)
                    row[pid_name] = to_float(response)
                except Exception as exc:
                    log.debug("Query failed for %s: %s", pid_name, exc)
                    row[pid_name] = float("nan")

            # Fill PIDs the ECU doesn't support with NaN (not zero)
            for pid_name in USEFUL_PIDS:
                if pid_name not in row:
                    row[pid_name] = float("nan")

            # Attach wall-clock timestamp so InferenceEngine can resample to
            # exactly 1 row/second regardless of raw adapter poll rate.
            # CsvStreamer rows do NOT carry __t — that's how the engine
            # distinguishes live from CSV path (None → CSV, float → live).
            row["__t"] = time.monotonic()

            # Push latest row; drop previous if consumer hasn't read it yet
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                pass  # extremely rare race; skip this tick

            self._elapsed_s += 1
            poll_duration = time.monotonic() - t_start
            self._last_poll_duration_s = poll_duration

            # Sleep the remainder of the sample interval (interruptible by stop())
            remaining = self._sample_interval - poll_duration
            if remaining > 0:
                self._stop_event.wait(timeout=remaining)
