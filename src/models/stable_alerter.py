"""Temporal voting filter to suppress single-window false positives.

Problem
-------
A raw per-window classifier fires an alert the instant one 60-second
window looks anomalous — but road bumps, brief throttle blips, and
sensor glitches all produce single-window spikes that disappear on the
next window.  In a real vehicle this causes the driver to see an alert
flash on and off, which destroys trust.

Fix: sliding majority vote
--------------------------
Keep a rolling buffer of the last N predictions.  Only fire an alert
(and hold it) when:
  1.  The *same* fault class wins a majority of the last N windows, AND
  2.  The classifier's confidence for that class exceeds ``min_confidence``
      on at least one of those winning windows.

This is equivalent to the "3-strikes" rule used in production OBD
scan-tool firmware, generalised to any window count.

Hysteresis
----------
Once an alert is raised it stays raised until at least one window
predicts "healthy" with confidence > ``clear_confidence``.  This prevents
the alert from flickering while the fault is clearly present.

Fault-to-fault transition
-------------------------
If all N windows in the buffer unanimously agree on a *different* fault
(same majority-vote rule as firing), the active alert transitions directly
to the new fault without going through a healthy intermediate.  This covers
the realistic scenario where one fault develops on top of another during a
live drive.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque

if TYPE_CHECKING:
    from src.diagnostics.cold_start_checker import ColdStartAlert

# Labels that are NOT faults and must not trigger fault alerts.
# "cold_start" is a normal operating regime (engine warming up), not a fault.
_NON_FAULT_LABELS: frozenset[str] = frozenset({"healthy", "cold_start"})


@dataclass
class AlertState:
    """Current alert output from the StableAlerter."""

    active: bool
    fault_type: str          # "healthy" when no alert
    confidence: float        # max confidence seen in the voting window
    windows_voted: int       # how many windows agreed
    # Rule-engine alerts are kept separately — they bypass the voting filter
    # because they are deterministic, not probabilistic.
    rule_alerts: list = field(default_factory=list)  # list[ColdStartAlert]


class StableAlerter:
    """Temporal voting filter wrapping a per-window classifier.

    Parameters
    ----------
    min_windows : int
        Minimum consecutive (or majority) windows that must agree before
        an alert fires.  Default 3 (30 seconds at 10 s stride).
    min_confidence : float
        Minimum softmax confidence required on at least one agreeing window.
        Default 0.70 (deliberately modest — classifier is well-calibrated).
    clear_confidence : float
        Minimum confidence for a "healthy" prediction to clear an active alert.
        Slightly higher than min_confidence to add hysteresis.
    """

    def __init__(
        self,
        min_windows: int = 3,
        min_confidence: float = 0.70,
        clear_confidence: float = 0.80,
    ) -> None:
        if min_windows < 1:
            raise ValueError("min_windows must be >= 1")
        if not (0.0 < min_confidence < 1.0):
            raise ValueError("min_confidence must be in (0, 1)")
        if not (0.0 < clear_confidence < 1.0):
            raise ValueError("clear_confidence must be in (0, 1)")

        self._min_windows = min_windows
        self._min_conf = min_confidence
        self._clear_conf = clear_confidence

        # Each entry: (predicted_label, confidence)
        self._buffer: Deque[tuple[str, float]] = deque(maxlen=min_windows)
        self._current_alert: AlertState = AlertState(
            active=False, fault_type="healthy", confidence=0.0, windows_voted=0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, predicted_label: str, confidence: float) -> AlertState:
        """Ingest one new window prediction and return the updated alert state.

        Parameters
        ----------
        predicted_label : str
            The classifier's top-1 label for this window
            (e.g. "healthy", "fuel_system").
        confidence : float
            Softmax probability for the predicted class (0–1).

        Returns
        -------
        AlertState
            Current alert status after this update.
        """
        self._buffer.append((predicted_label, confidence))

        if self._current_alert.active:
            self._current_alert = self._try_clear()
        else:
            self._current_alert = self._try_fire()

        return self._current_alert

    def ingest_rule_alert(self, rule_alert: "ColdStartAlert") -> AlertState:
        """Ingest one rule-engine alert and attach it to the current state.

        Rule-engine alerts bypass the temporal voting filter entirely — they
        are deterministic (high-confidence, fired once) and should reach the
        dashboard immediately.  They are stored alongside the ML alert state
        so the dashboard has one place to query both streams.

        Parameters
        ----------
        rule_alert : ColdStartAlert
            An alert from ColdStartChecker.

        Returns
        -------
        AlertState
            Updated state carrying the new rule alert.
        """
        updated_rule_alerts = list(self._current_alert.rule_alerts) + [rule_alert]
        self._current_alert = AlertState(
            active=self._current_alert.active,
            fault_type=self._current_alert.fault_type,
            confidence=self._current_alert.confidence,
            windows_voted=self._current_alert.windows_voted,
            rule_alerts=updated_rule_alerts,
        )
        return self._current_alert

    @property
    def state(self) -> AlertState:
        """Current alert state without consuming a new prediction."""
        return self._current_alert

    def reset(self) -> None:
        """Clear the buffer and alert state (e.g. between sessions)."""
        self._buffer.clear()
        self._current_alert = AlertState(
            active=False, fault_type="healthy", confidence=0.0, windows_voted=0
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_fire(self) -> AlertState:
        """Check if buffer justifies firing a NEW alert."""
        if len(self._buffer) < self._min_windows:
            # Not enough history yet
            return self._current_alert

        # Compute threshold first — the fault-window filter uses the same value.
        # floor(N/2)+1 means: with min_windows=3, need ≥2 fault windows total AND
        # ≥2 of them must agree on the same fault class.
        majority_threshold = self._min_windows // 2 + 1
        fault_windows = [(lbl, conf) for lbl, conf in self._buffer if lbl not in _NON_FAULT_LABELS]
        if len(fault_windows) < majority_threshold:
            return self._current_alert  # not enough fault windows for a majority

        # Simple majority of those fault windows must agree on the SAME fault type.
        labels = [lbl for lbl, _ in fault_windows]
        dominant = max(set(labels), key=labels.count)
        count = labels.count(dominant)
        if count < majority_threshold:
            return self._current_alert  # split between different fault types

        max_conf = max(conf for lbl, conf in fault_windows if lbl == dominant)
        if max_conf < self._min_conf:
            return self._current_alert  # not confident enough

        return AlertState(
            active=True,
            fault_type=dominant,
            confidence=max_conf,
            windows_voted=count,
        )

    def _try_clear(self) -> AlertState:
        """Check if a healthy window clears the alert, or transitions to a new fault.

        Two exit paths:
        1. High-confidence healthy → clear the alert entirely.
        2. All N windows unanimously point to a DIFFERENT fault → transition
           to that fault directly (same majority-vote rule as _try_fire).
        """
        # Path 1: high-confidence non-fault label clears the alert.
        # "cold_start" is a normal operating regime, not a fault — it must clear
        # an active alert just as "healthy" does, or a post-cold-start drive could
        # never suppress a false positive that fired during warm-up.
        last_label, last_conf = self._buffer[-1]
        if last_label in _NON_FAULT_LABELS and last_conf >= self._clear_conf:
            return AlertState(
                active=False, fault_type="healthy", confidence=last_conf, windows_voted=1
            )

        # Path 2: simple-majority different-fault buffer → direct transition.
        # Uses the same floor(N/2)+1 threshold as _try_fire so a single
        # misclassified window does not accidentally switch the fault type.
        if len(self._buffer) >= self._min_windows:
            fault_windows = [(lbl, conf) for lbl, conf in self._buffer if lbl not in _NON_FAULT_LABELS]
            majority_threshold = self._min_windows // 2 + 1
            if len(fault_windows) >= majority_threshold:
                labels = [lbl for lbl, _ in fault_windows]
                dominant = max(set(labels), key=labels.count)
                count = labels.count(dominant)
                if count >= majority_threshold and dominant != self._current_alert.fault_type:
                    max_conf = max(conf for lbl, conf in fault_windows if lbl == dominant)
                    if max_conf >= self._min_conf:
                        return AlertState(
                            active=True,
                            fault_type=dominant,
                            confidence=max_conf,
                            windows_voted=count,
                        )

        # Alert persists unchanged
        return self._current_alert
