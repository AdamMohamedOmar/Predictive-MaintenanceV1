"""Per-row inference pipeline for the live dashboard.

Architecture
------------
InferenceEngine wraps all five ML components into a single ``update(row)``
call that the Streamlit loop invokes once per row:

  CsvStreamer  →  InferenceEngine.update(row)  →  DashboardState
                       │
                       ├─ ColdStartChecker  (every row, deterministic rules)
                       ├─ rolling 60-row buffer
                       │      └─ every 10 rows (WINDOW_STRIDE_S):
                       │            ├─ extract_features()
                       │            ├─ XGBClassifier  → label + probabilities
                       │            ├─ SHAPExplainer  → top-5 features
                       │            ├─ FaultForecaster → 4× severity 60 s ahead
                       │            └─ compute_severity() → current severity ×4
                       └─ StableAlerter  (3-window majority vote + hysteresis)

Why update every row but classify every 10 rows?
-------------------------------------------------
The classifier operates on 60-second windows; re-running it every single
row (1 Hz) would be wasteful — the window changes by only 1 row.  The
stride of 10 matches what the dataset was built with, so feature
distributions seen at inference match the training distribution.

The ColdStartChecker, however, runs at 1 Hz because its rules need
second-level precision (e.g. "coolant was flat for exactly 90 seconds").

Healthy baselines for severity computation
------------------------------------------
``compute_severity()`` requires the vehicle's healthy baseline for STFT,
LTFT, and throttle-to-pedal ratio.  We derive these from the
StandardScaler that was fitted on healthy training windows: its ``mean_``
vector is exactly the per-feature healthy mean.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import MODELS_DIR, WINDOW_LENGTH_S, WINDOW_STRIDE_S
from src.diagnostics.cold_start_checker import ColdStartAlert, ColdStartChecker
from src.features.extractor import extract_features, feature_names
from src.features.severity import compute_severity
from src.models.anomaly import AnomalyDetector
from src.models.classifier import ALL_LABELS
from src.models.explainer import SHAPExplainer
from src.models.forecaster import FAULT_TYPES, FaultForecaster
from src.models.stable_alerter import AlertState, StableAlerter
from src.models.xgb_classifier import load_model as load_xgb_model

log = logging.getLogger(__name__)


# ── DashboardState ────────────────────────────────────────────────────────────


@dataclass
class DashboardState:
    """Single snapshot of everything the dashboard needs to render one frame."""

    elapsed_s: int
    latest_row: dict[str, float]

    # True once ≥ WINDOW_LENGTH_S rows have accumulated in the buffer.
    # The dashboard shows "warming up…" until this is True.
    buffer_ready: bool

    # Last classifier output (updated every WINDOW_STRIDE_S rows)
    classifier_label: str  # e.g. "fuel_system" or "healthy"
    classifier_confidence: float  # softmax prob for the predicted class
    all_class_probs: dict[str, float]  # {label: prob} for all 6 classes

    # Current fault severity in [0, 1] for each of the 4 fault types.
    # Computed from physics formulas, not the ML model.
    severities: dict[str, float]  # {fault_type: 0–1}

    # Forecaster output: predicted severity 60 s from now
    forecasts: dict[str, float]  # {fault_type: 0–1}

    # Temporal voting filter output (ML stream + rule stream)
    stable_alert: AlertState

    # Rule-engine alerts (kept separately for display)
    rule_alerts: list  # list[ColdStartAlert]

    # Top SHAP features for the current prediction (list of (name, value) pairs)
    top_features: list  # list[tuple[str, float]]

    # Input sanity check result — False when the adapter sent a physically impossible
    # row (e.g. RPM=0 + speed=50, fuel trims at ±75%).  Inference is skipped and
    # the last-known state is held until a good row arrives.
    data_quality_ok: bool = True
    data_quality_violations: list = field(default_factory=list)

    # One-class anomaly score in [0, 1] from the IsolationForest detector.
    # 0.0 when the detector is not loaded (model file missing) or the
    # buffer is still warming up — same convention as severities.
    anomaly_score: float = 0.0


def _initial_state() -> DashboardState:
    """Return a blank state used before the first window is ready."""
    blank_probs = {label: 0.0 for label in ALL_LABELS}
    blank_probs["healthy"] = 1.0
    blank_severities = {ft: 0.0 for ft in FAULT_TYPES}
    blank_alert = AlertState(
        active=False, fault_type="healthy", confidence=0.0, windows_voted=0
    )
    return DashboardState(
        elapsed_s=0,
        latest_row={},
        buffer_ready=False,
        classifier_label="warming_up",
        classifier_confidence=0.0,
        all_class_probs=blank_probs,
        severities=dict(blank_severities),
        forecasts=dict(blank_severities),
        stable_alert=blank_alert,
        rule_alerts=[],
        top_features=[],
    )


# ── InferenceEngine ───────────────────────────────────────────────────────────


class InferenceEngine:
    """Stateful per-row inference pipeline.

    Load once per Streamlit session with ``@st.cache_resource`` so the
    (expensive) SHAP TreeExplainer is built only once.

    Parameters
    ----------
    models_dir : Path or None
        Directory containing ``xgb_classifier_v1.pkl`` and
        ``forecaster_v1.pkl``.  Defaults to ``src.config.MODELS_DIR``.
    normalizer_override : Path or None
        If given, this normaliser replaces the one bundled with the
        XGBoost model.  Pass the path saved by live_baseline_capture.py
        for cross-vehicle inference (e.g. Skoda Roomster baseline).
        The classifier weights are unchanged — only the z-scoring
        reference distribution changes.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        normalizer_override: Optional[Path] = None,
    ) -> None:
        models_dir = Path(models_dir or MODELS_DIR)

        log.info("InferenceEngine: loading XGBoost classifier…")
        clf, norm = load_xgb_model(models_dir)
        self._clf = clf

        # Swap in the vehicle-specific normaliser if provided.
        # This is the cross-vehicle generalisation knob: same classifier,
        # different z-score baseline.
        if normalizer_override is not None:
            log.info(
                "InferenceEngine: loading normaliser override from %s",
                normalizer_override,
            )
            from src.features.normalizer import BaselineNormalizer

            norm = BaselineNormalizer.load(normalizer_override)
        self._norm = norm

        log.info("InferenceEngine: building SHAP TreeExplainer…")
        self._explainer = SHAPExplainer(clf)

        log.info("InferenceEngine: loading FaultForecaster…")
        self._forecaster = FaultForecaster.load(models_dir)

        # Optional one-class anomaly detector.  Loaded gracefully — if the
        # artefact is missing (fresh clone, model not yet trained), the
        # dashboard still runs and the anomaly panel shows "not loaded".
        anomaly_path = models_dir / "isolation_forest_v1.pkl"
        if anomaly_path.exists():
            log.info("InferenceEngine: loading AnomalyDetector…")
            try:
                self._anomaly: AnomalyDetector | None = AnomalyDetector.load(anomaly_path)
            except Exception as exc:  # pragma: no cover — corrupt artefact
                log.warning("AnomalyDetector load failed (%s); panel disabled", exc)
                self._anomaly = None
        else:
            log.info("InferenceEngine: AnomalyDetector artefact not found — panel disabled")
            self._anomaly = None
        # Share the same normalizer between classifier and forecaster.
        # Critical for cross-vehicle inference (Skoda normalizer_override): without
        # this, the forecaster still z-scores against the Etios healthy baseline
        # even when the caller has supplied a vehicle-specific normalizer.
        self._forecaster._norm = self._norm

        # Derive healthy baselines from the fitted scaler.
        # Works for both the training scaler and any override scaler because
        # BaselineNormalizer always stores a StandardScaler internally.
        feat_cols = feature_names()
        mean_arr = norm.feature_means
        self._baselines: dict[str, float] = {
            feat: float(mean_arr[i]) for i, feat in enumerate(feat_cols)
        }

        # Per-session stateful components (reset between files)
        self._cold_start = ColdStartChecker()
        self._alerter = StableAlerter()
        self._buffer: deque[dict] = deque(maxlen=WINDOW_LENGTH_S)
        self._rows_since_window: int = 0
        self._elapsed_s: int = 0
        self._last_state: DashboardState = _initial_state()
        # Tracks which NaN-filled features have already been logged so we warn
        # once per feature rather than spamming the log every window.
        self._nan_warned: set[str] = set()
        # Actual ECU poll rate — 1.0 for CSV replay (carOBD is 1 Hz); updated
        # each live tick via set_sample_hz() so rate-dependent features are correct.
        # After T3.1 the resampler ensures the buffer always sees 1-Hz rows,
        # so set_sample_hz() is called with 1.0 from app.py post-resampling.
        self._sample_hz: float = 1.0
        # T3.1 resampler: tracks the next 1-second tick boundary (monotonic).
        # None until the first live row arrives (CSV rows have no __t key).
        self._next_sample_t: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, row: dict[str, float]) -> DashboardState:
        """Ingest one raw row and return the updated state.

        Live path (row carries ``__t = time.monotonic()``):
            Rows are resampled to exactly 1-per-second using hold-last
            semantics.  A fast adapter (> 1 Hz) drops extra rows; a slow
            adapter (< 1 Hz) replays the last row at each missed 1-second
            slot so the 60-row buffer stays time-aligned with training data.

        CSV path (no ``__t`` key):
            Passes straight through to ``_process_one_row()`` — identical
            behaviour to the pre-T3.1 code path.

        Parameters
        ----------
        row : dict[str, float]
            OBD-II sensor readings.  Live rows include ``__t`` (wall-clock
            float); CSV rows do not.

        Returns
        -------
        DashboardState
            Most-recent computed state.
        """
        t = row.pop("__t", None)

        if t is not None:
            # ── Live path: resample to 1-Hz ───────────────────────────────
            if self._next_sample_t is None:
                self._next_sample_t = t  # anchor to first row's timestamp

            if t < self._next_sample_t:
                # Row arrived faster than 1 Hz — drop it.
                return self._last_state

            # Hold-last semantics: if the adapter is slow (< 1 Hz), the same
            # row fills every missed 1-second slot so the buffer length in
            # rows always equals elapsed seconds.
            while t >= self._next_sample_t:
                self._next_sample_t += 1.0
                self._process_one_row(row)

            return self._last_state

        # ── CSV path: no resampling needed ────────────────────────────────
        self._process_one_row(row)
        return self._last_state

    def _process_one_row(self, row: dict[str, float]) -> None:
        """Core per-row pipeline — called by update() after resampling.

        Mutates ``self._last_state`` in place so the resampler loop can call
        it 0–N times per incoming raw row without extra plumbing.
        """
        self._elapsed_s += 1

        # ── 0. Physics sanity check — skip inference on impossible rows ──
        # Cheap Bluetooth clones occasionally deliver garbage (RPM=0 + speed=50,
        # fuel trims at ±75%).  Classifying garbage produces confident false alarms.
        from src.dashboard.sanity import check_row
        verdict = check_row(row)
        if not verdict.ok:
            prev = self._last_state
            self._last_state = DashboardState(
                elapsed_s=self._elapsed_s,
                latest_row=row,
                buffer_ready=prev.buffer_ready,
                classifier_label=prev.classifier_label,
                classifier_confidence=prev.classifier_confidence,
                all_class_probs=prev.all_class_probs,
                severities=prev.severities,
                forecasts=prev.forecasts,
                stable_alert=prev.stable_alert,
                rule_alerts=prev.rule_alerts,
                top_features=prev.top_features,
                data_quality_ok=False,
                data_quality_violations=verdict.violations,
                anomaly_score=prev.anomaly_score,
            )
            return

        # ── 1. ColdStartChecker runs at 1 Hz (needs second-level resolution) ──
        # row.get(key, default) returns NaN (not the default) when the key exists
        # with a NaN value — e.g. an ECU that doesn't expose that PID.
        # Replace NaN with safe defaults so the warmup timer doesn't fire false alerts.
        raw_coolant = row.get("COOLANT_TEMPERATURE", 90.0)
        raw_rpm = row.get("ENGINE_RPM", 800.0)
        raw_speed = row.get("VEHICLE_SPEED", 0.0)
        raw_voltage = row.get("CONTROL_MODULE_VOLTAGE", 14.0)
        new_rule_alerts = self._cold_start.update(
            coolant=90.0 if math.isnan(raw_coolant) else raw_coolant,
            rpm=800.0 if math.isnan(raw_rpm) else raw_rpm,
            speed=0.0 if math.isnan(raw_speed) else raw_speed,
            voltage=14.0 if math.isnan(raw_voltage) else raw_voltage,
            now=time.monotonic(),
        )
        for ra in new_rule_alerts:
            self._alerter.ingest_rule_alert(ra)

        # ── 2. Add row to rolling 60-row buffer ──
        self._buffer.append(row)
        self._rows_since_window += 1
        buffer_ready = len(self._buffer) >= WINDOW_LENGTH_S

        # ── 3. Run window inference every WINDOW_STRIDE_S rows ──
        if buffer_ready and self._rows_since_window >= WINDOW_STRIDE_S:
            self._rows_since_window = 0
            self._last_state = self._run_window(row, buffer_ready)
        else:
            # Between windows: update elapsed_s, latest_row, and alert state
            # (alert state can change on every row via ColdStartChecker updates)
            prev = self._last_state
            self._last_state = DashboardState(
                elapsed_s=self._elapsed_s,
                latest_row=row,
                buffer_ready=buffer_ready,
                classifier_label=prev.classifier_label
                if buffer_ready
                else "warming_up",
                classifier_confidence=prev.classifier_confidence,
                all_class_probs=prev.all_class_probs,
                severities=prev.severities,
                forecasts=prev.forecasts,
                stable_alert=self._alerter.state,
                rule_alerts=list(self._alerter.state.rule_alerts),
                top_features=prev.top_features,
                data_quality_ok=True,
                data_quality_violations=[],
                anomaly_score=prev.anomaly_score,
            )

    def reset(self) -> None:
        """Clear all session state.  Call between CSV files."""
        self._cold_start.reset()
        self._alerter.reset()
        self._buffer.clear()
        self._rows_since_window = 0
        self._elapsed_s = 0
        self._last_state = _initial_state()
        self._nan_warned.clear()
        self._next_sample_t = None  # T3.1: restart resampler clock on session change

    def set_sample_hz(self, hz: float) -> None:
        """Update the ECU poll rate used by rate-dependent features.

        Call once per live tick (before update()) so COOLANT_WARMUP_RATE and
        FUEL_LOOP_ACTIVE stay calibrated as the adapter's measured throughput
        changes.  No-op for CSV replay (default 1.0 Hz is correct for carOBD).

        ``hz < 0.1`` means the adapter has not completed its first tick yet
        (``measured_poll_hz`` returns 0.0 on the first connect).  Treat this
        as "not yet known" and keep the previous value (1.0 default) rather
        than clipping to 0.05 Hz and producing wildly wrong time-axis features
        on the very first window.
        """
        if hz < 0.1:
            return  # adapter has not completed a tick yet — keep previous value
        import numpy as np
        self._sample_hz = float(np.clip(hz, 0.05, 5.0))

    @property
    def current_state(self) -> DashboardState:
        """Last computed state without consuming a new row."""
        return self._last_state

    @property
    def degraded_pid_count(self) -> int:
        """Count of distinct PIDs that have ever been NaN-filled.

        Each PID contributes 5 features (mean/std/min/max/delta), so dividing
        the warned-feature count by 5 gives an approximate PID count.
        Cross-PID and regime features (not multiples of 5) are excluded from
        the threshold check — only standard PID features are counted.
        """
        return len(self._nan_warned) // 5

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_window(self, latest_row: dict, buffer_ready: bool) -> DashboardState:
        """Run the full ML pipeline on the current 60-row buffer."""
        window_df = pd.DataFrame(list(self._buffer))

        # Feature extraction — pass real poll rate so time-axis features are correct
        feats = extract_features(window_df, sample_hz=self._sample_hz)

        # NaN-fill: when a PID is unsupported by the ECU its column is all NaN,
        # which propagates through mean/std/etc. in extract_features.  Substitute
        # the scaler's healthy mean so the z-score becomes 0 ("no signal → nominal").
        # Warn once per feature so the user can see which PIDs are missing without
        # the log being flooded on every window.
        nan_features = [k for k, v in feats.items() if math.isnan(v)]
        if nan_features:
            first_time = [k for k in nan_features if k not in self._nan_warned]
            if first_time:
                log.warning(
                    "NaN features filled with healthy baseline (unsupported PIDs): %s",
                    first_time,
                )
                self._nan_warned.update(first_time)
            for k in nan_features:
                feats[k] = self._baselines.get(k, 0.0)

        feats_df = pd.DataFrame([feats])

        # Classification + SHAP (explain_window normalises internally)
        try:
            result = self._explainer.explain_window(feats_df, self._norm, top_n=5)
            label = result["predicted_label"]
            probs = result["probabilities"]
            confidence = float(probs[label])
            top_features = result["top_features"]
        except Exception as exc:
            # SHAP failed (e.g. library version mismatch at demo time).
            # Fall back to raw clf.predict_proba() so the ML prediction is still
            # correct — only the explanation is lost.  NEVER silently return
            # "healthy" with 100% confidence here; a real fault would be missed.
            log.warning("SHAP explain failed (%s) — falling back to clf.predict_proba()", exc)
            try:
                from src.features.normalizer import normalised_feature_names as _nfn
                _fcols = _nfn()
                X_fb = self._norm.transform(feats_df)[_fcols].to_numpy(dtype=float)
                proba = self._clf.predict_proba(X_fb)[0]
                pred_id = int(proba.argmax())
                label = ALL_LABELS[pred_id]
                probs = {lbl: float(proba[i]) for i, lbl in enumerate(ALL_LABELS)}
                confidence = float(proba[pred_id])
                top_features = []
            except Exception as exc2:
                # Both SHAP and clf fallback failed.  Hold the last-known prediction
                # rather than silently reporting "healthy" — a stale label is safer
                # than a missed fault during a live Skoda demo.
                log.error(
                    "clf.predict_proba fallback also failed (%s) — holding last prediction",
                    exc2,
                )
                prev = self._last_state
                label = prev.classifier_label
                probs = prev.all_class_probs
                confidence = prev.classifier_confidence
                top_features = prev.top_features

        # Update temporal voting filter
        self._alerter.update(label, confidence)

        # Current severity (physics formulas)
        severities: dict[str, float] = {}
        for fault in FAULT_TYPES:
            try:
                severities[fault] = compute_severity(feats, fault, self._baselines)
            except (ValueError, KeyError):
                severities[fault] = 0.0

        # Forecaster trained on POST-ONSET windows only — calling it on healthy
        # data extrapolates off-distribution and produces phantom severities
        # (observed 0.91 on multiple clean carOBD sessions during expert review).
        # Skip the forecast on healthy / cold_start / warming_up labels.
        if label in ("healthy", "cold_start", "warming_up"):
            forecasts = {fault: 0.0 for fault in FAULT_TYPES}
        else:
            try:
                forecasts = self._forecaster.predict_all(feats)
            except Exception as exc:
                log.warning("predict_all failed (%s) — zeroing forecasts", exc)
                forecasts = {fault: 0.0 for fault in FAULT_TYPES}

        # One-class anomaly score (independent of the classifier label).
        # 0.0 when the detector is not loaded — the dashboard panel reads
        # this as "model not available" rather than "definitely healthy."
        anomaly_score = 0.0
        if self._anomaly is not None:
            try:
                anomaly_score = float(self._anomaly.score(feats, self._norm))
            except Exception as exc:
                log.warning("anomaly score failed (%s) — leaving at 0.0", exc)

        return DashboardState(
            elapsed_s=self._elapsed_s,
            latest_row=latest_row,
            buffer_ready=buffer_ready,
            classifier_label=label,
            classifier_confidence=confidence,
            all_class_probs=probs,
            severities=severities,
            forecasts=forecasts,
            stable_alert=self._alerter.state,
            rule_alerts=list(self._alerter.state.rule_alerts),
            top_features=top_features,
            data_quality_ok=True,
            data_quality_violations=[],
            anomaly_score=anomaly_score,
        )
