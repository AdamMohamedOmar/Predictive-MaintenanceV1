"""Record a healthy-drive baseline for cross-vehicle normaliser calibration.

The XGBoost classifier was trained on z-scored features computed relative to
the Etios healthy distribution.  To run on a different vehicle (e.g. Skoda
Roomster), we need the SAME classifier but a NEW normaliser fit on that
vehicle's own healthy windows.

This script collects N minutes of normal driving, windows the data exactly
as the training pipeline does, and fits + saves a BaselineNormalizer.

Usage
-----
    python -m scripts.live_baseline_capture --port COM3
    python -m scripts.live_baseline_capture --port COM3 --duration-min 5 \\
        --out models/skoda_normalizer.pkl --vehicle "Skoda Roomster 2007 1.4"

Driving instructions (printed at start)
----------------------------------------
  • Drive at normal city/road speeds — aim for at least some highway km.
  • Vary throttle naturally: don't hold fixed cruise control the whole time.
  • Let the engine reach operating temperature BEFORE starting (check the
    dashboard temperature gauge is in the normal band — not still rising).
  • Avoid deliberate fault simulation during baseline capture.

Guards (the script refuses to save if these fail)
--------------------------------------------------
  1. Coolant must reach >= 75°C at some point during capture.
     (Cold-engine baseline biases TIMING_VS_TEMP and cold-start features.)
  2. Mean vehicle speed must be >= 15 km/h.
     (Pure idle baseline biases THROTTLE_TO_PEDAL_RATIO and MAP_PER_THROTTLE.)
  3. At least 20 valid windows must be produced.
     (< 20 windows means < 3.5 min of data — not enough to fit a stable scaler.)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, USEFUL_PIDS, WINDOW_LENGTH_S, WINDOW_STRIDE_S
from src.features.extractor import extract_features, feature_names
from src.features.normalizer import BaselineNormalizer
from src.features.windowing import sliding_windows
from src.live.obd_source import LiveObdSource

# ── Guard thresholds ──────────────────────────────────────────────────────────
_MIN_COOLANT_TEMP = 75.0    # °C — engine must have reached operating temperature
_MIN_MEAN_SPEED = 15.0      # km/h — must include real driving (not just idle)
_MIN_WINDOWS = 20           # minimum valid windows for a stable scaler fit


# ── Pure processing (testable without hardware) ───────────────────────────────

def process_captured_rows(
    rows: list[dict],
    vehicle_name: str = "unknown",
    supported_pids: Optional[list[str]] = None,
    poll_hz: float = 1.0,
) -> tuple[BaselineNormalizer, dict]:
    """Fit a BaselineNormalizer from a list of raw OBD sensor rows.

    This function is hardware-free and fully testable.  The CLI calls it
    after collecting rows from LiveObdSource.

    Parameters
    ----------
    rows : list[dict[str, float]]
        Raw sensor rows, one per second, from ``LiveObdSource.next_row()``.
    vehicle_name : str
        Free-text label stored in the sidecar JSON (e.g. "Skoda Roomster 2007").
    supported_pids : list[str] or None
        PIDs the ECU exposed.  Used only for metadata; does not affect fitting.
    poll_hz : float
        Actual measured poll rate from the adapter (e.g. 0.3 for a slow Skoda
        ELM327).  Passed to ``extract_features`` so that rate-dependent features
        (COOLANT_WARMUP_RATE, FUEL_LOOP_ACTIVE) are calibrated on the correct
        time axis.  After T3.1 (1-Hz resampler in LiveObdSource), this should
        always be 1.0 because rows are resampled before reaching this function.

    Returns
    -------
    (BaselineNormalizer, metadata_dict)

    Raises
    ------
    ValueError
        If any of the three guard conditions fail (coolant, speed, windows).
    """
    if len(rows) < WINDOW_LENGTH_S:
        raise ValueError(
            f"Only {len(rows)} rows captured (need >= {WINDOW_LENGTH_S} for one window)."
        )

    df = pd.DataFrame(rows)

    # ── Guard 1: coolant temperature ──────────────────────────────────────────
    if "COOLANT_TEMPERATURE" in df.columns:
        max_coolant = float(df["COOLANT_TEMPERATURE"].dropna().max())
        if max_coolant < _MIN_COOLANT_TEMP:
            raise ValueError(
                f"Engine never reached {_MIN_COOLANT_TEMP}°C "
                f"(max observed: {max_coolant:.1f}°C).  "
                f"Wait until the temperature gauge is in the normal band "
                f"BEFORE starting baseline capture."
            )

    # ── Guard 2: mean vehicle speed ───────────────────────────────────────────
    if "VEHICLE_SPEED" in df.columns:
        mean_speed = float(df["VEHICLE_SPEED"].dropna().mean())
        if mean_speed < _MIN_MEAN_SPEED:
            raise ValueError(
                f"Mean vehicle speed {mean_speed:.1f} km/h is below "
                f"{_MIN_MEAN_SPEED} km/h.  The baseline must include real "
                f"driving, not just idling — THROTTLE_TO_PEDAL_RATIO and "
                f"MAP_PER_THROTTLE won't calibrate correctly at idle."
            )

    # ── Feature extraction (mirrors training pipeline exactly) ────────────────
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    feature_rows: list[dict] = []
    for window, _ in sliding_windows(df[pid_cols], label="healthy"):
        feats = extract_features(window, sample_hz=poll_hz)
        feature_rows.append(feats)

    # ── Guard 3: enough windows ───────────────────────────────────────────────
    if len(feature_rows) < _MIN_WINDOWS:
        raise ValueError(
            f"Only {len(feature_rows)} valid windows produced "
            f"(need >= {_MIN_WINDOWS}).  Drive for at least "
            f"{_MIN_WINDOWS * WINDOW_STRIDE_S // 60 + 1} minutes."
        )

    feat_df = pd.DataFrame(feature_rows)
    feat_df["label"] = "healthy"

    # Fill NaN features (from unsupported PIDs) with column mean, else 0.
    # This prevents StandardScaler from crashing on fully-absent PIDs.
    # Those features will be centred at 0 during inference — the least alarming
    # value — which is the correct behaviour for an unavailable sensor.
    for col in feature_names():
        if col in feat_df.columns and feat_df[col].isna().any():
            col_mean = feat_df[col].mean()
            feat_df[col] = feat_df[col].fillna(0.0 if pd.isna(col_mean) else col_mean)

    # ── Fit normaliser ────────────────────────────────────────────────────────
    norm = BaselineNormalizer()
    norm.fit(feat_df, healthy_label="healthy")

    # ── Metadata ──────────────────────────────────────────────────────────────
    feat_cols = feature_names()
    scaler_means = dict(zip(feat_cols, norm._scaler.mean_.tolist()))
    scaler_stds = dict(zip(feat_cols, np.sqrt(norm._scaler.var_).tolist()))

    metadata = {
        "vehicle": vehicle_name,
        "capture_date": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "duration_s": len(rows),
        "n_windows": len(feature_rows),
        "supported_pids": sorted(supported_pids or pid_cols),
        "missing_pids": [p for p in USEFUL_PIDS if p not in (supported_pids or pid_cols)],
        "feature_means": scaler_means,
        "feature_stds": scaler_stds,
    }
    return norm, metadata


# ── Hardware-touching shell ───────────────────────────────────────────────────

def run_capture(
    port: Optional[str],
    duration_min: float,
    out_path: Path,
    vehicle_name: str,
) -> int:
    """Connect, collect rows, process, save.  Returns shell exit code."""

    print("\n" + "=" * 60)
    print("  BASELINE CAPTURE — driving instructions")
    print("=" * 60)
    print("  1. Engine must be FULLY WARM before you start.")
    print("     Wait until the dashboard temperature gauge is in the")
    print("     normal band — NOT still rising from cold start.")
    print("  2. Drive at normal speeds. Include some variation:")
    print("     city streets, road, a roundabout. Not just a car park.")
    print("  3. Do NOT simulate any faults during this capture.")
    print("=" * 60)
    input("\n  Press ENTER when ready to begin capture…")

    print(f"\nConnecting to ELM327 (port={port or 'auto-detect'})…")
    src = LiveObdSource(port=port, sample_hz=1.0)
    if not src.connect():
        print("[FAIL] Could not connect.  Run live_discover.py first.")
        return 1

    duration_s = int(duration_min * 60)
    rows: list[dict] = []
    t_start = time.monotonic()
    t_end = t_start + duration_s

    src.start()
    print(f"\nCapturing {duration_min:.0f} min of data…  (Ctrl+C to abort)\n")
    try:
        while time.monotonic() < t_end:
            row = src.next_row()
            if row is not None:
                rows.append(row)
            else:
                time.sleep(0.05)

            elapsed = time.monotonic() - t_start
            remaining = max(0, duration_s - elapsed)
            print(
                f"\r  {elapsed:5.0f} s / {duration_s} s  |  "
                f"{len(rows)} rows  |  "
                f"{remaining:.0f} s remaining   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n\n  Capture interrupted.  Processing collected rows…")
    finally:
        src.stop()

    print(f"\n\n  Captured {len(rows)} rows.")

    # ── Process and save ──────────────────────────────────────────────────────
    try:
        norm, meta = process_captured_rows(
            rows,
            vehicle_name=vehicle_name,
            supported_pids=src.supported_pids,
            poll_hz=src.measured_poll_hz or 1.0,
        )
    except ValueError as exc:
        print(f"\n[FAIL] Guard check failed:\n  {exc}")
        print("\n  Redo the capture drive and try again.")
        return 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm.save(out_path)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Baseline saved:")
    print(f"     Normaliser : {out_path}")
    print(f"     Metadata   : {meta_path}")
    print(f"     Windows    : {meta['n_windows']}  |  Vehicle: {meta['vehicle']}")
    print(f"\n  Next step: launch the dashboard and select this normaliser.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a healthy baseline and save a vehicle normaliser."
    )
    parser.add_argument("--port", default=None,
                        help="ELM327 serial port (e.g. COM3). Omit to auto-detect.")
    parser.add_argument("--duration-min", type=float, default=5.0, metavar="N",
                        help="Minutes of healthy driving to capture (default: 5).")
    parser.add_argument(
        "--out", default=None,
        help="Output .pkl path. Default: models/<vehicle>_normalizer.pkl",
    )
    parser.add_argument("--vehicle", default="vehicle",
                        help='Free-text vehicle label, e.g. "Skoda Roomster 2007".')
    args = parser.parse_args(argv)

    slug = args.vehicle.lower().replace(" ", "_")
    default_out = MODELS_DIR / f"{slug}_normalizer.pkl"
    out_path = Path(args.out) if args.out else default_out

    return run_capture(args.port, args.duration_min, out_path, args.vehicle)


if __name__ == "__main__":
    sys.exit(main())
