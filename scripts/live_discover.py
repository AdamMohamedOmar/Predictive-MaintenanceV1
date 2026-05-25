"""Discover which OBD-II PIDs this ECU supports and measure ELM327 poll rate.

Run this BEFORE live_baseline_capture.py to confirm the adapter and ECU are
ready for live inference.  It checks two go/no-go criteria:

  1.  At least 12 of the 14 training PIDs are exposed by this ECU.
  2.  The adapter can sustain ≥ 0.8 Hz across all supported PIDs.

Exit code 0 = ready.  Exit code 1 = not ready (read the output).

Usage
-----
    python -m scripts.live_discover
    python -m scripts.live_discover --port COM3
    python -m scripts.live_discover --port /dev/ttyUSB0 --poll-seconds 8
"""

import argparse
import sys
import time
from typing import Optional

from src.config import USEFUL_PIDS
from src.live.obd_source import LiveObdSource

# ── Go / no-go thresholds ─────────────────────────────────────────────────────
MIN_SUPPORTED_PIDS = 12   # minimum PIDs for inference to be meaningful
MIN_POLL_HZ = 0.8          # below this the 60-row window accumulates too slowly


# ── Pure evaluation (testable without hardware) ───────────────────────────────

def evaluate(n_supported: int, actual_hz: float) -> tuple[bool, list[str]]:
    """Return (go_no_go, list_of_failure_reasons).

    Parameters
    ----------
    n_supported : int   — how many of the 14 USEFUL_PIDS the ECU exposes
    actual_hz   : float — measured rows-per-second from the adapter

    Returns
    -------
    (True, [])                    if everything is within spec
    (False, ["reason1", ...])     if one or more checks fail
    """
    reasons: list[str] = []

    if n_supported < MIN_SUPPORTED_PIDS:
        reasons.append(
            f"Only {n_supported}/{len(USEFUL_PIDS)} PIDs supported "
            f"(need >= {MIN_SUPPORTED_PIDS}).  Check for a diesel engine "
            f"(no LTFT/STFT) or a very old ECU."
        )

    if actual_hz < MIN_POLL_HZ:
        reasons.append(
            f"Poll rate {actual_hz:.2f} Hz is below {MIN_POLL_HZ} Hz.  "
            f"Switch to an ELM327 v1.5 USB adapter.  Bluetooth clones and "
            f"v2.1 chips frequently fail this threshold."
        )

    return (len(reasons) == 0), reasons


# ── Hardware-touching shell ───────────────────────────────────────────────────

def run_discover(port: Optional[str], poll_seconds: int) -> int:
    """Connect, report, evaluate.  Returns shell exit code (0=pass, 1=fail)."""
    print(f"\nConnecting to ELM327 (port={port or 'auto-detect'})…")
    src = LiveObdSource(port=port, sample_hz=2.0)  # fast for measurement

    if not src.connect():
        print("\n[FAIL] Could not connect.  Check:")
        print("  • Ignition is in position II (dash lit, engine optional)")
        print("  • ELM327 is seated in the OBD-II port")
        print("  • Correct COM port (try Device Manager on Windows)")
        return 1

    supported = src.supported_pids
    missing = src.missing_pids

    # ── PID report ────────────────────────────────────────────────────────────
    print(f"\nPID coverage ({len(supported)}/{len(USEFUL_PIDS)} supported):\n")
    for pid in USEFUL_PIDS:
        mark = "OK" if pid in supported else "XX"
        print(f"  [{mark}]  {pid}")

    if missing:
        print(f"\n  Missing: {', '.join(missing)}")

    # ── Poll rate measurement ─────────────────────────────────────────────────
    print(f"\nMeasuring poll rate over {poll_seconds} s…")
    src.start()
    t0 = time.monotonic()
    row_count = 0
    while time.monotonic() - t0 < poll_seconds:
        if src.next_row() is not None:
            row_count += 1
        time.sleep(0.02)
    actual_hz = row_count / poll_seconds
    src.stop()

    print(f"  Measured: {actual_hz:.2f} Hz  ({row_count} rows in {poll_seconds} s)")

    # ── Go / no-go ────────────────────────────────────────────────────────────
    go, reasons = evaluate(len(supported), actual_hz)
    if go:
        print(f"\n[OK] ECU is ready for live inference.")
        print(f"     Next step: python -m scripts.live_baseline_capture --port {port or 'COM3'}")
        return 0
    else:
        print("\n[FAIL] The following issues must be fixed before inference:")
        for r in reasons:
            print(f"  • {r}")
        return 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover OBD-II PIDs and measure ELM327 poll rate."
    )
    parser.add_argument(
        "--port", default=None,
        help="Serial port for the ELM327 (e.g. COM3 or /dev/ttyUSB0). "
             "Omit to auto-detect.",
    )
    parser.add_argument(
        "--poll-seconds", type=int, default=5, metavar="N",
        help="Seconds to measure poll rate (default: 5).",
    )
    args = parser.parse_args(argv)
    return run_discover(args.port, args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
