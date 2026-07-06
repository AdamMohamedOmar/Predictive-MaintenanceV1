"""CLI: adapt an ELM327 app-export CSV (any car) to the 14-PID, 1 Hz format.

Thin wrapper over src/live/app_csv.py — the SAME core the dashboard streamer
uses, so offline scoring and dashboard replay can never diverge again.

Output: a clean CSV with exactly the 14 USEFUL_PIDS columns at 1 Hz, ready for
    python -m scripts.score_recording <out.csv> --pre-adapted ...

WARNING (sampling): ~0.34 Hz app exports cannot represent the ~1 Hz closed-loop
fuel-trim oscillation the model trained on — it is aliased away. Hold-last is
the least-fabricating option, but window statistics (esp. STFT std) will not
match the training distribution, so any false-positive rate from this data
OVERSTATES the true rate. Report it with that caveat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from src.live.app_csv import adapt_app_df  # noqa: E402

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Adapt an ELM327 app CSV (any car) to 14-PID 1 Hz."
    )
    ap.add_argument("csv", help="Path to the raw ELM327 app CSV export (any car).")
    ap.add_argument("--out", required=True, help="Output path for the clean 1 Hz CSV.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = pd.read_csv(Path(args.csv))
    clean, missing = adapt_app_df(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    log.info("Wrote %d rows @1Hz x %d PIDs -> %s", len(clean), clean.shape[1], out)

    for pid, why in missing:
        log.warning(
            "  MISSING PID: %-30s (%s) -> all-NaN; features/faults depending on it "
            "are NOT evaluable on this vehicle (fault marked Untested at scoring).",
            pid,
            why,
        )
    if not missing:
        log.info("  All 14 USEFUL_PIDS present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())