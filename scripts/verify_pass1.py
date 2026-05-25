"""Pass 1 verification: no false positives on healthy carOBD sessions.

All 5 healthy sessions must report:
  max_sev < 0.30   — physics severity formulas stay below "suspected fault" zone
  max_fc  < 0.10   — forecaster is suppressed on healthy/cold_start/warming_up labels

Run with:
    python -m scripts.verify_pass1
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.inference import InferenceEngine
from src.dashboard.streamer import CsvStreamer

CAROBD_DIR = REPO_ROOT / "data" / "raw" / "carOBD"
SESSIONS = ["drive1.csv", "live5.csv", "live7.csv", "live10.csv", "live12.csv"]

eng = InferenceEngine()

all_passed = True
for fname in SESSIONS:
    path = CAROBD_DIR / fname
    if not path.exists():
        print(f"  SKIP {fname} (not found)")
        continue
    eng.reset()
    strm = CsvStreamer(path)
    state = None
    for _ in range(min(300, strm.total)):
        row = strm.next_row()
        if row is None:
            break
        state = eng.update(row)

    if state is None:
        print(f"  SKIP {fname} (no rows)")
        continue

    max_sev = max(state.severities.values())
    max_fc  = max(state.forecasts.values())
    label   = state.classifier_label
    ok_sev  = max_sev < 0.30
    ok_fc   = max_fc  < 0.10
    status  = "PASS" if (ok_sev and ok_fc) else "FAIL"
    if not (ok_sev and ok_fc):
        all_passed = False
    print(
        f"[{status}] {fname}: label={label}, "
        f"max_sev={max_sev:.2f} ({'OK' if ok_sev else 'FAIL>0.30'}), "
        f"max_fc={max_fc:.2f} ({'OK' if ok_fc else 'FAIL>0.10'})"
    )

    if not ok_sev:
        print("       severity detail:", {k: f"{v:.2f}" for k, v in state.severities.items()})
    if not ok_fc:
        print("       forecast detail:", {k: f"{v:.2f}" for k, v in state.forecasts.items()})

if all_passed:
    print("\nPASS: Pass 1 verification PASSED -- no false positives on healthy sessions.")
else:
    print("\nFAIL: Pass 1 verification FAILED -- investigate severity gates above.")
    sys.exit(1)
