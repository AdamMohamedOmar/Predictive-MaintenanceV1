"""Acceptance for §2c of the defense-sprint spec: a healthy real-car drive,
scored with its own baseline, must (a) be ≥ 70% healthy+cold_start windows and
(b) never fire a stable alert when its windows stream through StableAlerter.

Run:
    python -m scripts.acceptance_healthy_drive results/real_fault_eval/<file>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.models.stable_alerter import StableAlerter

_OK_LABELS = {"healthy", "cold_start", "warming_up"}


def main(result_json: str) -> int:
    d = json.loads(Path(result_json).read_text())
    windows = d["windows"]
    n = len(windows)
    ok = sum(1 for w in windows if w["label"] in _OK_LABELS)
    frac = ok / n if n else 0.0

    alerter = StableAlerter()
    fired: list[dict] = []
    for w in windows:
        state = alerter.update(w["label"], float(w["confidence"]))
        if state.active:
            fired.append({"elapsed_s": w["elapsed_s"], "fault_type": state.fault_type})

    print(f"windows={n}  healthy_or_regime={ok}  fraction={frac:.3f}  (need >= 0.70)")
    print(f"stable_alerts_fired={len(fired)}  (need 0)")
    for f in fired[:5]:
        print("  ", f)
    passed = frac >= 0.70 and not fired
    print("ACCEPTANCE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
