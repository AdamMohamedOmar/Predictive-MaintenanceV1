# `data/real_faults/` — real-vehicle fault recordings + plumbing fixture

This directory holds OBD-II CSVs used to evaluate the detection pipeline against
data that was **not produced by the synthetic injector** in
`src/injection/fault_injector.py`. As of this commit it contains only a
hand-crafted **plumbing fixture** under `mock/`; the `skoda/` and `etios/`
sub-directories are reserved for real induced-fault recordings that will land
when the collection protocol in `docs/REAL_FAULT_COLLECTION.md` runs.

---

## What this directory IS

- `mock/mock_lean_fault.csv` — a hand-edited derivative of
  `data/raw/carOBD/drive1.csv`. Rows 0–199 are unchanged. From row 200 onwards,
  `LONG_TERM_FUEL_TRIM_BANK_1` and `SHORT_TERM_FUEL_TRIM_BANK_1` are biased
  upward (ramp over 100 rows, then hold), clipped to ±20 % to stay within the
  OBD-II standard fuel-trim range. Used by
  `tests/test_real_fault_harness_plumbing.py` to exercise
  `src/eval/real_fault_eval.py`.
- `skoda/` — destination for real Skoda fault recordings collected per
  `docs/REAL_FAULT_COLLECTION.md` (vacuum leak primary; ECT-bias secondary).
  **Empty in this commit.**
- `etios/` — destination for any optional parity recordings on a junkyard
  Etios (e.g. bench-tested known-bad ECT sensor swapped in). **Empty in this
  commit.**

## What this directory IS NOT

**The mock CSV does not prove the model detects real faults.** It biases the
exact PIDs the injector biases, so the classifier flagging those biased
windows is the same logical loop as the synthetic self-consistency floor
(see project root `README.md` "Headline numbers"). The only thing the mock
exercises is whether the harness wires up — `CsvStreamer`/`pd.read_csv` →
`InferenceEngine.update()` per row → per-stride window record → summary JSON.

Real-fault detection claims wait on data in `skoda/` collected per the
protocol. The headline real-fault metric is **vacuum-leak recall ≥ 0.60**
(see `docs/CHARTER.md` §11 invariant #7).

---

## Recipe for `mock/mock_lean_fault.csv`

For reproducibility — re-generate with:

```python
import numpy as np
import pandas as pd

from src.config import USEFUL_PIDS
from src.data_loading import load_carobd_csv

df = load_carobd_csv("data/raw/carOBD/drive1.csv").head(600).copy()
rng = np.random.default_rng(1234)

ramp = np.zeros(600)
ramp[200:300] = np.linspace(0.0, 1.0, 100, endpoint=True)
ramp[300:] = 1.0

ltft_bias = ramp * 12.0 + rng.normal(0.0, 0.2, 600)
stft_bias = ramp * 6.0  + rng.normal(0.0, 0.5, 600)

df["LONG_TERM_FUEL_TRIM_BANK_1"] = np.clip(
    df["LONG_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + ltft_bias,
    -20.0, 20.0,
)
df["SHORT_TERM_FUEL_TRIM_BANK_1"] = np.clip(
    df["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float) + stft_bias,
    -20.0, 20.0,
)

pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
df[pid_cols].to_csv("data/real_faults/mock/mock_lean_fault.csv", index=False)
```

---

## Filename convention for real data (when it lands in `skoda/` / `etios/`)

`<vehicle>_<faulttype>_<YYYYMMDD>_<runN>.csv`, for example
`skoda_vacuumleak_20260605_run1.csv`. Each CSV has a sibling `.json` with
per-run metadata:

```json
{
  "vehicle": "skoda_roomster_2007_1.6L",
  "fault_type": "vacuum_leak",
  "induction_method": "PCV-line cap, 4 mm rubber cap with hose clamp",
  "materials": ["4 mm rubber cap", "4 mm hose clamp"],
  "mods_in_place_from_s": 300,
  "mods_removed_at_s": 1200,
  "operator": "Adam",
  "weather": "23 °C, dry",
  "fuel_grade": "92 RON",
  "notes": "Engine set P0171 at t=920s; reading dashboard logged it but did not clear."
}
```

The harness in `src/eval/real_fault_eval.py` consumes the CSV and uses the
JSON only to slice the evaluation into `pre` / `fault` / `post` intervals when
computing recall.
