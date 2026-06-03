# Testing-Phase Readiness Plan (for Sonnet)

**Goal:** make the project ready for the Skoda testing phase:
1. Record a driving session on the Skoda (ELM327 + a phone app).
2. Feed the recording to the model and read predictions.
3. Judge whether predictions are sensible (the car is healthy → should read healthy).
4. Optionally **induce a vacuum leak** during a recording and check the model flags it.

This document lists **verified** bugs/gaps (with code evidence), validated fixes,
and an ordered task list. Do P0 first — those block the workflow above. Each task
names the files, the acceptance test, and the definition of done. Re-run
`pytest -q` after each task; the suite is green at 340 passing today (one
pre-existing `test_classifier.py` psutil failure is unrelated — ignore it).

---

## 0. Read this first — two non-code realities that shape the plan

### 0.1 "The model learns the leak" is a misconception to correct
The deployed model is trained **offline** on synthetic Etios faults. Feeding it a
recorded Skoda leak session does **not** retrain it. A recorded leak session is a
**validation** input: "does the model flag this real leak?" To make the model
actually *learn* a real leak you would have to add the labelled real-leak windows
to a training set and retrain — a separate, larger effort (a future "fine-tune on
real faults" track). For now, the realistic testing-phase goal is **validation +
data collection**, not online learning. Say this to Adam explicitly.

### 0.2 The Skoda is probably MAF-based — this is the biggest scientific risk
The model is trained on the Toyota Etios, which is **speed-density** (MAP, no MAF).
Commit `ae47424` rewrote the air-system physics around that: on a speed-density
engine a vacuum leak shows as **raised idle RPM + slightly raised idle MAP + a
small idle-only fuel trim**.

The Skoda Roomster 1.6 (VW-group MPI) is **very likely MAF-based**. On a MAF
engine a vacuum leak downstream of the MAF is **unmetered air → big positive fuel
trims** — the *opposite* of the corrected speed-density model, and actually closer
to the *original* (pre-fix) injector. Consequences for the test:
- `MAP` on the Skoda will read ~barometric (no idle vacuum) — same as Ahmed's car —
  so the corrected air-system features carry little signal.
- A real Skoda vacuum leak will most likely surface as **positive LTFT/STFT**, so
  the model may label it **`fuel_system`** (or `air_system`) — both count as
  "lean condition detected" per `docs/REAL_FAULT_COLLECTION.md` §8, which is the
  agreed success criterion. **Do not expect a clean `air_system` label.**

**Task P0-A below adds a metering-type check** so this is confirmed from the data,
not assumed. The honest framing for the thesis: cross-architecture transfer
(speed-density training → MAF target) is a known threat to external validity.

---

## 1. Verified findings (evidence in parentheses)

| # | Type | Finding | Evidence |
|---|---|---|---|
| F1 | **GAP** | No way to capture a vehicle baseline **from a recorded CSV** — only live. The Ahmed baseline was done with inline Python. Without this, nobody but us can re-baseline the Skoda. | `scripts/live_baseline_capture.py` is live-only; its `process_captured_rows(rows, …)` is hardware-free and reusable. |
| F2 | **GAP** | No CLI flag to **score against a chosen baseline**. `eval_real_fault` always uses the Etios normalizer, so cross-vehicle scoring needs inline code. | `evaluate_real_fault(csv_path, models_dir, engine)` accepts a pre-built `engine`, and `InferenceEngine(…, normalizer_override=path)` exists — but `scripts/eval_real_fault.py` exposes neither. |
| F3 | **BUG** | `extract_features` raises `KeyError` if any of the 14 PIDs is missing from the window. The Skoda may not expose all 14. `live_baseline_capture` feeds it a `df[pid_cols]` subset → crash. | `src/features/extractor.py:78` `window[pid].to_numpy(...)`; `live_baseline_capture.py:127` `sliding_windows(df[pid_cols], …)`. (The `real_fault_eval` path is already safe — it backfills missing PIDs with NaN.) |
| F4 | **BUG + SILENT-CORRUPTION** | The Torque adapter is tuned to Ahmed's app/car and fails on a different app in two verified ways: (a) the time format `%H:%M:%S.%f` is hard-coded → a different format (e.g. ISO `2026-06-02T16:39:00`) makes every row `NaT` and the adapter **hard-crashes** (`ValueError: cannot convert float NaN to integer` at the reindex); (b) column selection picks the **first** in-range candidate — and even a "densest" rule is **not enough**: on a *moving* car the stuck-at-0 `Vehicle speed (km/h)` column (220 readings, var 0) is picked over the real-motion `Vehicle Speed (km/h)` (162 readings, var 4178), so the model silently sees **speed = 0 for a moving car**, corrupting regime/idle/TPS features. Throttle and load have the same ambiguity. | `scripts/adapt_torque_csv.py` (`format='%H:%M:%S.%f'`, `_find_column` returns first with `notna().sum() > 0`). Both reproduced 2026-06-03. |
| F5 | **SCIENCE RISK** | MAF-vs-speed-density mismatch (see §0.2). Not a code bug; a validity risk that must be checked and framed. | Etios training is speed-density; Skoda 1.6 likely MAF. |
| F6 | **LIMITATION** | Cross-vehicle re-baselining only swaps the **classifier's** normalizer. The anomaly detector's IsolationForest and calibration, and the PID forecaster, stay Etios-fit — so their cross-vehicle scores are only approximately corrected. | `InferenceEngine` applies `normalizer_override` to `self._norm` (used by classifier + anomaly z-scoring), but the IsolationForest and its `score_lo/hi` calibration are Etios-trained. Documented as charter R12. |
| F7 | **MINOR** | The harness records classifier label + anomaly score per window but **not** the forecaster output, so a tester can't see the 60-s-ahead prediction. | `src/eval/real_fault_eval.py` window dict. |
| F8 | **MINOR/DOC** | `extract_features` docstring still says "82 named features" (should be 83). | `src/features/extractor.py:73`. |

No invalid `FAULT_BEARING_FEATURES`, and the live OBD source already fills missing
PIDs — both checked, both fine.

---

## 2. Proposed solutions (each validated for feasibility)

- **F1 → `scripts/capture_baseline_from_csv.py`.** Read an adapted clean-column CSV,
  build rows, call the existing `process_captured_rows(rows, vehicle_name=…)`, save
  `models/<vehicle>_normalizer.pkl` + sidecar. **Valid & cheap** — reuses the tested
  function and the guard checks. (Note: the guards require a real *drive* baseline —
  coolant ≥ 75 °C, mean speed ≥ 15 km/h — so the baseline session must be driving,
  not idle. Keep the guards; they protect the calibration.)
- **F2 → add `--normalizer PATH` to `scripts/eval_real_fault.py`.** Construct
  `InferenceEngine(normalizer_override=Path(args.normalizer))` and pass it via the
  existing `engine=` parameter. **Valid & cheap** — the plumbing already exists.
- **F3 → make `extract_features` tolerant of missing PIDs.** Add a one-line guard
  at the top: for any `pid in USEFUL_PIDS` not in `window.columns`, treat its column
  as NaN (so the five stats become NaN, which every downstream caller already
  NaN-fills). Cleanest single fix; or add a shared `ensure_pid_columns(df)` helper
  used by both ingest paths. **Valid** — mirrors the fix already shipped in
  `real_fault_eval` and keeps all callers safe.
- **F4 → robustify the adapter (and make column choice explicit, not heuristic).**
  (a) Replace the hard-coded time parse with a fallback chain: try `%H:%M:%S.%f`,
  then `%H:%M:%S`, then ISO `pd.to_datetime(…, errors='coerce')` with no format, then
  a numeric epoch (seconds/ms) column; if all fail, fall back to row-index ÷ a
  `--rate-hz` argument. Guard the empty case so it raises a clear message instead of
  the NaN-to-int crash. (b) **Column selection must not be a silent heuristic.**
  "Densest" is insufficient (a stuck-0 speed column is densest). Do BOTH: (i) reject
  candidates that are constant / zero-variance for PIDs expected to vary (speed,
  rpm, throttle, load) — a stuck column is a decoy; (ii) support an explicit
  `--mapping mapping.json` ({clean_pid: source_column}) so the human can pin the
  correct column once per app. The auto-heuristic stays as a default but the
  inspector (P0-A) must **print every candidate + its fill + variance + the auto-pick**
  so the human verifies before the real drive. (c) Broaden candidate name lists
  (the Skoda app may use plain `Engine coolant temperature (°C)` without `_7E0`).
  **Valid** — all local to the adapter; cover with unit tests including the
  moving-car stuck-speed case.
- **F5 → metering-type check (P0-A).** Add a tiny report (in the adapter or a new
  `scripts/inspect_recording.py`) that flags: is `MAP` ~constant ≈ barometric across
  the session (⇒ MAF car, MAP uninformative)? Is `MAF (g/sec)` present in the raw
  export? Print a verdict + the cross-architecture caveat. **Valid & cheap.** Then
  document the expectation: a MAF-Skoda leak likely reads `fuel_system`/`air_system`
  via positive trims.
- **F6 → document + optional stretch.** Document clearly that only the classifier
  normalizer is re-baselined. *Stretch* (not required for first test): add a
  `--refit-anomaly-from BASELINE_CSV` path that re-fits the IsolationForest on the
  Skoda baseline. Leave as P2.
- **F7 → record forecaster output** in the harness window dict (`severities` /
  `forecasts` from the `DashboardState`). Cheap.
- **F8 → fix the docstring** ("82" → "83"). Trivial.

---

## 3. Ordered task list for Sonnet

### P0 — unblocks the Skoda workflow (do these first)

**P0-1 — CSV baseline capture tool.** Create `scripts/capture_baseline_from_csv.py`
with `--csv`, `--vehicle`, `--out`, reusing
`live_baseline_capture.process_captured_rows`. Output `models/<slug>_normalizer.pkl`
+ `.json`. The `*_normalizer.pkl` suffix makes it appear in the dashboard's
normalizer picker automatically.
*Acceptance:* `tests/test_capture_baseline_from_csv.py` — feed a synthetic warm,
moving, ≥4-min adapted CSV → a fitted `BaselineNormalizer` is saved and re-loads;
an idle/cold CSV raises the existing guard `ValueError`.
*DoD:* a user can run one command on an adapted Skoda drive and get a normalizer.

**P0-2 — `--normalizer` scoring flag.** Add `--normalizer PATH` to
`scripts/eval_real_fault.py`; when given, build
`InferenceEngine(normalizer_override=Path(...))` and pass via `engine=`.
*Acceptance:* extend `tests/test_harness_readiness.py` — scoring the pretend-real
CSV with vs without an override produces different label distributions and both
write valid JSON.
*DoD:* one command scores a Skoda recording against the Skoda baseline.

**P0-3 — `extract_features` missing-PID tolerance (F3).** Backfill absent PID
columns as NaN at the top of `extract_features` (or a shared helper). Keep all 83
feature keys present.
*Acceptance:* `tests/test_features.py` — a window missing one PID still returns all
83 features (the missing PID's five stats are NaN); no `KeyError`.
*DoD:* baseline capture and feature extraction never crash on a reduced-PID ECU.

**P0-A — metering-type / recording inspector (F5).** Add
`scripts/inspect_recording.py` (or a `--inspect` mode on the adapter) that reports,
from the raw export: MAP constancy vs barometric, presence of `MAF (g/sec)`, idle
vs driving fraction, per-PID fill rates, and prints the MAF caveat from §0.2.
*Acceptance:* runs on `data/real_faults/ahmed/...` raw export and on Ahmed's adapted
CSV without error and prints a metering verdict.
*DoD:* before the Skoda drive, the team can confirm metering type from a 30-s test
recording.

### P1 — makes the test trustworthy

**P1-1 — robustify the adapter (F4) — HIGHEST P1, it silently corrupts data.**
Flexible time parsing (fallback chain + `--rate-hz`, no NaN-to-int crash);
variance-aware candidate rejection (drop stuck/zero-variance decoys for
vary-expected PIDs); explicit `--mapping mapping.json` override; broadened
candidate names.
*Acceptance:* `tests/test_torque_adapter.py` gains cases: (1) an ISO-datetime
`time` column parses (today it **crashes** — see F4a); (2) `%H:%M:%S` no-millis
parses; (3) the **moving-car stuck-0 speed** fixture — the all-zero-but-densest
speed column must NOT be selected (the varying one wins, or `--mapping` pins it);
(4) `--mapping` forces a chosen column.
*DoD:* the adapter ingests a non-Torque export without crashing OR silently
feeding speed=0 for a moving car.

**P1-2 — record forecaster output in the harness (F7).** Add `severities` and
`forecasts` to each window dict in `evaluate_real_fault`.
*Acceptance:* `tests/test_harness_readiness.py` asserts the keys exist and are in
[0, 1].
*DoD:* testers can see the 60-s-ahead prediction alongside the label.

**P1-3 — one-command Skoda pipeline + doc.** A thin
`scripts/score_recording.py` (or a `Makefile`/README block) that chains:
adapt → (capture baseline | use given baseline) → score → print a summary
(label distribution, anomaly mean, top SHAP features). Update
`docs/REAL_FAULT_COLLECTION.md` §9 to reference the real commands.
*DoD:* Adam runs **one** command on a raw Skoda export and gets a verdict.

### P2 — stretch / polish

- **P2-1 — fix `extract_features` docstring** "82" → "83" (F8). Trivial.
- **P2-2 — per-vehicle anomaly re-fit (F6).** Optional `--refit-anomaly-from` so the
  IsolationForest + calibration are re-fit on the Skoda baseline, not just the
  classifier normalizer. Document the cross-vehicle limitation either way.
- **P2-3 — severity-stratified detection note.** When real leak data exists, report
  detection vs leak severity (the `injected_magnitude` column already supports the
  synthetic side).

---

## 4. Field procedure for the Skoda day (hand to Adam & Ahmed)

Pre-req: P0-1, P0-2, P0-3, P0-A done; ELM327 + a logging app (Torque/Car-Scanner)
that exports CSV.

1. **30-second test recording, engine warm, idling.** Run `inspect_recording.py`
   (P0-A) → confirm metering type and that the app exposes RPM, both fuel trims,
   coolant, load, throttle, MAP. If a needed PID is missing, change the app's PID
   selection and retry.
2. **Healthy baseline drive — 5–8 min of real driving** (warm engine, varied
   speed/throttle, some road km; **not** idling). Export → `adapt_torque_csv` →
   `capture_baseline_from_csv` → `models/skoda_normalizer.pkl`. (The guards will
   reject an idle/cold capture — that's intended.)
3. **Healthy validation drive — another normal drive.** Adapt → score with
   `eval_real_fault --normalizer models/skoda_normalizer.pkl`. **Expectation:**
   mostly `healthy`, anomaly mean well below 1.0. If it still screams faults, the
   baseline drive wasn't representative — redo step 2 with more varied driving.
4. **Vacuum-leak induction** (per `docs/REAL_FAULT_COLLECTION.md` §5, §2 safety
   doctrine): 5 min healthy → cap a small vacuum line → ~15 min driving → restore →
   5 min healthy. Keep the sibling metadata JSON (mods-in/mods-out seconds). Adapt →
   score against the **healthy** Skoda baseline. **Expectation (MAF car):** during
   the fault interval, more windows flag `fuel_system`/`air_system` and the anomaly
   score rises vs the healthy intervals. Score it as recall over the fault interval
   (protocol §10): success target ≥ 0.60.
5. **File everything** in `data/real_faults/skoda/` with the metadata JSON; run the
   harness to JSON; record the recall number.

**If the model misses the leak:** that is a *valid, reportable result* (the
synthetic-to-real / speed-density-to-MAF gap), not a failure to hide. The fallback
is the same as the charter: report the gap honestly and, as future work, fine-tune
on the collected real-leak windows (§0.1).

---

## 5. What NOT to do (scope guard for Sonnet)

- Do **not** retrain or "tune" anything to make the Skoda numbers look good. A leak
  the model misses is an honest finding.
- Do **not** change the Etios training pipeline or the 83-feature contract for this
  phase — the testing tools are additive (adapter, baseline-from-csv, scoring flag,
  inspector).
- Do **not** add the catalyst-temp PID or otherwise expand features now (deferred,
  see `audit_carobd.py`).
- Keep model `.pkl` artefacts untracked (regenerable); commit tools, tests, docs,
  and tracked results JSON together so README/results never disagree.
