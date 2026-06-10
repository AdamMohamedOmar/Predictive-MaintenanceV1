# Defense Sprint: Cross-Vehicle Calibration, Demo Spine, Metrics Push

**Date:** 2026-06-11 · **Defense:** 2026-06-15 · **Approach approved:** A — Calibrate-first, demo-hard
**Primary demo (user decision):** React web app + live Skoda Roomster 2007 via ELM327.
**Report numbers:** still open — retraining allowed until model freeze on 13 June EOD.

## 1. Findings that drive this design

1. **A healthy real-car drive reads as 100% vacuum leak.** Ahmed's 2 June drive
   (Toyota **Yaris 2014** — NOT the Skoda; the Skoda has never been connected)
   classifies 64/64 windows as `air_system` with confidence ~0.97, anomaly score
   pinned at 1.0 — verified against today's retrained pipeline
   (`results/real_fault_eval/ahmed_drive_20260602_v2_postfix.json`).
2. **Root cause: the per-vehicle baseline is mock junk, not the model.**
   `models/my_test_vehicle_normalizer.pkl` was captured against the mock serial
   source ("My Test Vehicle", 2026-06-10T18:49). Its "healthy" stats contain
   `COOLANT_TEMPERATURE = 90.0 ± 0.0` (the NaN-fallback constant), and std = 0
   for `COOLANT_*`, `COOLANT_WARMUP_RATE`, `FUEL_LOOP_ACTIVE`, `RPM_IDLE_DRIFT`.
   Z-scores against near-zero stds explode; SHAP blames
   `INTAKE_MANIFOLD_PRESSURE__min__z` and LTFT/STFT stats — a Yaris's normal
   idle MAP lands many σ high, which is exactly the vacuum-leak fingerprint.
3. **The capture guards have a variance hole.** `process_captured_rows` checks
   coolant ≥ 75 °C, mean speed ≥ 15 km/h, ≥ 20 windows — the mock data passes
   all three. Nothing checks that the data *varies like a real engine*.
4. **Classifier gap is structural, not diffuse.** Macro-F1 0.7974 vs ≥ 0.80
   target. `fuel_system` precision 0.457 absorbs 47 healthy + 45 air_system +
   46 TPS windows; TPS recall 0.61; healthy recall 0.70. The dataset builder
   already skips the first quarter of the injection ramp as ambiguous
   (`fault_start = onset_idx + ramp_len // 4`, dataset_builder.py:197) — the
   skip is tunable.
5. **air_system forecaster MAE 18.6%** misses the ≤ 15% target. Its severity
   target is idle-gated by physics (mean 0.15, mostly zeros): the leak is only
   observable at idle, so off-idle windows train/eval against a forced zero.
6. §10 vacuum-leak recall on the mock recording is 0.9655 (target ≥ 0.60) ✅.
   Latency is unmeasured. 416 tests pass.

## 2. Workstream 1 — Cross-vehicle calibration (P0)

### 2a. Real baselines
- Fit a **Yaris** normalizer from Ahmed's healthy drive via the existing
  `scripts/capture_baseline_from_csv.py` (`--vehicle "Toyota Yaris 2014"`).
  Re-run `eval_real_fault.py` with it on the same drive. Fit-on-self = smoke
  test of the mechanism (expect overwhelmingly healthy), not a validation claim.
- The **Skoda** gets its baseline from its first real session (see §5 schedule);
  per-car artifacts are stored via `Car.baseline_normalizer_path` (column exists).
- Delete `models/my_test_vehicle_normalizer.pkl/.json` so nothing can load it.

### 2b. Baseline guardrails (the never-again fix)
Added inside `process_captured_rows` (shared by live capture and CSV capture).
Rejection happens at **capture time only** — a load-time mirror was considered
and dropped during planning: vehicles with unsupported PIDs legitimately produce
std-0 *features* (NaN-fill design in live_baseline_capture.py:142-149), so a
load-time variance check cannot distinguish them from mock junk. Raw-PID-level
checks at capture time can. Guards:
- **Reject** if any fitted feature std == 0 (constant input — synthetic, dead
  PID, or NaN-fallback filled).
- **Reject** if `COOLANT_TEMPERATURE` is constant at exactly 90.0 across the
  capture (NaN-fallback signature).
- **Reject** if closed-loop (`FUEL_LOOP_ACTIVE`) is never reached.
- Keep existing guards (coolant ≥ 75 °C, speed ≥ 15 km/h, ≥ 20 windows).
- Every rejection raises `ValueError` with a plain-English, named reason
  (e.g. "coolant frozen at 90.0 °C for the whole capture — is a real engine
  connected?"). One pytest per rejection path.

### 2c. Acceptance criteria
- Yaris re-eval after refit: healthy + cold_start ≥ 70% of windows, and no
  stable alert would fire (windows are 60 s / stride 10 s; StableAlerter
  consecutive-window rule applied to the offline result).
- Skoda: after its calibration drive, a **separate fresh healthy drive** shows
  the same — measured through the product (web app), not a script.

## 3. Workstream 2 — Demo spine (web app + live path)

### 3a. Calibrate flow
- Backend endpoint on the live router: a calibration session records ~5 min of
  rows, calls the guarded fit, saves `models/car_<id>_normalizer.pkl`, writes
  the path to `Car.baseline_normalizer_path`, returns verdict (ok / rejected +
  reason, n_windows, vehicle string).
- CarPage state machine: *Not calibrated → Recording (live row/window count) →
  Fitting → Calibrated ✓ (date, windows) | Rejected (reason)*.
- Re-calibration allowed (overwrite after confirm).

### 3b. Alert arming rule
- Live sessions on a car with no valid baseline run **disarmed**: telemetry,
  PID strip and timeline all live, but fault alerts suppressed and a badge
  shows "Monitoring (uncalibrated) — calibrate this car to arm fault alerts."
- WS `telemetry` frame gains `armed: bool` so the UI states it explicitly.

### 3c. Live hardening
- Adapter death mid-session → visible banner + one auto-reconnect attempt
  before giving up (UI shows reconnecting state).
- `missing_pids` surfaced by name in the UI (which PIDs this ECU lacks), not
  just a count. Matters doubly for the 2007 Roomster (early EOBD — pedal-position
  PIDs `ACCELERATOR_PEDAL_POSITION_D/E` are the likely gaps; TPS-fault features
  degrade if absent, and the UI must say so rather than silently NaN-fill).
- Engine-off / no-data timeout produces a human message, not a stuck spinner.

### 3d. Replay-as-live fallback (demo insurance)
- `ReplayObdSource` implementing the `LiveObdSource` interface
  (`connect/start/next_row/stop/measured_poll_hz/missing_pids`), fed by a
  recorded CSV at true 1 Hz. Selected via a dev-only mechanism (env var or
  query param) — invisible in normal use.
- Same `_run_session`, same store, same UI. On-stage failure = switch source,
  identical screen continues.

### 3e. Sensor timeline / fault inspector (user-requested)
- New `SensorTimeline.tsx` (recharts `LineChart`): **X = time**, **Y = selected
  sensors** via a PID picker (checkbox multi-select of the 14 PIDs; default
  diagnostic set: RPM, MAP, STFT, LTFT, COOLANT; selection persisted in
  localStorage).
- Mixed units handled by per-series min-max normalization for *display*;
  **exact raw values** shown in the hover tooltip and in a readout panel for
  the selected timestamp (the requirement is "access the exact outputs of
  sensors at the time the fault appeared" — the readout is the contract).
- **Fault markers:** alert events drawn as vertical reference lines / shaded
  bands. Clicking a marker (or scrubbing to any time) snaps the readout to
  that second: alert label + all 14 sensor values at that timestamp.
- **Live mode** (in `LiveSession.tsx`): rolling window from the existing
  `pidHistory` buffer (extend `HISTORY_LEN` 300 → 600 ≈ 10 min).
- **Post-hoc mode** (Results page / session detail): full recording with a
  recharts `Brush` for scroll + zoom; data from the recording's adapted CSV
  rows + per-window results already served by `RecordingDetail`.
- Backend prerequisites: (1) WS frames carry discrete alert events (StableAlerter
  transitions + rule alerts), not just the current label; (2) `LiveSessionStore`
  persists them to `alerts.json` next to `rows.csv`/`marks.json`; (3) a session
  detail endpoint serves row-level PIDs for charting.

### 3f. Latency, measured
- Timestamps at OBD poll → WS send → browser receive/render; p50/p95 over a
  session written to `results/latency_v1.json` (report cites measured numbers
  against the ≤ 2 s target).

### 3g. Demo script
- One-page doc: exact click path (login → garage → Skoda → live), talking
  points per screen, the fallback drill (who clicks what when the adapter
  dies), and the §10 `mark_leak` moment if the optional real-leak test happens.

## 4. Workstream 3 — Timeboxed metrics push (hard stop 13 June EOD)

### 4a. Macro-F1 ≥ 0.80 (lever order, stop at target)
1. Widen the ambiguous-ramp skip: `fault_start = onset_idx + ramp_len // 4` →
   `// 2` in `dataset_builder.py`. Attacks exactly the early-ramp windows that
   look healthy/ambiguous and train the fuel_system black hole.
2. Class/sample weights to trade fuel_system recall (0.71) for precision (0.46).
3. (Only if still short) one feature: LTFT slope over the window, separating
   *developing* fuel faults (rising LTFT) from steady states.
- After every attempt: `rebuild_all` + record macro-F1 / fuel precision / TPS
  recall, **and re-run the two regression guards** — §10 mock recall ≥ 0.60 and
  the Yaris healthy re-eval staying clean. A retrain that wins F1 but
  reintroduces false alarms on a real car loses.

### 4b. air_system forecaster MAE — fix or honestly rescope
- Try: train/evaluate the air forecaster on idle-containing windows only
  (`ENGINE_LOAD__mean ≤ 40`, mirroring the severity gate) — the physics says
  severity is only defined there.
- If < 15% on that subset → done, document the gate. If not → keep the global
  number and document the physics-based rescope (precedent: TPS's 35% commit
  limit in `forecaster.py`).

### 4c. Freeze
- 13 June EOD: final `rebuild_all`, commit models + results JSONs, report
  tables sourced from those JSONs only. After freeze: demo-blocking bugfixes only.

## 5. Workstream 4 — Schedule & risks

| Day | Focus |
|---|---|
| **11 Jun** | W1 complete (Yaris fit, guardrails, re-eval). Start calibrate flow. **First Skoda session ASAP** — PID discovery, calibration drive, healthy verification drive. |
| **12 Jun** | W2 bulk: calibrate UI, arming, hardening, replay fallback, sensor timeline (live). W3 attempts between builds. |
| **13 Jun** | W3 closes EOD → **model freeze**. Timeline post-hoc mode + latency. Second Skoda drive through the product if available. |
| **14 Jun** | Dress rehearsal with the car + fallback drill + demo script final. Report tables. No code except demo-blockers. |
| **15 Jun** | Defense. |

**Risks:** (1) Skoda 2007 PID coverage unknown until first connection — that's
why the first session is on day 1, not day 4; `missing_pids` UI + NaN-handling
is the mitigation if pedal PIDs are absent. (2) Cheap ELM327 clones < 1 Hz —
existing slow-adapter warning + T3.1 resampler mitigate. (3) Retrain churn —
two regression guards re-run after every rebuild.

**Optional (user's call, not a dependency):** one rehearsal drive includes a
briefly cracked vacuum hose + `mark_leak` for a real §10 measurement.

## 6. Out of scope (explicit)

Streamlit changes (kept as-is as backup/internals view) · anomaly-detector AUC
improvements (0.689 is weak but it's a secondary OR-route §10 doesn't currently
need) · ONNX export · new fault types · web test infrastructure (lint + manual
checklist only) · dependency upgrades.

## 7. Testing strategy

- Python: TDD for guardrails, calibrate endpoint, ReplayObdSource, alert
  persistence; one end-to-end bench test driving a recorded CSV through
  mock source → resampler → engine → store → WS frames. 416-test suite stays
  green throughout; suite is the pre-commit gate for every task.
- Frontend: `npm run lint` + a manual checklist per screen (no test infra by
  design). The bench/replay path doubles as the manual test harness.
- Metrics: every retrain logs macro-F1, fuel precision, TPS recall, §10 mock
  recall, Yaris healthy fraction — one line per attempt in the plan's journal.
