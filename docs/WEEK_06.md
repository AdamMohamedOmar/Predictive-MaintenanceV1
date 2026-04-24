# Week 6 — Forecaster Final + Skoda Baseline

**Dates:** Mon 1 Jun – Sun 7 Jun 2026
**Hour budget:** ~15 per person
**Theme:** Finish both ML pillars. Get the Skoda baseline recorded so Week 7 can integrate live OBD.

---

## Goal of the week

By Sunday night, the forecaster has been improved beyond RF baseline (via 1D-CNN/LSTM or ordinal fallback, per Week 5 Sunday decision), the Skoda Roomster healthy baseline is recorded on real hardware, and both models are exported to ONNX for the dashboard.

---

## Pre-flight check (Monday morning)

- [ ] Week 5 Definition of Done all green
- [ ] **ELM327 adapter is physically in hand** (if not, trigger R2 fallback immediately — don't wait)
- [ ] Week 5 decision applied: DL forecaster OR ordinal fallback, committed to
- [ ] Post-Eid, both team members at full availability

---

## Daily tasks

### Monday 1 Jun — ELM327 test + Skoda plan (4h)

Hardware day. The demo depends on this working.

- **Both, 2h pair, at one of your cars (Skoda or any OBD-II compliant vehicle):**
  - Plug ELM327 into OBD-II port with ignition on, engine off
  - Install python-OBD, connect to adapter
  - Try reading 5 PIDs: RPM, speed, coolant temp, throttle position, fuel trim
  - Measure actual sample rate achievable
  - **⚠ Known issue:** cheap clones often can't hit 1 Hz on all PIDs simultaneously. If you're getting <0.5 Hz or intermittent drops, that's a real problem. Options: reduce number of PIDs polled, switch adapters, fallback to recorded sessions.
  - Record a 5-minute test — any vehicle is fine, doesn't need to be Skoda yet. Confirm data format matches what the models expect.

- **Both, 1h pair:** Write `src/obd_live.py`:
  - `OBDLiveReader` class wrapping python-OBD
  - Maintains rolling 60s buffer
  - Async read loop, thread-safe access to buffer
  - Raises clear errors if adapter disconnects
  - Unit test with a mock/fake adapter

- **One person, 1h:** Plan the Skoda baseline drive. What route? What duration (target 30–45 min)? Mix of idle, city, highway if possible. Avoid stop-and-go construction. Schedule for a day this week.

### Tuesday 2 Jun — Forecaster upgrade, branch A or B (5h)

Branch based on Week 5 Sunday decision.

**Branch A: Deep learning forecaster (if MAE was OK, we're going for the stretch)**

- **Both, 3h pair:** Implement `src/models/cnn_forecaster.py`. 1D-CNN operating on raw 60×N windows:
  - 2-3 conv layers, global pooling, MLP head with regression output
  - Small model — these datasets don't support anything huge
  - PyTorch, CPU training is fine for model this size
  - One model per fault class (matching RF baseline structure)

- **Both, 2h pair:** Train, evaluate, compare to RF baseline. Same session-level CV. Log to `docs/FORECASTER_RESULTS.md`.

**Branch B: Ordinal classification fallback (if MAE was too poor)**

- **Both, 3h pair:** Reframe severity regression as ordinal classification. Bin severity into low (0–0.33), medium (0.33–0.66), high (0.66–1.0). Train a 3-class classifier per fault class.
- **Both, 2h pair:** Evaluate with macro-F1 on the ordinal classes. This is a weaker result but honest and still useful — "we predict severity bucket 60s ahead."
- Document the pivot honestly in `docs/FORECASTER_RESULTS.md`. This is a finding, not a failure.

### Wednesday 3 Jun — Skoda baseline recording DAY (5h)

Block this day specifically for the Skoda drive. Weather-dependent somewhat — prefer dry conditions so the drive is consistent.

- **Both, 1h prep:** Double-check ELM327 + laptop + `src/obd_live.py` ready. Pre-test connection in the parked Skoda. Set up data logging to CSV with 1 Hz timestamping.
- **Both, 3h drive + record:**
  - Warm up engine (5–10 min idle)
  - 30–45 min drive: mix of city, highway if possible, a bit of idle
  - One person drives, other monitors the laptop — watch for connection drops, freeze frames, NaN injections
  - Second backup recording if first one is clean — redundancy matters
- **Both, 1h debrief:** Load the recorded CSV in a notebook. Sanity check: all PIDs populated? Reasonable ranges? Any gaps?
- Save as `data/skoda_baseline/skoda_healthy_2026-06-03.csv`. Commit it (small enough) — this is your ground truth healthy baseline for the demo.

**⚠ If the drive produces bad data:** reschedule for Thursday. Don't cave and use the bad file.

### Thursday 4 Jun — Skoda feature baseline + per-vehicle calibration (3h)

- **Both, 2h pair:** Compute the Skoda's healthy baseline statistics. Per-PID mean and std from the recorded session. Save as `data/skoda_baseline/baseline_stats.json`.
- **Both, 1h:** Validate cross-vehicle feature extraction works. Run the feature pipeline on the Skoda recording, using Skoda's baseline. Sanity check that baseline-normalized features are centered near 0 (since Skoda data is healthy, z-scores should be small).
- **⚠ Flag:** If Skoda baseline-normalized features look wildly different from Etios healthy-calibrated features (say, z-scores consistently above 2), that's a cross-vehicle transfer problem. May need to reframe live demo as "vehicle-specific classifier after per-vehicle calibration." This is still honest — note it in `docs/INTEGRATION_NOTES.md`.

### Friday 5 Jun — ONNX export + integration prep (3h)

- **Both, 2h pair:** Export both models (classifier and forecaster) to ONNX. Use `onnxruntime` for inference.
  - For XGBoost: use `onnxmltools` or `skl2onnx`
  - For PyTorch CNN: `torch.onnx.export`
  - Write a smoke test: load ONNX, run inference on a known window, verify output matches the original model within tolerance.
- **One person, 1h:** Update `src/dashboard/app.py` to use the ONNX models. This is the first real integration — dashboard + real models. Ugly is still fine.

### Saturday 6 Jun — Buffer + integration polish (2h)

- If anything from Mon–Fri rolled over, now's the time.
- Otherwise: clean up the dashboard a bit. Make the plots actually readable.

### Sunday 7 Jun — Weekly review + integration-readiness checkpoint (1h)

- Run Definition of Done.
- **Integration-readiness checkpoint** (from PLAN.md): are all pieces ready for Week 7's live integration? If not, Week 7 Monday starts with making them ready, not with paper drafting.

---

## Concrete deliverables

- `src/obd_live.py` + tests (mock adapter)
- `src/models/cnn_forecaster.py` OR ordinal classification fallback (per Week 5 decision)
- `data/skoda_baseline/skoda_healthy_2026-06-03.csv` (recorded Skoda data)
- `data/skoda_baseline/baseline_stats.json`
- ONNX model exports for both classifier and forecaster
- `src/dashboard/app.py` using ONNX inference
- `docs/FORECASTER_RESULTS.md` with final numbers
- `docs/INTEGRATION_NOTES.md` — cross-vehicle observations

---

## Definition of Done

- [ ] ELM327 reads OBD-II data at usable rate (≥0.5 Hz on required PIDs)
- [ ] Skoda healthy baseline CSV exists, 30+ min, all PIDs populated
- [ ] Final forecaster (CNN or ordinal) trained, evaluated, logged
- [ ] Forecaster MAE ≤ 15% of severity range OR ordinal macro-F1 documented as acceptable fallback
- [ ] Both models exported to ONNX and verified
- [ ] Dashboard runs end-to-end with ONNX models on a loaded CSV

---

## Week-specific risks

| Risk | Watch level | What to do |
|---|---|---|
| ELM327 still not arrived Monday | Critical | Trigger R2 fallback: plan the demo around the pre-recorded Skoda session. Edit Week 7 and 8 accordingly. |
| ELM327 arrives but can't hit adequate sample rate | High | Try reducing PID count. If still bad, buy a second adapter same-day shipping. If that fails, use recorded session only. |
| Skoda baseline drive produces bad data | Medium | Reschedule. One day's delay doesn't kill the week. |
| CNN forecaster doesn't beat RF baseline | Medium | Ship RF as final forecaster. Note in paper that "DL comparison showed no improvement on this dataset size" — this is an honest finding, not a failure. |
| Cross-vehicle features look wrong on Skoda | High | Document and reframe demo scope. Don't try to "fix" it in Week 6 — that's a research question, not an integration issue. |

---

## Handoff to Week 7

Week 7 needs:
- Working live OBD reader (check, tested in a real car)
- Final models, ONNX-exported (check)
- Skoda baseline recording for comparison (check)
- Cross-vehicle integration notes (check — whether positive or negative)
- A dashboard that runs end-to-end but is ugly (check)
- Both ML pillars at their final quality level — no more model training in Week 7 unless something is truly broken
