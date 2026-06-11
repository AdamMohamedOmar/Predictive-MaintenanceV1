# Defense Demo Script — 15 June 2026

## A. First Skoda session checklist (run on 11-12 June, NOT demo day)

1. Laptop + ELM327 in the Skoda, engine running, API server up (`uvicorn src.api.main:app`), web app up (`cd web && npm run dev`).
2. LIVE tab → select COM port → CONNECT. **Write down `missing_pids` from the UI.**
   If ACCELERATOR_PEDAL_POSITION_D/E are missing: TPS detection is degraded on
   this car — note it for the Q&A, the system says so on screen by design.
3. Drive until fully warm (coolant gauge mid-band), then CarPage → CALIBRATE → ~5 min
   normal driving → FINISH & FIT. Expect "✓ CALIBRATED". If REJECTED, the
   reason names the failed guard — fix and repeat.
4. Verification drive (≥ 10 min healthy): LIVE tab, confirm NO stable alert,
   timeline stays calm. Save the session dir name (`data/app/live_sessions/…`).
5. Back home: run the acceptance script over the session if desired:
   ```
   ./.venv/Scripts/python.exe scripts/eval_real_fault.py \
     data/app/live_sessions/<session>/rows.csv \
     --normalizer models/car_<id>_normalizer.pkl \
     --out results/real_fault_eval/skoda_verification_v1.json
   ./.venv/Scripts/python.exe -m scripts.acceptance_healthy_drive \
     results/real_fault_eval/skoda_verification_v1.json
   ```

## B. Demo-day click path (~8 min)

1. Login → Garage (three cars visible: Etios — training, Yaris — validation, Skoda — live).
2. Skoda CarPage → overview: point at "✓ CALIBRATED" and explain per-vehicle baselines
   (the torque-wrench-zeroing analogy — a wrench calibrated on steel reads wrong on aluminium).
3. LIVE tab → connect → telemetry + timeline streaming. Talking points:
   - Armed badge: fault alerts only fire after the car's own baseline is captured.
   - SHAP panel: "ENGINE_LOAD pushed this prediction toward air_system by 0.43" — not a black box.
   - Forecast column: severity 60 seconds from now.
4. Fault story: open HISTORY → `demo_fuel_system` recording → post-hoc timeline:
   onset marker at ~2:00, click it → exact sensor values at that second, LTFT
   climbing while STFT hands off. This is the STFT→LTFT adaptive-trim handoff the ECU performs.
5. (Optional, only if rehearsed and Ahmed approves) vacuum-hose pull with mark_leak; otherwise
   show the Yaris §10 / acceptance results instead.

### Numbers to mention on day

- Classifier macro-F1 = **0.80** on held-out sessions (drive1 + live12); §10 vacuum-leak recall = **0.966**.
- Forecaster MAE: coolant **0.7%**, TPS **6.3%**, fuel **12.4%**, air_system **19.2%** (structural limit — documented).
- Latency p95 = **4.6 ms** server-side; dashboard renders at ≤ 1 s end-to-end on localhost.
- Known limitation to state: fuel_system precision = 0.46 — the STFT→LTFT handoff means developing fuel faults
  overlap with healthy windows; per-vehicle calibration reduces false alarms.

## C. Fallback drill (rehearse TWICE on 14 June)

- Trigger: no telemetry 15 s after connect, or adapter error banner.
- Action: disconnect → port dropdown → "REPLAY — &lt;rehearsal session&gt;" → CONNECT.
  Same screen, recorded data. Keep talking; mention it is the morning's recorded session
  replaying through the identical pipeline.
- The system requires `PM_ALLOW_REPLAY=1` in the server environment (see launch commands below).

### Launch commands (put on a sticky note)

**Windows PowerShell:**
```powershell
$env:PM_ALLOW_REPLAY="1"; uvicorn src.api.main:app --port 8000
```
In a second terminal:
```powershell
cd web; npm run dev
```

**Bash (WSL / Git Bash):**
```bash
PM_ALLOW_REPLAY=1 uvicorn src.api.main:app --port 8000
cd web && npm run dev
```

## D. Rehearsal checklist (14 June, with the car)

- [ ] Full click path B end-to-end, timed (target ≤ 8 min).
- [ ] Fallback drill C, twice.
- [ ] Laptop power settings: no sleep, no updates pending, no antivirus scans scheduled.
- [ ] data/ rehearsal session committed so REPLAY has fresh material.
- [ ] Latency p95 from `results/latency_v1.json` noted on the slide.
- [ ] `missing_pids` list written down for Q&A.
- [ ] Demo-day URL / local address confirmed (http://localhost:5173 or similar).
- [ ] ELM327 firmware version and COM port number confirmed.
