# Week 5 — Forecaster Baseline (Eid-Compressed)

**Dates:** Mon 25 May – Sun 31 May 2026
**Hour budget:** ~10 per person (reduced from 15 due to Eid al-Adha)
**Theme:** Get a forecaster working. Work around the holiday honestly.

---

## Critical context — Eid al-Adha disruption

**Eid al-Adha falls Tue 26 May – Fri 29 May** (4 days). Arafat Day (26 May) + 3 Eid days (27–29 May). This is a major family holiday in Egypt; both team members will have limited availability.

**Plan:**
- Monday is the productive day — heavy work, clear deliverables before Eid starts
- Tue–Fri is reading/thinking time only — no heavy coding expected; design review and paper skimming is realistic
- Sat–Sun resume normal work

Hour budget this week is reduced to ~10 per person. Anything that doesn't get done this week rolls into Week 6 — which has slack because ELM327 testing depends on hardware being in hand.

---

## Goal of the week

By Sunday night, a regression-based forecaster is trained on ramped-injection data and produces a baseline MAE number. The forecaster pipeline reuses Week 3's feature infrastructure, differing only in:
- Injection mode (ramp, not step)
- Target (severity at T+60s, not class label at window end)

Ordinal classification fallback design is documented but not implemented (Week 6 if needed).

---

## Pre-flight check (Monday morning)

- [ ] Week 4 Definition of Done all green
- [ ] Mid-project checkpoint decisions applied to plan
- [ ] Injection engine frozen (per checkpoint)
- [ ] Both team members have ~5 hours available today

---

## Daily tasks

### Monday 25 May — PUSH DAY (6h)

Everything that needs focused pair-coding happens today.

- **Both, 1h pair:** Design review. Forecaster formulation locked in charter: regression, predict severity at T+60s given current 60s window. Confirm:
  - Input: same 60s window used by classifier, same features (or raw tensor for DL)
  - Target: severity value 60s after window's last timestamp
  - Only windows where the fault class is non-healthy at T+60s are used for training (healthy windows have no meaningful severity target)
  - One forecaster per fault class? Or one forecaster that also takes class as input? **Decision:** one per class initially. Simpler. Five models, each specialized.

- **Both, 2h pair:** Write `scripts/generate_forecaster_dataset.py`. For each healthy carOBD file, for each of 5 fault classes:
  - Apply a ramp injection: start at random t₀ (60–120s in), end at random t₁ (t₀ + 120–300s), target severity ~0.8–1.0
  - For each window in the ramped region, record: features, current severity, severity-at-T+60s
  - Save as a structured dataset indexed by (session, fault_class, window_start)

- **Both, 2h pair:** Implement `src/models/rf_forecaster.py` — sklearn RandomForestRegressor. Same session-level 5-fold split. One forecaster per fault class. Metric: MAE.

- **One person, 1h:** Training script `scripts/train_forecaster.py`. Run all 5 forecasters. Save MAE per class + overall. Log to `docs/FORECASTER_RESULTS.md`.

**Before end of day: commit everything. Push. Confirm Ahmed can pull on his laptop.** Eid starts tomorrow — no guarantee of easy coordination for 4 days.

### Tuesday 26 May – Friday 29 May — EID (minimal work, 1h total each)

- **Optional, ~1h total across the 4 days:** Light reading only. Options if time permits:
  - Read a relevant paper (e.g., remaining useful life forecasting with regression models)
  - Sketch notes in `docs/FORECASTER_NOTES.md` about what a good "severity scaling" between classes looks like (a severity of 0.5 in MAF fault is not the same physical impact as 0.5 in coolant fault)
  - Review Monday's results document if training completed

- **No pair work expected.** No PR reviews required. No coding sprints.

- **If you feel unwell or simply don't have bandwidth:** that's fine. The plan assumes minimal Tue–Fri output.

### Saturday 30 May — Post-Eid recovery + results analysis (3h)

- **Both, 1h pair:** Catch up sync. Walk through Monday's results together. Is MAE reasonable (≤ 20% of severity range is fine for baseline)? Which fault classes are hardest to forecast?
- **Both, 2h pair:** Notebook `notebooks/06_forecaster_results.ipynb`:
  - Per-class MAE table
  - Calibration plot: predicted severity vs actual severity (scatter, with y=x line)
  - Residuals over prediction horizon — does error grow as we extrapolate further?
  - Identify the hardest-to-forecast class. This informs Week 6's deep learning experiment.

### Sunday 31 May — Weekly review + Week 6 prep (1h)

- Run Definition of Done.
- **Critical check for Week 6:** has ELM327 arrived? If it's been 4+ weeks since order, follow up with the seller. Week 6 Skoda baseline depends on it.
- If forecaster MAE is bad (>25% severity range), write a note — Week 6 needs to consider whether to invest deep-learning effort or pivot to ordinal classification fallback.

---

## Concrete deliverables

- `scripts/generate_forecaster_dataset.py`
- `src/models/rf_forecaster.py`
- `scripts/train_forecaster.py` + results saved
- `docs/FORECASTER_RESULTS.md` started
- `notebooks/06_forecaster_results.ipynb`

---

## Definition of Done

- [ ] Ramped injection dataset generated for all 5 fault classes
- [ ] Random Forest forecaster trained, evaluated with session-level CV, per-class MAE saved
- [ ] Calibration plot produced
- [ ] `docs/FORECASTER_RESULTS.md` contains baseline numbers
- [ ] Week 6 go/no-go decision documented: proceed with DL forecaster, or pivot to ordinal classification?

---

## Week-specific risks

| Risk | Watch level | What to do |
|---|---|---|
| Monday doesn't produce a working baseline because of unforeseen bugs | HIGH — no recovery days until Saturday | Budget Monday's work tight. If something breaks at hour 5, stop adding features and ship what works. A barely-working RF forecaster on Monday is worth more than an ambitious broken one. |
| Team member unavailability extends beyond Friday | Medium | Saturday is catch-up day; if unavailability extends to Sunday, Week 6 absorbs the delay. Forecaster polish can slip to Week 7 if needed. |
| ELM327 still not arrived by Sunday | Medium | Week 6 must trigger the R2 fallback — pre-recorded Skoda session for demo. Follow up with the seller today, not tomorrow. |
| MAE is terrible (>30% severity range) | Medium | Document as a finding, not a failure. Fallback to ordinal classification is charter-sanctioned. Decide Sunday. |

---

## Handoff to Week 6

Week 6 needs:
- A working forecaster baseline with known MAE (check)
- A decision on whether to pursue DL forecaster or ordinal fallback
- ELM327 in hand (critical — drives Skoda baseline recording)
- Calibration and residual plots to understand where the forecaster fails
- A rested team — Eid is intense, and Week 6 restores full 15h/person
