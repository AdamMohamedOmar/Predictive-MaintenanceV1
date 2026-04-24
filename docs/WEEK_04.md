# Week 4 — Classifier Final + Mid-Project Checkpoint

**Dates:** Mon 18 May – Sun 24 May 2026
**Hour budget:** ~15 per person
**Theme:** Ship the final classifier. Decide if we're on track.

---

## Goal of the week

By Sunday night, the final classifier (XGBoost with hyperparameter tuning) is trained, evaluated with SHAP for interpretability, and hits the macro-F1 ≥ 0.80 target. A dashboard skeleton (Streamlit) is stood up with mock predictions. Mid-project checkpoint is held and documented; scope adjustments are made if needed.

**This is the week where the project either stays on Level 2.5 priority or formally shifts.**

---

## Pre-flight check (Monday morning)

- [ ] Week 3 Definition of Done all green
- [ ] RF baseline numbers are known
- [ ] Confusion matrix reviewed; the hardest class(es) identified
- [ ] Session-leakage test still passing (re-run as sanity check)

---

## Daily tasks

### Monday 18 May — XGBoost baseline + targeted features (5h)

- **Both, 1h pair:** Review Week 3 results. Identify 2-3 targeted feature additions based on the confusion matrix. Examples:
  - Coolant vs healthy confusion → add "time since engine start" normalized coolant temp (cold starts should have rising coolant; a stuck sensor won't)
  - TPS vs throttle confusion → add standard deviation of throttle-pedal residual (a drifting TPS creates a residual with non-zero mean; healthy creates one with mean ~0)
- **One person, 2h:** Implement the new features in `src/features/extract.py`. Version the feature set — `v1` = Week 3 features, `v2` = Week 4 features. Being able to compare is important.
- **Other person, 2h:** Write `src/models/xgb_classifier.py`. Default hyperparameters first. Train it with the same session-level CV. Compare to RF baseline. Log numbers to `docs/CLASSIFIER_RESULTS.md`.

### Tuesday 19 May — Hyperparameter search (4h)

- **Both, 3h pair:** Grid search or random search on XGBoost. Keep it tractable — focus on `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`.
  - **Critical:** search must respect session-level splitting. Use sklearn's `GridSearchCV` with a `PredefinedSplit` built from your session folds. Don't let sklearn shuffle.
  - Budget: ~50 configurations max. Full search takes maybe 30 min, fine.
- **One person, 1h:** Pick the best configuration by macro-F1. Retrain once with that config on all 5 folds. Save model artifact: `models/xgb_classifier_v1.pkl`.

### Wednesday 20 May — SHAP interpretability (4h)

- **Both, 2h pair:** Notebook `notebooks/05_shap_analysis.ipynb`:
  - Load the trained XGBoost model
  - Compute SHAP values on a held-out fold
  - Summary plot (global feature importance, replaces RF's basic plot)
  - Per-class dependence plots for the top 3 features
  - **Identify 2-3 "headline" SHAP figures that will go into the paper.** Things like "the model relies primarily on fuel-trim sum for distinguishing fuel vs O2 faults" — this is what makes the thesis defensible.
- **One person, 2h:** Add SHAP values to `docs/CLASSIFIER_RESULTS.md`. Write a few paragraphs of narrative — what does the model look at, and does that match the physics we baked into the injections? If SHAP says the model mostly uses a feature we didn't expect, that's a finding worth discussing in the paper.

### Thursday 21 May — Dashboard skeleton (Streamlit) (4h)

- **Both, 4h pair:** Stand up `src/dashboard/app.py`:
  - Streamlit page layout: title, sidebar for file upload / "use live" toggle
  - Main panel: current class prediction (as text + probability bars), recent sensor plot (last 60s of a couple of key PIDs), a "forecast" placeholder box (Week 7 fills this in)
  - For now, load a saved CSV and run the classifier on 60s windows, updating every second in a loop (simulated live)
  - Don't worry about OBD-II integration yet — that's Week 7

This is deliberately ugly this week. A clean dashboard is a Week 7 deliverable. The point of Thursday is to prove the classifier can run in a streaming context.

### Friday 22 May — **MID-PROJECT CHECKPOINT** (3h)

- **Both, 2h meeting:** Formal checkpoint review. Answer in writing (in `docs/CHECKPOINT_WEEK4.md`):
  1. **Classifier status.** Final macro-F1 = ? Per-class F1 = ? Is the ≥0.80 target met?
  2. **Injection engine status.** Is it stable enough that the forecaster's ramped dataset can be generated without changes?
  3. **ELM327 status.** Has the adapter arrived? If not, expected arrival?
  4. **Timeline status.** Are we on schedule, ahead, or behind?

- **Decision rules** (from charter Section 9):
  - If classifier macro-F1 < 0.70: **drop the deep-learning classifier comparison**, ship XGBoost as the final classifier. Reclaim those hours for forecaster work.
  - If injection engine still needs changes: **freeze the engine at current state**. Accept whatever fault quality we have. Focus Week 5 on forecaster, not engine polish.
  - If ELM327 hasn't arrived with expected delivery after Week 6: **fall back to pre-recorded Skoda session** for the demo. Reframe the live demo as "offline demonstration using Skoda data recorded last week."
  - If classifier macro-F1 ≥ 0.80 AND engine is stable AND ELM327 arrived: **on track, proceed as planned.**

- **Both, 1h:** Update PLAN.md overview with any changes. If decisions were made, edit WEEK_05.md through WEEK_08.md accordingly. Commit as a charter amendment.

### Saturday 23 May — Buffer + documentation polish (2h)

- Clean up code from the week. Docstrings. Type hints.
- Update `docs/CLASSIFIER_RESULTS.md` with final numbers.
- If a deep-learning classifier comparison was kept (stretch goal), start it here. 1D-CNN is the obvious choice — it operates directly on the 60×N window tensor without handcrafted features.
- **Don't** let this Saturday turn into "let me just try one more hyperparameter config." The classifier is done on Friday.

### Sunday 24 May — Weekly review (1h)

- Run Definition of Done.
- Preview Week 5 — note that Eid al-Adha hits Tue–Fri. Forecaster baseline must be designed by Monday night so Tuesday's partial work is productive.

---

## Concrete deliverables

- `src/models/xgb_classifier.py` + trained model `models/xgb_classifier_v1.pkl`
- `notebooks/05_shap_analysis.ipynb` with headline figures
- `src/dashboard/app.py` skeleton with classifier integrated
- `docs/CLASSIFIER_RESULTS.md` with final numbers and SHAP narrative
- `docs/CHECKPOINT_WEEK4.md` — formal checkpoint record
- Updated PLAN.md if decisions changed downstream weeks

---

## Definition of Done

- [ ] XGBoost classifier trained, evaluated, saved as artifact
- [ ] Macro-F1 ≥ 0.80 OR formal checkpoint decision made and documented
- [ ] SHAP summary plot + 2-3 dependence plots produced
- [ ] Dashboard skeleton runs locally and shows predictions on a loaded CSV
- [ ] Checkpoint document committed, with explicit decisions on each of the 4 checkpoint questions
- [ ] If any scope changes were made, WEEK_05 through WEEK_08 updated to reflect them

---

## Week-specific risks

| Risk | Watch level | What to do |
|---|---|---|
| Macro-F1 stuck below 0.70 after XGBoost + new features | High impact | Checkpoint decision triggers: drop DL classifier, reallocate to forecaster. Don't over-engineer the classifier trying to rescue it. |
| SHAP reveals the model is relying on an unexpected feature | Medium — could be interesting or troubling | If unexpected-but-reasonable: great paper material. If unexpected-and-wrong (e.g., the model uses a feature that's actually an artifact of injection): serious — investigate before trusting the result. |
| Dashboard takes longer than 4h because Streamlit is new | Likely | Keep it ugly. Week 7 has full polish time budgeted. Don't let dashboard eat into SHAP time. |
| Hyperparameter search takes too long | Low | Fix seed, narrow grid. 50 configs in 30 min is plenty. |

---

## Handoff to Week 5

Week 5 needs:
- A frozen injection engine (check, per checkpoint)
- The feature pipeline reused (check — forecaster uses same windows)
- The session-level splitting infrastructure (check)
- A clear go/no-go from the Friday checkpoint on whether Week 5's forecaster scope is intact or reduced
- Awareness that Eid al-Adha (Tue–Fri) is a productivity cliff
