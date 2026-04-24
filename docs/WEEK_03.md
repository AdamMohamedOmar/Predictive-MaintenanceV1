# Week 3 — Classifier Baseline

**Dates:** Mon 11 May – Sun 17 May 2026
**Hour budget:** ~15 per person
**Theme:** Get a first working classifier end-to-end. Ugly but honest numbers are fine.

---

## Goal of the week

By Sunday night, a Random Forest classifier is trained on step-injected carOBD data, evaluated with session-level 5-fold cross-validation, and produces per-class precision/recall/F1 + a confusion matrix. The feature pipeline is tested. The session-leakage regression test is in place and green.

Ugly first, pretty later. Week 4 is for improvement.

---

## Pre-flight check (Monday morning)

- [ ] Week 2 Definition of Done all green
- [ ] Visualization notebook plots reviewed and agreed physically plausible
- [ ] Any Week 2 rollover (e.g., stubborn fault class) either complete or explicitly deferred

---

## Daily tasks

### Monday 11 May — Windowing + session-split regression test (4h)

Do the session-split test first. Everything else depends on it being correct.

- **Both, 2h pair:** Implement `src/features/windowing.py`:
  - `make_windows(df, window_s=60, stride_s=10) -> list[Window]`
  - Each window carries: session_id (derived from source filename), start_time, end_time, data (60xN array), label (class at last timestamp), severity
- **Both, 2h pair:** Implement `src/features/splitting.py`:
  - `session_level_kfold(windows, n_splits=5, seed=42) -> list[(train_indices, val_indices)]`
  - Each fold's val set contains windows from sessions NOT in that fold's train set
- Write `tests/test_splitting.py` with **the critical regression test**:
  ```python
  def test_no_session_leakage():
      windows = [...]  # mix from 10 different session_ids
      folds = session_level_kfold(windows, n_splits=5)
      for train_idx, val_idx in folds:
          train_sessions = {windows[i].session_id for i in train_idx}
          val_sessions = {windows[i].session_id for i in val_idx}
          assert not (train_sessions & val_sessions), \
              "SESSION LEAKAGE DETECTED — this is a project-invariant failure"
  ```

  This test fails loudly if anyone ever changes the splitting code in a way that would leak. **It is the single most important test in the project.**

### Tuesday 12 May — Synthetic dataset generation (4h)

- **Both, 1h pair:** Design the dataset generation script. For each healthy carOBD file, produce:
  - 1 fully-healthy version (labels all HEALTHY)
  - 5 versions, one per fault class, each with a step injection starting at a randomized point (say, 60–120s into the recording), randomized target severity (0.4–1.0)
  - Net: ~6x data augmentation, reasonable class balance
- **One person, 2h:** Implement `scripts/generate_dataset.py`. Output: a single directory of CSVs + a metadata JSON mapping each file to (source_session, fault_class, severity, injection_start).
- **Other person, 1h:** Run it. Confirm outputs. Sanity-check class balance.

### Wednesday 13 May — Feature extraction (5h)

- **Both, 3h pair:** Implement `src/features/extract.py`. For each window, produce a feature vector:
  - **Per-PID stats** (for each of the ~12 chosen PIDs): mean, std, min, max, first-to-last delta → ~60 features
  - **Cross-PID ratios:** throttle/accel_pedal ratio mean + std, STFT+LTFT sum mean, commanded-vs-observed throttle residual mean → ~6 features
  - **Baseline-normalized stats:** for each PID, z-score mean against the session's per-file healthy baseline → ~12 features
  - Total feature vector size: roughly 80 features. Document it.
- **One person, 1h:** Write `tests/test_features.py` — basic shape and finiteness tests (no NaN/inf in output).
- **Other person, 1h:** Notebook `notebooks/03_feature_inspection.ipynb`. Load 100 random windows of each class, plot feature distributions by class. Eyeball it — do the features look separable?

### Thursday 14 May — Baseline model training (3h)

- **Both, 1h pair:** Write `src/models/rf_baseline.py`. Sklearn RandomForestClassifier. Default hyperparameters — don't tune. The point is to see a number, not to win.
- **Both, 2h pair:** Training script `scripts/train_classifier.py`:
  - Load windows, compute features
  - Run session-level 5-fold CV
  - For each fold: train RF, predict on val, store predictions
  - Aggregate: per-class precision/recall/F1, macro-F1, full confusion matrix
  - Save results to `results/week3_rf_baseline.json` and a confusion matrix PNG

### Friday 15 May — Results analysis + feature importance (3h)

- **Both, 2h pair:** Notebook `notebooks/04_classifier_results.ipynb`.
  - Load saved results, display confusion matrix
  - Identify which classes are confused with which (this tells us where the injection engine or features need work)
  - Plot RF feature importance — top 20 features
  - **⚠ Watch for:** macro-F1 > 0.95 on the very first model. If this happens, the most likely cause is session leakage (our test should catch it) or a feature that is literally the label in disguise (e.g., a baseline-normalized value that's only computable when you know which class the window is). Investigate before celebrating.
- **One person, 1h:** Write `docs/CLASSIFIER_RESULTS.md` — a running document capturing headline numbers per model version. This becomes the thesis results chapter.

### Saturday 16 May — Buffer / feature engineering based on results (2h)

- If certain classes are poorly classified, add 1-2 targeted features. Common wins:
  - If fuel vs O2 sensor are confused: ratio of STFT variability to mean
  - If MAF vs fuel are confused: engine-load-normalized intake pressure
- Don't spend more than 2 hours. Major improvements belong in Week 4.

### Sunday 17 May — Weekly review (1h)

- Run Definition of Done.
- Check macro-F1. If ≥ 0.70, Week 4 is XGBoost-and-polish. If < 0.70, Week 4 Monday starts with a retrospective — what went wrong? Bad features? Bad injections? Plan accordingly.

---

## Concrete deliverables

- `src/features/windowing.py`, `src/features/splitting.py`, `src/features/extract.py`
- `tests/test_splitting.py` — **session-leakage regression test** (non-negotiable)
- `tests/test_features.py`
- `scripts/generate_dataset.py`, `scripts/train_classifier.py`
- `results/week3_rf_baseline.json` + confusion matrix PNG
- `notebooks/03_feature_inspection.ipynb`, `notebooks/04_classifier_results.ipynb`
- `docs/CLASSIFIER_RESULTS.md` started

---

## Definition of Done

- [ ] `pytest` all green, including session-leakage test
- [ ] RF baseline trained and evaluated with 5-fold session-level CV
- [ ] Macro-F1 ≥ 0.70 OR a written post-mortem in the weekly review note explaining why not and what Week 4 will do about it
- [ ] Confusion matrix saved as PNG, reviewed by both team members
- [ ] Feature importance plot exists

---

## Week-specific risks

| Risk (from charter) | Watch level | What to do |
|---|---|---|
| R1 session-level leakage | HIGH — this is the week it can happen | Regression test on Monday. If macro-F1 > 0.95, suspect leakage first. |
| R5 injection physics implausible | Medium | If one class gets near-perfect F1 and another gets near-random, the injection for the "too easy" class may be producing an impossibly strong signature. Investigate, don't just accept. |
| Feature pipeline too slow | Low but annoying | 60s windows with stride 10 on ~50 CSVs = manageable. If it takes >10 min to generate features, optimize later (Week 4 Saturday). |

---

## Handoff to Week 4

Week 4 needs:
- Working feature pipeline (check)
- Session-level CV infrastructure (check)
- Baseline numbers to beat (RF macro-F1)
- A clear sense of which classes are hardest (from confusion matrix)
- All the feature-importance knowledge to inform XGBoost tuning and SHAP analysis
