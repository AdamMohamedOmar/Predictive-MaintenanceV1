# Project Charter

**A Predictive Maintenance Framework for Engine Fault Classification and Early-Stage Forecasting from OBD-II Data**

| | |
|---|---|
| **Version** | 1.1 |
| **Date** | 24 April 2026 |
| **Authors** | Adam, Ahmed |
| **Deadline** | 15 June 2026 |
| **Priority Level** | 2.5 — Equal effort on classification and forecasting |
| **Repository** | Private GitHub (TBA) |

---

## 1. Project Identity

This project delivers a software framework that detects and forecasts engine faults from On-Board Diagnostic II (OBD-II) sensor data. The framework is trained on healthy driving data from a Toyota Etios and validated in a live demonstration on a Skoda Roomster via an ELM327 adapter. It is submitted as the final-year graduation project for a Computer Engineering degree and is intended to double as a portfolio piece for post-graduation applications to Valeo and Brightskies.

The work has two co-equal technical pillars:

1. **Fault detection and classification** — given a window of recent OBD-II data, decide which of six states the engine is in (healthy or one of five fault classes).
2. **Early-stage fault forecasting** — given a window of recent OBD-II data showing a fault in progress, predict the fault's severity one minute ahead.

Both pillars share a common foundation: a physics-respecting fault injection engine that transforms healthy OBD-II recordings into synthetic fault scenarios.

---

## 2. Problem Statement

Modern vehicles continuously stream sensor data through the OBD-II interface, but this data is almost always used reactively — the Malfunction Indicator Lamp (MIL) turns on after a fault has already tripped a manufacturer-defined diagnostic trouble code (DTC). By then, the fault has progressed far enough to cross a fixed threshold; subtler early-stage deviations and gradual sensor drift are invisible to the driver and typically invisible to standard OBD-II diagnostics.

A data-driven framework that can (a) classify faults from multivariate sensor patterns rather than isolated DTC thresholds and (b) forecast where a developing fault is heading before it crosses the MIL threshold would provide earlier warning, reduce unplanned breakdowns, and generate evidence for condition-based maintenance.

The scientific obstacle is that public OBD-II datasets containing labeled, naturally-occurring faults at scale do not exist. This project addresses that by synthesizing labeled fault data from healthy recordings using physically-grounded sensor models, then evaluating whether models trained on synthetic faults produce useful behavior on real vehicle data in live operation.

---

## 3. Objectives and Success Criteria

### 3.1 Primary objectives

1. Build a unified fault injection engine that applies physically-plausible modifications to healthy OBD-II time series, supporting both step faults (instantaneous full severity, used for classifier training) and ramp faults (gradual severity increase over time, used for forecaster training).
2. Train and evaluate a classifier that distinguishes six engine states: healthy plus five fault classes.
3. Train and evaluate a forecaster that predicts fault severity 60 seconds ahead of the current window.
4. Integrate both models into a live dashboard that reads OBD-II data in real time from a Skoda Roomster via ELM327.
5. Produce an IEEE-style conference paper, a full thesis book, and a presentation deck.

### 3.2 Quantitative success criteria

| Deliverable | Target (commit) | Stretch |
|---|---|---|
| Classifier macro-F1 on held-out sessions | ≥ 0.80 | ≥ 0.88 |
| Forecaster MAE (severity at T+60s, held-out) | ≤ 15% of severity range | ≤ 10% |
| Live dashboard latency (OBD read → displayed prediction) | ≤ 2 s | ≤ 1 s |
| Reproducibility: fresh clone to passing tests | ≤ 30 min | ≤ 10 min |

Macro-F1 is reported in preference to accuracy because the synthetic dataset will be class-balanced by construction, but any class imbalance that does emerge (for example, if some fault types produce fewer valid windows) must not hide behind a misleading accuracy number.

### 3.3 Non-quantitative success criteria

- Every claim in the paper and book is either supported by an experimental result or explicitly labeled as limitation or future work.
- A reader with no prior knowledge of the project can clone the repository, follow the README, and reproduce the headline results.
- The live demo runs on a laptop during the thesis defense without requiring an internet connection.

---

## 4. Scope

### 4.1 In scope

- Six-class classification over the fault taxonomy defined in Section 6.
- 60-second-ahead severity regression for ramped faults, with an ordinal classification fallback if regression underperforms.
- Classical ML baseline (Random Forest, XGBoost) with SHAP-based explainability.
- Deep learning comparison (1D-CNN and/or LSTM), used primarily for the forecaster and as a comparison point for the classifier.
- Streamlit dashboard consuming either recorded CSV files or a live ELM327 stream.
- Live validation on a 2007 Skoda Roomster 1.6L.
- Full documentation: README, requirements.txt, a reproducibility script, unit tests on the fault injection engine and feature pipeline.

### 4.2 Out of scope

- Misfire detection. 1 Hz OBD-II data cannot resolve per-cylinder combustion events; any model claiming to do so would be physically unjustified.
- Run-to-failure validation on naturally-degrading components. This is explicitly labeled as future work in the paper and book.
- Benchmarking against EngineFaultDB. This dataset was considered and dropped: it is lab dynamometer data, not OBD-II, and a fair comparison would require extensive caveats that dilute the paper's main contribution.
- Cloud deployment, mobile app, and Raspberry Pi deployment. Raspberry Pi is noted as future work; the scope for 15 June is laptop-based only.
- FastAPI + React dashboard. Streamlit is the committed dashboard. A React upgrade happens only if all other deliverables are complete with at least one week of slack remaining.

---

## 5. Datasets

### 5.1 Primary training dataset — carOBD

- Source: public GitHub repository (eron93br/carOBD), Toyota Etios 2014, 1.5L.
- Sample rate: 1 Hz across 27 PIDs.
- Contents: healthy driving only, across multiple trip modes (idle, highway, city, campus, long trips).
- Usage: the sole source of training data for both classifier and forecaster. All fault-labeled data is generated by applying the injection engine to these recordings.

### 5.2 Live validation vehicle — Skoda Roomster 2007

- 1.6L manual transmission.
- Read via an ELM327 Bluetooth or USB adapter (to be ordered; see Section 10).
- Used for (a) recording a healthy baseline before the demo and (b) the live dashboard demonstration on defense day.

### 5.3 Hard invariant — split by session, never by window

**This is a project rule, not a methodology preference.** All train/validation/test splits are performed at the level of entire recording sessions (individual CSV files from the carOBD dataset). Splitting by window, by row, or by shuffled timestamp causes information from the same driving episode to leak across the split boundary, producing inflated scores that collapse the moment the model sees a genuinely new trip. If at any point during development macro-F1 jumps above 0.95, the first hypothesis to rule out is session-level leakage.

---

## 6. Fault Taxonomy

The classifier and forecaster operate over six classes. Each is grounded in an identifiable physical mechanism and a clear OBD-II signature.

| Class | Mechanism | Primary PID signature |
|---|---|---|
| Healthy | Nominal operation | All PIDs within vehicle-specific baseline |
| Air system fault | MAF drift or intake vacuum leak | Intake manifold pressure, long-term fuel trim |
| Oxygen sensor fault | Stuck or biased O₂ sensor | Short-term fuel trim, fuel–air equivalence ratio |
| Fuel system fault | Injector clogging or fuel pressure drop | Long-term fuel trim, short-term fuel trim (both biased) |
| Coolant temperature sensor fault | Stuck or drifting ECT sensor | Coolant temperature, timing advance |
| Throttle position sensor fault | TPS drift | Throttle position vs. accelerator pedal position mismatch |

The injection engine must produce, for each non-healthy class, a modification that is both detectable in the listed signature PIDs and physically consistent with secondary effects (for example, a biased O₂ sensor causing the ECU to commanded-compensate via fuel trims).

---

## 7. Methodology

### 7.1 Fault injection engine

The engine takes a healthy OBD-II recording and a fault specification and returns a modified recording with per-timestamp labels. Two injection modes are supported:

- **Step**: at timestamp *t*₀, the fault jumps from severity 0 to a specified target severity and holds. Used to generate training data for the classifier, which operates per-window and does not need to model progression.
- **Ramp**: severity increases linearly (or along a specified profile) from 0 at *t*₀ to the target severity at *t*₁. Used to generate training and evaluation data for the forecaster, which must learn the relationship between current-window observations and near-future severity.

Severity is defined per fault class as a continuous value in a physically meaningful range — for example, a MAF drift severity might range from 0 (no drift) to 1.0 (30% under-reporting, a value beyond which the ECU would typically store a DTC).

### 7.2 Windowing

- **Window length**: 60 seconds (60 samples at 1 Hz).
- **Stride**: 10 seconds during training, for a 6× data-augmentation effect.
- **Label**: the fault class of the window, computed as the class active at the last timestamp of the window. For the forecaster, the regression target is the severity value 60 seconds after the window's last timestamp.

### 7.3 Features

Windows are summarized into a feature vector that is deliberately vehicle-agnostic, so that models trained on Etios data have a chance of transferring to the Skoda:

- Per-PID statistics: mean, standard deviation, min, max, and first-to-last delta within the window.
- Cross-PID ratios and residuals: throttle-to-accelerator-pedal ratio, fuel-trim sum (STFT + LTFT), commanded-vs-observed throttle residual.
- Baseline-normalized values: each PID is z-scored against the vehicle's healthy baseline (mean and standard deviation computed from a designated healthy recording per vehicle).

The baseline-normalization step is what makes cross-vehicle generalization possible in principle. Raw PID values differ between vehicles; deviations from each vehicle's own healthy baseline are comparable.

### 7.4 Models

- **Classical baseline**: Random Forest and XGBoost on the feature vectors above. Primary classifier for the initial submission; also a reasonable forecasting baseline.
- **Deep learning**: 1D-CNN and/or LSTM operating on the raw 60×N window (N = number of PIDs used). Primary candidate for the forecaster; comparison point for the classifier.
- **Explainability**: SHAP values on the XGBoost classifier for the paper, providing per-feature contribution plots that make the model's decisions defensible.

### 7.5 Evaluation protocol

Cross-validation is reported two ways:

- **(a) Fixed session-level holdout** — `{drive1, live12}` held out; the remaining 7 sessions train the model. Used by `scripts/rebuild_all.py` to produce the deployed artefacts and the headline F1 number.
- **(b) Leave-one-session-out (LOSO)** — all 9 usable sessions iterated; each held out in turn. Executed by `scripts/loso_cv.py`. Reports mean ± std macro-F1 as the honest generalisation estimate with error bars. The thesis must report both (a) and (b).

Additional evaluation items:
- Per-class precision, recall, and F1, plus macro-F1 and a confusion matrix.
- For the forecaster: MAE, RMSE, and a calibration plot of predicted vs. actual severity on held-out ramped injections.

### 7.6 Live demonstration

A Streamlit dashboard reads OBD-II data from the ELM327 adapter, computes baseline-normalized features in a rolling 60-second window, and displays (a) the current classifier prediction with class probabilities, (b) SHAP-style feature contributions, and (c) the forecaster's 60-seconds-ahead severity estimate. The dashboard runs on a laptop during the defense, with the vehicle idling in the parking lot if necessary.

---

## 8. Deliverables

| # | Deliverable | Owner | Format |
|---|---|---|---|
| D1 | Project charter (this document) | Both | CHARTER.md |
| D2 | Fault injection engine with unit tests | Both | Python package |
| D3 | Feature pipeline with unit tests | Both | Python package |
| D4 | Classifier model + evaluation notebook | Both | .pkl, .ipynb |
| D5 | Forecaster model + evaluation notebook | Both | .pkl, .ipynb |
| D6 | Streamlit dashboard with OBD-II live integration | Both | Python app |
| D7 | Skoda healthy baseline recording | Both | CSV |
| D8 | IEEE-style conference paper (4–6 pages) | Both | LaTeX |
| D9 | Thesis book | Both | LaTeX |
| D10 | Defense presentation | Both | PowerPoint |
| D11 | README with reproducibility instructions | Both | README.md |

---

## 9. Timeline (8 weeks)

Weeks are numbered from the project start. Week 8 ends on the 15 June deadline.

| Week | Focus | Key milestones |
|---|---|---|
| 1 | Foundation | Repo initialized, environment locked, carOBD downloaded and explored, ELM327 ordered |
| 2 | Injection engine | Step and ramp injection for all 6 classes, unit tests pass, physics-sanity-checked |
| 3 | Classifier — baseline | Feature pipeline, session-level split, Random Forest baseline, first confusion matrix |
| 4 | **Checkpoint** + classifier — final | XGBoost + SHAP, classifier hits ≥0.70 macro-F1 or the scope is trimmed; dashboard skeleton stood up; re-evaluate pillar balance |
| 5 | Forecaster — baseline | Ramped dataset generated, regression baseline trained, calibration plot reviewed |
| 6 | Forecaster — final + Skoda baseline | 1D-CNN or LSTM forecaster, ELM327 arrived, healthy baseline recorded on Skoda |
| 7 | Integration + paper draft | Live dashboard end-to-end, paper draft v1, book outline with Chapter 1 seeded from this charter |
| 8 | Polish + deliverables | Book complete, paper final, presentation deck, dry-run defense, submission |

### Week 4 checkpoint — explicit decision rule

At the end of Week 4, review:

1. Is the classifier hitting ≥0.70 macro-F1 on the session-split validation set?
2. Is the fault injection engine stable enough that the forecaster's ramped dataset can be generated without further engine changes?
3. Is the ELM327 adapter in hand?

If (1) fails: drop the deep-learning classifier comparison, ship XGBoost as the final classifier, and pour the saved time into the forecaster.
If (2) fails: freeze the injection engine, accept whatever faults are working, and move on.
If (3) fails: fall back to a recorded Skoda session for the demo instead of live streaming.

---

## 10. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Session-level leakage inflates scores | Medium | High | Enforce split-by-session as a project invariant; add a regression test that fails if any window's session ID appears in more than one fold |
| R2 | ELM327 adapter doesn't arrive in time | Medium | Medium | Order by end of Week 1; fallback = pre-recorded Skoda session for demo |
| R3 | Skoda baseline differs too much from Etios for cross-vehicle transfer | Medium | Medium | Baseline-normalize all features; if transfer still fails, reframe demo as "same-vehicle after baseline calibration" rather than "cross-vehicle generalization" |
| R4 | Forecaster regression underperforms MAE target | Medium | Medium | Fall back to ordinal classification (low/medium/high severity); same pipeline, different head |
| R5 | Fault injection produces physically implausible signatures | Medium | High | Each injected class reviewed against OBD-II physics before use; unit tests assert that secondary PID effects move in the correct direction |
| R6 | Scope creep during paper/book writing | High | Medium | Paper and book content is bounded by what is in this charter; any addition requires a charter amendment |
| R7 | Git conflict or accidental force-push loses work | Low | High | main + feature branches only; no rebasing on shared branches; weekly `git bundle` backup to a second location |
| R8 | One team member blocked on environment setup | Medium | Low | `requirements.txt` locked in Week 1; both members verify a clean install on their own machine |
| R9 | Misfire-related question during defense | High | Low | Prepared answer: "1 Hz OBD-II cannot resolve per-cylinder combustion; misfire detection is explicitly out of scope and would require high-rate CAN access" |
| R10 | Claims in paper exceed experimental evidence | Medium | High | Every claim reviewed against Section 11 honest-framing rules before submission |

---

## 11. Honest Framing Invariants

These rules govern how the work is described in the paper, book, presentation, and any conversations with supervisors or interviewers. They are not stylistic preferences; they are the line between credible research and overclaim.

1. The forecaster predicts **injected, early-stage faults**, not naturally-occurring degradation. Every mention of forecasting in the paper and book includes this qualifier on first use, and the limitations section makes it explicit.
2. The classifier is trained on **a single vehicle's data** (Toyota Etios). Generalization to the Skoda Roomster is a hypothesis the live demo tests, not a proven property of the model.
3. Run-to-failure validation is **future work**, not something the project delivers.
4. **Misfire detection is not claimed** anywhere, for the reason given in Section 4.2.
5. Fault severity values are defined **per-class in physically meaningful units** and reported as such in the paper, not as unitless numbers whose meaning is hidden in the code.
6. The fault injection engine is a **reasonable approximation** of real fault mechanisms, not a certified simulator. The paper's discussion section acknowledges the gap between injected and real faults as the principal threat to external validity.

---

## 12. Team and Roles

Adam and Ahmed work as a pair on all technical components at the same skill level. There is no fixed role split. For coordination:

- All code goes through pull requests, even for solo-written features, so that the second person has seen and understood it.
- The person opening a pull request is the author; the other is the reviewer. Reviewer approval is required before merging to main.
- Weekly 30-minute progress review on Sundays, checking actual progress against the Section 9 timeline.

---

## 13. Tooling and Environment

| Category | Choice | Locked |
|---|---|---|
| Language | Python 3.11 | Yes |
| ML (classical) | pandas, numpy, scikit-learn, xgboost | Yes |
| ML (deep) | PyTorch | Yes |
| Explainability | SHAP | Yes |
| OBD-II live | python-OBD, pyserial | Yes |
| Dashboard | Streamlit | Yes (FastAPI+React only if ahead of schedule) |
| Model export | ONNX Runtime | Yes |
| Version control | Git, private GitHub | Yes |
| OS | Windows + VS Code | Yes |

---

## 14. Charter Amendment Process

This charter is the single source of truth for project scope. Changes happen explicitly, not by drift:

1. Any scope change is proposed as an edit to this file in a pull request.
2. Both team members must approve before the PR merges.
3. The version number at the top is incremented, and the change is noted in Section 15.
4. The amended charter is the one that governs from that point forward; previous versions remain in Git history.

---

## 15. Change Log

| Version | Date | Changes |
|---|---|---|
| 1.0 | 24 April 2026 | Initial charter. Level 2.5 priority locked. EngineFaultDB dropped from scope. Forecaster formulated as 60s-ahead severity regression with ordinal fallback. Window = 60s, stride = 10s. Classifier target = macro-F1 ≥ 0.80. Forecaster target = MAE ≤ 15% of severity range. |
| 1.1 | 24 April 2026 | Added Section 16 (Budget). ELM327 price left as a placeholder variable pending adapter selection. |

---

## 16. Budget

All costs are tracked here so that spending is visible and agreed before it happens, not discovered afterward. Prices are placeholders until confirmed at purchase time; update this table and increment the charter version when a number is finalized.

Currency: EGP (Egyptian Pound). Convert to other currencies only when reporting to external parties.

| # | Item | Purpose | Estimated cost | Status | Notes |
|---|---|---|---|---|---|
| B1 | ELM327 adapter | Live OBD-II read from Skoda for demo and baseline | `{{ELM327_PRICE}}` EGP | Not ordered | Order by end of Week 1. Bluetooth or USB; USB preferred for demo-day reliability. Avoid sub-100 EGP clones — many are counterfeit ELM327 v1.5 chips that drop PIDs intermittently. |
| B2 | OBD-II extension cable (optional) | Reach from OBD port to laptop on passenger seat | `{{OBD_CABLE_PRICE}}` EGP | Not ordered | Only needed if ELM327 is USB and the cable is too short to reach the laptop comfortably. |
| B3 | Fuel for Skoda baseline recording | One clean 30–45 minute drive to record healthy baseline | `{{FUEL_BASELINE_PRICE}}` EGP | Not incurred | Single line item; not a recurring cost. |
| B4 | Fuel for live demo day | Vehicle running during defense demonstration | `{{FUEL_DEMO_PRICE}}` EGP | Not incurred | Idle time mostly; budget small. |
| B5 | Printing (thesis book, paper, poster if required) | University submission copies | `{{PRINTING_PRICE}}` EGP | Not incurred | Check department requirements before printing — some accept digital-only. |
| B6 | Contingency | Unplanned costs (cable replacement, failed adapter, etc.) | `{{CONTINGENCY_PRICE}}` EGP | Reserve | Recommended: 20% of sum of B1–B5. |
| | **Total** | | `{{BUDGET_TOTAL}}` EGP | | |

### 16.1 Cost-free by design

The following are deliberately excluded from the budget because the project is designed to avoid them:

- **Cloud GPU compute.** Models are small enough to train on a laptop CPU or a free-tier Colab session. If this assumption breaks in Week 5 (forecaster training times exceed what a laptop can reasonably run overnight), the fallback is a one-off Colab Pro month, which becomes a new line item in a charter amendment.
- **Paid datasets.** carOBD is public and free. EngineFaultDB was dropped from scope in v1.0.
- **Paid software.** The entire stack is open source. VS Code, Python, PyTorch, Streamlit, Git all have no license cost.
- **Domain hosting for demo.** The dashboard runs locally for the defense. If a hosted version becomes needed later, it is future work.

### 16.2 Cost-sharing

Costs are split equally between Adam and Ahmed unless otherwise noted per line item. Receipts are kept in a shared folder so that split reconciliation is a reading exercise, not a memory exercise.

