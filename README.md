# Predictive Maintenance Framework for OBD-II Engine Fault Detection & Forecasting

Graduation project by **Adam** and **Ahmed**, Computer Engineering.
Deadline: 15 June 2026.

A framework that detects and forecasts engine faults from On-Board Diagnostic II (OBD-II) sensor data. Trained on healthy driving data from a Toyota Etios 2014 (the [carOBD dataset](https://github.com/eron93br/carOBD)) with physically-grounded synthetic fault injection. A live Skoda Roomster 2007 validation via an ELM327 adapter is the deployment target (real-fault data collection in progress).

**Two co-equal pillars** (v1.2 framing):

1. **Fault classification** — given a 60-second window of OBD-II data, classify engine state into one of 6 classes (healthy, cold_start, air_system, fuel_system, coolant_temp_sensor, throttle_position_sensor). XGBoost with SHAP explainability.
2. **Forecasting + anomaly detection** — predict raw next-window PID values 60 seconds ahead (`PIDForecaster`, trained on healthy windows only — no severity formula, no injector inverse) and score the current window against the learned healthy manifold (`AnomalyDetector`, IsolationForest). Residuals between predicted-vs-actual PID values plus the out-of-distribution score together form a model-agnostic fault sentinel. The legacy severity forecaster is preserved as a self-consistency baseline — see "Headline numbers".

See [`docs/CHARTER.md`](docs/CHARTER.md) for full scope, methodology, and success criteria.
See [`docs/PLAN.md`](docs/PLAN.md) for the 8-week execution plan.

---

## Headline numbers — what they actually measure

The classifier's synthetic scores, taken straight from the committed artefacts:

- **Fixed-holdout (drive1 + live12) macro-F1 = 0.965** — `results/xgb_classifier_v1_results.json`.
- **LOSO mean macro-F1 = 0.957**, but the mean is misleading: it averages near-duplicate commute sessions (live5–live11 each ≈ 0.99) against the one genuinely different trip. The honest hard case is the lone highway session **drive1 = 0.82** (`results/loso_cv_results.json`, `min_f1`). We headline 0.82, not the mean.

All of these measure recovery of labels written by the deterministic injector in `src/injection/fault_injector.py`, scored against a severity metric that — in the current artefacts — mirrors the injector's own coefficients. Read them as a **synthetic self-consistency floor**, not real-fault detection. (The severity ↔ injector coupling is being removed; see the work tracked in `docs/CHARTER.md` §15.)

Real-fault detection is validated separately against induced-fault recordings collected per `docs/REAL_FAULT_COLLECTION.md` and scored by `src/eval/real_fault_eval.py`. The paper's headline real-fault metric is **vacuum-leak recall ≥ 0.60**.

---

## Quickstart

```bash
# Clone
git clone <repo-url> carobd-pdm
cd carobd-pdm

# Create environment (Python 3.11)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
pytest
```

If `pytest` passes, the environment is working. You'll see one trivial test pass for now; more arrive as the project progresses.

---

## Project structure

```
.
├── src/
│   ├── injection/          # Fault injection engine
│   ├── features/           # Windowing, feature extraction, normalisation
│   ├── models/             # XGBoost classifier, forecasters, SHAP explainer
│   ├── dashboard/          # Streamlit live dashboard
│   ├── diagnostics/        # Rule-based cold-start checker
│   ├── live/               # ELM327 / OBD-II live source
│   ├── config.py           # Paths, constants
│   └── data_loading.py     # carOBD CSV loading
├── tests/                  # pytest suite
├── notebooks/              # Jupyter exploration and results
├── scripts/                # Production entry points (rebuild, train, live tools)
│   └── investigations/     # Research/tuning experiments (not production)
├── docs/                   # Charter, plan, per-week files, results docs
├── data/                   # Datasets (gitignored)
├── models/                 # Trained model artifacts (gitignored)
├── results/                # Evaluation outputs, plots (gitignored)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Data

The `data/` directory is **gitignored**. To get the training data:

```bash
# Download carOBD dataset
git clone https://github.com/eron93br/carOBD.git data/raw/carOBD
```

Skoda baseline recordings (Week 6 onwards) are in `data/skoda_baseline/` and are committed to the repo since they're small and project-specific.

---

## Reproducing results

*(Fleshed out Week 3 onwards as results appear.)*

| Result | Command | Outputs |
|---|---|---|
| Build dataset + train all models | `python scripts/rebuild_all.py` | `data/synthetic/`, `models/`, `results/` |
| Train forecasters only | `python scripts/train_forecasters.py` | `models/forecaster_v1.pkl`, `results/forecaster_v1_results.json` |
| Run dashboard (CSV replay) | `streamlit run src/dashboard/app.py` | Local web UI at localhost:8501 |
| Check live OBD adapter | `python -m scripts.live_discover` | Go/no-go report for ELM327 |
| Capture Skoda baseline | `python -m scripts.live_baseline_capture --port COM3` | `models/<vehicle>_normalizer.pkl` |

---

## Key design decisions

- **Split by session, never by window.** All cross-validation folds keep entire recording sessions together. Enforced by a regression test in `tests/test_splitting.py`. Violating this invariant inflates scores by 10–20 F1 points on synthetic OBD-II data.
- **Vehicle-agnostic features.** All PID statistics are z-scored against the source vehicle's own healthy baseline, enabling cross-vehicle generalization (Etios → Skoda).
- **Physics-respecting fault injection.** Every injected fault has a primary sensor effect AND a secondary ECU-response effect. See `docs/INJECTION_ENGINE.md` (Week 2 onwards).
- **Honest framing.** The classifier's macro-F1 numbers measure recovery of the injector's own labels — see "Headline numbers" above. The forecaster is evaluated on injected early-stage faults, not run-to-failure data. Misfire detection is explicitly out of scope (1 Hz OBD-II cannot resolve per-cylinder combustion). See `docs/CHARTER.md` §11.
- **Two classifiers, one in production.** `src/models/classifier.py` is the Random Forest baseline (Week 3); `src/models/xgb_classifier.py` is the production XGBoost model (Week 4, fixed-holdout macro-F1 = 0.965 / worst LOSO fold 0.82 — see "Headline numbers" above for what this measures). Both are kept in the repo so the thesis can report the RF→XGB improvement. The dashboard and live inference load exclusively from `models/xgb_classifier_v1.pkl`. The RF module also provides shared types (`ALL_LABELS`, `_HELD_OUT_SESSIONS`, `session_split`) imported by the XGBoost module.

---

## Status

| Week | Deliverable | Status |
|---|---|---|
| 1 | Config, data loader, project scaffold | ✅ Complete |
| 2 | Fault injection engine (4 faults, ramp mode) | ✅ Complete |
| 3 | Feature pipeline, Random Forest baseline | ✅ Complete |
| 4 | XGBoost classifier (synthetic floor: 0.965 fixed-holdout, 0.82 worst LOSO fold), SHAP explainer | ✅ Complete |
| 5 | FaultForecaster (4× XGBRegressor, 60 s horizon) | ✅ Complete |
| 6 | Streamlit dashboard, live ELM327 integration, cross-vehicle baseline | ✅ Complete |
| 7 | Live Skoda validation, polish | 🔄 In progress |
| 8 | Thesis write-up, final demo | 📅 Scheduled |

---

## Citation

If you use anything from this repo in your own work, please cite the project once the paper is published. The underlying carOBD dataset should be cited as described in the [original repository](https://github.com/eron93br/carOBD).

## License

TBD before Week 8. Likely MIT or similar permissive license for code; data usage governed by carOBD's upstream license.
