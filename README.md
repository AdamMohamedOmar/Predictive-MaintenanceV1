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

## Headline numbers — what they actually measure (model freeze 13 June 2026)

All numbers sourced directly from `results/` JSON files — never from memory.

### Classifier (`results/xgb_classifier_v1_results.json`)

**Fixed-holdout (drive1 + live12) macro-F1 = 0.8006** — per-class F1:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| healthy | 0.824 | 0.717 | 0.767 |
| air_system | 0.682 | 0.749 | 0.714 |
| fuel_system | 0.462 | 0.736 | 0.567 |
| coolant_temp_sensor | 1.000 | 0.985 | 0.993 |
| throttle_position_sensor | 1.000 | 0.617 | 0.763 |
| cold_start | 1.000 | 1.000 | 1.000 |

Fuel is hardest: the STFT→LTFT handoff plus mild jittered severities make developing fuel faults genuinely subtle. Precision = 0.46 is a structural floor; sample-weight tuning (Task 16) moved macro-F1 from 0.797 to 0.801 without meaningfully improving fuel precision.

**Known limitation:** on the healthy Yaris idle drive (Ahmed's 2 June session), the fuel_system cluster at elapsed 280–330 s fires 9 stable alerts. Healthy fraction = 84.4% (passes ≥ 70% threshold); the stable-alert count does not pass zero. This is a calibration gap, not a classifier regression — per-vehicle baseline calibration (the car page CALIBRATE flow) is the recommended mitigation.

**LOSO mean = 0.85** with large spread (σ ≈ 0.17): near-duplicate commute sessions score ≈ 0.98; structurally-different trips score lower. Honest hard case **live12 = 0.53** (`results/loso_cv_results.json`). We headline 0.53, not the mean.

These are still **synthetic** — faults are injected by `src/injection/fault_injector.py` — severity anchored to external diagnostic thresholds, withheld-coefficient evaluation gap ≈ 0.03 (`results/withheld_coeff_results.json`). Read as a synthetic baseline, not real-fault detection.

### Real-fault recall (`results/real_fault_eval/`)

**§10 vacuum-leak recall = 0.966** (28/29 windows, `mock_lean_fault_v1.json`). Pass criterion ≥ 0.60.

Yaris healthy drive (fit-on-self calibration): healthy fraction 0.844 with 9 residual fuel_system stable alerts (structural floor; see above).

### Forecasters (`results/forecaster_v1_results.json`)

| Fault | MAE % of range | Target | Status |
|---|---|---|---|
| coolant_temp_sensor | 0.7% | ≤ 15% | ✓ |
| throttle_position_sensor | 6.3% | ≤ 35% | ✓ |
| fuel_system | 12.4% | ≤ 15% | ✓ |
| air_system | 19.2% | ≤ 15% | ✗ structural floor |

Air_system misses: idle-gate applied (ENGINE_LOAD ≤ 40%), but MAP anomaly at idle is small (~3–5 kPa) and ECU self-compensates via fuel trim, leaving a low-SNR severity signal. 19% is the honest structural limit for this dataset.

### Latency (`results/latency_v1.json`)

Server-side poll→WS-send: **p95 = 4.61 ms** (n = 3, bench run; browser adds network + render on localhost).

Forecasting + anomaly (honest, mixed):
- **PID forecaster** (delta target, P1-3) beats a persistence baseline on MAP, coolant, and throttle-to-pedal ratio; **LTFT does not** — its 60-second change is near-zero noise.
- **Anomaly detector** AUC 0.69 (95% CI [0.66, 0.72]). 1% false-alarm budget holds within-distribution; inflates to ~16% on held-out sessions — addressed by per-vehicle baseline re-fit.

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

If `pytest` passes, the environment is working. The full suite is ~1200 tests;
the data-dependent ones skip unless `data/raw/carOBD/` is populated (see Data below).

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

The `data/` directory is **gitignored**. To get the training data, clone the
carOBD repository and copy its CSVs **flat** into `data/raw/carOBD/` (the
loaders glob `data/raw/carOBD/*.csv` directly — a nested `obdiidata/` folder
will not be found):

```bash
# Download carOBD dataset (CSVs live in the repo's obdiidata/ folder)
git clone https://github.com/eron93br/carOBD.git /tmp/carOBD
mkdir -p data/raw/carOBD
cp /tmp/carOBD/obdiidata/*.csv data/raw/carOBD/
```

Skoda baseline recordings (Week 6 onwards) are in `data/skoda_baseline/` and are committed to the repo since they're small and project-specific.

---

## Reproducing results

*(Fleshed out Week 3 onwards as results appear.)*

| Result | Command | Outputs |
|---|---|---|
| Build dataset + train all models | `python -m scripts.rebuild_all` | `data/synthetic/`, `models/`, `results/` |
| Train forecasters only | `python -m scripts.train_forecasters` | `models/forecaster_v1.pkl`, `results/forecaster_v1_results.json` |
| Run dashboard (CSV replay) | `streamlit run src/dashboard/app.py` | Local web UI at localhost:8501 |
| Check live OBD adapter | `python -m scripts.live_discover` | Go/no-go report for ELM327 |
| Capture Skoda baseline | `python -m scripts.live_baseline_capture --port COM3` | `models/<vehicle>_normalizer.pkl` |

---

## Key design decisions

- **Split by session, never by window.** All cross-validation folds keep entire recording sessions together. Enforced by a regression test in `tests/test_splitting.py`. Violating this invariant inflates scores by 10–20 F1 points on synthetic OBD-II data.
- **Vehicle-agnostic features.** All PID statistics are z-scored against the source vehicle's own healthy baseline, enabling cross-vehicle generalization (Etios → Skoda).
- **Physics-respecting fault injection.** Every injected fault has a primary sensor effect AND a secondary ECU-response effect. See `docs/INJECTION_ENGINE.md` (Week 2 onwards).
- **Honest framing.** The classifier's macro-F1 numbers measure recovery of the injector's own labels — see "Headline numbers" above. The forecaster is evaluated on injected early-stage faults, not run-to-failure data. Misfire detection is explicitly out of scope (1 Hz OBD-II cannot resolve per-cylinder combustion). See `docs/CHARTER.md` §11.
- **Two classifiers, one in production.** `src/models/classifier.py` is the Random Forest baseline (Week 3); `src/models/xgb_classifier.py` is the production XGBoost model (Week 4, corrected-physics fixed-holdout macro-F1 = 0.80 / worst LOSO fold 0.53 — see "Headline numbers" above for what this measures). Both are kept in the repo so the thesis can report the RF→XGB improvement. The dashboard and live inference load exclusively from `models/xgb_classifier_v1.pkl`. The RF module also provides shared types (`ALL_LABELS`, `_HELD_OUT_SESSIONS`, `session_split`) imported by the XGBoost module.

---

## Status

| Week | Deliverable | Status |
|---|---|---|
| 1 | Config, data loader, project scaffold | ✅ Complete |
| 2 | Fault injection engine (4 faults, ramp mode) | ✅ Complete |
| 3 | Feature pipeline, Random Forest baseline | ✅ Complete |
| 4 | XGBoost classifier (corrected-physics: 0.80 fixed-holdout, 0.53 worst LOSO fold), SHAP explainer | ✅ Complete |
| 5 | FaultForecaster (4× XGBRegressor, 60 s horizon) | ✅ Complete |
| 6 | Streamlit dashboard, live ELM327 integration, cross-vehicle baseline | ✅ Complete |
| 7 | Live Skoda validation, polish, defense sprint (WS calibrate, SensorTimeline, replay fallback) | ✅ Complete (model freeze 13 June) |
| 8 | Defense 15 June, thesis write-up | 🔄 In progress |

---

## Citation

If you use anything from this repo in your own work, please cite the project once the paper is published. The underlying carOBD dataset should be cited as described in the [original repository](https://github.com/eron93br/carOBD).

## License

TBD before Week 8. Likely MIT or similar permissive license for code; data usage governed by carOBD's upstream license.
