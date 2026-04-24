# Predictive Maintenance Framework for OBD-II Engine Fault Detection & Forecasting

Graduation project by **Adam** and **Ahmed**, Computer Engineering.
Deadline: 15 June 2026.

A framework that detects and forecasts engine faults from On-Board Diagnostic II (OBD-II) sensor data. Trained on healthy driving data from a Toyota Etios 2014 (the [carOBD dataset](https://github.com/eron93br/carOBD)) with physically-grounded synthetic fault injection, validated live on a Skoda Roomster 2007 via an ELM327 adapter.

**Two co-equal pillars:**

1. **Fault classification** — given a 60-second window of OBD-II data, classify engine state into one of 6 classes (healthy + 5 fault types).
2. **Fault forecasting** — given a window of data during a developing fault, predict fault severity 60 seconds ahead.

See [`docs/CHARTER.md`](docs/CHARTER.md) for full scope, methodology, and success criteria.
See [`docs/PLAN.md`](docs/PLAN.md) for the 8-week execution plan.

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
│   ├── injection/          # Fault injection engine (Week 2)
│   ├── features/           # Windowing, splitting, feature extraction (Week 3)
│   ├── models/             # Classifier and forecaster (Weeks 3-6)
│   ├── dashboard/          # Streamlit live dashboard (Weeks 4, 7)
│   ├── config.py           # Paths, constants
│   ├── data_loading.py     # carOBD CSV loading
│   └── obd_live.py         # Live OBD-II reader (Week 6)
├── tests/                  # pytest suite
├── notebooks/              # Jupyter exploration and results
├── scripts/                # Dataset generation, training entry points
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
| Generate synthetic dataset | `python scripts/generate_dataset.py` | `data/synthetic/` |
| Train classifier | `python scripts/train_classifier.py` | `models/xgb_classifier.pkl`, `results/classifier_metrics.json` |
| Train forecaster | `python scripts/train_forecaster.py` | `models/forecaster_*.pkl`, `results/forecaster_metrics.json` |
| Run dashboard | `streamlit run src/dashboard/app.py` | Local web UI |

---

## Key design decisions

- **Split by session, never by window.** All cross-validation folds keep entire recording sessions together. Enforced by a regression test in `tests/test_splitting.py`. Violating this invariant inflates scores by 10–20 F1 points on synthetic OBD-II data.
- **Vehicle-agnostic features.** All PID statistics are z-scored against the source vehicle's own healthy baseline, enabling cross-vehicle generalization (Etios → Skoda).
- **Physics-respecting fault injection.** Every injected fault has a primary sensor effect AND a secondary ECU-response effect. See `docs/INJECTION_ENGINE.md` (Week 2 onwards).
- **Honest framing.** The forecaster is evaluated on injected early-stage faults, not run-to-failure data. Misfire detection is explicitly out of scope (1 Hz OBD-II cannot resolve per-cylinder combustion). See `docs/CHARTER.md` §11.

---

## Status

| Week | Status |
|---|---|
| 1 | Not started — begins Mon 27 Apr 2026 |
| 2–8 | Scheduled |

Weekly progress is tracked in `docs/WEEK_XX.md` files.

---

## Citation

If you use anything from this repo in your own work, please cite the project once the paper is published. The underlying carOBD dataset should be cited as described in the [original repository](https://github.com/eron93br/carOBD).

## License

TBD before Week 8. Likely MIT or similar permissive license for code; data usage governed by carOBD's upstream license.
