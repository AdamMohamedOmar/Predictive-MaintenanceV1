# Predictive Maintenance for OBD-II Engine Faults

Early detection of *developing* engine faults from standard OBD-II sensor data —
before the check-engine light. A 6-class XGBoost classifier + physics-constrained
fault injection + a live Streamlit dashboard, built as a Computer Engineering
graduation project (AAST) by **Adam & Ahmed**.

> **Honest scope, up front:** no public dataset of real, labeled OBD-II engine
> faults existed for this setup, so training faults are *synthetically injected*
> into healthy data under strict physics constraints. All accuracy numbers below
> measure recovery of injected faults across unseen driving contexts — a
> validated detection **algorithm and data-collection protocol**, with
> real-fault field validation as the roadmap's top item. The system is designed
> to say "Untested" and "insufficient data" rather than fake a verdict.

---

## What it does

- Reads OBD-II telemetry (recorded CSV or live ELM327 adapter) in rolling
  60-second windows.
- Classifies each window into **healthy / cold_start / air_system /
  fuel_system / coolant_temp_sensor / throttle_position_sensor**, with SHAP
  explanations for every call.
- Suppresses one-off blips with a 3-window temporal vote before alerting.
- Prints an **end-of-read health report** for recorded drives: per-fault
  Detected / Healthy / Untested / Inconclusive, physics-based severity, and
  explicit caveats.
- **Cross-vehicle by design:** features are z-scored against each car's own
  healthy baseline; faults whose required sensor a car doesn't expose are
  reported *Untested* instead of scored (e.g. the air-leak fault on MAF-based
  cars with no MAP sensor).

## Results (synthetic validation, honestly framed)

Trained on the public **carOBD** dataset (Toyota Etios 2014, 1 Hz) — all 129
sessions across five context families (commute / highway / long-trip / campus /
idle), ~84.5 hours.

| Evaluation | Result | How to read it |
|---|---|---|
| Leave-One-**Family**-Out CV (hold out an entire driving context) | mean macro-F1 **0.977**, worst fold **0.958** (`live` family) | Context-transfer robustness on injected faults. Report the worst fold. |
| Leave-One-Session-Out CV | mean **0.917**, min **0.699** | Per-session spread; the min is the honest floor. |
| Fixed 2-session holdout | 0.962 | Legacy split with near-duplicate leakage — superseded by LOFO; do not headline. |
| Real healthy MAF car (Toyota Yaris) | Air fault **Untested** (no MAP); verdict **INSUFFICIENT EVALUABLE DATA** | The system declining to invent a verdict is the correct behavior. |

**Why these numbers are a ceiling, not field performance:** the same injector
writes the faults and defines the answers (a *self-consistency* bound). No
synthetic split — however careful — can substitute for one real induced-fault
recording. See `docs/CHARTER.md` §11.

### The data-integrity story (worth knowing)

An early audit kept only 8–9 of 129 carOBD files, believing the rest violated
physical bounds (timing advance of 30–776°). The real cause was a **trailing
comma** on ~120 files: a bare `pd.read_csv` silently shifted every column one
place left. One parser flag (`index_col=False`) plus a physical-bounds guard
recovered the full dataset (4.4 h → 84.5 h) and now makes any future
misalignment fail loudly instead of entering training as scrambled "healthy"
data. Regression tests lock this shut (`tests/test_data_integrity.py`).

## Quick start

Requires **Python 3.11** and ~1 GB disk for data + models.

```bash
git clone https://github.com/AdamMohamedOmar/Predictive-MaintenanceV1.git
cd Predictive-MaintenanceV1

# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate
# Linux/macOS
python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# Training data (gitignored):
git clone https://github.com/eron93br/carOBD.git
# copy carOBD/obdiidata/*.csv into data/raw/carOBD/

pytest                          # verify environment + invariants
python scripts/rebuild_all.py   # build synthetic dataset + train all models
python scripts/lofo_eval.py     # the honest cross-context evaluation

streamlit run src/dashboard/app.py   # dashboard (CSV replay or live ELM327)
```

### Scoring a real recording

```bash
# Adapt a phone-app ELM327 export (any car) to the canonical 14-PID 1 Hz format
python -m scripts.adapt_app_csv data/demo/your_drive.csv --out data/real/your_drive.csv

# Fit that car's healthy baseline and score with an honest verdict
python -m scripts.score_recording data/real/your_drive.csv --pre-adapted \
    --vehicle "Your Car" --out-dir results/your_car
```

The dashboard also accepts raw app exports directly — it auto-detects the
format, renames columns, and resamples to 1 Hz.

## Project structure

```
src/
  config.py            constants: 1 Hz, 60 s windows, the 14 usable PIDs
  data_loading.py      safe carOBD loader (column-shift guard, bounds check)
  features/            windowing → 83 features → per-vehicle z-scoring
  injection/           physics-constrained synthetic fault engine
  models/              XGBoost classifier, SHAP, IsolationForest, forecasters,
                       StableAlerter (temporal alert gating)
  eval/                Untested-fault contract, real-recording harness,
                       end-of-read session report
  dashboard/           Streamlit app + per-row InferenceEngine
  live/                ELM327 live source, app-CSV adapter core, replay
  diagnostics/         rule-based cold-start checks, DTC codes, advice
  api/                 optional FastAPI backend (auth, garage, uploads, live WS)
scripts/               rebuild, evaluations, adapters, baseline capture
tests/                 ~50 files; includes real-data integrity suite
docs/                  CHARTER.md (scope + honest-framing invariants), DATA_NOTES.md
```

A beginner-friendly walkthrough of every file: **`docs/PROJECT_GUIDE.md`**.

## Known limitations

1. **Not validated on real faults.** Synthetic recall ≠ field recall.
2. **Per-car sensor coverage varies** — unavailable-sensor faults are Untested.
3. **Self-baselining detects developing faults only**; a fault present from
   key-on becomes the baseline. Printed on every report.
4. **~0.34 Hz phone-app sampling** aliases fast fuel-trim dynamics; FP rates
   measured on such data overstate the truth.

## Roadmap

1. One controlled real-fault recording (vacuum leak protocol in
   `docs/CHARTER.md`) or a public real-fault OBD-II dataset — the single
   highest-value item.
2. Per-regime baselining (idle vs. driving "normal") — evaluated, deferred:
   measured idle→drive feature shift is mild (2–3σ, throttle-side only) and
   absorbed by regime features + alert voting.
3. Intermittent / non-monotonic fault shapes in the injector (robustness).

## Data & citation

Training data: [carOBD](https://github.com/eron93br/carOBD) by Eron J.
Maranhão (Toyota Etios 2014, 1 Hz) — public, citation requested; see
`references.bib`.

## License & authors

Graduation project — Arab Academy for Science and Technology (AAST),
Computer Engineering. Authors: Adam & Ahmed.
