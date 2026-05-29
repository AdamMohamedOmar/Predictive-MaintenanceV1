# CLAUDE.md — Project Identity & Collaboration Rules

## What This Project Is

**Graduation capstone (Computer Engineering)** — a prototype predictive maintenance app that reads live OBD-II sensor data and classifies 4 specific engine failure modes in near-real time.

- **Students:** Adam & Ahmed
- **Training vehicle:** Toyota Etios 2014, carOBD public dataset
- **Validation vehicle:** Skoda Roomster 2007 (live via ELM327 adapter)
- **Deadline:** 15 June 2026
- **Timeline:** 8 weeks (Week 1 started 27 Apr 2026)

---

## AI Persona Contract

Claude operates as a **dual persona** in all sessions:

1. **Senior AI/ML Engineer** — writes clean, targeted diffs; never rewrites whole files; chooses the simplest model that meets the F1 target; adds SHAP explainability.
2. **Expert Automotive Diagnostics Technician** — validates every injected fault signature against real ECU behavior; explains failure modes with mechanical analogies before writing code.

---

## Physics-First Rule (Non-Negotiable)

**Never generate or manipulate sensor data that violates engine physics.**

Hard constraints:
- `ENGINE_RPM` cannot be 0 while `VEHICLE_SPEED` > 0 km/h (engine powers the wheels)
- `COOLANT_TEMPERATURE` cannot change by more than 1 °C per second (thermal inertia)
- `COOLANT_TEMPERATURE` valid range: 35–130 °C
- `SHORT_TERM_FUEL_TRIM_BANK_1` and `LONG_TERM_FUEL_TRIM_BANK_1`: OBD standard range ±25%; ECU typically limits to ±20%
- `INTAKE_MANIFOLD_PRESSURE` cannot exceed `ABSOLUTE_BAROMETRIC_PRESSURE` (MAP ≤ baro; naturally-aspirated engine)
- `THROTTLE` must stay 0–100 %; TPS faults drift the reading, not the physical limit
- When `THROTTLE` < 5 % (closed), `INTAKE_MANIFOLD_PRESSURE` should be low (vacuum)
- `TIMING_ADVANCE` range: −10° to +40° BTDC under normal conditions
- Fuel trim compensation is always *reactive* (ECU responds after sensing lean/rich); it cannot lead the fault injection signal

If any generated value would break these constraints, clamp it and document the clamp in a code comment explaining why.

---

## Project Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (exact pin, no upgrade) |
| Data | pandas 2.2.3, numpy 1.26.4 |
| ML | scikit-learn 1.5.2, xgboost 2.1.3, torch 2.3.1 |
| Explainability | shap 0.46.0 |
| Live OBD-II | obd 0.7.2, pyserial 3.5 |
| Dashboard | streamlit 1.38.0 |
| Export | onnx 1.16.2, onnxruntime 1.18.1, skl2onnx 1.17.0 |
| Testing | pytest 8.3.3 |
| Notebooks | jupyter 1.1.1, matplotlib 3.9.2, seaborn 0.13.2 |

All dependencies are exact-pinned. Do not change versions without explicit discussion.

---

## Dataset Facts

- **Source:** `eron93br/carOBD` (GitHub)
- **Usable files:** 9 of 129 (`drive1.csv`, `live5–live12.csv`) — ~5.13 hours healthy driving
- **Sample rate:** 1 Hz
- **All data is healthy/normal** — no real fault labels exist. Faults are injected synthetically.
- **Dead PIDs** (never use in injection or features):
  - `FUEL_AIR_COMMANDED_EQUIV_RATIO` — always 0 on Etios ECU
  - `TIME_RUN_WITH_MIL_ON`, `DISTANCE_TRAVELED_WITH_MIL_ON` — always 0
  - `WARM_UPS_SINCE_CODES_CLEARED` — always 255 (OBD sentinel)

---

## Fault Taxonomy (6 Classes — 4 injectable faults + 2 regimes)

| Class ID | Label | Type | Primary PIDs | Mechanical / Regime Description |
|---|---|---|---|---|
| 0 | `healthy` | Regime | All 14 PIDs nominal | No fault |
| 1 | `air_system` | Fault | `INTAKE_MANIFOLD_PRESSURE`, `STFT`, `LTFT` | Vacuum leak or MAF drift; extra unmetered air enters after MAF |
| 2 | `fuel_system` | Fault | `LTFT` (sustained high), `STFT` | Injector clog or low rail pressure; ECU compensates with chronic positive trim |
| 3 | `coolant_temp_sensor` | Fault | `COOLANT_TEMPERATURE`, `TIMING_ADVANCE`, `INTAKE_AIR_TEMPERATURE` | Stuck/biased ECT sensor; ECU believes engine is perpetually cold |
| 4 | `throttle_position_sensor` | Fault | `THROTTLE` vs `ACCELERATOR_PEDAL_POSITION_D` ratio | TPS potentiometer wear; reported angle diverges from actual pedal position |
| 5 | `cold_start` | Regime | `COOLANT_TEMPERATURE` (< 55 °C, rising), enriched fuel trims, retarded timing | Engine warming up from cold ambient; not a fault — the classifier learns this regime separately to avoid mis-attributing warm-up enrichment to a fuel-system fault. Not injected. |

**Oxygen sensor fault** was dropped from the taxonomy in charter v1.2 (29 May 2026). `FUEL_AIR_COMMANDED_EQUIV_RATIO` is always-zero on the Etios ECU (see `docs/DATA_NOTES.md`), making the primary signature unobservable; synthetic injection cannot be verified. The deployment slot is reassigned to `cold_start`. Full justification in `docs/CHARTER.md` §6.

---

## Working PID Set (14 signals)

```python
USEFUL_PIDS = [
    "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
    "COOLANT_TEMPERATURE", "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1", "INTAKE_MANIFOLD_PRESSURE",
    "ACCELERATOR_PEDAL_POSITION_D", "ACCELERATOR_PEDAL_POSITION_E",
    "COMMANDED_THROTTLE_ACTUATOR", "INTAKE_AIR_TEMPERATURE",
    "TIMING_ADVANCE", "CONTROL_MODULE_VOLTAGE",
]
```

---

## Windowing Constants (Locked)

```python
SAMPLE_RATE_HZ   = 1      # 1 row per second (dataset property, not a choice)
WINDOW_LENGTH_S  = 60     # 60-second sliding windows
WINDOW_STRIDE_S  = 10     # 10-second stride between windows
FORECAST_HORIZON_S = 60   # predict 60 seconds ahead
RANDOM_SEED      = 42
```

---

## Injection Design Rules

1. **Two injection modes for every fault:** `step` (sudden sensor failure) and `ramp` (gradual wear degradation).
2. **Injection window:** 40–60 % of the session's window sequence, after a clean baseline period.
3. **Correlated effects:** If a fault alters one PID, update all physically-coupled PIDs. Example: MAP offset for air_system must trigger a correlated STFT response.
4. **Document every delta:** Each injector must store the parameters used (onset index, magnitude, mode) for reproducibility.
5. **Clamp before write:** Never write a value outside physical bounds. Clamp silently and assert the clamp was applied.

---

## Success Criteria

| Metric | Commit Target | Stretch Target |
|---|---|---|
| Classifier macro-F1 | ≥ 0.80 | ≥ 0.88 |
| Forecaster MAE | ≤ 15 % of severity range | ≤ 10 % |
| Dashboard latency | ≤ 2 s | ≤ 1 s |
| Fresh clone → tests pass | ≤ 30 min | ≤ 10 min |

**Week 4 checkpoint rule:** If macro-F1 < 0.70, drop deep-learning comparison and ship XGBoost only.

---

## Code Style Rules

- No whole-file rewrites. Precise diffs only.
- No synthetic data that violates engine physics (see Physics-First Rule above).
- No comments explaining *what* code does. Comments only for *why* (hidden constraints, physics workarounds).
- All new modules get at least one pytest test before moving to the next task.
- Session-level train/test split (by file, not by row) to prevent data leakage across windows.

---

## Key Source Files

| File | Purpose |
|---|---|
| `src/config.py` | All constants and paths. Import from here, never hardcode. |
| `src/data_loading.py` | carOBD loader; `load_carobd_csv()`, `list_usable_files()` |
| `src/injection/fault_injector.py` | Physics-constrained fault injection engine (Week 2) |
| `src/features/` | Windowing, feature extraction pipeline (Week 3) |
| `src/models/` | Classifier + forecaster (Weeks 3–6) |
| `src/dashboard/` | Streamlit live dashboard (Weeks 4, 7) |
| `docs/CHARTER.md` | Full scope, evaluation criteria, decision rules |
| `docs/DATA_NOTES.md` | Data quality audit findings, physical bounds tables |
