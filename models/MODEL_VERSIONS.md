# Model artefact versions

This file tracks **semantic** changes to deployed model artefacts —
changes to what a saved `.pkl` file actually predicts. Add an entry
whenever a forecaster, classifier, or detector's target distribution
changes such that an old artefact would not be safely interchangeable
with a new one.

The file lives in `models/` so it ships alongside the artefacts and
diff'd alongside them in PRs that introduce breaking changes.

---

## Format

For each artefact, list the version, the date, the new semantics, and
explicitly what would break in code that loaded the old version.

---

## `forecaster_v1.pkl`

| Version | Date | Class | Target | Notes |
|---|---|---|---|---|
| **v1** (current) | 26 May 2026 | `src.models.forecaster.FaultForecaster` | Per-fault severity scalar in [0, 1] derived from `src.features.severity.compute_severity` | This is the **legacy severity forecaster**. The target is the algebraic inverse of the injector's own coefficients (see project root README "Headline numbers"). |

**Status:** preserved for reproducibility of the synthetic self-consistency
floor in the paper. Not the centrepiece of v2 framing.

---

## `pid_forecaster_v1.pkl`

| Version | Date | Class | Target | Notes |
|---|---|---|---|---|
| **v1** | 29 May 2026 | `src.models.pid_forecaster.PIDForecaster` | Four z-scored next-window PID values (LTFT, MAP, coolant, throttle-to-pedal ratio) at t + 60 s | Built as the principled replacement for `forecaster_v1.pkl`'s severity target. Trained on healthy carOBD windows only — no fault labels, no severity formula, no injector inverse. Health-residual = ‖predicted_z − actual_z‖ per PID. |

**Initial training results** (`results/pid_forecaster_v1_results.json`):

| PID | MAE (z-units) | Persistence baseline | Verdict |
|---|---:|---:|---|
| `COOLANT_TEMPERATURE__mean` | 0.05 | 0.11 | beats persistence 2× |
| `INTAKE_MANIFOLD_PRESSURE__mean` | 0.79 | 0.80 | tied with persistence |
| `THROTTLE_TO_PEDAL_RATIO` | 0.91 | 0.90 | marginally worse |
| `LONG_TERM_FUEL_TRIM_BANK_1__mean` | 2.90 | 0.49 | **dramatically worse — session-overfit** |

**Honest finding for the paper:** healthy-only forecasting on a 60-s
horizon works for slow thermal signals (coolant) but degrades sharply
for ECU-state signals that encode session/vehicle-specific operating
context (LTFT). A production deployment needs per-vehicle baseline-fit
of the forecaster, or a hybrid "predict residual relative to per-vehicle
baseline" approach. This is the same cross-session generalisation
problem the anomaly detector exhibits (`results/anomaly_v1_results.json`).

---

## Planned migration: `forecaster_v1.pkl` → severity-legacy

When the v1.2 charter amendment lands (after the honest-framing PR
series completes), the following swap is planned:

1. `src/models/forecaster.py` → relocate to `src/legacy/severity_forecaster.py`
   (file moves; class name preserved; the existing `forecaster_v1.pkl`
   artefact still loads via the legacy class).
2. `forecaster_v1.pkl` stays as the legacy filename — gives anyone with
   an old checkout a deterministic load path; the file is gitignored
   anyway.
3. The dashboard's "60-Second Forecast" panel will be re-labelled as
   "Synthetic forecast (sanity)" with a tooltip pointing at this file,
   and a new "PID-residual" panel will read from `pid_forecaster_v1.pkl`.

That swap is scheduled as a follow-up to Step 4 of the honest-framing
PR series. It is not done in this commit because the dashboard's panel
geometry and SHAP/forecast wiring have several callers — splitting the
new-model build from the panel swap keeps each PR's blast radius small.
