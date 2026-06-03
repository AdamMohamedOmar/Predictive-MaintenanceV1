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
| **v1** | 29 May 2026 | `src.models.pid_forecaster.PIDForecaster` | Next-window PID **DELTA** (future − now) for LTFT, MAP, coolant, throttle-to-pedal ratio at t + 60 s, scaled by each PID's healthy σ (P1-3) | Replaces `forecaster_v1.pkl`'s severity target. Trained on healthy carOBD windows only — no fault labels, no severity formula, no injector inverse. Forecasting the DELTA (not the absolute level) cancels the per-session baseline offset. Level reconstructed at inference as current + predicted_delta; health-residual = ‖predicted_level_z − actual_level_z‖ per PID. |

**Training results after the P1-3 delta retarget** (`results/pid_forecaster_v1_results.json`):

| PID | MAE (z) | Persistence (predict Δ=0) | Verdict |
|---|---:|---:|---|
| `COOLANT_TEMPERATURE__mean` | 0.05 | 0.11 | **beats** persistence 2× |
| `INTAKE_MANIFOLD_PRESSURE__mean` | 0.77 | 0.80 | **beats** persistence |
| `THROTTLE_TO_PEDAL_RATIO` | 0.83 | 0.90 | **beats** persistence (was losing pre-P1-3) |
| `LONG_TERM_FUEL_TRIM_BANK_1__mean` | 1.15 | 0.49 | still worse (improved from 2.90) |

**Honest finding for the paper:** the delta retarget (P1-3) flipped MAP and
the throttle ratio from tied/losing to **beating** persistence, and roughly
halved the LTFT error — but LTFT **still does not beat** the persistence
baseline. That is itself a legitimate result: a healthy engine's LTFT barely
moves over 60 s, so "predict no change" is near-optimal and a learned model
only adds variance. LTFT forecasting is reported as **explored, did not beat
baseline**; the other three PIDs are shipped. The remaining LTFT gap is the
same cross-session generalisation limit the anomaly detector shows
(`results/anomaly_v1_results.json`) — addressed by per-vehicle baseline re-fit.

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
