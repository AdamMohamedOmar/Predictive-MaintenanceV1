# Roomster Pass-3 Fix Plan

Third external review surfaced 15 suggestions, framed as comparisons against
mature OBD/predictive-maintenance repos on GitHub. Each was validated against
the current tree. This document is the execution plan for Sonnet.

The classification:
- **DONE in Tier 3** — already implemented in pass-2 work
- **Open in Tier 4/5** — overlaps an item from `ROOMSTER_PASS2_FIXES.md` that is
  still pending; no duplicate work, just confirm the pass-2 recipe still holds
- **NEW (Tier 6)** — genuinely new suggestions; full recipe below

> **Discipline rule unchanged:** physics-first, precise diffs, no whole-file rewrites.
> Each task has a verification step — do not move on until that step passes.

---

## Validation Summary

| # | Report claim | Verdict | Status | Fix tier |
|---|---|---|---|---|
| 1 | Wall-clock 60-s windows, resample to 1 Hz | ✅ True | **DONE in T3.1** | — |
| 2 | Time-aware cold-start rules | ✅ True | **DONE in T3.2** | — |
| 3 | Fix DTC and UI copy | ✅ True | Open | **T4.1/T4.2/T4.3** (pass-2) — *decision point on codes* |
| 4 | E-gas feature: THROTTLE − COMMANDED_THROTTLE_ACTUATOR | ✅ True | Open | **T5.7** (pass-2) |
| 5 | Stronger plausibility layer (MAP≤baro, |Δcoolant|≤2°C/s, thr-vs-cmd) | ✅ True | Open | **T5.3** (pass-2) |
| 6a | Make LOSO the headline metric | ✅ True | Open | **T5.5** (pass-2) |
| 6b | Down-weight overlapping windows | ✅ True | Already considered "intentional" in pass-2 | **T6.6** (revisit) |
| 6c | Probability calibration (Platt/isotonic) | ✅ True | NEW | **T6.1** |
| 6d | Rename UI "probability" → "model score" | ✅ True | NEW | **T6.2** |
| 7 | Vehicle profile JSON for Roomster | ✅ True | NEW | **T6.3** |
| 8 | One real fault recording (vacuum leak / MAF disconnect) | ✅ True | NEW (procedural) | **T6.10** |
| 9a | Forecaster trained on mid-range severity (0.1–0.9) | ✅ True | NEW | **T6.7** |
| 9b | TPS ordinal-buckets fallback | ✅ True | NEW (conditional) | **T6.8** |
| 10 | Idle-weighted air-system signal | ✅ True | Open | **T5.6** (pass-2) |
| 11 | Minimal PID fallback set in live_discover.py | ✅ True | NEW | **T6.4** |
| 12a | Group SHAP by PID family (one bar per signal) | ✅ True | NEW | **T6.5** |
| 12b | PDF session report export | ✅ True | NEW (optional) | **T6.9** |
| 12c | VIN / engine-code field per session | ✅ True | NEW | rolled into **T6.3** |
| 13 | Thesis framing: cite carOBD author, LOSO as main number | ✅ True | Mostly **T5.5** + small additions | **T6.11** |
| 14 | Do not z-score regime flags | ✅ True | Open | **T5.1** (pass-2) |
| 15a | Reconnect: re-run RPM liveness | ✅ True | **DONE in T3.4** | — |
| 15b | Pause ML until poll rate stable for N seconds | ✅ True | NEW | **T6.12** |

**Items already shipped (Tier 3):** 1, 2, 15a
**Items already planned in pass-2 (Tier 4/5):** 3, 4, 5, 6a, 10, 14
**New work (Tier 6):** 6b, 6c, 6d, 7, 8, 9a, 9b, 11, 12a, 12b, 12c, 13, 15b

---

## Resolved Decisions (locked in 2026-05-26)

### D1 ✓ fuel_system DTC = **"P0171 / P0087"** (paired)

Pass-2's T4.1 recipe is **superseded** by this dual-code entry. The final
DTC map entry for `fuel_system` is:

```python
"fuel_system": {
    "code": "P0171 / P0087",
    "name": "Lean Mixture + Low Fuel Pressure",
    "short": "Lean+Fuel",
    "description": (
        "Sustained positive LTFT (P0171) consistent with under-delivery "
        "of fuel. Probable mechanical root cause: low fuel rail pressure "
        "(P0087) from clogged filter, weak pump, partially blocked "
        "injector, or stuck pressure regulator."
    ),
},
```

The recommendations text (pass-2 T4.3) is updated to match:
```python
"Pull DTCs — confirm P0171 (lean, Bank 1) and/or P0087 (low rail pressure). "
"If misfire codes accompany either, suspect injector clog rather than "
"pump/filter.",
```

### D2 ✓ throttle_position_sensor DTC = **"P2135"**

Pass-2's T4.2 recipe stands unchanged. The Roomster is drive-by-wire;
P2135 (TPS A/B Correlation) is the workshop-scan-tool code that matches
our `_inject_tps` divergence injection.

### D3 ✓ Window overlap — report both stride-10 and stride-60 LOSO

Stride-10 stays in training (matches dataset spec, preserves the 0.958
LOSO number). The thesis reports **both** stride-10 (in-training-distribution)
and stride-60 (non-overlapping, honest generalisation) numbers — the gap
between them is the disclosed "overlap inflation". Implemented in **T6.6**.

---

## Tier 6 — Pass-3 additions

### T6.1  Probability calibration (Platt / isotonic on held-out sessions)

**Why:** XGBoost's `predict_proba` outputs are not probabilistically calibrated.
A "73% confidence" prediction is currently around 50% likely to be correct in
practice. Thesis reviewers know this and will challenge any unflagged
"probability" plot.

**Files:** `src/models/xgb_classifier.py`, `src/models/calibration.py` (new)

**Approach:** wrap the trained `XGBClassifier` in a `CalibratedClassifierCV`
fit on held-out sessions (NOT the training sessions — leakage), using
`method="isotonic"` (more flexible than Platt for multi-class).

```python
# In src/models/xgb_classifier.py, after the existing fit() call:
from sklearn.calibration import CalibratedClassifierCV

# Split a small calibration set from train (session-level, NOT row-level).
# Pick 2 sessions held out from both training and the final test set.
CALIB_SESSIONS = {"live8", "live9"}  # holdout-of-holdout — different from _HELD_OUT_SESSIONS
calib_mask = train_df["session_id"].isin(CALIB_SESSIONS)
calib_df = train_df[calib_mask]
train_subset = train_df[~calib_mask]

# Retrain on train_subset, then calibrate using calib_df.
clf_raw.fit(X_train_subset, y_train_subset)
clf = CalibratedClassifierCV(clf_raw, method="isotonic", cv="prefit")
clf.fit(X_calib, y_calib)
```

**Acceptance:** macro-F1 must drop ≤ 0.02 after calibration (calibration
typically costs ~1 % F1). Add Brier score and reliability diagram to
`results/calibration_diagnostics.json`.

**Verification:** `tests/test_calibration.py` — assert `np.abs(clf.predict_proba(X).sum(axis=1) - 1.0).max() < 1e-6` and that the reliability slope is in [0.8, 1.2].

---

### T6.2  Rename UI "probability" → "model score"

**Files:** `src/dashboard/app.py`, `src/dashboard/inference.py` (no logic change).

Search for the user-facing strings `confidence`, `probability`, `prob` in
labels/captions and replace with `model score` (lowercase). The internal
field names in `DashboardState` (`classifier_confidence`, `all_class_probs`)
stay unchanged — only the rendered text changes.

Add a one-line caption under the score: *"Calibrated score from XGBoost +
isotonic regression on held-out sessions. Not a true probability — see thesis
§5.3."*

**Verification:** open the dashboard, every banner/tooltip/chart legend reads
"model score" not "probability" / "confidence".

---

### T6.3  Vehicle profile JSON (Roomster 1.6 BTS)

**Why:** Etios-tuned constants (`_IAC_HIGH_RPM_THRESHOLD=1100`, severity
baselines, sanity bounds) drive false alarms on the Roomster. A small
per-vehicle profile loaded at startup decouples ML weights (vehicle-agnostic
after z-scoring) from rule thresholds (vehicle-specific).

**Files:** `vehicles/roomster_1.6_bts.json` (new), `vehicles/etios_1.5.json`
(new — extract current Etios constants), `src/dashboard/inference.py`,
`src/diagnostics/cold_start_checker.py`.

**Schema:**
```json
{
  "vehicle": "Skoda Roomster 2007 1.6 MPI",
  "engine_code": "BTS",
  "vin": null,
  "warmup_target_c": 80.0,
  "closed_loop_min_coolant_c": 60.0,
  "nominal_idle_rpm": 750,
  "iac_high_rpm_threshold": 1000,
  "healthy_ltft_band_pct": [-2.0, 6.0],
  "healthy_stft_band_pct": [-5.0, 5.0],
  "barometric_kpa_fallback": 100.0,
  "min_supported_pids_for_demo": 8
}
```

**Loader:** `src/config/vehicle_profile.py`:
```python
@dataclass(frozen=True)
class VehicleProfile:
    vehicle: str
    engine_code: str
    vin: str | None
    warmup_target_c: float
    closed_loop_min_coolant_c: float
    nominal_idle_rpm: int
    iac_high_rpm_threshold: int
    healthy_ltft_band_pct: tuple[float, float]
    healthy_stft_band_pct: tuple[float, float]
    barometric_kpa_fallback: float
    min_supported_pids_for_demo: int

    @classmethod
    def load(cls, path: Path) -> "VehicleProfile":
        ...
```

**Integration:** `ColdStartChecker.__init__` accepts an optional
`profile: VehicleProfile | None`. When given, override `_WARMUP_TARGET_TEMP`,
`_IAC_HIGH_RPM_THRESHOLD`, `_IAC_WARM_MIN_S` etc. on the instance.
`InferenceEngine.__init__` takes `profile_path: Path | None`; the dashboard
sidebar exposes a profile selector that defaults to "etios_1.5" but switches
to "roomster_1.6_bts" for the live demo.

**Verification:** `tests/test_vehicle_profile.py` — load both JSONs, assert
their fields type-check; `tests/test_cold_start_checker.py::test_iac_uses_profile_threshold` — feed a row at 950 RPM with the Roomster profile (threshold 1000) and confirm no alert fires.

---

### T6.4  Minimal PID fallback set in live_discover.py

**Why:** The 14-PID query at 0.3 Hz is the bottleneck on cheap ELM327 clones.
If we accept losing 6 of the 14 PIDs and querying only the 8 high-value ones,
the same hardware can sustain 0.8 Hz.

**Files:** `scripts/live_discover.py`, `src/config.py`.

Add to `src/config.py`:
```python
MIN_DEMO_PIDS: tuple[str, ...] = (
    "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE",
    "COOLANT_TEMPERATURE", "INTAKE_MANIFOLD_PRESSURE",
    "SHORT_TERM_FUEL_TRIM_BANK_1", "LONG_TERM_FUEL_TRIM_BANK_1",
    "ACCELERATOR_PEDAL_POSITION_D",
)
```

In `live_discover.py::evaluate`, when poll rate is below 0.8 Hz, check
whether the 8-PID subset would have made the cut and recommend it:
```python
if actual_hz < MIN_POLL_HZ:
    expected_hz_8 = actual_hz * (n_supported / 8.0)
    if expected_hz_8 >= MIN_POLL_HZ:
        reasons.append(
            f"At full 14 PIDs the adapter sustains only {actual_hz:.2f} Hz. "
            f"Restricting to MIN_DEMO_PIDS (8 signals) would project ≈ "
            f"{expected_hz_8:.2f} Hz. Use --pid-subset min_demo for a "
            f"degraded-feature demo run."
        )
    else:
        reasons.append(...)
```

**Acceptance:** classifier still produces a defensible output with 6
NaN-filled PID features (existing NaN-fill logic already handles this).

**Verification:** mental walkthrough on a real Roomster log.

---

### T6.5  Group SHAP bars by PID family

**Why:** A single window often shows the top-5 SHAP features as five
slightly-different LTFT-derived bars (`LTFT__mean`, `LTFT__std`,
`LTFT__delta`, `FUEL_TRIM_DIVERGENCE`, `FUEL_LOOP_ACTIVE`). To a workshop
tech this is noise — they want to see "LTFT contributed +0.42, MAP
contributed +0.18, all else negligible."

**File:** `src/dashboard/app.py` (the `_render_shap_bars` or equivalent
function — locate via `Grep` for `top_features`).

**Approach:** aggregate SHAP contributions by PID before rendering. The
"family" of a feature is the prefix before `__` (e.g. `LTFT` for
`LONG_TERM_FUEL_TRIM_BANK_1__mean`). Cross-PID features (`THROTTLE_TO_PEDAL_RATIO`, `MAP_PER_THROTTLE`, `FUEL_TRIM_DIVERGENCE`,
`COOLANT_WARMUP_RATE`, etc.) each get their own group.

Add `src/dashboard/shap_grouper.py`:
```python
def group_shap_by_family(top_features: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Aggregate SHAP values by PID family.

    Returns groups sorted by absolute contribution, descending.
    """
    PID_FAMILIES = {
        "ENGINE_RPM": "RPM", "VEHICLE_SPEED": "Speed", "THROTTLE": "Throttle",
        "ENGINE_LOAD": "Load", "COOLANT_TEMPERATURE": "Coolant",
        "LONG_TERM_FUEL_TRIM_BANK_1": "LTFT",
        "SHORT_TERM_FUEL_TRIM_BANK_1": "STFT",
        "INTAKE_MANIFOLD_PRESSURE": "MAP", ...
    }
    groups: dict[str, float] = {}
    for name, val in top_features:
        prefix = name.split("__")[0]
        family = PID_FAMILIES.get(prefix, prefix)
        groups[family] = groups.get(family, 0.0) + val
    return sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True)
```

Render the grouped list in the UI; keep the raw per-feature list available
behind an "Expand details" toggle for thesis-grade debugging.

**Verification:** `tests/test_shap_grouper.py` — feed 5 LTFT-derived
features, confirm output is a single `LTFT` row with the sum of contributions.

---

### T6.6  Stride-60 LOSO as a secondary headline number

**Files:** `scripts/loso_cv.py`, `results/loso_cv_results.json`.

After the main stride-10 LOSO run, add a second pass with stride matching
window length so the test windows do not overlap:

```python
# In loso_cv.py after the main loop:
from src.features.dataset_builder import build_dataset
ds_nonoverlap = build_dataset(stride_s=60)   # build a non-overlapping copy
f1_scores_no = []
for held in sessions:
    train_df = ds_nonoverlap[ds_nonoverlap["session_id"] != held]
    test_df  = ds_nonoverlap[ds_nonoverlap["session_id"] == held]
    clf, norm = train_clf(train_df, n_estimators=300, random_seed=42)
    res = eval_clf(clf, norm, test_df)
    f1_scores_no.append(res["macro_f1"])
```

Save both means in `loso_cv_results.json`:
```json
{
  "stride_10_mean_f1": 0.958,
  "stride_60_mean_f1": <reported>,
  "delta": <difference — the overlap inflation>
}
```

**Verification:** the script runs without error and the gap between
stride-10 and stride-60 is reported.

---

### T6.7  Forecaster trained on mid-range severity (0.1–0.9)

**Why:** the forecaster currently learns to predict 0.0 most of the time
(healthy + early-injection windows have severity ≈ 0). It also learns to
saturate at 1.0 for fully developed faults. The interesting band is the
middle, where prediction is most useful for early warning.

**Files:** `src/models/forecaster.py`.

In `_train_one`, after building `(X_train, y_train)`:
```python
# Keep only mid-range severity windows for fitting; healthy and saturated
# rows train the model to ignore the early-warning band.
keep = (y_train > 0.1) & (y_train < 0.9)
if keep.sum() < 50:
    log.warning("%s: only %d mid-range rows — falling back to all rows.",
                fault_type, int(keep.sum()))
else:
    X_train, y_train = X_train[keep], y_train[keep]
```

**Acceptance:** mid-range MAE (computed on test rows with `0.1 < y < 0.9`)
must improve by ≥ 10 % vs the pass-2 baseline; overall MAE may worsen
slightly (acceptable — the band we care about is mid-range).

**Verification:** rerun `python -m scripts.rebuild_all`; check
`results/forecaster_v1_results.json` shows `mid_range_mae < baseline_mid_range_mae`.

---

### T6.8  TPS ordinal-buckets fallback (conditional)

**Trigger condition:** only execute T6.8 if T6.7 fails to bring
`throttle_position_sensor` MAE below 25 % (current pass-2 ceiling is 35 %).

**Files:** `src/models/forecaster.py`.

Replace the regression head for the TPS forecaster with a 3-class
classifier (`low: 0–0.33`, `medium: 0.33–0.66`, `high: 0.66–1.0`) and
return the bucket midpoint when called via `predict()`. The bucket band
is what a workshop tech actually wants — "is this getting worse?" not
"is this 0.43 or 0.51?"

**Verification:** add `tests/test_forecaster_tps_ordinal.py` — assert
`predict()` returns one of `{0.165, 0.495, 0.825}` for the TPS fault.

---

### T6.9  PDF session report (optional / time-permitting)

**Why:** workshops accept printed paperwork. A one-page PDF per session
makes the project demo-able beyond the laptop.

**File:** `src/dashboard/pdf_export.py` (new), `src/dashboard/app.py`.

Use `reportlab` (already in the `fourpoints` reference) to render: vehicle
identification, capture date, regime breakdown, top-3 alerts (rule + ML),
SHAP grouped bars, recommended steps. Add a "Export PDF" button to the
dashboard sidebar.

**Verification:** dashboard generates a non-empty PDF; manual eyeball check.

---

### T6.10  One real fault recording — protocol document

**File:** `docs/REAL_FAULT_PROTOCOL.md` (new).

Procedural document (no code) capturing:
- Exact step to inject a real, reversible fault on the Roomster
  (recommended: pull a vacuum line at the brake booster or PCV with the
  engine warm and idling — induces a real air_system fault for ~30 s)
- Required dashboard state before disconnecting
- CSV save path
- Cleanup / safety steps
- Expected dashboard reaction (`air_system` label, severity climbing
  from 0.0 to ~0.5)

This becomes the thesis credibility moment — one real labelled fault
beats ten more synthetic ones.

---

### T6.11  CHARTER additions — carOBD baseline citation

**File:** `docs/CHARTER.md`.

Already partly covered by pass-2's T5.5. Add a paragraph citing
[eron93br/carOBD](https://github.com/eron93br/carOBD) — the original
thesis on the same Etios 1 Hz dataset used **one-class ECT anomaly
detection only**. Our six-class synthetic-fault scope is a deliberate
extension; honest scoping in §1 of the thesis.

---

### T6.12  Stable-poll-rate gate

**Why:** at connect time the first 2–3 windows are fed at whatever the
ELM327 happens to deliver. If the poll rate is bouncing between 0.2 and
0.8 Hz, the hold-last resampler (T3.1) over- or under-fills early
windows. Pause ML classification until the rate has been stable for
≥ 10 consecutive ticks within ±10 % of its running mean.

**Files:** `src/live/obd_source.py`, `src/dashboard/inference.py`.

In `LiveObdSource`, track a rolling-mean filter on `_last_poll_duration_s`:
```python
def __init__(...):
    ...
    self._poll_durations: deque[float] = deque(maxlen=10)
    self._stable_since_s: float | None = None

# Inside _poll_loop after computing poll_duration:
self._poll_durations.append(poll_duration)
if len(self._poll_durations) == 10:
    mean = sum(self._poll_durations) / 10
    spread = (max(self._poll_durations) - min(self._poll_durations)) / mean
    if spread < 0.1:
        if self._stable_since_s is None:
            self._stable_since_s = time.monotonic()
    else:
        self._stable_since_s = None

@property
def poll_stable_s(self) -> float:
    if self._stable_since_s is None:
        return 0.0
    return time.monotonic() - self._stable_since_s
```

In `InferenceEngine._process_one_row`, gate the classifier on
`poll_stable_s ≥ 10.0` (passed in via `set_poll_stable_s(...)` from
`app.py`). Until stable, return a `state.classifier_label = "warming_up"`
banner reading "Waiting for stable poll rate…".

**Verification:** `tests/test_poll_stability.py` — feed 10 synthetic
poll_duration values; assert `poll_stable_s` advances only when spread
< 10 %.

---

## Execution order (Tier 6)

Recommended sequence — smallest correctness wins first, larger refactors last:

1. **T6.2** (UI text relabel) — 5-minute change, removes a thesis liability
2. **T6.6** (stride-60 LOSO) — script-only, no model change
3. **T6.11** (CHARTER additions) — docs only
4. **T6.4** (minimal PID set) — 30 lines in `live_discover.py` + `config.py`
5. **T6.10** (real-fault protocol) — docs only
6. **T6.5** (SHAP grouping) — UI refactor; no model change
7. **T6.7** (mid-range forecaster training) — needs rebuild
8. **T6.1** (probability calibration) — needs rebuild + new test
9. **T6.3** (vehicle profile JSON) — largest change; touches multiple modules
10. **T6.12** (poll-stability gate) — depends on T6.3 for the gate-time constant
11. **T6.8** (TPS ordinal buckets) — only if T6.7 result is not good enough
12. **T6.9** (PDF export) — if and only if time remains before the demo

**Tier 4/5 ordering reminder:** pass-2's recommended Tier-4 (DTC fixes) and
Tier-5 items remain valid; the user must first resolve **D1** and **D2**
above, then a Sonnet session can run the pass-2 plan in its prescribed order.

---

## Bundling into PRs

| PR | Contents |
|---|---|
| 1 | **Tier 4** (T4.1+T4.2+T4.3) — DTC + recommendation copy. Single small commit. |
| 2 | **Tier 5 quick wins** (T5.2 + T5.4 + T5.5) — degraded-PID threshold, dead-code, CHARTER. |
| 3 | **Tier 5 sanity** (T5.3) — three new sanity rules. |
| 4 | **Tier 5 feature/model rebuild** (T5.1 + T5.6 + T5.7) — regime z-score fix, idle-weighted air injection, commanded-vs-actual throttle. Triggers a full `rebuild_all`. |
| 5 | **Tier 6 docs + UI** (T6.2 + T6.5 + T6.6 + T6.10 + T6.11) — no model changes. |
| 6 | **Tier 6 PID gating** (T6.4 + T6.12) — discover script + poll-stability gate. |
| 7 | **Tier 6 model rebuild** (T6.1 + T6.7) — calibration + mid-range forecaster. |
| 8 | **Tier 6 vehicle profile** (T6.3) — vehicle JSON + loader + integration. |
| 9 | **Tier 6 optional** (T6.8, T6.9) — only if there's time before the demo. |

---

## Verification gate before the Roomster demo (extends pass-2 list)

After Tier 6 lands, repeat the pass-2 bench dry-run plus:

6. Open the dashboard with the **Roomster** profile selected. Confirm: the
   warmup target reads 80 °C, the IAC threshold reads 1000 RPM, the
   barometric fallback reads 100 kPa.
7. Connect with the engine cold. Confirm: classifier stays on
   "warming_up" until the poll rate has been stable for 10 s, then
   transitions to live classification.
8. Pull a vacuum line on the warm engine. Confirm within 60 s:
   `air_system` model score climbs above 0.6, top SHAP groups show
   `LTFT > STFT > MAP`. Re-attach the line; severity decays toward 0
   over the next 90 s.
9. Print the session as PDF. Confirm engine code "BTS" and chosen
   profile name appear on the printed page.
