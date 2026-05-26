# Roomster Pass-2 Fix Plan

Second expert review identified 17 issues. All were independently validated by reading the current tree. This document is the **execution plan for Sonnet**: each fix has a verdict, the exact files/lines, and an implementation recipe.

The plan is partitioned into three tiers:

- **Tier 3 — Live-deployment blockers** (T3.x) — affect a real Roomster demo via ELM327. Fix before the demo.
- **Tier 4 — Workshop-facing correctness** (T4.x) — wrong DTC codes, wrong recommendations. Cheap to fix, large credibility win.
- **Tier 5 — ML & evaluation polish** (T5.x) — methodology cleanup and feature engineering improvements. Worth it for the thesis.

> **Discipline rule:** physics-first, precise diffs, no whole-file rewrites. Each task has a verification step — do not move on until that step passes.

---

## Validation Summary

| # | Report claim | Verdict | Severity | Fix tier |
|---|---|---|---|---|
| 1 | 60-row deque ≠ 60 wall-clock seconds at sub-1 Hz polling | ✅ True | **CRITICAL** | T3.1 |
| 2 | `ColdStartChecker` timers count rows, not seconds | ✅ True | **CRITICAL** | T3.2 |
| 3 | `measured_poll_hz=0` on first connect → clipped to 0.05 Hz | ✅ True | HIGH | T3.3 |
| 4 | `_try_reconnect()` skips the RPM liveness check | ✅ True | HIGH | T3.4 |
| 5 | `fuel_system` mapped to P0172 ("rich") but injection is lean | ✅ True | HIGH | T4.1 |
| 6 | `throttle_position_sensor` mapped to P0122 ("low") but TPS drifts upward | ✅ True | MEDIUM | T4.2 |
| 7 | Recommendations text mentions wrong DTC codes | ✅ True | LOW | T4.3 |
| 8 | 60-row training windows with 10-step stride → heavy overlap | ✅ True (intentional) | LOW | (no change — note in thesis) |
| 9 | StandardScaler z-scores binary regime one-hots | ✅ True | LOW-MEDIUM | T5.1 |
| 10 | NaN imputation; degraded-PID warning fires only at ≥ 3 | ✅ True | MEDIUM | T5.2 |
| 11 | `sanity.py` missing MAP ≤ baro, throttle vs commanded, coolant-rate checks | ✅ True | MEDIUM | T5.3 |
| 12 | `scripts/loso_cv.py:38` assigns `norm` then ignores it | ✅ True | TRIVIAL | T5.4 |
| 13 | CHARTER says 5-fold CV; rebuild uses fixed holdout `{drive1, live12}` | ✅ True | MEDIUM (thesis) | T5.5 |
| 14 | `_inject_air_system` not idle-weighted | ✅ True (design choice) | LOW | T5.6 |
| 15 | `TIMING_VS_TEMP` linear-in-coolant; `COMMANDED_THROTTLE_ACTUATOR` unused as residual | ✅ True | MEDIUM | T5.7 |
| 16 | `ColdStartChecker` cannot detect ECT stuck-at-warm or post-warmup freeze | ✅ True (documented limitation) | LOW | (no change — already documented at line 261-275) |
| 17 | `scripts/live_baseline_capture.py:121` calls `extract_features(window)` without `sample_hz` | ✅ True | **HIGH** | T3.5 |

---

## Tier 3 — Live-deployment blockers

### T3.1  Resample live rows so the 60-row window is 60 wall-clock seconds

**Files:** `src/dashboard/inference.py`, `src/live/obd_source.py`

**Why this matters:** T1.2 fixed two rate-dependent features (`COOLANT_WARMUP_RATE`, `FUEL_LOOP_ACTIVE`), but the buffer still feeds the classifier 60 raw rows. At a measured 0.3 Hz on the Skoda, one "60-sample window" spans 200 seconds of real driving. The XGBoost model was trained on windows that were always 60 seconds. Every PID's `__std`, `__delta`, and regime one-hot is now off-distribution, and the LOSO F1 of 0.96 does **not** apply to that live signal.

**Approach:** Add a clock-driven resampler in `InferenceEngine.update()` so the deque only receives one row per real second.

1. Attach a wall-clock timestamp to each row at the source. In `LiveObdSource._poll_loop()` right before pushing to the queue:
   ```python
   row["__t"] = time.monotonic()
   ```
   In `CsvStreamer` (csv mode), set `row["__t"] = self._elapsed_s` (or omit — see step 3).

2. In `InferenceEngine.__init__`, track the next 1-Hz tick:
   ```python
   self._next_sample_t: float | None = None
   ```

3. In `InferenceEngine.update(row)`, **before** any other work:
   ```python
   t = row.pop("__t", None)
   if t is not None:                 # live path
       if self._next_sample_t is None:
           self._next_sample_t = t
       if t < self._next_sample_t:
           # Row arrived faster than 1 Hz — drop and return last state.
           return self._last_state
       # Hold-last semantics: if poll was slower than 1 Hz, replay the same
       # row at each missed 1-second slot so the buffer stays time-aligned.
       while t >= self._next_sample_t:
           self._next_sample_t += 1.0
           self._process_one_row(row)   # extract the body of the current update()
       return self._last_state
   # Csv path falls through to original logic.
   ```

4. Refactor: move the existing body of `update()` into `_process_one_row(row)` so the resampler can call it 0…N times per incoming row.

5. **Important:** keep the existing `set_sample_hz()` plumbing — the rate-dependent features (T1.2) still need the *raw* poll rate to compute `COOLANT_WARMUP_RATE` correctly *within* the resampled stream. The resampler upstreams 1-Hz rows; the features downstream should be told `sample_hz=1.0` after resampling. Update `app.py` line 1113: pass `1.0` (not `measured_poll_hz`) once the resampler is in place.

6. **Refuse to classify at extreme slow polls.** If `measured_poll_hz < 0.3`, the hold-last buffer fills with near-identical rows and `__std` features collapse to zero. Add to the dashboard's status banner:
   ```python
   if poll_hz > 0 and poll_hz < 0.3:
       st.error(
           "Adapter poll rate {:.2f} Hz is below the 0.3 Hz floor — "
           "classifier features are unreliable. Use a faster ELM327.".format(poll_hz)
       )
   ```

**Verification:** Add `tests/test_inference_resampler.py` with two cases:
- Feed 200 rows at synthetic `__t = i * 3.33` (0.3 Hz). Assert the deque advances by ~60 1-second ticks for every 18 input rows.
- Feed 200 rows at `__t = i * 0.5` (2 Hz). Assert every other row is dropped.

---

### T3.2  Make `ColdStartChecker` time-based, not row-based

**File:** `src/diagnostics/cold_start_checker.py`

**Why this matters:** Every threshold (`_OVERHEAT_CONSECUTIVE_S = 30`, `_FROZEN_SENSOR_MIN_S = 90`, `_WARMUP_TIMEOUT_S = 480`, `_IAC_WARM_MIN_S = 120`, `_LOW_VOLTAGE_CONSECUTIVE_S = 60`) is documented in *seconds* but compared against `_elapsed_s` which increments by **1 per `update()` call**. At 0.3 Hz, the "90 s frozen ECT" rule needs ~5 minutes of real time. At 2 Hz, it fires in 45 seconds.

**Approach (preferred):** add an optional `now: float | None = None` parameter to `update()` and use wall-clock time when provided.

1. Change `__init__` to track a session start time:
   ```python
   self._session_start_t: float | None = None
   ```

2. Change `update()` signature:
   ```python
   def update(self, coolant, rpm, speed, voltage=14.0, *, now: float | None = None):
   ```

3. Replace the `self._elapsed_s += 1` line:
   ```python
   if now is not None:
       if self._session_start_t is None:
           self._session_start_t = now
       self._elapsed_s = int(now - self._session_start_t)
   else:
       self._elapsed_s += 1  # legacy fallback for 1 Hz csv mode
   ```

4. Update the only caller — `InferenceEngine.update()` — to pass `now=time.monotonic()` (live) or `now=self._elapsed_s` (csv). After T3.1 is in, the resampler makes this a constant 1-second tick anyway, so the difference becomes a safety belt rather than a hot path.

5. Audit the rule bodies for buffer-length assumptions:
   - `_check_frozen_sensor` slices `self._coolant_buf[-self._frozen_min_s:]` (last 90 entries). If `update()` is now called less than once per second, the buffer has fewer than 90 rows at 90 s elapsed. Replace with: `count = sum(1 for _ in self._coolant_buf if it's within the last 90 s)` — but the easiest fix is to **also resample upstream** (T3.1 covers this), then the buffers stay at 1 entry per second by construction.

   - `_check_overheat` and `_check_alternator` slice the last `_OVERHEAT_CONSECUTIVE_S` / `_LOW_VOLTAGE_CONSECUTIVE_S` entries. Same logic — depends on upstream resampling.

**Verification:** add `tests/test_cold_start_timing.py` that feeds 90 updates at `now = i * 3.0` (one update every 3 s) and asserts the frozen-sensor rule fires only after ≥ 30 updates (= 90 seconds), not 90 updates.

---

### T3.3  Treat `measured_poll_hz = 0` as "unknown" not "0.05 Hz"

**Files:** `src/dashboard/inference.py` (`set_sample_hz`), `src/dashboard/app.py` (the call site)

**Why this matters:** On the very first tick after connect, `LiveObdSource._last_poll_duration_s` is still 0, so `measured_poll_hz` returns 0. The current `set_sample_hz` clips that with `np.clip(hz, 0.05, 5.0)`, anchoring early windows to a 0.05 Hz time axis.

**Fix:** in `set_sample_hz`, treat `hz <= 0` (or `< 0.1`) as "not yet measured" and keep the previous value:

```python
def set_sample_hz(self, hz: float) -> None:
    if hz < 0.1:                # adapter has not completed a tick yet
        return                  # keep the previous (default 1.0) value
    self._sample_hz = float(np.clip(hz, 0.05, 5.0))
```

In `app.py` (the live branch around line 1112), keep the existing call — the guard moves into the engine method so the call site stays simple.

**Verification:** mental walkthrough. On reconnect after a drop, `_last_poll_duration_s` is held at the pre-drop value (the reconnect doesn't reset it), so this case is already covered.

---

### T3.4  Re-run the RPM liveness check on auto-reconnect

**File:** `src/live/obd_source.py` (`_try_reconnect`)

**Why this matters:** `connect()` blocks any reading until `ENGINE_RPM ≥ 50` to avoid pumping garbage from an ACC-mode ECU into the classifier. `_try_reconnect()` only checks `OBDStatus.CAR_CONNECTED`, which a cheap clone returns even with the key in ACC.

**Fix:** extract the RPM-liveness block from `connect()` lines 146-162 into a private helper and call it from both places.

```python
def _verify_engine_running(self) -> bool:
    """Return True only if ENGINE_RPM is being reported and is >= 50."""
    if "ENGINE_RPM" not in self._supported_pids:
        return True            # ECU doesn't expose RPM - can't verify either way
    import math
    try:
        resp = self._conn.query(PID_MAP["ENGINE_RPM"])
        rpm = to_float(resp)
        if math.isnan(rpm) or rpm < 50:
            log.warning("Reconnect: ENGINE_RPM=%s - ignition not in RUN.", rpm)
            return False
    except Exception as exc:
        log.warning("Reconnect: RPM liveness check failed: %s", exc)
        return False
    return True
```

In `_try_reconnect()`, after `self._supported_pids = self._discover_pids()` and before `self._connected = True`:
```python
if not self._verify_engine_running():
    self._connected = False
    return False
```

Apply the same refactor inside `connect()` for parity.

**Verification:** unit-testing this requires mocking `obd.OBD`, which we already do partially in `tests/test_live_obd_source.py`. Add one fake that returns RPM=0 on the second connect and assert `_try_reconnect()` returns False.

---

### T3.5  Pass `sample_hz` from `live_baseline_capture` into `extract_features`

**File:** `scripts/live_baseline_capture.py` line 121

**Why this matters:** the script polls at a requested 1 Hz but the Skoda ELM327 actually delivers ~0.3 Hz. Right now feature extraction defaults to `sample_hz=1.0`, so the captured baseline normalizer is fit on features computed with the *wrong* time axis. The Skoda will then run inference against a baseline that doesn't match its own rate.

**Fix:** read the actual measured poll rate from the source and pass it through.

```python
poll_hz = src.measured_poll_hz or 1.0
...
for window, _ in sliding_windows(df[pid_cols], label="healthy"):
    feats = extract_features(window, sample_hz=poll_hz)
    feature_rows.append(feats)
```

(After T3.1 is in, `LiveObdSource` will emit 1-Hz resampled rows during baseline capture as well — at that point the right value to pass is `1.0`. Document this dependency in the script's docstring.)

**Verification:** dry-run the script against a stubbed CsvStreamer at 0.5 Hz; assert `feature_rows[0]["COOLANT_WARMUP_RATE"]` is in the expected °C/min range.

---

## Tier 4 — Workshop-facing DTC correctness

These are the codes a mechanic will read off our UI and compare to their scan tool. Wrong codes erode trust during the live demo.

### T4.1  Fix `fuel_system` DTC: P0172 → "P0171 / P0087" (paired)

**File:** `src/diagnostics/dtc_map.py`

**Problem:** entry has `code: "P0172"` (System Too Rich) but `name`, `short`, and `description` all describe a chronic-lean condition with positive LTFT. This is internally inconsistent.

**Pass-3 final decision (D1, locked 2026-05-26):** show **both** codes
joined by " / " — the symptom (P0171) and the typical root cause (P0087).
This matches what a real workshop scan tool prints when this fault develops
on a real car. The same workshop tech who would pull P0171 alone on a
vacuum leak (`air_system`) will pull P0171 + P0087 together on a
fuel-pressure problem.

Update the entry:
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

**Verification:** open the dashboard, replay any session containing a `fuel_system` window, and confirm the right column reads `DTC: P0171 / P0087  Lean+Fuel`.

---

### T4.2  Fix `throttle_position_sensor` DTC: P0122 → P2135

**File:** `src/diagnostics/dtc_map.py`

**Problem:** `_inject_tps` drifts THROTTLE **upward** (see `src/injection/fault_injector.py:337` "drifts upward relative to the actual pedal position"). The corresponding DTC is **P0123** (TPS Circuit High Input) — not P0122 (Low Input).

For an E-gas system like the Roomster, the more diagnostic-realistic code is **P2135** (TPS A/B Correlation), because the fault is exactly a divergence between the two redundant sensors. Recommendation:

```python
"throttle_position_sensor": {
    "code": "P2135",
    "name": "Throttle Position Sensor A/B Correlation",
    "short": "TPS Corr",
    "description": (
        "Reported throttle angle diverges from pedal command. On E-gas "
        "(drive-by-wire) systems this triggers a correlation fault rather "
        "than a single-sensor circuit code. Most common causes: worn TPS "
        "resistive track, harness contamination, contaminated throttle body."
    ),
},
```

**Verification:** same as T4.1.

---

### T4.3  Update fuel_system recommendation text to match the new dual DTC

**File:** `src/diagnostics/recommendations.py` line 19

Pass-3 decision D1 paired P0171 + P0087, so the recommendation text reflects
both codes explicitly:

Change:
```python
"Pull DTCs - confirm P0171 or P0172. If misfire codes present, suspect injector clog instead.",
```
to:
```python
"Pull DTCs - confirm P0171 (lean, Bank 1) and/or P0087 (low rail pressure). If misfire codes accompany either, suspect injector clog rather than pump/filter.",
```

**Verification:** sidebar opens the fault on a fuel_system window; check the third bullet matches.

---

## Tier 5 — ML & evaluation polish

### T5.1  Exclude regime one-hots from `StandardScaler`

**File:** `src/features/normalizer.py`

**Why:** the 5 `REGIME__*` features are 0/1 indicators. Z-scoring them produces values like {-0.4, +2.5} that the model treats as continuous magnitudes — a window labelled `REGIME__CRUISE` on the Skoda (low cruise share) will read more "anomalous" than the same regime on the Etios (high cruise share), even when both are healthy. This is a soft transfer-learning failure.

**Fix:** pass through the regime columns unchanged.

```python
from src.features.regime import regime_feature_names

_REGIME_COLS = set(regime_feature_names())
_SCALED_COLS = [c for c in _FEAT_COLS if c not in _REGIME_COLS]

# In fit():
healthy_X = df[df["label"] == healthy_label][_SCALED_COLS]
self._scaler.fit(healthy_X.to_numpy(dtype=float))

# In transform():
X_scaled = self._scaler.transform(df[_SCALED_COLS].to_numpy(dtype=float))
z_cols = {f"{c}__z": X_scaled[:, i] for i, c in enumerate(_SCALED_COLS)}
# Regime one-hots pass through unmodified as their own z columns:
for c in regime_feature_names():
    z_cols[f"{c}__z"] = df[c].to_numpy(dtype=float)
return pd.concat([df.reset_index(drop=True), pd.DataFrame(z_cols)], axis=1)
```

**Compatibility:** existing saved normalizers were fit on 82 features. After this change they fit on 77. The `feature_order` check in `BaselineNormalizer.load()` will refuse to load the old file — that is the desired behaviour. Retrain the production normalizer with `python -m scripts.rebuild_all` after merging.

**Verification:** `tests/test_normalizer.py` — assert `transform(df)` returns regime z columns whose values are exactly 0.0 or 1.0.

---

### T5.2  Tighten degraded-PID warning threshold

**File:** `src/dashboard/app.py` line 1083

Replace `>= 3` with `>= 1`. One missing PID already degrades the classifier — silence on 2 missing PIDs is the wrong default for a graduation demo.

```python
if engine.degraded_pid_count >= 1:
    st.warning(
        f"WARNING: {engine.degraded_pid_count} PID(s) unsupported by this ECU - "
        f"classifier z-scores fall back to the healthy mean. Confidence is degraded."
    )
```

---

### T5.3  Extend `sanity.py` cross-PID rules

**File:** `src/dashboard/sanity.py`

Add three rules to `check_row()`:

1. **MAP <= barometric (NA engine):**
   ```python
   map_val = row.get("INTAKE_MANIFOLD_PRESSURE")
   baro = row.get("ABSOLUTE_BAROMETRIC_PRESSURE", 105.0)  # NA Skoda fallback
   if map_val is not None and not (isinstance(map_val, float) and math.isnan(map_val)):
       if map_val > baro + 5:  # +5 kPa sensor noise tolerance
           violations.append(f"MAP={map_val:.1f} > BARO={baro:.1f} (NA engine)")
   ```

2. **Throttle vs commanded actuator divergence (gross fault, not the slow TPS drift):**
   ```python
   thr = row.get("THROTTLE")
   cmd = row.get("COMMANDED_THROTTLE_ACTUATOR")
   if thr is not None and cmd is not None:
       both_ok = not any(isinstance(v, float) and math.isnan(v) for v in (thr, cmd))
       if both_ok and abs(thr - cmd) > 30:  # > 30 % is hardware fault, not wear
           violations.append(f"THROTTLE={thr:.0f} but COMMANDED={cmd:.0f} (delta > 30)")
   ```

3. **Coolant rate-of-change (cross-row):** this needs state, so add a small `class RowSanityChecker` that holds the previous coolant value + timestamp. Skip for now if check_row stays stateless; otherwise:
   ```python
   class RowSanityChecker:
       def __init__(self):
           self._prev_coolant = None
           self._prev_t = None
       def check(self, row, now):
           verdict = check_row(row)
           c = row.get("COOLANT_TEMPERATURE")
           if c is not None and self._prev_coolant is not None and self._prev_t is not None:
               dt = max(now - self._prev_t, 1e-3)
               if abs(c - self._prev_coolant) / dt > 1.0:  # > 1 deg C/s = sensor glitch
                   verdict.violations.append(
                       f"Coolant jumped {c - self._prev_coolant:+.1f}C in {dt:.1f}s"
                   )
                   verdict.ok = False
           self._prev_coolant, self._prev_t = c, now
           return verdict
   ```

Wire `RowSanityChecker` into `InferenceEngine` if you adopt the stateful variant. Otherwise ship just rules 1 + 2.

**Verification:** extend `tests/test_sanity.py` with one case per rule.

---

### T5.4  Remove dead `norm` assignment in `loso_cv.py`

**File:** `scripts/loso_cv.py` line 38

Delete the line:
```python
norm = BaselineNormalizer().fit(train_df)
```

The next line already gets the trained normalizer from `train_clf`. Also remove the unused import on line 26 (`from src.features.normalizer import BaselineNormalizer`).

**Verification:** `python -m scripts.loso_cv` still runs and produces the same mean F1 (0.958).

---

### T5.5  Make CHARTER match what the code actually does

**File:** `docs/CHARTER.md` line 163

Change "Session-level **5-fold** cross-validation, with fold assignment fixed and committed to the repository before any results are reported." to:

> Cross-validation is reported two ways: (a) a fixed session-level holdout `{drive1, live12}` used by `scripts/rebuild_all.py` — this produces the headline model and the deployed artefacts; (b) leave-one-session-out (LOSO) across all 9 usable sessions, executed by `scripts/loso_cv.py` — this produces mean ± std macro-F1 as the honest generalisation estimate. The thesis must report both.

**Verification:** `git diff docs/CHARTER.md` reads sensibly to a thesis reviewer.

---

### T5.6  Idle-weight `_inject_air_system` (deferred / optional)

**File:** `src/injection/fault_injector.py`

This is a *design* improvement, not a bug. Vacuum leaks are a larger fraction of total airflow at idle than at WOT, so the MAP offset and STFT correction should both be ramped down at high engine load.

**Approach (sketch — only do this if T3.1–T3.5 finish with time left):**
```python
load = df["ENGINE_LOAD"].to_numpy(dtype=float)
idle_weight = np.clip(1.0 - load / 60.0, 0.3, 1.0)  # full effect at idle, 30 % at WOT
map_delta = ramp * magnitude_kpa * idle_weight + noise(0.3)
```

The new injection still has to obey existing clamps. Retrain after the change and re-run `scripts/loso_cv.py`.

**Verification:** `tests/test_injection.py` — assert injected MAP delta is larger at low-load rows than at high-load rows.

---

### T5.7  Add commanded-vs-actual throttle residual feature

**Files:** `src/features/extractor.py`, `src/features/severity.py`

This was listed earlier as T2.1 in the original Roomster plan but never implemented.

1. In `extract_features`, after the existing TPS ratio:
   ```python
   commanded = window["COMMANDED_THROTTLE_ACTUATOR"].to_numpy(dtype=float)
   actual    = window["THROTTLE"].to_numpy(dtype=float)
   open_mask = commanded > 5.0
   if open_mask.any():
       features["THROTTLE_CMD_ACTUAL_DELTA"] = float(
           np.mean(actual[open_mask] - commanded[open_mask])
       )
   else:
       features["THROTTLE_CMD_ACTUAL_DELTA"] = 0.0
   ```
   Bump the feature count from 82 to 83 in the docstring and `tests/test_features.py::test_feature_names_length`.

2. In `compute_severity` for `throttle_position_sensor`, add a second term and average:
   ```python
   ratio_term = (delta - _TPS_DEADBAND) / _TPS_SCALE
   cmd_delta  = features.get("THROTTLE_CMD_ACTUAL_DELTA", 0.0)
   cmd_term   = np.clip(cmd_delta / 10.0, 0.0, 1.0)   # > 10 % gap → full severity
   return float(np.clip(0.5 * ratio_term + 0.5 * cmd_term, 0.0, 1.0))
   ```

**Compatibility:** existing saved model artefacts will fail the feature-count check. Rebuild after the change.

**Verification:** `tests/test_features.py` — assert the new feature is present and reads ≈ 0 on synthetic balanced data.

---

## Execution order

1. **T3.5** (smallest live fix — 1 line)
2. **T3.3** (smallest engine fix — guard `set_sample_hz`)
3. **T3.4** (refactor RPM liveness into shared helper)
4. **T3.2** (cold-start timer becomes time-based — drop-in change, but touches buffer indexing)
5. **T3.1** (1-Hz resampler — largest single change, do it last in Tier 3 because T3.2's correctness depends on whether the upstream is resampled or raw)
6. **T4.1 + T4.2 + T4.3** (DTC text — review together, single commit)
7. **T5.2** (degraded-PID threshold ≥ 1 — one line)
8. **T5.4** (dead-code removal in loso_cv.py)
9. **T5.5** (CHARTER text)
10. **T5.3** (sanity rules)
11. **T5.1** (regime-aware normalizer — needs a rebuild)
12. **T5.7** (commanded-vs-actual feature — needs a rebuild)
13. **T5.6** (idle-weighted air injection — needs a rebuild; only if time permits)

Bundle 1–5 into one PR ("Tier 3: live deployment correctness"), 6 into a second ("Tier 4: workshop DTC fixes"), 7–10 into a third ("Tier 5: sanity & methodology polish"), and 11–13 into a final retrain PR ("Tier 5: feature pipeline upgrades + rebuild").

---

## Verification gate before the Roomster demo

After Tier 3 lands, do this dry-run on the bench:
1. Plug in the ELM327 with the laptop **and** Skoda ignition in ACC (no engine running).
2. Hit Connect in the dashboard. Confirm: refuses to start, logs `ENGINE_RPM=… — ignition in ACC`.
3. Start the engine. Hit Connect again. Confirm: connects, banner shows `measured_poll_hz` between 0.2–0.6 Hz, no `0.05 Hz` artefact in the first window.
4. Idle for 3 minutes. Confirm: `COOLANT_WARMUP_RATE` is in the 1–3 °C/min range (a real warming engine), not 10+ (the old bug).
5. After warmup, drive 1 km. Confirm: classifier label remains `healthy` with confidence ≥ 0.7 for the entire run.

If step 5 is unstable, **do not** demo. Investigate which feature is off-distribution — most likely the standard-deviation features over a window that's still effectively too long. Consider lowering `WINDOW_LENGTH_S` to 30 seconds and retraining as a fallback.
