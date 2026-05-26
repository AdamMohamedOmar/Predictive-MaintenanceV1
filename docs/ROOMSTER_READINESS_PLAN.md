# Roomster Readiness Plan — Tier 1 + Tier 2

**Goal:** make the dashboard demo-ready on the 2007 Skoda Roomster 1.6 MPI (BTS engine, 16V petrol, port-injected, NA) before **15 June 2026**.

**Source review:** `report_audit.md` (the external critique). Items below address every report claim that is (a) **verified against the code** and (b) **tractable within the deadline**. Items deferred to thesis "Limitations" are listed at the bottom — they are not in this branch.

**Order:** complete Tier 1 in numeric order; only start Tier 2 once Tier 1 verifies. Each task ends with an explicit verify step.

---

## Tier 1 — Demo-blocking (must complete before plugging into the Roomster)

### T1.1 — Engine identification ✅ DONE

- Vehicle: 2007 Skoda Roomster
- Engine: **VW Group EA111 1.6 MPI (BTS code), 1595 cc, 16V DOHC, 77 kW / 105 hp, port injection, NA**
- Code-relevant implications:
  - Petrol → `TIMING_ADVANCE`, fuel trims, closed-loop O₂ semantics all valid
  - Port-injected → STFT/LTFT match Etios training distribution
  - NA → MAP ≤ barometric pressure holds
  - Spark-ignition → `TIMING_VS_TEMP` feature is physically meaningful

**No code action.** Document the engine code in the Limitations section of the thesis ("validated on BTS 1.6 MPI; other Roomster variants — 1.2 HTP, 1.4 MPI, 1.9 TDI — would require separate validation").

---

### T1.2 — Fix 1 Hz hard-coding in feature extraction (CRITICAL CODE FIX)

**Why:** ELM327 polling 14 PIDs over Bluetooth typically delivers 0.2–0.5 Hz on a 2007 ECU. Two features assume the row index equals seconds:

1. **`COOLANT_WARMUP_RATE`** (`extractor.py:118-120`) — uses `np.arange(n)` as time axis, multiplies slope by 60 to get °C/min. At 0.3 Hz, every reported rate is **3.3× too large**. This silently breaks the `cold_start_checker` thermostat rule (≤ 0.3 °C/min for 480 s) and the coolant-severity gating.
2. **`FUEL_LOOP_ACTIVE`** (`extractor.py:123-124`) — fires when ≥10 rows show `|STFT| > 0.5%`. At 1 Hz that's 10 seconds of trim activity. At 0.3 Hz, 10 rows = 33 seconds — much stricter; the closed-loop gate may **never fire** on a healthy Roomster, suppressing fuel/coolant severity entirely.

**Scope:** `src/features/extractor.py` (1 signature change + 2 feature formulas), `src/dashboard/inference.py` (plumb live `measured_poll_hz` through). Backwards compatible — default `sample_hz=1.0` so all existing tests still pass.

**File: `src/features/extractor.py`**

Change signature at line 54:
```python
def extract_features(window: pd.DataFrame, sample_hz: float = 1.0) -> dict[str, float]:
```

Update docstring (line ~55–67) to add:
```
sample_hz : float, default 1.0
    Actual sampling rate of `window` in Hz.  Defaults to 1.0 to match
    the carOBD training dataset.  Pass live ELM327 poll rate at inference
    so time-based features (warmup rate, fuel-loop active threshold)
    scale correctly with adapter throughput.
```

Replace `COOLANT_WARMUP_RATE` block (lines 115–120):

**Before:**
```python
    # Trajectory features — rate-of-change signals for cold-start diagnostics
    coolant = window["COOLANT_TEMPERATURE"].to_numpy(dtype=float)
    n = len(coolant)
    # Slope in °C/min via linear regression over the 60-s window
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, coolant, 1)[0]) * 60.0  # convert /s → /min
    features["COOLANT_WARMUP_RATE"] = slope
```

**After:**
```python
    # Trajectory features — rate-of-change signals for cold-start diagnostics
    coolant = window["COOLANT_TEMPERATURE"].to_numpy(dtype=float)
    n = len(coolant)
    # Slope in °C/min — uses *actual* time axis derived from sample_hz so the
    # rate is correct when the live ELM327 polls below 1 Hz (typical on a 2007
    # ECU over Bluetooth).  At sample_hz=1.0 this is identical to the original
    # np.arange(n) formula.
    t_sec = np.arange(n, dtype=float) / max(sample_hz, 1e-3)
    slope_per_sec = float(np.polyfit(t_sec, coolant, 1)[0])
    features["COOLANT_WARMUP_RATE"] = slope_per_sec * 60.0  # °C/min
```

Replace `FUEL_LOOP_ACTIVE` block (lines 122–124):

**Before:**
```python
    stft = window["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float)
    active_rows = int(np.sum(np.abs(stft) > 0.5))
    features["FUEL_LOOP_ACTIVE"] = 1.0 if active_rows >= 10 else 0.0
```

**After:**
```python
    # FUEL_LOOP_ACTIVE fires when the ECU has shown ≥10 *seconds* of measurable
    # STFT activity.  Translate that to a row count using sample_hz so a slow
    # ELM327 (0.3 Hz → 3 rows of activity = 10 s) still detects closed-loop.
    # Floor at 3 rows to keep the rule meaningful against single-sample noise.
    stft = window["SHORT_TERM_FUEL_TRIM_BANK_1"].to_numpy(dtype=float)
    active_rows = int(np.sum(np.abs(stft) > 0.5))
    threshold_rows = max(3, int(round(10.0 * sample_hz)))
    features["FUEL_LOOP_ACTIVE"] = 1.0 if active_rows >= threshold_rows else 0.0
```

**File: `src/dashboard/inference.py`**

`_run_window` currently calls `extract_features(window_df)`. Make it pass the live sample rate when known:

In `__init__`, add a default sample-rate field (after `self._elapsed_s = 0`):
```python
        # Live sample rate (Hz) — overridden via set_sample_hz() from the
        # dashboard once the live ELM327 source reports measured_poll_hz.
        # CSV replay leaves this at 1.0 (the training dataset's rate).
        self._sample_hz: float = 1.0
```

Add a setter near `reset()`:
```python
    def set_sample_hz(self, hz: float) -> None:
        """Update the sample rate used by time-based features.

        Called by the dashboard each rerun in live mode so COOLANT_WARMUP_RATE
        and FUEL_LOOP_ACTIVE stay calibrated against actual ELM327 throughput.
        """
        # Clamp to a sane range — outside [0.05, 5.0] is almost certainly a
        # measurement glitch, not a real poll rate.
        self._sample_hz = float(np.clip(hz, 0.05, 5.0))
```

(Add `import numpy as np` at the top if not already present.)

Change line 333 from:
```python
        feats = extract_features(window_df)
```
to:
```python
        feats = extract_features(window_df, sample_hz=self._sample_hz)
```

**File: `src/dashboard/app.py`**

In `main()`, just before `engine.update(row)` (around line 1093), wire in the live rate:
```python
        # In live mode, keep the inference engine in sync with the actual
        # adapter throughput so time-based features stay calibrated.
        if st.session_state.source_type == "live" and source is not None:
            engine.set_sample_hz(getattr(source, "measured_poll_hz", 1.0))
```

**New test: `tests/test_features.py`**

Add a regression test that confirms the rate scaling holds:
```python
def test_warmup_rate_scales_with_sample_hz():
    """At half sample rate, the same coolant rise should report the same °C/min."""
    import pandas as pd
    from src.features.extractor import extract_features

    base = _make_window()  # existing helper that builds a 60-row healthy window
    base = base.copy()
    # Inject a clear linear warmup: 30°C → 60°C across the window
    base["COOLANT_TEMPERATURE"] = np.linspace(30.0, 60.0, len(base))

    feats_1hz = extract_features(base, sample_hz=1.0)
    feats_05hz = extract_features(base, sample_hz=0.5)

    # At 1 Hz, 60 rows = 60 s. At 0.5 Hz, 60 rows = 120 s. Same rise in twice
    # the time = half the rate.
    assert feats_05hz["COOLANT_WARMUP_RATE"] == pytest.approx(
        feats_1hz["COOLANT_WARMUP_RATE"] / 2.0, rel=0.01
    )


def test_fuel_loop_threshold_scales_with_sample_hz():
    """Closed-loop should fire on equivalent *time* of activity at any rate."""
    import pandas as pd
    from src.features.extractor import extract_features

    win = _make_window().copy()
    # 5 rows of STFT activity = 5 s at 1 Hz (below 10 s threshold → off),
    # but the same 5 rows at 0.5 Hz cover 10 s of real time → should fire.
    win["SHORT_TERM_FUEL_TRIM_BANK_1"] = 0.0
    win.loc[:4, "SHORT_TERM_FUEL_TRIM_BANK_1"] = 2.0

    assert extract_features(win, sample_hz=1.0)["FUEL_LOOP_ACTIVE"] == 0.0
    assert extract_features(win, sample_hz=0.5)["FUEL_LOOP_ACTIVE"] == 1.0
```

(Place `_make_window` references next to the existing helpers in the same test file; if no such helper exists, build a 60-row DataFrame inline.)

**Verify:**
```
.venv/Scripts/pytest.exe tests/test_features.py -q
.venv/Scripts/pytest.exe tests/ -q   # full suite (268+ pass)
```

Expected: all existing tests still pass (sample_hz defaults to 1.0); two new tests pass.

---

### T1.3 — Capture Skoda baseline normalizer (procedure, no code change)

**Why:** XGBoost reads z-scores; z-scores are healthy-relative; the saved Etios scaler is wrong for the Roomster. The classifier weights are vehicle-agnostic, but the normalization reference must be Roomster-specific.

**Tool:** `scripts/live_baseline_capture.py` (already exists and has guards).

**Procedure (do this on the Roomster, after T1.2 is merged):**

1. Plug ELM327 in, ignition ON, wait for engine to be **fully warm** (coolant ≥ 87 °C — BTS thermostat opening temp).
2. From repo root:
   ```
   python -m scripts.live_baseline_capture \
     --port COM3 \
     --duration-min 7 \
     --out models/skoda_roomster_normalizer.pkl \
     --vehicle "Skoda Roomster 2007 1.6 MPI BTS"
   ```
3. **Drive normally for the full 7 minutes** — mix of urban + at least one stretch at ≥ 60 km/h. Do not stay at idle.
4. Script will refuse to save if any guard fails (script docstring lists them). If it refuses, drive longer / hotter and re-run.

**Expected baseline values for BTS engine** (sanity check the output JSON):
- `LONG_TERM_FUEL_TRIM_BANK_1__mean`: −2 to +5 % (lean-of-stoich calibration is normal)
- `SHORT_TERM_FUEL_TRIM_BANK_1__mean`: −3 to +3 %
- `THROTTLE_TO_PEDAL_RATIO`: 0.8–1.2 (electronic throttle, generally linear in cruise)
- `COOLANT_TEMPERATURE__mean`: 87–95 °C
- `ENGINE_RPM__mean`: 1500–2500 (depends on drive mix)

If any value is wildly off these ranges, the baseline drive wasn't representative — re-capture.

**Verify (after re-running dashboard with new normalizer):**
- Sidebar **Normalizer** dropdown now lists `skoda_roomster_normalizer.pkl`
- Selecting it and pressing Play on `drive1.csv` (Etios commute) should produce **lower-confidence healthy predictions** than the built-in normalizer (expected — the Etios data now z-scores differently against a Skoda baseline). Severity gauges should stay near 0%.

**No code change required** — the dashboard already accepts normalizer overrides.

---

### T1.4 — Run LOSO-CV for honest F1

**Why:** Charter §7.5 commits to session-level k-fold. Production training uses a fixed `{drive1, live12}` holdout — a single point estimate on a regime-biased test set. LOSO-CV gives a mean ± std across all 9 sessions, which is the metric the thesis must cite.

**Tool:** `scripts/loso_cv.py` (already exists from earlier this session).

**Procedure:**
```
python -m scripts.loso_cv
```
Wait ~5–10 min (trains 9 models).

**Verify:**
- `results/loso_cv_results.json` exists
- `mean_f1` should be in the **0.85 – 0.95** range
- `std_f1` should be **< 0.05** (variance across sessions)
- Any individual session F1 below 0.75 — note it; that session has signature characteristics worth discussing in the thesis (probably highway-only drive1)

**Thesis impact:** quote the headline as `mean_f1 ± std_f1` from LOSO, **not** the fixed-holdout 0.96. Drop any "macro-F1 = 0.96" claim in favor of "macro-F1 = X.XX ± Y.YY (9-fold LOSO)".

---

## Tier 2 — Thesis quality (do this week)

### T2.1 — Add commanded-vs-actual throttle divergence to TPS severity

**Why (report 2.F):** workshop diagnostics for TPS drift compare *commanded* throttle (what the ECU asked for) vs *actual* throttle (what the TPS reports). The Roomster has both PIDs (`COMMANDED_THROTTLE_ACTUATOR`, `THROTTLE`). Currently `compute_severity("throttle_position_sensor", …)` uses only the pedal-vs-actual ratio, which can be confounded by ECU "drive-by-wire" nonlinearity on VAG vehicles.

**File:** `src/features/severity.py`

Locate the `throttle_position_sensor` branch (~line 114). Augment the severity formula to take the **max** of the two diagnostic paths — pedal-ratio (existing) and commanded-vs-actual divergence (new). Use `max`, not weighted average, so either signature alone is enough to flag the fault.

Add a constant at the top (with the others around line 41):
```python
_TPS_COMMANDED_SCALE = 8.0   # %-pts of commanded-vs-actual divergence at full fault
                              # BTS healthy divergence stays < 2 %-pts
                              # 8 % matches the throttle-fault injection magnitude
```

Replace the body of the `throttle_position_sensor` branch with:
```python
    if fault_type == "throttle_position_sensor":
        # Path A — pedal vs actual throttle (existing logic) ──────────────
        if features.get("THROTTLE__mean", 0.0) < _TPS_MIN_THROTTLE_MEAN:
            sev_pedal = 0.0
        else:
            ratio = features["THROTTLE_TO_PEDAL_RATIO"]
            ratio_base = baselines["THROTTLE_TO_PEDAL_RATIO"]
            delta = ratio - ratio_base
            if delta < _TPS_DEADBAND:
                sev_pedal = 0.0
            else:
                sev_pedal = float(np.clip((delta - _TPS_DEADBAND) / _TPS_SCALE, 0.0, 1.0))

        # Path B — commanded-vs-actual throttle divergence ─────────────────
        # Healthy ECU: throttle plate tracks commanded position within ~2 %-pts.
        # Worn TPS reports back to ECU different angle than ECU commanded → divergence.
        commanded = features.get("COMMANDED_THROTTLE_ACTUATOR__mean", float("nan"))
        actual    = features.get("THROTTLE__mean", float("nan"))
        if not (math.isnan(commanded) or math.isnan(actual)):
            divergence = abs(actual - commanded)
            sev_cmd = float(np.clip(divergence / _TPS_COMMANDED_SCALE, 0.0, 1.0))
        else:
            sev_cmd = 0.0

        return float(max(sev_pedal, sev_cmd))
```

Add `import math` at the top of the file if not already present.

**New test:** `tests/test_severity.py` (or wherever severity tests live) — add a case where pedal-ratio is healthy but commanded-vs-actual diverges by 10 %, asserting severity > 0.5.

**Verify:**
```
.venv/Scripts/pytest.exe tests/ -q
```

---

### T2.2 — Tighten degraded-sensor warning (≥1 PID) + add "DEGRADED" banner state

**Why (report 1.H):** Currently when an ECU doesn't expose a PID, `inference.py` fills feature values from the healthy baseline (so z-scores become ~0). The warning only fires at ≥3 NaN-filled PIDs. A workshop demo where 1–2 PIDs are missing on the Roomster (e.g. `COMMANDED_THROTTLE_ACTUATOR` on some BTS variants) currently shows confident "healthy" with no indication anything is missing.

**File: `src/dashboard/app.py`**

Replace the existing warning block (around line 1066):

**Before:**
```python
    if engine.degraded_pid_count >= 3:
        st.warning(
            f"WARNING: {engine.degraded_pid_count} PIDs unsupported by this ECU — "
            f"classifier confidence is degraded. Run with a vehicle that supports all 14 PIDs."
        )
```

**After:**
```python
    if engine.degraded_pid_count >= 1:
        # Threshold at 1 PID — even a single NaN-imputed PID can mask a real
        # fault by anchoring its features to the healthy training mean.
        # Workshop reality: missing PIDs need to be loudly visible, not silent.
        st.warning(
            f"⚠ {engine.degraded_pid_count} PID(s) unsupported by this ECU — "
            f"their features are imputed from the healthy training baseline.  "
            f"Classifier output is **screening only** for these channels; "
            f"check the SHAP panel before trusting any prediction."
        )
```

**Optional (do only if time permits):** Add a `DEGRADED` banner mode to `_render_status_banner` that fires when `engine.degraded_pid_count >= 3` and the classifier is reporting `healthy`. The current banner shows "ALL SYSTEMS NOMINAL" which is misleading on a partially-supported ECU. Two extra branches in the if/elif chain are enough — copy the WARMING_UP styling, change the title to "DEGRADED SENSOR INPUT" and accent to `ACCENT_WARN`.

**Verify:**
- Load `drive1.csv` in dashboard → no warning visible (all 14 PIDs present)
- Manually rename a column in a test CSV to force NaN → warning appears immediately at count = 1

---

### T2.3 — Extend sanity check with MAP-vs-baro and °C/s rate limit

**Why (report 2.A):** Current `check_row` enforces per-PID min/max + one cross-PID rule (RPM<200 while speed>5). It misses:
- **MAP > 105 kPa on a NA petrol engine** (atmospheric is ~101 kPa at sea level; BTS is NA, MAP physically cannot exceed baro). Bad ELM327 frames can return MAP = 250 kPa and slip through the existing per-PID upper bound of 250.
- **Coolant changing by > 2 °C between consecutive rows** — physically impossible (thermal inertia). This catches a stuck-sensor jumping from 40 °C → 90 °C in one frame.

**File: `src/dashboard/sanity.py`**

1. **Tighten MAP upper bound** for NA engines. The existing bound (250 kPa) was permissive to allow turbo. We're targeting a NA petrol; cap at 110 kPa (atmospheric + small margin):
```python
    "INTAKE_MANIFOLD_PRESSURE": (10, 110),   # was (0, 250); NA petrol cannot exceed baro
```
*If you later want to support turbo vehicles, gate this with a `naturally_aspirated=True` constructor arg.*

2. **Add coolant-rate cross-row rule.** `check_row(row)` currently takes only the current row. Extend signature to optionally accept previous row:
```python
def check_row(
    row: dict[str, float],
    previous: dict[str, float] | None = None,
) -> QualityVerdict:
    """...
    previous : dict[str, float] | None
        The most recent row that passed sanity.  Used for cross-row rate limits
        (e.g. coolant Δ > 2 °C/s is physically impossible).  Pass None on the
        first call after reset.
    """
    ...
    # ... existing per-PID checks ...

    # Cross-row physics: coolant cannot change > 2 °C in one second.
    # (Conservative — true thermal limit is ~1 °C/s, we leave headroom for
    # the slowest plausible ELM327 poll rate.)
    if previous is not None:
        cur_t  = row.get("COOLANT_TEMPERATURE", float("nan"))
        prev_t = previous.get("COOLANT_TEMPERATURE", float("nan"))
        if not (math.isnan(cur_t) or math.isnan(prev_t)):
            if abs(cur_t - prev_t) > 2.0:
                violations.append(
                    f"COOLANT_TEMPERATURE jumped {prev_t:.1f}→{cur_t:.1f} (>2 °C/row)"
                )
```

3. **Plumb `previous` through `inference.py`.** Store the last *passing* row on `InferenceEngine`:
```python
self._last_good_row: dict | None = None    # in __init__, after _last_state
...
# In update(), before/after the verdict.ok branch:
verdict = check_row(row, previous=self._last_good_row)
if verdict.ok:
    self._last_good_row = row
```

**New tests** in `tests/test_sanity.py`:
- `test_map_over_baro_rejected` — row with `INTAKE_MANIFOLD_PRESSURE=150` fails
- `test_coolant_jump_rejected` — passing previous row at 40 °C and current at 90 °C fails
- `test_coolant_smooth_passes` — 40 → 41 °C between consecutive rows passes

**Verify:**
```
.venv/Scripts/pytest.exe tests/test_sanity.py -v
```

---

### T2.4 — Filter sidebar dropdown to usable session files only

**Why (report context + your drive13 mistake earlier):** `_DATA_DIR.glob("*.csv")` shows ALL files including the 120 with firmware-encoding bugs (drive13, live1, live2, etc.). Users can pick a broken file and see the dashboard freeze at "SENSOR DATA INVALID" with no indication that the file itself is bad.

**File: `src/dashboard/app.py`**

Replace the CSV file list near line 217:

**Before:**
```python
        csv_files = sorted(_DATA_DIR.glob("*.csv"))
        if not csv_files:
            st.sidebar.warning("No CSV files found. Run scripts/rebuild_all.py first.")
            return normalizer_path, 1.0
```

**After:**
```python
        from src.data_loading import USABLE_CAROBD_FILES

        # Show only the 9 audited-clean files.  The other 120 in carOBD have
        # firmware-encoding bugs in TIMING_ADVANCE / STFT that fail sanity
        # and freeze the dashboard with no useful signal.
        csv_files = sorted(
            p for p in _DATA_DIR.glob("*.csv")
            if p.name in USABLE_CAROBD_FILES
        )
        if not csv_files:
            st.sidebar.warning(
                "No usable CSV files found in carOBD/.  Expected one of: "
                f"{', '.join(sorted(USABLE_CAROBD_FILES))}"
            )
            return normalizer_path, 1.0
```

**Verify:**
- Restart Streamlit, open sidebar — only 9 files visible (`drive1`, `live5`–`live12`)
- drive13, live1–live4, etc. are gone

---

### T2.5 — Stop citing RF metrics in thesis/README (TEXT-ONLY)

**Why (report 1.F):** The Random Forest baseline trained on absolute (non-z-scored) features hit 1.0 F1 because it memorized Etios-specific thresholds. The README and any presentation slides should not quote that number as evidence of separability — it overstates and contradicts the cross-vehicle story.

**Action (text edits only, no code):**

1. **README.md** — locate any line mentioning RF F1 = 1.00 or similar. Replace with:
   > "An early Random Forest baseline (no normalization, absolute features) scored macro-F1 = 1.00 on the held-out test split, indicating it had learned vehicle-specific thresholds rather than generalizable fault signatures. The production pipeline uses XGBoost on z-scored features (`BaselineNormalizer`) and reports macro-F1 = X.XX ± Y.YY (9-fold LOSO; see `results/loso_cv_results.json`)."

2. **Thesis methodology section** — same substitution. Cite LOSO numbers from T1.4 as the headline.

3. **`docs/CHARTER.md`** if it quotes the RF number — leave the charter as-is (historical document), but add a "Realized Results" appendix that points to LOSO results.

**Verify:** grep README + docs for "Random Forest" / "RF" / "1.0 F1" — ensure no remaining standalone quotes; every mention is contextualized as the "memorization baseline".

---

### T2.6 — Document O₂ → cold_start substitution in methodology (TEXT-ONLY)

**Why (report 1.I, 3.E):** Charter §6 commits to 5 faults including O₂. Code has 6 classes with `cold_start` replacing O₂. `CLAUDE.md` documents the substitution and the reason (`FUEL_AIR_COMMANDED_EQUIV_RATIO` is constant 0 on the Etios ECU, so O₂ ground truth cannot be injected). The thesis must state this explicitly — failure to do so is the kind of thing a defense committee will pick on.

**Action (text-only):**

Add a "Taxonomy Adjustment" subsection to the thesis methodology, ~½ page, covering:
1. Why O₂ was dropped (dead PID on Etios ECU; documented in `DATA_NOTES.md`)
2. Why `cold_start` was added (real diagnostic regime on this ECU; rule-detectable; useful gate for severity formulas)
3. What this means for thesis claims (no validated detection of O₂ faults — `air_system` and `fuel_system` will partially flag the same lean conditions an O₂ failure would produce)
4. Map to OBD-II DTCs (`P0131`/`P0132` O₂ codes not directly mapped; `P0171`/`P0172` lean/rich codes covered via `air_system`/`fuel_system`)

**Verify:** ½ page of thesis text reviewed; cross-reference to `CLAUDE.md` fault taxonomy and `src/diagnostics/dtc_map.py`.

---

## Tier 3 — Limitations chapter content (text-only, do during writing)

Things to **explicitly state** in the thesis's Limitations & Future Work section. Do not try to fix these in code before defense.

1. **Synthetic-only fault labels.** No real-world Roomster fault data; performance numbers reflect injected signature detectability, not in-field calibration.
2. **Commute-biased training set.** 8/9 usable carOBD files are urban commute; 1 is highway. Cross-regime generalization untested.
3. **Within-session window correlation.** 60s windows at 10s stride share 50/60 rows; effective sample size is below the row count. Mitigated by session-level splits and LOSO-CV; not eliminated.
4. **Hard-coded vehicle constants.** Coolant operating temp (90 °C), warmup target (75 °C), IAC threshold (1100 RPM) are EA111-petrol generic. A V2 should load a YAML vehicle profile per engine code.
5. **No plausibility-fusion engine.** ML output, physics severities, forecasts, and rule alerts run in parallel with simple temporal voting. A workshop-grade tool would cross-check MAP↔load↔trim↔coolant consistency before raising an alert.
6. **Sample-rate-dependent windowing.** Even after T1.2, the 60-*row* buffer covers different real-time windows at different poll rates. True fix is timestamp-based resampling to 1 Hz upstream of `extract_features`. Deferred to V2.

---

## Execution order summary

```
T1.1  ✅ done (engine confirmed BTS petrol)
T1.2  ← START HERE — code fix, today
T1.3  ← needs car plugged in; after T1.2 lands
T1.4  ← 5 min, can do anytime after T1.2
─── Tier 1 complete; dashboard ready for Roomster ───
T2.1  ← code fix, ~30 min
T2.2  ← code fix, ~30 min
T2.3  ← code fix, ~1 hr
T2.4  ← code fix, ~15 min
T2.5  ← text only, thesis chapter
T2.6  ← text only, thesis chapter
─── Tier 2 complete; thesis defensible ───
Tier 3 → discuss in Limitations chapter, no code work
```

---

## Final acceptance test (run after every Tier 1 + Tier 2 code task)

```
.venv/Scripts/pytest.exe tests/ -q                  # must be 268+ passing
.venv/Scripts/streamlit.exe run src/dashboard/app.py
# manual: load drive1.csv, press Play, watch 120 s
#   - no traceback anywhere
#   - status banner renders (Bug B regression check)
#   - PID strip 4 unified cards updating live (Bug E regression check)
#   - severity gauges stay at 0% (drive1 is healthy)
#   - SHAP + recommendations render
#   - degraded-PID warning hidden (all 14 PIDs present on Etios)
```

When this passes after each task, commit. When all Tier 1 + Tier 2 code tasks pass, push to main.
