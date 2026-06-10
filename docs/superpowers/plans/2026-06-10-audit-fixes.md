# Audit Fixes Implementation Plan (Fable-5 audit, 2026-06-10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the logical errors found in the 2026-06-10 full-project audit — most critically, the headline real-fault recall metric that is currently documented (docs/REAL_FAULT_COLLECTION.md §10) but implemented nowhere correctly — before the 15 June defense.

**Architecture:** All fixes are surgical diffs to existing modules (project rule: no whole-file rewrites). One new shared function (`compute_fault_recall`) becomes the single source of truth for the §10 metric, consumed by both the CLI and the web API. One new small class (`LiveSessionStore`) persists live-session telemetry and fault marks. Everything else is targeted edits with regression tests.

**Tech Stack:** Python 3.11, pandas 2.2.3, scikit-learn 1.5.2, xgboost 2.1.3, pytest 8.3.3, FastAPI (api layer). Exact-pinned — do not bump versions.

**Note on model artefacts:** the project persists trained models with Python's standard serialization (`.pkl` bundles under `models/`, loaded only from this repo's own output — established project convention; do not introduce new serialization formats in this plan).

**Project conventions (from CLAUDE.md — binding):**
- Precise diffs only; never rewrite whole files. Use Edit, not Write, on existing files.
- Comments only for *why* (physics constraints, ECU quirks) — never *what*.
- Every new module/function gets a pytest test before moving on.
- Never generate sensor values that violate the Physics-First bounds in CLAUDE.md.
- Run commands with the project venv: `D:\Predictive-MaintenanceV1\Predictive-MaintenanceV1\.venv\Scripts\python.exe` (referred to below as `python`). Working dir for all commands: `D:\Predictive-MaintenanceV1\Predictive-MaintenanceV1`.

**Current ground truth (verified 2026-06-10):** 391 tests pass, 8 fail (all in `tests/test_classifier.py`, env-caused — see Task 0). Working tree has uncommitted changes to `src/api/routers/live.py` and `models/my_test_vehicle_normalizer.json`.

---

## Audit findings this plan fixes (context for the executor)

| # | Severity | Finding | Where |
|---|---|---|---|
| F1 | **P0** | Headline metric (§10 vacuum-leak recall) is implemented nowhere as specified. Docs promise `--fault-from/--fault-to` on `scripts/eval_real_fault.py` — flags don't exist. The only recall code (`src/api/routers/recordings.py:_compute_recall`) counts **any** non-healthy label as detection, so `cold_start`, `coolant_temp_sensor`, `throttle_position_sensor` labels all count as detecting a vacuum leak. A leak run that starts with a cold engine gets free "detections" for the whole warm-up. | `src/api/routers/recordings.py:27-36`, `scripts/eval_real_fault.py`, `docs/REAL_FAULT_COLLECTION.md:435-449` |
| F2 | **P0** | `fault_fraction` in the eval harness counts `cold_start` as a fault window (`lbl != "healthy"`). | `src/eval/real_fault_eval.py:158` |
| F3 | **P0** | `InferenceEngine` passes wall-clock `now=time.monotonic()` to `ColdStartChecker` on **every** path, violating the checker's own contract ("Omit in tests and CSV replay"). At 10× CSV replay the duration rules (90 s frozen-ECT, 480 s thermostat, 120 s IAC) measure wall seconds while data streams 10× faster → the rule panel silently never fires during the accelerated defense demo. The wall-clock path is obsolete anyway: the T3.1 resampler guarantees 1 row = 1 data-second on the live path too. | `src/dashboard/inference.py:327-333`, `src/diagnostics/cold_start_checker.py:141-147` |
| F4 | **P0** | Live WebSocket sessions persist nothing: telemetry only feeds the browser chart, and `mark_leak` (ground-truth annotation for the recall metric) is acked then dropped. A live Skoda fault run leaves no permanent record. | `src/api/routers/live.py:176-178`, `web/src/components/LiveSession.tsx:69` |
| F5 | **P1** | Train/serve skew on TPS severity baseline: training-time forecast targets use `compute_baselines()` (active-throttle windows only, by design — see severity.py:201-213), but `InferenceEngine` derives the same baseline from `norm.feature_means` (mean over ALL healthy windows, including idle windows where the extractor emits its 1.0 fallback). The dashboard severity gauge and the trained forecaster disagree about what "healthy ratio" means. | `src/dashboard/inference.py:206-213`, `src/features/severity.py:181-226` |
| F6 | **P1** | `_TPS_DEADBAND = 0.20` is justified in-code by "live12 had 0.19 Δ" — live12 is a held-out **test** session. A hyperparameter was tuned on the test set. | `src/features/severity.py:71`, `src/models/forecaster.py:111-116` |
| F7 | **P1** | 8 test failures: `psutil` install is broken in the venv (`module 'psutil' has no attribute 'Process'` from joblib/loky), breaking RandomForest `n_jobs=-1` tests only. Environment fix, not code. | venv |
| F8 | **P2** | Fuel-system injector lowers `ENGINE_LOAD`. OBD calculated load is an **airflow** ratio; a clogged injector cuts fuel, not air → at fixed throttle load is unchanged (driver compensation, not modeled in replay data, would *raise* it). Sign is wrong; magnitude small (~1.4%). | `src/injection/fault_injector.py:387-394` |
| F9 | **P2** | CLAUDE.md Injection Design Rule #1 requires step **and** ramp modes; only ramp exists. `generate_demo_data.py` configs carry a `"step"` field that is silently ignored. | `src/injection/fault_injector.py:8`, `scripts/generate_demo_data.py:42-46` |
| F10 | **P2** | `regime.py` docstring promises hysteresis bands; `detect_regime` is stateless with single thresholds. Doc lies. | `src/features/regime.py:14-19` |
| F11 | **P2** | Torque adapter rejects IAT > 90 °C as garbage. Heat soak at idle in a June demo can legitimately read 60–95 °C → real rows get NaN'd. | `scripts/adapt_torque_csv.py:163` |
| F12 | **P2** | `dataset_v1_meta.json` omits `magnitude_jitter` even when the dataset was built with it (`rebuild_all.py` uses `(0.6, 1.4)`) → metadata under-specifies reproduction. | `src/features/dataset_builder.py:236-267` |
| F13 | **P2** | Stale docstrings: `pid_forecast_dataset.py` header still claims severity is "the algebraic inverse of the injector's coefficients" (fixed by P0-2 in severity.py); `inference.py:set_sample_hz` docstring contradicts the post-T3.1 call pattern. | `src/features/pid_forecast_dataset.py:5-9`, `src/dashboard/inference.py:380-392` |
| F14 | **P3** | Skoda baseline σ fitted on as few as 20 windows is statistically thin (±16% noise on every z-score). | `scripts/live_baseline_capture.py` |
| F15 | **P3** | fuel_system precision 0.46 / LOSO live12 = 0.53 trace to per-session baseline offsets that the *global* train normalizer can't remove — yet deployment refits per vehicle. Training never mirrors that. Worth an experiment. | `results/loso_cv_results.json` |
| F16 | **P3** | No quantified evidence for the slow-adapter case: hold-last resampling at 0.3 Hz damps `__std` features vs 1 Hz training; F1 under that shift is unmeasured. | — |

**Do NOT touch (verified correct — strengths to preserve):** session-level split + normalizer fit on train only (`xgb_classifier.train`); forecast pairing with zero window overlap (`forecast_dataset.py`, window *t* ends exactly where *t+60s* begins); persistence-baseline comparison (`pid_forecaster.py`); withheld-coefficient eval (`eval_withheld_coeff.py`); README honest-numbers framing; the T3.1 1-Hz resampler in `inference.py:update()`; physics clamps in the injector.

**Execution order:** Task 0 → 1 → 2 → 3 → 4 → 5 → 6 (P0 done) → 7 → 8 (P1) → 9–11 (P2, any order) → 12–14 (P3, only if time before 13 June). Re-run `python scripts/rebuild_all.py` once after Task 7, not after every task.

---

### Task 0: Fix the broken `psutil` install (8 failing tests) and commit the dirty tree

**Files:** none (environment + git hygiene)

- [ ] **Step 0.1: Check what's uncommitted and commit or stash it**

```powershell
git status --short
git diff src/api/routers/live.py
```

If the `live.py` diff is intentional work-in-progress, commit it before this plan's work starts so plan commits stay atomic:

```powershell
git add src/api/routers/live.py models/my_test_vehicle_normalizer.json
git commit -m "WIP: live router + test vehicle normalizer (pre-audit-fixes snapshot)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 0.2: Verify nothing in the repo shadows psutil, then force-reinstall it**

```powershell
Get-ChildItem -Path . -Filter "psutil*" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "\.venv" }
.venv\Scripts\python.exe -m pip install --force-reinstall psutil
```

Expected: no shadow files found; pip reinstalls psutil cleanly.

- [ ] **Step 0.3: Run the previously-failing tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_classifier.py -q`
Expected: `8 passed` (previously 8 failed with `AttributeError: module 'psutil' has no attribute 'Process'`).

- [ ] **Step 0.4: Run the full suite to establish the green baseline**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: `399 passed`. No commit (environment-only).

---

### Task 1: Shared §10-compliant recall function

The single source of truth for the headline metric. Per `docs/REAL_FAULT_COLLECTION.md:444`:
`recall = |{windows where label ∈ {fuel_system, air_system}} ∪ {windows where anomaly_score ≥ 0.85}| / |fault-interval windows|`

**Files:**
- Modify: `src/eval/real_fault_eval.py` (append after `evaluate_real_fault`)
- Test: `tests/test_fault_recall.py` (create)

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_fault_recall.py`:

```python
"""§10 headline-metric recall — the detection set is {fuel_system, air_system}
labels OR anomaly_score >= 0.85.  cold_start and wrong-fault labels are NOT
detections of a vacuum leak (docs/REAL_FAULT_COLLECTION.md §10)."""

import pytest

from src.eval.real_fault_eval import compute_fault_recall


def _w(elapsed_s: int, label: str, anomaly: float = 0.0) -> dict:
    return {"elapsed_s": elapsed_s, "label": label, "anomaly_score": anomaly}


def test_air_and_fuel_labels_count_as_detection():
    windows = [_w(100, "air_system"), _w(110, "fuel_system"), _w(120, "healthy")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=130)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["n_fault_windows"] == 3
    assert r["detected_by_label"] == 2


def test_cold_start_is_not_a_detection():
    windows = [_w(100, "cold_start"), _w(110, "cold_start"), _w(120, "air_system")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=130)
    assert r["recall"] == pytest.approx(1 / 3)


def test_wrong_fault_label_is_not_a_detection():
    windows = [_w(100, "coolant_temp_sensor"), _w(110, "throttle_position_sensor")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["recall"] == 0.0


def test_anomaly_route_counts_even_when_label_healthy():
    windows = [_w(100, "healthy", anomaly=0.90), _w(110, "healthy", anomaly=0.10)]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["recall"] == pytest.approx(0.5)
    assert r["detected_by_anomaly_only"] == 1


def test_windows_outside_interval_are_ignored():
    windows = [_w(50, "air_system"), _w(100, "healthy"), _w(500, "fuel_system")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["n_fault_windows"] == 1
    assert r["recall"] == 0.0


def test_empty_interval_returns_zero_not_crash():
    r = compute_fault_recall([], fault_from_s=0, fault_to_s=100)
    assert r["recall"] == 0.0
    assert r["n_fault_windows"] == 0


def test_label_and_anomaly_on_same_window_not_double_counted():
    windows = [_w(100, "air_system", anomaly=0.99)]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["n_detected"] == 1
    assert r["recall"] == 1.0
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fault_recall.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_fault_recall'`.

- [ ] **Step 1.3: Implement `compute_fault_recall`**

Append to `src/eval/real_fault_eval.py` (after `evaluate_real_fault`):

```python
# §10 headline metric (docs/REAL_FAULT_COLLECTION.md): a vacuum leak may present
# through the trim route (fuel_system) or the mechanical route (air_system) —
# both count.  cold_start / coolant / TPS labels do NOT detect a vacuum leak.
_VACUUM_LEAK_DETECTION_LABELS: frozenset[str] = frozenset({"fuel_system", "air_system"})
_ANOMALY_DETECTION_THRESHOLD = 0.85


def compute_fault_recall(
    windows: list[dict],
    fault_from_s: int,
    fault_to_s: int,
    *,
    detection_labels: frozenset[str] = _VACUUM_LEAK_DETECTION_LABELS,
    anomaly_threshold: float = _ANOMALY_DETECTION_THRESHOLD,
) -> dict:
    """Vacuum-leak recall over the fault interval, exactly as defined in §10.

    Parameters
    ----------
    windows : list of dict
        Per-stride window records from ``evaluate_real_fault`` (each must
        carry ``elapsed_s``, ``label``, ``anomaly_score``).
    fault_from_s, fault_to_s : int
        Fault interval (mods-in / mods-out timestamps), seconds since start.
    detection_labels : frozenset
        Labels that constitute a detection.  Default = §10's set.
    anomaly_threshold : float
        Anomaly-route OR-branch threshold.  Default = §10's 0.85.

    Returns
    -------
    dict with recall, n_fault_windows, n_detected, detected_by_label,
    detected_by_anomaly_only.
    """
    in_interval = [w for w in windows if fault_from_s <= w["elapsed_s"] <= fault_to_s]
    if not in_interval:
        return {
            "recall": 0.0,
            "n_fault_windows": 0,
            "n_detected": 0,
            "detected_by_label": 0,
            "detected_by_anomaly_only": 0,
        }
    by_label = sum(1 for w in in_interval if w["label"] in detection_labels)
    detected = sum(
        1
        for w in in_interval
        if w["label"] in detection_labels
        or float(w.get("anomaly_score", 0.0)) >= anomaly_threshold
    )
    return {
        "recall": detected / len(in_interval),
        "n_fault_windows": len(in_interval),
        "n_detected": detected,
        "detected_by_label": by_label,
        "detected_by_anomaly_only": detected - by_label,
    }
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fault_recall.py -q`
Expected: `7 passed`.

- [ ] **Step 1.5: Commit**

```powershell
git add src/eval/real_fault_eval.py tests/test_fault_recall.py
git commit -m "Implement §10 vacuum-leak recall as a shared function (F1)

cold_start and wrong-fault labels no longer count as detections;
anomaly>=0.85 OR-branch implemented per REAL_FAULT_COLLECTION.md §10.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `fault_fraction` must not count cold_start as a fault

**Files:**
- Modify: `src/eval/real_fault_eval.py:156-170`
- Test: `tests/test_real_fault_harness_plumbing.py` (extend)

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_real_fault_harness_plumbing.py`:

```python
def test_fault_fraction_excludes_cold_start():
    from src.eval.real_fault_eval import _summarise_labels

    label_counts = {"healthy": 5, "cold_start": 3, "air_system": 2}
    summary = _summarise_labels(label_counts, n_windows=10)
    assert summary["fault_window_count"] == 2
    assert summary["fault_fraction"] == pytest.approx(0.2)
    assert summary["non_fault_window_count"] == 8
```

(Add `import pytest` at the top of the file if it is not already imported.)

- [ ] **Step 2.2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_real_fault_harness_plumbing.py::test_fault_fraction_excludes_cold_start -q`
Expected: FAIL — `ImportError: cannot import name '_summarise_labels'`.

- [ ] **Step 2.3: Extract and fix the summary computation**

In `src/eval/real_fault_eval.py`, add near the top of the module (after the imports):

```python
# Labels that are NOT fault detections — mirrors StableAlerter._NON_FAULT_LABELS.
# cold_start is a normal regime; warming_up is the pre-buffer placeholder.
_NON_FAULT_LABELS: frozenset[str] = frozenset({"healthy", "cold_start", "warming_up"})


def _summarise_labels(label_counts: dict[str, int], n_windows: int) -> dict:
    fault_count = sum(c for lbl, c in label_counts.items() if lbl not in _NON_FAULT_LABELS)
    return {
        "fault_window_count": fault_count,
        "non_fault_window_count": n_windows - fault_count,
        "fault_fraction": (fault_count / n_windows) if n_windows else 0.0,
        "label_counts": label_counts,
    }
```

Then replace the existing summary block at the end of `evaluate_real_fault` (currently lines 155-170):

```python
    label_counts: dict[str, int] = {}
    for w in windows:
        label_counts[w["label"]] = label_counts.get(w["label"], 0) + 1

    return {
        "csv_path": str(csv_path),
        "n_rows": len(rows),
        "n_windows": len(windows),
        "windows": windows,
        "summary": _summarise_labels(label_counts, len(windows)),
    }
```

- [ ] **Step 2.4: Run the harness tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_real_fault_harness_plumbing.py tests/test_cross_vehicle_skeleton.py -q`
Expected: all pass. (`cross_vehicle_eval.py` reads `fault_fraction` through the summary dict, so it inherits the fix with no change.)

- [ ] **Step 2.5: Commit**

```powershell
git add src/eval/real_fault_eval.py tests/test_real_fault_harness_plumbing.py
git commit -m "fault_fraction no longer counts cold_start as a fault (F2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire `--fault-from/--fault-to` into the CLI (the documented §9c workflow)

**Files:**
- Modify: `scripts/eval_real_fault.py`
- Test: manual CLI run against the mock fixture (script has no unit tests; the function it calls is tested in Task 1)

- [ ] **Step 3.1: Add the arguments**

In `scripts/eval_real_fault.py`, after the `--normalizer` argument block (line ~59), add:

```python
    parser.add_argument(
        "--fault-from",
        type=int,
        default=None,
        metavar="S",
        help="Fault-interval start (seconds since recording start; mods-in mark).",
    )
    parser.add_argument(
        "--fault-to",
        type=int,
        default=None,
        metavar="S",
        help="Fault-interval end (seconds since recording start; mods-out mark).",
    )
```

- [ ] **Step 3.2: Compute and report recall when both flags are given**

In the same file, after `result = evaluate_real_fault(csv_path, **engine_kwargs)` (line ~79), add:

```python
    if (args.fault_from is None) != (args.fault_to is None):
        log.error("--fault-from and --fault-to must be given together.")
        return 1
    if args.fault_from is not None:
        from src.eval.real_fault_eval import compute_fault_recall

        recall_block = compute_fault_recall(
            result["windows"], args.fault_from, args.fault_to
        )
        result["fault_interval"] = {
            "from_s": args.fault_from,
            "to_s": args.fault_to,
            **recall_block,
        }
```

And after the existing summary logging (line ~97), add:

```python
    if "fault_interval" in result:
        fi = result["fault_interval"]
        log.info(
            "  §10 vacuum-leak recall: %.3f  (%d/%d windows; by-label %d, anomaly-only %d)",
            fi["recall"], fi["n_detected"], fi["n_fault_windows"],
            fi["detected_by_label"], fi["detected_by_anomaly_only"],
        )
        log.info("  Pass criterion: recall >= 0.60 (REAL_FAULT_COLLECTION.md §10)")
```

- [ ] **Step 3.3: Verify against the mock fixture**

```powershell
.venv\Scripts\python.exe -m scripts.eval_real_fault data/real_faults/mock/mock_lean_fault.csv --fault-from 120 --fault-to 400
```

Expected: runs to completion, prints a `§10 vacuum-leak recall:` line, and the output JSON in `results/real_fault_eval/` contains a `fault_interval` block. (If the mock fixture lives at a different filename, list `data/real_faults/mock/` and use the CSV found there.)

- [ ] **Step 3.4: Commit**

```powershell
git add scripts/eval_real_fault.py
git commit -m "eval_real_fault CLI: --fault-from/--fault-to per §9c (F1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Web API recall uses the shared §10 function

**Files:**
- Modify: `src/api/routers/recordings.py:27-36` and the call site at line ~113

- [ ] **Step 4.1: Replace `_compute_recall` with the shared function**

In `src/api/routers/recordings.py`, delete the local `_compute_recall` (lines 27-36) and change the recall computation (currently `recall = _compute_recall(windows, fault_from_s, fault_to_s)`) to:

```python
    recall = None
    recall_detail = None
    if fault_from_s is not None and fault_to_s is not None:
        from src.eval.real_fault_eval import compute_fault_recall

        recall_detail = compute_fault_recall(windows, fault_from_s, fault_to_s)
        recall = recall_detail["recall"]
```

The `Recording` row keeps storing the float `recall`, so no schema change. If you want the detail surfaced, append it to the result JSON file instead of the DB (optional, not required).

- [ ] **Step 4.2: Run the API-adjacent tests + full suite**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: all pass (the deleted helper had no direct tests; the shared function is covered by Task 1).

- [ ] **Step 4.3: Commit**

```powershell
git add src/api/routers/recordings.py
git commit -m "recordings API: recall via shared §10 function — no more cold_start credit (F1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: ColdStartChecker — one time base (row counting), demo-speed regression test

The T3.1 resampler guarantees 1 processed row = 1 data-second on the live path, and CSV rows are 1 Hz by construction. Row counting is therefore the single correct time base everywhere; the wall-clock path actively breaks accelerated CSV replay.

**Files:**
- Modify: `src/dashboard/inference.py:319-333` (the ColdStartChecker call), `src/diagnostics/cold_start_checker.py:141-147` (docstring only)
- Test: `tests/test_cold_start_timing.py` (extend)

- [ ] **Step 5.1: Write the failing regression test**

Append to `tests/test_cold_start_timing.py`:

```python
def test_frozen_ect_fires_on_data_time_during_fast_replay():
    """At 10x CSV replay 95 rows arrive in well under 90 wall-seconds.
    The frozen-ECT rule is defined in DATA seconds (rows), so it must fire."""
    from collections import deque

    from src.config import WINDOW_LENGTH_S
    from src.dashboard.inference import InferenceEngine, _initial_state
    from src.diagnostics.cold_start_checker import ColdStartChecker
    from src.models.stable_alerter import StableAlerter

    eng = InferenceEngine.__new__(InferenceEngine)  # no model artefacts needed
    eng._cold_start = ColdStartChecker()
    eng._alerter = StableAlerter()
    eng._buffer = deque(maxlen=WINDOW_LENGTH_S)
    eng._rows_since_window = 0
    eng._elapsed_s = 0
    eng._last_state = _initial_state()
    eng._nan_warned = set()
    eng._sample_hz = 1.0
    eng._next_sample_t = None
    eng._run_window = lambda row, ready: eng._last_state  # ML path not under test

    row = {
        "COOLANT_TEMPERATURE": 50.0,  # cold AND perfectly flat -> stuck sensor
        "ENGINE_RPM": 850.0,
        "VEHICLE_SPEED": 0.0,
        "CONTROL_MODULE_VOLTAGE": 14.1,
    }
    for _ in range(95):  # 95 data-seconds streamed as fast as Python loops
        eng.update(dict(row))

    fired_rules = {a.rule for a in eng._cold_start.alerts}
    assert "ect_sensor_frozen" in fired_rules
```

- [ ] **Step 5.2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cold_start_timing.py::test_frozen_ect_fires_on_data_time_during_fast_replay -q`
Expected: FAIL — `_elapsed_s` derived from `time.monotonic()` stays ~0 during the fast loop, so the 90 s gate never opens and no alert fires.

- [ ] **Step 5.3: Stop passing wall-clock time from the engine**

In `src/dashboard/inference.py`, the ColdStartChecker call currently ends with `now=time.monotonic(),`. Edit it to:

```python
        # Row counting IS the time base: CSV rows are 1 Hz by construction and
        # the T3.1 resampler guarantees 1 processed row per data-second on the
        # live path.  Wall-clock `now` would desynchronise duration rules from
        # data time at any replay speed != 1x.
        new_rule_alerts = self._cold_start.update(
            coolant=90.0 if math.isnan(raw_coolant) else raw_coolant,
            rpm=800.0 if math.isnan(raw_rpm) else raw_rpm,
            speed=0.0 if math.isnan(raw_speed) else raw_speed,
            voltage=14.0 if math.isnan(raw_voltage) else raw_voltage,
        )
```

Also remove the now-unused `import time` **only if** nothing else in `inference.py` uses `time` (search first — if other uses exist, leave the import).

- [ ] **Step 5.4: Update the checker's `now` docstring to match reality**

In `src/diagnostics/cold_start_checker.py`, replace the `now` parameter paragraph (lines ~141-147) with:

```python
        now : float or None
            Optional wall-clock time from ``time.monotonic()``.  Only pass this
            when feeding rows at a non-1-Hz rate directly (bypassing the
            InferenceEngine resampler).  All engine-fed paths — CSV replay at
            any speed and resampled live rows — must omit it: one row equals
            one data-second there, and wall-clock time would desynchronise the
            duration rules from the data timeline.
```

- [ ] **Step 5.5: Run the regression test and the related suites**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cold_start_timing.py tests/test_cold_start_checker.py tests/test_dashboard_inference.py tests/test_inference_resampler.py -q`
Expected: all pass, including the new test.

- [ ] **Step 5.6: Commit**

```powershell
git add src/dashboard/inference.py src/diagnostics/cold_start_checker.py tests/test_cold_start_timing.py
git commit -m "ColdStartChecker: row-count time base everywhere — fixes dead rule panel at fast replay (F3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Persist live sessions — telemetry rows + mark_leak annotations

Minimal-robust design: a `LiveSessionStore` writes (a) `rows.csv` (elapsed_s + 14 PIDs, one line per data-second), (b) `marks.json` (every mark_leak with its data-time). Recall is then computed offline by replaying `rows.csv` through the already-tested Task 3 CLI — no new scoring path.

**Files:**
- Create: `src/api/live_store.py`
- Modify: `src/api/routers/live.py`
- Test: `tests/test_live_store.py` (create)

- [ ] **Step 6.1: Write the failing tests**

Create `tests/test_live_store.py`:

```python
import json
import math

from src.api.live_store import LiveSessionStore
from src.config import USEFUL_PIDS


def test_rows_csv_has_header_and_dedupes_by_elapsed(tmp_path):
    store = LiveSessionStore(tmp_path / "s1")
    row = {pid: 1.0 for pid in USEFUL_PIDS}
    store.append_row(elapsed_s=1, row=row)
    store.append_row(elapsed_s=1, row=row)  # same second -> ignored
    store.append_row(elapsed_s=2, row=row)
    store.close()

    lines = (tmp_path / "s1" / "rows.csv").read_text().strip().splitlines()
    assert lines[0].split(",")[0] == "elapsed_s"
    assert len(lines) == 3  # header + 2 unique seconds


def test_nan_pid_serialised_as_empty_cell(tmp_path):
    store = LiveSessionStore(tmp_path / "s2")
    row = {pid: 1.0 for pid in USEFUL_PIDS}
    row["TIMING_ADVANCE"] = float("nan")
    store.append_row(elapsed_s=1, row=row)
    store.close()

    header, data = (tmp_path / "s2" / "rows.csv").read_text().strip().splitlines()
    idx = header.split(",").index("TIMING_ADVANCE")
    assert data.split(",")[idx] == ""


def test_marks_written_immediately_with_elapsed(tmp_path):
    store = LiveSessionStore(tmp_path / "s3")
    store.record_mark(state="start", elapsed_s=42)
    store.record_mark(state="stop", elapsed_s=99)

    marks = json.loads((tmp_path / "s3" / "marks.json").read_text())
    assert [m["state"] for m in marks] == ["start", "stop"]
    assert [m["elapsed_s"] for m in marks] == [42, 99]
    store.close()
```

- [ ] **Step 6.2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_live_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.live_store'`.

- [ ] **Step 6.3: Implement `LiveSessionStore`**

Create `src/api/live_store.py`:

```python
"""Disk persistence for live ELM327 sessions.

A live Skoda run is unrepeatable evidence: rows.csv lets the §10 recall be
recomputed offline through the tested CLI path, and marks.json preserves the
mods-in / mods-out ground truth that the WebSocket used to discard.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.config import USEFUL_PIDS


class LiveSessionStore:
    """Append-only writer for one live session (rows.csv + marks.json)."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._marks: list[dict] = []
        self._last_elapsed: int = -1
        self._rows_f = open(
            self.session_dir / "rows.csv", "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._rows_f, fieldnames=["elapsed_s", *USEFUL_PIDS]
        )
        self._writer.writeheader()

    def append_row(self, elapsed_s: int, row: dict) -> None:
        if elapsed_s <= self._last_elapsed:
            return  # resampler can re-deliver the same data-second; keep one
        self._last_elapsed = elapsed_s
        rec: dict = {"elapsed_s": elapsed_s}
        for pid in USEFUL_PIDS:
            v = row.get(pid)
            try:
                f = float(v)
                rec[pid] = "" if math.isnan(f) else f
            except (TypeError, ValueError):
                rec[pid] = ""
        self._writer.writerow(rec)
        self._rows_f.flush()  # a crash mid-drive must not lose the run

    def record_mark(self, state: str, elapsed_s: int) -> None:
        self._marks.append({"state": str(state), "elapsed_s": int(elapsed_s)})
        (self.session_dir / "marks.json").write_text(
            json.dumps(self._marks, indent=2)
        )

    def close(self) -> None:
        if not self._rows_f.closed:
            self._rows_f.close()
        if not (self.session_dir / "marks.json").exists():
            (self.session_dir / "marks.json").write_text("[]")
```

- [ ] **Step 6.4: Run the store tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_live_store.py -q`
Expected: `3 passed`.

- [ ] **Step 6.5: Wire the store into the WebSocket session**

In `src/api/routers/live.py`:

(a) Add imports at the top (after the existing imports):

```python
from datetime import datetime, timezone

from src.api.config import DATA_APP_DIR
from src.api.live_store import LiveSessionStore
```

(b) In `_run_session`, right after `obd_src.start()` succeeds (line ~157), create the store and a shared elapsed tracker:

```python
    session_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = LiveSessionStore(DATA_APP_DIR / "live_sessions" / session_ts)
    last_elapsed = {"s": 0}
    log.info("Live WS: persisting session to %s", store.session_dir)
```

(c) In `_recv`, replace the `mark_leak` branch:

```python
                elif action == "mark_leak":
                    state_val = action_msg.get("state", "")
                    store.record_mark(state=state_val, elapsed_s=last_elapsed["s"])
                    await ws.send_json({
                        "type": "mark_ack",
                        "state": state_val,
                        "elapsed_s": last_elapsed["s"],
                    })
```

(d) In `_poll`, right after `state = await asyncio.to_thread(engine.update, row)` succeeds, add:

```python
            last_elapsed["s"] = state.elapsed_s
            store.append_row(elapsed_s=state.elapsed_s, row=row)
```

(e) Extend the existing `finally` around `await asyncio.gather(_recv(), _poll())`:

```python
    try:
        await asyncio.gather(_recv(), _poll())
    finally:
        obd_src.stop()
        store.close()
        log.info("Live WS: session saved — %s", store.session_dir)
```

- [ ] **Step 6.6: Verify the API module still imports and the suite passes**

Run: `.venv\Scripts\python.exe -c "from src.api.main import app; print('api ok')"`
Expected: `api ok`.
Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6.7: Commit**

```powershell
git add src/api/live_store.py src/api/routers/live.py tests/test_live_store.py
git commit -m "Persist live sessions: rows.csv + mark_leak ground truth (F4)

Score offline with: python -m scripts.eval_real_fault <rows.csv> --fault-from/--fault-to from marks.json

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: TPS severity baseline — same definition at train and serve

`compute_baselines()` deliberately excludes idle windows from the TPS ratio baseline (extractor emits a 1.0 fallback there). `InferenceEngine` must use that same definition instead of the all-windows `feature_means`. Fix: the normalizer computes and persists the severity baselines at `fit()` time; the engine prefers them.

**Files:**
- Modify: `src/features/normalizer.py` (fit/save/load + property), `src/dashboard/inference.py:206-213`
- Test: `tests/test_normalizer.py` (extend)

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_normalizer.py`:

```python
def test_severity_baselines_use_active_throttle_windows_only():
    """The TPS ratio baseline must ignore idle windows (extractor fallback=1.0).
    A global mean would be 0.75 here; the active-window baseline is 0.5."""
    import pandas as pd
    import pytest

    from src.features.extractor import feature_names
    from src.features.normalizer import BaselineNormalizer

    rows = []
    for _ in range(10):  # active-throttle windows: true vehicle ratio 0.5
        r = {c: 0.0 for c in feature_names()}
        r["THROTTLE__mean"] = 30.0
        r["THROTTLE_TO_PEDAL_RATIO"] = 0.5
        r["label"] = "healthy"
        rows.append(r)
    for _ in range(10):  # idle windows: extractor neutral fallback
        r = {c: 0.0 for c in feature_names()}
        r["THROTTLE__mean"] = 3.0
        r["THROTTLE_TO_PEDAL_RATIO"] = 1.0
        r["label"] = "healthy"
        rows.append(r)

    norm = BaselineNormalizer().fit(pd.DataFrame(rows))
    assert norm.severity_baselines is not None
    assert norm.severity_baselines["THROTTLE_TO_PEDAL_RATIO"] == pytest.approx(0.5)


def test_severity_baselines_survive_save_load(tmp_path):
    import pandas as pd

    from src.features.extractor import feature_names
    from src.features.normalizer import BaselineNormalizer

    rows = []
    for _ in range(5):
        r = {c: 1.0 for c in feature_names()}
        r["THROTTLE__mean"] = 30.0
        r["label"] = "healthy"
        rows.append(r)
    norm = BaselineNormalizer().fit(pd.DataFrame(rows))
    norm.save(tmp_path / "n.pkl")

    loaded = BaselineNormalizer.load(tmp_path / "n.pkl")
    assert loaded.severity_baselines == norm.severity_baselines
```

- [ ] **Step 7.2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_normalizer.py -k severity_baselines -q`
Expected: FAIL — `AttributeError: ... no attribute 'severity_baselines'`.

- [ ] **Step 7.3: Implement in the normalizer**

In `src/features/normalizer.py`:

(a) Add to the imports: `from src.features.severity import compute_baselines` (severity.py imports only numpy — no import cycle).

(b) In `__init__`, after `self._regime_means ... = None`, add:

```python
        # Severity-formula baselines captured at fit() time with the SAME
        # definition used to build forecast targets (compute_baselines —
        # active-throttle windows only for the TPS ratio).  Deriving these
        # from feature_means at inference silently used a different
        # definition (all windows incl. idle fallback=1.0) — train/serve skew.
        self._severity_baselines: dict[str, float] | None = None
```

(c) In `fit()`, after `self._regime_means = ...`, add:

```python
        self._severity_baselines = compute_baselines(healthy)
```

(d) In `save()`, after the `regime_means` line, add:

```python
        if self._severity_baselines is not None:
            bundle["severity_baselines"] = self._severity_baselines
```

(e) In `load()`, inside the `isinstance(bundle, dict)` branch, after `norm._regime_means = ...`, add:

```python
            norm._severity_baselines = bundle.get("severity_baselines")
```

(f) Add a property next to `feature_means`:

```python
    @property
    def severity_baselines(self) -> "dict[str, float] | None":
        """Baselines for compute_severity, captured at fit() time.  None for
        artefacts saved before this field existed (engine falls back to
        feature_means for those)."""
        return getattr(self, "_severity_baselines", None)
```

- [ ] **Step 7.4: Use them in the engine**

In `src/dashboard/inference.py`, the `__init__` baseline block currently reads:

```python
        feat_cols = feature_names()
        mean_arr = norm.feature_means
        self._baselines: dict[str, float] = {
            feat: float(mean_arr[i]) for i, feat in enumerate(feat_cols)
        }
```

Append directly after it:

```python
        # Prefer the fit-time severity baselines (same definition as the
        # forecast targets — TPS ratio from active-throttle windows only).
        # Old artefacts lack the field; feature_means stays the fallback.
        fit_baselines = getattr(norm, "severity_baselines", None)
        if fit_baselines:
            self._baselines.update(fit_baselines)
```

- [ ] **Step 7.5: Run normalizer + inference tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_normalizer.py tests/test_dashboard_inference.py tests/test_capture_baseline_from_csv.py -q`
Expected: all pass (live_baseline_capture flows through `BaselineNormalizer.fit`, so Skoda captures get the fix for free).

- [ ] **Step 7.6: Rebuild artefacts so the shipped bundles carry the baselines**

Run: `.venv\Scripts\python.exe scripts\rebuild_all.py`
Expected output shape: dataset build log, `Macro-F1: 0.79…` (within ±0.02 of 0.7970 — this change must NOT move the classifier, only the severity gauge), forecaster MAE lines, `Done.`
Then: `.venv\Scripts\python.exe -m pytest tests -q` → all pass.

- [ ] **Step 7.7: Commit**

```powershell
git add src/features/normalizer.py src/dashboard/inference.py tests/test_normalizer.py
git commit -m "Severity baselines captured at fit() — ends TPS train/serve skew (F5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Re-derive `_TPS_DEADBAND` from TRAIN sessions only (test-set hygiene)

The constant 0.20 is currently justified by live12 — a held-out test session. Re-derive from train sessions; whether the number changes or not, the justification must.

**Files:**
- Create: `scripts/derive_tps_deadband.py`
- Modify: `src/features/severity.py:71` (comment, possibly value)

- [ ] **Step 8.1: Write the derivation script**

Create `scripts/derive_tps_deadband.py`:

```python
"""Derive the TPS deadband from TRAIN sessions only.

_TPS_DEADBAND was set citing live12's healthy ratio scatter — but live12 is a
held-out TEST session, so the threshold was tuned on test data.  This script
recomputes the healthy cross-session ratio scatter using train sessions only,
which is the defensible derivation for the thesis.

Run:
    python -m scripts.derive_tps_deadband
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.features.dataset_builder import load_dataset
from src.models.classifier import _HELD_OUT_SESSIONS


def main() -> int:
    ds = load_dataset()
    train = ds[~ds["session_id"].isin(_HELD_OUT_SESSIONS)]
    healthy_active = train[
        (train["label"] == "healthy") & (train["THROTTLE__mean"] > 10.0)
    ]
    per_session = healthy_active.groupby("session_id")["THROTTLE_TO_PEDAL_RATIO"].mean()

    print("Per-session healthy active-throttle ratio means (TRAIN only):")
    print(per_session.round(4).to_string())
    scatter = float(per_session.max() - per_session.min())
    print(f"\nmax cross-session scatter: {scatter:.4f}")
    print(f"suggested _TPS_DEADBAND  : {scatter + 0.02:.2f}  (scatter + 0.02 margin)")
    print("\nDecision rule:")
    print("  suggestion <  0.20 -> update severity.py constant, rerun rebuild_all + loso_cv")
    print("  suggestion >= 0.20 -> keep 0.20, but fix the comment to cite this derivation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.2: Run it and record the output**

Run: `.venv\Scripts\python.exe -m scripts.derive_tps_deadband`
Expected: a table of ~7 train sessions and a suggested value. Paste the output into the commit message.

- [ ] **Step 8.3: Apply the decision rule**

Whatever the outcome, edit the comment at `src/features/severity.py:71`. If the value stays 0.20:

```python
_TPS_DEADBAND = 0.20             # healthy cross-session ratio scatter, derived from TRAIN sessions only (scripts/derive_tps_deadband.py) — test sessions excluded
```

If the derived suggestion is **below** 0.20, set the constant to the suggestion (2 decimal places), use the same comment, then re-run:

```powershell
.venv\Scripts\python.exe scripts\rebuild_all.py
.venv\Scripts\python.exe -m scripts.loso_cv
.venv\Scripts\python.exe -m pytest tests -q
```

and update the two numbers in `README.md` ("Headline numbers") and the forecaster comment at `src/models/forecaster.py:111-116` if the TPS MAE limit story changes.

- [ ] **Step 8.4: Commit**

```powershell
git add scripts/derive_tps_deadband.py src/features/severity.py
git commit -m "TPS deadband derived from train sessions only — removes test-set tuning (F6)

<paste script output here>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(Include `README.md`, `src/models/forecaster.py`, and regenerated `results/*.json` in the same commit if the value changed.)

---

### Task 9: Fuel-system injector — stop lowering ENGINE_LOAD

OBD "calculated load" is normalised **airflow**, not torque. A clogged injector reduces fuel, not air: at fixed throttle, load is unchanged. (On a real drive the driver presses further to hold speed, which would *raise* it — driver compensation isn't modeled in replayed data, so unchanged is the physically safe choice.)

**Files:**
- Modify: `src/injection/fault_injector.py:387-394`
- Test: `tests/test_injection.py` (extend)

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_injection.py` (it already defines `_make_session()` and `_params()` helpers — reuse them):

```python
def test_fuel_system_engine_load_unchanged():
    """Calculated load is an airflow ratio; a clogged injector cuts fuel, not
    air -> at fixed throttle the PID must not move."""
    df = _make_session()
    original_load = df["ENGINE_LOAD"].copy()
    out = inject_fault(df, _params("fuel_system", noise_std=0.0))
    pd.testing.assert_series_equal(
        out["ENGINE_LOAD"], original_load, check_names=False
    )
```

- [ ] **Step 9.2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_injection.py::test_fuel_system_engine_load_unchanged -q`
Expected: FAIL — the injector currently subtracts `load_delta`.

- [ ] **Step 9.3: Remove the load delta**

In `src/injection/fault_injector.py`, delete this block from `_inject_fuel_system` (lines ~387-394):

```python
    # ENGINE_LOAD: lean misfires mean less useful work per cycle — load drops
    # slightly at full ramp (~1.5 % load reduction per % LTFT bias).
    load_delta = ramp * magnitude_pct * 0.08  # ~1.4 % drop at full 18 % LTFT fault
    df["ENGINE_LOAD"] = np.clip(
        df["ENGINE_LOAD"].to_numpy(dtype=float) - load_delta,
        0.0,
        100.0,
    )
```

and replace it with:

```python
    # ENGINE_LOAD deliberately unchanged: the PID is normalised AIRFLOW, and a
    # clogged injector cuts fuel, not air.  (A real driver compensating for
    # lost torque would RAISE it — driver behaviour is outside replayed data.)
```

- [ ] **Step 9.4: Run the injection suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_injection.py tests/test_injection_severity_variation.py tests/test_air_system_physics.py -q`
Expected: all pass (no existing test asserted the old load drop — verified during the audit).

- [ ] **Step 9.5: Rebuild + sanity-check the classifier**

Run: `.venv\Scripts\python.exe scripts\rebuild_all.py`
Expected: Macro-F1 within ±0.03 of the previous run (the deleted signal was tiny and physically wrong; if F1 *drops* more than that, the model was leaning on the artifact — report the delta honestly in the commit message either way).

- [ ] **Step 9.6: Commit**

```powershell
git add src/injection/fault_injector.py tests/test_injection.py
git commit -m "Fuel-system fault: ENGINE_LOAD unchanged — calculated load is airflow, not torque (F8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Step injection mode (charter compliance) + honest demo configs

CLAUDE.md Injection Design Rule #1 promises step AND ramp for every fault; only ramp exists, and the demo generator's `"step"` field is silently ignored. A step is the `ramp_len = 1` degenerate case — the physics clamps (1 °C/s coolant inertia) still shape the trajectory, which is exactly right: a stuck sensor's *reading* can jump, but our injected coolant path stays inertia-limited by design.

**Files:**
- Modify: `src/injection/fault_injector.py` (`inject_session` signature), `scripts/generate_demo_data.py:60-68`
- Test: `tests/test_injection.py` (extend)

- [ ] **Step 10.1: Write the failing tests**

Append to `tests/test_injection.py`:

```python
def test_step_mode_reaches_full_magnitude_within_one_row():
    df = _make_session()
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = 0.0
    out = inject_session(
        df, "fuel_system", onset_fraction=0.5, magnitude=18.0,
        noise_std=0.0, random_seed=42, mode="step",
    )
    params = out.attrs["injection"]
    assert params.ramp_len == 1
    # well after onset, LTFT has integrated to ~the full magnitude
    onset = params.onset_idx
    post = out["LONG_TERM_FUEL_TRIM_BANK_1"].iloc[onset + 80:]
    assert post.mean() > 15.0


def test_step_mode_coolant_still_respects_thermal_inertia():
    """Physics-First rule: even a step fault cannot move the coolant READING
    faster than 1 degC per second."""
    import numpy as np

    df = _make_session()
    out = inject_session(
        df, "coolant_temp_sensor", onset_fraction=0.5, magnitude=42.0,
        noise_std=0.0, random_seed=42, mode="step",
    )
    deltas = np.abs(np.diff(out["COOLANT_TEMPERATURE"].to_numpy()))
    assert deltas.max() <= 1.0 + 1e-9
```

(Ensure `inject_session` is imported at the top of the test file: `from src.injection.fault_injector import inject_fault, inject_session, InjectionParams` — extend the existing import if needed.)

- [ ] **Step 10.2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_injection.py -k step_mode -q`
Expected: FAIL — `inject_session() got an unexpected keyword argument 'mode'`.

- [ ] **Step 10.3: Add `mode` to `inject_session`**

In `src/injection/fault_injector.py`:

(a) Change the `inject_session` signature (line ~195) to add the parameter after `ramp_fraction`:

```python
def inject_session(
    df: pd.DataFrame,
    fault_type: FaultType,
    *,
    onset_fraction: float = 0.40,
    ramp_fraction: float = 0.15,
    mode: Literal["ramp", "step"] = "ramp",
    magnitude: float | None = None,
    noise_std: float = 0.3,
    random_seed: int | None = None,
) -> pd.DataFrame:
```

(b) Change the `ramp_len` line inside it from `ramp_len = max(1, int(n * ramp_fraction))` to:

```python
    # A step fault is the ramp's 1-row degenerate case; per-PID physics clamps
    # (e.g. the 1 degC/s coolant inertia loop) still bound the trajectory.
    ramp_len = 1 if mode == "step" else max(1, int(n * ramp_fraction))
```

(c) Add a `mode` line to the docstring's Parameters section:

```python
    mode : "ramp" or "step"
        ramp = gradual wear (default); step = sudden failure (ramp_len=1).
```

(d) Update the module docstring line 8 from `Step mode (sudden onset) is reserved for a later iteration.` to:

```python
Step mode (sudden onset) is the ramp_len=1 degenerate case via inject_session(mode="step").
```

- [ ] **Step 10.4: Pass the demo config's mode through**

In `scripts/generate_demo_data.py`, the loop currently ignores the `mode` tuple field. Change the `inject_session` call (lines ~62-68) to:

```python
        injected = inject_session(
            df,
            fault_type=fault_type,
            onset_fraction=onset,
            mode=mode,
            magnitude=magnitude,
            random_seed=42,
        )
```

- [ ] **Step 10.5: Run tests + regenerate demo files**

Run: `.venv\Scripts\python.exe -m pytest tests/test_injection.py -q`
Expected: all pass.
Run: `.venv\Scripts\python.exe -m scripts.generate_demo_data`
Expected: 4 demo CSVs regenerated; the coolant demo is now genuinely a step fault.

- [ ] **Step 10.6: Commit**

```powershell
git add src/injection/fault_injector.py scripts/generate_demo_data.py tests/test_injection.py
git commit -m "Step injection mode (ramp_len=1) — charter rule 1 compliance; demo config no longer decorative (F9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Small correctness/doc fixes (batch)

**Files:**
- Modify: `src/features/regime.py:14-19`, `scripts/adapt_torque_csv.py:163`, `src/features/dataset_builder.py` (meta), `src/features/pid_forecast_dataset.py:5-9`, `src/dashboard/inference.py:380-392`

- [ ] **Step 11.1: regime.py — remove the false hysteresis claim (F10)**

Replace the docstring block (lines 14-19):

```python
Threshold behaviour
-------------------
Thresholds are single-valued and the detector is stateless: each 60-second
window is classified independently from its own mean values.  Window means
plus the 10-second stride already smooth boundary flicker, and the
StableAlerter provides the temporal hysteresis at the alert layer — adding
per-window hysteresis here would duplicate that responsibility.
```

- [ ] **Step 11.2: adapt_torque_csv.py — IAT heat-soak ceiling (F11)**

Change line 163 from `"INTAKE_AIR_TEMPERATURE": (-40.0, 90.0),` to:

```python
    "INTAKE_AIR_TEMPERATURE": (-40.0, 120.0),  # heat soak at idle reads 60-95 degC in summer; decoy columns read 200+
```

- [ ] **Step 11.3: dataset metadata records the jitter (F12)**

In `src/features/dataset_builder.py`:
(a) `build_dataset` — pass it through: change the `_save_metadata(...)` call to add `magnitude_jitter` as the final argument:

```python
    _save_metadata(
        output_dir, dataset, usable_files, random_seed, noise_std,
        onset_fraction, ramp_fraction, magnitudes or _DEFAULT_MAGNITUDE, output_name,
        magnitude_jitter,
    )
```

(b) `_save_metadata` — extend the signature with `magnitude_jitter: tuple[float, float] | None = None,` (after `output_name: str,`) and inside `meta["injection"]` add:

```python
            "magnitude_jitter": list(magnitude_jitter) if magnitude_jitter else None,
```

- [ ] **Step 11.4: pid_forecast_dataset.py — stale severity claim (F13)**

Replace the second docstring paragraph (lines 5-9, "The legacy forecast dataset … not a predictive model of physical reality.") with:

```python
The legacy forecast dataset (`src/features/forecast_dataset.py`) targets the
*severity scalar* from `src/features/severity.py`.  Since P0-2 that scalar is
anchored to external diagnostic thresholds rather than the injector's own
coefficients, but it still presupposes a known fault type and a formula.
This dataset instead targets *raw next-window PID values*.
```

- [ ] **Step 11.5: inference.py — set_sample_hz docstring matches the post-T3.1 reality (F13)**

Replace the first paragraph of the `set_sample_hz` docstring ("Call once per live tick…is correct for carOBD).") with:

```python
        """Update the ECU poll rate used by rate-dependent features.

        Post-T3.1 the resampler guarantees the buffer sees 1-Hz rows, so
        app.py correctly passes 1.0 after resampling.  Keep this method for
        callers that feed non-resampled rows into the engine directly
        (none in the current codebase).
```

(Keep the existing `hz < 0.1` paragraph below it unchanged.)

- [ ] **Step 11.6: Run the affected suites + rebuild meta**

Run: `.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_torque_adapter.py tests/test_pid_forecaster.py -q`
Expected: all pass.
Run: `.venv\Scripts\python.exe scripts\rebuild_all.py` (refreshes `dataset_v1_meta.json` with the jitter field).

- [ ] **Step 11.7: Commit**

```powershell
git add src/features/regime.py scripts/adapt_torque_csv.py src/features/dataset_builder.py src/features/pid_forecast_dataset.py src/dashboard/inference.py
git commit -m "Doc/metadata truth pass: regime hysteresis claim, IAT heat-soak range, jitter provenance, stale severity docstrings (F10-F13)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12 (P3, only if ≥1 day remains): per-session normalization LOSO experiment

Deployment refits the normalizer per vehicle, but training z-scores against a *global* train baseline — so per-session offsets (the live12 = 0.53 story, fuel precision 0.46) never get removed at train time even though they would be at deploy time. Measure whether mirroring deployment in training helps. **Experiment only — adoption is a team decision.**

**Files:**
- Create: `scripts/experiment_session_norm.py`

- [ ] **Step 12.1: Write the experiment script**

Create `scripts/experiment_session_norm.py`:

```python
"""LOSO with session-relative normalization (deployment-mirroring experiment).

Deployment refits the BaselineNormalizer on each vehicle's own healthy
baseline; training uses one global train-split scaler.  This experiment
z-scores every session against ITS OWN healthy windows before LOSO, which is
exactly the information available at deployment (the pre-fault capture).

Compare against results/loso_cv_results.json (global-scaler LOSO).
Adoption rule of thumb: min_f1 must improve by >= +0.05 with mean_f1 not
dropping by more than 0.02 — and the team signs off before any pipeline change.

Run:
    python -m scripts.experiment_session_norm
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from src.features.dataset_builder import load_dataset
from src.features.normalizer import BaselineNormalizer, normalised_feature_names
from src.models.classifier import ALL_LABELS


def per_session_transform(ds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in ds.groupby("session_id"):
        norm = BaselineNormalizer().fit(g)  # healthy windows of THIS session only
        out.append(norm.transform(g))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ds_z = per_session_transform(load_dataset())
    feat_z = normalised_feature_names()
    sessions = sorted(ds_z["session_id"].unique())

    scores: list[float] = []
    for held in sessions:
        tr = ds_z[ds_z["session_id"] != held]
        te = ds_z[ds_z["session_id"] == held]
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob",
            num_class=len(ALL_LABELS), eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        y_tr = tr["label_id"].to_numpy(int)
        clf.fit(
            tr[feat_z].to_numpy(float), y_tr,
            sample_weight=compute_sample_weight("balanced", y_tr),
        )
        f1 = f1_score(
            te["label_id"].to_numpy(int),
            clf.predict(te[feat_z].to_numpy(float)),
            average="macro",
        )
        scores.append(float(f1))
        print(f"  held out {held}: F1 = {f1:.4f}")

    result = {
        "per_session": dict(zip(sessions, scores)),
        "mean_f1": statistics.mean(scores),
        "std_f1": statistics.stdev(scores),
        "min_f1": min(scores),
        "max_f1": max(scores),
    }
    print(f"\nmean={result['mean_f1']:.4f}  min={result['min_f1']:.4f}")
    print("baseline (global scaler): see results/loso_cv_results.json (mean 0.855, min 0.525)")

    out = _REPO / "results" / "experiment_session_norm.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12.2: Run it (~10 min: 9 XGB fits)**

Run: `.venv\Scripts\python.exe -m scripts.experiment_session_norm`
Expected: per-session F1 lines + summary + `results/experiment_session_norm.json`.

- [ ] **Step 12.3: Commit the experiment + result (no pipeline change without team sign-off)**

```powershell
git add scripts/experiment_session_norm.py results/experiment_session_norm.json
git commit -m "Experiment: session-relative normalization LOSO (deployment-mirroring) (F15)

<paste mean/min comparison here>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13 (P3): slow-adapter robustness number for the thesis

**Files:**
- Create: `scripts/eval_low_rate_robustness.py`

- [ ] **Step 13.1: Write the script**

Create `scripts/eval_low_rate_robustness.py`:

```python
"""Quantify classifier F1 under a slow ELM327 adapter (hold-last resampling).

The live path resamples sub-1-Hz adapters by replaying the last row into each
missed 1-second slot (inference.py T3.1).  That stair-steps the signals: a
60-row window at a 0.33 Hz adapter holds only ~20 independent samples, so
__std/__delta features are damped relative to the 1-Hz training data.  This
script measures the macro-F1 cost of that shift on the held-out sessions, so
the thesis can state it instead of hand-waving it.

Method: monkeypatch the dataset builder's loader to decimate each session to
every 3rd row and hold-last back to 1 Hz, rebuild the dataset with identical
seeds, and score the SHIPPED model (trained at true 1 Hz) on the held-out
split of the degraded dataset.

Run:
    python -m scripts.eval_low_rate_robustness
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np

import src.features.dataset_builder as db
from src.models import xgb_classifier
from src.models.classifier import session_split

_KEEP_EVERY = 3  # simulate a ~0.33 Hz adapter


def hold_last_resample(df, keep_every: int = _KEEP_EVERY):
    out = df.copy()
    mask = (np.arange(len(out)) % keep_every) != 0
    out.iloc[mask, :] = np.nan
    out = out.ffill().bfill()
    out.attrs = df.attrs
    return out


_orig_load = db.load_carobd_csv


def _load_decimated(path):
    return hold_last_resample(_orig_load(path))


def main() -> int:
    db.load_carobd_csv = _load_decimated
    try:
        ds_low = db.build_dataset(
            magnitude_jitter=(0.6, 1.4), output_name="dataset_lowrate_sim"
        )
    finally:
        db.load_carobd_csv = _orig_load

    _, test_low = session_split(ds_low)
    clf, norm = xgb_classifier.load_model()
    res = xgb_classifier.evaluate(clf, norm, test_low)

    out = {
        "simulated_adapter_hz": round(1.0 / _KEEP_EVERY, 3),
        "macro_f1_lowrate": res["macro_f1"],
        "per_class": res["per_class"],
        "note": (
            "Model trained at 1 Hz, scored on hold-last-resampled held-out "
            "sessions. Compare macro_f1 against "
            "results/xgb_classifier_v1_results.json (1 Hz: ~0.797)."
        ),
    }
    out_path = _REPO / "results" / "lowrate_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"low-rate macro-F1: {res['macro_f1']:.4f}  (1 Hz reference: ~0.797)")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 13.2: Run and record**

Run: `.venv\Scripts\python.exe -m scripts.eval_low_rate_robustness`
Expected: a macro-F1 number for the 0.33 Hz simulation + `results/lowrate_robustness.json`. Whatever it is, it goes into the thesis robustness/limitations section as a measured number.

- [ ] **Step 13.3: Commit**

```powershell
git add scripts/eval_low_rate_robustness.py results/lowrate_robustness.json
git commit -m "Measured slow-adapter robustness: F1 at simulated 0.33 Hz hold-last resampling (F16)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14 (P3): Skoda pre-flight checklist (defense-day insurance)

**Files:**
- Create: `docs/ROOMSTER_PREFLIGHT.md`

- [ ] **Step 14.1: Write the checklist**

Create `docs/ROOMSTER_PREFLIGHT.md`:

```markdown
# Roomster Pre-Flight Checklist (run BEFORE the recall drive and BEFORE the defense demo)

Run `python -m scripts.live_discover` with the engine warm and idling, then verify:

1. **MAP PID present?** The air_system injector and MAP features assume a
   speed-density (MAP-based) engine like the Etios. If the Roomster's ECU
   reports MAF instead of MAP (or MAP ~= barometric at idle, no vacuum), a real
   vacuum leak will present mainly through fuel trims — the §10 metric already
   counts the fuel_system label as a detection, so the experiment still works,
   but the thesis must say which route fired and why.
2. **PEDAL_D / PEDAL_E / COMMANDED_THROTTLE_ACTUATOR present?** If absent
   (common on 2007 K-line ECUs via generic OBD), the TPS fault class is
   undetectable live — the engine NaN-fills those features with the healthy
   baseline (z = 0). Expected behaviour, but state it in Limitations rather
   than discovering it at the defense.
3. **Measured poll rate >= 0.5 Hz?** Below ~0.3 Hz the dashboard warns and
   accuracy degrades (see results/lowrate_robustness.json for the measured
   cost). If the clone adapter is slower, reduce the polled PID set or use the
   wired adapter.
4. **Baseline capture is a real drive**: engine fully warm, >= 15 km/h mean
   speed, >= 20 windows (~4 min) — the capture guards enforce this, but plan
   the route before parking-lot improvisation.
5. **Before inducing the leak**: start the live session (rows.csv + marks.json
   now persist server-side) OR start the phone-app recording, and note
   wall-clock mods-in/mods-out times as backup ground truth.
```

- [ ] **Step 14.2: Commit**

```powershell
git add docs/ROOMSTER_PREFLIGHT.md
git commit -m "Roomster pre-flight checklist: PID availability, MAF-vs-MAP route, poll-rate floor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final verification (after the last task you execute)

- [ ] Run: `.venv\Scripts\python.exe -m pytest tests -q` → expected all pass, 0 failed.
- [ ] Run: `.venv\Scripts\python.exe scripts\rebuild_all.py` once more if any injector/normalizer task changed after the last rebuild; confirm Macro-F1 printed and `results/*.json` regenerated.
- [ ] Confirm README "Headline numbers" still match `results/xgb_classifier_v1_results.json` and `results/loso_cv_results.json`; update the two numbers if Tasks 8/9 moved them.
- [ ] Smoke the demo path: `streamlit run src/dashboard/app.py`, select a `[DEMO]` file, set speed 10×, verify the cold-start rule panel fires (Task 5's fix is exactly this scenario).

## Out of scope (deliberately)

- σ-shrinkage for small Skoda baselines (F14): worthwhile, but riskier than its payoff 5 days out; the ≥20-window guard already bounds the worst case. Revisit post-defense.
- Adopting per-session normalization in the production pipeline (Task 12 is the experiment only).
- Any change to the locked windowing constants, fault taxonomy, or split design.
