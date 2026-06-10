# Defense Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live-Skoda web-app demo defense-ready: fix the cross-vehicle false-alarm root cause with guarded per-car calibration, harden the live path with a replay fallback and a sensor timeline, and close the macro-F1 / MAE gaps in a timeboxed push before a 13 June model freeze.

**Architecture:** All baseline fitting funnels through `process_captured_rows()` (scripts/live_baseline_capture.py), which gains variance/closed-loop guardrails — both CLIs and the new WebSocket calibrate mode call it. The live WS router (`src/api/routers/live.py::_run_session`) gains a `mode` (monitor/calibrate), an `armed` flag from `Car.baseline_normalizer_path`, discrete alert events, and a swappable `ReplayObdSource`. The React app adds a `SensorTimeline` (recharts) fed by the same WS frames live and by a new rows endpoint post-hoc.

**Tech Stack:** Python 3.11, FastAPI WebSockets, pytest 8.3.3, XGBoost 2.1.3, React 19 + recharts 3, venv at `.venv/Scripts/python.exe` (run everything from repo root `D:\Predictive-MaintenanceV1\Predictive-MaintenanceV1`).

**Spec:** `docs/superpowers/specs/2026-06-11-defense-sprint-design.md`

**Hard dates:** model freeze 13 June EOD · dress rehearsal 14 June · defense 15 June.

**Standing rules for every task:**
- Run tests with `./.venv/Scripts/python.exe -m pytest` (Bash) — never bare `python`.
- Full suite green before every commit: `./.venv/Scripts/python.exe -m pytest tests/ -q` (416+ passing at plan time; live-router tests need no hardware after Task 6).
- Synthetic test fixtures obey the Physics-First rules in CLAUDE.md (coolant ≤ 1 °C/s, trims ±25%, RPM>0 when moving).
- Tasks marked **[HUMAN]** need Adam/Ahmed with the car — the executor stops and asks rather than skipping.

---

## Phase A — P0: cross-vehicle calibration (no car needed)

### Task 0: Fit the Yaris baseline and prove the mechanism

Ahmed's 2 June drive was a HEALTHY Toyota Yaris 2014 that scored 64/64 windows
`air_system` because the only per-vehicle normalizer on disk was fit on mock
data. Fit a real one from his drive and re-evaluate.

**Files:**
- Create: `models/yaris_2014_normalizer.pkl` + `.json` (artifacts)
- Create: `results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json`

- [ ] **Step 1: Fit the baseline from the healthy drive**

```bash
./.venv/Scripts/python.exe -m scripts.capture_baseline_from_csv \
  --csv data/real_faults/ahmed/ahmed_drive_20260602.csv \
  --vehicle "Toyota Yaris 2014" \
  --out models/yaris_2014_normalizer.pkl
```
Expected: prints saved path; `models/yaris_2014_normalizer.json` sidecar exists with non-zero `feature_stds` for coolant features.

- [ ] **Step 2: Re-run the eval with the new normalizer**

```bash
./.venv/Scripts/python.exe scripts/eval_real_fault.py \
  data/real_faults/ahmed/ahmed_drive_20260602.csv \
  --normalizer models/yaris_2014_normalizer.pkl \
  --out results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json
```
Expected: label counts dominated by `healthy` (+ possibly `cold_start` early). Record the healthy fraction. This is **fit-on-self** — a mechanism smoke test, not a validation claim (say so in the commit message).

- [ ] **Step 3: Sanity-compare against the broken run**

Read `summary.label_counts` in the new JSON vs `results/real_fault_eval/ahmed_drive_20260602_v2_postfix.json` (64/64 air_system). If healthy+cold_start < 70% of windows, STOP and report — do not proceed to Task 1 with an unexplained result.

- [ ] **Step 4: Commit**

```bash
git add models/yaris_2014_normalizer.pkl models/yaris_2014_normalizer.json results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json
git commit -m "Yaris baseline from Ahmed's healthy drive — false alarms collapse (fit-on-self smoke test)"
```

### Task 1: Baseline guardrails in process_captured_rows (TDD)

The mock baseline passed all three existing guards (coolant ≥ 75 °C, speed ≥ 15,
≥ 20 windows) because nothing checks that the data *varies like a real engine*.

**Files:**
- Modify: `scripts/live_baseline_capture.py` (guards live in `process_captured_rows`, after the Guard-2 block at ~line 122 and after the Guard-3 block at ~line 137)
- Create: `tests/test_baseline_guardrails.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baseline_guardrails.py`:

```python
"""Guardrails: process_captured_rows must reject captures that cannot be a
real engine (the mock-baseline incident: constant 90.0 coolant, std=0 stats
poisoned every later z-score, and a healthy Yaris read 64/64 air_system)."""

import numpy as np
import pytest

from scripts.live_baseline_capture import process_captured_rows


def _mk_rows(n: int = 400, *, coolant: float | None = None,
             stft_amp: float = 2.0, seed: int = 0) -> list[dict]:
    """Physically plausible warmed-up drive: ~50 km/h, coolant climbing to 92°C
    at 0.08°C/s (< 1°C/s thermal inertia), trims inside ±25%."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "ENGINE_RPM": 1900 + 250 * np.sin(i / 30) + rng.normal(0, 40),
            "VEHICLE_SPEED": 50 + 8 * np.sin(i / 45) + rng.normal(0, 1.5),
            "THROTTLE": 22 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "ENGINE_LOAD": 35 + 5 * np.sin(i / 35) + rng.normal(0, 1),
            "COOLANT_TEMPERATURE": (coolant if coolant is not None
                                    else min(92.0, 76.0 + i * 0.08) + rng.normal(0, 0.2)),
            "LONG_TERM_FUEL_TRIM_BANK_1": 1.5 + rng.normal(0, 0.5),
            "SHORT_TERM_FUEL_TRIM_BANK_1": rng.normal(0, stft_amp),
            "INTAKE_MANIFOLD_PRESSURE": 55 + 8 * np.sin(i / 40) + rng.normal(0, 1),
            "ACCELERATOR_PEDAL_POSITION_D": 24 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "ACCELERATOR_PEDAL_POSITION_E": 24 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "COMMANDED_THROTTLE_ACTUATOR": 22 + 6 * np.sin(i / 25) + rng.normal(0, 1),
            "INTAKE_AIR_TEMPERATURE": 32 + rng.normal(0, 0.5),
            "TIMING_ADVANCE": 16 + 3 * np.sin(i / 20) + rng.normal(0, 1),
            "CONTROL_MODULE_VOLTAGE": 14.0 + rng.normal(0, 0.05),
        })
    return rows


def test_accepts_healthy_varied_capture():
    norm, meta = process_captured_rows(_mk_rows(), vehicle_name="fixture")
    assert meta["n_windows"] >= 20


def test_rejects_coolant_frozen_at_fallback():
    """Constant 90.0 is the NaN-fallback constant — the mock-capture signature."""
    with pytest.raises(ValueError, match="90.0"):
        process_captured_rows(_mk_rows(coolant=90.0), vehicle_name="mock")


def test_rejects_any_constant_present_pid():
    rows = _mk_rows()
    for r in rows:
        r["INTAKE_MANIFOLD_PRESSURE"] = 55.0
    with pytest.raises(ValueError, match="INTAKE_MANIFOLD_PRESSURE"):
        process_captured_rows(rows, vehicle_name="mock")


def test_rejects_open_loop_never_reached():
    """|STFT| sigma=0.2% stays under the 0.5% FUEL_LOOP_ACTIVE threshold in
    extractor.py — closed loop never detected, baseline must be refused."""
    with pytest.raises(ValueError, match="closed-loop"):
        process_captured_rows(_mk_rows(stft_amp=0.2), vehicle_name="openloop")


def test_absent_pid_is_exempt_from_variance_guard():
    """An unsupported PID arrives as NaN every row (LiveObdSource contract).
    That is legitimate (2007 Skoda may lack pedal PIDs) — must NOT reject."""
    rows = _mk_rows()
    for r in rows:
        r["ACCELERATOR_PEDAL_POSITION_D"] = float("nan")
    norm, meta = process_captured_rows(rows, vehicle_name="no-pedal")
    assert meta["n_windows"] >= 20
```

- [ ] **Step 2: Run tests, verify the 3 rejection tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_baseline_guardrails.py -v`
Expected: `test_accepts_healthy_varied_capture` PASSES, the three `rejects_*` tests FAIL (no ValueError raised), `absent_pid` PASSES.

- [ ] **Step 3: Implement the guards**

In `scripts/live_baseline_capture.py`, immediately AFTER the existing Guard-2
(mean speed) block and BEFORE the "Feature extraction" comment, insert:

```python
    # ── Guard 2b: per-PID variance ────────────────────────────────────────────
    # A real engine never produces a perfectly constant sensor over minutes of
    # driving (even battery voltage jitters).  A constant present-PID means the
    # capture source was synthetic/mock — fitting on it poisons every later
    # z-score (the my_test_vehicle incident: healthy Yaris read 64/64
    # air_system).  Absent PIDs (all-NaN) are exempt: NaN-fill below handles
    # them by design.
    cool = df.get("COOLANT_TEMPERATURE")
    if cool is not None and cool.notna().any():
        c = cool.dropna()
        if float(c.std(ddof=0)) == 0.0 and float(c.iloc[0]) == 90.0:
            raise ValueError(
                "COOLANT_TEMPERATURE is frozen at exactly 90.0 °C for the whole "
                "capture — that is the NaN-fallback constant, not a real engine. "
                "Is a real vehicle connected (not the mock source)?"
            )
    for pid in USEFUL_PIDS:
        col = df.get(pid)
        if col is None or not col.notna().any():
            continue  # absent PID — legitimately NaN-filled later
        if float(col.dropna().std(ddof=0)) == 0.0:
            raise ValueError(
                f"{pid} has zero variance across {len(df)} rows — a real engine "
                f"never produces a perfectly constant sensor. Baseline rejected "
                f"(synthetic/mock capture suspected)."
            )
```

Immediately AFTER the existing Guard-3 (window count) block, insert:

```python
    # ── Guard 4: closed-loop must be reached ──────────────────────────────────
    # STFT/LTFT carry no information in open loop; a baseline whose every
    # window is open-loop centres the trim features on frozen values.
    if all(fr.get("FUEL_LOOP_ACTIVE", 0.0) < 0.5 for fr in feature_rows):
        raise ValueError(
            "ECU never reached closed-loop fuel control during the capture "
            "(FUEL_LOOP_ACTIVE = 0 in every window). Warm the engine fully and "
            "drive normally before capturing a baseline."
        )
```

- [ ] **Step 4: Run the guardrail tests — all pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_baseline_guardrails.py -v`
Expected: 5 passed.

- [ ] **Step 5: Full suite + commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
git add scripts/live_baseline_capture.py tests/test_baseline_guardrails.py
git commit -m "Baseline guardrails: reject zero-variance / frozen-coolant / open-loop captures"
```

### Task 2: Remove the poisoned mock artifact + cover the CSV entry point

**Files:**
- Delete: `models/my_test_vehicle_normalizer.pkl`, `models/my_test_vehicle_normalizer.json`
- Test: append to `tests/test_baseline_guardrails.py`

- [ ] **Step 1: Append an integration test for the CSV capture path**

```python
def test_csv_capture_path_rejects_mock_like_file(tmp_path):
    """capture_baseline_from_csv funnels into process_captured_rows — the
    guards must hold for file-based captures too."""
    import pandas as pd
    from scripts.capture_baseline_from_csv import capture_baseline_from_csv

    rows = _mk_rows(coolant=90.0)
    csv = tmp_path / "mock_drive.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="90.0"):
        capture_baseline_from_csv(csv, vehicle_name="mock", out_path=tmp_path / "m.pkl")
```

- [ ] **Step 2: Run it**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_baseline_guardrails.py -v`
Expected: 6 passed (guards already shared — this should pass immediately; if it fails, capture_baseline_from_csv is bypassing process_captured_rows and that bypass must be removed).

- [ ] **Step 3: Delete the mock artifact and commit**

```bash
git rm models/my_test_vehicle_normalizer.pkl models/my_test_vehicle_normalizer.json
git add tests/test_baseline_guardrails.py
git commit -m "Delete mock-fitted normalizer; guard the CSV capture entry point"
```
Also clear any `baseline_normalizer_path` DB rows pointing at the deleted file:
```bash
./.venv/Scripts/python.exe -c "
from src.api.db import SessionLocal
from src.api.models import Car
db = SessionLocal()
for car in db.query(Car).all():
    if car.baseline_normalizer_path and 'my_test_vehicle' in car.baseline_normalizer_path:
        car.baseline_normalizer_path = None
db.commit(); db.close()
print('cleared')
"
```
(If `SessionLocal` is named differently in `src/api/db.py`, open that file and use its session factory — same pattern `get_db` uses.)

### Task 3: Acceptance check — healthy drive must not fire a stable alert

**Files:**
- Create: `scripts/acceptance_healthy_drive.py`

- [ ] **Step 1: Write the acceptance script**

```python
"""Acceptance for §2c of the defense-sprint spec: a healthy real-car drive,
scored with its own baseline, must (a) be ≥ 70% healthy+cold_start windows and
(b) never fire a stable alert when its windows stream through StableAlerter.

Run:
    python -m scripts.acceptance_healthy_drive results/real_fault_eval/<file>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.models.stable_alerter import StableAlerter

_OK_LABELS = {"healthy", "cold_start", "warming_up"}


def main(result_json: str) -> int:
    d = json.loads(Path(result_json).read_text())
    windows = d["windows"]
    n = len(windows)
    ok = sum(1 for w in windows if w["label"] in _OK_LABELS)
    frac = ok / n if n else 0.0

    alerter = StableAlerter()
    fired: list[dict] = []
    for w in windows:
        state = alerter.update(w["label"], float(w["confidence"]))
        if state.active:
            fired.append({"elapsed_s": w["elapsed_s"], "fault_type": state.fault_type})

    print(f"windows={n}  healthy_or_regime={ok}  fraction={frac:.3f}  (need >= 0.70)")
    print(f"stable_alerts_fired={len(fired)}  (need 0)")
    for f in fired[:5]:
        print("  ", f)
    passed = frac >= 0.70 and not fired
    print("ACCEPTANCE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

- [ ] **Step 2: Run it on the Task-0 result**

Run: `./.venv/Scripts/python.exe -m scripts.acceptance_healthy_drive results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json`
Expected: `ACCEPTANCE: PASS`. If FAIL, report the numbers and stop — Phase D retraining decisions need this context.

- [ ] **Step 3: Commit**

```bash
git add scripts/acceptance_healthy_drive.py
git commit -m "Healthy-drive acceptance script: >=70% healthy windows, zero stable alerts"
```

---

## Phase B — Demo spine: live path (backend)

### Task 4: Alert events in WS frames + alerts.json persistence (TDD)

The frame currently carries only the rolling label. The timeline needs
*discrete events*: stable-alert transitions and new rule alerts.

**Files:**
- Modify: `src/api/live_store.py` (add `record_alert`)
- Modify: `src/api/routers/live.py` (`_poll` inside `_run_session`, frame dict ~line 225)
- Test: append to `tests/test_live_store.py`

- [ ] **Step 1: Failing test for the store**

Append to `tests/test_live_store.py`:

```python
def test_alerts_written_immediately(tmp_path):
    store = LiveSessionStore(tmp_path / "s1")
    store.record_alert({"kind": "stable", "fault_type": "fuel_system",
                        "confidence": 0.91, "elapsed_s": 130})
    store.record_alert({"kind": "rule", "rule": "ect_sensor_frozen", "elapsed_s": 95})
    data = json.loads((tmp_path / "s1" / "alerts.json").read_text())
    assert len(data) == 2
    assert data[0]["fault_type"] == "fuel_system"
    store.close()
```
(Match the existing imports/fixtures style at the top of the file; `json` is already imported there.)

- [ ] **Step 2: Run it — fails with AttributeError**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_store.py -v`
Expected: FAIL — `'LiveSessionStore' object has no attribute 'record_alert'`.

- [ ] **Step 3: Implement record_alert in LiveSessionStore**

Mirror `record_mark` exactly (immediate flush — a crash must not lose alert history):

```python
    def record_alert(self, event: dict) -> None:
        self._alerts.append(dict(event))
        (self.session_dir / "alerts.json").write_text(json.dumps(self._alerts, indent=2))
```
Add `self._alerts: list[dict] = []` in `__init__` next to `self._marks`, and in
`close()` mirror the marks behaviour: write `[]` if `alerts.json` doesn't exist.

- [ ] **Step 4: Run store tests — pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_store.py -v`

- [ ] **Step 5: Emit events from _poll**

In `src/api/routers/live.py::_run_session`, inside `_poll()` before the loop add:

```python
        prev_stable_active = False
        prev_stable_fault = ""
        prev_rule_count = 0
```

After `state = await asyncio.to_thread(engine.update, row)` and the
`store.append_row(...)` line, add:

```python
            # Discrete alert events: stable-alert transitions + new rule alerts.
            alert_events: list[dict] = []
            sa = state.stable_alert
            if sa.active and (not prev_stable_active or sa.fault_type != prev_stable_fault):
                alert_events.append({
                    "kind": "stable", "fault_type": sa.fault_type,
                    "confidence": round(float(sa.confidence), 4),
                    "elapsed_s": state.elapsed_s,
                })
            elif prev_stable_active and not sa.active:
                alert_events.append({"kind": "clear", "elapsed_s": state.elapsed_s})
            prev_stable_active, prev_stable_fault = sa.active, sa.fault_type

            if len(state.rule_alerts) > prev_rule_count:
                for ra in state.rule_alerts[prev_rule_count:]:
                    alert_events.append({
                        "kind": "rule",
                        "rule": getattr(ra, "rule", str(ra)),
                        "elapsed_s": state.elapsed_s,
                    })
                prev_rule_count = len(state.rule_alerts)

            for ev in alert_events:
                store.record_alert(ev)
```

Add to the `frame` dict: `"alert_events": alert_events,`

- [ ] **Step 6: Full suite + commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/api/live_store.py src/api/routers/live.py tests/test_live_store.py
git commit -m "Live WS: discrete alert events in frames, persisted to alerts.json"
```

### Task 5: ReplayObdSource — recorded CSV behind the live interface (TDD)

**Files:**
- Create: `src/live/replay_source.py`
- Create: `tests/test_replay_source.py`

- [ ] **Step 1: Failing tests**

```python
"""ReplayObdSource must be drop-in for LiveObdSource: same methods, same
row contract (all 14 USEFUL_PIDS keys, NaN for absent ones)."""

import math
from pathlib import Path

import pandas as pd

from src.config import USEFUL_PIDS
from src.live.replay_source import ReplayObdSource


def _demo_csv(tmp_path: Path, drop: str | None = None) -> Path:
    n = 5
    data = {p: [float(i) + 1.0 for i in range(n)] for p in USEFUL_PIDS}
    if drop:
        del data[drop]
    p = tmp_path / "session.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def test_drains_all_rows_instantly_when_not_realtime(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path), realtime=False, loop=False)
    assert src.connect()
    src.start()
    rows = []
    while (r := src.next_row()) is not None:
        rows.append(r)
    assert len(rows) == 5
    assert set(rows[0].keys()) == set(USEFUL_PIDS)


def test_missing_column_becomes_nan_and_missing_pid(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path, drop="ACCELERATOR_PEDAL_POSITION_D"),
                          realtime=False, loop=False)
    assert src.connect()
    src.start()
    row = src.next_row()
    assert math.isnan(row["ACCELERATOR_PEDAL_POSITION_D"])
    assert "ACCELERATOR_PEDAL_POSITION_D" in src.missing_pids


def test_connect_false_for_missing_file(tmp_path):
    src = ReplayObdSource(tmp_path / "nope.csv")
    assert src.connect() is False


def test_loop_mode_wraps_around(tmp_path):
    src = ReplayObdSource(_demo_csv(tmp_path), realtime=False, loop=True)
    src.connect(); src.start()
    for _ in range(7):
        assert src.next_row() is not None  # 5 rows + wrap + 2 more
```

- [ ] **Step 2: Run — fails with ModuleNotFoundError**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_replay_source.py -v`

- [ ] **Step 3: Implement src/live/replay_source.py**

```python
"""Replay a recorded 1 Hz session CSV through the LiveObdSource interface.

Demo insurance: if the ELM327 or the car misbehaves on stage, the SAME
LiveSession UI keeps running from a recorded drive — identical rendering path,
identical frames. Enabled only when PM_ALLOW_REPLAY=1 (see routers/live.py).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import USEFUL_PIDS


class ReplayObdSource:
    """Drop-in for LiveObdSource: connect/start/stop/next_row/connected/
    missing_pids/measured_poll_hz. Paces rows at true 1 Hz unless
    realtime=False (tests drain instantly)."""

    def __init__(self, csv_path: Path | str, *, realtime: bool = True,
                 loop: bool = True) -> None:
        self.csv_path = Path(csv_path)
        self.realtime = realtime
        self.loop = loop
        self._df: Optional[pd.DataFrame] = None
        self._idx = 0
        self._t0: Optional[float] = None

    def connect(self, timeout: float = 0.0) -> bool:
        if not self.csv_path.exists():
            return False
        df = pd.read_csv(self.csv_path)
        if not any(p in df.columns for p in USEFUL_PIDS):
            return False
        self._df = df
        return True

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._idx = 0

    def stop(self) -> None:
        self._t0 = None

    @property
    def connected(self) -> bool:
        return self._df is not None

    @property
    def missing_pids(self) -> list[str]:
        if self._df is None:
            return list(USEFUL_PIDS)
        return [p for p in USEFUL_PIDS
                if p not in self._df.columns or self._df[p].isna().all()]

    @property
    def measured_poll_hz(self) -> float:
        return 1.0 if (self._t0 is not None and self._idx > 0) else 0.0

    def next_row(self) -> Optional[dict[str, float]]:
        if self._df is None or self._t0 is None:
            return None
        if self._idx >= len(self._df):
            if not self.loop:
                return None
            self._idx = 0
            self._t0 = time.monotonic()
        if self.realtime and (time.monotonic() - self._t0) < self._idx:
            return None  # not yet time for the next 1 Hz row
        i = self._idx
        self._idx += 1
        out: dict[str, float] = {}
        for p in USEFUL_PIDS:
            if p in self._df.columns and pd.notna(self._df[p].iloc[i]):
                out[p] = float(self._df[p].iloc[i])
            else:
                out[p] = float("nan")
        return out
```

- [ ] **Step 4: Run tests — pass; commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_replay_source.py -v
git add src/live/replay_source.py tests/test_replay_source.py
git commit -m "ReplayObdSource: recorded CSV behind the live source interface (demo fallback)"
```

### Task 6: Wire replay into the live router + bench E2E test

**Files:**
- Modify: `src/api/routers/live.py` (source construction ~line 145; serial-ports helper ~lines 50-62)
- Create: `tests/test_live_ws_replay.py`

- [ ] **Step 1: Source selection in _run_session**

Replace the construction
`obd_src = LiveObdSource(port=port, sample_hz=1.0)` with:

```python
    import os
    if port and port.startswith("replay:") and os.environ.get("PM_ALLOW_REPLAY") == "1":
        from src.live.replay_source import ReplayObdSource
        # PM_REPLAY_FAST=1 is test-only: drains rows without 1 Hz pacing so the
        # bench/calibrate tests finish in seconds instead of minutes.
        obd_src = ReplayObdSource(
            csv_path=port[len("replay:"):],
            realtime=os.environ.get("PM_REPLAY_FAST") != "1",
        )
        log.info("Live WS: REPLAY source — %s", obd_src.csv_path)
    else:
        obd_src = LiveObdSource(port=port, sample_hz=1.0)
```

- [ ] **Step 2: Advertise replay pseudo-ports**

In the serial-ports helper (the function feeding `listSerialPorts`, ~line 50),
after building the real ports list, append before returning:

```python
        import os
        if os.environ.get("PM_ALLOW_REPLAY") == "1":
            # Relative paths: the API server always runs from the repo root
            # (config.py has no demo-dir constant — generate_demo_data builds
            # its own). Replay entries are invisible without the env flag.
            ports += [
                {"device": f"replay:{p}", "description": f"REPLAY (demo) — {p.name}"}
                for p in sorted(Path("data/demo").glob("demo_*.csv"))
            ]
            ahmed = Path("data/real_faults/ahmed/ahmed_drive_20260602.csv")
            if ahmed.exists():
                ports.append({"device": f"replay:{ahmed}", "description": "REPLAY — Yaris healthy drive"})
```

- [ ] **Step 3: Bench E2E test — the whole live stack with zero hardware**

Create `tests/test_live_ws_replay.py`:

```python
"""End-to-end bench: replay CSV -> _run_session -> engine -> store -> WS frames.
Covers the live stack without an ELM327. Slow (~10-20 s: SHAP init)."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

DEMO = Path("data/demo/demo_fuel_system.csv")


@pytest.mark.skipif(not DEMO.exists(), reason="demo CSV not generated")
def test_replay_session_streams_frames(monkeypatch, tmp_path):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({
            "action": "connect",
            "port": f"replay:{DEMO}",
            "car_id": None,
        }))
        frames = []
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "telemetry":
                frames.append(msg)
            if len(frames) >= 3:
                break
        ws.send_text(json.dumps({"action": "stop"}))

    assert len(frames) >= 3
    f = frames[-1]
    assert "telemetry" in f and "label" in f and "alert_events" in f
    assert f["poll_hz"] >= 0.0
```
Note: the WS route path must match the router's actual mount (`/ws/live` per
`@router.websocket("/ws/live")` plus any prefix in `src/api/main.py` —
check `app.include_router` there and adjust the URL).
Replay paces at 1 Hz, so 3 frames ≈ 3-4 s wall time — acceptable.

- [ ] **Step 4: Run it, then the full suite, then commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_ws_replay.py -v
./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/api/routers/live.py tests/test_live_ws_replay.py
git commit -m "Replay source wired into live WS behind PM_ALLOW_REPLAY; bench E2E test"
```

### Task 7: Armed flag — no fault alerts on an uncalibrated car

**Files:**
- Modify: `src/api/routers/live.py`

- [ ] **Step 1: Compute armed and put it in every frame**

In `_run_session`, right after the normalizer lookup block (~line 135):

```python
    # A car with no valid per-vehicle baseline runs DISARMED: telemetry and
    # timeline stream, but fault alerts are suppressed — the Etios-trained
    # z-scores are meaningless on another vehicle (the Yaris 64/64 incident).
    armed = normalizer_path is not None
```
Replay sessions score recorded data from known vehicles — treat
`port.startswith("replay:")` as `armed = True` (add `or port.startswith("replay:")`).

In `_poll`, wrap the alert-event emission from Task 4:

```python
            if not armed:
                alert_events = []
```
(after building them, before `store.record_alert` / frame assembly), and add
`"armed": armed,` to the frame dict.

- [ ] **Step 2: Extend the bench test**

Append to `tests/test_live_ws_replay.py`:

```python
@pytest.mark.skipif(not DEMO.exists(), reason="demo CSV not generated")
def test_replay_sessions_are_armed(monkeypatch):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "connect", "port": f"replay:{DEMO}", "car_id": None}))
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "telemetry":
                assert msg["armed"] is True
                break
        ws.send_text(json.dumps({"action": "stop"}))
```

- [ ] **Step 3: Run, full suite, commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_ws_replay.py -v
./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/api/routers/live.py tests/test_live_ws_replay.py
git commit -m "Armed flag: fault alerts suppressed on uncalibrated cars"
```

### Task 8: Calibrate mode over the live WebSocket

**Files:**
- Modify: `src/api/routers/live.py`
- Modify: `scripts/capture_baseline_from_csv.py` + `scripts/live_baseline_capture.py` (extract the shared save helper)
- Test: append to `tests/test_live_ws_replay.py`

- [ ] **Step 1: Extract the save helper (refactor, no behaviour change)**

Both `capture_baseline_from_csv.py` and `live_baseline_capture.py::run_capture`
end with near-identical "save pkl + write sidecar json" blocks after
`process_captured_rows` succeeds. Move those exact lines into ONE function in
`scripts/live_baseline_capture.py`:

```python
def save_normalizer_bundle(norm, metadata: dict, out_path: Path) -> Path:
    """Persist a fitted per-vehicle normalizer + its sidecar JSON."""
    # (moved verbatim from the save block that previously lived in the CLIs)
```
Call it from both CLI paths. Verify no behaviour change by re-running the
Task-0 fit command and diffing the sidecar JSON against the committed one
(only `capture_date` may differ):

```bash
./.venv/Scripts/python.exe -m scripts.capture_baseline_from_csv --csv data/real_faults/ahmed/ahmed_drive_20260602.csv --vehicle "Toyota Yaris 2014" --out models/yaris_2014_normalizer.pkl
git diff --stat models/yaris_2014_normalizer.json
```

- [ ] **Step 2: Implement calibrate mode in _run_session**

The connect action gains `mode`: `{action:'connect', port, car_id, mode:'calibrate'}`
(default `'monitor'`). After the source is connected and started, branch:

```python
    mode = msg.get("mode", "monitor")
    if mode == "calibrate":
        await _run_calibration(ws, obd_src, car_id)
        return
```

Add the coroutine (same file, after `_run_session`):

```python
async def _run_calibration(ws: WebSocket, obd_src, car_id: Optional[int]) -> None:
    """Collect rows until the client sends finish_calibration (or 12 min cap),
    then fit through the guarded process_captured_rows and persist per-car."""
    from scripts.live_baseline_capture import process_captured_rows, save_normalizer_bundle

    session_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = LiveSessionStore(DATA_APP_DIR / "live_sessions" / f"{session_ts}_calibration")
    rows: list[dict] = []
    finish = asyncio.Event()

    async def _recv() -> None:
        while not finish.is_set():
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.25)
                action = json.loads(raw).get("action")
                if action in ("finish_calibration", "stop"):
                    finish.set()
            except asyncio.TimeoutError:
                pass
            except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
                finish.set()

    async def _collect() -> None:
        while not finish.is_set() and len(rows) < 720:  # 12 min hard cap
            row = obd_src.next_row()
            if row is None:
                await asyncio.sleep(0.05)
                continue
            rows.append(row)
            store.append_row(elapsed_s=len(rows) - 1, row=row)
            if len(rows) % 5 == 0:
                try:
                    await ws.send_json({
                        "type": "calibrate_progress",
                        "rows_collected": len(rows),
                        "elapsed_s": len(rows) - 1,
                    })
                except (RuntimeError, WebSocketDisconnect):
                    finish.set()

    try:
        await asyncio.gather(_recv(), _collect())
    finally:
        obd_src.stop()
        store.close()

    try:
        norm, meta = await asyncio.to_thread(
            process_captured_rows, rows,
            f"car_{car_id}" if car_id is not None else "uncatalogued",
            None, 1.0,
        )
    except ValueError as exc:
        await ws.send_json({"type": "calibrate_result", "ok": False, "reason": str(exc)})
        return

    out = MODELS_DIR / f"car_{car_id}_normalizer.pkl"
    await asyncio.to_thread(save_normalizer_bundle, norm, meta, out)

    if car_id is not None:
        db = next(get_db())
        try:
            car = db.get(Car, int(car_id))
            if car:
                car.baseline_normalizer_path = str(out)
                db.commit()
        finally:
            db.close()

    await ws.send_json({
        "type": "calibrate_result", "ok": True,
        "n_windows": meta["n_windows"], "path": str(out),
    })
```
Imports needed at top of live.py: `MODELS_DIR` from `src.config` (check the
existing import line — `DATA_APP_DIR` is already imported there).
Skip the engine load when `mode == "calibrate"` (move the `_load_engine` call
below the mode branch — calibration needs no model).

- [ ] **Step 3: Bench test calibrate happy path + rejection path**

Append to `tests/test_live_ws_replay.py`:

```python
YARIS = Path("data/real_faults/ahmed/ahmed_drive_20260602.csv")


@pytest.mark.skipif(not YARIS.exists(), reason="Yaris drive not present")
def test_calibrate_mode_fits_and_reports(monkeypatch):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"action": "connect", "port": f"replay:{YARIS}",
                                 "car_id": None, "mode": "calibrate"}))
        progressed = False
        for _ in range(400):
            msg = ws.receive_json()
            if msg["type"] == "calibrate_progress":
                progressed = True
                if msg["rows_collected"] >= 400:
                    ws.send_text(json.dumps({"action": "finish_calibration"}))
            elif msg["type"] == "calibrate_result":
                assert msg["ok"] is True, msg
                assert msg["n_windows"] >= 20
                break
        assert progressed
```
NOTE: replay paces at 1 Hz → 400 rows ≈ 400 s. That is too slow for a test.
Make the pacing patchable: in the test, before connecting, add
`monkeypatch.setattr("src.live.replay_source.ReplayObdSource", lambda csv_path, **kw: __import__("src.live.replay_source", fromlist=["ReplayObdSource"]).ReplayObdSource(csv_path, realtime=False, loop=False))` —
or cleaner: have `_run_session` read `PM_REPLAY_FAST=1` env and pass
`realtime=False` to ReplayObdSource. Implement the env-var route (3 lines) and
set both env vars in the test. Document `PM_REPLAY_FAST` as test-only.

- [ ] **Step 4: Run, full suite, commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_ws_replay.py -v
./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/api/routers/live.py scripts/live_baseline_capture.py scripts/capture_baseline_from_csv.py tests/test_live_ws_replay.py
git commit -m "WS calibrate mode: guarded per-car baseline fit + DB path update"
```

### Task 9: Live hardening — missing PIDs by name, no-data watchdog, one reconnect

**Files:**
- Modify: `src/api/routers/live.py` (`_poll`)

- [ ] **Step 1: Names, not counts**

In the frame dict, alongside `degraded_pid_count`, add:
`"missing_pids": list(obd_src.missing_pids),`

- [ ] **Step 2: No-data watchdog with one reconnect attempt**

In `_poll`, the `row is None` branch currently just sleeps. Initialise
`no_data_s = 0.0` and `reconnect_tried = False` before the loop (next to the
Task-4 `prev_*` variables), then replace the None branch with:

```python
            if row is None:
                no_data_s += 0.05
                if no_data_s > 10.0 and not reconnect_tried:
                    reconnect_tried = True
                    await ws.send_json({"type": "warning",
                                        "message": "No data for 10 s — attempting one reconnect…"})
                    await asyncio.to_thread(obd_src.stop)
                    ok = await asyncio.to_thread(obd_src.connect)
                    if ok:
                        obd_src.start()
                        no_data_s = 0.0
                        await ws.send_json({"type": "warning", "message": "Reconnected."})
                    else:
                        await ws.send_json({"type": "error",
                                            "message": "Adapter unresponsive — session ended. "
                                                       "Check ignition and USB, then reconnect."})
                        stop_event.set()
                        break
                elif no_data_s > 20.0:
                    await ws.send_json({"type": "error",
                                        "message": "No data after reconnect — session ended."})
                    stop_event.set()
                    break
                await asyncio.sleep(0.05)
                continue
            no_data_s = 0.0
```

- [ ] **Step 3: Verify by bench, full suite, commit**

The replay bench tests still pass (replay always yields rows within pacing).
```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_ws_replay.py tests/ -q
git add src/api/routers/live.py
git commit -m "Live hardening: missing PIDs by name, 10s no-data watchdog, single reconnect"
```

### Task 10: Latency measurement

**Files:**
- Modify: `src/api/routers/live.py` (`_poll`, session finally block)

- [ ] **Step 1: Stamp and collect**

In `_poll`, when a row arrives: `t_poll = time.time()` (import `time` at top if
absent). Add `"t_poll": round(t_poll, 3),` to the frame. After `ws.send_json(frame)`
append `latencies_ms.append((time.time() - t_poll) * 1000.0)`
(`latencies_ms: list[float] = []` initialised before the loop, declared in
`_run_session` scope so the finally block sees it).

- [ ] **Step 2: Write percentiles on session end**

In the `finally` block of `_run_session` (after `store.close()`):

```python
        if latencies_ms:
            import numpy as np
            (Path("results") / "latency_v1.json").write_text(json.dumps({
                "p50_ms": float(np.percentile(latencies_ms, 50)),
                "p95_ms": float(np.percentile(latencies_ms, 95)),
                "n_frames": len(latencies_ms),
                "note": "poll->ws-send server-side; browser adds network+render (localhost demo)",
            }, indent=2))
```

- [ ] **Step 3: Verify via bench test + commit**

Run the bench tests, then check `results/latency_v1.json` exists with p95 well
under 2000 ms:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_ws_replay.py -q
cat results/latency_v1.json
git add src/api/routers/live.py results/latency_v1.json
git commit -m "Latency measured: poll->send percentiles persisted to results/latency_v1.json"
```

---

## Phase C — Demo spine: web app (frontend)

For all frontend tasks: `cd web`, verify with `npm run lint` and `npm run build`
(tsc catches type drift). Manual check via `npm run dev` against the API with
`PM_ALLOW_REPLAY=1` — select a REPLAY pseudo-port in the LIVE tab.

### Task 11: api.ts types for the new protocol

**Files:**
- Modify: `web/src/api.ts`

- [ ] **Step 1: Extend the frame types**

```ts
export interface AlertEvent { kind: 'stable' | 'rule' | 'clear'; fault_type?: string; rule?: string; confidence?: number; elapsed_s: number }
```
Extend `TelemetryFrame`:
- widen `type` union with `'calibrate_progress' | 'calibrate_result'`
- add optional fields: `armed?: boolean; alert_events?: AlertEvent[]; missing_pids?: string[]; t_poll?: number; rows_collected?: number; ok?: boolean; reason?: string; n_windows?: number; path?: string;`

- [ ] **Step 2: Lint, build, commit**

```bash
cd web && npm run lint && npm run build
git add web/src/api.ts
git commit -m "web: frame types for armed/alert events/calibrate protocol"
```

### Task 12: SensorTimeline component + live integration

**Files:**
- Create: `web/src/components/SensorTimeline.tsx`
- Modify: `web/src/components/LiveSession.tsx`

- [ ] **Step 1: Create the component**

```tsx
import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Brush } from 'recharts';
import { T } from '../theme';
import { USEFUL_PIDS } from '../pids';

export interface TimelineAlert { elapsed_s: number; label: string }
interface Props {
  rows: Record<string, number>[];   // [{ elapsed_s, <PID>: value, ... }]
  alerts: TimelineAlert[];
  withBrush?: boolean;              // post-hoc mode: scroll + zoom
}

const DEFAULT_SEL = ['ENGINE_RPM', 'INTAKE_MANIFOLD_PRESSURE',
  'SHORT_TERM_FUEL_TRIM_BANK_1', 'LONG_TERM_FUEL_TRIM_BANK_1', 'COOLANT_TEMPERATURE'];
const COLORS = [T.ACCENT_DATA, T.ACCENT_WARN, T.ACCENT_OK, T.ACCENT_INFO,
  '#b88aff', '#ff9a6b', '#6bd6ff', '#ffd66b'];
const LS_KEY = 'pm_timeline_pids';

export default function SensorTimeline({ rows, alerts, withBrush = false }: Props) {
  const [selected, setSelected] = useState<string[]>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(LS_KEY) ?? 'null');
      if (Array.isArray(s) && s.length) return s;
    } catch { /* corrupted localStorage — fall through */ }
    return DEFAULT_SEL;
  });
  const [cursor, setCursor] = useState<number | null>(null);

  const toggle = (pid: string) => setSelected(prev => {
    const next = prev.includes(pid) ? prev.filter(p => p !== pid) : [...prev, pid];
    localStorage.setItem(LS_KEY, JSON.stringify(next));
    return next.length ? next : prev;   // never allow empty selection
  });

  // Display is per-series min-max normalised (RPM 0-3000 and trims ±25% can't
  // share a raw axis); EXACT raw values live in the readout panel below.
  const data = useMemo(() => {
    const ranges: Record<string, [number, number]> = {};
    for (const pid of selected) {
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) {
        const v = r[pid];
        if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
      ranges[pid] = lo === Infinity ? [0, 1] : [lo, hi === lo ? lo + 1 : hi];
    }
    return rows.map(r => {
      const d: Record<string, number> = { elapsed_s: r.elapsed_s };
      for (const pid of selected) {
        const v = r[pid]; const [lo, hi] = ranges[pid];
        if (Number.isFinite(v)) d[pid] = (v - lo) / (hi - lo);
      }
      return d;
    });
  }, [rows, selected]);

  const readout = useMemo(() => {
    if (cursor == null || !rows.length) return null;
    let best = rows[0];
    for (const r of rows)
      if (Math.abs(r.elapsed_s - cursor) < Math.abs(best.elapsed_s - cursor)) best = r;
    const alert = alerts.find(a => Math.abs(a.elapsed_s - best.elapsed_s) <= 5);
    return { row: best, alert };
  }, [cursor, rows, alerts]);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
        {USEFUL_PIDS.map(pid => (
          <button key={pid} onClick={() => toggle(pid)} style={{
            fontFamily: T.FONT_MONO, fontSize: 9, padding: '3px 7px', cursor: 'pointer',
            border: `1px solid ${selected.includes(pid) ? T.ACCENT_DATA : T.BORDER}`,
            background: selected.includes(pid) ? `${T.ACCENT_DATA}22` : 'transparent',
            color: selected.includes(pid) ? T.TEXT_PRIMARY : T.TEXT_MUTED,
          }}>{pid.replace(/_/g, ' ')}</button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} onClick={(e) => { if (e && e.activeLabel != null) setCursor(Number(e.activeLabel)); }}>
          <XAxis dataKey="elapsed_s" tickFormatter={fmt}
                 stroke={T.TEXT_MUTED} fontSize={10} fontFamily={T.FONT_MONO} />
          <YAxis domain={[0, 1]} hide />
          <Tooltip
            labelFormatter={(v) => `t = ${fmt(Number(v))}`}
            contentStyle={{ background: T.BG_SURFACE, border: `1px solid ${T.BORDER}`, fontFamily: T.FONT_MONO, fontSize: 10 }}
          />
          {selected.map((pid, i) => (
            <Line key={pid} dataKey={pid} dot={false} strokeWidth={1.5}
                  stroke={COLORS[i % COLORS.length]} isAnimationActive={false} />
          ))}
          {alerts.map((a, i) => (
            <ReferenceLine key={i} x={a.elapsed_s} stroke={T.ACCENT_ALERT}
                           strokeDasharray="4 3"
                           label={{ value: a.label, fill: T.ACCENT_ALERT, fontSize: 9, position: 'top' }} />
          ))}
          {cursor != null && <ReferenceLine x={cursor} stroke={T.TEXT_PRIMARY} />}
          {withBrush && <Brush dataKey="elapsed_s" height={18} travellerWidth={8}
                               stroke={T.ACCENT_DATA} tickFormatter={fmt} />}
        </LineChart>
      </ResponsiveContainer>

      {readout && (
        <div style={{ marginTop: 8, padding: 10, border: `1px solid ${T.BORDER}`, background: T.BG_SURFACE }}>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, marginBottom: 6 }}>
            SENSORS @ {fmt(readout.row.elapsed_s)}
            {readout.alert && <span style={{ color: T.ACCENT_ALERT }}> — ALERT: {readout.alert.label}</span>}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 4 }}>
            {USEFUL_PIDS.map(pid => (
              <div key={pid} style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_SECONDARY }}>
                {pid.replace(/_/g, ' ')}: <span style={{ color: T.TEXT_PRIMARY }}>
                  {Number.isFinite(readout.row[pid]) ? readout.row[pid].toFixed(1) : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```
Integration note: the hover tooltip shows *normalised* values by design — the
click-readout panel is the raw-value contract. `web/src/pids.ts` doesn't exist
yet — create it:
```ts
export const USEFUL_PIDS = [
  'ENGINE_RPM', 'VEHICLE_SPEED', 'THROTTLE', 'ENGINE_LOAD',
  'COOLANT_TEMPERATURE', 'LONG_TERM_FUEL_TRIM_BANK_1',
  'SHORT_TERM_FUEL_TRIM_BANK_1', 'INTAKE_MANIFOLD_PRESSURE',
  'ACCELERATOR_PEDAL_POSITION_D', 'ACCELERATOR_PEDAL_POSITION_E',
  'COMMANDED_THROTTLE_ACTUATOR', 'INTAKE_AIR_TEMPERATURE',
  'TIMING_ADVANCE', 'CONTROL_MODULE_VOLTAGE',
] as const;
```

- [ ] **Step 2: Integrate into LiveSession**

In `web/src/components/LiveSession.tsx`:
- `const HISTORY_LEN = 600;` (was 300 — 10 min rolling window)
- Add state: `const [alertEvents, setAlertEvents] = useState<TimelineAlert[]>([]);`
- In the telemetry branch of the frame handler, after `setPidHistory`:
```ts
          if (f.alert_events?.length) {
            setAlertEvents(prev => [...prev,
              ...f.alert_events!.filter(e => e.kind !== 'clear')
                .map(e => ({ elapsed_s: e.elapsed_s, label: e.fault_type ?? e.rule ?? 'alert' }))]);
          }
```
- Render `<SensorTimeline rows={pidHistory} alerts={alertEvents} />` where
  `<PidStrip …>` currently renders (keep PidStrip's import removed; the file
  itself stays — Results may still use it).
- Reset `setAlertEvents([])` inside `disconnect()`.
- Armed badge: where StatusBanner renders, add above it:
```tsx
      {frame && frame.armed === false && (
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_WARN,
                      border: `1px solid ${T.ACCENT_WARN}`, padding: '6px 10px', marginBottom: 8 }}>
          MONITORING (UNCALIBRATED) — calibrate this car to arm fault alerts
        </div>
      )}
```
- Missing PIDs: under the warnings list, when `frame?.missing_pids?.length`:
```tsx
      {!!frame?.missing_pids?.length && (
        <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED }}>
          ECU does not report: {frame.missing_pids.join(', ')}
        </div>
      )}
```

- [ ] **Step 3: Lint, build, manual check, commit**

```bash
cd web && npm run lint && npm run build
```
Manual: API server with `PM_ALLOW_REPLAY=1`, LIVE tab → REPLAY demo_fuel_system →
timeline draws, an alert line appears around the fault onset, clicking shows the
readout with raw values.
```bash
git add web/src/components/SensorTimeline.tsx web/src/pids.ts web/src/components/LiveSession.tsx
git commit -m "web: SensorTimeline — selectable sensors vs time, alert markers, exact-value readout"
```

### Task 13: CalibrationCard on CarPage

**Files:**
- Create: `web/src/components/CalibrationCard.tsx`
- Modify: `web/src/pages/CarPage.tsx` (overview tab)

- [ ] **Step 1: Create the component**

```tsx
import { useEffect, useRef, useState } from 'react';
import { listSerialPorts, openLiveSocket, type Car, type SerialPort, type TelemetryFrame } from '../api';
import { T } from '../theme';

type Phase = 'idle' | 'recording' | 'fitting' | 'done' | 'rejected';
interface Props { car: Car; onCalibrated: () => void }

export default function CalibrationCard({ car, onCalibrated }: Props) {
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [port, setPort] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [rows, setRows] = useState(0);
  const [message, setMessage] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    listSerialPorts().then(ps => { setPorts(ps); if (ps.length) setPort(ps[0].device); }).catch(() => {});
  }, []);

  const start = () => {
    setPhase('recording'); setRows(0); setMessage('');
    const ws = openLiveSocket((f: TelemetryFrame) => {
      if (f.type === 'calibrate_progress') setRows(f.rows_collected ?? 0);
      else if (f.type === 'calibrate_result') {
        if (f.ok) { setPhase('done'); setMessage(`${f.n_windows} windows`); onCalibrated(); }
        else { setPhase('rejected'); setMessage(f.reason ?? 'rejected'); }
        wsRef.current?.close(); wsRef.current = null;
      } else if (f.type === 'error') {
        setPhase('rejected'); setMessage(f.message ?? 'connection error');
        wsRef.current?.close(); wsRef.current = null;
      }
    }, () => { if (phase === 'recording') setPhase('idle'); wsRef.current = null; });
    ws.onopen = () => ws.send(JSON.stringify({ action: 'connect', port, car_id: car.id, mode: 'calibrate' }));
    wsRef.current = ws;
  };

  const finish = () => { setPhase('fitting'); wsRef.current?.send(JSON.stringify({ action: 'finish_calibration' })); };

  const calibrated = !!car.baseline_normalizer_path;
  const mins = Math.floor(rows / 60), secs = rows % 60;

  return (
    <div style={{ border: `1px solid ${T.BORDER}`, padding: 16, marginBottom: 16, background: T.BG_SURFACE }}>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED, letterSpacing: '0.12em', marginBottom: 8 }}>
        BASELINE CALIBRATION
      </div>
      <div style={{ fontFamily: T.FONT_MONO, fontSize: 12, marginBottom: 10,
                    color: calibrated ? T.ACCENT_OK : T.ACCENT_WARN }}>
        {calibrated ? '✓ CALIBRATED — fault alerts armed' : '○ NOT CALIBRATED — fault alerts disarmed'}
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select value={port} onChange={e => setPort(e.target.value)}
                  style={{ background: T.BG_BASE, color: T.TEXT_PRIMARY, border: `1px solid ${T.BORDER}`,
                           fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 8px' }}>
            {ports.map(p => <option key={p.device} value={p.device}>{p.description}</option>)}
          </select>
          <button onClick={start} style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 14px',
                  background: T.ACCENT_DATA, color: T.BG_BASE, border: 'none', cursor: 'pointer' }}>
            {calibrated ? 'RE-CALIBRATE' : 'CALIBRATE'}
          </button>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 10, color: T.TEXT_MUTED }}>
            engine warm · ~5 min of normal driving
          </span>
        </div>
      )}

      {phase === 'recording' && (
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_PRIMARY }}>
            ● RECORDING {mins}:{String(secs).padStart(2, '0')} ({rows} rows)
          </span>
          <button onClick={finish} disabled={rows < 240}
                  style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '6px 14px', cursor: 'pointer',
                           background: rows < 240 ? T.BG_BASE : T.ACCENT_OK,
                           color: rows < 240 ? T.TEXT_MUTED : T.BG_BASE, border: `1px solid ${T.BORDER}` }}>
            {rows < 240 ? `FINISH (need ${240 - rows}s more)` : 'FINISH & FIT'}
          </button>
        </div>
      )}

      {phase === 'fitting' && <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.TEXT_SECONDARY }}>FITTING…</span>}
      {phase === 'done' && <span style={{ fontFamily: T.FONT_MONO, fontSize: 12, color: T.ACCENT_OK }}>✓ SAVED — {message}</span>}
      {phase === 'rejected' && (
        <div>
          <div style={{ fontFamily: T.FONT_MONO, fontSize: 11, color: T.ACCENT_ALERT, marginBottom: 6 }}>REJECTED: {message}</div>
          <button onClick={() => setPhase('idle')} style={{ fontFamily: T.FONT_MONO, fontSize: 11, padding: '4px 10px', cursor: 'pointer' }}>TRY AGAIN</button>
        </div>
      )}
    </div>
  );
}
```
(240 rows = 4 min minimum before FINISH unlocks — under 20 windows the backend
guard would reject anyway; the gate just saves the user a wasted fit.)

- [ ] **Step 2: Mount in CarPage overview tab**

In `web/src/pages/CarPage.tsx`, inside the `tab === 'overview'` render branch
(read the file to find it), add at the top:
```tsx
<CalibrationCard car={car} onCalibrated={() => getCar(id).then(setCar)} />
```
with the import `import CalibrationCard from '../components/CalibrationCard';`.

- [ ] **Step 3: Lint, build, manual check, commit**

Manual: with `PM_ALLOW_REPLAY=1` + `PM_REPLAY_FAST=1`, calibrate against the
REPLAY Yaris pseudo-port → progresses → FINISH → "✓ CALIBRATED", car refreshes,
LIVE tab now shows armed (no uncalibrated badge).
```bash
cd web && npm run lint && npm run build
git add web/src/components/CalibrationCard.tsx web/src/pages/CarPage.tsx
git commit -m "web: per-car calibration flow with guardrail verdicts"
```

### Task 14: Post-hoc timeline on the recording detail view

**Files:**
- Modify: `src/api/routers/recordings.py` (new rows endpoint)
- Modify: `web/src/api.ts`, the recording-detail rendering (in `web/src/pages/Results.tsx` or wherever `RecordingDetail` is displayed — locate with `grep -rn "RecordingDetail" web/src/pages/`)

- [ ] **Step 1: Rows endpoint**

In `src/api/routers/recordings.py`, copy the auth/db dependency pattern from the
existing `get_recording` handler (line ~127) and add:

```python
@router.get("/recordings/{recording_id}/rows")
def get_recording_rows(recording_id: int, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    # IMPORTANT: before writing this, read get_recording (line ~127 in this
    # file) and mirror its EXACT dependency names and its lookup + 404 +
    # ownership-check lines — auth helpers may be named differently here.
    rec = _get_owned_recording(db, user, recording_id)  # ← replace with the real pattern
    if not rec.adapted_csv_path or not Path(rec.adapted_csv_path).exists():
        raise HTTPException(status_code=404, detail="No adapted CSV for this recording.")
    df = pd.read_csv(rec.adapted_csv_path)
    stride = max(1, len(df) // 1200)   # cap payload ~1200 points for charting
    pid_cols = [p for p in USEFUL_PIDS if p in df.columns]
    rows = []
    for i in range(0, len(df), stride):
        r = {"elapsed_s": int(i)}
        for p in pid_cols:
            v = df[p].iloc[i]
            r[p] = None if pd.isna(v) else float(v)
        rows.append(r)
    return {"rows": rows, "stride_s": stride, "n_total": len(df)}
```

- [ ] **Step 2: Frontend fetch + render**

In `web/src/api.ts` add:
```ts
export interface RecordingRows { rows: Record<string, number | null>[]; stride_s: number; n_total: number }
export const getRecordingRows = (id: number) => fetchJson<RecordingRows>(`/recordings/${id}/rows`);
```
In the recording-detail view, fetch rows alongside the existing detail and render:
```tsx
<SensorTimeline
  rows={rowsData.rows.map(r => ({ ...r, elapsed_s: r.elapsed_s as number })) as Record<string, number>[]}
  alerts={detail.result ? detail.result.windows
    .filter((w, i, ws) => w.label !== 'healthy' && w.label !== 'cold_start'
                          && (i === 0 || ws[i - 1].label !== w.label))
    .map(w => ({ elapsed_s: w.elapsed_s, label: w.label })) : []}
  withBrush
/>
```
(Markers = label *transitions* into a fault, so a 300-window fault region draws
one line at onset, not 300.)

- [ ] **Step 3: Lint, build, manual check, full pytest, commit**

Manual: open a recording with a fault (upload one of the `data/demo/*.csv` via
ADD RECORDING if history is empty) → timeline shows the trace, onset marker at
~2 min, Brush zooms, clicking shows raw sensor values at any second.
```bash
cd web && npm run lint && npm run build
cd .. && ./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/api/routers/recordings.py web/src/api.ts web/src/pages/
git commit -m "Post-hoc sensor timeline: rows endpoint + brush/zoom + onset markers"
```

---

## Phase D — Timeboxed metrics push (HARD STOP: 13 June EOD)

Keep a journal table at the bottom of this file — one row per attempt:
`| attempt | change | macro-F1 | fuel_precision | tps_recall | §10 mock recall | yaris healthy frac | kept? |`

Regression guards after EVERY rebuild (both must pass to keep a change):
```bash
./.venv/Scripts/python.exe scripts/eval_real_fault.py data/real_faults/mock/mock_lean_fault.csv --fault-from 120 --fault-to 400 --out results/real_fault_eval/mock_lean_fault_v1.json
./.venv/Scripts/python.exe scripts/eval_real_fault.py data/real_faults/ahmed/ahmed_drive_20260602.csv --normalizer models/yaris_2014_normalizer.pkl --out results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json
./.venv/Scripts/python.exe -m scripts.acceptance_healthy_drive results/real_fault_eval/ahmed_drive_20260602_v3_yaris_baseline.json
```
Guard pass = §10 `fault_interval.recall ≥ 0.60` AND acceptance prints PASS.

### Task 15: F1 lever 1 — widen the ambiguous-ramp skip

**Files:**
- Modify: `src/features/dataset_builder.py:197`

- [ ] **Step 1: Make the change**

```python
            # Skip HALF the ramp, not a quarter: windows over the early ramp
            # carry a fault label but a near-healthy signature — they trained
            # the fuel_system black hole (precision 0.457, absorbing healthy/
            # air/TPS windows). A <50%-developed fault is below the diagnostic
            # thresholds the severity scales cite, so excluding it is honest.
            fault_start = params.onset_idx + max(1, params.ramp_len // 2)
```

- [ ] **Step 2: Rebuild + evaluate**

```bash
./.venv/Scripts/python.exe -m scripts.rebuild_all
```
Read macro-F1 + per-class from the console / `results/xgb_classifier_v1_results.json`.
Run both regression guards. Fill the journal row.

- [ ] **Step 3: Decision gate**

- macro-F1 ≥ 0.80 AND guards pass → keep, commit, SKIP Task 16.
- Improved but < 0.80 → keep, proceed to Task 16.
- Worse or guards fail → revert (`git checkout -- src/features/dataset_builder.py`, rebuild), proceed to Task 16.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
git add -A src/features/dataset_builder.py data/synthetic models results
git commit -m "Dataset: skip half the injection ramp — ambiguous early-ramp windows out (macro-F1 <old>-><new>)"
```

### Task 16: F1 lever 2 — fuel_system sample weight (only if still < 0.80)

**Files:**
- Modify: `src/models/xgb_classifier.py` (the `train()` function — read it first)

- [ ] **Step 1: Add a fuel down-weight**

In `train()`, before `clf.fit(...)`:
```python
    # fuel_system over-fires (precision 0.46 at plan time): down-weight its
    # samples so the boundary retreats toward higher-confidence fuel windows.
    from src.features.dataset_builder import LABEL_TO_ID
    fuel_id = LABEL_TO_ID["fuel_system"]
    sample_weight = np.where(y_train == fuel_id, _FUEL_WEIGHT, 1.0)
```
and pass `sample_weight=sample_weight` to `clf.fit`. Define `_FUEL_WEIGHT = 0.6`
at module top. (Verify the actual variable name for training labels inside
`train()` — adjust `y_train` to match.)

- [ ] **Step 2: Sweep 0.5 / 0.6 / 0.7**

Rebuild + guards per value; journal each. Keep the best macro-F1 whose
fuel_system precision ≥ 0.60 and guards pass; if none beats the Task-15 state,
revert this change entirely.

If macro-F1 is STILL < 0.80 after both levers: stop here and report the journal
to Adam & Ahmed — do not invent new features inside the timebox. 0.797→0.80 is
not worth a destabilised pipeline two days before the defense.
(Also verify the `LABEL_TO_ID` import source and the training-label variable
name by reading `src/models/xgb_classifier.py::train` before editing.)

- [ ] **Step 3: Commit (or revert) with the journal updated**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
git add -A src/models data/synthetic models results
git commit -m "Classifier: fuel_system sample weight <W> — precision <old>-><new>, macro-F1 <new>"
```

### Task 17: air_system forecaster MAE — idle-gate or rescope

**Files:**
- Modify: `src/models/forecaster.py` (`_train_one`, after the split at ~line 80)

- [ ] **Step 1: Gate air_system to idle-observable windows**

```python
    if fault_type == "air_system":
        # Physics gate (mirrors _AIR_IDLE_LOAD_MAX in severity.py): a
        # speed-density vacuum leak is only observable at idle/low load —
        # off-idle windows force the target to 0 and dominate the MAE with
        # a value the formula defines away rather than predicts.
        m_train = train_df["ENGINE_LOAD__mean"].to_numpy(dtype=float) <= 40.0
        m_test = test_df["ENGINE_LOAD__mean"].to_numpy(dtype=float) <= 40.0
        train_df, test_df = train_df[m_train], test_df[m_test]
        log.info("  air_system idle gate: train %d, test %d windows", len(train_df), len(test_df))
```
Place BEFORE `train_norm = norm.transform(train_df)`.

- [ ] **Step 2: Rebuild, read air MAE**

```bash
./.venv/Scripts/python.exe -m scripts.rebuild_all
```
- MAE ≤ 15% → keep; note in `results/forecaster_v1_results.json` it's the
  idle-gated population, and add one sentence to the module docstring.
- Still > 15% → revert and instead add the honest rescope comment next to
  `_COMMIT_LIMIT` in `forecaster.py` (same precedent as TPS's 35%):
```python
    # air_system: severity is idle-gated by physics (speed-density leak washes
    # out off-idle), so most windows have a forced-zero target; the global MAE
    # overstates error on the population where the fault is observable.
```

- [ ] **Step 3: Full suite, guards, commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
git add src/models/forecaster.py models results
git commit -m "air_system forecaster: idle-gated training population (MAE <old>-><new>)"
```

### Task 18: Model freeze (13 June EOD)

- [ ] **Step 1: Final rebuild + full eval battery**

```bash
./.venv/Scripts/python.exe -m scripts.rebuild_all
./.venv/Scripts/python.exe -m pytest tests/ -q
```
Plus both regression guards from the Phase-D preamble.

- [ ] **Step 2: Update the README headline numbers**

Open `README.md`, find the headline metrics section, update macro-F1, MAE table,
§10 recall, latency p95 from the results JSONs (never from memory).

- [ ] **Step 3: Commit + tag**

```bash
git add -A models results data/synthetic README.md
git commit -m "MODEL FREEZE for 15 June defense — all headline numbers final"
git tag defense-freeze-2026-06-13
```

---

## Phase E — Demo readiness

### Task 19: Demo script + Skoda session checklists **[HUMAN]**

**Files:**
- Create: `docs/DEMO_SCRIPT.md`

- [ ] **Step 1: Write docs/DEMO_SCRIPT.md**

```markdown
# Defense Demo Script — 15 June 2026

## A. First Skoda session checklist (run on 11-12 June, NOT demo day)
1. Laptop + ELM327 in the Skoda, engine running, API server up (`uvicorn src.api.main:app`), web app up (`cd web && npm run dev`).
2. LIVE tab → select COM port → CONNECT. **Write down `missing_pids` from the UI.**
   If ACCELERATOR_PEDAL_POSITION_D/E are missing: TPS detection is degraded on
   this car — note it for the Q&A, the system says so on screen by design.
3. Drive until fully warm (gauge mid-band), then CarPage → CALIBRATE → ~5 min
   normal driving → FINISH & FIT. Expect "✓ CALIBRATED". If REJECTED, the
   reason names the failed guard — fix and repeat.
4. Verification drive (≥ 10 min healthy): LIVE tab, confirm NO stable alert,
   timeline stays calm. Save the session dir name (data/app/live_sessions/...).
5. Back home: run the acceptance script over the session if desired.

## B. Demo-day click path (~8 min)
1. Login → Garage (three cars visible: Etios — training, Yaris — validation, Skoda — live).
2. Skoda CarPage → overview: point at "✓ CALIBRATED" and explain per-vehicle
   baselines (the torque-wrench-zeroing analogy).
3. LIVE tab → connect → telemetry + timeline streaming. Talking points:
   armed badge, SHAP panel, forecast columns.
4. Fault story: open HISTORY → demo_fuel_system recording → post-hoc timeline:
   onset marker at 2:00, click it → exact sensor values at that second, LTFT
   climbing while STFT hands off.
5. (Optional, only if rehearsed and Ahmed approves) mods-in vacuum-hose moment
   with mark_leak; otherwise show the Yaris §10/acceptance results instead.

## C. Fallback drill (rehearse this TWICE on 14 June)
- Trigger: no telemetry 15 s after connect, or adapter error banner.
- Action: disconnect → port dropdown → "REPLAY — <rehearsal session>" →
  CONNECT. Same screen, recorded data. Keep talking; mention it's the morning's
  recorded session replaying through the identical pipeline. (Requires server
  started with PM_ALLOW_REPLAY=1 on demo day — put it in the launch command.)
- Launch commands (put on a sticky note):
  - `PM_ALLOW_REPLAY=1 uvicorn src.api.main:app --port 8000`  (PowerShell: `$env:PM_ALLOW_REPLAY="1"; uvicorn src.api.main:app --port 8000`)
  - `cd web; npm run dev`

## D. Rehearsal checklist (14 June, with the car)
- [ ] Full click path B end-to-end, timed.
- [ ] Fallback drill C, twice.
- [ ] Laptop power settings: no sleep, no updates pending.
- [ ] data/ rehearsal session committed so REPLAY has fresh material.
- [ ] Latency p95 from results/latency_v1.json noted on the slide.
```

- [ ] **Step 2: Commit**

```bash
git add docs/DEMO_SCRIPT.md
git commit -m "Demo script: click path, Skoda session checklist, fallback drill"
```

### Task 20 **[HUMAN]**: Skoda calibration + verification drives

Not executable by the agent. Adam/Ahmed follow `docs/DEMO_SCRIPT.md` §A with
the car (target 11-12 June). The executor's only role: after the drives, run
the acceptance script over the verification session and report:

```bash
./.venv/Scripts/python.exe scripts/eval_real_fault.py data/app/live_sessions/<verification-session>/rows.csv --normalizer models/car_<skoda-id>_normalizer.pkl --out results/real_fault_eval/skoda_verification_v1.json
./.venv/Scripts/python.exe -m scripts.acceptance_healthy_drive results/real_fault_eval/skoda_verification_v1.json
```
Expected: `ACCEPTANCE: PASS`. If FAIL — STOP, report the label counts and the
top SHAP features from the result JSON; do not improvise fixes after freeze.

---

## Phase-D journal

| attempt | change | macro-F1 | fuel_precision | tps_recall | §10 mock | yaris healthy | kept? |
|---|---|---|---|---|---|---|---|
| baseline | (plan time) | 0.7974 | 0.457 | 0.612 | 0.9655 | (fill at Task 0) | — |
