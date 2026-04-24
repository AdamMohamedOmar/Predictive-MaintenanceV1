# Week 2 — Fault Injection Engine (Step Mode)

**Dates:** Mon 4 May – Sun 10 May 2026
**Hour budget:** ~15 per person
**Theme:** Build the single most important piece of infrastructure in the project.

---

## Goal of the week

By Sunday night, `src/injection/` contains a working engine that takes a healthy carOBD recording and a fault specification, and returns a modified recording with per-timestamp labels. Step mode is complete; ramp mode is stubbed for Week 5. All 6 fault classes produce physically plausible output. Unit tests pass.

---

## Pre-flight check (Monday morning)

- [ ] Week 1 Definition of Done checklist all green
- [ ] `docs/DATA_NOTES.md` exists with baseline stats per PID
- [ ] Agreed PID list (10–15 of the 27) committed
- [ ] Both laptops pass `pytest`

---

## Context: what "physically plausible" means

This is the week where the project either earns its credibility or silently loses it. A fault injection that just adds noise to a PID produces garbage data that any classifier can learn to separate trivially — and the paper reviewer will see through it.

The rule: **every fault injection must have a primary effect (the sensor that "breaks") and at least one secondary effect (what the ECU does in response).** For example:

- **Oxygen sensor stuck lean** → primary effect: O₂ reading drops. Secondary effect: ECU sees "lean" and commands more fuel, so short-term fuel trim goes positive. Over minutes, long-term fuel trim follows.
- **MAF under-reporting (vacuum leak)** → primary: intake manifold pressure sensor reads lower than it should for the engine load. Secondary: ECU injects less fuel than needed, engine runs lean, O₂ reports lean, fuel trims compensate positively.

If your injection only does the primary effect, the fuel trim PIDs will stay at their healthy baseline while the O₂ sensor is "broken," and that's a physically impossible combination. A good classifier will use this impossibility as a feature and look brilliant — on synthetic data. On real faults, it'll fail.

---

## Daily tasks

### Monday 5 May — Engine architecture + healthy-data loader (5h)

- **Both, 1h pair:** Whiteboard (or Excalidraw) the injection engine architecture. Target design:
  ```python
  # High-level API
  from src.injection import FaultInjector, FaultSpec, FaultClass

  spec = FaultSpec(
      fault_class=FaultClass.O2_SENSOR_STUCK_LEAN,
      mode="step",           # or "ramp"
      start_time_s=120,      # fault begins at t=120s
      end_time_s=None,       # step mode holds; ramp mode would set this
      target_severity=0.7,   # 0 to 1 per class definition
  )
  injector = FaultInjector()
  modified_df, labels = injector.apply(healthy_df, spec)
  ```
- **One person, 2h:** Implement `src/data_loading.py` properly. Functions:
  - `load_carobd_csv(path) -> pd.DataFrame` with timestamp index
  - `get_healthy_baseline(df, pid_list) -> dict[pid, (mean, std)]`
  - Write tests that pass on real carOBD files.
- **Other person, 2h:** Implement `src/injection/base.py`:
  - `FaultClass` enum with the 6 classes
  - `FaultSpec` dataclass
  - `FaultInjector` abstract base class with `.apply(df, spec)` signature
  - Empty concrete subclasses for each of the 5 fault classes (raise NotImplementedError for now)

### Tuesday 6 May — First fault: O2 sensor (4h)

We implement O2 sensor first because its physics chain is the clearest pedagogically: one sensor breaks, fuel trims respond, everything else stays normal.

- **Both, 3h pair:** Implement `O2SensorFault` in `src/injection/o2_sensor.py`:
  - **Primary effect:** bias the fuel–air equivalence ratio toward lean (value > 1.0) or rich (< 1.0) by an amount scaled by severity
  - **Secondary effect:** short-term fuel trim (STFT) moves opposite the bias (lean bias → positive STFT, ECU trying to add fuel). Use a simple first-order filter to model the lag — ECU doesn't respond instantly.
  - **Tertiary effect:** long-term fuel trim (LTFT) follows STFT with a longer time constant (minutes, not seconds).
- **One person, 1h:** Write tests in `tests/test_o2_injection.py`:
  - Before injection start, all PIDs unchanged
  - After injection, equiv ratio is biased by expected amount
  - After injection, STFT moves in the correct direction
  - After ~5 min, LTFT has shifted toward STFT's new baseline

**⚠ Physics sanity check:** Plot a healthy trip alongside its O2-injected version. Do the fuel trim curves look like they'd come from the same car with a bad sensor? If they look like straight lines or perfect sine waves, something's wrong — real fuel trims are noisy.

### Wednesday 7 May — Remaining 4 faults, parallel implementation (5h)

One person owns each of 4 faults (MAF, fuel, coolant, TPS); the 5th person-hour is pair time for cross-review.

Assign owners at the start of the day — this is one of the few days where Adam and Ahmed work on different files simultaneously.

- **Owner A, 2h:** `src/injection/air_system.py` (MAF drift / vacuum leak)
  - Primary: intake manifold pressure reading biased low
  - Secondary: positive fuel trims (ECU sees less air, should inject less fuel, but because the bias is in the sensor not reality, engine runs lean and O2 demands compensation)
  - Secondary: engine load calculation affected
- **Owner A, 2h:** `src/injection/fuel_system.py` (injector issue)
  - Primary: biased fuel trims directly (because actual fuel delivery doesn't match commanded)
  - Secondary: equivalence ratio shifts

- **Owner B, 2h:** `src/injection/coolant_sensor.py`
  - Primary: coolant temperature reading drifts or sticks
  - Secondary: timing advance affected (cold engine → different timing strategy)
  - Secondary (subtle): intake air temperature relationship becomes odd
- **Owner B, 2h:** `src/injection/tps.py` (throttle position sensor drift)
  - Primary: throttle position reading drifts from accelerator pedal position
  - Secondary: the throttle-vs-pedal ratio becomes the dead giveaway feature

- **Both, 1h pair review:** Read each other's code. Each reviewer specifically checks: does this have a secondary effect? Is the secondary physically correct?

### Thursday 8 May — Tests, labels, healthy class (4h)

- **Both, 2h pair:** Write `tests/test_all_injections.py` — for each fault class:
  - Apply a step injection at t=60s, target severity=0.8
  - Assert the primary PID is changed after t=60s
  - Assert at least one secondary PID is changed after t=60s (with some lag allowed)
  - Assert PIDs not in the fault's signature are unchanged (within noise)
- **One person, 1h:** Implement "healthy" as a no-op fault class. Returns the input unchanged with labels = `FaultClass.HEALTHY` for all timestamps.
- **Other person, 1h:** Label format — decide and document. Recommend a DataFrame column `fault_class` (enum value) + `severity` (float 0–1) per timestamp. This structure is what Week 3's windowing will consume.

### Friday 9 May — Ramp mode stub + visualization notebook (3h)

- **One person, 1h:** Add `mode="ramp"` support to each injector. Ramp = severity linearly interpolates from 0 at `start_time_s` to `target_severity` at `end_time_s`. Don't optimize — just make it work. Week 5 will exercise it properly.
- **Both, 2h pair:** Notebook `notebooks/02_injection_showcase.ipynb`. For each of the 5 fault classes:
  - Load a healthy trip
  - Apply step injection
  - Plot primary PID + affected secondary PID, before and after injection
  - Add a paragraph of text explaining the physics

This notebook is **gold for the thesis book** — it becomes a figure-heavy section of the methodology chapter with minimal rewrite.

### Saturday 10 May — Buffer / polish / documentation (2h)

- If Mon–Fri ran long, Saturday is catch-up time.
- Otherwise: clean code review. Docstrings on every public function. Type hints. Run `pytest` one more time, confirm everything green.
- Start `docs/INJECTION_ENGINE.md` — a design document explaining the physics logic for each class. This becomes thesis content.

### Sunday 10 May — Weekly review + engine-sanity checkpoint (1h)

- **Both, 1h:** Run Week 2 Definition of Done. Specifically: run the visualization notebook and look at every plot. If any fault's secondary effect looks wrong, log it and fix in Week 3 Monday (but flag it in the weekly review note).

---

## Concrete deliverables

- `src/injection/` package with 6 fault classes (5 faults + healthy no-op)
- Step mode fully implemented, ramp mode stubbed
- `tests/test_all_injections.py` with per-fault assertions, all green
- `notebooks/02_injection_showcase.ipynb` with physics plots
- `docs/INJECTION_ENGINE.md` design doc started

---

## Definition of Done

- [ ] `pytest tests/test_all_injections.py` passes
- [ ] Visualization notebook produces a plot for each of 5 faults that "looks like a real car with a broken sensor" — subjective but important; both team members agree
- [ ] Each fault has a documented primary effect AND at least one secondary effect
- [ ] Labels DataFrame format is documented in `docs/INJECTION_ENGINE.md`
- [ ] Healthy no-op class works — returns input unchanged

---

## Week-specific risks

| Risk | What happens if it hits | Mitigation |
|---|---|---|
| Fault physics more complex than expected for one class (likely: fuel system, because injector faults are genuinely subtle) | That class slips to Week 3 | Ship Week 2 with 4 faults + healthy; complete the 5th fault Monday of Week 3 before classifier work. Don't delay the classifier for one stubborn fault. |
| Secondary effects don't look right in visualization | Discovery in Week 4 that the classifier is learning impossible patterns | The visualization notebook on Friday is the checkpoint. If it looks wrong on Friday, spend Saturday fixing it, not polishing. |
| Engine over-engineered for Week 5's ramp needs | Waste of time | Keep ramp mode as a minimal stub. Don't add profile shapes, noise models, or multi-phase ramps until Week 5 demands it. |

---

## Handoff to Week 3

Week 3 needs:
- A working injection engine (check) that can produce labeled windowed datasets on demand
- A documented label format so feature pipeline knows what to consume
- Confidence that the synthetic data is physically credible (via visualization notebook review)
