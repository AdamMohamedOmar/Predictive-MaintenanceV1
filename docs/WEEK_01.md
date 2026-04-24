# Week 1 — Foundation

**Dates:** Mon 27 Apr – Sun 3 May 2026
**Hour budget:** ~15 per person (Fri 1 May = Labour Day, both off)
**Theme:** Set up everything that makes Week 2 work on Monday morning.

---

## Goal of the week

By Sunday night, the project repo exists on GitHub, both team members can clone it and run a trivial test, carOBD data is downloaded and explored in a notebook, and the ELM327 adapter is ordered.

---

## Pre-flight check (Monday morning)

- [ ] Charter v1.1 reviewed and both team members agree on scope
- [ ] GitHub accounts exist for both, 2FA enabled
- [ ] Python 3.11 installable on both laptops (confirm, don't install yet)
- [ ] At least one of you has admin rights on the laptop that will run the live demo

---

## Daily tasks

### Monday 27 Apr — Repo + environment (5h)

- **Both, 1h:** Read through `CHARTER.md` together one more time. Any disagreement is flagged now, not in Week 3.
- **One person, 2h:** Create private GitHub repo. Initial structure:
  ```
  /src          (empty with __init__.py)
  /notebooks    (empty)
  /data         (gitignored except README)
  /tests        (empty)
  /docs         (CHARTER.md, PLAN.md, WEEK_*.md go here)
  README.md
  requirements.txt
  .gitignore
  ```
- **Other person, 2h:** In parallel, install Python 3.11 on their laptop, create a venv, install pandas + numpy + jupyter. Verify `jupyter notebook` opens. Push a "hello from Adam" or "hello from Ahmed" file to prove access works.
- **Both, end of day:** Brief sync — does every path above actually work? If either laptop can't run jupyter, fix it before Tuesday.

### Tuesday 28 Apr — Dependencies locked, carOBD explored (4h)

- **Both, 2h pair:** Write `requirements.txt` with exact pins (not `>=`). Include everything from the charter Section 13 stack. Install on both laptops. If install fails on one, debug together — this is where Windows-specific DLL issues show up for PyTorch and we'd rather hit them now.
  - Pins to use (verified stable combos as of April 2026):
    ```
    python==3.11.*
    pandas==2.2.3
    numpy==1.26.4
    scikit-learn==1.4.2
    xgboost==2.0.3
    torch==2.3.0
    shap==0.45.0
    obd==0.7.2
    pyserial==3.5
    streamlit==1.35.0
    onnxruntime==1.17.3
    pytest==8.2.0
    jupyter==1.0.0
    ```
  - Note: confirm these resolve together before committing; if pip complains, loosen the offending pin and note it in README.
- **One person, 2h:** Clone carOBD from GitHub, copy CSVs into `/data/raw/carOBD/` (which is gitignored). Open one of the `drive*.csv` files in a notebook, plot ENGINE_RPM, VEHICLE_SPEED, and COOLANT_TEMPERATURE over time. Sanity check: do they look like a real car drive?

### Wednesday 29 Apr — carOBD deep dive (4h)

- **Both, 3h pair:** Notebook `notebooks/01_data_exploration.ipynb`. For each file type in carOBD (idle, drive, live, ufpe, long), answer:
  - How many rows (= seconds of data)?
  - Are all 27 PIDs actually populated, or are some always NaN?
  - What does healthy baseline look like per PID? Mean, std, min, max.
  - Are there any PIDs with suspicious constant values or wild outliers?
- **One person, 1h:** Start a document `docs/DATA_NOTES.md` that captures findings. This becomes input to Week 2's injection engine design and eventually feeds into the thesis's dataset chapter.

**⚠ Flag to watch:** If any of the 6 "signature PIDs" we need for the fault taxonomy (MAF/MAP, O2/STFT/LTFT, fuel trims, coolant temp, throttle, accel pedal) turn out to be absent or unreliable in carOBD, that's a scope-breaking discovery. Raise it immediately — we may need to drop a fault class or redefine it.

### Thursday 30 Apr — ELM327 ordered, project scaffolding (3h)

- **Both, 1h:** Pick the specific ELM327 adapter. Prioritize reliability over price (see charter Section 16 warnings about sub-100 EGP clones). Target: genuine or known-good USB adapter. Place the order. Update `CHARTER.md` Section 16 with actual price, bump to v1.2.
- **Both, 2h pair:** Project scaffolding commit. Empty module skeletons in `/src`:
  ```
  /src
    __init__.py
    config.py              # paths, constants
    data_loading.py        # carOBD CSV reader, empty stub
    injection/             # Week 2 lives here
      __init__.py
    features/              # Week 3 lives here
      __init__.py
    models/                # Week 3-6 lives here
      __init__.py
  /tests
    test_data_loading.py   # one trivial test that loads a CSV
  ```
  Add a `pytest.ini`. Verify `pytest` runs and the one test passes.

### Friday 1 May — Labour Day, OFF

- No work scheduled. If anyone does work, it's bonus and not in the plan.

### Saturday 2 May — README skeleton, small cleanup (2h)

- **One person, 2h:** Write the initial README.md:
  - Project title (from charter)
  - One-paragraph description (can lift from charter Section 1)
  - Installation instructions (just `pip install -r requirements.txt` for now)
  - How to run the data exploration notebook
  - Link to CHARTER.md, PLAN.md
  - "Reproducibility" section stubbed — fill in as project grows

### Sunday 3 May — Weekly review (1h)

- **Both, 1h:** Run the Week 1 "Definition of Done" checklist below. If anything's missing, write a rollover item into WEEK_02.md's Monday block — don't silently skip it.

---

## Concrete deliverables

- Private GitHub repo, initial commit from both authors
- `requirements.txt` locked, installable on both laptops
- `notebooks/01_data_exploration.ipynb` with plots and findings
- `docs/DATA_NOTES.md` summarizing per-PID observations
- ELM327 ordered, charter v1.2 with real price
- `/src` and `/tests` scaffolding with one passing test
- `README.md` v0.1

---

## Definition of Done

- [ ] Both team members can run `pytest` on the repo and see at least one green test
- [ ] Both team members have successfully loaded a carOBD CSV in a notebook on their own laptop
- [ ] `docs/DATA_NOTES.md` contains a table of mean/std per PID across healthy files
- [ ] ELM327 order confirmation email exists; expected delivery date is logged in `docs/DATA_NOTES.md`
- [ ] No signature PID from the fault taxonomy is missing from carOBD (or if one is, it's flagged and the taxonomy is amended in a charter PR)

---

## Week-specific risks

| Risk | What happens if it hits | Mitigation |
|---|---|---|
| `requirements.txt` doesn't resolve cleanly on Windows | Tuesday is burned debugging | Loosen the worst-offending pin, note it in README, move on. PyTorch is the most likely offender. |
| carOBD data turns out to be lower quality than expected | Fault taxonomy may need trimming | Document missing/bad PIDs in DATA_NOTES.md. Raise as charter amendment if a class needs to drop. |
| ELM327 expected delivery > 4 weeks | Week 6 Skoda baseline is at risk | Order a backup from a different seller. Worst case: R2 fallback from charter — pre-recorded Skoda session. |

---

## Handoff to Week 2

Week 2 needs:
- A working dev environment (check)
- A clean understanding of what healthy carOBD data looks like per PID (from DATA_NOTES.md)
- The `/src/injection/` empty package ready to be filled in
- An agreed list of which 10–15 PIDs (out of 27) are the "useful" ones — trimming noise reduces feature-pipeline pain later
