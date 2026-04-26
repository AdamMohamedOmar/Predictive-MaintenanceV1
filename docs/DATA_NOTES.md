# Data Notes — carOBD

> Living document. Updated as the project's understanding of the dataset evolves.
> Final version of this content becomes Chapter 3 of the thesis book.

## Source

The carOBD dataset is a public OBD-II logging dataset published by Eron J. Maranhão
alongside his master's thesis. Available at: https://github.com/eron93br/carOBD

- **Vehicle:** Toyota Etios (2014, 1.5L)
- **Logger:** Carloop (open-source ELM327-compatible development kit with cellular connectivity)
- **Sample rate:** 1 Hz across 27 PIDs
- **License:** Public, with author requesting citation if used. Citation tracked in the project's `references.bib`.

## File categories

The 129 CSV files are grouped by recording context (described in the upstream README):

| Prefix  | Meaning                                | Files in dataset |
|---------|----------------------------------------|------------------|
| `idle*` | Engine on, vehicle parked              | 47 |
| `drive*`| Highway-style driving                  | 13 |
| `live*` | Specific recurring trip (work → home)  | 39 |
| `ufpe*` | Low-speed driving on university campus | 18 |
| `long*` | Long trips                             | 12 |

All files contain the same 27-column header schema (verified Week 1).

## Schema and column-name normalization

The raw CSVs have:

- A trailing ` ()` on every column name (an artefact of the logging firmware)
- One typo: `ENGINE_RUN_TINE ()` instead of the intended `ENGINE_RUN_TIME`
- Several names that diverge from the upstream README's prose (e.g. `PEDAL_D` for what the README calls `ACCELERATOR_PEDAL_POSITION_D`)

All consumers in this project use the cleaned names produced by
`src.data_loading.load_carobd_csv()`. The rename map in that module is the single
source of truth.

## Data-quality findings (Week 1 audit, 2026-04-27)

A column-by-column audit was performed against physically plausible bounds for each
PID. Two distinct issues were found.

### Issue 1: PIDs that are constant across the entire dataset

| PID                              | Value across files     | Cause |
|----------------------------------|------------------------|-------|
| `FUEL_AIR_COMMANDED_EQUIV_RATIO` | always 0               | PID likely not exposed by the Etios ECU, or not requested correctly by the carloop firmware. |
| `TIME_RUN_WITH_MIL_ON`           | always 0               | MIL never on (healthy car). |
| `DISTANCE_TRAVELED_WITH_MIL_ON`  | always 0               | MIL never on (healthy car). |
| `WARM_UPS_SINCE_CODES_CLEARED`   | always 255             | OBD-II "no data / unsupported" sentinel. |

These four PIDs are dropped by default in `load_carobd_csv()`. They can be retained
for explicit data-quality auditing via `drop_unusable=False`.

**Implication for the project:** the Charter's original Oxygen Sensor fault definition
(Section 6) listed both `SHORT_TERM_FUEL_TRIM_BANK_1` and `FUEL_AIR_COMMANDED_EQUIV_RATIO`
as signature PIDs. With the equivalence ratio unavailable, the O2 fault is detectable
via STFT (primary) with LTFT drift as the slower secondary indicator. This is the
canonical OBD-II observable for O2 sensor health and is sufficient for the project's
classification and forecasting goals. The Charter is not formally amended — this is
a documented implementation detail, not a scope change.

### Issue 2: Cross-file inconsistency in 4 critical signature PIDs

A physical-bounds check across all 129 files revealed three distinct file populations:

| Pattern (Speed \| Coolant \| Timing \| STFT) | File count | Status |
|----------------------------------------------|------------|--------|
| OK \| OK \| OK \| OK                         | 9          | Usable |
| OK \| OK \| BAD \| OK                        | 12         | Timing column out of physical range |
| OK \| OK \| BAD \| BAD                       | 108        | Both timing and STFT columns out of physical range |

In the 120 files marked BAD, the affected columns contain values that are physically
impossible for the named PID (e.g. timing advance values of 30°–776° where the
OBD-II spec bounds timing at ±64°). Vehicle speed and coolant temperature are
within bounds in **all** 129 files, ruling out a simple column-shift hypothesis.

**Suspected cause:** firmware-version inconsistency in the original carOBD
recordings. The author's repository contains two firmware programs
(`discover-pids.ino` and `obd-logger.ino`); some recordings appear to use raw
or differently-decoded values for select PIDs. The exact decoding rule was not
investigated in Week 1 to preserve schedule.

**Decision:** use only the 9 fully-clean files for this project. Documented in
`src.data_loading.USABLE_CAROBD_FILES`. The 9-file working set:

| File         | Rows | Minutes |
|--------------|------|---------|
| `drive1.csv` | 2709 | 45.1    |
| `live5.csv`  | 2384 | 39.7    |
| `live6.csv`  | 2287 | 38.1    |
| `live7.csv`  | 2453 | 40.9    |
| `live8.csv`  | 2413 | 40.2    |
| `live9.csv`  | 2168 | 36.1    |
| `live10.csv` | 1176 | 19.6    |
| `live11.csv` | 1727 | 28.8    |
| `live12.csv` | 1151 | 19.2    |
| **Total**    | **18468** | **307.8** (~5.13 hours) |

**Class-balance note:** 1 of 9 files is "drive" mode; 8 are "live" mode (work-to-home
commute). The training set is therefore heavily weighted toward the "live" regime.
This is documented as a known dataset limitation; mitigation strategies will be
considered in Week 3 (e.g. session-level stratification in cross-validation, weighted
sampling) and revisited at the Week 4 mid-project checkpoint.

**Stretch path:** if at the Week 4 checkpoint the classifier is starved for data,
the 120 BAD files may be reverse-engineered to recover their decoded values, OR
healthy recordings from the Skoda Roomster may supplement the dataset. Charter
amendment would be required only in the latter case.

## Working PID list (12 PIDs)

The 12 PIDs in `src.config.USEFUL_PIDS` are the working feature set, derived from
the Charter Section 6 fault taxonomy plus context PIDs for operating-regime
normalization. Rationale lives inline in `config.py`. This list is locked for
Weeks 2–3 and will be re-evaluated at Week 4.

## ELM327 adapter

[Order status to be updated when the adapter is ordered. Pending team decision
on price tier; Wednesday is the deadline for ordering.]

## Audit reproducibility

The audit that produced the 9-file working set was performed in a Jupyter notebook
during Week 1 Monday. The full audit table for all 129 files (per-file, per-PID
summary statistics) is captured at `docs/data_audit_2026-04-27.txt`. Re-running
the audit on a new dataset version should produce a similar table; if the
populations of file patterns shift significantly, the working set must be
re-derived.

## Change log

| Date       | Change |
|------------|--------|
| 2026-04-27 | Initial audit, 9-file working set established, 4 unusable PIDs dropped. |