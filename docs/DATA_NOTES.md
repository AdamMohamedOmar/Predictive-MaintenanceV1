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

**Implication for the project:** the always-zero `FUEL_AIR_COMMANDED_EQUIV_RATIO`
made the primary O₂-sensor signature unobservable, and the oxygen sensor fault was
formally **dropped from the taxonomy in charter v1.2** (29 May 2026); its deployment
slot was reassigned to the `cold_start` regime class. See `docs/CHARTER.md` §6.

### Issue 2: Trailing-comma column shift (RESOLVED — was misdiagnosed as "120 unusable files")

The Week 1 audit found three file populations via a physical-bounds check:

| Pattern (Speed \| Coolant \| Timing \| STFT) | File count | Week 1 verdict |
|----------------------------------------------|------------|----------------|
| OK \| OK \| OK \| OK                         | 9          | Usable |
| OK \| OK \| BAD \| OK                        | 12         | "Timing out of range" |
| OK \| OK \| BAD \| BAD                       | 108        | "Timing and STFT out of range" |

The Week 1 interpretation (firmware-version inconsistency in the logger) was
**wrong**. The June 2026 data-integrity investigation found the real cause:

**~120 of the carOBD files end each data row with a trailing comma** (28 fields
against 27 header names). A bare `pd.read_csv()` resolves that mismatch by
silently promoting the first column to the DataFrame index, which shifts **every
column one position left** — `COOLANT_TEMPERATURE` reads fuel-trim values,
`TIMING_ADVANCE` reads catalyst temperatures (hence the impossible 30°–776°
readings), and so on. The data itself was never corrupt; the parse was.

**Fix (all three pieces are load-bearing — see CLAUDE.md "Protected invariants"):**

1. `pd.read_csv(path, index_col=False)` in `src/data_loading.py`,
   `scripts/audit_carobd.py`, and `src/live/replay_source.py` disables the
   index promotion so all 27 columns align in every file.
2. `df.apply(pd.to_numeric, errors="coerce")` in the loader handles isolated
   non-numeric cells (e.g. a stray `' '` in `INTAKE_AIR_TEMPERATURE` in
   `live16.csv`) that would otherwise object-type an entire column.
3. `_assert_physical_bounds()` remains as a misalignment tripwire: a shifted
   file now fails **loudly** at load time instead of entering training as
   scrambled "healthy" data.

**Result: all 129 files parse cleanly and pass the physical-bounds guard.**
The hardcoded 9-file whitelist (`USABLE_CAROBD_FILES`) was removed;
`src.data_loading.list_usable_files()` now validates each file dynamically
against the bounds guard and skips (with a warning) anything that fails.

**Class-balance note (historical):** the frozen v1 models (13 June 2026) were
trained on the original 9-file working set (drive1, live5–live12, ~5.13 h),
which is heavily weighted toward the "live" commute regime. The parse fix
expands the usable pool to all 129 files (idle/drive/live/ufpe/long) for any
future retraining; the frozen artefacts and their headline numbers are
unchanged.

## Working PID list (14 PIDs)

The 14 PIDs in `src.config.USEFUL_PIDS` are the working feature set, derived from
the Charter Section 6 fault taxonomy plus context PIDs for operating-regime
normalization. Rationale lives inline in `config.py`.

## ELM327 adapter

Adapter acquired and used for the Week 6–7 live integration (Skoda Roomster
baseline capture and live dashboard validation). See `scripts/live_discover.py`
for the go/no-go adapter check.

## Audit reproducibility

The bounds audit can be re-run at any time with `scripts/audit_carobd.py`,
which uses the corrected `index_col=False` parse; on the current dataset it
reports all 129 files within physical bounds. (The original Week 1 audit and
its "120 unusable files" conclusion were produced by the same bounds check run
on the misparsed, column-shifted data.)

## Change log

| Date       | Change |
|------------|--------|
| 2026-04-27 | Initial audit, 9-file working set established, 4 unusable PIDs dropped. |
| 2026-06    | Trailing-comma column shift identified as the real cause of the "120 unusable files"; `index_col=False` parse fix + `pd.to_numeric` coercion + physical-bounds guard land in the loader; 9-file whitelist replaced by dynamic `list_usable_files()`; all 129 files usable. |
| 2026-07-05 | Doc updated to reflect the above (Issue 2 rewritten, O₂ note aligned with charter v1.2, PID count corrected to 14). |