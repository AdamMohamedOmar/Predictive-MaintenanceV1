# 8-Week Plan — Overview

**Project:** A Predictive Maintenance Framework for Engine Fault Classification and Early-Stage Forecasting from OBD-II Data
**Team:** Adam, Ahmed
**Start:** Monday 27 April 2026
**Submission deadline:** Monday 15 June 2026
**Week 8 end:** Sunday 21 June 2026 (buffer week after submission)
**Version:** 1.0 (24 April 2026)

---

## How this plan works

This file is the map. Daily work happens in the per-week files: `WEEK_01.md` through `WEEK_08.md`. Open the current week's file on Monday morning, check off tasks as you go, and commit the updated file to `main` at the end of each week.

At the end of each week, you run a **weekly review** (Sunday, ~30 min) that answers three questions:
1. Did we hit this week's "Definition of Done"?
2. What rolled over? Why?
3. Does next week's plan still make sense, or does it need amending?

If a week slips badly, edit the affected WEEK files directly and note the change in the log at the bottom of this file. The plan is a hypothesis; amendments are expected, not a sign of failure.

---

## Calendar (dated)

| Week | Dates | Focus | Key constraint |
|---|---|---|---|
| 1 | Mon 27 Apr – Sun 3 May | Foundation | Fri 1 May = Labour Day (both off) |
| 2 | Mon 4 May – Sun 10 May | Fault injection engine (step) | — |
| 3 | Mon 11 May – Sun 17 May | Classifier baseline (RF) | — |
| 4 | Mon 18 May – Sun 24 May | Classifier final (XGBoost + SHAP) + **checkpoint** | — |
| 5 | Mon 25 May – Sun 31 May | Forecaster baseline | **Eid al-Adha 26–29 May (Tue–Fri)** — 4-day disruption |
| 6 | Mon 1 Jun – Sun 7 Jun | Forecaster final + Skoda baseline recording | ELM327 must have arrived |
| 7 | Mon 8 Jun – Sun 14 Jun | Integration + paper draft | — |
| 8 | Mon 15 Jun – Sun 21 Jun | **SUBMIT MON 15 JUN**, then polish/buffer | Submission day 1, rest is contingency |

The 15 June deadline landing on Monday of Week 8 is genuinely lucky. Treat the submission as happening at end of Week 7 (Sunday 14 June) and use Mon 15 Jun as the final-check-before-upload day. Weeks 7→8 have zero slack; everything upstream has buffer.

---

## Dependency graph (what unblocks what)

```
Week 1: repo, env, carOBD, ELM327 order        ─┐
                                                 │
Week 2: fault injection engine (step mode)     ──┤
              │                                  │
              ├──> Week 3: classifier baseline (RF)
              │                     │
              │                     └──> Week 4: classifier final + checkpoint
              │                                     │
              └──> Week 5: forecaster baseline (ramp mode)
                                   │
                                   ├──> Week 6: forecaster final + Skoda baseline
                                   │                │
                                   └────────────────┴──> Week 7: dashboard, paper draft
                                                                      │
                                                                      └──> Week 8: submit
```

The single most important dependency: **Week 2 must ship a working injection engine** because Weeks 3 *and* 5 both depend on it. If Week 2 slips, both pillars slip.

---

## Hour budget

- ~15 focused hours/week per person, 30 person-hours/week total
- Pair work = hours on each person's ledger (2h pair = 2h each)
- Saturday is light (4h max), Sunday is weekly review only (2h)
- **Week 5 reduced to ~20 person-hours total** because of Eid al-Adha

| Week | Total person-hours | Per-person hours |
|---|---|---|
| 1 | 30 | 15 |
| 2 | 30 | 15 |
| 3 | 30 | 15 |
| 4 | 30 | 15 |
| 5 | 20 | 10 — Eid disruption |
| 6 | 30 | 15 |
| 7 | 35 | 17–18 — paper drafting pushes up |
| 8 | 25 | 12–13 — polish + defense prep |
| **Total** | **230** | **115/person** |

If either of you is consistently spending more than the weekly budget, the plan is wrong and we amend.

---

## Deliverables by week (from charter Section 8)

| Week produced | Deliverable ID | What |
|---|---|---|
| 1 | D1, D11 | Charter committed, README skeleton |
| 2 | D2 | Fault injection engine + tests |
| 3 | D3 (part) | Feature pipeline |
| 4 | D4 | Classifier model + eval notebook |
| 5 | D5 (part) | Forecaster baseline |
| 6 | D5, D7 | Forecaster final, Skoda baseline CSV |
| 7 | D6, D8 (draft), D9 (outline) | Dashboard integrated, paper v1, book outline |
| 8 | D8 (final), D9, D10 | Paper, book, presentation |

---

## Checkpoints

**End of Week 2 — engine-sanity checkpoint.** Before Week 3 begins, the fault injection engine must produce physically plausible output for all 6 classes. If it doesn't, Week 3's classifier work cannot proceed and the schedule is already broken.

**End of Week 4 — mid-project checkpoint** (also defined in charter Section 9). Decision rules:
- If classifier macro-F1 < 0.70 on session-split validation → drop deep-learning classifier comparison, ship XGBoost as final classifier, save time for forecaster.
- If injection engine still needs changes → freeze engine, accept current fault quality, move on.
- If ELM327 hasn't arrived → fall back to pre-recorded Skoda session for demo.

**End of Week 6 — integration-readiness checkpoint.** Forecaster shipped, Skoda baseline recorded, both models exportable. If not, Week 7 paper drafting starts before integration is done (acceptable — paper drafting can happen in parallel with integration polish).

---

## Risk register (from charter Section 10) — active items by week

| Week | Main risks to watch |
|---|---|
| 1 | R2 (ELM327 delay), R8 (env setup blocker) |
| 2 | R5 (injection physics implausible) |
| 3 | R1 (session-level leakage), R5 (injection) |
| 4 | R4 (if forecaster fallback becomes likely) |
| 5 | R4 (forecaster regression underperforms) — + Eid disruption |
| 6 | R2 (ELM327 late → demo fallback), R3 (Skoda baseline too different) |
| 7 | R6 (scope creep in paper), R10 (overclaim) |
| 8 | R9 (defense Q&A prep), R10 (final overclaim check) |

---

## Change log

| Date | Change |
|---|---|
| 24 Apr 2026 | Initial plan v1.0. Week 5 reduced for Eid. Submission on Mon 15 Jun confirmed. |
