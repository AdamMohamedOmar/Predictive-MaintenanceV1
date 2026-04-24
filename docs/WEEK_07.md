# Week 7 — Integration + Paper Draft

**Dates:** Mon 8 Jun – Sun 14 Jun 2026
**Hour budget:** ~17–18 per person (extra hours for paper drafting)
**Theme:** Polish the dashboard. Write the paper. No more model training.

---

## Goal of the week

By Sunday night, the live dashboard is demo-ready, a full paper draft (IEEE style, 4–6 pages) exists, and the thesis book outline is complete with Chapter 1 drafted (lifting from charter). Sunday 14 June is the effective submission day for the paper — Monday 15 June is buffer-check only.

**No more ML experiments this week.** If something feels broken, fix it. If something feels improvable, leave it. Writing and polishing is the job.

---

## Pre-flight check (Monday morning)

- [ ] Week 6 Definition of Done all green
- [ ] Live OBD reader tested in a real car (check)
- [ ] Both models ONNX-exported (check)
- [ ] Dashboard runs end-to-end (may be ugly)
- [ ] `docs/CLASSIFIER_RESULTS.md` and `docs/FORECASTER_RESULTS.md` have final numbers

---

## Daily tasks

### Monday 8 Jun — Dashboard polish (4h)

- **Both, 4h pair:**
  - Clean up layout — Streamlit has good defaults, use them. Sidebar for controls, main area for 3 panels: (1) live sensor plot, (2) classifier prediction with probability bars, (3) forecaster severity + trend
  - SHAP-style contribution display for the current prediction (a small bar chart of top 5 contributing features)
  - Status indicator: "Live OBD connected" or "Replaying recorded file"
  - Graceful error handling: if OBD drops, show last known prediction + warning, don't crash
- Goal for end of day: you could show this to your advisor and not be embarrassed.

### Tuesday 9 Jun — Live integration test + dashboard edge cases (3h)

- **Both, 2h pair, with ELM327 + a real car:**
  - Full live test of the dashboard. Warm up engine, drive around a parking lot, watch the dashboard.
  - Confirm: classifier says "healthy" most of the time (occasional misclassifications are expected on real data; if it's consistently wrong, that's a calibration issue)
  - Confirm: forecaster output is stable and doesn't oscillate wildly
  - Record a screen capture — this becomes a backup demo video if live hardware fails on defense day
- **One person, 1h:** Fix any obvious bugs from the live test. Don't add features. Fix and move on.

### Wednesday 10 Jun — Paper draft v1 (5h)

IEEE-style conference paper, target 4–6 pages in two-column format.

- **Both, 1h pair:** Outline the paper section by section:
  - Abstract (~150 words, write LAST)
  - I. Introduction (1 column)
  - II. Related Work (0.5–1 column) — brief
  - III. Methodology
    - A. Fault injection engine
    - B. Feature pipeline
    - C. Classifier
    - D. Forecaster
  - IV. Experiments
    - A. Dataset
    - B. Evaluation protocol (emphasize session-level split)
    - C. Classifier results (table + confusion matrix + SHAP figures)
    - D. Forecaster results (table + calibration plot)
    - E. Live demonstration
  - V. Discussion (limitations — use honest framing invariants directly)
  - VI. Conclusion + Future Work
  - References

- **Split and write in parallel, 4h each (8h total):**
  - Adam: Introduction, Related Work, Methodology A + B
  - Ahmed: Methodology C + D, Experiments A + B
  - Leave C, D, E of Experiments for Thursday (you need the final figures in the paper ready)

Use IEEEtran.cls template. Any paper writing tool (Overleaf is easiest for two-person collaboration).

### Thursday 11 Jun — Paper draft v1 finished + book outline (5h)

- **Both, 2h pair:** Finish the paper draft:
  - Experiments C/D/E — copy numbers from results docs, embed key figures
  - Discussion section: use the charter's Section 11 honest framing invariants as the skeleton. Each invariant becomes a paragraph: "The forecaster is evaluated on injected, early-stage faults; true run-to-failure validation remains future work."
  - Conclusion + future work
  - Abstract (now you can write it)

- **One person, 2h:** Thesis book outline. Target chapter structure:
  - Ch 1: Introduction & Motivation (lift 60% from charter Sections 1–3)
  - Ch 2: Background (OBD-II, common engine faults, ML for predictive maintenance)
  - Ch 3: Dataset & Fault Injection (lift from charter + DATA_NOTES.md + INJECTION_ENGINE.md)
  - Ch 4: Fault Classification (lift from CLASSIFIER_RESULTS.md + SHAP notebook)
  - Ch 5: Fault Forecasting (lift from FORECASTER_RESULTS.md)
  - Ch 6: Live Demonstration & Cross-Vehicle Considerations (INTEGRATION_NOTES.md)
  - Ch 7: Discussion & Future Work
  - Ch 8: Conclusion
  - Appendices: feature list, hyperparameters, reproducibility instructions

- **Other person, 1h:** Draft Chapter 1 using charter Sections 1–3 as input. First pass is fine.

### Friday 12 Jun — Internal review + paper v2 (4h)

- **Both, 1h:** Read each other's sections. Red pen everything. Specific things to check:
  - Honest framing invariants respected everywhere
  - No claims beyond experimental evidence
  - Numbers match between prose and tables
  - Session-level split is mentioned explicitly in evaluation protocol
  - No mention of misfire detection
  - No overclaim about run-to-failure
- **Both, 3h pair:** Revise paper based on internal review. Produce v2 of the draft.

### Saturday 13 Jun — Paper v3 + presentation skeleton (4h)

- **Both, 2h pair:** One more paper pass. Polish prose, tighten claims, proof-read. Check that figure captions are standalone-readable. Check references format.
- **Both, 2h pair:** PowerPoint skeleton for the defense:
  - Target 20–25 slides for ~20 min presentation
  - Title + team
  - Problem statement (1 slide)
  - Approach overview (1 slide)
  - Fault injection engine (2–3 slides, reuse showcase notebook figures)
  - Classifier (3–4 slides: method, results, SHAP)
  - Forecaster (3–4 slides: method, results, calibration)
  - Live demo (1 slide + live or backup video)
  - Limitations & future work (2 slides — do this honestly)
  - Q&A slide

### Sunday 14 Jun — Final review + SUBMIT (3h)

- **Both, 2h:** Full end-to-end review of the paper. Every figure, every claim, every reference. This is the submission-ready version.
- **Both, 30 min:** Check submission requirements. Format, page limit, file naming, authorship info. Save as PDF.
- **Both, 30 min:** Upload / submit / whatever the delivery mechanism is. If the deadline is digital upload to a department portal, do it tonight. If it's a printed copy on Monday, print tonight.

---

## Concrete deliverables

- Polished dashboard, demo-ready, tested on real vehicle
- Demo backup video (screen recording of the live test)
- Paper draft v3 (submission-ready) in IEEE format, PDF
- Thesis book outline (all chapters) + Chapter 1 draft
- PowerPoint skeleton with outline + placeholder slides

---

## Definition of Done

- [ ] Dashboard runs on a real car, with live OBD, without crashing
- [ ] Backup demo video recorded and stored safely
- [ ] Paper v3 complete, proofread twice, submission-ready PDF exists
- [ ] Paper submitted (or queued for Monday submission)
- [ ] Book outline committed, Chapter 1 drafted
- [ ] Presentation skeleton exists with slide count close to target

---

## Week-specific risks

| Risk | Watch level | What to do |
|---|---|---|
| Dashboard bugs found during Tuesday live test | Medium | Fix the critical ones only. Cosmetic bugs wait for Week 8 buffer. |
| Paper runs over page limit | Medium | Cut related work section first, then tighten methodology prose. Keep experiments and discussion. |
| Disagreement between team members on paper wording | Low but annoying | Escalate to "whichever version is more conservative/honest wins." Don't debate for >15 min on a phrase. |
| Scope creep — "let's add one more experiment" | HIGH | **Rule of the week: NO new experiments.** If a thought crosses your mind, write it as future work in the paper. |

---

## Handoff to Week 8

Week 8 needs:
- Paper submitted Sunday (check) — Monday is buffer only
- Book outline + Ch 1 (check) — rest of book written Week 8
- Presentation skeleton (check) — content filled in Week 8
- Demo working on real hardware (check) — no hardware changes in Week 8

**Week 7 Sunday is effectively the submission deadline in this plan.** Week 8 exists to write the book, finalize the presentation, and prepare for the defense — not to rescue the paper.
