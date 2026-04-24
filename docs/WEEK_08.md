# Week 8 — Submit, Book, Defense Prep

**Dates:** Mon 15 Jun – Sun 21 Jun 2026
**Hour budget:** ~12–13 per person
**Theme:** Submit Monday. Write the book. Prepare the defense.

---

## Goal of the week

By end of week: paper officially submitted, thesis book complete, presentation polished and dry-run at least twice, defense prep done.

This week has deliberate slack. If Weeks 1–7 all hit their targets, Week 8 is mostly polish and nerves-management. If something slipped, Week 8 absorbs it.

---

## Pre-flight check (Monday morning)

- [ ] Week 7 Definition of Done all green
- [ ] Paper PDF finalized and ready to upload
- [ ] Dashboard still works (run it once, confirm)
- [ ] ELM327 + laptop + charger + backup adapter all accounted for physically

---

## Daily tasks

### Monday 15 Jun — SUBMIT (3h)

- **Both, 1h:** Morning final check on paper. One last proof-read. Check submission system one more time. Submit.
- **Both, 1h:** Email / Slack / however you communicate — confirm submission with advisor. Get written acknowledgment.
- **Both, 1h:** Celebrate, take a breath, then start book writing. The rest of this week is just execution.

### Tuesday 16 Jun — Book writing (4h)

- **Adam, 2h:** Draft Chapter 2 (Background). Short is fine — this is a grad thesis book, not a textbook. 5–8 pages.
- **Ahmed, 2h:** Draft Chapter 3 (Dataset & Fault Injection). Longer — 10–15 pages. Heavy lifting from INJECTION_ENGINE.md + DATA_NOTES.md + Section from charter.

### Wednesday 17 Jun — Book writing continued (4h)

- **Adam, 2h:** Draft Chapter 4 (Fault Classification). 10–12 pages. Lift from CLASSIFIER_RESULTS.md + the paper's methodology section + SHAP analysis. Book version has more detail than paper — longer tables, more figures, deeper discussion.
- **Ahmed, 2h:** Draft Chapter 5 (Fault Forecasting). 10–12 pages. Same treatment.

### Thursday 18 Jun — Book writing + presentation (4h)

- **Adam, 2h:** Draft Chapter 6 (Live Demonstration & Cross-Vehicle). 6–10 pages including dashboard screenshots + Skoda baseline discussion.
- **Ahmed, 2h:** Draft Chapter 7 (Discussion & Future Work) + Chapter 8 (Conclusion). 8–12 pages combined. Strongest-word-for-strongest-word match to the paper's discussion, expanded with more nuance.

### Friday 19 Jun — Presentation content + book polish (4h)

- **Both, 2h pair:** Fill in the presentation content from Week 7's skeleton. Every slide gets real content (text, figures, numbers). Don't worry about aesthetics yet.
- **Both, 2h pair:** Book polish pass:
  - Consistent terminology (fault vs failure vs anomaly — pick one per meaning and stick to it)
  - Consistent notation
  - Cross-references (Ch 4 refs Ch 3 where appropriate)
  - Appendices: feature list, hyperparameters, reproducibility instructions (README content adapted)

### Saturday 20 Jun — Dry run defense + final polish (4h)

- **Both, 1h:** First dry run. Full 20-min presentation, out loud, with slides. Time it. Note awkward transitions.
- **Both, 1h:** Fix the worst 3 slides based on dry run.
- **Both, 1h:** Second dry run. Time it again. Should be closer to 20 min.
- **Both, 1h:** Final book PDF export. Final presentation export. Final README pass — make sure a clean clone + install gets someone to "running the dashboard."

### Sunday 21 Jun — Q&A prep + buffer (3h)

- **Both, 2h:** Anticipate defense questions. Write answers to:
  - "Why didn't you validate on real faults?" → The forecaster is trained on physically-grounded injected faults; run-to-failure validation requires a multi-month fleet study, which is future work. The injection engine models secondary effects correctly (demonstrate with a figure).
  - "Why not more data / different vehicles / deeper models?" → Scope choices given 8 weeks. See limitations section.
  - "How does this compare to commercial systems like [X]?" → Commercial OBD diagnostic tools use threshold-based DTCs, which by construction fire after a fault is already severe. Our approach aims to catch earlier, subtler patterns.
  - "Can it detect misfires?" → Explicitly out of scope. 1 Hz OBD-II data cannot resolve per-cylinder combustion events. (This is charter Section 4.2.)
  - "Why no benchmark against public datasets?" → We evaluated EngineFaultDB; it's lab dynamometer data, not OBD-II, and comparing would require caveats that dilute the contribution. Out of scope, noted as future work.
  - "What would you do with more time?" → Real run-to-failure data, cross-vehicle fleet study, Raspberry Pi deployment, richer physics in the injection engine.

- **Both, 1h:** Final logistics check. Location confirmed for defense. Time confirmed. Laptop + charger + adapter + backup cables packed. Presentation on laptop AND on a USB stick AND on cloud backup. Paper PDF printed if needed.

---

## Concrete deliverables

- Paper submitted (Monday)
- Thesis book complete, PDF exported
- Presentation finalized
- Defense Q&A prep document
- Final README with reproducibility section

---

## Definition of Done

- [ ] Paper submission confirmed in writing
- [ ] Book PDF complete (~60–80 pages including appendices, roughly)
- [ ] Presentation exports cleanly to PDF (backup if laptop fails)
- [ ] Dashboard still works — tested Saturday
- [ ] Q&A prep document exists with ≥5 anticipated questions answered
- [ ] Both of you feel ready (subjective — but important)

---

## Week-specific risks

| Risk | Watch level | What to do |
|---|---|---|
| Submission system fails on Monday | Medium | Have the paper ready by Sunday night. Don't leave submission for Monday afternoon. If the system is down, email the advisor proof of on-time submission with the PDF attached. |
| Book runs very long / short | Low | Target is a deliverable book, not a perfect one. 60–80 pages is typical. 50 is fine if content is dense; 100 is fine if content justifies it. |
| Last-minute "let me improve the classifier one more time" temptation | HIGH | **Hard rule: no code changes to models after Week 7.** If you find a bug, note it as future work. Changing numbers in the paper after submission is a disaster. |
| Dry run runs way over or under time | Medium | Cut or add 2-3 slides as needed. 20 min presentation = roughly 20 slides at 1 slide/min. |
| Defense nerves | High likelihood, low impact | You've done the work. The plan was honest. The scope was appropriate. Answer questions truthfully, including "we don't know" or "that's future work" — those are professional answers. |

---

## Post-project (beyond 21 Jun)

After the defense, some optional follow-ups worth noting but not planning:

- Clean up the repo, move from private to public (with advisor approval)
- Add a demo GIF to the README
- Write a blog post / LinkedIn post about the project — useful for Valeo / Brightskies applications
- Archive a tagged release: `v1.0-thesis-defended`

---

## Final note

If you hit 21 Jun having done this plan, you've shipped:
- A working classifier with >0.80 macro-F1
- A working forecaster (regression or ordinal)
- A live dashboard on real hardware
- A paper + book + presentation all internally consistent and honestly framed

That's a strong graduation project by any measure, and a portfolio piece that holds up in interviews. Good luck.
