# Results directory

Evaluation outputs: metrics JSON, confusion matrices, calibration plots, figures. **Gitignored** — regenerate by re-running training scripts.

Figures intended for the paper or book should be copied from here into `docs/figures/` (tracked) once final.

## Headline numbers — important caveat

The metrics in `xgb_classifier_v1_results.json` (macro-F1 ≈ 0.87) and `loso_cv_results.json` (mean macro-F1 ≈ 0.96) are measured on labels produced by the deterministic injector in `src/injection/fault_injector.py`, scored against the algebraic inverse of the same injector's coefficients (see `src/features/severity.py` lines 32–35). They are a **synthetic self-consistency floor**, not a real-fault detection result.

The same applies to `forecaster_v1_results.json`: the target severity is the injector's own ramp re-derived through the severity formula. The coolant forecaster's near-zero MAE (0.002) reflects targets that are themselves near-zero due to cold-start gating (`src/features/severity.py` lines 105–108), not predictive skill on the underlying physical signal.

Real-fault evaluation outputs will land in `results/real_fault_eval/` once the harness (`src/eval/real_fault_eval.py`) and data collection (`docs/REAL_FAULT_COLLECTION.md`) are in place — see the project root `README.md` "Headline numbers" section.
