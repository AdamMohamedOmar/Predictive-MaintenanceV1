"""Evaluation harnesses for non-injector-generated data.

`real_fault_eval` runs the per-window classification pipeline on a CSV and
emits structured JSON. Intended for two callers:
  1. tests/test_real_fault_harness_plumbing.py — smoke test against the
     mock fixture in data/real_faults/mock/.
  2. scripts/eval_real_fault.py — CLI for evaluating real Skoda recordings
     collected per docs/REAL_FAULT_COLLECTION.md.

None of the harnesses in this package claim to validate real-fault
detection. They produce per-window predictions; "did the model detect the
fault" is interpreted downstream against per-run metadata (mods-in / mods-out
timestamps) by the caller.
"""
