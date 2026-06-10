"""PLUMBING SMOKE TEST — proves the real-fault evaluation harness wires up.

THIS DOES NOT PROVE THE MODEL DETECTS REAL FAULTS.

The mock fixture in ``data/real_faults/mock/mock_lean_fault.csv`` is a
hand-edited derivative of ``drive1.csv`` where ``LONG_TERM_FUEL_TRIM_BANK_1``
and ``SHORT_TERM_FUEL_TRIM_BANK_1`` are manually biased upward starting at
row 200. This is the **same logical loop** as the synthetic injector in a
different costume — the fixture biases the same PIDs the injector biases.

So when the classifier labels post-bias windows as ``fuel_system`` or
``air_system``, that demonstrates the **harness wires up**:
  1. ``_read_csv`` reads both raw carOBD and clean-column formats.
  2. ``InferenceEngine.update()`` runs end-to-end without exceptions.
  3. The harness collects one record per stride window into the documented
     JSON shape with the expected keys.

It does **not** demonstrate that the model would detect a real vacuum leak,
real injector clog, or any naturally-occurring fault. Real-fault validation
is performed only against data collected per ``docs/REAL_FAULT_COLLECTION.md``
once the Skoda induction has been run (Step 5 of the honest-framing PR
series; headline metric: vacuum-leak recall ≥ 0.60).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import MODELS_DIR
from src.eval.real_fault_eval import _read_csv, evaluate_real_fault


def test_fault_fraction_excludes_cold_start():
    from src.eval.real_fault_eval import _summarise_labels

    label_counts = {"healthy": 5, "cold_start": 3, "air_system": 2}
    summary = _summarise_labels(label_counts, n_windows=10)
    assert summary["fault_window_count"] == 2
    assert summary["fault_fraction"] == pytest.approx(0.2)
    assert summary["non_fault_window_count"] == 8

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOCK_CSV = _REPO_ROOT / "data" / "real_faults" / "mock" / "mock_lean_fault.csv"
_RAW_DRIVE1 = _REPO_ROOT / "data" / "raw" / "carOBD" / "drive1.csv"
_MODELS_PRESENT = (MODELS_DIR / "xgb_classifier_v1.pkl").exists() and (
    MODELS_DIR / "forecaster_v1.pkl"
).exists()


# ─── Always-run plumbing checks (no model loading) ───────────────────────────


def test_mock_fixture_exists():
    """The mock fixture lives where the harness expects it."""
    assert _MOCK_CSV.exists(), (
        f"Mock fixture missing at {_MOCK_CSV}. "
        f"Re-generate per the recipe in data/real_faults/README.md."
    )


def test_read_csv_clean_column_format():
    """_read_csv handles the clean-name format used by the mock fixture."""
    df = _read_csv(_MOCK_CSV)
    assert "LONG_TERM_FUEL_TRIM_BANK_1" in df.columns
    assert "ENGINE_RPM" in df.columns
    assert len(df) == 600


@pytest.mark.skipif(not _RAW_DRIVE1.exists(), reason="drive1.csv not present")
def test_read_csv_raw_carobd_format():
    """_read_csv also handles the raw carOBD format with ``()`` suffixes."""
    df = _read_csv(_RAW_DRIVE1)
    assert "LONG_TERM_FUEL_TRIM_BANK_1" in df.columns
    assert "ENGINE_RPM" in df.columns


def test_mock_fixture_baseline_then_bias_shape():
    """The mock fixture has a clean baseline (rows 0–199) then a bias.

    This is what the harness depends on for the smoke test below — proven
    structurally so the smoke test's pre/post split is meaningful.
    """
    df = pd.read_csv(_MOCK_CSV)
    baseline_ltft = df["LONG_TERM_FUEL_TRIM_BANK_1"].iloc[:200].mean()
    biased_ltft = df["LONG_TERM_FUEL_TRIM_BANK_1"].iloc[400:].mean()
    assert biased_ltft - baseline_ltft > 8.0, (
        f"Mock fixture lost its bias. Baseline LTFT={baseline_ltft:.2f}, "
        f"biased LTFT={biased_ltft:.2f}. Re-generate per data/real_faults/README.md."
    )


# ─── Smoke test that needs the trained model ─────────────────────────────────


@pytest.mark.skipif(
    not _MODELS_PRESENT,
    reason=(
        "Trained model artefacts not present in models/. Run "
        "`python -m scripts.rebuild_all` to build them."
    ),
)
class TestEndToEnd:
    """End-to-end plumbing smoke test — requires the trained model artefacts.

    Skipped when artefacts are absent so a fresh clone still runs the
    model-free plumbing checks above.
    """

    def test_evaluate_returns_documented_shape(self):
        """evaluate_real_fault produces the JSON shape promised in its docstring."""
        result = evaluate_real_fault(_MOCK_CSV)

        assert set(result.keys()) >= {
            "csv_path",
            "n_rows",
            "n_windows",
            "windows",
            "summary",
        }
        assert result["n_rows"] == 600
        assert result["n_windows"] > 0
        assert all(
            {"elapsed_s", "label", "confidence", "all_probs"} <= set(w.keys())
            for w in result["windows"]
        )
        summary = result["summary"]
        assert set(summary.keys()) >= {
            "fault_window_count",
            "fault_fraction",
            "label_counts",
        }

    def test_harness_flags_biased_window_majority(self):
        """≥30 % of post-row-200 windows label != 'healthy'.

        NOT A FAULT-DETECTION TEST. The fixture biases the same PIDs the
        injector biases, so the classifier flagging those windows only
        confirms the pipeline is wired up. See module docstring for why
        this is plumbing, not detection.

        Threshold = 30 % is a smoke margin: well below what we'd expect
        from a working pipeline on a +12 % LTFT bias, well above what
        random noise on the healthy baseline produces.
        """
        result = evaluate_real_fault(_MOCK_CSV)
        post_bias = [w for w in result["windows"] if w["elapsed_s"] >= 260]
        # Row 260+ is past the full-ramp boundary (row 300 holds full bias),
        # but a window centred at row 260 already sees mostly-biased data
        # because the buffer holds rows 200–259.
        assert len(post_bias) > 0, "No post-bias windows recorded — harness bug."
        fault_post = [w for w in post_bias if w["label"] != "healthy"]
        frac = len(fault_post) / len(post_bias)
        assert frac >= 0.30, (
            f"Plumbing smoke test failed: only {frac:.0%} of post-bias windows "
            f"({len(fault_post)}/{len(post_bias)}) carry a non-healthy label. "
            f"Expected ≥30%. Either the harness is mis-wired or the mock fixture "
            f"has drifted. Re-generate per data/real_faults/README.md."
        )
