"""P0-2a: withheld-coefficient evaluation harness.

These tests check the harness MECHANICS on synthetic data — that the two
generator configs are genuinely distinct and that the gap computation runs
end-to-end and returns the documented shape. The real synthetic-to-withheld
gap is produced by `python -m scripts.eval_withheld_coeff` on carOBD data;
its size is an honest empirical outcome, not something tests pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.extractor import feature_names
from src.injection.eval_configs import CONFIG_A, CONFIG_B
from src.features.dataset_builder import LABEL_TO_ID
from src.models.classifier import ALL_LABELS
from scripts.eval_withheld_coeff import compute_withheld_gap

_FEAT = feature_names()


# ─── Configs are genuinely different ─────────────────────────────────────────


def test_configs_differ_on_every_axis():
    assert CONFIG_A.magnitudes != CONFIG_B.magnitudes
    assert CONFIG_A.onset_fraction != CONFIG_B.onset_fraction
    assert CONFIG_A.ramp_fraction != CONFIG_B.ramp_fraction
    assert CONFIG_A.noise_std != CONFIG_B.noise_std


def test_configs_cover_all_four_faults():
    faults = {"air_system", "fuel_system", "coolant_temp_sensor", "throttle_position_sensor"}
    assert set(CONFIG_A.magnitudes) == faults
    assert set(CONFIG_B.magnitudes) == faults


# ─── Gap computation runs end-to-end on synthetic data ───────────────────────


def _make_ds(shift: float, seed: int) -> pd.DataFrame:
    """Separable synthetic dataset with the held-out sessions present.

    `shift` offsets the class means so a model trained on shift=0 scores
    slightly worse on a shifted set — mimicking a withheld generator.
    """
    rng = np.random.default_rng(seed)
    sessions = ["s1", "s2", "s3", "drive1", "live12"]  # held-out set is {drive1, live12}
    rows = []
    for ci, label in enumerate(ALL_LABELS):
        for sess in sessions:
            for _ in range(20):
                row = {c: float(rng.normal(ci * 6.0 + shift, 1.0)) for c in _FEAT}
                row["label"] = label
                row["label_id"] = LABEL_TO_ID[label]
                row["session_id"] = sess
                row["fault_type"] = label
                rows.append(row)
    return pd.DataFrame(rows)


def test_compute_withheld_gap_returns_documented_shape():
    ds_a = _make_ds(shift=0.0, seed=1)
    ds_b = _make_ds(shift=2.0, seed=2)  # a different "generator"
    out = compute_withheld_gap(ds_a, ds_b, n_estimators=20, random_seed=0)

    assert {"same_config_macro_f1", "withheld_config_macro_f1", "gap"} <= set(out.keys())
    assert 0.0 <= out["same_config_macro_f1"] <= 1.0
    assert 0.0 <= out["withheld_config_macro_f1"] <= 1.0
    assert np.isfinite(out["gap"])
    # gap is defined as same − withheld
    assert abs(out["gap"] - (out["same_config_macro_f1"] - out["withheld_config_macro_f1"])) < 1e-9
