"""Tests for BaselineNormalizer — regime pass-through and feature_means shape."""

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import feature_names
from src.features.regime import regime_feature_names
from src.features.normalizer import BaselineNormalizer, normalised_feature_names


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 100, label: str = "healthy") -> pd.DataFrame:
    """Synthetic feature matrix with all 83 columns and a label column."""
    rng = np.random.default_rng(99)
    cols = {c: rng.normal(0.0, 1.0, n) for c in feature_names()}
    # Regime one-hots: exactly one of the five is 1.0 per row
    for i, rcol in enumerate(regime_feature_names()):
        cols[rcol] = 0.0
    cols[regime_feature_names()[4]] = 1.0  # REGIME__CRUISE = 1 for every row
    df = pd.DataFrame(cols)
    df["label"] = label
    return df


# ─── Regime pass-through (T5.1) ──────────────────────────────────────────────

def test_regime_z_columns_are_binary():
    """Regime __z columns must be exactly 0.0 or 1.0 (not z-scored floats)."""
    df = _make_feature_df()
    norm = BaselineNormalizer().fit(df)
    out = norm.transform(df)
    for rcol in regime_feature_names():
        z_col = f"{rcol}__z"
        assert z_col in out.columns, f"Missing {z_col}"
        unique_vals = set(out[z_col].unique())
        assert unique_vals <= {0.0, 1.0}, (
            f"{z_col} has non-binary values {unique_vals} after transform — "
            f"regime one-hots must be copied verbatim, not z-scored."
        )


def test_continuous_z_columns_are_not_binary():
    """At least some continuous __z columns should have non-binary values."""
    df = _make_feature_df()
    norm = BaselineNormalizer().fit(df)
    out = norm.transform(df)
    # ENGINE_RPM__mean is a continuous feature; its z-scored version must have
    # more than 2 unique values (it's a real-valued z-score, not a binary flag).
    z_col = "ENGINE_RPM__mean__z"
    assert z_col in out.columns
    # Should have more than 2 unique values (it's a real-valued z-score)
    assert out[z_col].nunique() > 2


# ─── feature_means shape ─────────────────────────────────────────────────────

def test_feature_means_length():
    """feature_means must return one value per base feature (83 total)."""
    df = _make_feature_df()
    norm = BaselineNormalizer().fit(df)
    means = norm.feature_means
    assert len(means) == len(feature_names()), (
        f"feature_means returned {len(means)} values but feature_names() has "
        f"{len(feature_names())} entries."
    )


def test_feature_means_is_copy():
    """Mutating the returned array must not affect the internal scaler state."""
    df = _make_feature_df()
    norm = BaselineNormalizer().fit(df)
    means1 = norm.feature_means
    means1[:] = 999.0
    means2 = norm.feature_means
    assert not np.all(means2 == 999.0), "feature_means is not returning a copy"


# ─── normalised_feature_names ────────────────────────────────────────────────

def test_normalised_feature_names_count():
    assert len(normalised_feature_names()) == len(feature_names())


def test_normalised_feature_names_suffix():
    for name in normalised_feature_names():
        assert name.endswith("__z"), f"{name!r} does not end with '__z'"


# ─── unfitted guard ──────────────────────────────────────────────────────────

def test_transform_before_fit_raises():
    norm = BaselineNormalizer()
    df = _make_feature_df()
    with pytest.raises(RuntimeError, match="fit"):
        norm.transform(df)


def test_feature_means_before_fit_raises():
    norm = BaselineNormalizer()
    with pytest.raises(RuntimeError, match="fit"):
        _ = norm.feature_means
