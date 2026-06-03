"""P2-2: injected severity must vary, not be a single fixed shape per fault.

PdM is about a continuum of degradation. The old builder injected one fixed
magnitude/onset/ramp per fault, so the classifier only ever saw one "shape" of
each fault. `jitter_injection` draws a per-injection magnitude and shape so the
training set spans a severity range.
"""

from __future__ import annotations

import numpy as np

from src.features.dataset_builder import jitter_injection


def test_jitter_varies_magnitude_across_injections():
    """Independent draws produce different magnitudes spanning the base range."""
    base = 13.0
    jitter = (0.5, 1.3)
    mags = [
        jitter_injection(base, 0.40, 0.15, jitter, np.random.default_rng(s))[0]
        for s in range(20)
    ]
    assert len(set(mags)) > 1, "Jittered magnitude is constant — no severity variation."
    # All within base × [lo, hi].
    assert min(mags) >= base * 0.5 - 1e-9
    assert max(mags) <= base * 1.3 + 1e-9
    # And genuinely spread (not all clustered).
    assert np.std(mags) > 0.5


def test_jitter_onset_and_ramp_stay_in_bounds():
    jitter = (0.5, 1.3)
    for s in range(50):
        _, onset, ramp = jitter_injection(13.0, 0.40, 0.15, jitter, np.random.default_rng(s))
        assert 0.20 <= onset <= 0.60
        assert 0.08 <= ramp <= 0.35


def test_jitter_is_reproducible_for_a_seed():
    a = jitter_injection(13.0, 0.40, 0.15, (0.5, 1.3), np.random.default_rng(7))
    b = jitter_injection(13.0, 0.40, 0.15, (0.5, 1.3), np.random.default_rng(7))
    assert a == b
