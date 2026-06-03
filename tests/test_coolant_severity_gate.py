"""P1-1 acceptance: coolant severity must fire for a stuck-cold sensor and
stay quiet for a legitimately warming engine.

The old gate returned 0 whenever the coolant was cold (cold_start/warmup
regime). A stuck-cold ECT reads cold forever, so the fault triggered the gate
that nullified it. The fix discriminates on warm-up DYNAMICS: cold AND flat =
stuck sensor; cold AND climbing = normal warm-up.
"""

from __future__ import annotations

from src.features.severity import compute_severity


def _coolant_features(temp_mean: float, warmup_rate: float) -> dict[str, float]:
    """Minimal feature dict for the coolant severity branch."""
    return {
        "COOLANT_TEMPERATURE__mean": temp_mean,
        "COOLANT_WARMUP_RATE": warmup_rate,
        # The coolant branch ignores these, but keep a realistic dict shape.
        "FUEL_LOOP_ACTIVE": 1.0,
        "REGIME__COLD_START": 1.0 if temp_mean < 55 else 0.0,
        "REGIME__WARMUP": 1.0 if 55 <= temp_mean < 75 else 0.0,
    }


def test_stuck_cold_sensor_scores_high():
    """Coolant pinned at ~42 °C with ~0 °C/min warm-up = stuck sensor → high severity.

    This is the case the OLD code failed: it returned 0 because the cold
    reading set REGIME__COLD_START, which gated severity to zero.
    """
    feats = _coolant_features(temp_mean=42.0, warmup_rate=0.0)
    sev = compute_severity(feats, "coolant_temp_sensor", baselines={})
    assert sev > 0.5, f"Stuck-cold sensor should score > 0.5, got {sev:.3f}."


def test_normal_cold_start_scores_near_zero():
    """Coolant at 50 °C but climbing 1.5 °C/min = healthy warm-up → ~0 severity."""
    feats = _coolant_features(temp_mean=50.0, warmup_rate=1.5)
    sev = compute_severity(feats, "coolant_temp_sensor", baselines={})
    assert sev < 0.05, f"Healthy warm-up should score ≈ 0, got {sev:.3f}."


def test_warm_reading_scores_zero():
    """A warm coolant reading cannot be a stuck-COLD sensor → 0."""
    feats = _coolant_features(temp_mean=90.0, warmup_rate=0.0)
    sev = compute_severity(feats, "coolant_temp_sensor", baselines={})
    assert sev == 0.0


def test_stuck_cold_severity_scales_with_deficit():
    """Colder stuck value → higher severity (monotonic in the temperature deficit)."""
    s42 = compute_severity(_coolant_features(42.0, 0.0), "coolant_temp_sensor", {})
    s60 = compute_severity(_coolant_features(60.0, 0.0), "coolant_temp_sensor", {})
    assert s42 > s60 > 0.0
