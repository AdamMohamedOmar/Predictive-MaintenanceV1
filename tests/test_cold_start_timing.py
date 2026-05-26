"""T3.2 regression: ColdStartChecker timers must count real seconds, not rows.

At 0.3 Hz the rule "frozen ECT after 90 s" must fire after ≥ 90 real
seconds, NOT after 90 row-counting steps.
"""

import pytest
from src.diagnostics.cold_start_checker import ColdStartChecker


def test_frozen_sensor_requires_90_real_seconds_not_rows():
    """Feed 90 updates at 3-second intervals (= 270 real seconds, 90 rows).

    Without the T3.2 fix, row-counting would fire the rule at row 90.
    With the fix, the rule uses the `now` timestamp and sees 270 s elapsed,
    so it fires (270 s >> 90 s threshold).  The key assertion is the *inverse*:
    at fewer than 30 updates (= 90 s elapsed) the rule must NOT have fired.
    """
    c = ColdStartChecker(frozen_sensor_min_s=90)

    # Feed 29 updates at 3 s intervals → 87 real seconds → must NOT fire yet
    for i in range(29):
        new = c.update(coolant=42.0, rpm=800.0, speed=0.0, now=float(i * 3))
        assert not any(a.rule == "ect_sensor_frozen" for a in new), (
            f"Frozen-sensor rule fired too early at update {i} "
            f"(t={i*3}s, expected ≥90 s)"
        )

    # 30th update: t=87 s — still below 90 s threshold
    new = c.update(coolant=42.0, rpm=800.0, speed=0.0, now=87.0)
    assert not any(a.rule == "ect_sensor_frozen" for a in new), (
        "Frozen-sensor fired at 87 s real time (threshold is 90 s)"
    )

    # Feed until ≥90 real seconds have elapsed
    new = c.update(coolant=42.0, rpm=800.0, speed=0.0, now=91.0)
    assert any(a.rule == "ect_sensor_frozen" for a in c.alerts), (
        "Frozen-sensor rule never fired even after 91 real seconds with flat coolant"
    )


def test_legacy_row_counting_still_works_without_now():
    """Existing tests that call update() without `now` must keep working.

    The fallback _elapsed_s += 1 path is preserved for CSV replay and tests.
    """
    c = ColdStartChecker(frozen_sensor_min_s=10)
    for _ in range(10):
        c.update(coolant=42.0, rpm=800.0, speed=0.0)  # no `now` kwarg
    assert any(a.rule == "ect_sensor_frozen" for a in c.alerts)


def test_reset_clears_session_start_time():
    """After reset(), the next call with `now` starts a fresh session clock."""
    c = ColdStartChecker(frozen_sensor_min_s=5)
    # First session
    for i in range(10):
        c.update(coolant=42.0, rpm=800.0, speed=0.0, now=float(i))
    assert any(a.rule == "ect_sensor_frozen" for a in c.alerts)

    c.reset()
    # Second session: 3 updates at t=1000…1002 — start fresh, only 2 s elapsed
    for i in range(3):
        new = c.update(coolant=42.0, rpm=800.0, speed=0.0, now=float(1000 + i))
        assert not any(a.rule == "ect_sensor_frozen" for a in new), (
            "Session clock not reset — timer leaked across sessions"
        )
