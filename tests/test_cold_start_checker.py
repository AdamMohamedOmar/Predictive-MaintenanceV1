"""Tests for the cold-start rule-based diagnostic engine."""
from src.diagnostics.cold_start_checker import ColdStartChecker


# ── Helpers ──────────────────────────────────────────────────────────────────

def _feed(checker, coolant, rpm=800.0, speed=0.0, n=1):
    """Feed n identical rows and return all new alerts."""
    alerts = []
    for _ in range(n):
        alerts.extend(checker.update(coolant=coolant, rpm=rpm, speed=speed))
    return alerts


def _feed_warmup(checker, start_temp=40.0, target=80.0, rate=1.0):
    """Feed a realistic warm-up curve until target is reached."""
    temp = start_temp
    while temp < target:
        checker.update(coolant=temp, rpm=1200.0, speed=0.0)
        temp = min(temp + rate / 60.0, target)  # rate °C/min → /s


# ── Construction ─────────────────────────────────────────────────────────────

def test_initial_state_no_alerts():
    c = ColdStartChecker()
    assert c.alerts == []
    assert not c.is_dormant


def test_reset_clears_everything():
    c = ColdStartChecker(warmup_timeout_s=5)
    _feed(c, coolant=40.0, n=10)
    c.reset()
    assert c.alerts == []
    assert not c.is_dormant


# ── Thermostat stuck-open rule ────────────────────────────────────────────────

def test_thermostat_fires_when_coolant_never_warms(  ):
    c = ColdStartChecker(warmup_timeout_s=10)
    # Feed 10 seconds of cold engine — should trigger
    _feed(c, coolant=40.0, n=10)
    rules = [a.rule for a in c.alerts]
    assert "thermostat_stuck_open" in rules


def test_thermostat_does_not_fire_before_timeout():
    c = ColdStartChecker(warmup_timeout_s=30)
    _feed(c, coolant=40.0, n=20)
    assert not any(a.rule == "thermostat_stuck_open" for a in c.alerts)


def test_thermostat_does_not_fire_if_engine_warms_up():
    c = ColdStartChecker(warmup_timeout_s=10)
    # Engine warms up on second 5 — no alert
    _feed(c, coolant=40.0, n=4)
    _feed(c, coolant=80.0, n=7)  # engine warm now
    assert not any(a.rule == "thermostat_stuck_open" for a in c.alerts)


def test_thermostat_fires_only_once():
    c = ColdStartChecker(warmup_timeout_s=5)
    _feed(c, coolant=40.0, n=20)
    thermostat_alerts = [a for a in c.alerts if a.rule == "thermostat_stuck_open"]
    assert len(thermostat_alerts) == 1


# ── ECT sensor frozen rule ────────────────────────────────────────────────────

def test_frozen_sensor_fires_on_flat_reading():
    c = ColdStartChecker(frozen_sensor_min_s=10)
    # Perfectly flat reading for 10 seconds
    _feed(c, coolant=42.0, n=10)
    assert any(a.rule == "ect_sensor_frozen" for a in c.alerts)


def test_frozen_sensor_does_not_fire_on_varying_reading():
    c = ColdStartChecker(frozen_sensor_min_s=10)
    # Alternating readings — plenty of variance
    for i in range(10):
        c.update(coolant=40.0 + i * 0.5, rpm=800.0, speed=0.0)
    assert not any(a.rule == "ect_sensor_frozen" for a in c.alerts)


def test_frozen_sensor_does_not_fire_before_min_window():
    c = ColdStartChecker(frozen_sensor_min_s=20)
    _feed(c, coolant=42.0, n=10)
    assert not any(a.rule == "ect_sensor_frozen" for a in c.alerts)


def test_frozen_sensor_fires_only_once():
    c = ColdStartChecker(frozen_sensor_min_s=5)
    _feed(c, coolant=42.0, n=20)
    frozen_alerts = [a for a in c.alerts if a.rule == "ect_sensor_frozen"]
    assert len(frozen_alerts) == 1


# ── IAC valve rule ────────────────────────────────────────────────────────────

def test_iac_fires_on_high_idle_rpm_after_warmup():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    # Warm up first
    _feed(c, coolant=80.0, n=1)
    # Now feed high-idle rows for iac_warm_min_s + buffer
    _feed(c, coolant=85.0, rpm=1500.0, speed=0.0, n=10)
    assert any(a.rule == "iac_valve_stuck_open" for a in c.alerts)


def test_iac_does_not_fire_on_normal_idle_rpm():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    _feed(c, coolant=80.0, n=1)
    _feed(c, coolant=85.0, rpm=800.0, speed=0.0, n=10)
    assert not any(a.rule == "iac_valve_stuck_open" for a in c.alerts)


def test_iac_does_not_fire_before_engine_warms():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    # High RPM but engine still cold
    _feed(c, coolant=40.0, rpm=1500.0, speed=0.0, n=20)
    assert not any(a.rule == "iac_valve_stuck_open" for a in c.alerts)


def test_iac_does_not_fire_when_driving():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    _feed(c, coolant=80.0, n=1)
    # High RPM but car is moving — not idle
    _feed(c, coolant=85.0, rpm=3000.0, speed=50.0, n=10)
    assert not any(a.rule == "iac_valve_stuck_open" for a in c.alerts)


# ── Dormancy ──────────────────────────────────────────────────────────────────

def test_checker_goes_dormant_after_full_warmup():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    _feed(c, coolant=80.0, n=1)
    _feed(c, coolant=90.0, rpm=800.0, speed=0.0, n=10)
    assert c.is_dormant


def test_dormant_checker_returns_no_new_alerts():
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    _feed(c, coolant=80.0, n=1)
    _feed(c, coolant=90.0, rpm=800.0, speed=0.0, n=10)
    assert c.is_dormant
    # Feed more rows — nothing should fire
    new = _feed(c, coolant=40.0, rpm=2000.0, speed=0.0, n=100)
    assert new == []


# ── Overheat rule ─────────────────────────────────────────────────────────────

def test_overheat_alert_fires_at_high_coolant():
    """30 consecutive seconds above 108°C must trigger thermostat_stuck_closed."""
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    # First warm the engine normally
    _feed(c, coolant=90.0, rpm=900.0, speed=0.0, n=120)
    # Now simulate 30s of dangerously high coolant
    all_new = []
    for _ in range(30):
        all_new += c.update(coolant=112.0, rpm=900.0, speed=0.0)
    assert any(a.rule == "thermostat_stuck_closed" for a in all_new), (
        "overheat alert not fired — check _check_overheat() consecutive-seconds logic"
    )


def test_overheat_alert_does_not_fire_on_brief_spike():
    """A single spike above 108°C must not trigger — only sustained heat does."""
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    _feed(c, coolant=90.0, rpm=900.0, speed=0.0, n=120)
    # 10 seconds above threshold — less than the 30s consecutive requirement
    all_new = []
    for _ in range(10):
        all_new += c.update(coolant=112.0, rpm=900.0, speed=0.0)
    assert not any(a.rule == "thermostat_stuck_closed" for a in all_new)


def test_overheat_alert_fires_even_after_dormancy():
    """Overheat check must run even after the checker goes dormant."""
    c = ColdStartChecker(warmup_timeout_s=300, iac_warm_min_s=5)
    # Warm up to dormancy
    _feed(c, coolant=80.0, n=1)
    _feed(c, coolant=90.0, rpm=800.0, speed=0.0, n=10)
    assert c.is_dormant

    # Then simulate sustained overheat while dormant
    all_new = []
    for _ in range(30):
        all_new += c.update(coolant=112.0, rpm=900.0, speed=0.0)
    assert any(a.rule == "thermostat_stuck_closed" for a in all_new), (
        "overheat alert not fired from dormant checker"
    )


# ── Alert structure ───────────────────────────────────────────────────────────

def test_alert_has_required_fields():
    c = ColdStartChecker(warmup_timeout_s=5)
    _feed(c, coolant=40.0, n=5)
    alerts = c.alerts
    assert len(alerts) > 0
    a = alerts[0]
    assert isinstance(a.rule, str)
    assert isinstance(a.description, str)
    assert 0.0 < a.confidence <= 1.0
    assert isinstance(a.triggered_at_s, int)
