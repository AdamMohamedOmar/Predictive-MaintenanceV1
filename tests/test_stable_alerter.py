"""Tests for StableAlerter temporal voting filter."""
import pytest
from src.models.stable_alerter import StableAlerter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def alerter():
    """Default alerter: 3 windows, 70% confidence threshold."""
    return StableAlerter(min_windows=3, min_confidence=0.70, clear_confidence=0.80)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

def test_initial_state_is_no_alert(alerter):
    assert alerter.state.active is False
    assert alerter.state.fault_type == "healthy"


def test_invalid_min_windows_raises():
    with pytest.raises(ValueError):
        StableAlerter(min_windows=0)


def test_invalid_min_confidence_raises():
    with pytest.raises(ValueError):
        StableAlerter(min_confidence=1.1)


# ---------------------------------------------------------------------------
# No alert for fewer than min_windows predictions
# ---------------------------------------------------------------------------

def test_single_fault_window_does_not_alert(alerter):
    state = alerter.update("fuel_system", 0.95)
    assert state.active is False


def test_two_fault_windows_does_not_alert(alerter):
    alerter.update("fuel_system", 0.90)
    state = alerter.update("fuel_system", 0.90)
    assert state.active is False


# ---------------------------------------------------------------------------
# Alert fires on simple majority (⌊N/2⌋+1 windows must agree on same fault)
# ---------------------------------------------------------------------------

def test_three_consecutive_fault_windows_alerts(alerter):
    for _ in range(3):
        state = alerter.update("fuel_system", 0.90)
    assert state.active is True
    assert state.fault_type == "fuel_system"


def test_alert_fires_on_majority_same_fault(alerter):
    """2 of 3 windows on the same fault IS enough (majority, not unanimity)."""
    alerter.update("fuel_system", 0.90)
    alerter.update("air_system", 0.90)
    state = alerter.update("fuel_system", 0.90)
    # fuel_system has 2/3 ≥ ⌊3/2⌋+1 = 2 → majority → alert fires
    assert state.active is True
    assert state.fault_type == "fuel_system"


def test_evenly_split_fault_labels_do_not_fire(alerter):
    """1 window per distinct fault — no majority → no alert."""
    alerter.update("fuel_system", 0.90)
    alerter.update("air_system", 0.90)
    state = alerter.update("coolant_temp_sensor", 0.90)
    assert state.active is False


def test_alert_suppressed_by_low_confidence(alerter):
    for _ in range(3):
        state = alerter.update("air_system", 0.50)  # below 0.70 threshold
    assert state.active is False


# ---------------------------------------------------------------------------
# Healthy window in the middle resets the vote
# ---------------------------------------------------------------------------

def test_lone_fault_window_among_healthy_does_not_fire(alerter):
    """1 fault window out of 3 (2 healthy) — below majority → no alert."""
    alerter.update("healthy", 0.95)
    alerter.update("coolant_temp_sensor", 0.85)
    alerter.update("healthy", 0.95)
    # Only 1/3 fault windows — below majority threshold of 2 → no alert
    assert alerter.state.active is False


def test_two_of_three_fault_windows_fire(alerter):
    """2 fault windows out of 3 (1 healthy) — equals majority → alert fires."""
    alerter.update("coolant_temp_sensor", 0.85)
    alerter.update("healthy", 0.95)
    alerter.update("coolant_temp_sensor", 0.85)
    # 2/3 coolant_temp_sensor — equals majority threshold of ⌊3/2⌋+1=2 → fires
    assert alerter.state.active is True
    assert alerter.state.fault_type == "coolant_temp_sensor"


# ---------------------------------------------------------------------------
# Hysteresis: alert stays active until high-confidence healthy clears it
# ---------------------------------------------------------------------------

def test_active_alert_persists_through_low_conf_healthy(alerter):
    # Trigger the alert
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.active is True

    # Low-confidence healthy should NOT clear it
    state = alerter.update("healthy", 0.60)
    assert state.active is True


def test_active_alert_cleared_by_high_conf_healthy(alerter):
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.active is True

    state = alerter.update("healthy", 0.85)
    assert state.active is False
    assert state.fault_type == "healthy"


def test_cold_start_clears_active_alert(alerter):
    """A high-confidence cold_start label must clear an active alert.

    Without this, a fault that fires during the warm-up phase can never be
    suppressed once the engine enters the cold_start operating regime.
    """
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.active is True

    state = alerter.update("cold_start", 0.85)
    assert state.active is False
    assert state.fault_type == "healthy"


# ---------------------------------------------------------------------------
# reset() clears everything
# ---------------------------------------------------------------------------

def test_reset_clears_buffer_and_state(alerter):
    for _ in range(3):
        alerter.update("air_system", 0.90)
    assert alerter.state.active is True
    alerter.reset()
    assert alerter.state.active is False
    # After reset, should need 3 more windows to fire again
    alerter.update("air_system", 0.90)
    assert alerter.state.active is False


# ---------------------------------------------------------------------------
# Return value matches state property
# ---------------------------------------------------------------------------

def test_update_return_matches_state_property(alerter):
    returned = alerter.update("throttle_position_sensor", 0.75)
    assert returned is alerter.state


# ---------------------------------------------------------------------------
# Rule-engine alert integration
# ---------------------------------------------------------------------------

def _make_rule_alert(rule="thermostat_stuck_open"):
    from src.diagnostics.cold_start_checker import ColdStartAlert
    return ColdStartAlert(
        rule=rule,
        description="test",
        confidence=0.90,
        triggered_at_s=120,
    )


def test_rule_alert_attaches_to_state(alerter):
    ra = _make_rule_alert()
    state = alerter.ingest_rule_alert(ra)
    assert len(state.rule_alerts) == 1
    assert state.rule_alerts[0].rule == "thermostat_stuck_open"


def test_rule_alert_does_not_affect_ml_active_flag(alerter):
    """Rule alerts are independent of the ML voting — active flag stays False."""
    ra = _make_rule_alert()
    state = alerter.ingest_rule_alert(ra)
    assert state.active is False  # ML hasn't voted yet


def test_multiple_rule_alerts_accumulate(alerter):
    alerter.ingest_rule_alert(_make_rule_alert("thermostat_stuck_open"))
    state = alerter.ingest_rule_alert(_make_rule_alert("ect_sensor_frozen"))
    assert len(state.rule_alerts) == 2


def test_reset_clears_rule_alerts(alerter):
    alerter.ingest_rule_alert(_make_rule_alert())
    alerter.reset()
    assert alerter.state.rule_alerts == []


def test_both_streams_active_simultaneously(alerter):
    """ML alert and rule alert can be active at the same time."""
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.active is True

    alerter.ingest_rule_alert(_make_rule_alert("ect_sensor_frozen"))
    assert alerter.state.active is True
    assert len(alerter.state.rule_alerts) == 1


# ---------------------------------------------------------------------------
# Fault-to-fault transitions (A3 — live demo robustness)
# ---------------------------------------------------------------------------

def test_fault_transitions_to_different_fault(alerter):
    """Active fuel_system alert transitions directly to air_system on full consensus."""
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.fault_type == "fuel_system"

    # Three consecutive air_system windows displace fuel_system in the buffer
    for _ in range(3):
        state = alerter.update("air_system", 0.85)

    assert state.active is True
    assert state.fault_type == "air_system"
    assert state.confidence == pytest.approx(0.85)


def test_single_divergent_window_does_not_transition(alerter):
    """1 of 3 windows on a different fault — below majority → no transition."""
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.fault_type == "fuel_system"

    # 2 fuel_system + 1 air_system: air_system has only 1/3 — below majority
    alerter.update("fuel_system", 0.80)
    alerter.update("fuel_system", 0.80)
    state = alerter.update("air_system", 0.85)

    assert state.fault_type == "fuel_system"  # 1/3 air_system not enough to transition


def test_same_fault_type_does_not_retransition(alerter):
    """Buffer agreeing on the SAME fault as the active one should not change anything."""
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.fault_type == "fuel_system"

    state = alerter.update("fuel_system", 0.95)
    assert state.fault_type == "fuel_system"
    assert state.active is True


def test_transition_requires_min_confidence(alerter):
    """Fault transition should NOT fire if new-fault confidence is below threshold."""
    for _ in range(3):
        alerter.update("fuel_system", 0.90)
    assert alerter.state.fault_type == "fuel_system"

    # air_system at 0.50 — below min_confidence of 0.70
    for _ in range(3):
        state = alerter.update("air_system", 0.50)

    assert state.fault_type == "fuel_system"  # no transition
