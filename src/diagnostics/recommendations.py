"""Workshop-style recommended diagnostic steps for each fault label.

Treat each list as an ordered checklist a tech would follow.  Steps are kept
to 3-5 items so the dashboard doesn't sprawl.
"""

from __future__ import annotations

RECOMMENDATIONS: dict[str, list[str]] = {
    "air_system": [
        "Smoke-test the intake — listen for hiss at idle (vacuum lines, brake booster).",
        "Inspect intake elbow & MAF housing for cracks (post-MAF leaks raise STFT).",
        "Check MAF sensor for contamination — clean with MAF-safe spray if dirty.",
        "Confirm PCV system seals; replace PCV valve if older than 80k km.",
    ],
    "fuel_system": [
        "Measure fuel rail pressure key-on (spec: 300-400 kPa nominal).",
        "Inspect fuel filter — replace if > 50k km since last change.",
        "Pull DTCs — confirm P0171 or P0172. If misfire codes present, suspect injector clog instead.",
        "Run injector balance test if pressure is in spec.",
    ],
    "coolant_temp_sensor": [
        "Compare ECT vs IAT after engine has soaked overnight — they must agree within +/-3 C.",
        "Check ECT connector for corrosion / loose pins.",
        "Measure ECT resistance vs spec table (cold ~2.5 kOhm, hot ~200 Ohm).",
        "If sensor reads stable at one value regardless of state, replace.",
    ],
    "throttle_position_sensor": [
        "Inspect throttle body for carbon build-up — clean if visible.",
        "Sweep pedal slowly key-on, engine-off — TPS reading should track linearly.",
        "Compare ACCELERATOR_PEDAL_POSITION_D vs E — they should agree (mismatch = pedal sensor).",
        "If throttle body recently replaced, re-learn idle position (check service manual).",
    ],
    "thermostat_stuck_open": [
        "Check coolant level — top up if low.",
        "Replace thermostat (standard service item; ~30 min on most cars).",
        "Re-test warm-up time — should reach 75 C within 4-5 minutes in normal weather.",
    ],
    "thermostat_stuck_closed": [
        "STOP DRIVING — let engine cool to ambient before any inspection.",
        "Check coolant level (could be low — overheating cause OR effect).",
        "Replace thermostat AND inspect radiator for blockage.",
        "Compression test cylinders 1-4 — overheating may have damaged head gasket.",
    ],
    "ect_sensor_frozen": [
        "Verify sensor variance — running engine should show +/-0.5 C oscillation.",
        "Replace ECT sensor (cheap, ~10 min job).",
    ],
    "iac_valve_stuck_open": [
        "Confirm A/C is OFF when re-testing (compressor adds ~150 rpm).",
        "Clean throttle body and IAC passage.",
        "If still elevated, perform idle re-learn procedure for the vehicle.",
    ],
    "alternator_low_output": [
        "Load-test battery with a dedicated tester (not just voltage — test CCA).",
        "Check alternator belt tension and condition.",
        "Measure alternator output at 2000 rpm: should be 13.8-14.7 V.",
        "If voltage is low under load, replace alternator.",
    ],
}


def get_steps(label: str) -> list[str]:
    """Return ordered diagnostic steps for a fault label; empty list if unmapped."""
    return RECOMMENDATIONS.get(label, [])
