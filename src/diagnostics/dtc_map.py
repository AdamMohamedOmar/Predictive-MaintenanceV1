"""Mapping from our internal fault labels to OBD-II Diagnostic Trouble Codes.

These are the codes a workshop tech reads on a scan tool — using them in the UI
lets us speak the same language as the customer's mechanic.
"""

from __future__ import annotations

DTC_MAP: dict[str, dict[str, str]] = {
    "air_system": {
        "code": "P0171",
        "name": "System Too Lean (Bank 1)",
        "short": "Lean",
        "description": (
            "ECU is adding fuel to compensate for excess unmetered air. "
            "Most common cause: vacuum leak after MAF, or worn MAF sensor."
        ),
    },
    "fuel_system": {
        "code": "P0171 / P0087",
        "name": "Lean Mixture + Low Fuel Pressure",
        "short": "Lean+Fuel",
        "description": (
            "ECU has biased LTFT chronically positive — fuel delivery is "
            "below demand. P0171 = Bank 1 lean (LTFT compensation exhausted); "
            "P0087 = fuel rail pressure too low. Most common cause: clogged "
            "fuel filter, weak fuel pump, or stuck-closed pressure regulator."
        ),
    },
    "coolant_temp_sensor": {
        "code": "P0117",
        "name": "ECT Sensor Circuit Low Input",
        "short": "ECT Low",
        "description": (
            "Coolant temp sensor reports value lower than physically plausible "
            "given engine run time. Likely sensor short to ground or "
            "open circuit at the sensor connector."
        ),
    },
    "throttle_position_sensor": {
        "code": "P2135",
        "name": "Throttle Position Sensor A/B Correlation",
        "short": "TPS Corr",
        "description": (
            "Reported throttle angle diverges from pedal command — "
            "TPS-A and TPS-B signals disagree beyond tolerance. "
            "Most common cause: worn TPS resistive track, "
            "harness contamination, or contaminated throttle body."
        ),
    },
    # Cold-start rule alerts
    "thermostat_stuck_open": {
        "code": "P0128",
        "name": "Coolant Below Thermostat Regulating Temp",
        "short": "Thermostat",
        "description": (
            "Coolant never reached operating temperature within expected "
            "time window. Thermostat is failing open."
        ),
    },
    "thermostat_stuck_closed": {
        "code": "P0217",
        "name": "Engine Over-Temperature",
        "short": "Overheat",
        "description": (
            "Coolant exceeded safe operating temperature. URGENT: stop driving "
            "before head gasket damage occurs. Thermostat may be stuck closed."
        ),
    },
    "ect_sensor_frozen": {
        "code": "P0116",
        "name": "ECT Sensor Range / Performance",
        "short": "ECT Stuck",
        "description": (
            "Coolant temperature signal is too stable — engine should show "
            "thermal variance even at operating temp. Sensor likely stuck."
        ),
    },
    "iac_valve_stuck_open": {
        "code": "P0507",
        "name": "Idle Air Control RPM Higher Than Expected",
        "short": "IAC High",
        "description": (
            "Warm idle RPM elevated above normal range. "
            "IAC valve stuck open OR throttle body dirty OR vacuum leak."
        ),
    },
    "alternator_low_output": {
        "code": "P0562",
        "name": "System Voltage Low",
        "short": "Low Volt",
        "description": (
            "CONTROL_MODULE_VOLTAGE sustained below threshold with engine running. "
            "Alternator output low or battery failing."
        ),
    },
}


def get_dtc(label: str) -> dict[str, str]:
    """Return the DTC info for a label; fall back to label name if unmapped."""
    return DTC_MAP.get(label, {
        "code": "—",
        "name": label.replace("_", " ").title(),
        "short": label,
        "description": "",
    })
