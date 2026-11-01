package com.example.obdlogger

// One OBD-II PID definition: command we send, response prefix a valid reply
// starts with, data-byte count, unit, a fault-relevance tier, and the decode
// formula. `a`/`b` are the first/second data bytes.
//
// tier 1 = Core (highest engine-fault signal), 2 = Extended (useful context),
// 3 = Full (niche / redundant for engine faults; kept for throughput testing).
data class ObdPid(
    val name: String,
    val command: String,
    val responsePrefix: String,
    val pidNumber: Int,
    val bytes: Int,
    val unit: String,
    val tier: Int,
    val formula: (a: Int, b: Int) -> Double
)

object Pids {

    // Order here = column order in the CSV (all 23 are always written; mode only
    // changes which are actually polled/filled).
    val ALL: List<ObdPid> = listOf(
        // --- Tier 1: Core fault signals ---
        ObdPid("ENGINE_RPM",                        "010C", "410C", 0x0C, 2, "rpm",   1) { a, b -> (a * 256 + b) / 4.0 },
        ObdPid("ENGINE_LOAD",                       "0104", "4104", 0x04, 1, "%",     1) { a, _ -> a * 100.0 / 255 },
        ObdPid("INTAKE_MANIFOLD_PRESSURE",          "010B", "410B", 0x0B, 1, "kPa",   1) { a, _ -> a.toDouble() },
        // Added after your Yaris returned no MAP (010B). MAF is the real airflow
        // channel on a MAF-based car. This is a 24th column beyond your 23-PID
        // training set — delete this line to keep the schema frozen at 23.
        ObdPid("MASS_AIR_FLOW",                     "0110", "4110", 0x10, 2, "g/s",   1) { a, b -> (a * 256 + b) / 100.0 },
        ObdPid("COOLANT_TEMPERATURE",               "0105", "4105", 0x05, 1, "C",     1) { a, _ -> a - 40.0 },
        ObdPid("INTAKE_AIR_TEMP",                   "010F", "410F", 0x0F, 1, "C",     1) { a, _ -> a - 40.0 },
        ObdPid("SHORT_TERM_FUEL_TRIM_BANK_1",       "0106", "4106", 0x06, 1, "%",     1) { a, _ -> (a - 128) * 100.0 / 128 },
        ObdPid("LONG_TERM_FUEL_TRIM_BANK_1",        "0107", "4107", 0x07, 1, "%",     1) { a, _ -> (a - 128) * 100.0 / 128 },
        ObdPid("TIMING_ADVANCE",                    "010E", "410E", 0x0E, 1, "deg",   1) { a, _ -> a / 2.0 - 64 },

        // --- Tier 2: Extended context ---
        ObdPid("VEHICLE_SPEED",                     "010D", "410D", 0x0D, 1, "km/h",  2) { a, _ -> a.toDouble() },
        ObdPid("THROTTLE",                          "0111", "4111", 0x11, 1, "%",     2) { a, _ -> a * 100.0 / 255 },
        ObdPid("FUEL_AIR_COMMANDED_EQUIV_RATIO",    "0144", "4144", 0x44, 2, "ratio", 2) { a, b -> (a * 256 + b) / 32768.0 },
        ObdPid("CONTROL_MODULE_VOLTAGE",            "0142", "4142", 0x42, 2, "V",     2) { a, b -> (a * 256 + b) / 1000.0 },
        ObdPid("ABSOLUTE_BAROMETRIC_PRESSURE",      "0133", "4133", 0x33, 1, "kPa",   2) { a, _ -> a.toDouble() },

        // --- Tier 3: Full (throughput test / completeness) ---
        ObdPid("ENGINE_RUN_TIME",                   "011F", "411F", 0x1F, 2, "s",     3) { a, b -> (a * 256 + b).toDouble() },
        ObdPid("FUEL_TANK",                         "012F", "412F", 0x2F, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("ABSOLUTE_THROTTLE_B",               "0147", "4147", 0x47, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("PEDAL_D",                           "0149", "4149", 0x49, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("PEDAL_E",                           "014A", "414A", 0x4A, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("COMMANDED_THROTTLE_ACTUATOR",       "014C", "414C", 0x4C, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("RELATIVE_THROTTLE_POSITION",        "0145", "4145", 0x45, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
        ObdPid("CATALYST_TEMPERATURE_BANK1_SENSOR1","013C", "413C", 0x3C, 2, "C",     3) { a, b -> (a * 256 + b) / 10.0 - 40 },
        ObdPid("CATALYST_TEMPERATURE_BANK1_SENSOR2","013E", "413E", 0x3E, 2, "C",     3) { a, b -> (a * 256 + b) / 10.0 - 40 },
        ObdPid("COMMANDED_EVAPORATIVE_PURGE",       "012E", "412E", 0x2E, 1, "%",     3) { a, _ -> a * 100.0 / 255 },
    )

    // PIDs to poll for a given capture mode (1=Core, 2=Extended, 3=Full).
    fun forTier(maxTier: Int): List<ObdPid> = ALL.filter { it.tier <= maxTier }

    // Turn a raw ELM327 reply into a number, or null if unsupported / no data.
    fun parse(raw: String, pid: ObdPid): Double? {
        val upper = raw.uppercase()
        if (upper.contains("NO DATA") || upper.contains("ERROR") ||
            upper.contains("UNABLE") || upper.contains("STOPPED") || upper.contains("?")) return null

        val clean = upper.replace(Regex("[^0-9A-F]"), "")
        val idx = clean.indexOf(pid.responsePrefix)
        if (idx < 0) return null

        val data = clean.substring(idx + pid.responsePrefix.length)
        if (data.length < pid.bytes * 2) return null

        return try {
            val a = data.substring(0, 2).toInt(16)
            val b = if (pid.bytes >= 2) data.substring(2, 4).toInt(16) else 0
            pid.formula(a, b)
        } catch (e: Exception) {
            null
        }
    }
}
