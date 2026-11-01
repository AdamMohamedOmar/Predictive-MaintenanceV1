package com.example.obdlogger

// Higher-level OBD operations built on top of the raw Bluetooth link:
// initialise the ELM327, detect which PIDs the car supports, and read a PID.
class ObdLogger(private val bt: BluetoothObd) {

    // Standard ELM327 init sequence. Returns a readable log so you can see what
    // the adapter replied (useful when a clone misbehaves).
    fun initElm(): String {
        val log = StringBuilder()
        log.append("ATZ   -> ").append(bt.send("ATZ", 3000).trim()).append('\n')   // reset
        log.append("ATE0  -> ").append(bt.send("ATE0").trim()).append('\n')        // echo off (important!)
        log.append("ATL0  -> ").append(bt.send("ATL0").trim()).append('\n')        // linefeeds off
        log.append("ATS0  -> ").append(bt.send("ATS0").trim()).append('\n')        // spaces off (simpler parsing)
        log.append("ATSP0 -> ").append(bt.send("ATSP0").trim()).append('\n')       // auto-detect protocol
        log.append("0100  -> ").append(bt.send("0100", 5000).trim()).append('\n')  // wake bus / first query
        return log.toString()
    }

    // Ask the ECU which PIDs it supports via the support bitmasks, then keep only
    // the requested PIDs the car actually has. This replaces any "choose your car"
    // dropdown — the car tells us itself.
    fun detectSupported(requested: List<ObdPid>): List<ObdPid> {
        val supported = mutableSetOf<Int>()
        // 0100 -> PIDs 0x01..0x20, 0120 -> 0x21..0x40, 0140 -> 0x41..0x60
        listOf("0100" to 0x00, "0120" to 0x20, "0140" to 0x40).forEach { (cmd, base) ->
            val bits = parseSupportMask(bt.send(cmd, 3000), cmd.substring(2)) ?: return@forEach
            for (i in 0 until 32) {
                if ((bits shr (31 - i)) and 1L == 1L) supported.add(base + i + 1)
            }
        }
        // If detection failed completely, fall back to trying all requested PIDs.
        if (supported.isEmpty()) return requested
        return requested.filter { supported.contains(it.pidNumber) }
    }

    private fun parseSupportMask(raw: String, pidHex: String): Long? {
        val upper = raw.uppercase()
        if (upper.contains("NO DATA") || upper.contains("ERROR")) return null
        val clean = upper.replace(Regex("[^0-9A-F]"), "")
        val idx = clean.indexOf("41$pidHex")
        if (idx < 0) return null
        val data = clean.substring(idx + 4)
        if (data.length < 8) return null
        return try { data.substring(0, 8).toLong(16) } catch (e: Exception) { null }
    }

    // Read one PID and decode it. Null = unsupported or no data this cycle.
    fun readPid(pid: ObdPid): Double? = Pids.parse(bt.send(pid.command, 1000), pid)
}
