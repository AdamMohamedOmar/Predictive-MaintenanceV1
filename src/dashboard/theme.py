"""Design token constants for the diagnostic instrument-cluster UI.

This module contains ONLY constants — no logic, no Streamlit calls.
All panel renderers import from here so hex codes never appear in app.py.

Note: this module has no pytest tests by design — constants have nothing to
assert. UI correctness is verified visually (screenshot each pass).
"""

# ── Background ────────────────────────────────────────────────────────────────

BG_BASE     = "#0A0E12"   # page background — near-black, slight blue cast
BG_SURFACE  = "#161B22"   # cards, panels, sidebar backgrounds
BG_RAISED   = "#1F262E"   # hover / active surface, inputs
BORDER      = "#262E38"   # 1px dividers, card outlines
BORDER_STRONG = "#3A4554" # emphasised borders (active state, focused inputs)

# ── Text ──────────────────────────────────────────────────────────────────────

TEXT_PRIMARY   = "#E8E8E5"  # body text, headings — warm white (not pure #FFF)
TEXT_SECONDARY = "#8B95A1"  # captions, axis labels, secondary metadata
TEXT_MUTED     = "#5A6470"  # disabled, placeholders, time prefixes in log

# ── Accent / state colors ─────────────────────────────────────────────────────

ACCENT_DATA  = "#22D3EE"   # neutral data / charts / sparkline lines
ACCENT_OK    = "#10B981"   # healthy state — green
ACCENT_WARN  = "#F59E0B"   # cold_start, warming-up, suspected fault — amber
ACCENT_ALERT = "#EF4444"   # confirmed fault active — red
ACCENT_INFO  = "#60A5FA"   # informational states (connecting, warming up) — blue

# ── State → accent mapping (used in status banner + gauges) ──────────────────

def state_accent(is_fault: bool, is_suspected: bool, is_coldstart: bool) -> str:
    """Return the appropriate accent hex for the current machine state."""
    if is_fault:
        return ACCENT_ALERT
    if is_suspected or is_coldstart:
        return ACCENT_WARN
    return ACCENT_OK

# ── Typography ────────────────────────────────────────────────────────────────

# Google Fonts loaded by inject_global_styles() in styles.py
FONT_DISPLAY = "'Saira Condensed', sans-serif"    # headings, status title, section labels
FONT_BODY    = "'Outfit', system-ui, sans-serif"  # paragraphs, captions, body text
FONT_MONO    = "'JetBrains Mono', ui-monospace, monospace"  # ALL numeric readouts, log lines

# ── Severity tier thresholds ──────────────────────────────────────────────────

SEV_LOW_THRESH  = 0.30   # below → ACCENT_OK
SEV_MID_THRESH  = 0.60   # below → ACCENT_WARN; above → ACCENT_ALERT


def severity_color(severity: float) -> str:
    """Map a [0, 1] severity to the correct accent color."""
    if severity < SEV_LOW_THRESH:
        return ACCENT_OK
    if severity < SEV_MID_THRESH:
        return ACCENT_WARN
    return ACCENT_ALERT
