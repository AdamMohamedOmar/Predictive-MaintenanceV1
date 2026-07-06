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

# ── Typography ────────────────────────────────────────────────────────────────

# Google Fonts loaded by inject_global_styles() in styles.py
FONT_DISPLAY = "'Saira Condensed', sans-serif"    # headings, status title, section labels
FONT_BODY    = "'Outfit', system-ui, sans-serif"  # paragraphs, captions, body text
FONT_MONO    = "'JetBrains Mono', ui-monospace, monospace"  # ALL numeric readouts, log lines

# ── Severity tiers (severity strip + end-of-read report) ─────────────────────

# Deliberately softer than ACCENT_OK / ACCENT_WARN: these colour steady-state
# status readouts (bars, verdict text), not attention-demanding alerts.
SEVERITY_OK      = "#3FB27F"   # green — severity near 0, healthy verdicts
SEVERITY_CAUTION = "#C9A227"   # amber — mid severity, inconclusive verdicts

SEV_CAUTION_THRESH = 0.33   # at/above → SEVERITY_CAUTION
SEV_ALERT_THRESH   = 0.66   # at/above → ACCENT_ALERT


def severity_color(severity: float) -> str:
    """Map a [0, 1] severity to the severity-strip tier colour."""
    if severity >= SEV_ALERT_THRESH:
        return ACCENT_ALERT
    if severity >= SEV_CAUTION_THRESH:
        return SEVERITY_CAUTION
    return SEVERITY_OK
