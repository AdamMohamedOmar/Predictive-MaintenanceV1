# UI Redesign Plan — "Diagnostic Instrument Cluster"

**For Sonnet to execute.** Cold-start friendly: includes everything needed without conversation history.

---

## 0. Context & Constraints

This is the Streamlit dashboard for a predictive maintenance graduation capstone (Toyota Etios training, Skoda Roomster validation, 15 June deadline). The current UI works but looks like a generic Streamlit prototype. We're rebuilding the visual language to look like a real diagnostic tool worth demoing to a panel.

**Read these first:**
- `CLAUDE.md` — project rules (precise diffs, comments-for-why-only, no whole-file rewrites)
- `src/dashboard/app.py` — current dashboard (~530 lines, single file, no theme system yet)
- `src/dashboard/inference.py` — `DashboardState` shape (what we render)
- The `frontend-design` skill (already installed locally) — apply its taste rules: distinctive typography, committed aesthetic, no generic AI-slop colors. Treat its guidance as the design north star, but **adapt** to Streamlit's HTML-via-`st.markdown(unsafe_allow_html=True)` constraints — there is no React/Motion library here.

**Decisions already made (do NOT relitigate):**

| Decision | Locked answer |
|---|---|
| Aesthetic direction | **Instrument cluster (dark)** — Bloomberg-terminal × automotive HUD |
| New dependency | **Add `plotly==5.24.1`** to `requirements.txt` |
| Scope | **Hero panels first** — 3 independently-shippable passes |

---

## 1. Design System (single source of truth)

These tokens drive every visual choice. Create them in `src/dashboard/theme.py` in Pass 1. Reference them by name everywhere — never inline a hex code in a panel renderer.

### Colors
```python
# src/dashboard/theme.py
BG_BASE        = "#0A0E12"   # page background — near-black, slight blue
BG_SURFACE     = "#161B22"   # cards, panels, sidebar
BG_RAISED      = "#1F262E"   # hover / active surface
BORDER         = "#262E38"   # 1px dividers
BORDER_STRONG  = "#3A4554"   # emphasised borders (top of status banner, focused inputs)

TEXT_PRIMARY   = "#E8E8E5"   # body text, headings — warm white, NOT pure white
TEXT_SECONDARY = "#8B95A1"   # captions, axis labels, secondary metadata
TEXT_MUTED     = "#5A6470"   # disabled, placeholders

ACCENT_DATA    = "#22D3EE"   # neutral data / charts / sparklines
ACCENT_OK      = "#10B981"   # healthy state
ACCENT_WARN    = "#F59E0B"   # cold_start, suspected-but-not-confirmed
ACCENT_ALERT   = "#EF4444"   # confirmed fault
ACCENT_INFO    = "#60A5FA"   # informational (warming up, connecting)
```

### Typography
```python
# Google Fonts — loaded via styles.py
FONT_DISPLAY = "'Saira Condensed', sans-serif"   # headings, status banner, page title
FONT_BODY    = "'Outfit', system-ui, sans-serif" # paragraphs, labels, captions
FONT_MONO    = "'JetBrains Mono', ui-monospace, monospace"  # ALL numeric readouts, log lines, SHAP feature names
```

**Rule:** every number on screen (RPM, %, seconds, confidence) renders in `FONT_MONO`. Every heading or section title renders in `FONT_DISPLAY` with `letter-spacing: 0.04em; text-transform: uppercase`. Body text in `FONT_BODY`.

### Spacing scale
4 / 8 / 12 / 16 / 24 / 32 / 48 px. No other values. Surface radius = 8 px. Panel padding = 16 px.

---

## 2. Pass-by-pass execution

Each pass ends in a visually verifiable state. The user (Adam) will screenshot after each pass — if it looks wrong, fix before moving on.

---

### Pass 1 — Foundation + Status Banner

**Goal:** Global dark theme applied; status banner becomes the new visual signature of the app.

**Step 1.1 — Add plotly**

Append to `requirements.txt`:
```
# Dashboard charts (gauges, dark-themed live plots)
plotly==5.24.1
```

Then `./.venv/Scripts/python.exe -m pip install plotly==5.24.1`. Do NOT touch other pins.

**Step 1.2 — Streamlit global theme**

Create `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#22D3EE"
backgroundColor = "#0A0E12"
secondaryBackgroundColor = "#161B22"
textColor = "#E8E8E5"
font = "sans serif"

[server]
runOnSave = true
```

This handles Streamlit's own chrome (sidebar, widgets, default chart axes). Our custom CSS layers on top.

**Step 1.3 — `src/dashboard/theme.py`**

Create with the constants from §1 above. Export everything. No logic, just constants.

**Step 1.4 — `src/dashboard/styles.py`**

Create with a single function `inject_global_styles()` that calls `st.markdown(..., unsafe_allow_html=True)` once at app startup. The CSS string must:

1. `@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@500;700&family=Outfit:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');`
2. Override `body`, `.stApp`, `.main`, `.block-container` background and text colors using the tokens.
3. Style `h1, h2, h3, h4` with `FONT_DISPLAY`, uppercase, letter-spacing.
4. Style `code, pre, .stCode` with `FONT_MONO`.
5. Restyle `.stButton > button` — dark surface, border, monospace, hover state shifting to `BG_RAISED` + `ACCENT_DATA` border.
6. Restyle `.stSlider`, `.stSelectbox`, `.stRadio` containers — surface backgrounds, border = `BORDER`, accent color for active track = `ACCENT_DATA`.
7. Add a utility class `.panel-card` (surface bg, border, 8px radius, 16px padding) we'll use in later passes.
8. Hide Streamlit's default "Made with Streamlit" footer and the hamburger menu (`#MainMenu, footer { visibility: hidden; }`).

Call `inject_global_styles()` at the top of `main()` in `app.py`, immediately after `_init_session_state()`.

**Step 1.5 — Status banner rewrite**

Replace `_render_status_banner` in `app.py`. The new banner is a single full-width panel with FOUR vertical lanes laid out horizontally:

```
┌─ ICON LANE ──┬─ TITLE / SUBTITLE ────────────┬─ KEY METRICS ───────────┬─ TIME LANE ──┐
│   ⬢          │  FAULT — FUEL SYSTEM          │  CONF  87%              │  ELAPSED     │
│   (40px      │  3 of 3 windows agreed        │  WINDOWS  3/3           │  04:12       │
│    accent)   │                               │  SEVERITY  68% ↑        │  REGIME      │
│              │                               │                         │  cruise      │
└──────────────┴───────────────────────────────┴─────────────────────────┴──────────────┘
```

- **Icon lane** — a 40px tall colored square (the state color: `ACCENT_OK` / `ACCENT_WARN` / `ACCENT_ALERT` / `ACCENT_INFO`), no glyph needed (the color IS the icon — "live status LED" metaphor). The square has a soft glow: `box-shadow: 0 0 24px <accent>40` (the trailing `40` is alpha).
- **Title / subtitle** — `FONT_DISPLAY`, uppercase, big (22px) title; `FONT_BODY` 13px subtitle line below.
- **Key metrics** — three small stat blocks. Each is a tiny uppercase `FONT_DISPLAY` label (10px, muted) above a `FONT_MONO` value (18px, primary text). Metrics shown depend on state — when alert active: confidence / windows / current-severity. When healthy: classifier confidence / dominant class. When warming up: rows collected / target.
- **Time lane** — elapsed session time in `FONT_MONO` (24px), plus the current driving regime (e.g. "cruise", "idle") from the latest row if we can infer it (use a feature column like `REGIME_*` if it exists in `DashboardState.latest_row`, else hide).

Implementation: build a single HTML string with inline CSS (we don't have a separate CSS file system in Streamlit). Use CSS Grid for the lane layout (`display:grid; grid-template-columns: 56px 1fr auto auto; gap: 24px;`).

The whole banner sits on `BG_SURFACE` with a 1px `BORDER` line, 8px radius, and a top-edge accent strip 3px tall in the state color (creates the "this band is the alert" reading).

**Verification for Pass 1:**
- Page background near-black, fonts loaded (open devtools, check `Saira Condensed` is applied to headings).
- Status banner full-width, four lanes, accent strip visible at top in the right color for each state.
- Sidebar background = `BG_SURFACE`, not Streamlit's default gray.
- Buttons restyled — dark, monospace, hover shifts color.
- No regressions: play a CSV file, banner transitions through warming-up → healthy → (if injected) fault active.

---

### Pass 2 — Severity Gauges + PID Strip

**Goal:** Replace the flat HTML progress bars and Streamlit's default line charts with plotly visuals tuned to our dark theme.

**Step 2.1 — Severity gauge (plotly)**

Replace `_render_severity_grid` in `app.py`. Render 4 columns, each containing one plotly figure built with `go.Indicator`:

```
mode="gauge+number+delta"
value=current_severity_pct
delta={"reference": forecast_severity_pct, "relative": False, "suffix": " forecast"}
gauge={
    "shape": "angular",          # half-circle dial
    "axis": {"range": [0, 100], "tickwidth": 0, "tickfont": {"color": TEXT_MUTED, "size": 9}},
    "bar": {"color": <accent based on severity tier>},
    "bgcolor": BG_SURFACE,
    "borderwidth": 0,
    "steps": [
        {"range": [0, 30],  "color": "#10B98115"},   # green tier (15% alpha)
        {"range": [30, 60], "color": "#F59E0B15"},   # amber tier
        {"range": [60, 100],"color": "#EF444415"},   # red tier
    ],
    "threshold": {                 # 60-second forecast marker
        "line": {"color": ACCENT_DATA, "width": 3},
        "thickness": 0.85,
        "value": forecast_severity_pct,
    },
}
```

Tier colors:
- severity < 0.30 → `ACCENT_OK`
- 0.30 ≤ severity < 0.60 → `ACCENT_WARN`
- severity ≥ 0.60 → `ACCENT_ALERT`

Apply to every gauge: `fig.update_layout(paper_bgcolor=BG_SURFACE, font={"color": TEXT_PRIMARY, "family": "JetBrains Mono"}, margin={"l":20,"r":20,"t":40,"b":20}, height=200)`.

Above each gauge, render the fault name as an uppercase `FONT_DISPLAY` title (e.g. "FUEL SYSTEM") in a small div — do this with `st.markdown(..., unsafe_allow_html=True)` so we control the spacing exactly.

Below each gauge, a 2-line caption in `FONT_MONO`: "NOW · 47%" and "+60s · 63% ▲" (with up/down/flat triangle for delta sign).

The plotly `threshold` line acts as the visual "where we'll be in 60 s" marker — much more intuitive than two separate numbers.

**Step 2.2 — PID strip (plotly sparklines)**

Replace `_render_pid_strip`. The previous version showed 4 PIDs in 2 side-by-side default line charts. New version: a 2×2 grid of dark-themed sparklines, each with the current value displayed large to the left of the chart.

For each PID:
```
┌────────────┬────────────────────────────────────────────┐
│  ENGINE    │                                            │
│  RPM       │     /\          /\__/\                     │
│            │  __/  \____/\__/      \___                 │
│  2143      │                                            │
│  rpm       │                                            │
└────────────┴────────────────────────────────────────────┘
```

- Left column (fixed 110px): label in uppercase `FONT_DISPLAY` 11px muted; big current value in `FONT_MONO` 28px primary text; unit in `FONT_BODY` 11px muted.
- Right column: plotly sparkline (`go.Scatter`, mode="lines", line.color=`ACCENT_DATA`, line.width=1.5). Height 90px. No axes, no grid, no legend. `fig.update_xaxes(visible=False); fig.update_yaxes(visible=False); fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor=BG_SURFACE, plot_bgcolor=BG_SURFACE)`.
- Each sparkline sits on a `.panel-card` (BG_SURFACE, border, radius).
- The 4 PIDs to show: `ENGINE_RPM`, `COOLANT_TEMPERATURE`, `LONG_TERM_FUEL_TRIM_BANK_1`, `THROTTLE`. Keep the existing pair-split or rearrange to 2×2 — your call, but document which.

When the deque is empty, render the same card layout with a placeholder ("—") so the page doesn't jump when data starts flowing.

**Verification for Pass 2:**
- Four radial gauges fill the row, each animates smoothly as severity changes, threshold line marks 60s forecast.
- PID strip cards have stable layout, sparklines update without flicker, current value is the most prominent thing.
- Inject a faulty CSV (`live5_air.csv` or similar) and watch the air_system gauge climb past 30% → color shifts to amber, then past 60% → red.

---

### Pass 3 — Alert Log + SHAP + Sidebar polish

**Goal:** Finish the secondary panels and the sidebar so nothing looks "out of place" relative to the redesigned hero panels.

**Step 3.1 — Alert log as "diagnostic feed"**

Replace `_render_alert_log`. New design: a scrollable monospace feed (max-height 320px, `overflow-y: auto`) with each entry as a single HTML row containing:

```
[ 0124 s ]  ▸  ML ALERT   fuel system    87% conf · 3 windows
[ 0118 s ]  ▸  RULE       thermostat_stuck_open
[ 0042 s ]  ✓  CLEARED    system healthy
```

- Render each entry with `st.markdown(unsafe_allow_html=True)` inside a single parent div with the `.panel-card` class.
- Time prefix (`[ 0124 s ]`) in `FONT_MONO` `TEXT_MUTED`.
- Marker glyph (▸ for new alert, ✓ for cleared, ⚠ for rule) — color from accent palette.
- Category tag (`ML ALERT`, `RULE`, `CLEARED`) in `FONT_DISPLAY` uppercase 10px, with a tiny 1px border around it, color-coded.
- Detail in `FONT_BODY` `TEXT_PRIMARY`.

To classify an entry's type, parse the existing log strings — they're already prefixed with "ML ALERT —", "RULE —", or "Alert cleared". This keeps the data flow untouched.

Newest entries at top (current behavior). Cap at 15 visible. Each row gets a faint bottom border (`1px solid BORDER`) except the last.

**Step 3.2 — SHAP panel**

Replace `_render_shap_panel`. Use plotly `go.Bar` (horizontal):
- `orientation="h"`, sorted by absolute SHAP value descending.
- Bar colors: positive SHAP (pushed prediction toward this class) = `ACCENT_DATA`; negative = `TEXT_MUTED`.
- Y-axis tick labels in `FONT_MONO` 11px, no axis line.
- X-axis hidden, but show the SHAP value as text at the end of each bar (`text=values, textposition="outside"`).
- Title strip above the chart: "PREDICTING — FUEL SYSTEM · 87% CONFIDENCE" in `FONT_DISPLAY` uppercase. Style as a 1-line header bar inside the same `.panel-card`.

Same dark layout settings as Pass 2 gauges.

**Step 3.3 — Sidebar polish**

The buttons and selectboxes inherit the global styles from Pass 1, so most of the work is already done. Remaining touch-ups in `_render_sidebar`:

- Replace `st.sidebar.header(":wrench: Session Controls")` with a custom HTML block using `FONT_DISPLAY` uppercase, 14px, letter-spacing 0.06em. Include a tiny colored square next to the title (`ACCENT_DATA`) as a brand mark.
- Replace `st.sidebar.divider()` with a custom 1px line using `BORDER`.
- The progress bar (`st.sidebar.progress`) — restyle via CSS in `styles.py`: track = `BG_RAISED`, fill = `ACCENT_DATA`.
- Connection status banners (`st.sidebar.success`, `st.sidebar.warning`, `st.sidebar.error`) — wrap in custom HTML so they use our color palette instead of Streamlit's default green/yellow/red boxes. Match the status-banner color tokens.
- "Elapsed" metric at the bottom — render as a custom mono readout (`FONT_MONO` 22px, `TEXT_PRIMARY`) with a `FONT_DISPLAY` label above ("SESSION TIME"). Drop the default `st.sidebar.metric`.

**Verification for Pass 3:**
- Alert log feels like a terminal feed, color-coded, monospace, time-prefixed.
- SHAP bars are dark, sorted, labeled, fit inside a card matching the other panels.
- Sidebar looks like part of the same product — no Streamlit-default chrome leaking through.

---

## 3. File map (what gets created vs edited)

| File | Action | Notes |
|---|---|---|
| `requirements.txt` | +1 line | `plotly==5.24.1` |
| `.streamlit/config.toml` | NEW | Streamlit native theme |
| `src/dashboard/theme.py` | NEW | Color + font constants only |
| `src/dashboard/styles.py` | NEW | `inject_global_styles()` + any HTML builder helpers |
| `src/dashboard/app.py` | EDIT (targeted) | Each panel renderer replaced one at a time. Do NOT rewrite the whole file in one shot — that violates `CLAUDE.md` rules. |
| `docs/UI_REDESIGN_PLAN.md` | (this file) | Reference, no edits |

No changes to `inference.py`, `streamer.py`, or anything outside `dashboard/`. The data contract (`DashboardState`) stays identical.

---

## 4. Anti-patterns to avoid

These are the failure modes the `frontend-design` skill explicitly warns about, mapped to this project:

1. **Don't reach for Inter / Roboto / Arial / system fonts.** They're banned. We're using Saira Condensed + Outfit + JetBrains Mono.
2. **Don't use purple gradients on white.** Trivially avoided — we're going dark.
3. **Don't scatter micro-animations everywhere.** One well-orchestrated transition (e.g. gauge needle easing) > a dozen jittery hovers.
4. **Don't inline hex codes in panel renderers.** Always import from `theme.py`. If you find yourself typing `#22D3EE` outside `theme.py` or `styles.py`, stop.
5. **Don't add emoji to fault names or section headings.** The aesthetic is "diagnostic instrument" — emoji breaks the tone. The wrench in `st.set_page_config` is fine (it's the favicon); kill it everywhere else.
6. **Don't whole-file-rewrite `app.py`.** Replace one renderer at a time, run the app between, screenshot, move on.
7. **Don't add tests for the UI layer.** Streamlit UI is verified by eye, not pytest. `CLAUDE.md` says "all new modules get at least one pytest test" but `theme.py` (constants) and `styles.py` (CSS string + Streamlit calls) are exempt — they have no logic to test. Note this in the module docstring.

---

## 5. Verification checklist

After all 3 passes, the user should be able to:

- [ ] Open the app on a projector (or low-brightness screen) and have it look intentional, not "default Streamlit".
- [ ] Show a screenshot to someone unfamiliar with the project and have them guess "automotive diagnostic tool" without being told.
- [ ] Read every number on screen in `JetBrains Mono`, every heading in `Saira Condensed`.
- [ ] Watch a fault inject and see: status banner color flip → severity gauge climb past threshold → alert log row appear at top → SHAP bars reshuffle. The story reads top-to-bottom on a single glance.
- [ ] Re-run `pytest tests/` — still 250/250 green (no UI tests added, no regressions to the data layer).

---

## 6. Decision log (for Sonnet, while executing)

Things you may need to decide mid-flight without going back to Adam:

| If you encounter… | Do this |
|---|---|
| A Google Font fails to load offline at the demo | Add fallback: `font-family: 'Saira Condensed', 'Bebas Neue', sans-serif;` (Bebas Neue is bundled in many systems). Mention in the commit message. |
| Plotly gauges feel too "dial-y" / over-the-top | Switch `gauge.shape` from `"angular"` to `"bullet"` (horizontal bullet chart). Keep the threshold marker. |
| The status banner runs out of horizontal space on a small window | Stack lanes vertically below 900 px viewport using a CSS media query inside `styles.py`. |
| You need to add CSS for a Streamlit internal class but the class name changes between versions | Hard-code the class name and add a comment with the Streamlit version (`1.38.0`). Future-Adam will fix when they upgrade. |
| You're tempted to add a new viz library | Stop. Plotly + matplotlib + Streamlit's built-ins are enough. |

---

## 7. Out of scope (do not do these without explicit approval)

- Rewriting the sidebar's information architecture (file picker, source toggle, etc.)
- Adding new dashboard sections (vehicle vitals strip, diagnostic timeline, OBD-II raw view)
- Replacing Streamlit with a React/Vue frontend
- Animating the page-load with staggered reveals (nice-to-have, not in this scope)
- Mobile responsiveness — this is a laptop-on-projector demo tool
- Dark-light mode toggle — we committed to dark

---

End of plan.
