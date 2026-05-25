# PID Strip Fix Plan — Round 2

**Context:** UI_FIXES_PLAN.md (round 1) fixed the gauge `ValueError` and the banner blank-line bug, but two issues remain in the **Live PID Readings** section. Screenshot confirms: page renders correctly through the gauges, then crashes at the PID strip with a new Plotly hex error. Visual state appears frozen because every Streamlit rerun re-crashes at the same point.

**Scope:** `src/dashboard/app.py` only. One critical bug (page crash) + one layout regression from round 1 (over-corrected card wrapping).

---

## Bug E — Plotly Scatter `fillcolor` 8-char hex crash (CRITICAL — page crash)

### Root cause
`_render_pid_strip` at line **750**:
```python
fig = go.Figure(
    go.Scatter(
        y=series.values,
        mode="lines",
        line={"color": line_color, "width": 1.8},
        fill="tozeroy",
        fillcolor=line_color + "18",      # ← BROKEN: "#22D3EE18" rejected by Plotly
    )
)
```

This is the **same root cause** as Bug A in round 1 (Plotly's color validator rejects 8-char hex `#RRGGBBAA`), but in a different Plotly property (`Scatter.fillcolor` instead of `Indicator.gauge.steps[].color`). Round 1's plan said "*only Plotly-bound colors need the helper*" — that rule was correct, but the implementation only patched the gauge. The sparkline fillcolor was missed.

The error fires the moment `_render_pid_strip` is called with a non-empty history (which happens on the very first frame after Play). The error trace replaces the rest of the page (SHAP, recommendations, sidebar session-time readout), which is why the user sees "the bars won't rise / all is static" — every Streamlit rerun crashes at this same line, the error pinwheel replaces the live content, and nothing below the PID strip header updates.

### Fix
Replace line 750:

**Before:**
```python
                fillcolor=line_color + "18",
```

**After:**
```python
                fillcolor=_hex_with_alpha(line_color, 0.094),
```

The `_hex_with_alpha` helper already exists (added in round 1). No new helper needed.

### Audit (confirms no other Plotly-bound 8-char hex remain)

A grep of all `go.Figure / go.Scatter / go.Bar / go.Indicator` blocks and their `color` / `fillcolor` / `bgcolor` parameters confirms only this one site is broken. Every other Plotly color in the file consumes a base 6-char constant (`ACCENT_DATA`, `TEXT_PRIMARY`, `BG_SURFACE`, etc.) directly. The remaining `+ "60"`, `+ "40"`, `+ "80"` patterns in the file are all inside inline CSS in `st.markdown` calls (LED box-shadows, banner top-strip glow, alert-log tag borders) — those go through CSS, which accepts 8-char hex and is unaffected.

### Verify
After fix: load any usable session file (see "Data note" below), press Play. PID strip must render 4 sparkline cards without `ValueError`. Sparkline trace lines should still have the faint background fill visible (confirms alpha preserved — `0x18 / 0xFF ≈ 0.094`).

---

## Bug D-v2 — PID-strip layout: one unified card per PID, not two (regression from round 1)

### Root cause
Round 1's fix wrapped **both** inner sub-columns in their own `st.container(border=True)`:
```python
with cols[i % 2]:
    inner_c1, inner_c2 = st.columns([1, 3])
    with inner_c1:
        with st.container(border=True):    # ← border #1 (around text)
            st.markdown(left_block, ...)
    with inner_c2:
        with st.container(border=True):    # ← border #2 (around chart)
            st.plotly_chart(fig, ...)
```

The docstring at line 695 reads "*Card layout (per PID): Left column (fixed 110px) ... Right column (flex)*" — clearly one card per PID with two columns *inside* it, not two side-by-side cards. The round-1 fix solved "right side has no card" by giving the right side its own card — but that produced **two disconnected boxes per PID**, eight bordered boxes total in the 2×2 grid. Visually worse than the original half-card layout.

### Fix
Wrap the row in a **single outer** `st.container(border=True)` and put both inner columns inside it with **no inner borders**. Replace lines 763–790 (the `with cols[i % 2]:` block):

**Before:**
```python
        with cols[i % 2]:
            # 2-column nested layout: text left, sparkline right
            # Both sides use st.container(border=True) for consistent card styling.
            # Previously the left used a raw <div> wrapper while the right had none —
            # this produced a half-card with a floating chart beside it.
            inner_c1, inner_c2 = st.columns([1, 3])
            with inner_c1:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="min-width:110px;padding:4px 0;">'
                        f'<div style="font-family:{FONT_DISPLAY};font-size:9px;'
                        f"text-transform:uppercase;letter-spacing:0.1em;"
                        f'color:{TEXT_MUTED};">{label}</div>'
                        f'<div style="font-family:{FONT_MONO};font-size:28px;'
                        f'font-weight:700;color:{TEXT_PRIMARY};line-height:1.1;">'
                        f"{val_str}</div>"
                        f'<div style="font-family:{FONT_BODY};font-size:10px;'
                        f'color:{TEXT_SECONDARY};">{unit}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            with inner_c2:
                with st.container(border=True):
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
```

**After:**
```python
        with cols[i % 2]:
            # One bordered card per PID, with two columns *inside* the card:
            # value/label on the left, sparkline on the right.  Round 1's fix
            # mistakenly put a border around each sub-column, producing two
            # disconnected boxes per PID — the docstring above describes one
            # unified card.
            with st.container(border=True):
                inner_c1, inner_c2 = st.columns([1, 3])
                with inner_c1:
                    st.markdown(
                        f'<div style="min-width:110px;padding:4px 0;">'
                        f'<div style="font-family:{FONT_DISPLAY};font-size:9px;'
                        f"text-transform:uppercase;letter-spacing:0.1em;"
                        f'color:{TEXT_MUTED};">{label}</div>'
                        f'<div style="font-family:{FONT_MONO};font-size:28px;'
                        f'font-weight:700;color:{TEXT_PRIMARY};line-height:1.1;">'
                        f"{val_str}</div>"
                        f'<div style="font-family:{FONT_BODY};font-size:10px;'
                        f'color:{TEXT_SECONDARY};">{unit}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with inner_c2:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
```

The only structural change: the `with st.container(border=True):` line moves **outside** the inner column split, and the two `with st.container(border=True):` lines inside the sub-columns are **removed**. Inner markdown / plotly_chart bodies are unchanged.

### Verify
After fix: PID strip renders 4 PID rows in a 2×2 grid. Each row is **one** bordered card containing a labeled value on the left and a sparkline on the right, both sharing the same border. The 4 cards should align with consistent height across the grid.

---

## Execution order

1. **Bug E first** — without this, you can't see the layout result (page crashes).
2. **Bug D-v2 second** — visual polish; verifiable only after E lands.

---

## Data note (NOT a code bug — flag for the user, do not fix)

The screenshot showed `drive13.csv` selected with a **SENSOR DATA INVALID** banner reporting `SHORT_TERM_FUEL_TRIM_BANK_1=100.0` (outside `[-30, 30]`) and `TIMING_ADVANCE=200.9` (outside `[-30, 60]`). This is **correct behavior**:

- `CLAUDE.md` documents that of the 129 carOBD files, only **9 are usable**: `drive1.csv` and `live5.csv`–`live12.csv`. The rest have firmware-encoding bugs in `TIMING_ADVANCE` and `STFT`.
- `drive13.csv` is one of the broken files. STFT=100% and TIMING_ADVANCE=200° are impossible — those are the firmware encoding bugs.
- The `check_row()` sanity gate in `src/dashboard/sanity.py` is correctly rejecting these rows and holding the inference state. That's why the gauges stay at 0% — inference never runs on garbage data.
- The user should pick `drive1.csv`, `live5.csv`, ... `live12.csv` for the demo.

**Optional UX improvement (out of scope for this branch):** filter the sidebar `Session file` dropdown (around line 222 in `app.py`) to show only the 9 usable files, or annotate the unusable ones (e.g. `drive13.csv (broken — firmware bug)`). Skip this for now — the data-quality banner already explains why the screen looks frozen on bad files. Decide whether to add the filter in a follow-up.

---

## Tests to run after both fixes

```bash
.venv/Scripts/pytest.exe tests/ -q                 # all 268 tests still pass
.venv/Scripts/streamlit.exe run src/dashboard/app.py
# manual: load drive1.csv, press Play, watch for 90+ seconds
```

Expected after fixes:
- No `ValueError` traceback anywhere on the page.
- PID strip renders 4 unified bordered cards in a 2×2 grid.
- Sparkline traces update as the streamer advances — RPM and throttle should visibly change; coolant should slowly climb.
- Severity gauges remain at 0% on healthy `drive1.csv` data (correct) but the SHAP panel and recommendations panel below should now render (they were being hidden by the crash).
- Sidebar session-time readout updates each second (was being hidden by the crash).
