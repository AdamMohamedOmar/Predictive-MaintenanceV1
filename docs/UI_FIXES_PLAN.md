# UI Fix Plan — Dashboard Rendering Bugs

**Scope:** `src/dashboard/app.py` only. Two critical bugs (page errors / unreadable banner) plus two visual polish bugs (orphaned panel cards). No model code, no inference code, no theme constants change.

**Verification baseline:** `streamlit run src/dashboard/app.py` → load `drive1.csv` → press Play. Page must render past warmup without a `ValueError`, and the status banner must display rendered HTML (not source code).

---

## Bug A — Plotly gauge `ValueError: '#10B98118'` (CRITICAL — crashes page)

### Root cause
`_render_severity_grid` builds gauge step colors by string-concatenating an 8-bit alpha byte onto a 6-char hex:

```python
# src/dashboard/app.py lines 630–634
"steps": [
    {"range": [0, 30], "color": ACCENT_OK + "18"},       # "#10B98118"
    {"range": [30, 60], "color": ACCENT_WARN + "18"},
    {"range": [60, 100], "color": ACCENT_ALERT + "18"},
],
```

Inline CSS (used elsewhere in this file) accepts `#RRGGBBAA`. **Plotly does not** — its `gauge.step.color` validator rejects anything that isn't 6-char hex / named / `rgb()` / `rgba()` / `hsl()` / `hsla()`. This crashes every render after the buffer is ready.

### Fix
Add a small hex→rgba helper near the top of `app.py` (after the imports block, before `_FAULT_DISPLAY`):

```python
def _hex_with_alpha(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + alpha 0–1 → 'rgba(r,g,b,a)'.  Plotly-safe."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"
```

Then replace lines 630–634:

```python
"steps": [
    {"range": [0, 30],   "color": _hex_with_alpha(ACCENT_OK,    0.094)},
    {"range": [30, 60],  "color": _hex_with_alpha(ACCENT_WARN,  0.094)},
    {"range": [60, 100], "color": _hex_with_alpha(ACCENT_ALERT, 0.094)},
],
```

(`0x18 / 0xFF ≈ 0.094` — preserves the original visual.)

### Out of scope
**Do NOT touch** any inline-CSS uses of `+ "18"`, `+ "60"`, `+ "40"`, `+ "80"`, `+ "18"` elsewhere (LED box-shadows, banner top-strip glow, sparkline fillcolor — lines 431, 535, 550, 732, 733, etc.). Those go through CSS, which DOES accept 8-char hex with alpha. Only **Plotly-bound colors** need the helper.

### Verify
After fix: `streamlit run src/dashboard/app.py` → load any session → press Play → wait 60s for warmup to complete. The 4 severity gauges must render without exception. Each gauge bar should still show three faint horizontal bands (green / amber / red zones) — confirming the alpha is preserved.

---

## Bug B — Status banner renders as literal HTML text (CRITICAL — unreadable banner)

### Root cause
`_render_status_banner` at lines 524–576 builds a multi-line `banner_html` f-string with **blank lines inside the HTML**:

```python
banner_html = f"""
<div style="...">
  <!-- Accent strip across the top -->
  <div style="height:3px;..."></div>
                                     ← blank line at line 536
  <!-- Content grid: LED | title+sub | stats | elapsed -->
  <div style="
    display:grid;
    ...
  ">
    <!-- LED square... -->
    <div style="...">{...}</div>
                                     ← blank line
    <!-- Title / subtitle -->
    ...
```

Streamlit's markdown parser follows CommonMark: **a blank line terminates an HTML block.** Everything after the first blank line reverts to text/markdown processing — which is why screenshot 2 shows the raw HTML source rendered in a code-block-style font (and `<!--` getting smart-quote-converted to `<!—`).

### Fix
Strip blank lines from the f-string before passing to `st.markdown`. Two options — pick **option 1** (smallest diff):

**Option 1 — one-line collapse (recommended):**
At line 577 (just before `st.markdown(banner_html, ...)`), add:
```python
banner_html = "".join(line for line in banner_html.splitlines() if line.strip())
st.markdown(banner_html, unsafe_allow_html=True)
```

This preserves the f-string for readability but flattens it to a single-line HTML blob before rendering. Markdown can't break a single-line HTML block.

**Apply the same fix to the data-quality banner** at line 454 (the `dq_banner` block, lines 427–453). Even though it doesn't currently contain blank lines, the next refactor could re-introduce them — collapse it the same way for consistency:

```python
dq_banner = "".join(line for line in dq_banner.splitlines() if line.strip())
st.markdown(dq_banner, unsafe_allow_html=True)
```

### Verify
After fix: refresh dashboard before pressing Play. Status banner must display the warming-up state as a styled card (LED square + "WARMING UP" title + stat blocks + session-time readout) — **not** as visible HTML source. After warmup completes, the active "ALL SYSTEMS NOMINAL" banner must also render correctly.

---

## Bug C — SHAP panel card wrapper is orphaned (cosmetic)

### Root cause
Lines 946–950:
```python
st.markdown(f'<div class="panel-card" style="padding:12px 16px;">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)
```

Streamlit auto-closes each `st.markdown` call's HTML internally. The opening `<div>` is closed before `st.plotly_chart` runs, so the chart renders outside the card. The trailing `</div>` is also closed inside its own markdown call — net effect: no wrapper, possibly a stray empty container.

### Fix
Use `st.container(border=True)` (Streamlit 1.29+; we're on 1.38). Replace lines 946–950 with:

```python
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```

This produces a real wrapped card without HTML-injection gymnastics. If the look doesn't match `panel-card`, fall back to wrapping the chart inside a single `st.markdown` only if Streamlit's container border styling can't be themed to match.

### Verify
SHAP panel chart sits inside a bordered card matching the alert-log panel beside it.

---

## Bug D — PID-strip right sub-column has no card (cosmetic)

### Root cause
Lines 745–773 in `_render_pid_strip`. The left `inner_c1` column wraps `left_block` inside a styled `<div>` with `background:{BG_SURFACE};border:1px solid {BORDER};border-radius:8px;...`. The right `inner_c2` column calls `st.plotly_chart(fig, ...)` directly with no wrapper, so the sparkline renders without a card border — visually splitting each PID into "card + floating chart."

### Fix
Wrap the right sub-column the same way as the left, using `st.container(border=True)` for both:

```python
with cols[i % 2]:
    inner_c1, inner_c2 = st.columns([1, 3])
    with inner_c1:
        with st.container(border=True):
            st.markdown(left_block, unsafe_allow_html=True)
    with inner_c2:
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```

Drop the manual `<div style="background:..."` wrapper around `left_block` since `st.container(border=True)` provides the card.

### Verify
Each of the 4 PID cells displays as a unified card with the value/label on the left and the sparkline on the right, both inside the same border.

---

## Execution order (recommended)

1. **Bug A first** — fixes the page crash so you can actually see the rest.
2. **Bug B second** — restores the banner so you can see the layout.
3. **Bug C + D last** — visual polish; verify with a screenshot pass once A & B are done.

## Out of scope (do not touch in this branch)

- Model/inference code (`src/dashboard/inference.py`, `src/features/`, `src/models/`)
- Theme constants (`src/dashboard/theme.py`) — colors are correct, only the consumer is wrong
- Sidebar layout, alert log, recommendations panel — these render correctly per the screenshots
- The 8-character hex pattern in inline CSS — CSS accepts it; only Plotly needs the helper
- Streamlit version bump

## Tests to run after all four fixes

```bash
pytest tests/ -q                          # all 268 tests still pass
streamlit run src/dashboard/app.py        # manual: load drive1.csv, play to 120s
```

Expected: no `ValueError`, banner renders styled, severity gauges show needles + threshold lines + faint background zones, SHAP and PID strips display as cards.
