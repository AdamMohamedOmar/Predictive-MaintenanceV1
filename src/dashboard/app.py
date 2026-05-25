"""Streamlit live dashboard — Predictive Maintenance.

Run with:
    streamlit run src/dashboard/app.py

Layout (top to bottom)
-----------------------
  Sidebar      — file selector, speed slider, play/pause, reset
  Status banner— full-width colour-coded alert status
  Severity grid— 4 fault columns (current + 60s forecast)
  PID strip    — 4 live sensor channels over a rolling 5-min window
  Alert log    — timestamped ML + rule-engine events
  SHAP panel   — top-5 features driving the current prediction

Loop design
-----------
Each Streamlit rerun advances exactly ONE simulated second (one row).
After rendering, the script calls st.rerun() after sleeping 1/speed
seconds — so at speed=10 we advance 10 rows per real second.
This gives smooth animation without batching.

@st.cache_resource
------------------
InferenceEngine loads the XGBoost model and builds the SHAP
TreeExplainer once per Streamlit server process (survives reruns,
survives multiple browser tabs).  Between CSV files the engine is
explicitly reset() — the cached object persists, its internal state
is flushed.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

# When launched via `streamlit run src/dashboard/app.py` (or directly with
# python), the project root is NOT automatically on sys.path.  Add it so
# `from src.xxx import` works regardless of how the script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import serial.tools.list_ports
import streamlit as st

from src.config import MODELS_DIR, USEFUL_PIDS
from src.dashboard.inference import DashboardState, InferenceEngine
from src.dashboard.streamer import CsvStreamer
from src.dashboard.styles import inject_global_styles
from src.dashboard.theme import (
    ACCENT_DATA,
    ACCENT_OK,
    ACCENT_WARN,
    ACCENT_ALERT,
    ACCENT_INFO,
    BG_BASE,
    BG_SURFACE,
    BG_RAISED,
    BORDER,
    BORDER_STRONG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    FONT_DISPLAY,
    FONT_BODY,
    FONT_MONO,
    severity_color,
    state_accent,
)
from src.live.obd_source import LiveObdSource
from src.models.classifier import ALL_LABELS
from src.models.forecaster import FAULT_TYPES

# ── Page-level config (must be first Streamlit call) ─────────────────────────

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon=":wrench:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "raw" / "carOBD"

# How many rows to keep for the live PID chart (~5 min at 1 Hz)
_HISTORY_LEN = 300

# The four PIDs shown in the live chart.  Two per panel to keep charts readable.
_LEFT_PIDS = ["ENGINE_RPM", "COOLANT_TEMPERATURE"]
_RIGHT_PIDS = ["LONG_TERM_FUEL_TRIM_BANK_1", "THROTTLE"]

_FAULT_DISPLAY = {
    "air_system": "Air System",
    "fuel_system": "Fuel System",
    "coolant_temp_sensor": "Coolant Sensor",
    "throttle_position_sensor": "TPS",
}


# ── Cached InferenceEngine (expensive: SHAP TreeExplainer built once) ─────────


@st.cache_resource
def _load_engine(normalizer_path: str | None = None) -> InferenceEngine | None:
    """Load XGBoost + SHAP + FaultForecaster.  Returns None if models missing.

    Keyed by normalizer_path so switching baselines (Etios → Skoda) rebuilds
    the engine once and then caches the new instance.
    """
    try:
        from pathlib import Path

        override = Path(normalizer_path) if normalizer_path else None
        return InferenceEngine(normalizer_override=override)
    except FileNotFoundError:
        return None


# ── Session state bootstrap ───────────────────────────────────────────────────


def _init_session_state() -> None:
    defaults = {
        "source_type": "csv",  # "csv" | "live"
        "streamer": None,  # CsvStreamer | None  (csv mode)
        "live_source": None,  # LiveObdSource | None  (live mode)
        "normalizer_path": None,  # str | None — path to active normalizer pkl
        "playing": False,
        "pid_history": deque(maxlen=_HISTORY_LEN),
        "latest_state": None,  # DashboardState | None
        "alert_log": [],  # list[str]  — timestamped event strings
        "last_active_fault": "",  # track transitions for alert log
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar(engine: InferenceEngine | None) -> tuple[str | None, float]:
    """Render controls.  Returns (normalizer_path, speed).  Handles all interactions."""
    st.sidebar.markdown(
        f'<div style="font-family:{FONT_DISPLAY};font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:{TEXT_MUTED};padding:4px 0 8px 0;">'
        f'<span style="display:inline-block;width:8px;height:8px;background:{ACCENT_DATA};'
        f'border-radius:2px;margin-right:8px;vertical-align:middle;"></span>'
        f"Session Controls</div>",
        unsafe_allow_html=True,
    )

    # ── Source selector ───────────────────────────────────────────────────────
    source_type = st.sidebar.radio(
        "Source",
        ["CSV replay", "Live vehicle"],
        index=0 if st.session_state.source_type == "csv" else 1,
        horizontal=True,
    )
    new_source_type = "csv" if source_type == "CSV replay" else "live"

    # Stop live source if user switched away from live mode
    if new_source_type != st.session_state.source_type:
        if st.session_state.live_source is not None:
            st.session_state.live_source.stop()
            st.session_state.live_source = None
        st.session_state.playing = False
        st.session_state.source_type = new_source_type

    st.sidebar.divider()

    # ── Normalizer picker (shared by both modes) ──────────────────────────────
    norm_files = sorted(MODELS_DIR.glob("*_normalizer.pkl"))
    norm_options = ["Built-in (training baseline)"] + [p.name for p in norm_files]
    norm_index = st.sidebar.selectbox("Normalizer", norm_options, index=0)
    if norm_index == "Built-in (training baseline)":
        normalizer_path = None
    else:
        normalizer_path = str(MODELS_DIR / norm_index)

    # Warn if live mode selected but no vehicle normalizer exists
    if new_source_type == "live" and not norm_files:
        st.sidebar.warning(
            "No vehicle normalizer found.  Run:\n"
            "`python -m scripts.live_baseline_capture`"
        )

    # Store normalizer choice so main() can key the engine cache
    st.session_state.normalizer_path = normalizer_path

    speed = 1.0  # default; overridden in CSV section below

    # ── CSV mode controls ─────────────────────────────────────────────────────
    if new_source_type == "csv":
        if not _DATA_DIR.exists():
            st.sidebar.error(f"Data directory not found:\n{_DATA_DIR}")
            return normalizer_path, 1.0

        csv_files = sorted(_DATA_DIR.glob("*.csv"))
        if not csv_files:
            st.sidebar.warning("No CSV files found. Run scripts/rebuild_all.py first.")
            return normalizer_path, 1.0

        selected_name = st.sidebar.selectbox(
            "Session file", [p.name for p in csv_files]
        )
        selected_path = _DATA_DIR / selected_name
        speed = float(
            st.sidebar.slider(
                "Playback speed (×real-time)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
            )
        )

        c1, c2 = st.sidebar.columns(2)
        play_label = "Play" if not st.session_state.playing else "Pause"
        play_clicked = c1.button(play_label, use_container_width=True)
        reset_clicked = c2.button("Reset", use_container_width=True)

        if play_clicked:
            current = st.session_state.streamer
            if current is None or current.session_id != selected_path.stem:
                if engine is not None:
                    engine.reset()
                st.session_state.streamer = CsvStreamer(selected_path, speed=speed)
                _clear_session_display()
            st.session_state.playing = not st.session_state.playing

        if reset_clicked:
            if engine is not None:
                engine.reset()
            if st.session_state.streamer is not None:
                st.session_state.streamer.reset()
            _clear_session_display()
            st.session_state.playing = False

        streamer = st.session_state.streamer
        if streamer is not None:
            st.sidebar.progress(streamer.elapsed_s / max(1, streamer.total))
            st.sidebar.caption(
                f"{streamer.elapsed_s} s / {streamer.total} s "
                f"| {streamer.remaining} s remaining"
            )

    # ── Live vehicle controls ─────────────────────────────────────────────────
    else:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["(no ports found)"]
        selected_port = st.sidebar.selectbox("COM port", ports)

        live_src = st.session_state.live_source
        is_connected = live_src is not None and live_src.connected

        c1, c2 = st.sidebar.columns(2)
        connect_clicked = c1.button(
            "Disconnect" if is_connected else "Connect",
            use_container_width=True,
        )
        reset_clicked = c2.button("Reset", use_container_width=True)

        if connect_clicked:
            if is_connected:
                # Disconnect
                live_src.stop()
                st.session_state.live_source = None
                st.session_state.playing = False
            else:
                # Connect
                with st.sidebar.status("Connecting to ELM327…", expanded=False):
                    new_src = LiveObdSource(
                        port=selected_port
                        if selected_port != "(no ports found)"
                        else None,
                        sample_hz=1.0,
                    )
                    ok = new_src.connect()
                if ok:
                    new_src.start()
                    st.session_state.live_source = new_src
                    if engine is not None:
                        engine.reset()
                    _clear_session_display()
                    st.session_state.playing = True
                else:
                    st.sidebar.error(
                        "Could not connect.  Check adapter, port, and ignition."
                    )

        if reset_clicked:
            if engine is not None:
                engine.reset()
            if live_src is not None:
                live_src.reset()
            _clear_session_display()
            st.session_state.playing = False

        # Live status indicator
        live_src = st.session_state.live_source
        if live_src is not None and live_src.connected:
            hz = live_src.measured_poll_hz
            st.sidebar.success(
                f"Connected — {hz:.1f} Hz | {len(live_src.supported_pids)}/14 PIDs"
            )
            st.sidebar.caption(f"{live_src.elapsed_s} s elapsed")
        elif live_src is not None:
            st.sidebar.warning("Adapter found, waiting for ECU…")
        else:
            st.sidebar.info("Not connected.  Press Connect to start.")

    return normalizer_path, speed


# ── Panel renderers ───────────────────────────────────────────────────────────


def _section_header(title: str) -> None:
    """Render a thin section label in Saira Condensed with a left accent bar."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'margin:20px 0 10px 0;">'
        f'<div style="width:3px;height:16px;background:{ACCENT_DATA};'
        f'border-radius:2px;flex-shrink:0;"></div>'
        f'<span style="font-family:{FONT_DISPLAY};font-size:11px;'
        f'text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_SECONDARY};">'
        f"{title}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_status_banner(state: DashboardState) -> None:
    """Full-width 4-lane instrument-cluster status banner.

    Lanes: [LED square] [Title / subtitle] [3 stat blocks] [Elapsed / regime]
    A 3px accent strip runs across the top edge — the dominant visual cue for
    the current machine state without relying on text alone.
    """
    alert = state.stable_alert

    # ── Determine state category ────────────────────────────────────────────
    if not state.buffer_ready:
        accent = ACCENT_INFO
        title = "WARMING UP"
        subtitle = "Collecting first 60 seconds of sensor data…"
        stat1_lbl, stat1_val = "ELAPSED", f"{state.elapsed_s} s"
        stat2_lbl, stat2_val = "TARGET", "60 s"
        stat3_lbl, stat3_val = "REGIME", "—"
    elif alert.active:
        accent = ACCENT_ALERT
        fault = alert.fault_type.replace("_", " ").upper()
        title = f"FAULT — {fault}"
        subtitle = f"{alert.windows_voted} windows confirmed · majority vote passed"
        stat1_lbl, stat1_val = "CONFIDENCE", f"{alert.confidence:.0%}"
        stat2_lbl, stat2_val = "WINDOWS", f"{alert.windows_voted}/3"
        # Pull current severity for the active fault if available
        sev = state.severities.get(alert.fault_type, 0.0)
        fcast = state.forecasts.get(alert.fault_type, 0.0)
        delta_sym = "▲" if fcast > sev + 0.02 else ("▼" if fcast < sev - 0.02 else "—")
        stat3_lbl, stat3_val = "SEVERITY", f"{sev:.0%} {delta_sym}"
    elif state.classifier_label == "cold_start":
        accent = ACCENT_WARN
        title = "COLD START"
        subtitle = "Monitoring engine warm-up arc · rules active"
        coolant = state.latest_row.get("COOLANT_TEMPERATURE", 0.0)
        stat1_lbl, stat1_val = "COOLANT", f"{coolant:.0f} °C"
        stat2_lbl, stat2_val = "TARGET", "75 °C"
        stat3_lbl, stat3_val = "ELAPSED", f"{state.elapsed_s} s"
    elif state.classifier_label in ("healthy", "warming_up"):
        accent = ACCENT_OK
        title = "ALL SYSTEMS NOMINAL"
        subtitle = f"Classifier · {state.classifier_confidence:.0%} confidence"
        stat1_lbl, stat1_val = "LABEL", "HEALTHY"
        stat2_lbl, stat2_val = "CONF", f"{state.classifier_confidence:.0%}"
        stat3_lbl, stat3_val = "ELAPSED", f"{state.elapsed_s} s"
    else:
        # Fault suspected but voting not complete
        accent = ACCENT_WARN
        label_d = state.classifier_label.replace("_", " ").upper()
        title = f"SUSPECTED — {label_d}"
        subtitle = "Awaiting majority vote confirmation…"
        stat1_lbl, stat1_val = "CONF", f"{state.classifier_confidence:.0%}"
        stat2_lbl, stat2_val = "ELAPSED", f"{state.elapsed_s} s"
        stat3_lbl, stat3_val = "STATUS", "UNCONFIRMED"

    # ── Stat block HTML builder ──────────────────────────────────────────────
    def _stat(label: str, value: str) -> str:
        return (
            f'<div style="display:flex;flex-direction:column;gap:2px;">'
            f'<span style="font-family:{FONT_DISPLAY};font-size:9px;text-transform:uppercase;'
            f'letter-spacing:0.12em;color:{TEXT_MUTED};">{label}</span>'
            f'<span style="font-family:{FONT_MONO};font-size:18px;font-weight:700;'
            f'color:{TEXT_PRIMARY};line-height:1;">{value}</span>'
            f"</div>"
        )

    # ── Elapsed lane (right) ─────────────────────────────────────────────────
    elapsed_html = (
        f'<div style="text-align:right;min-width:96px;">'
        f'<div style="font-family:{FONT_DISPLAY};font-size:9px;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{TEXT_MUTED};">SESSION TIME</div>'
        f'<div style="font-family:{FONT_MONO};font-size:26px;font-weight:700;'
        f'color:{TEXT_PRIMARY};line-height:1.1;">{state.elapsed_s:04d} s</div>'
        f"</div>"
    )

    # ── Assemble banner ──────────────────────────────────────────────────────
    banner_html = f"""
<div style="
    background-color:{BG_SURFACE};
    border:1px solid {BORDER};
    border-radius:8px;
    overflow:hidden;
    margin-bottom:16px;
    box-shadow:0 2px 12px {BG_BASE}80;
">
  <!-- Accent strip across the top -->
  <div style="height:3px;background:{accent};
              box-shadow:0 0 20px {accent}60;"></div>

  <!-- Content grid: LED | title+sub | stats | elapsed -->
  <div style="
    display:grid;
    grid-template-columns:52px 1fr auto auto;
    align-items:center;
    gap:20px;
    padding:16px 20px;
  ">
    <!-- LED square -->
    <div style="
      width:40px;height:40px;
      background:{accent};
      border-radius:6px;
      box-shadow:0 0 24px {accent}40;
      flex-shrink:0;
    "></div>

    <!-- Title / subtitle -->
    <div>
      <div style="font-family:{FONT_DISPLAY};font-size:20px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.05em;
                  color:{TEXT_PRIMARY};line-height:1.1;">{title}</div>
      <div style="font-family:{FONT_BODY};font-size:12px;color:{TEXT_SECONDARY};
                  margin-top:4px;">{subtitle}</div>
    </div>

    <!-- 3 stat blocks -->
    <div style="display:flex;gap:28px;padding:0 8px;">
      {_stat(stat1_lbl, stat1_val)}
      {_stat(stat2_lbl, stat2_val)}
      {_stat(stat3_lbl, stat3_val)}
    </div>

    <!-- Elapsed time lane -->
    {elapsed_html}
  </div>
</div>
"""
    st.markdown(banner_html, unsafe_allow_html=True)


def _render_severity_grid(state: DashboardState) -> None:
    """Four plotly angular gauges — current severity + 60-second forecast threshold.

    The cyan threshold line on each gauge shows WHERE we expect severity to be
    in 60 seconds, so the driver reads "needle position = now, cyan line = soon".
    """
    import plotly.graph_objects as go

    cols = st.columns(4)
    for col, fault in zip(cols, FAULT_TYPES):
        sev = state.severities.get(fault, 0.0)
        fc = state.forecasts.get(fault, 0.0)
        display = _FAULT_DISPLAY.get(fault, fault)

        sev_pct = round(sev * 100, 1)
        fc_pct = round(fc * 100, 1)
        bar_col = severity_color(sev)

        # Delta symbol for caption
        if fc > sev + 0.02:
            delta_sym, delta_col = "▲", ACCENT_ALERT
        elif fc < sev - 0.02:
            delta_sym, delta_col = "▼", ACCENT_OK
        else:
            delta_sym, delta_col = "—", TEXT_MUTED

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sev_pct,
                number={
                    "suffix": "%",
                    "font": {"family": FONT_MONO, "size": 26, "color": TEXT_PRIMARY},
                },
                gauge={
                    "shape": "angular",
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 0,
                        "tickcolor": BORDER,
                        "tickfont": {
                            "color": TEXT_MUTED,
                            "size": 8,
                            "family": FONT_MONO,
                        },
                        "nticks": 6,
                    },
                    "bar": {"color": bar_col, "thickness": 0.26},
                    "bgcolor": BG_RAISED,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": ACCENT_OK + "18"},
                        {"range": [30, 60], "color": ACCENT_WARN + "18"},
                        {"range": [60, 100], "color": ACCENT_ALERT + "18"},
                    ],
                    # Forecast marker — cyan needle shows where we'll be in 60 s
                    "threshold": {
                        "line": {"color": ACCENT_DATA, "width": 3},
                        "thickness": 0.82,
                        "value": fc_pct,
                    },
                },
            )
        )
        fig.update_layout(
            paper_bgcolor=BG_SURFACE,
            plot_bgcolor=BG_SURFACE,
            font={"color": TEXT_PRIMARY, "family": FONT_MONO},
            margin={"l": 16, "r": 16, "t": 32, "b": 0},
            height=190,
        )

        with col:
            # Fault name heading
            st.markdown(
                f'<div style="font-family:{FONT_DISPLAY};font-size:11px;'
                f"text-transform:uppercase;letter-spacing:0.1em;"
                f'color:{TEXT_SECONDARY};text-align:center;margin-bottom:4px;">'
                f"{display}</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )
            # Caption: now · forecast
            st.markdown(
                f'<div style="font-family:{FONT_MONO};font-size:11px;'
                f'text-align:center;color:{TEXT_MUTED};margin-top:-12px;">'
                f'NOW <b style="color:{bar_col}">{sev_pct:.0f}%</b>'
                f"&nbsp;&nbsp;+60s "
                f'<b style="color:{delta_col}">{fc_pct:.0f}% {delta_sym}</b>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_pid_strip(history: deque) -> None:
    """2×2 grid of dark sparkline cards — each shows a large live value + rolling plot.

    Card layout (per PID):
      Left column  (fixed 110px): unit label, big current value, unit suffix
      Right column (flex):        axes-free plotly sparkline in ACCENT_DATA
    """
    import plotly.graph_objects as go

    _PID_META = {
        "ENGINE_RPM": ("ENGINE RPM", "", ACCENT_DATA),
        "COOLANT_TEMPERATURE": ("COOLANT", "°C", ACCENT_WARN),
        "LONG_TERM_FUEL_TRIM_BANK_1": ("LTFT", "%", ACCENT_OK),
        "THROTTLE": ("THROTTLE", "%", ACCENT_ALERT),
    }
    _PID_ORDER = list(_PID_META.keys())

    if not history:
        # Placeholder cards with no data yet
        cols = st.columns(2)
        for i, pid in enumerate(_PID_ORDER):
            label, unit, _ = _PID_META[pid]
            with cols[i % 2]:
                st.markdown(
                    f'<div class="panel-card" style="height:96px;display:flex;'
                    f'align-items:center;justify-content:center;">'
                    f'<span style="font-family:{FONT_MONO};color:{TEXT_MUTED};font-size:12px;">'
                    f"{label} — press Play</span></div>",
                    unsafe_allow_html=True,
                )
        return

    hist_df = pd.DataFrame(list(history))
    cols = st.columns(2)

    for i, pid in enumerate(_PID_ORDER):
        label, unit, line_color = _PID_META[pid]
        if pid not in hist_df.columns:
            continue

        series = hist_df[pid].dropna()
        current_val = series.iloc[-1] if len(series) else 0.0

        # Format current value: integers for RPM, 1 decimal for others
        if pid == "ENGINE_RPM":
            val_str = f"{int(current_val):,}"
        else:
            val_str = f"{current_val:.1f}"

        # Sparkline — no axes, no labels, just the signal shape
        fig = go.Figure(
            go.Scatter(
                y=series.values,
                mode="lines",
                line={"color": line_color, "width": 1.8},
                fill="tozeroy",
                fillcolor=line_color + "18",
            )
        )
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
        fig.update_layout(
            paper_bgcolor=BG_SURFACE,
            plot_bgcolor=BG_SURFACE,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False,
            height=80,
        )

        with cols[i % 2]:
            # Card: left value block + right sparkline side by side
            left_block = (
                f'<div style="min-width:110px;padding-right:12px;">'
                f'<div style="font-family:{FONT_DISPLAY};font-size:9px;'
                f"text-transform:uppercase;letter-spacing:0.1em;"
                f'color:{TEXT_MUTED};">{label}</div>'
                f'<div style="font-family:{FONT_MONO};font-size:28px;'
                f'font-weight:700;color:{TEXT_PRIMARY};line-height:1.1;">'
                f"{val_str}</div>"
                f'<div style="font-family:{FONT_BODY};font-size:10px;'
                f'color:{TEXT_SECONDARY};">{unit}</div>'
                f"</div>"
            )
            # Use a 2-column nested layout: text left, chart right
            inner_c1, inner_c2 = st.columns([1, 3])
            with inner_c1:
                st.markdown(
                    f'<div style="background:{BG_SURFACE};border:1px solid {BORDER};'
                    f"border-radius:8px;padding:12px 12px 8px 16px;height:96px;"
                    f'display:flex;align-items:center;">{left_block}</div>',
                    unsafe_allow_html=True,
                )
            with inner_c2:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )


def _render_alert_log(log_entries: list) -> None:
    """Diagnostic terminal feed — monospace, color-coded, time-prefixed rows."""
    if not log_entries:
        st.markdown(
            f'<div class="panel-card" style="font-family:{FONT_MONO};'
            f'font-size:12px;color:{TEXT_MUTED};padding:20px 16px;">'
            f"// no alerts yet</div>",
            unsafe_allow_html=True,
        )
        return

    rows_html = ""
    for entry in reversed(log_entries[-15:]):
        # Parse entry format: "[{elapsed}s] {TYPE} — {detail}"
        # Classify entry for color coding
        if "ML ALERT" in entry:
            glyph, tag, tag_color = "▸", "ML ALERT", ACCENT_ALERT
        elif "RULE" in entry.upper():
            glyph, tag, tag_color = "⚠", "RULE", ACCENT_WARN
        elif "cleared" in entry.lower() or "healthy" in entry.lower():
            glyph, tag, tag_color = "✓", "CLEARED", ACCENT_OK
        else:
            glyph, tag, tag_color = "·", "INFO", TEXT_MUTED

        # Extract time prefix if present
        if entry.startswith("["):
            bracket_end = entry.find("]")
            time_part = entry[1:bracket_end] if bracket_end > 0 else ""
            detail = entry[bracket_end + 1 :].strip().lstrip("—").strip()
        else:
            time_part = ""
            detail = entry

        # Remove the redundant type prefix from detail (we re-render it as a tag)
        for prefix in ("ML ALERT —", "RULE —", "Alert cleared —"):
            if detail.startswith(prefix):
                detail = detail[len(prefix) :].strip()
                break

        rows_html += (
            f'<div style="display:grid;grid-template-columns:80px 20px 90px 1fr;'
            f"gap:8px;align-items:baseline;padding:8px 0;"
            f'border-bottom:1px solid {BORDER};">'
            # Time
            f'<span style="font-family:{FONT_MONO};font-size:10px;color:{TEXT_MUTED};">'
            f"[ {time_part} ]</span>"
            # Glyph
            f'<span style="font-size:12px;color:{tag_color};">{glyph}</span>'
            # Category tag
            f'<span style="font-family:{FONT_DISPLAY};font-size:9px;text-transform:uppercase;'
            f"letter-spacing:0.1em;color:{tag_color};"
            f'border:1px solid {tag_color}40;border-radius:3px;padding:1px 5px;">'
            f"{tag}</span>"
            # Detail
            f'<span style="font-family:{FONT_BODY};font-size:12px;color:{TEXT_PRIMARY};">'
            f"{detail}</span>"
            f"</div>"
        )

    st.markdown(
        f'<div class="panel-card" style="max-height:320px;overflow-y:auto;padding:8px 16px;">'
        f"{rows_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_shap_panel(state: DashboardState) -> None:
    """Dark horizontal SHAP bar chart with title strip and outside-bar labels."""
    import plotly.graph_objects as go

    if not state.top_features:
        st.markdown(
            f'<div class="panel-card" style="font-family:{FONT_MONO};'
            f'font-size:12px;color:{TEXT_MUTED};padding:20px 16px;">'
            f"// available after first 60-second window</div>",
            unsafe_allow_html=True,
        )
        return

    # Title strip — label + confidence in Saira Condensed
    label_display = state.classifier_label.replace("_", " ").upper()
    st.markdown(
        f'<div style="font-family:{FONT_DISPLAY};font-size:11px;'
        f"text-transform:uppercase;letter-spacing:0.1em;"
        f'color:{TEXT_SECONDARY};margin-bottom:6px;">'
        f"PREDICTING &nbsp;"
        f'<span style="color:{TEXT_PRIMARY};">{label_display}</span>'
        f"&nbsp;·&nbsp;"
        f'<span style="color:{ACCENT_DATA};font-family:{FONT_MONO};">'
        f"{state.classifier_confidence:.0%} CONFIDENCE</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    names = [
        f[0].replace("__z", "").replace("__", " ").replace("_", " ").title()
        for f in state.top_features
    ]
    values = [f[1] for f in state.top_features]

    # Sort by absolute SHAP value descending, orient horizontal
    pairs = sorted(
        zip(names, values), key=lambda x: abs(x[1])
    )  # ascending for plotly (bottom-up)
    sorted_names = [p[0] for p in pairs]
    sorted_vals = [p[1] for p in pairs]

    # Color: positive → pushes toward active label (ACCENT_DATA), negative → TEXT_MUTED
    bar_colors = [ACCENT_DATA if v >= 0 else TEXT_MUTED for v in sorted_vals]

    fig = go.Figure(
        go.Bar(
            x=sorted_vals,
            y=sorted_names,
            orientation="h",
            marker={"color": bar_colors},
            text=[f"{v:+.3f}" for v in sorted_vals],
            textposition="outside",
            textfont={"family": FONT_MONO, "size": 10, "color": TEXT_SECONDARY},
            cliponaxis=False,
        )
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False,
        tickfont={"family": FONT_MONO, "size": 10, "color": TEXT_SECONDARY},
    )
    fig.update_layout(
        paper_bgcolor=BG_SURFACE,
        plot_bgcolor=BG_SURFACE,
        font={"family": FONT_MONO, "color": TEXT_PRIMARY},
        margin={"l": 0, "r": 60, "t": 0, "b": 0},
        height=220,
        showlegend=False,
    )

    st.markdown(
        f'<div class="panel-card" style="padding:12px 16px;">', unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── Alert log helpers ─────────────────────────────────────────────────────────


def _log_append(entry: str) -> None:
    """Append to alert log only if it's not a duplicate of the last entry."""
    log = st.session_state.alert_log
    if not log or log[-1] != entry:
        log.append(entry)


def _update_alert_log(state: DashboardState) -> None:
    """Emit log entries for new rule alerts and ML alert state transitions."""
    # Rule alerts (deterministic; fire once)
    for ra in state.rule_alerts:
        entry = (
            f"[{state.elapsed_s}s] RULE — {ra.rule}: {ra.description[:80].rstrip()}…"
        )
        if entry not in st.session_state.alert_log:
            _log_append(entry)

    # ML stable alert transitions
    alert = state.stable_alert
    current_fault = alert.fault_type if alert.active else "healthy"
    if current_fault != st.session_state.last_active_fault:
        if alert.active:
            _log_append(
                f"[{state.elapsed_s}s] ML ALERT — {alert.fault_type.replace('_', ' ')} "
                f"({alert.confidence:.0%} conf, {alert.windows_voted} windows)"
            )
        elif st.session_state.last_active_fault not in ("healthy", ""):
            _log_append(f"[{state.elapsed_s}s] Alert cleared — system healthy")
        st.session_state.last_active_fault = current_fault


# ── Session display helper ────────────────────────────────────────────────────


def _clear_session_display() -> None:
    """Reset per-session display buffers without touching the source."""
    st.session_state.pid_history.clear()
    st.session_state.alert_log.clear()
    st.session_state.latest_state = None
    st.session_state.last_active_fault = ""


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _init_session_state()
    inject_global_styles()

    st.title("Predictive Maintenance — Live OBD-II Dashboard")

    # Load engine keyed by current normalizer; sidebar may update it this rerun
    engine = _load_engine(st.session_state.normalizer_path)
    _normalizer_path, speed = _render_sidebar(engine)
    # Reload after sidebar in case normalizer selectbox changed (cache hit if unchanged)
    engine = _load_engine(st.session_state.normalizer_path)

    if engine is None:
        st.error(
            "Model files not found. Run `python -m scripts.rebuild_all` first "
            "to train the classifier and forecasters."
        )
        return

    # ── Advance one row (if playing) ──────────────────────────────────────────
    source = (
        st.session_state.streamer
        if st.session_state.source_type == "csv"
        else st.session_state.live_source
    )

    if st.session_state.playing and source is not None:
        row = source.next_row()

        if row is None:
            if getattr(source, "exhausted", False):
                # CSV finished
                st.session_state.playing = False
            else:
                # Live source: no fresh row ready this tick — reschedule quickly
                time.sleep(0.05)
                st.rerun()
                return
        else:
            state = engine.update(row)
            st.session_state.latest_state = state
            st.session_state.pid_history.append(row)
            _update_alert_log(state)

    # ── Render current state ──────────────────────────────────────────────────
    state: DashboardState | None = st.session_state.latest_state

    if state is None:
        if st.session_state.source_type == "live":
            st.info("Connect to a vehicle in the sidebar to begin live monitoring.")
        else:
            st.info("Select a session file in the sidebar and press **Play** to begin.")
    else:
        _render_status_banner(state)

        _section_header("Fault Severity · 60-Second Forecast")
        _render_severity_grid(state)

        _section_header("Live PID Readings")
        _render_pid_strip(st.session_state.pid_history)

        col_log, col_shap = st.columns([1, 1])
        with col_log:
            _section_header("Alert Log")
            _render_alert_log(st.session_state.alert_log)
        with col_shap:
            _section_header("Top SHAP Features")
            _render_shap_panel(state)

        # Elapsed time — custom mono readout (avoids default metric chrome)
        st.sidebar.markdown(
            f'<div style="border-top:1px solid {BORDER};margin-top:12px;padding-top:12px;">'
            f'<div style="font-family:{FONT_DISPLAY};font-size:9px;text-transform:uppercase;'
            f'letter-spacing:0.12em;color:{TEXT_MUTED};">SESSION TIME</div>'
            f'<div style="font-family:{FONT_MONO};font-size:22px;font-weight:700;'
            f'color:{TEXT_PRIMARY};line-height:1.2;">{state.elapsed_s:,} s</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Schedule next rerun (auto-advance) ───────────────────────────────────
    if st.session_state.playing:
        if st.session_state.source_type == "csv":
            csv_speed = (
                st.session_state.streamer.speed if st.session_state.streamer else 1.0
            )
            # Floor at 20 ms to avoid hammering Streamlit's render loop
            time.sleep(max(0.02, 1.0 / csv_speed))
        else:
            # Live mode: poll at ~20 Hz; actual OBD rows arrive at 1 Hz
            time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
