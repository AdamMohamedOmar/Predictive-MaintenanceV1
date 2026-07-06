"""Global CSS injected once at app startup via inject_global_styles().

Call inject_global_styles() at the top of main() in app.py.
This module has no pytest tests — it contains only CSS strings and one
Streamlit call; correctness is verified visually.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.theme import (
    ACCENT_DATA, BG_BASE, BG_SURFACE, BG_RAISED,
    BORDER, BORDER_STRONG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_DISPLAY, FONT_BODY, FONT_MONO,
)

_CSS = f"""
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@500;700&family=Outfit:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Page & app shell ── */
html, body, [class*="css"] {{
    font-family: {FONT_BODY};
    color: {TEXT_PRIMARY};
}}

.stApp {{
    background-color: {BG_BASE};
}}

.main .block-container {{
    background-color: {BG_BASE};
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {BG_SURFACE};
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] .block-container {{
    background-color: {BG_SURFACE};
    padding-top: 1rem;
}}

/* ── Headings — Saira Condensed, uppercase, spaced ── */
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: {FONT_DISPLAY} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEXT_PRIMARY};
    font-weight: 700;
}}

/* ── Code / monospace — JetBrains Mono ── */
code, pre, .stCode, .stCodeBlock {{
    font-family: {FONT_MONO} !important;
    background-color: {BG_RAISED} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: {FONT_MONO} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background-color: {BG_RAISED} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
    transition: border-color 0.15s, background-color 0.15s;
}}

.stButton > button:hover {{
    background-color: {BG_SURFACE} !important;
    border-color: {ACCENT_DATA} !important;
    color: {ACCENT_DATA} !important;
}}

/* ── Selectbox / radio ── */
.stSelectbox > div > div {{
    background-color: {BG_RAISED} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    font-family: {FONT_BODY} !important;
}}

.stRadio > div {{
    gap: 8px;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {{
    padding: 0 !important;
}}

[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {ACCENT_DATA} !important;
    border-color: {ACCENT_DATA} !important;
}}

/* ── Progress bar (session playback) ── */
.stProgress > div > div > div > div {{
    background-color: {ACCENT_DATA} !important;
}}

.stProgress > div > div > div {{
    background-color: {BG_RAISED} !important;
    border-radius: 4px !important;
}}

/* ── Alert/status boxes (info/warning/success/error) ── */
.stAlert {{
    background-color: {BG_SURFACE} !important;
    border-radius: 6px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Metric widget ── */
[data-testid="metric-container"] {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 12px 16px;
}}

[data-testid="metric-container"] label {{
    font-family: {FONT_DISPLAY} !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_MUTED} !important;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: {FONT_MONO} !important;
    font-size: 1.6rem !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ── Reusable panel card ── */
.panel-card {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
}}

/* ── Dividers ── */
hr, .stDivider {{
    border-color: {BORDER} !important;
    margin: 12px 0 !important;
}}

/* ── Captions ── */
.stCaption, small {{
    font-family: {FONT_BODY} !important;
    color: {TEXT_SECONDARY} !important;
    font-size: 0.75rem !important;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

/* ── Plotly charts — remove white border Streamlit adds ── */
.stPlotlyChart {{
    border: none !important;
    background: transparent !important;
}}

/* ── Column gaps ── */
[data-testid="column"] {{
    padding-left: 6px !important;
    padding-right: 6px !important;
}}
"""


def inject_global_styles() -> None:
    """Inject the instrument-cluster CSS once at app startup.

    Call this at the top of main(), after _init_session_state().
    Safe to call on every rerun — Streamlit deduplicates injected HTML.
    """
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
