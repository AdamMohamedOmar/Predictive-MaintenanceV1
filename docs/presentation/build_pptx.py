"""Build the graduation-defense slide deck: docs/presentation/defense.pptx.

Reproducible: run from the repo root with the project venv.
    python docs/presentation/build_pptx.py

All metrics are pulled from results/*.json (see verify_numbers()); the AAST
logo and the EDA / confusion figures are reused from docs/book/.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "defense.pptx"
FIG = ROOT / "docs" / "book" / "figures"
LOGO = ROOT / "docs" / "book" / "aast.png"

# ── Theme ─────────────────────────────────────────────────────────────────────
NAVY = RGBColor(0x0B, 0x25, 0x45)      # title bars, dividers
NAVY2 = RGBColor(0x13, 0x36, 0x5C)     # divider gradient-ish second tone
TEAL = RGBColor(0x22, 0xB8, 0xC4)      # accent rule / highlights
GOLD = RGBColor(0xC9, 0xA2, 0x27)      # honest-number accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)       # body text
MUTED = RGBColor(0x5A, 0x64, 0x70)     # captions / footer
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)     # placeholder fill

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

FOOTER_TEXT = "AAST — Computer Engineering  |  Predictive Maintenance for OBD-II Engine Faults"


def _set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _tf(shape, wrap=True):
    tf = shape.text_frame
    tf.word_wrap = wrap
    return tf


def _para(tf, text, size, color, bold=False, align=PP_ALIGN.LEFT,
          space_after=6, first=False, font="Calibri"):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def _footer(slide, idx):
    """Logo + footer text + slide number on every non-title slide."""
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(0.28), Inches(6.95),
                                 height=Inches(0.42))
    box = slide.shapes.add_textbox(Inches(0.9), Inches(6.98), Inches(10.6), Inches(0.4))
    p = _para(_tf(box), FOOTER_TEXT, 9, MUTED, first=True)
    num = slide.shapes.add_textbox(Inches(12.4), Inches(6.98), Inches(0.7), Inches(0.4))
    _para(_tf(num), str(idx), 11, MUTED, bold=True, align=PP_ALIGN.RIGHT, first=True)


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_title_bar(slide, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(1.15))
    _set_fill(bar, NAVY)
    # teal accent rule under the bar
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.15), SW, Inches(0.06))
    _set_fill(rule, TEAL)
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.3), Inches(0.85))
    tf = _tf(tb)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, title, 30, WHITE, bold=True, first=True)


def content_slide(idx, title, bullets, image=None, caption=None,
                  placeholder=None, notes=None):
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, title)
    has_visual = image is not None or placeholder is not None
    bullets_w = Inches(6.2) if has_visual else Inches(12.3)

    box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), bullets_w, Inches(5.1))
    tf = _tf(box)
    for i, b in enumerate(bullets):
        p = _para(tf, "▪  " + b, 20, INK, first=(i == 0), space_after=14)

    if image is not None and Path(image).exists():
        pic = slide.shapes.add_picture(str(image), Inches(6.7), Inches(1.55),
                                       width=Inches(6.3))
        # Cap height without distorting: rescale width proportionally BEFORE
        # overwriting height, then re-centre in the right-hand column.
        max_h = Inches(4.8)
        if pic.height > max_h:
            ratio = max_h / pic.height
            pic.width = int(pic.width * ratio)
            pic.height = int(max_h)
            pic.left = int(Inches(6.7) + (Inches(6.3) - pic.width) / 2)
        if caption:
            cap = slide.shapes.add_textbox(Inches(6.7), Inches(6.5), Inches(6.3), Inches(0.4))
            _para(_tf(cap), caption, 11, MUTED, align=PP_ALIGN.CENTER, first=True)
    elif placeholder is not None:
        ph = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7),
                                    Inches(1.7), Inches(6.3), Inches(4.4))
        ph.fill.solid(); ph.fill.fore_color.rgb = LIGHT
        ph.line.color.rgb = MUTED
        ph.line.width = Pt(1.0)
        ptf = _tf(ph); ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _para(ptf, placeholder, 14, MUTED, align=PP_ALIGN.CENTER, first=True)

    _footer(slide, idx)
    if notes:
        _notes(slide, notes)
    return slide


def divider_slide(idx, part_label, title, presenters, notes=None):
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    _set_fill(bg, NAVY)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.2),
                                  Inches(3.2), Inches(0.08))
    _set_fill(rule, TEAL)
    lab = slide.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(0.8))
    _para(_tf(lab), part_label, 22, TEAL, bold=True, first=True)
    ttl = slide.shapes.add_textbox(Inches(0.9), Inches(3.45), Inches(11.5), Inches(1.4))
    _para(_tf(ttl), title, 40, WHITE, bold=True, first=True)
    pres = slide.shapes.add_textbox(Inches(0.9), Inches(4.9), Inches(11.5), Inches(1.0))
    _para(_tf(pres), "Presented by " + presenters, 18, RGBColor(0xC9, 0xD3, 0xE0),
          first=True)
    num = slide.shapes.add_textbox(Inches(12.4), Inches(6.98), Inches(0.7), Inches(0.4))
    _para(_tf(num), str(idx), 11, RGBColor(0x8B, 0x95, 0xA1), bold=True,
          align=PP_ALIGN.RIGHT, first=True)
    if notes:
        _notes(slide, notes)
    return slide


def title_slide():
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    _set_fill(bg, NAVY)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.55), SW, Inches(0.06))
    _set_fill(band, TEAL)
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(5.92), Inches(0.35),
                                 height=Inches(1.5))
    inst = slide.shapes.add_textbox(Inches(0.5), Inches(1.95), Inches(12.3), Inches(0.5))
    _para(_tf(inst), "Arab Academy for Science, Technology and Maritime Transport",
          15, RGBColor(0xC9, 0xD3, 0xE0), align=PP_ALIGN.CENTER, first=True)
    dept = slide.shapes.add_textbox(Inches(0.5), Inches(2.35), Inches(12.3), Inches(0.4))
    _para(_tf(dept), "College of Engineering and Technology  ·  Computer Engineering Department",
          12, RGBColor(0x9F, 0xB0, 0xC4), align=PP_ALIGN.CENTER, first=True)

    ttl = slide.shapes.add_textbox(Inches(0.7), Inches(2.85), Inches(11.9), Inches(1.3))
    tf = _tf(ttl)
    _para(tf, "Predictive Maintenance for OBD-II Engine Faults", 34, WHITE,
          bold=True, align=PP_ALIGN.CENTER, first=True)
    sub = slide.shapes.add_textbox(Inches(0.7), Inches(4.05), Inches(11.9), Inches(0.5))
    _para(_tf(sub), "Detecting developing engine faults from live 1 Hz sensor data",
          16, TEAL, align=PP_ALIGN.CENTER, first=True)

    # Presenters (two columns) + supervisors
    names = [
        ("Omar Hossam Eldin Mohamed Hafez", "20101737"),
        ("Adam Mohamed Ibrahim", "20101978"),
        ("Ahmed Abdelaal Saad", "16101027"),
        ("Mohamed Sameh Mohamed El Bassuni", "18100896"),
        ("Mohamed Mohamed Talaat Abokhatwa", "12103172"),
    ]
    left = slide.shapes.add_textbox(Inches(1.4), Inches(4.75), Inches(6.2), Inches(1.7))
    ltf = _tf(left)
    _para(ltf, "Presented by", 13, TEAL, bold=True, first=True)
    for nm, rg in names:
        _para(ltf, f"{nm}  —  {rg}", 12, WHITE, space_after=3)

    right = slide.shapes.add_textbox(Inches(8.2), Inches(4.75), Inches(4.0), Inches(1.7))
    rtf = _tf(right)
    _para(rtf, "Supervised by", 13, TEAL, bold=True, first=True)
    _para(rtf, "Dr. Yahia Elderiny", 12, WHITE, space_after=3)
    _para(rtf, "Dr. Rania Kadry", 12, WHITE, space_after=3)
    _para(rtf, "", 6, WHITE)
    _para(rtf, "Defense — July 2026", 12, RGBColor(0x9F, 0xB0, 0xC4))
    return slide


# ══════════════════════════ BUILD ══════════════════════════
MECH_REVIEW = ("[REVIEW-NEEDED] This mechanical-engineering content uses the "
               "task's default outline (OBD-II, test vehicle + ELM327, fault "
               "physics, data collection). Replace with the two mechanical "
               "students' actual contribution before the defense.")

# 1 — Title
title_slide()

# 2 — Agenda
content_slide(
    2, "Agenda — one project, two halves",
    [
        "Part 1 — Mechanical: the car, its sensors, the physics",
        "Part 2 — Computer Engineering: the data and the model",
        "What each engine fault looks like in the sensors",
        "How we detect a fault before the warning light",
        "Honest results — what works and what does not",
        "A live dashboard and where this goes next",
    ],
    image=FIG / "fig_architecture.png",
    caption="System overview: car → adapter → phone → model → dashboard",
    notes="Team lead opens. Keep this to ~1 minute. The figure is the whole "
          "system in one picture; we walk it left to right later.",
)

# 3 — PART 1 divider
divider_slide(
    3, "PART 1", "Mechanical Engineering",
    "Mohamed Sameh & Mohamed Abokhatwa",
    notes=MECH_REVIEW,
)

# 4 — OBD-II
content_slide(
    4, "What every car already tells us",
    [
        "Since ~1996 every car has an OBD-II diagnostic port",
        "The engine computer reports its sensors once a second",
        "Engine speed, coolant temp, fuel trims, throttle, and more",
        "A cheap adapter can read this live stream",
        "Standardized: SAE J1979 over ISO 15765-4 CAN",
    ],
    placeholder="[FIGURE: photo of the OBD-II port\nunder the dashboard + connected adapter]",
    notes=MECH_REVIEW + "\n\nKeep it plain: the car is already publishing this "
          "data for free; we just listen.",
)

# 5 — Test vehicle + hardware
content_slide(
    5, "The test car and the reader",
    [
        "Test vehicle: 2015 Toyota Yaris",
        "ELM327 Bluetooth adapter plugs into the OBD-II port",
        "Our Android app (OBD-LOG) records every reading",
        "About $15 of hardware — no garage tools needed",
        "More sensors polled = slower rate (7 PIDs 1.10 Hz, 21 PIDs 0.37 Hz)",
    ],
    placeholder="[FIGURE: 2015 Toyota Yaris +\nELM327 adapter + phone running OBD-LOG]",
    notes=MECH_REVIEW + "\n\nThe poll-rate trade-off is measured, in the book "
          "Table 3.1. Say it simply: asking for more sensors slows all of them.",
)

# 6 — Sensors behind the faults
content_slide(
    6, "The sensors behind each fault",
    [
        "Air leak → manifold pressure (MAP) and idle speed",
        "Fuel problem → short- and long-term fuel trims",
        "Temperature → the coolant sensor",
        "Throttle wear → throttle angle vs pedal position",
        "Each fault moves specific sensors in a known direction",
    ],
    notes=MECH_REVIEW + "\n\nThis slide is the bridge: name the sensor for each "
          "fault so the physics on the next two slides lands.",
)

# 7 — Physics 1: air + fuel
content_slide(
    7, "The physics (1): air and fuel",
    [
        "Vacuum leak: extra air sneaks in past the throttle",
        "Idle speed rises — like leaving the throttle cracked open",
        "Injector clog: less fuel sprayed, engine runs lean",
        "The computer adds fuel — long-term fuel trim climbs",
        "MAP stays normal for a fuel fault — that tells them apart",
    ],
    placeholder="[FIGURE: simple diagram — intake manifold,\nthrottle, injector, airflow arrows]",
    notes=MECH_REVIEW + "\n\nUse the analogy out loud: a vacuum leak behaves "
          "like a stuck-open throttle. That single sentence explains the whole slide.",
)

# 8 — Physics 2: coolant + throttle
content_slide(
    8, "The physics (2): coolant and throttle",
    [
        "Stuck coolant sensor: computer thinks the engine is cold",
        "It retards timing and over-fuels — trims go rich",
        "Throttle sensor wear: reported angle drifts from the pedal",
        "Only shows under load; commanded throttle stays correct",
        "That mismatch is a second, independent clue",
    ],
    notes=MECH_REVIEW + "\n\nKeep 'timing' light for a lay audience: the engine "
          "acts like it never warmed up.",
)

# 9 — Data collection
content_slide(
    9, "Collecting real driving data",
    [
        "Drives on public roads, with the driver's consent",
        "The phone logs every sensor once per second",
        "No fault was ever induced on the car — safety first",
        "We record a healthy baseline for each vehicle",
        "That baseline is what the model compares against",
    ],
    placeholder="[FIGURE: map/photo of a data-collection drive\nor the app logging screen]",
    notes=MECH_REVIEW + "\n\nEnd Part 1 on the ethics point: everything is "
          "healthy, real, consented data. Hand off to the CE team.",
)

# 10 — PART 2 divider
divider_slide(
    10, "PART 2", "Computer Engineering",
    "Omar Hossam, Adam Mohamed & Ahmed Saad",
)

# 11 — Problem + honest scope
content_slide(
    11, "The problem, stated honestly",
    [
        "Goal: flag a developing fault before the warning light",
        "No public dataset of real, labeled OBD-II faults exists",
        "So we train on synthetic faults over real healthy data",
        "We report this as synthetic validation, not field proof",
        "That honesty shapes every number in this talk",
    ],
    image=FIG / "fig_architecture.png",
    caption="The two-pillar pipeline: classify the fault, and watch for drift",
    notes="Say the hard truth early: the model has never seen a real fault. "
          "Everything after this is measured against that limit.",
)

# 12 — Dataset + trailing comma
content_slide(
    12, "Rescuing the dataset",
    [
        "First audit: 120 of 129 files looked physically impossible",
        "Real cause: a trailing comma shifted every column left",
        "Timing read catalyst temperatures — hence 776° readings",
        "One parser flag recovered all 129 files",
        "Data grew from 9 files (~4.4 h) to 129 (~84.5 h): 19× more",
    ],
    image=FIG / "fig_timing_fix.png",
    caption="Timing advance, before vs after the one-line parser fix",
    notes="This is our best 'data engineering' story. The bug was silent and "
          "the symptom appeared two layers from the cause. 19x more data from "
          "one character.",
)

# 13 — Fault injection
content_slide(
    13, "Making faults we could learn from",
    [
        "Inject 4 faults into real healthy drives",
        "Every written value obeys engine physics",
        "Coolant can't jump >1°C per second; fuel trims capped ±25%",
        "Ramp mode models gradual wear, not a sudden break",
        "Fully reproducible: seed, onset, magnitude all stored",
    ],
    placeholder="[FIGURE: injected fuel-trim ramp\noverlaid on a healthy drive]",
    notes="The physics constraints are the point. A naive injector writes "
          "signatures real ECUs never produce; ours refuses to.",
)

# 14 — Features
content_slide(
    14, "Turning sensors into features",
    [
        "Each 60-second window → 83 numeric features",
        "Statistics, cross-sensor ratios, trajectories, regime flags",
        "Every feature z-scored vs that car's own healthy baseline",
        "So the model reads '4σ above normal for THIS car'",
        "That baseline trick is what lets one model fit many cars",
    ],
    image=FIG / "fig_regime_windows.png",
    caption="Driving splits into regimes; cruise dominates the windows",
    notes="The z-scoring is the cross-vehicle mechanism. Raw values differ car "
          "to car for healthy reasons; deviations from each car's own normal are "
          "comparable.",
)

# 15 — Classifier + Untested contract
content_slide(
    15, "The classifier, and knowing its limits",
    [
        "XGBoost over 6 classes: healthy, cold start, 4 faults",
        "SHAP explains every call — which sensors, how much",
        "Untested contract: missing sensor → 'Untested', not a guess",
        "A MAF car with no MAP → air fault marked Untested",
        "The model is allowed to say 'I can't test this here'",
    ],
    notes="SHAP matters for a workshop: 'LTFT pushed this toward fuel by 0.43' "
          "is a checkable claim, not a black box.",
)

# 16 — Results
content_slide(
    16, "Results, read honestly",
    [
        "Cross-context worst fold: macro-F1 0.958 (our headline)",
        "Hardest single session: 0.699 (the honest floor)",
        "Random Forest scored 1.000 — a memorization warning, not a win",
        "Withheld-settings gap only 0.033 — not just memorizing",
        "All numbers are synthetic self-consistency, not field detection",
    ],
    image=FIG / "fig_confusion_xgb.png",
    caption="Fixed-holdout confusion matrix (macro-F1 0.962)",
    notes="Lead with 0.958, not the 1.000. The perfect Random-Forest score is "
          "the tell that the synthetic task is memorizable; we report it as a "
          "warning on purpose.",
)

# 17 — Dashboard
content_slide(
    17, "The live dashboard",
    [
        "Shows the classification, fuel trims, and SHAP live",
        "Severity strip comes from physics, not the forecaster",
        "End-of-read report: one plain verdict per fault",
        "Refuses to score when the data is insufficient",
        "Explains itself so a mechanic can check it",
    ],
    placeholder="[FIGURE: Streamlit dashboard screenshot —\nstatus banner, severity strip,\nSHAP panel, end-of-read report]",
    notes="No screenshot is committed to the repo, so this is a placeholder. "
          "Grab a live screenshot before the defense.",
)

# 18 — Real-vehicle test
content_slide(
    18, "Testing on a real (healthy) car",
    [
        "Healthy 2015 Yaris — MAF-based, no MAP sensor",
        "Air fault correctly marked 'Untested'",
        "Only 7 of 169 windows evaluable → 'Insufficient data'",
        "That refusal is the correct answer, not a failure",
        "An earlier version wrongly cried '97% air fault' — we fixed it",
    ],
    notes="This non-result is our strongest evidence the reporting layer works. "
          "The old version guessed with 97% confidence on a healthy car; the new "
          "one says 'I can't test this'.",
)

# 19 — Future work
content_slide(
    19, "What comes next",
    [
        "Real fault data — the one change that matters most",
        "Live phone → cloud server feed (FastAPI, already built)",
        "Apache Kafka to score many cars at once",
        "Per-regime baselines and more fault shapes",
        "One real recording turns 'algorithm' into 'validated detector'",
    ],
    notes="Order is by impact, not chronology. Real data is item one because it "
          "is the only thing that converts synthetic scores into field proof.",
)

# 20 — Conclusion
content_slide(
    20, "Conclusion",
    [
        "Built: a physics fault injector and a cross-vehicle detector",
        "Built: a live dashboard with an honest health report",
        "It reports what it cannot test and refuses to guess",
        "Not built: a proven real-fault detector — that needs real data",
        "The discipline of refusal is the real contribution",
    ],
    notes="Close on the honest framing. Fewer confident-looking numbers, but a "
          "tool you could actually trust on a car.",
)

# 21 — Thank you
slide = prs.slides.add_slide(BLANK)
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
_set_fill(bg, NAVY)
if LOGO.exists():
    slide.shapes.add_picture(str(LOGO), Inches(5.92), Inches(1.4), height=Inches(1.5))
ty = slide.shapes.add_textbox(Inches(0.7), Inches(3.2), Inches(11.9), Inches(1.2))
_para(_tf(ty), "Thank you", 44, WHITE, bold=True, align=PP_ALIGN.CENTER, first=True)
q = slide.shapes.add_textbox(Inches(0.7), Inches(4.4), Inches(11.9), Inches(0.6))
_para(_tf(q), "Questions & discussion", 20, TEAL, align=PP_ALIGN.CENTER, first=True)
repo = slide.shapes.add_textbox(Inches(0.7), Inches(5.2), Inches(11.9), Inches(0.5))
_para(_tf(repo), "github.com/AdamMohamedOmar/Predictive-MaintenanceV1", 13,
      RGBColor(0x9F, 0xB0, 0xC4), align=PP_ALIGN.CENTER, first=True)
numb = slide.shapes.add_textbox(Inches(12.4), Inches(6.98), Inches(0.7), Inches(0.4))
_para(_tf(numb), "21", 11, RGBColor(0x8B, 0x95, 0xA1), bold=True,
      align=PP_ALIGN.RIGHT, first=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved {OUT} with {len(prs.slides._sldIdLst)} slides")
