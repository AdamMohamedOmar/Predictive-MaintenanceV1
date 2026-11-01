"""Build the defense speaker script: docs/presentation/script.docx.

Reproducible: run from the repo root with the project venv.
    python docs/presentation/build_docx.py

Slide assignment (covers all 21 slides, no gaps, no overlaps):
    Mohamed Sameh      — Slides 1 to 6
    Mohamed Abokhatwa  — Slides 7 to 9
    Omar Hossam        — Slides 10 to 12
    Adam Mohamed       — Slides 13 to 15
    Ahmed Saad         — Slides 16 to 21
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = Path(__file__).resolve().parent / "script.docx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
TEAL = RGBColor(0x14, 0x7A, 0x86)
GOLD = RGBColor(0x8A, 0x6D, 0x10)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(8)
style.paragraph_format.line_spacing = 1.15


def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = NAVY
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    return p


def speaker_header(name, srange):
    p = doc.add_paragraph()
    r = p.add_run(f"{name} — {srange}")
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = TEAL
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    # thin rule
    return p


def cue(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = GOLD
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    return p


def say(text):
    p = doc.add_paragraph(text)
    return p


def note(label, text):
    p = doc.add_paragraph()
    r = p.add_run(f"{label} ")
    r.bold = True
    r.italic = True
    r2 = p.add_run(text)
    r2.italic = True
    return p


# ── Cover ─────────────────────────────────────────────────────────────────────
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Defense Speaker Script")
r.bold = True
r.font.size = Pt(24)
r.font.color.rgb = NAVY

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Predictive Maintenance for OBD-II Engine Faults")
r.font.size = Pt(14)
r.font.color.rgb = TEAL

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run(
    "Arab Academy for Science, Technology and Maritime Transport\n"
    "College of Engineering and Technology — Computer Engineering Department\n"
    "Defense — July 2026"
).font.size = Pt(11)

doc.add_paragraph()
note("How to use this script:",
     "Each speaker's section is marked with the exact slides they present. "
     "Slide cues in gold show where to click to the next slide. Delivery is "
     "paced for roughly one and a half minutes per slide, about 140 words a "
     "minute. Say the numbers the way they are written out — they are meant to "
     "be spoken, not read off the slide.")

roster = doc.add_paragraph()
roster.add_run("Speaking order and slide ranges:\n").bold = True
roster.add_run(
    "Mohamed Sameh — Slides 1 to 6      (Mechanical: intro, the car, the sensors)\n"
    "Mohamed Abokhatwa — Slides 7 to 9  (Mechanical: fault physics, data collection)\n"
    "Omar Hossam — Slides 10 to 12       (CE: problem, dataset)\n"
    "Adam Mohamed — Slides 13 to 15      (CE: injection, features, model)\n"
    "Ahmed Saad — Slides 16 to 21        (CE: results, dashboard, future work, close)"
)

doc.add_page_break()

# ══════════════════════ MOHAMED SAMEH — Slides 1 to 6 ══════════════════════
speaker_header("Mohamed Sameh", "Slides 1 to 6")

cue("(Slide 1 — Title)")
say("Good morning, and thank you all for being here. We are five students from "
    "the Computer and Mechanical Engineering departments, and this is our "
    "graduation project: Predictive Maintenance for OBD-II Engine Faults. In "
    "plain terms, we built a tool that listens to a car's own sensors and tries "
    "to spot an engine fault while it is still developing — before the "
    "check-engine light ever comes on. My name is Mohamed Sameh, and I'll start "
    "us off on the mechanical side.")

cue("(Slide 2 — Agenda)")
say("The talk comes in two halves. In the first half, my colleague and I cover "
    "the car itself — the diagnostic port, the adapter we used, the sensors "
    "behind each fault, and the physics of how each fault actually behaves in "
    "the engine. In the second half, the computer-engineering team takes over "
    "and shows how we turned that sensor data into a model that can flag a "
    "fault. This one picture on the right is the whole system, and by the end "
    "you'll have seen every box in it.")

cue("(Slide 3 — Part 1: Mechanical Engineering)")
say("So let's start with the foundation everything else sits on: what the car "
    "is actually telling us, and what physically goes wrong inside the engine.")

cue("(Slide 4 — What every car already tells us)")
say("Here's the thing most drivers never realize. Almost every car built since "
    "the mid-nineties has a small diagnostic port under the dashboard, called "
    "OBD-II. It was originally put there for emissions testing, but it does far "
    "more than that. Through that port, the engine's computer will happily "
    "report its sensor readings — about once every second. Engine speed, "
    "coolant temperature, how much fuel it's adding or subtracting, the "
    "throttle position, the ignition timing, and more. Think about what that "
    "means. The car is already measuring its own health, continuously, and "
    "writing it down where anyone can read it.")
say("It's all standardized — there are formal specifications, SAE J1979 and "
    "ISO 15765-4, that define what each reading means and how to ask for it. "
    "So the same simple adapter works across brands and model years. The data "
    "is already there, in every car, all the time. The problem isn't getting "
    "the data — it's that almost nobody is listening to it until something has "
    "already broken. That gap is the opening our whole project aims at.")

cue("(Slide 5 — The test car and the reader)")
say("For our testing we used a 2015 Toyota Yaris — an ordinary car, which is "
    "exactly the point. To read the port, we used an ELM327 Bluetooth adapter. "
    "It just plugs in, and it costs around fifteen dollars. No special garage "
    "equipment, no dealer tools. That adapter talks over Bluetooth to an "
    "Android app our team wrote, called OBD-LOG, and the app records every "
    "reading to a file with a timestamp, so we can replay the drive later.")
say("There's one honest detail worth knowing up front, because it shaped our "
    "design. The adapter reads the sensors one at a time, in a loop — so the "
    "more sensors you ask for, the slower each one comes back around. When we "
    "polled seven sensors we got a little over one reading per second. When we "
    "pushed it to twenty-one sensors, that dropped to about a third of a "
    "reading per second. So there's a real trade-off: watch more channels, or "
    "watch fewer channels faster. We grouped the sensors into tiers so we could "
    "choose deliberately, keeping the most fault-relevant ones in the fast "
    "group.")

cue("(Slide 6 — The sensors behind each fault)")
say("This slide is the bridge into the physics, so let me connect each fault to "
    "the sensor that gives it away. An air leak shows up in the manifold "
    "pressure and the idle speed. A fuel problem shows up in the fuel trims — "
    "and I'll explain trims in a moment, but for now think of them as the "
    "correction the computer is applying to keep the mixture right. A "
    "temperature fault shows up in the coolant sensor reading. And throttle "
    "wear shows up as a mismatch between the throttle angle the car reports and "
    "where the pedal actually is.")
say("The key idea — and this is what makes the whole project possible — is "
    "that each fault pushes its own sensors in a known, predictable direction. "
    "A vacuum leak doesn't move things randomly; it raises idle speed and "
    "manifold pressure, every time, for the same physical reason. Once you know "
    "the direction each fault pushes, you know what pattern to look for. My "
    "colleague Mohamed Abokhatwa will now walk through what's physically "
    "happening inside the engine in each case.")

# ══════════════════════ MOHAMED ABOKHATWA — Slides 7 to 9 ══════════════════════
speaker_header("Mohamed Abokhatwa", "Slides 7 to 9")

cue("(Slide 7 — The physics, part one: air and fuel)")
say("Thanks, Sameh. Let me make the physics concrete, starting with air and "
    "fuel. A vacuum leak is exactly what it sounds like — a cracked hose or a "
    "bad seal lets extra air sneak into the engine past the throttle. The "
    "engine ends up breathing more air than the driver asked for, so it idles "
    "faster, almost as if you'd left the throttle cracked open a little. On the "
    "engine in our data, the manifold pressure sensor sees that extra air "
    "directly, and the computer mostly corrects for it on its own. So the "
    "clearest sign is simply that raised idle. That's the whole picture for the "
    "air fault, and keeping it simple is the honest description here.")
say("The fuel fault is a different story. Picture an injector that's partly "
    "clogged, or a fuel pump that can't quite hold pressure, so each spray "
    "delivers a little less fuel than it should. Now the mixture runs lean — "
    "too much air for the fuel. The computer notices the exhaust is lean and "
    "starts adding fuel to compensate. That running correction is called the "
    "long-term fuel trim, and under this fault it climbs and stays high. Here's "
    "the detail we really lean on: a fuel fault leaves the manifold pressure "
    "completely normal, because the air side is fine. So if the trims climb but "
    "the pressure is untouched, that points at fuel, not air. That one contrast "
    "separates the two faults that otherwise look alike.")

cue("(Slide 8 — The physics, part two: coolant and throttle)")
say("Two more faults. The first is a coolant temperature sensor that's stuck "
    "reading too cold. The engine warms up normally, but the sensor keeps "
    "insisting it's cold — so the computer believes the engine never warmed up. "
    "It reacts the way it would on a freezing morning: it adjusts the ignition "
    "timing and it over-fuels, running the mixture rich to help a cold engine "
    "that isn't actually cold. So the trims go rich, and the timing sits back. "
    "The engine is basically being managed for weather that isn't there.")
say("The last one is the throttle position sensor, and it's a wear problem. "
    "That sensor is essentially a little potentiometer, and over thousands of "
    "miles its reading starts to drift away from where the pedal actually is. "
    "The interesting part is that it only really shows under load — when you're "
    "genuinely pressing the pedal — because at rest the sensor isn't stressed. "
    "And here's the clue we use: the command the computer sends to the throttle "
    "stays correct, only the reported position drifts. So the gap between what "
    "was commanded and what's reported becomes a second, independent signal "
    "that the sensor is wearing out. Each of these four faults leaves its own "
    "fingerprint.")

cue("(Slide 9 — Collecting real driving data)")
say("Now, how did we actually get our data? We drove the car on public roads, "
    "with the driver's consent, at legal speeds, with the phone mounted and the "
    "app logging every sensor once a second. A mix of conditions — idling, city "
    "streets, steady cruising — because a fault can hide in one and show in "
    "another, and we wanted the full range of normal behavior on record.")
say("I want to be very clear about one thing, because it matters both "
    "ethically and technically. We never induced a fault on the car. Not once. "
    "Every drive we recorded was a healthy car behaving normally. We did "
    "consider deliberately creating a fault under supervision, and we decided "
    "against it for this stage — safety came first. What all that healthy "
    "driving gives us is a clean baseline for each vehicle: a picture of what "
    "normal looks like for that specific car. And that baseline is exactly what "
    "the model on the computer-engineering side measures everything against. "
    "With that foundation in place, I'll pass it to Omar to start the second "
    "half.")

# ══════════════════════ OMAR HOSSAM — Slides 10 to 12 ══════════════════════
speaker_header("Omar Hossam", "Slides 10 to 12")

cue("(Slide 10 — Part 2: Computer Engineering)")
say("Thank you, Mohamed. I'm Omar, and I'll set up the computer-engineering "
    "half — the problem we're really solving, and the data we had to fight to "
    "get.")

cue("(Slide 11 — The problem, stated honestly)")
say("Our goal is to flag a developing fault before the warning light comes on. "
    "That's simple to say. But here's the wall we hit, and I want to say it "
    "plainly, right at the start, because it shapes everything that follows. "
    "There is no public dataset of real, labeled engine faults recorded through "
    "this port at one reading per second. We looked. Nobody has published one, "
    "and there are structural reasons why — collecting it means either "
    "deliberately breaking cars or waiting for a fleet of them to break while "
    "you log everything, and whoever pays for that keeps the result private.")
say("So we did the honest thing that was actually available to us. We trained "
    "on synthetic faults — faults we inject ourselves into real, healthy "
    "driving data. And I want to be upfront: we report our results as synthetic "
    "validation, not as proof that the system catches real faults out on the "
    "road. Every single number you'll hear in the next few minutes carries that "
    "asterisk. What we're really presenting is a detection method and an honest "
    "way to measure it, not a finished product that's been proven on broken "
    "cars.")

cue("(Slide 12 — Rescuing the dataset)")
say("Before we could train anything, we had a data problem — and honestly, "
    "it's one of my favorite parts of the whole project, because it taught us "
    "more than any lecture could. Our training data is a public set of a "
    "hundred and twenty-nine driving files. When we first audited them, a "
    "hundred and twenty of those files looked physically impossible. The "
    "ignition timing column had values like three hundred, seven hundred "
    "degrees, when the sensor physically can't exceed sixty-four. We concluded "
    "the files were corrupt and nearly threw all hundred and twenty away, "
    "keeping only the nine that looked clean.")
say("We were wrong, and the real cause was almost funny. Each row in those "
    "files ended with one extra comma. When our code loaded the file, that "
    "single trailing comma shifted every column over by one position. So the "
    "column we thought was timing was actually holding catalyst temperature — "
    "and seven hundred degrees is a perfectly normal catalyst temperature. The "
    "data was never broken; our reading of it was. One flag in the file loader "
    "fixed it, and we added a guard that now fails loudly if it ever happens "
    "again. That recovery took us from nine usable files, about four and a half "
    "hours, to all one hundred twenty-nine — roughly eighty-four and a half "
    "hours. Nineteen times more data, from fixing one character. Adam will take "
    "it from here into how we build the model on top of that data.")

# ══════════════════════ ADAM MOHAMED — Slides 13 to 15 ══════════════════════
speaker_header("Adam Mohamed", "Slides 13 to 15")

cue("(Slide 13 — Making faults we could learn from)")
say("Thanks, Omar. So if there's no real fault data to learn from, how do we "
    "make our own? We inject faults — but very carefully. We take a real, "
    "healthy drive and we write four fault signatures into it, one at a time. "
    "The one rule we never break is physics. If our injected fault produces a "
    "sensor reading a real engine could never produce, the whole approach is "
    "worthless, because the model would learn a fantasy.")
say("So we clamp everything to what's physically real. A coolant reading can't "
    "jump more than one degree per second, because metal has thermal inertia — "
    "it simply can't heat or cool faster than that. Fuel trims are capped at "
    "plus or minus twenty-five percent, because that's the real limit of the "
    "system. We ramp each fault in gradually, over time, to mimic a component "
    "slowly wearing out rather than snapping all at once. And every injection "
    "is fully reproducible — the random seed, the timing, the size, all stored, "
    "so any result can be traced back to exactly how it was made. A naive "
    "injector would happily write textbook signatures that real engines never "
    "actually produce. Ours refuses to, and that discipline is what makes the "
    "training data trustworthy.")

cue("(Slide 14 — Turning sensors into features)")
say("Next we turn raw sensor readings into something a model can actually use. "
    "We slice the drive into sixty-second windows, and we summarize each window "
    "into eighty-three numbers. Some are simple statistics — the average and "
    "the spread of each sensor. Some are ratios between sensors, like throttle "
    "against pedal. Some capture trends, like how fast the coolant is warming. "
    "And a few are just flags for what the car is doing — idling, cruising, "
    "accelerating — so the model has context.")
say("But here's the step that really matters. Every one of those eighty-three "
    "numbers is scored against that specific car's own healthy baseline. So the "
    "model never thinks in raw values like 'fuel trim equals fifteen percent.' "
    "It thinks in terms like 'this is four standard deviations above normal, "
    "for this car.' That distinction is the whole trick that lets a model "
    "trained on one car have any chance on a completely different one. A healthy "
    "Toyota and a healthy anything-else will disagree on the raw numbers, for "
    "perfectly innocent reasons. But they agree on what 'far from my own "
    "normal' means. That shared language of deviation is what travels across "
    "vehicles.")

cue("(Slide 15 — The classifier, and knowing its limits)")
say("The model itself is XGBoost — a gradient-boosted tree model — over six "
    "classes: healthy, cold start, and our four faults. Two things I'd "
    "highlight. First, it explains itself: for every prediction, it tells us "
    "which sensors drove the decision and by how much, so a mechanic can check "
    "the reasoning instead of trusting a black box. Second — and this is a "
    "design choice we're proud of — if a car doesn't even have the sensor a "
    "fault depends on, the model is allowed to say 'I can't test this here' "
    "instead of guessing. It marks that fault Untested. Ahmed will show you why "
    "that one decision mattered so much.")

# ══════════════════════ AHMED SAAD — Slides 16 to 21 ══════════════════════
speaker_header("Ahmed Saad", "Slides 16 to 21")

cue("(Slide 16 — Results, read honestly)")
say("Thanks, Adam. Let's talk numbers — and let's read them honestly, which is "
    "the whole spirit of this project. We measure with a score called macro-F1, "
    "where one point oh is perfect. Our headline comes from the hardest fair "
    "test we ran: we hold out an entire driving context the model has never "
    "seen, and ask it to classify faults there. On that test it scored just "
    "under ninety-six percent — zero point nine five eight. On the single "
    "toughest individual recording, it dropped to about zero point seven-oh, "
    "and we deliberately report that floor rather than hiding it behind an "
    "average.")
say("Now here's the honest part I really want to land. A simpler model, a "
    "Random Forest, scored a perfect one point oh on the same task. And we "
    "treat that perfect score as a warning, not a trophy. A perfect score "
    "doesn't mean the problem is solved — it means the synthetic task was easy "
    "enough to memorize. That's a red flag, and we kept it in the report on "
    "purpose to stay honest with ourselves. We also ran a check: we changed the "
    "injector's settings between training and testing, and the score barely "
    "moved — a gap of only zero point oh-three-three. So the model isn't just "
    "memorizing our exact recipe. But I'll repeat the asterisk one more time: "
    "every number on this slide is synthetic self-consistency. It is not "
    "real-world fault detection.")

cue("(Slide 17 — The live dashboard)")
say("All of this runs live in a dashboard. It shows the current classification, "
    "the fuel trims, and the explanation, updating as the data streams. The "
    "severity bars come from physics, not from guessing. And when a recording "
    "ends, it produces one plain verdict per fault. The part I care about most: "
    "when the data isn't good enough to judge, it refuses to give a verdict. It "
    "would rather say 'I don't know' than make something up.")

cue("(Slide 18 — Testing on a real, healthy car)")
say("Here's where that refusal earned its keep. We ran the whole pipeline on a "
    "real, healthy 2015 Yaris. That car is MAF-based — it has no manifold "
    "pressure sensor at all, which is the main sensor our air fault depends on. "
    "The system correctly marked the air fault Untested. Only seven of a "
    "hundred sixty-nine windows were even judgeable, so it returned "
    "'insufficient data' — and that's the right answer, not a failure. I'll be "
    "blunt about why this matters: an earlier version of our system, without "
    "that safeguard, looked at the same healthy car and confidently announced a "
    "ninety-seven percent air fault. Fabricated sensor values produce confident "
    "wrong answers. Fixing that is one of the things I'm proudest of.")

cue("(Slide 19 — What comes next)")
say("So where does this go from here? We ordered the future work by impact, not "
    "by what's easy, and the single most important next step is real fault "
    "data. One properly supervised recording of an actual induced fault — done "
    "safely, with a mechanic — is the one thing that turns this from an "
    "algorithm into a validated detector. Everything else is secondary to that.")
say("After that, there's a clear path. A live feed from the phone up to a cloud "
    "server, scoring the car in real time — and we've already built the backend "
    "for it. Then Apache Kafka, so the system can handle many cars streaming at "
    "once instead of one at a time. And richer fault shapes in the injector, "
    "because real components don't only fail on a smooth ramp — some flicker, "
    "some accelerate. Each of these has a specific reason it wasn't done yet, "
    "and each one is a concrete next step rather than a vague 'improve the "
    "model.'")

cue("(Slide 20 — Conclusion)")
say("To wrap up. We built a physics-based fault injector, a detector that "
    "works across different cars, and a live dashboard with an honest health "
    "report. It reports what it cannot test, and it refuses to guess. What we "
    "did not build is a proven real-fault detector — and I'll say it straight, "
    "because it's the truth: this model has never detected a real fault, "
    "because the data to prove that doesn't exist yet. We think the discipline "
    "of knowing what it can't do is the real contribution here.")

cue("(Slide 21 — Thank you and questions)")
say("That's our project. Thank you for your time and attention — and we'd be "
    "glad to take your questions.")

# ── TO FILL IN ────────────────────────────────────────────────────────────────
doc.add_page_break()
h1("TO FILL IN — remaining placeholders and review items")
note("1. Mechanical content (Slides 3 to 9):",
     "flagged REVIEW-NEEDED. The mechanical-engineering slides and the "
     "corresponding script (Mohamed Sameh, Mohamed Abokhatwa) use the task's "
     "default outline — OBD-II basics, the test vehicle and adapter, the fault "
     "physics, and the data-collection process. Replace with the two mechanical "
     "students' actual contribution and wording before the defense.")
note("2. Slide images to capture:",
     "Slides 4, 5, 7, 9, 13, and 17 use labeled [FIGURE] placeholder boxes. "
     "Add real images before the defense — in particular the OBD-II port photo "
     "(Slide 4), the Yaris-plus-adapter photo (Slide 5), and a live Streamlit "
     "dashboard screenshot (Slide 17).")
note("3. Defense date:",
     "the deck and this script show 'July 2026' as a placeholder date. Set the "
     "exact defense date on the title slide once it is confirmed.")
note("4. Supervisor titles:",
     "titles on the title slide are shown as 'Dr.' for both supervisors "
     "(Yahia Elderiny, Rania Kadry). Correct the honorifics if needed.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(f"Saved {OUT}")
