"""Generate all figures for the graduation book (docs/book/figures/*.png).

Adapted from notebooks/phase1_eda.ipynb (same regime thresholds, same plots),
plus three book-only figures: the before/after parser comparison, the XGBoost
confusion matrix, and the system-architecture diagram.

Run from the repo root:
    python docs/book/figures/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
FIG = Path(__file__).resolve().parent
FIG.mkdir(exist_ok=True)

from src.config import DATA_CAROBD_DIR, USEFUL_PIDS
from src.data_loading import load_carobd_csv

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 150

WINDOW_LEN, STRIDE = 60, 10
REGIMES = ["cold_start", "warmup", "idle", "accel", "cruise"]


def detect_regime(win: pd.DataFrame) -> str:
    coolant = win["COOLANT_TEMPERATURE"].mean() if "COOLANT_TEMPERATURE" in win else 90.0
    speed = win["VEHICLE_SPEED"].mean() if "VEHICLE_SPEED" in win else 0.0
    pedal_s = win["ACCELERATOR_PEDAL_POSITION_D"].std() if "ACCELERATOR_PEDAL_POSITION_D" in win else 0.0
    if coolant < 55.0:
        return "cold_start"
    if coolant < 75.0:
        return "warmup"
    if speed < 3.0:
        return "idle"
    if pedal_s > 8.0:
        return "accel"
    return "cruise"


def family(stem: str) -> str:
    return stem.rstrip("0123456789")


# ── Load corpus (fixed parser) ────────────────────────────────────────────────
files = sorted(DATA_CAROBD_DIR.glob("*.csv"))
frames, win_rows = [], []
for p in files:
    df = load_carobd_csv(p)
    df["session_id"] = p.stem
    df["family"] = family(p.stem)
    frames.append(df)
    arr = df.reset_index(drop=True)
    for start in range(0, max(0, len(arr) - WINDOW_LEN) + 1, STRIDE):
        w = arr.iloc[start:start + WINDOW_LEN]
        win_rows.append({"session_id": p.stem, "family": family(p.stem),
                         "regime": detect_regime(w)})

data = pd.concat(frames, ignore_index=True)
windows = pd.DataFrame(win_rows)

stats = {
    "n_files": len(files),
    "n_rows": int(len(data)),
    "hours": round(len(data) / 3600, 1),
    "n_windows": int(len(windows)),
    "families": data.groupby("family")["session_id"].nunique().to_dict(),
    "regime_share_pct": (windows["regime"].value_counts(normalize=True)
                         .reindex(REGIMES) * 100).round(1).to_dict(),
    "duplicate_rows": int(data.duplicated().sum()),
}

# ── Fig 1: sessions per family ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fam_order = ["idle", "live", "drive", "long", "ufpe"]
fc = data.groupby("family")["session_id"].nunique().reindex(fam_order)
fc.plot.bar(ax=ax, color="#4C72B0")
ax.set_title("Sessions per family (129 carOBD files)")
ax.set_ylabel("number of files")
ax.set_xlabel("session family")
ax.tick_params(axis="x", rotation=0)
for i, v in enumerate(fc):
    ax.text(i, v + 0.5, str(int(v)), ha="center")
plt.tight_layout()
fig.savefig(FIG / "fig_family_sessions.png")
plt.close(fig)

# ── Fig 2: windows per regime ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
rc = windows["regime"].value_counts().reindex(REGIMES)
rc.plot.bar(ax=ax, color="#DD8452")
ax.set_title("60-second windows per operating regime")
ax.set_ylabel("number of windows")
ax.set_xlabel("operating regime")
ax.tick_params(axis="x", rotation=15)
for i, v in enumerate(rc):
    share = v / rc.sum() * 100
    ax.text(i, v + rc.max() * 0.01, f"{int(v):,}\n({share:.1f}%)",
            ha="center", fontsize=8)
plt.tight_layout()
fig.savefig(FIG / "fig_regime_windows.png")
plt.close(fig)

# ── Fig 3: TIMING_ADVANCE before vs after the parser fix ────────────────────
naive_vals, fixed_vals = [], []
for p in files:
    try:
        raw = pd.read_csv(p)          # the buggy parse (column shift on ~120 files)
        col = [c for c in raw.columns if "TIMING_ADVANCE" in c]
        if col:
            naive_vals.append(raw[col[0]].dropna())
    except Exception:
        pass
    good = pd.read_csv(p, index_col=False)
    col = [c for c in good.columns if "TIMING_ADVANCE" in c]
    if col:
        fixed_vals.append(pd.to_numeric(good[col[0]], errors="coerce").dropna())

naive = pd.concat(naive_vals, ignore_index=True)
fixed = pd.concat(fixed_vals, ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].hist(naive, bins=80, color="#C44E52")
axes[0].set_title("Bare pd.read_csv (column shift)")
axes[0].set_xlabel("TIMING_ADVANCE column value")
axes[0].set_ylabel("row count (log scale)")
axes[0].set_yscale("log")
axes[0].axvline(64, color="k", ls="--", lw=1)
axes[0].axvline(-64, color="k", ls="--", lw=1)
axes[0].annotate("OBD-II bound ±64°", xy=(64, 1e5), fontsize=8, rotation=90,
                 xytext=(80, 3e3))
axes[1].hist(fixed, bins=80, color="#55A868")
axes[1].set_title("index_col=False (aligned)")
axes[1].set_xlabel("TIMING_ADVANCE column value")
axes[1].set_yscale("log")
axes[1].axvline(64, color="k", ls="--", lw=1)
axes[1].axvline(-64, color="k", ls="--", lw=1)
fig.suptitle("The trailing-comma column shift, seen through one PID (all 129 files)")
plt.tight_layout()
fig.savefig(FIG / "fig_timing_fix.png")
plt.close(fig)
stats["naive_timing_max"] = float(naive.max())
stats["fixed_timing_min"] = float(fixed.min())
stats["fixed_timing_max"] = float(fixed.max())

# ── Fig 4: correlation heatmap ───────────────────────────────────────────────
corr = data[USEFUL_PIDS].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8},
            annot_kws={"size": 6.5}, ax=ax)
ax.set_title("Correlation heatmap of the 14 working PIDs")
plt.tight_layout()
fig.savefig(FIG / "fig_corr_heatmap.png")
plt.close(fig)

# ── Fig 5: fault-bearing PID distributions ───────────────────────────────────
fault_pids = ["LONG_TERM_FUEL_TRIM_BANK_1", "SHORT_TERM_FUEL_TRIM_BANK_1",
              "INTAKE_MANIFOLD_PRESSURE", "COOLANT_TEMPERATURE", "TIMING_ADVANCE"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axes.ravel(), fault_pids):
    sns.histplot(data[c].dropna(), bins=60, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(c, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
axes.ravel()[-1].axis("off")
fig.suptitle("Distributions of the five key fault-bearing PIDs (all 129 files)")
plt.tight_layout()
fig.savefig(FIG / "fig_fault_pid_distributions.png")
plt.close(fig)

# ── Fig 6: bivariate views ───────────────────────────────────────────────────
samp = data.sample(min(15000, len(data)), random_state=42).copy()


def row_regime(r):
    if r.COOLANT_TEMPERATURE < 55:
        return "cold_start"
    if r.COOLANT_TEMPERATURE < 75:
        return "warmup"
    if r.VEHICLE_SPEED < 3:
        return "idle"
    return "driving"


samp["regime"] = samp.apply(row_regime, axis=1)
pal = {"cold_start": "#8172B3", "warmup": "#CCB974",
       "idle": "#DD8452", "driving": "#4C72B0"}
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
sns.scatterplot(data=samp, x="ENGINE_RPM", y="INTAKE_MANIFOLD_PRESSURE",
                hue="regime", palette=pal, s=8, alpha=.4, ax=ax[0], legend=True)
ax[0].set_title("MAP vs RPM")
sns.scatterplot(data=samp, x="VEHICLE_SPEED", y="ENGINE_RPM", hue="regime",
                palette=pal, s=8, alpha=.4, ax=ax[1], legend=False)
ax[1].set_title("RPM vs Speed (gear bands)")
sns.scatterplot(data=samp, x="COOLANT_TEMPERATURE", y="TIMING_ADVANCE",
                hue="regime", palette=pal, s=8, alpha=.4, ax=ax[2], legend=False)
ax[2].set_title("Timing vs Coolant")
plt.tight_layout()
fig.savefig(FIG / "fig_bivariate.png")
plt.close(fig)

# ── Fig 7: XGBoost fixed-holdout confusion matrix ────────────────────────────
with open(ROOT / "results" / "xgb_classifier_v1_results.json") as f:
    xgb = json.load(f)
cm = np.array(xgb["confusion_matrix"])
labels = [l.replace("_", "\n") for l in xgb["label_order"]]
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("predicted label")
ax.set_ylabel("true (injected) label")
ax.set_title(f"XGBoost fixed-holdout confusion matrix "
             f"(macro-F1 = {xgb['macro_f1']:.3f})")
plt.tight_layout()
fig.savefig(FIG / "fig_confusion_xgb.png")
plt.close(fig)

# ── Fig 8: system architecture diagram ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.axis("off")


def box(x, y, w, h, text, fc="#E8F0FE", ec="#1A73E8", fs=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4,
                               zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, wrap=True)


def arrow(x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, text, ha="center",
                fontsize=8, color="#444")


# acquisition row
box(0.2, 4.6, 1.7, 1.1, "Vehicle\n(OBD-II port,\nCAN / ISO 15765-4)")
box(2.6, 4.6, 1.7, 1.1, "ELM327\nBluetooth adapter")
box(5.0, 4.6, 2.0, 1.1, "OBD-LOG\nAndroid app\n(Kotlin / Compose)")
box(7.7, 4.6, 1.6, 1.1, "CSV export\n(26 columns)")
arrow(1.9, 5.15, 2.6, 5.15)
arrow(4.3, 5.15, 5.0, 5.15, "SPP/RFCOMM")
arrow(7.0, 5.15, 7.7, 5.15)

# training row
box(0.2, 2.6, 2.1, 1.1, "carOBD corpus\n129 sessions, 1 Hz\n(Toyota Etios)")
box(2.9, 2.6, 2.1, 1.1, "Fault injection\nengine\n(physics-constrained)")
box(5.6, 2.6, 2.1, 1.1, "Feature pipeline\n60 s windows,\n83 features, z-scored")
box(8.3, 2.6, 2.3, 1.1, "Models\nXGBoost classifier,\nPID forecaster,\nIsolationForest", fs=8)
arrow(2.3, 3.15, 2.9, 3.15)
arrow(5.0, 3.15, 5.6, 3.15)
arrow(7.7, 3.15, 8.3, 3.15)

# inference row
box(2.9, 0.6, 2.1, 1.1, "Adapter\n(rename + 1 Hz\nresample)")
box(5.6, 0.6, 2.1, 1.1, "InferenceEngine\nclassifier + SHAP,\nphysics severity,\nStableAlerter", fs=8)
box(8.3, 0.6, 2.3, 1.1, "Streamlit dashboard\n+ end-of-read\nhealth report")
arrow(8.5, 4.6, 4.0, 1.7)           # csv export -> adapter
arrow(5.0, 1.15, 5.6, 1.15)
arrow(7.7, 1.15, 8.3, 1.15)
arrow(9.4, 2.6, 6.9, 1.7)           # models -> inference

ax.text(0.2, 1.0, "Pillar 1: classification\n(what fault?)\nPillar 2: forecaster +\nanomaly (is something\ndrifting?)",
        fontsize=8.5, va="center",
        bbox=dict(facecolor="#FFF8E1", edgecolor="#F9A825"))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.2)
fig.savefig(FIG / "fig_architecture.png", bbox_inches="tight")
plt.close(fig)

with open(FIG / "figure_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
print("figures written to", FIG)
