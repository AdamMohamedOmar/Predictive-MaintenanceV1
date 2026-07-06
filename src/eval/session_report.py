"""End-of-read engine-health report for CSV replay (self-baselined).

Design (all four decisions locked with the team):

1. Self-baseline: the first 20% of the recording (by elapsed time) is assumed
   healthy and used only to define "normal". It is EXCLUDED from the verdict —
   scoring baseline windows against themselves is circular.
2. Severity comes from the physics `compute_severity` per window (carried on
   DashboardState.severities), NOT the forecaster — the forecaster extrapolates
   off-distribution on healthy data and produced phantom 17%/27% severities.
3. Baseline-period windows are excluded from all verdict/severity aggregation.
4. If too few evaluable post-baseline windows remain, the report says
   "INSUFFICIENT DATA" rather than inventing a health verdict.

Honest scope (must appear on the rendered report):
  * This detects a fault that DEVELOPS during the recording (deviation from the
    car's own early-session healthy baseline).
  * It CANNOT detect a fault present from key-on — that becomes the baseline and
    reads as normal.
  * It assumes the first 20% is healthy. A car already faulty at key-on poisons
    the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Faults that carry a physics severity (the 4 injectable faults). healthy /
# cold_start have no severity.
SEVERITY_FAULTS = (
    "air_system",
    "fuel_system",
    "coolant_temp_sensor",
    "throttle_position_sensor",
)

# A fault is "Detected" if it is the winning label in at least this fraction of
# evaluable post-baseline windows (mirrors the score_recording verdict gate).
_DETECT_FRACTION = 0.40
# Below this many evaluable post-baseline windows, refuse to give a verdict.
_MIN_EVAL_WINDOWS = 10


@dataclass
class FaultReport:
    fault: str
    status: str  # "detected" | "healthy" | "untested"
    severity_pct: float | None  # 0..100, only for detected+evaluable; else None
    window_share_pct: float  # % of evaluable windows that got this label


@dataclass
class SessionReport:
    verdict: str
    n_windows_total: int
    n_baseline_windows: int
    n_evaluable_windows: int
    n_untested_windows: int
    faults: list[FaultReport] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (avoids a numpy dependency here)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_session_report(
    records: list[dict],
    untested_fault_set: set[str],
    *,
    baseline_frac: float = 0.20,
    detect_fraction: float = _DETECT_FRACTION,
    min_eval_windows: int = _MIN_EVAL_WINDOWS,
) -> SessionReport:
    """Build the end-of-read report.

    Parameters
    ----------
    records : list of dict
        One entry per scored window, in time order, each with:
          "elapsed_s": float
          "label":     str   (winning classifier label)
          "severities": dict[str, float]  (physics severity 0..1 per fault)
        Baseline windows are identified here by elapsed_s, so pass ALL windows.
    untested_fault_set : set[str]
        Faults whose primary PID is unavailable on this vehicle (from
        src.eval.pid_availability.untested_faults). These are never Detected;
        they are reported "untested".
    """
    n_total = len(records)
    if n_total == 0:
        return SessionReport("INSUFFICIENT DATA (no windows)", 0, 0, 0, 0)

    # Baseline cut-off by elapsed time (first `baseline_frac` of the recording).
    t_max = max(r["elapsed_s"] for r in records)
    cutoff = t_max * baseline_frac
    post = [r for r in records if r["elapsed_s"] >= cutoff]
    n_baseline = n_total - len(post)

    # Untested windows (winning label is an untested fault) are set aside.
    evaluable = [r for r in post if r["label"] not in untested_fault_set]
    n_untested = len(post) - len(evaluable)
    n_eval = len(evaluable)

    caveats = [
        "Detects faults that DEVELOP during the recording; cannot detect a fault "
        "present from key-on (it becomes the baseline).",
        f"Assumes the first {int(baseline_frac * 100)}% of the recording is healthy.",
    ]

    # Untested faults are reported regardless of how much evaluable data exists —
    # "we can't test this on this vehicle" is information even when the rest is thin.
    untested_reports = [
        FaultReport(f, "untested", None, 0.0)
        for f in SEVERITY_FAULTS
        if f in untested_fault_set
    ]

    if n_eval < min_eval_windows:
        return SessionReport(
            verdict=f"INSUFFICIENT DATA ({n_eval} evaluable windows; need >= {min_eval_windows})",
            n_windows_total=n_total,
            n_baseline_windows=n_baseline,
            n_evaluable_windows=n_eval,
            n_untested_windows=n_untested,
            faults=untested_reports,
            caveats=caveats,
        )

    # Per-fault status over evaluable post-baseline windows.
    faults: list[FaultReport] = list(untested_reports)
    any_detected = False
    for fault in SEVERITY_FAULTS:
        if fault in untested_fault_set:
            continue  # already reported as untested
        hits = [r for r in evaluable if r["label"] == fault]
        share = 100.0 * len(hits) / n_eval
        if share >= detect_fraction * 100.0:
            sev_vals = [r["severities"].get(fault, 0.0) for r in evaluable]
            sev_pct = round(_percentile(sev_vals, 95.0) * 100.0, 1)
            if sev_pct < 1.0:
                # Label dominance with ~zero physical severity is the signature
                # of a missing-PID / feature artifact, not a physical fault —
                # report it as inconclusive rather than "DETECTED, severity 0%".
                faults.append(FaultReport(fault, "inconclusive", None, round(share, 1)))
            else:
                faults.append(FaultReport(fault, "detected", sev_pct, round(share, 1)))
                any_detected = True
        else:
            faults.append(FaultReport(fault, "healthy", None, round(share, 1)))

    if any_detected:
        verdict = "DEVELOPING FAULT(S) DETECTED"
    else:
        verdict = "HEALTHY (no developing fault within session)"

    return SessionReport(
        verdict=verdict,
        n_windows_total=n_total,
        n_baseline_windows=n_baseline,
        n_evaluable_windows=n_eval,
        n_untested_windows=n_untested,
        faults=faults,
        caveats=caveats,
    )