"""Tests for the end-of-read session health report."""

from src.eval.session_report import (
    SEVERITY_FAULTS,
    build_session_report,
)


def _rec(t, label, **sev):
    sevs = {f: 0.0 for f in SEVERITY_FAULTS}
    sevs.update(sev)
    return {"elapsed_s": float(t), "label": label, "severities": sevs}


def test_all_healthy_gives_healthy_verdict():
    recs = [_rec(t, "healthy") for t in range(60)]
    rep = build_session_report(recs, set())
    assert rep.verdict.startswith("HEALTHY")
    assert all(f.status == "healthy" for f in rep.faults)


def test_baseline_windows_excluded():
    # 100 windows; first 20% (t < 20) are baseline and must be excluded.
    recs = [_rec(t, "healthy") for t in range(100)]
    rep = build_session_report(recs, set())
    assert rep.n_baseline_windows == 20
    assert rep.n_evaluable_windows == 80


def test_developing_fault_detected_with_severity():
    # healthy baseline, then fuel_system dominates the post-baseline window
    recs = [_rec(t, "healthy") for t in range(20)]
    recs += [_rec(t, "fuel_system", fuel_system=0.6) for t in range(20, 100)]
    rep = build_session_report(recs, set())
    assert rep.verdict == "DEVELOPING FAULT(S) DETECTED"
    fuel = next(f for f in rep.faults if f.fault == "fuel_system")
    assert fuel.status == "detected"
    assert fuel.severity_pct is not None and fuel.severity_pct > 0


def test_untested_fault_never_detected():
    # air_system dominates but is untested (no MAP) -> reported untested, not fault
    recs = [_rec(t, "healthy") for t in range(20)]
    recs += [_rec(t, "air_system") for t in range(20, 100)]
    rep = build_session_report(recs, {"air_system"})
    air = next(f for f in rep.faults if f.fault == "air_system")
    assert air.status == "untested"
    # with air set aside, remaining evaluable windows are ~0 -> insufficient/healthy
    assert "FAULT" not in rep.verdict or rep.verdict.startswith("INSUFFICIENT")


def test_insufficient_data_floor():
    recs = [_rec(t, "healthy") for t in range(12)]  # ~2 evaluable after baseline
    rep = build_session_report(recs, set())
    assert rep.verdict.startswith("INSUFFICIENT DATA")


def test_severity_is_95th_percentile_not_max():
    # one spike shouldn't dominate; 95th percentile of mostly-0.3 with a 0.9 spike
    recs = [_rec(t, "healthy") for t in range(20)]
    recs += [_rec(t, "coolant_temp_sensor", coolant_temp_sensor=0.3) for t in range(20, 99)]
    recs += [_rec(99, "coolant_temp_sensor", coolant_temp_sensor=0.9)]
    rep = build_session_report(recs, set())
    cool = next(f for f in rep.faults if f.fault == "coolant_temp_sensor")
    assert cool.status == "detected"
    assert cool.severity_pct < 90.0  # spike didn't set the number