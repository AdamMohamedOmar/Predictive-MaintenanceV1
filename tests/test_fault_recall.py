"""§10 headline-metric recall — the detection set is {fuel_system, air_system}
labels OR anomaly_score >= 0.85.  cold_start and wrong-fault labels are NOT
detections of a vacuum leak (docs/REAL_FAULT_COLLECTION.md §10)."""

import pytest

from src.eval.real_fault_eval import compute_fault_recall


def _w(elapsed_s: int, label: str, anomaly: float = 0.0) -> dict:
    return {"elapsed_s": elapsed_s, "label": label, "anomaly_score": anomaly}


def test_air_and_fuel_labels_count_as_detection():
    windows = [_w(100, "air_system"), _w(110, "fuel_system"), _w(120, "healthy")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=130)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["n_fault_windows"] == 3
    assert r["detected_by_label"] == 2


def test_cold_start_is_not_a_detection():
    windows = [_w(100, "cold_start"), _w(110, "cold_start"), _w(120, "air_system")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=130)
    assert r["recall"] == pytest.approx(1 / 3)


def test_wrong_fault_label_is_not_a_detection():
    windows = [_w(100, "coolant_temp_sensor"), _w(110, "throttle_position_sensor")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["recall"] == 0.0


def test_anomaly_route_counts_even_when_label_healthy():
    windows = [_w(100, "healthy", anomaly=0.90), _w(110, "healthy", anomaly=0.10)]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["recall"] == pytest.approx(0.5)
    assert r["detected_by_anomaly_only"] == 1


def test_windows_outside_interval_are_ignored():
    windows = [_w(50, "air_system"), _w(100, "healthy"), _w(500, "fuel_system")]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["n_fault_windows"] == 1
    assert r["recall"] == 0.0


def test_empty_interval_returns_zero_not_crash():
    r = compute_fault_recall([], fault_from_s=0, fault_to_s=100)
    assert r["recall"] == 0.0
    assert r["n_fault_windows"] == 0


def test_label_and_anomaly_on_same_window_not_double_counted():
    windows = [_w(100, "air_system", anomaly=0.99)]
    r = compute_fault_recall(windows, fault_from_s=90, fault_to_s=120)
    assert r["n_detected"] == 1
    assert r["recall"] == 1.0
