"""Tests for the feature pipeline: windowing, extraction, and dataset builder."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.windowing import sliding_windows, count_windows
from src.features.extractor import extract_features, feature_names
from src.features.dataset_builder import LABEL_TO_ID, FAULT_TYPES, build_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CAROBD_DIR = REPO_ROOT / "data" / "raw" / "carOBD"


# ─── Synthetic session fixture ────────────────────────────────────────────────

def _make_session(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    cols = {
        "ENGINE_RPM": rng.uniform(1800, 2200, n),
        "VEHICLE_SPEED": rng.uniform(45, 55, n),
        "THROTTLE": rng.uniform(15, 25, n),
        "ENGINE_LOAD": rng.uniform(30, 40, n),
        "COOLANT_TEMPERATURE": rng.uniform(88, 92, n),
        "LONG_TERM_FUEL_TRIM_BANK_1": rng.uniform(-3, 3, n),
        "SHORT_TERM_FUEL_TRIM_BANK_1": rng.uniform(-5, 5, n),
        "INTAKE_MANIFOLD_PRESSURE": rng.uniform(35, 45, n),
        "ABSOLUTE_BAROMETRIC_PRESSURE": np.full(n, 101.0),
        "ACCELERATOR_PEDAL_POSITION_D": rng.uniform(18, 28, n),
        "ACCELERATOR_PEDAL_POSITION_E": rng.uniform(18, 28, n),
        "COMMANDED_THROTTLE_ACTUATOR": rng.uniform(15, 25, n),
        "INTAKE_AIR_TEMPERATURE": rng.uniform(28, 32, n),
        "TIMING_ADVANCE": rng.uniform(12, 18, n),
        "CONTROL_MODULE_VOLTAGE": rng.uniform(13.8, 14.2, n),
    }
    df = pd.DataFrame(cols)
    df.attrs["session_id"] = "synthetic"
    return df


# ─── Windowing ────────────────────────────────────────────────────────────────

def test_window_count_formula():
    # (300 - 60) / 10 + 1 = 25
    assert count_windows(300, window_len=60, stride=10) == 25


def test_window_count_too_short():
    assert count_windows(50, window_len=60, stride=10) == 0


def test_sliding_windows_yields_correct_count():
    df = _make_session(300)
    windows = list(sliding_windows(df, "healthy", window_len=60, stride=10))
    assert len(windows) == count_windows(300, 60, 10)


def test_sliding_windows_label_propagated():
    df = _make_session(300)
    for _, label in sliding_windows(df, "air_system", window_len=60, stride=10):
        assert label == "air_system"


def test_sliding_windows_each_is_60_rows():
    df = _make_session(300)
    for window, _ in sliding_windows(df, "healthy", window_len=60, stride=10):
        assert len(window) == 60


def test_sliding_windows_index_reset():
    df = _make_session(300)
    for window, _ in sliding_windows(df, "healthy", window_len=60, stride=10):
        assert list(window.index) == list(range(60))


def test_sliding_windows_too_short_yields_nothing():
    df = _make_session(30)
    windows = list(sliding_windows(df, "healthy", window_len=60, stride=10))
    assert windows == []


# ─── Feature extraction ───────────────────────────────────────────────────────

def test_extract_features_returns_correct_count():
    df = _make_session(60)
    feats = extract_features(df)
    expected = 14 * 5 + 4 + 4 + 5  # 70 PID stats + 4 cross-PID + 4 trajectory + 5 regime = 83
    assert len(feats) == expected


def test_feature_names_length():
    assert len(feature_names()) == 83  # 70 PID stats + 4 cross-PID + 4 trajectory + 5 regime


def test_throttle_cmd_actual_delta_near_zero_when_equal():
    """When THROTTLE ≈ COMMANDED_THROTTLE_ACTUATOR the delta should be ~0."""
    df = _make_session(60)
    df["THROTTLE"] = 20.0
    df["COMMANDED_THROTTLE_ACTUATOR"] = 20.0
    feats = extract_features(df)
    assert abs(feats["THROTTLE_CMD_ACTUAL_DELTA"]) < 0.1


def test_feature_names_matches_extract_keys():
    df = _make_session(60)
    feats = extract_features(df)
    assert set(feats.keys()) == set(feature_names())


def test_extract_features_all_finite():
    df = _make_session(60)
    feats = extract_features(df)
    assert all(np.isfinite(v) for v in feats.values())


def test_extract_features_tolerates_missing_pid():
    """A window missing one PID must still return all 83 feature keys with no
    KeyError.  The five stats for the absent PID will be NaN — that is correct
    (downstream callers already NaN-fill with the healthy-baseline mean)."""

    df = _make_session(60)
    # Drop one PID entirely from the window
    df = df.drop(columns=["CONTROL_MODULE_VOLTAGE"])
    assert "CONTROL_MODULE_VOLTAGE" not in df.columns

    feats = extract_features(df)  # must not raise KeyError

    # All 83 feature keys must be present
    assert len(feats) == 83
    assert set(feats.keys()) == set(feature_names())

    # The five stats for the dropped PID are NaN (not missing)
    for stat in ("mean", "std", "min", "max", "delta"):
        key = f"CONTROL_MODULE_VOLTAGE__{stat}"
        assert key in feats
        assert np.isnan(feats[key]), f"{key} should be NaN for a missing PID"


def test_throttle_to_pedal_ratio_near_one_when_equal():
    df = _make_session(60)
    df["THROTTLE"] = 20.0
    df["ACCELERATOR_PEDAL_POSITION_D"] = 20.0
    feats = extract_features(df)
    # ratio should be close to 1.0 when throttle ≈ pedal
    assert abs(feats["THROTTLE_TO_PEDAL_RATIO"] - 1.0) < 0.01


def test_fuel_trim_divergence_zero_when_equal():
    df = _make_session(60)
    df["LONG_TERM_FUEL_TRIM_BANK_1"] = 5.0
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = 5.0
    feats = extract_features(df)
    assert feats["FUEL_TRIM_DIVERGENCE"] == pytest.approx(0.0)


def test_map_per_throttle_safe_when_throttle_zero():
    df = _make_session(60)
    df["THROTTLE"] = 0.0
    feats = extract_features(df)
    # Should not raise and should be finite (ε prevents /0)
    assert np.isfinite(feats["MAP_PER_THROTTLE"])


def test_warmup_rate_scales_with_sample_hz():
    """At 0.5 Hz the same row-sequence takes twice as long → slope is half."""
    df = _make_session(60)
    # Impose a clear linear coolant rise so polyfit has a strong signal
    df["COOLANT_TEMPERATURE"] = np.linspace(60.0, 80.0, 60)
    feats_1hz = extract_features(df, sample_hz=1.0)
    feats_05hz = extract_features(df, sample_hz=0.5)
    # Half the sample rate → real time is doubled → °C/min rate is halved
    assert feats_05hz["COOLANT_WARMUP_RATE"] == pytest.approx(
        feats_1hz["COOLANT_WARMUP_RATE"] / 2.0, rel=0.01
    )


def test_fuel_loop_threshold_scales_with_sample_hz():
    """FUEL_LOOP_ACTIVE threshold is 10 *seconds* of activity, not 10 rows."""
    df = _make_session(60)
    # Exactly 5 rows have |STFT| > 0.5 %
    df["SHORT_TERM_FUEL_TRIM_BANK_1"] = 0.0
    df.loc[df.index[:5], "SHORT_TERM_FUEL_TRIM_BANK_1"] = 1.0
    # At 1 Hz: threshold_rows = max(3, round(10*1.0)) = 10 → 5 < 10 → should NOT fire
    assert extract_features(df, sample_hz=1.0)["FUEL_LOOP_ACTIVE"] == 0.0
    # At 0.5 Hz: threshold_rows = max(3, round(10*0.5)) = 5 → 5 >= 5 → should fire
    assert extract_features(df, sample_hz=0.5)["FUEL_LOOP_ACTIVE"] == 1.0


# ─── Dataset builder ──────────────────────────────────────────────────────────

def test_label_to_id_covers_all_faults():
    assert "healthy" in LABEL_TO_ID
    for fault in FAULT_TYPES:
        assert fault in LABEL_TO_ID


def test_label_ids_are_unique():
    ids = list(LABEL_TO_ID.values())
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_dataset_produces_parquet(tmp_path):
    build_dataset(carobd_dir=CAROBD_DIR, output_dir=tmp_path)
    assert (tmp_path / "dataset_v1.parquet").exists()
    assert (tmp_path / "dataset_v1_meta.json").exists()


@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_dataset_has_all_classes(tmp_path):
    dataset = build_dataset(carobd_dir=CAROBD_DIR, output_dir=tmp_path)
    present_labels = set(dataset["label"].unique())
    # cold_start is a 6th label produced from cold-engine windows in clean sessions
    expected = {"healthy", "cold_start"} | set(FAULT_TYPES)
    assert present_labels == expected


@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_dataset_feature_columns(tmp_path):
    dataset = build_dataset(carobd_dir=CAROBD_DIR, output_dir=tmp_path)
    for col in feature_names():
        assert col in dataset.columns


@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_dataset_no_nan_in_features(tmp_path):
    dataset = build_dataset(carobd_dir=CAROBD_DIR, output_dir=tmp_path)
    feat_cols = feature_names()
    assert dataset[feat_cols].notna().all().all()


@pytest.mark.skipif(not CAROBD_DIR.exists(), reason="carOBD data not present")
def test_build_dataset_label_ids_consistent(tmp_path):
    dataset = build_dataset(carobd_dir=CAROBD_DIR, output_dir=tmp_path)
    for label, expected_id in LABEL_TO_ID.items():
        rows = dataset[dataset["label"] == label]
        if len(rows):
            assert (rows["label_id"] == expected_id).all()
