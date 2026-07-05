"""End-to-end bench: replay CSV -> _run_session -> engine -> store -> WS frames.
Covers the live stack without an ELM327. Slow (~10-20 s: SHAP init)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

DEMO = Path("data/demo/demo_fuel_system.csv")
YARIS = Path("data/real_faults/ahmed/ahmed_drive_20260602.csv")


@pytest.mark.skipif(not DEMO.exists(), reason="demo CSV not generated")
def test_replay_session_streams_frames(monkeypatch, tmp_path):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    monkeypatch.setenv("PM_REPLAY_FAST", "1")
    client = TestClient(app)
    with client.websocket_connect("/api/ws/live") as ws:
        ws.send_text(json.dumps({
            "action": "connect",
            "port": f"replay:{DEMO}",
            "car_id": None,
        }))
        frames = []
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "telemetry":
                frames.append(msg)
            if len(frames) >= 3:
                break
        ws.send_text(json.dumps({"action": "stop"}))

    assert len(frames) >= 3
    f = frames[-1]
    assert "telemetry" in f and "label" in f and "alert_events" in f
    assert f["poll_hz"] >= 0.0


@pytest.mark.skipif(not DEMO.exists(), reason="demo CSV not generated")
def test_replay_sessions_are_armed(monkeypatch):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    monkeypatch.setenv("PM_REPLAY_FAST", "1")
    client = TestClient(app)
    with client.websocket_connect("/api/ws/live") as ws:
        ws.send_text(json.dumps({"action": "connect", "port": f"replay:{DEMO}", "car_id": None}))
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "telemetry":
                assert msg["armed"] is True
                break
        ws.send_text(json.dumps({"action": "stop"}))


@pytest.mark.skipif(not YARIS.exists(), reason="Yaris drive not present")
def test_calibrate_mode_fits_and_reports(monkeypatch):
    monkeypatch.setenv("PM_ALLOW_REPLAY", "1")
    monkeypatch.setenv("PM_REPLAY_FAST", "1")
    client = TestClient(app)
    with client.websocket_connect("/api/ws/live") as ws:
        ws.send_text(json.dumps({
            "action": "connect",
            "port": f"replay:{YARIS}",
            "car_id": None,
            "mode": "calibrate",
            "allow_idle": True,
        }))
        progressed = False
        for _ in range(400):
            msg = ws.receive_json()
            if msg["type"] == "calibrate_progress":
                progressed = True
                if msg["rows_collected"] >= 400:
                    ws.send_text(json.dumps({"action": "finish_calibration"}))
            elif msg["type"] == "calibrate_result":
                assert msg["ok"] is True, msg
                assert msg["n_windows"] >= 20
                break
        assert progressed
