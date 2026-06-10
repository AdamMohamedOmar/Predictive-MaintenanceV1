"""End-to-end bench: replay CSV -> _run_session -> engine -> store -> WS frames.
Covers the live stack without an ELM327. Slow (~10-20 s: SHAP init)."""

import json
import os
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


