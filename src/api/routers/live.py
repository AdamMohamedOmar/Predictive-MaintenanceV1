"""Live session router — serial ports + WebSocket telemetry.

Protocol (WS /api/ws/live):
    Client → server (JSON):
        {action:'connect', port:'COM3', car_id:N}   – start; port=null for auto-detect
        {action:'mark_leak', state:'start'|'stop'}  – annotate fault window
        {action:'stop'}                             – request clean shutdown

    Server → client (JSON):
        {type:'telemetry', elapsed_s, telemetry:{14 PIDs or null},
         label, confidence, severities, forecasts, anomaly_score,
         top_shap:[[name,val],...], degraded_pid_count, poll_hz}
        {type:'warning', message}
        {type:'error',   message}
        {type:'mark_ack',state}
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.config import DATA_APP_DIR
from src.api.db import get_db
from src.api.live_store import LiveSessionStore
from src.api.models import Car

log = logging.getLogger(__name__)
router = APIRouter(tags=["live"])

# One live session at a time — the serial port and InferenceEngine are not
# designed for concurrent access.  asyncio.Lock is loop-local, which is fine
# for a single-worker uvicorn process (the normal dev/demo setup).
_session_lock = asyncio.Lock()


# ── 4.1  Serial port discovery ────────────────────────────────────────────────

@router.get("/serial/ports")
def list_serial_ports() -> list[dict]:
    """Return all connected serial ports.

    ELM327 USB adapters typically show as 'USB Serial Port' or 'ELM327'.
    Bluetooth adapters appear after pairing — pair first, then refresh.
    Returns an empty list when pyserial's list_ports finds nothing or when
    the call itself fails (e.g. on a machine with no serial subsystem).
    """
    try:
        from serial.tools.list_ports import comports  # pyserial==3.5
        return [
            {"device": p.device, "description": p.description or "Serial port"}
            for p in comports()
        ]
    except Exception:
        return []


# ── 4.2  Live WebSocket ───────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def live_ws(ws: WebSocket) -> None:
    """Live ELM327 inference WebSocket — see module docstring for protocol."""
    await ws.accept()

    if _session_lock.locked():
        await ws.send_json({
            "type": "error",
            "message": "A live session is already active on this server. Disconnect first.",
        })
        await ws.close(code=1008)
        return

    async with _session_lock:
        obd_src = None
        try:
            await _run_session(ws)
        except WebSocketDisconnect:
            log.info("Live WS: client disconnected cleanly.")
        except Exception as exc:
            log.exception("Live WS: unexpected error — %s", exc)
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            if obd_src is not None:          # safety net (normally stopped inside)
                await asyncio.to_thread(obd_src.stop)
            log.info("Live WS: session ended.")


async def _run_session(ws: WebSocket) -> None:
    """Inner session coroutine — separated so the finally in live_ws always runs."""
    # ── Receive connect action (30 s timeout) ─────────────────────────────────
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
        msg = json.loads(raw)
    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "Timed out waiting for connect action."})
        return
    except json.JSONDecodeError as exc:
        await ws.send_json({"type": "error", "message": f"Invalid JSON: {exc}"})
        return

    if msg.get("action") != "connect":
        await ws.send_json({
            "type": "error",
            "message": "First message must be {action:'connect', port, car_id}.",
        })
        return

    port: Optional[str] = msg.get("port") or None
    car_id: Optional[int] = msg.get("car_id")

    # ── Look up car's normalizer path ─────────────────────────────────────────
    normalizer_path: Optional[Path] = None
    if car_id is not None:
        db = next(get_db())
        try:
            car = db.get(Car, int(car_id))
            if car and car.baseline_normalizer_path:
                p = Path(car.baseline_normalizer_path)
                if p.exists():
                    normalizer_path = p
                    log.info("Live WS: using normalizer %s", normalizer_path)
                else:
                    log.info("Live WS: normalizer path in DB not found on disk — using Etios default")
        finally:
            db.close()

    # ── Load InferenceEngine (blocking SHAP init → run in thread) ─────────────
    try:
        engine = await asyncio.to_thread(_load_engine, normalizer_path)
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"Model load failed: {exc}"})
        return

    # ── Connect ELM327 (blocks up to 15 s) ────────────────────────────────────
    from src.live.obd_source import LiveObdSource
    obd_src = LiveObdSource(port=port, sample_hz=1.0)
    log.info("Live WS: connecting to ELM327 on port=%s…", port or "auto")

    connected = await asyncio.to_thread(obd_src.connect)
    if not connected:
        await ws.send_json({
            "type": "error",
            "message": (
                f"Could not connect to ELM327 on {port or 'auto-detected port'}. "
                "Ensure the engine is running, the adapter is plugged in, and "
                "the correct COM port is selected."
            ),
        })
        return

    obd_src.start()
    log.info("Live WS: poll thread started — streaming.")

    session_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = LiveSessionStore(DATA_APP_DIR / "live_sessions" / session_ts)
    last_elapsed = {"s": 0}
    log.info("Live WS: persisting session to %s", store.session_dir)

    # ── Concurrent recv + poll ────────────────────────────────────────────────
    stop_event = asyncio.Event()

    async def _recv() -> None:
        """Listen for client actions (mark_leak, stop) until disconnect."""
        while not stop_event.is_set():
            try:
                raw_msg = await asyncio.wait_for(ws.receive_text(), timeout=0.25)
                try:
                    action_msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue
                action = action_msg.get("action")
                if action == "stop":
                    stop_event.set()
                    break
                elif action == "mark_leak":
                    state_val = action_msg.get("state", "")
                    store.record_mark(state=state_val, elapsed_s=last_elapsed["s"])
                    await ws.send_json({
                        "type": "mark_ack",
                        "state": state_val,
                        "elapsed_s": last_elapsed["s"],
                    })
            except asyncio.TimeoutError:
                pass  # normal — client is just not sending anything
            except (WebSocketDisconnect, RuntimeError):
                stop_event.set()
                break
            except Exception as exc:
                log.debug("Live WS recv: %s", exc)

    async def _poll() -> None:
        """Drain OBD rows, run inference, emit telemetry frames."""
        warned_slow = False
        while not stop_event.is_set():
            row = obd_src.next_row()
            if row is None:
                await asyncio.sleep(0.05)
                continue

            # XGBoost + SHAP are CPU-bound — run in the default thread executor
            # so the event loop stays responsive to incoming actions.
            try:
                state = await asyncio.to_thread(engine.update, row)
            except Exception as exc:
                log.warning("Live WS engine.update: %s", exc)
                await asyncio.sleep(0.1)
                continue

            last_elapsed["s"] = state.elapsed_s
            store.append_row(elapsed_s=state.elapsed_s, row=row)

            poll_hz = obd_src.measured_poll_hz
            degraded = len(obd_src.missing_pids)

            frame: dict = {
                "type": "telemetry",
                "elapsed_s": state.elapsed_s,
                # NaN PIDs (unsupported by this ECU) are serialised as null —
                # JSON disallows NaN; the frontend treats null as "no data".
                "telemetry": _safe_pid_dict(state.latest_row),
                "label": state.classifier_label,
                "confidence": _r4(state.classifier_confidence),
                "severities": {k: _r4(v) for k, v in state.severities.items()},
                "forecasts": {k: _r4(v) for k, v in state.forecasts.items()},
                "anomaly_score": _r4(state.anomaly_score),
                "top_shap": [
                    [str(n), _r4(v)]
                    for n, v in (state.top_features or [])
                    if not _is_nan(v)
                ],
                "degraded_pid_count": degraded,
                "poll_hz": round(poll_hz, 3),
            }

            try:
                await ws.send_json(frame)
            except (RuntimeError, WebSocketDisconnect):
                stop_event.set()
                break

            # Warn once when the adapter is too slow (< 0.3 Hz)
            if not warned_slow and 0 < poll_hz < 0.3:
                warned_slow = True
                try:
                    await ws.send_json({
                        "type": "warning",
                        "message": (
                            f"Slow adapter: {poll_hz:.2f} Hz — "
                            "cheap ELM327 clones can't always sustain 14 PIDs at 1 Hz. "
                            "Classifier accuracy may be reduced."
                        ),
                    })
                except (RuntimeError, WebSocketDisconnect):
                    stop_event.set()
                    break

    try:
        await asyncio.gather(_recv(), _poll())
    finally:
        obd_src.stop()
        store.close()
        log.info("Live WS: session saved — %s", store.session_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_engine(normalizer_path: Optional[Path]):
    """Build InferenceEngine in a thread (SHAP TreeExplainer is expensive)."""
    from src.dashboard.inference import InferenceEngine
    return InferenceEngine(normalizer_override=normalizer_path)


def _is_nan(v) -> bool:
    """True when v is a float NaN (NaN != NaN in IEEE 754)."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _r4(v) -> float:
    """Round to 4 d.p.; return 0.0 on NaN/None."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def _safe_pid_dict(row: dict) -> dict:
    """Serialise a PID row to JSON-safe values.

    Private keys (starting with '__') are stripped.
    NaN floats are converted to null (None) — JSON does not allow NaN.
    """
    out = {}
    for k, v in row.items():
        if k.startswith("__"):
            continue
        try:
            f = float(v)
            out[k] = None if math.isnan(f) else f
        except (TypeError, ValueError):
            out[k] = None
    return out
