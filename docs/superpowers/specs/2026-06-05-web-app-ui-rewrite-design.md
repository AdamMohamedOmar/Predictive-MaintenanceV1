# Design Spec — Web App UI Rewrite (React + FastAPI + SQLite)

**Date:** 2026-06-05
**Author:** brainstormed with Adam; to be implemented by Sonnet
**Status:** approved design → feeds the implementation plan

---

## 1. Goal

Re-imagine the Predictive-Maintenance UI as a small multi-user web app:

1. **Login / Signup** (mock auth — real enough to demo, not production security).
2. **Garage** — add your car (make / model / year / engine type) and keep it.
3. **Add a recording (CSV)** exported from your car (phone/Torque app) → the app
   runs it through the model and shows the verdict.
4. **Live session** — connect an **ELM327 directly to the laptop** and stream live
   predictions. This path must work *reliably* (Adam's explicit priority).
5. Surface honest caveats (degraded PIDs, MAF-vs-speed-density) in the UI.

The existing Streamlit dashboard stays alive as a fallback until the React app
reaches parity. **We never have a broken demo.**

---

## 2. Verified technical facts (checked 2026-06-05, not assumed)

| Fact | Evidence | Consequence |
|---|---|---|
| ML core is **Streamlit-free** | `import` of `InferenceEngine`, `evaluate_real_fault`, `LiveObdSource`, `adapt_torque_csv`, `capture_baseline_from_csv` → `streamlit in sys.modules == False` | FastAPI imports `src/` directly; **no ML/inference rewrite** |
| `LiveObdSource` is **already headless + threaded** | `src/live/obd_source.py` docstring + lifecycle: background poll thread, size-1 queue (latest-row-wins), `connect()/start()/next_row()/stop()/reset()`, **auto-reconnect every 2 s** | A FastAPI background loop pumps `next_row()` into a WebSocket; auto-reconnect serves the "effective live data" requirement |
| Backend deps **not installed** | `pip list` shows no fastapi/uvicorn/bcrypt/etc. | Phase 0 adds + pins them in `requirements.txt` |
| **Node.js not installed** | not on git-bash PATH, not on PowerShell PATH, absent from `Program Files\nodejs`, nvm, `AppData\npm` | Phase 0 **gate**: install Node LTS before any frontend work |

These four facts are the load-bearing assumptions of the whole design. All hold.

---

## 3. Decisions (locked)

- **Stack:** React + Vite + TypeScript (frontend) · FastAPI (backend) · SQLite (storage).
- **Auth:** mock — bcrypt-hashed passwords in SQLite, signed session token. No email
  verification, OAuth, or password reset.
- **Live transport:** **WebSocket** (bidirectional: client sends connect/port/start/stop/
  mark-leak; server streams telemetry + predictions).
- **Live scope:** **direct ELM327 → laptop only** (USB/Bluetooth COM port). Phone→laptop
  real-time relay is **out of scope**; "recorded on phone" is covered by the CSV upload path.
- **Theme:** evolve the existing **dark automotive-cockpit** look (reuse `theme.py` palette
  as the design-token source), polished via the `frontend-design` skill.
- **No ML changes:** the 83-feature contract, models, physics, and adapters are untouched.

---

## 4. Architecture

```
┌──────────────────────┐   HTTP/JSON + WebSocket    ┌───────────────────────────┐
│  React (Vite + TS)   │ ◄────────────────────────► │     FastAPI backend       │
│  Login / Signup      │                            │  routers:                 │
│  Garage              │     multipart CSV upload   │   /api/auth               │
│  Car page (tabs)     │ ─────────────────────────► │   /api/cars               │
│  Results view        │                            │   /api/cars/{id}/recordings│
│  Live session        │ ◄═══ WebSocket /ws/live ══► │   /api/serial/ports       │
└──────────────────────┘                            │   /ws/live (telemetry)    │
                                                     └──────────┬────────────────┘
        ┌────────────────┐                                      │ imports (no rewrite)
        │ SQLite app.db  │ ◄──────── SQLAlchemy ────────────────┤
        └────────────────┘                            ┌─────────▼─────────────────┐
        data/app/users/<uid>/cars/<cid>/...           │ existing src/ core:       │
                                                       │  InferenceEngine, SHAP,   │
                                                       │  evaluate_real_fault,     │
                                                       │  adapt_torque_csv,        │
                                                       │  capture_baseline_from_csv│
                                                       │  inspect_recording,       │
                                                       │  LiveObdSource            │
                                                       └─────────┬─────────────────┘
                                              ELM327 ────────────┘ (COM port, pyserial)
```

**Why FastAPI improves the live path:** Streamlit re-runs the whole script every tick — a
poor host for a serial port. FastAPI is a long-lived process: one owner of the ELM327, a
background reader, predictions pushed over WS. This makes the direct-laptop path the
*best-supported* one.

### 4.1 Module boundaries (each unit, one purpose)

- `src/api/main.py` — FastAPI app factory, CORS, router registration, startup/shutdown.
- `src/api/db.py` — SQLAlchemy engine/session, `Base`, `init_db()`.
- `src/api/models.py` — ORM models (User, Car, Recording).
- `src/api/schemas.py` — Pydantic request/response models.
- `src/api/auth.py` — signup/login, password hashing, token issue/verify, `current_user` dep.
- `src/api/routers/cars.py` — garage CRUD.
- `src/api/routers/recordings.py` — CSV upload → adapt → (baseline | score) → store.
- `src/api/routers/live.py` — `/api/serial/ports` + `/ws/live` WebSocket loop.
- `src/api/service.py` — thin orchestration that calls existing `src/` functions (the only
  place the API touches the ML core; keeps routers dumb and testable).
- `web/` — React app (Vite). Pages under `web/src/pages/`, shared API client in
  `web/src/api.ts`, components under `web/src/components/`.

`src/dashboard/` (Streamlit) is **left intact** — fallback, deleted only after parity.

---

## 5. Data model (SQLite via SQLAlchemy)

```
users
  id INTEGER PK
  username TEXT UNIQUE NOT NULL
  password_hash TEXT NOT NULL
  created_at TEXT

cars
  id INTEGER PK
  user_id INTEGER FK->users.id
  make TEXT, model TEXT, year INTEGER
  engine_metering TEXT            -- 'speed_density' | 'maf' | 'unknown'
  baseline_normalizer_path TEXT   -- nullable; set after a healthy baseline upload
  created_at TEXT

recordings
  id INTEGER PK
  car_id INTEGER FK->cars.id
  kind TEXT                       -- 'csv' | 'live'
  original_filename TEXT
  adapted_csv_path TEXT
  result_json_path TEXT
  label_summary TEXT              -- JSON: {label: count}
  anomaly_mean REAL
  recall REAL                     -- nullable; set when a fault interval is marked
  fault_from_s INTEGER, fault_to_s INTEGER   -- nullable
  created_at TEXT
```

**Per-car baseline memory:** a car stores its own `baseline_normalizer_path`. Scoring a
recording uses that path as `normalizer_override`, so the cross-vehicle (Skoda/MAF) problem
is handled by the UX, not by hand. Files live under `data/app/users/<uid>/cars/<cid>/`.

---

## 6. API surface (REST + one WebSocket)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/signup` | create user (bcrypt) → token |
| POST | `/api/auth/login` | verify → token |
| GET | `/api/cars` | list current user's cars |
| POST | `/api/cars` | add a car |
| GET | `/api/cars/{id}` | car detail + recordings |
| DELETE | `/api/cars/{id}` | remove a car |
| POST | `/api/cars/{id}/baseline` | upload healthy drive → `capture_baseline_from_csv` → set normalizer |
| POST | `/api/cars/{id}/recordings` | upload CSV → `inspect`+`adapt`+`evaluate_real_fault` → store |
| GET | `/api/recordings/{id}` | full result JSON for the results view |
| GET | `/api/serial/ports` | list COM ports (`serial.tools.list_ports`) |
| WS | `/ws/live` | live session: client {connect, port, car_id, start, stop, mark_leak}; server streams frames |

**Upload pipeline (server side):** save raw → `inspect_recording` (metering + PID coverage +
warnings) → `adapt_torque_csv` (robust time parse + variance-aware columns) → if "is baseline"
then `capture_baseline_from_csv` else `evaluate_real_fault(adapted, engine=InferenceEngine(
normalizer_override=car.baseline_normalizer_path))` → persist result JSON + summary row.

---

## 7. Live session design (the priority path)

WebSocket handler in `routers/live.py`:

1. Client connects `/ws/live`, sends `{action:"connect", port:"COM3", car_id:N}`.
2. Server enforces a **global single-session lock** (one physical ELM327 → one live session
   at a time; reject a second with a clear message).
3. Server builds `LiveObdSource(port)`, `connect()` (≤15 s), `start()` (spawns its poll thread).
   Engine = `InferenceEngine(normalizer_override=car.baseline_normalizer_path)`.
4. Async loop: `await asyncio.sleep(~0.1)`; `row = src.next_row()`; if a row arrived, run
   `engine.update(row)` **in a worker thread** (`asyncio.to_thread`) so SHAP doesn't block the
   event loop; `await ws.send_json(frame)`.
5. Frame = `{elapsed_s, telemetry:{14 PIDs}, label, confidence, severities, forecasts,
   anomaly_score, top_shap:[...], degraded_pid_count, poll_hz}`.
6. `{action:"mark_leak", state:"start"|"stop"}` records interval bounds → live recall.
7. `{action:"stop"}` or disconnect → `src.stop()`, release the lock.
8. Surface the existing slow-adapter guard: if `poll_hz < 0.3`, send a warning frame.

The client renders frames into the re-imagined panels (status banner, severity grid,
anomaly, PID strip, SHAP), reusing the visual language of the current dashboard.

---

## 8. Screen flow

1. **Login / Signup** — dark, branded; tab toggle.
2. **Garage** — card grid of cars + "Add car" modal (make/model/year/engine metering, with a
   one-line explainer of MAF vs speed-density). Empty state guides first add.
3. **Car page** — tabs:
   - **Overview** — specs, baseline status ("No baseline yet — upload a healthy drive"),
     recent recordings.
   - **Add recording** — drag-drop CSV; "this is a healthy baseline" toggle; shows the
     `inspect_recording` report (metering verdict, PID coverage, warnings) before scoring.
   - **Live session** — COM-port picker → Connect → live panels + "Mark leak start/stop".
   - **History** — past recordings; open any → Results; compare a run vs the baseline run.
4. **Results view** — status banner, severity + 60 s forecast grid, anomaly score, PID
   timeline, top SHAP, diagnostic steps, honest caveats. Per-window scrub (stretch).

---

## 9. Proposed upgrades

**In-scope (build):**
- Auto metering-type check on every upload (reuse `inspect_recording`) → inline MAF caveat.
- Per-car baseline memory (§5) — fixes the #1 cross-vehicle footgun inside the UX.
- Live "mark leak" → live recall (turns the test-day protocol into one button).
- Baseline-vs-fault comparison in History.

**Stretch (flag optional, only if time remains):**
- Printable PDF report per recording.
- Per-window SHAP scrubber on the results timeline.

**Honesty carryover:** degraded-PID count and the MAF caveat stay visible, never hidden —
consistent with the project's anti-overclaim doctrine.

---

## 10. Build order (phased; Streamlit stays alive throughout)

- **Phase 0 — Prerequisites & skeleton.** Install Node LTS (gate). Add+pin backend deps.
  FastAPI app + SQLite `init_db` + reuse-core smoke endpoint (`POST /api/recordings` works via
  `curl` on an existing adapted CSV). Vite app boots with a placeholder page.
- **Phase 1 — Auth + Garage.** Signup/login (mock), token, cars CRUD, garage UI.
- **Phase 2 — CSV upload → score → Results.** The full upload pipeline + results view. This is
  the first end-to-end demo.
- **Phase 3 — Baseline capture.** Upload-healthy-drive → per-car normalizer; scoring uses it.
- **Phase 4 — Live session.** WebSocket + `LiveObdSource` reader + live panels + mark-leak.
  Adam's priority; verify against a real/simulated ELM327.
- **Phase 5 — Re-imagine polish + upgrades.** `frontend-design` pass; comparison view; honest
  caveats; stretch items if time remains.

Each phase ends green (`pytest -q` for backend; app boots for frontend) and is independently
demoable. If Phase 0 Node setup blocks, the Streamlit app still demos the science.

---

## 11. Out of scope (scope guard for Sonnet)

- No changes to the 83-feature contract, models, physics, or adapters.
- No phone→laptop real-time relay (CSV upload covers "recorded on phone").
- No production auth (OAuth, email verify, password reset, RBAC).
- No cloud/hosting; runs locally (`uvicorn` + `vite dev`, or `vite build` served by FastAPI).
- No new ML faults or PID expansion.
- Do not delete the Streamlit dashboard until React reaches parity.

---

## 12. Risks & mitigations

- **Deadline (June 15) vs. full rewrite** → ML reused untouched; phased vertical slices;
  Streamlit fallback. Biggest scope is presentation only.
- **Node toolchain on Windows** → Phase 0 gate with an explicit install + verify step.
- **New backend deps must have Windows wheels** → pin known-good versions (fastapi, uvicorn,
  sqlalchemy, bcrypt/passlib, python-multipart, websockets) and verify install before building.
- **Single ELM327** → global single-live-session lock; clear "busy" message.
- **SHAP latency blocking the WS event loop** → run `engine.update` in `asyncio.to_thread`.
