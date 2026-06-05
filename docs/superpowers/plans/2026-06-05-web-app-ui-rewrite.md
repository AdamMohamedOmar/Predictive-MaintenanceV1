# Web App UI Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. When building any React screen, invoke the `frontend-design` skill for the visual layer.

**Goal:** Re-imagine the Predictive-Maintenance UI as a small multi-user web app (login → garage → add car → upload CSV or run a live ELM327 session → model verdict), reusing the existing ML core untouched.

**Architecture:** React (Vite + TS) frontend ⇄ FastAPI backend ⇄ existing `src/` ML core, with SQLite persistence. The backend imports `InferenceEngine`, `evaluate_real_fault`, `adapt_torque_csv`, `capture_baseline_from_csv`, `inspect_recording`, and `LiveObdSource` directly — **no ML rewrite**. The existing Streamlit app stays alive until React reaches parity.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, SQLAlchemy 2.x (SQLite), bcrypt, PyJWT, python-multipart, websockets (via uvicorn[standard]); React 18 + Vite + TypeScript; charts via Recharts (static) + uPlot (live strip).

**Design spec:** `docs/superpowers/specs/2026-06-05-web-app-ui-rewrite-design.md` (read it first).

---

## Ground rules (carry through every task)

- **Reuse, don't rewrite.** The 83-feature contract, models, physics, and adapters are frozen. The API only *calls* `src/` functions; it never reimplements them.
- **Streamlit stays.** Do not touch or delete `src/dashboard/` until Phase 5 parity sign-off.
- **Tests per module** (CLAUDE.md): every new backend module gets at least one pytest. Run `pytest -q` after each backend task; keep it green (357 passing today; one pre-existing `test_classifier.py` psutil failure is unrelated — ignore it).
- **Honesty doctrine.** Degraded-PID count and the MAF caveat stay visible in the UI; never tune or hide a result to look good.
- **Commit frequently**, one logical unit per commit, message ending with the `Co-Authored-By` trailer. Keep model `.pkl` and the SQLite `.db` untracked (regenerable / local data).

---

## File structure (created across the plan)

```
src/api/
  __init__.py
  main.py            # FastAPI app factory, CORS, router registration, startup
  config.py          # API paths: DB_URL, DATA_APP_DIR, JWT secret, CORS origins
  db.py              # SQLAlchemy engine/session, Base, init_db(), get_db dep
  models.py          # ORM: User, Car, Recording
  schemas.py         # Pydantic request/response models
  auth.py            # hashing, token issue/verify, current_user dependency
  service.py         # orchestration that calls existing src/ functions (ONLY ML touchpoint)
  routers/
    __init__.py
    auth.py          # /api/auth/signup, /login
    cars.py          # /api/cars CRUD + /baseline
    recordings.py    # /api/cars/{id}/recordings upload, /api/recordings/{id}
    live.py          # /api/serial/ports, /ws/live
tests/api/
  test_auth_api.py
  test_cars_api.py
  test_recordings_api.py
  test_live_api.py
  conftest.py        # FastAPI TestClient + temp SQLite fixtures

web/                 # Vite React app
  index.html
  package.json
  vite.config.ts
  src/
    main.tsx
    api.ts           # typed fetch client + WS helper; mirrors the API contract
    auth.tsx         # token storage + auth context
    theme.ts         # design tokens ported from src/dashboard/theme.py
    pages/
      Login.tsx
      Garage.tsx
      CarPage.tsx     # tabs: Overview / Upload / Live / History
      Results.tsx
    components/
      StatusBanner.tsx  SeverityGrid.tsx  AnomalyPanel.tsx
      PidStrip.tsx      ShapPanel.tsx     LiveSession.tsx

data/app/            # untracked: app.db + users/<uid>/cars/<cid>/ files
```

Server-rendered fallback: `uvicorn` can also serve the built `web/dist` (mount static) so the whole app runs from one process for the demo.

---

# Phase 0 — Prerequisites & skeleton

*Outcome: backend boots, SQLite initializes, one endpoint runs an existing adapted CSV through the model via `curl`; Vite app boots with a placeholder. Streamlit untouched.*

### Task 0.1 — Install & verify Node.js LTS (GATE)

**Files:** none (environment).

- [ ] Install Node LTS (≥ 20) on Windows: `winget install OpenJS.NodeJS.LTS` (or the official MSI from nodejs.org).
- [ ] Open a **new** terminal so PATH refreshes.
- [ ] Verify: `node --version` (expect v20.x or v22.x) and `npm --version`.

**Acceptance:** both commands print versions in a fresh PowerShell AND git-bash shell.
**DoD:** Node toolchain available. **If this blocks, stop and tell Adam — the Streamlit app still demos the science meanwhile.**

### Task 0.2 — Pin & install backend dependencies

**Files:** Modify `requirements.txt`.

- [ ] Append a "Web app (FastAPI backend)" block. Suggested pins (verify a Windows wheel exists on install; loosen to a minor range if a pin lacks a wheel, per the file's own rule):

```
# Web app (FastAPI backend)
fastapi==0.115.6
uvicorn[standard]==0.32.1     # bundles websockets + httptools for /ws/live
sqlalchemy==2.0.36
python-multipart==0.0.18      # multipart CSV upload
bcrypt==4.2.1                 # password hashing (mock auth)
pyjwt==2.10.1                 # session tokens
```

- [ ] Install: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
- [ ] Verify import: `.venv/Scripts/python.exe -c "import fastapi, uvicorn, sqlalchemy, jwt, bcrypt, multipart; print('backend deps OK')"`

**Acceptance:** import line prints `backend deps OK`.
**DoD:** deps installed and pinned. Commit `requirements.txt`.

### Task 0.3 — API config + SQLite engine + `init_db`

**Files:** Create `src/api/__init__.py`, `src/api/config.py`, `src/api/db.py`. Test `tests/api/test_db.py`.

- [ ] `src/api/config.py`: define `DATA_APP_DIR = _REPO_ROOT/"data"/"app"`, `DB_URL = f"sqlite:///{DATA_APP_DIR/'app.db'}"`, `JWT_SECRET` (read env `PM_JWT_SECRET`, default a dev constant), `JWT_ALGO="HS256"`, `CORS_ORIGINS=["http://localhost:5173"]`.
- [ ] `src/api/db.py`: SQLAlchemy 2.x `create_engine(DB_URL, connect_args={"check_same_thread": False})`, `SessionLocal`, `Base = declarative_base()`, `init_db()` that `Base.metadata.create_all`, and a `get_db()` generator dependency.
- [ ] Test: `init_db()` against a temp DB file creates the tables; `get_db()` yields a usable session.

**Acceptance:** `pytest tests/api/test_db.py -q` passes.
**DoD:** SQLite bootstraps. Commit.

### Task 0.4 — ORM models & Pydantic schemas

**Files:** Create `src/api/models.py`, `src/api/schemas.py`. Test `tests/api/test_models.py`.

- [ ] `models.py` — exactly the schema in spec §5:

```python
class User(Base):
    __tablename__ = "users"
    id = mapped_column(Integer, primary_key=True)
    username = mapped_column(String, unique=True, nullable=False)
    password_hash = mapped_column(String, nullable=False)
    created_at = mapped_column(String)

class Car(Base):
    __tablename__ = "cars"
    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    make = mapped_column(String); model = mapped_column(String)
    year = mapped_column(Integer)
    engine_metering = mapped_column(String, default="unknown")  # speed_density|maf|unknown
    baseline_normalizer_path = mapped_column(String, nullable=True)
    created_at = mapped_column(String)

class Recording(Base):
    __tablename__ = "recordings"
    id = mapped_column(Integer, primary_key=True)
    car_id = mapped_column(Integer, ForeignKey("cars.id"), nullable=False)
    kind = mapped_column(String)               # csv|live
    original_filename = mapped_column(String)
    adapted_csv_path = mapped_column(String, nullable=True)
    result_json_path = mapped_column(String, nullable=True)
    label_summary = mapped_column(String, nullable=True)   # JSON string
    anomaly_mean = mapped_column(Float, nullable=True)
    recall = mapped_column(Float, nullable=True)
    fault_from_s = mapped_column(Integer, nullable=True)
    fault_to_s = mapped_column(Integer, nullable=True)
    created_at = mapped_column(String)
```

- [ ] `schemas.py` — Pydantic v2 models: `UserCreate`, `UserOut`, `TokenOut`, `CarCreate`, `CarOut`, `RecordingOut`. Use `model_config = ConfigDict(from_attributes=True)` on the `*Out` models.
- [ ] Test: create a User+Car+Recording in a temp session, query back, and round-trip through the `*Out` schemas.

**Acceptance:** `pytest tests/api/test_models.py -q` passes.
**DoD:** data layer complete. Commit.

### Task 0.5 — `service.py`: the single ML touchpoint (smoke endpoint)

**Files:** Create `src/api/service.py`, `src/api/main.py`, `tests/api/conftest.py`, `tests/api/test_smoke_api.py`.

- [ ] `service.py` — one function for now:

```python
def score_adapted_csv(adapted_csv: Path, normalizer_path: Optional[Path]) -> dict:
    """Score an already-adapted clean-column CSV. Wraps evaluate_real_fault."""
    from src.dashboard.inference import InferenceEngine
    from src.eval.real_fault_eval import evaluate_real_fault
    engine = InferenceEngine(normalizer_override=normalizer_path) if normalizer_path else None
    kwargs = {"engine": engine} if engine else {}
    return evaluate_real_fault(adapted_csv, **kwargs)
```

- [ ] `main.py` — `create_app()` factory: FastAPI instance, CORS middleware (`CORS_ORIGINS`), `@app.on_event("startup")` → `init_db()`, a `GET /api/health` returning `{"status":"ok"}`, and a temporary `POST /api/_smoke` that accepts a server-side path and returns `score_adapted_csv(...)["summary"]`. Expose `app = create_app()`.
- [ ] `conftest.py` — `TestClient(create_app())` fixture with a temp `DATA_APP_DIR`/DB via monkeypatch.
- [ ] Test: `GET /api/health` → 200 `{"status":"ok"}`; `POST /api/_smoke` with `data/real_faults/ahmed/ahmed_drive_20260602.csv` and the ahmed normalizer → returns a `label_counts` dict with > 0 windows (skip if models absent).

**Acceptance:** `pytest tests/api/test_smoke_api.py -q` passes; `uvicorn src.api.main:app --port 8000` then `curl localhost:8000/api/health` → ok.
**DoD:** backend serves the ML core end-to-end. Remove `/api/_smoke` in Task 2.x once the real upload route exists. Commit.

### Task 0.6 — Vite React skeleton boots

**Files:** Create `web/` via Vite.

- [ ] From the repo root: `npm create vite@latest web -- --template react-ts` (creates `web/`).
- [ ] `cd web && npm install`.
- [ ] Replace `web/src/App.tsx` body with a placeholder "Predictive Maintenance — coming soon" using the dark background from spec theme.
- [ ] Add `web/.gitignore` entries for `node_modules`, `dist`.
- [ ] Verify: `npm run dev` serves on `http://localhost:5173`.

**Acceptance:** browser shows the placeholder at :5173 with no console errors.
**DoD:** frontend toolchain live. Commit `web/` (excluding `node_modules`).

---

# Phase 1 — Auth (mock) + Garage

*Outcome: a user can sign up, log in, add cars, see their garage. First real screens.*

### Task 1.1 — Password hashing + JWT helpers

**Files:** Create `src/api/auth.py`. Test `tests/api/test_auth_unit.py`.

- [ ] `hash_password(pw)->str` (bcrypt), `verify_password(pw, hash)->bool`, `make_token(user_id)->str` (PyJWT, `exp` ~7 days), `decode_token(tok)->user_id`.
- [ ] `current_user` FastAPI dependency: read `Authorization: Bearer <tok>`, decode, load `User` from DB, 401 on failure.
- [ ] Test: hash≠plaintext; verify round-trips; token round-trips to the same user_id; tampered token raises.

**Acceptance:** `pytest tests/api/test_auth_unit.py -q` passes.
**DoD:** auth primitives ready. Commit.

### Task 1.2 — Auth router (signup/login)

**Files:** Create `src/api/routers/__init__.py`, `src/api/routers/auth.py`; register in `main.py`. Test `tests/api/test_auth_api.py`.

- [ ] `POST /api/auth/signup` (`UserCreate`) → unique-username check (409 if taken) → store hashed → return `TokenOut`.
- [ ] `POST /api/auth/login` → verify → `TokenOut` (401 on bad creds).
- [ ] Test: signup returns a token; duplicate signup → 409; login with right/wrong password → 200/401; the token authorizes `GET /api/cars` (empty list).

**Acceptance:** `pytest tests/api/test_auth_api.py -q` passes.
**DoD:** mock auth works end-to-end. Commit.

### Task 1.3 — Cars router (garage CRUD)

**Files:** Create `src/api/routers/cars.py`; register. Test `tests/api/test_cars_api.py`.

- [ ] `GET /api/cars` (current user only), `POST /api/cars` (`CarCreate`), `GET /api/cars/{id}` (404 if not owner), `DELETE /api/cars/{id}`.
- [ ] On create, make `data/app/users/<uid>/cars/<cid>/` directories.
- [ ] Test: a user sees only their own cars; create→list→get→delete cycle; cross-user access → 404.

**Acceptance:** `pytest tests/api/test_cars_api.py -q` passes.
**DoD:** garage backend complete. Commit.

### Task 1.4 — Frontend: API client + auth context

**Files:** Create `web/src/api.ts`, `web/src/auth.tsx`, `web/src/theme.ts`.

- [ ] `theme.ts`: port the palette/fonts from `src/dashboard/theme.py` (BG_BASE, BG_SURFACE, ACCENT_OK/WARN/ALERT/DATA, TEXT_*, FONT_*) as TS constants — the design-token source of truth.
- [ ] `api.ts`: typed `fetchJson` wrapper that injects `Authorization`, plus `signup`, `login`, `listCars`, `createCar`, `getCar`, `deleteCar`, `uploadRecording`, `uploadBaseline`, `getRecording`, `listSerialPorts`, and `openLiveSocket` (returns a `WebSocket`). Base URL from `import.meta.env.VITE_API_URL ?? "http://localhost:8000"`.
- [ ] `auth.tsx`: React context storing the token in `localStorage`; `useAuth()` exposes `{token, login, signup, logout}`.

**Acceptance:** `npm run build` compiles with no type errors.
**DoD:** frontend talks to the API contract. Commit.

### Task 1.5 — Frontend: Login + Garage screens

**Files:** Create `web/src/pages/Login.tsx`, `web/src/pages/Garage.tsx`; wire routing in `main.tsx` (add `react-router-dom`).

- [ ] Invoke the **`frontend-design`** skill for these two screens — distinctive dark automotive-cockpit aesthetic, not generic. Inputs to honor: theme tokens from `theme.ts`; Login has a Sign-up/Sign-in toggle; Garage is a responsive card grid with an "Add car" action opening a modal (make/model/year + engine-metering select with a one-line MAF-vs-speed-density helper).
- [ ] Routing: `/login` (public) → `/garage` (auth-gated, redirect to `/login` if no token).
- [ ] Behavior: signup/login call `api.ts`, store token, navigate to `/garage`; garage lists cars, add-car posts and refreshes, delete confirms first.

**Acceptance:** with `uvicorn` + `npm run dev` both up: sign up → land on an empty garage → add a car → it appears → reload keeps it (persisted) → logout returns to login.
**DoD:** auth + garage demoable end-to-end. Commit.

---

# Phase 2 — CSV upload → score → Results (first full E2E demo)

*Outcome: drop a phone/Torque CSV on a car → see the verdict. The headline feature.*

### Task 2.1 — Service: full upload pipeline

**Files:** Modify `src/api/service.py`. Test `tests/api/test_service_pipeline.py`.

- [ ] Add `process_upload(raw_csv, out_dir, normalizer_path, is_baseline) -> dict`:
  1. `inspect_recording(raw_csv)` → capture the metering/coverage/warnings report.
  2. `adapt_torque_csv(raw_csv)` → write `<out_dir>/adapted.csv`. (If the file is already clean-column, `inspect` flags `is_clean_column_format`; in that case skip adapt and copy.)
  3. If `is_baseline`: `capture_baseline_from_csv(adapted, vehicle_name, out_dir/"normalizer.pkl")` → return `{mode:"baseline", normalizer_path, inspect}`.
  4. Else: `score_adapted_csv(adapted, normalizer_path)` → write `result.json`; return `{mode:"score", result, inspect, adapted_csv}`.
- [ ] Test (skip if models absent): run on the ahmed raw export → returns `mode:"score"` with > 0 windows and an `inspect` report carrying a `metering_type`.

**Acceptance:** `pytest tests/api/test_service_pipeline.py -q` passes.
**DoD:** the pipeline is one callable. Commit.

### Task 2.2 — Recordings router (upload + fetch)

**Files:** Create `src/api/routers/recordings.py`; register; delete the temporary `/api/_smoke`. Test `tests/api/test_recordings_api.py`.

- [ ] `POST /api/cars/{id}/recordings` (multipart `file`, form `is_baseline: bool`, optional `fault_from_s`, `fault_to_s`): save raw under the car dir → `process_upload(...)` with `car.baseline_normalizer_path` → if baseline, update `car.baseline_normalizer_path`; else insert a `Recording` row (label_summary, anomaly_mean, recall if fault interval given) → return `RecordingOut` (or baseline confirmation).
- [ ] `GET /api/recordings/{id}` → owner-checked; returns the stored `result.json` plus the row metadata for the Results view.
- [ ] Compute `recall` when `fault_from_s/to_s` provided (windows in interval labelled non-healthy / total in interval).
- [ ] Test: upload the ahmed CSV to a car → 200 with a label summary; baseline upload sets the car's normalizer; `GET /api/recordings/{id}` returns the full window list.

**Acceptance:** `pytest tests/api/test_recordings_api.py -q` passes.
**DoD:** upload→score persists and is retrievable. Commit.

### Task 2.3 — Frontend: Car page shell + Upload tab

**Files:** Create `web/src/pages/CarPage.tsx`; route `/cars/:id`.

- [ ] Invoke **`frontend-design`** for the Car page tab shell (Overview / Add recording / Live / History) and the Upload tab.
- [ ] Upload tab: drag-drop a CSV; a "this is a healthy baseline drive" toggle; on drop, POST to the recordings endpoint with a loading state; render the returned `inspect` report **before/above** the verdict (metering verdict + PID coverage + any warnings — honesty doctrine), then link to Results.
- [ ] Overview tab: car specs + baseline status ("No baseline yet — upload a healthy drive" vs "Baseline captured ✓").

**Acceptance:** upload the ahmed CSV through the UI → see the metering warning (MAF caveat) and a label distribution → navigate to Results.
**DoD:** upload UX works. Commit.

### Task 2.4 — Frontend: Results view (re-imagined panels)

**Files:** Create `web/src/pages/Results.tsx` + `components/StatusBanner.tsx`, `SeverityGrid.tsx`, `AnomalyPanel.tsx`, `PidStrip.tsx`, `ShapPanel.tsx`.

- [ ] Invoke **`frontend-design`** for the Results layout and each panel — re-imagine the current Streamlit panels (status banner, severity + 60 s forecast grid, anomaly score, PID timeline via Recharts, top-SHAP bars, diagnostic steps). Data contract = the `GET /api/recordings/{id}` payload (`windows[]` with `label, confidence, severities, forecasts, anomaly_score, all_probs` + summary).
- [ ] Add `recharts` to `web`. PID timeline plots from the stored window series; a window scrubber selects which window drives the SHAP/severity panels (per-window scrub is a stretch — a single summary view is acceptable for Phase 2).
- [ ] Keep honest caveats visible (degraded PIDs, MAF note) at the top of Results.

**Acceptance:** Results renders a real ahmed recording: label mix, anomaly mean, PID charts, top SHAP — no console errors.
**DoD:** **first full end-to-end demo** (signup → add car → upload → verdict). Commit. Tag this as the parity-candidate checkpoint.

---

# Phase 3 — Baseline capture polish

*Outcome: a car "remembers" its baseline; scoring auto-uses it.*

### Task 3.1 — Baseline endpoint surfaced in UI + guard messaging

**Files:** Modify `recordings.py` (baseline path already exists from 2.2); `CarPage.tsx` Overview/Upload.

- [ ] Ensure the baseline upload returns the guard outcome cleanly: on `ValueError` from `capture_baseline_from_csv` (idle/cold/too-short), return HTTP 422 with the guard message; the UI shows it inline ("baseline must be a real warm drive, mean speed > 15 km/h…").
- [ ] Overview reflects baseline status and the captured vehicle metadata (n_windows, date).
- [ ] After a baseline is set, the Upload tab notes "scoring will use this car's baseline."
- [ ] Test (`tests/api/test_recordings_api.py`): a cold/idle CSV → 422 with guard text; a warm/moving CSV → sets `baseline_normalizer_path`.

**Acceptance:** test passes; UI shows guard failure gracefully.
**DoD:** per-car baseline loop closed. Commit.

---

# Phase 4 — Live session (WebSocket + ELM327) — ADAM'S PRIORITY

*Outcome: connect ELM327 to the laptop, stream live predictions reliably.*

### Task 4.1 — Serial ports endpoint

**Files:** Create `src/api/routers/live.py`; register. Test `tests/api/test_live_api.py`.

- [ ] `GET /api/serial/ports` → `[{device, description}]` from `serial.tools.list_ports.comports()`.
- [ ] Test: endpoint returns a list (possibly empty) with 200.

**Acceptance:** `pytest tests/api/test_live_api.py -q` passes.
**DoD:** UI can enumerate adapters. Commit.

### Task 4.2 — Live WebSocket loop (single-session lock)

**Files:** Modify `src/api/routers/live.py`. Test `tests/api/test_live_ws.py`.

- [ ] Module-level `asyncio.Lock` / boolean enforcing **one live session at a time** (single physical ELM327). A second connect → send `{"type":"error","message":"A live session is already running"}` and close.
- [ ] `WS /ws/live` protocol:
  - client → `{"action":"connect","port":"COM3","car_id":N}`; server builds `LiveObdSource(port)`, `connect()` (≤15 s; on fail send error+close), `start()`; builds `InferenceEngine(normalizer_override=car.baseline_normalizer_path)`.
  - loop: `await asyncio.sleep(0.1)`; `row = src.next_row()`; if row → `state = await asyncio.to_thread(engine.update, row)` (keeps the event loop free during SHAP) → `await ws.send_json(frame)` where frame =

```json
{"type":"telemetry","elapsed_s":N,"telemetry":{...14 PIDs...},
 "label":"...","confidence":0.0,"severities":{...},"forecasts":{...},
 "anomaly_score":0.0,"top_shap":[["FEATURE",0.0],...],
 "degraded_pid_count":0,"poll_hz":1.0}
```

  - `{"action":"mark_leak","state":"start"|"stop"}` → record interval seconds, echo `{"type":"mark_ack",...}`.
  - `{"action":"stop"}` or WS disconnect → `src.stop()`, release the lock, optionally persist a `Recording(kind="live")` with the marked interval + a recall.
  - If `poll_hz < 0.3`, send `{"type":"warning","message":"Adapter poll rate below 0.3 Hz…"}` (reuse the existing slow-adapter guard text).
- [ ] Test with a **fake source** (monkeypatch `LiveObdSource` to a stub that yields synthetic rows): connect → receive ≥ 3 telemetry frames with all contract keys → mark_leak ack → stop releases the lock (a second session can then connect).

**Acceptance:** `pytest tests/api/test_live_ws.py -q` passes (no hardware needed — stubbed source).
**DoD:** live backend correct and event-loop-safe. Commit.

### Task 4.3 — Frontend: Live session tab

**Files:** Create `web/src/components/LiveSession.tsx`; wire into `CarPage` Live tab. Add `uplot` for the live strip.

- [ ] Invoke **`frontend-design`** for the live cockpit. Controls: COM-port `<select>` (from `/api/serial/ports`), Connect/Disconnect, "Mark leak start/stop". Live panels reuse `StatusBanner/SeverityGrid/AnomalyPanel/ShapPanel` fed by WS frames; PID strip uses uPlot for smooth 1 Hz streaming over a rolling window.
- [ ] Connection UX: "Connecting…", "Waiting for ECU…", "Live", and the busy/slow-adapter warnings as banners. Disconnect cleanly closes the socket.

**Acceptance:** **Manual hardware check (Adam):** plug the ELM327 into the laptop, pick the COM port, Connect → live telemetry + predictions update ~1 Hz; "Mark leak" toggles; Disconnect stops cleanly. If no adapter is on hand, verify against the Streamlit live path or a stubbed WS in dev.
**DoD:** the direct ELM327 → laptop live path works end-to-end in the new UI. Commit.

---

# Phase 5 — Re-imagine polish, upgrades, parity sign-off

*Outcome: the app looks intentional; the in-scope upgrades land; Streamlit can retire.*

### Task 5.1 — History + baseline-vs-fault comparison

**Files:** `CarPage.tsx` History tab; maybe `GET /api/cars/{id}/recordings` (list).
- [ ] List past recordings (label mix, anomaly mean, recall, date). Open → Results. Select a fault run + the baseline run → side-by-side label distributions and anomaly means.
**Acceptance:** two uploads on one car show in History and compare.
**DoD:** comparison works. Commit.

### Task 5.2 — frontend-design coherence pass

- [ ] Invoke **`frontend-design`** once more across all screens for visual coherence (spacing, motion, empty/error/loading states, responsive behavior). No new features — polish only.
**Acceptance:** a walkthrough of every screen looks intentional and consistent.
**DoD:** the "re-imagine" goal is visibly met. Commit.

### Task 5.3 — Honest-caveat audit + README

- [ ] Verify degraded-PID count and the MAF caveat appear wherever a result is shown (upload, results, live). Add a `web/README.md` and a `docs/` note: how to run (`uvicorn src.api.main:app` + `npm run dev`, or build + single-process serve).
**Acceptance:** caveats present in all three surfaces; README lets a fresh user start the app.
**DoD:** documented + honest. Commit.

### Task 5.4 — Parity sign-off (then, only then, retire Streamlit)

- [ ] Confirm the React app covers everything the Streamlit dashboard did (CSV scoring, live session, all panels). Get Adam's explicit OK.
- [ ] Only after sign-off: optionally move `src/dashboard/` to `legacy/` (do not delete history). Keep until Adam confirms.
**Acceptance:** Adam signs off on parity.
**DoD:** project re-imagined; fallback retired deliberately, not accidentally.

---

## Self-review (spec coverage)

- Login/signup → Phase 1 (1.1–1.5). ✓
- Add car model + data → Phase 1 (1.3–1.5). ✓
- Add CSV from your car → model → Phase 2 (2.1–2.4). ✓
- Live session, direct ELM327 → Phase 4 (4.1–4.3), priority-flagged. ✓
- SQLite storage → Phase 0 (0.3–0.4), per-car baseline → Phases 2–3. ✓
- Mock auth → Phase 1 (1.1–1.2). ✓
- WebSocket live transport → Phase 4 (4.2). ✓
- Evolve dark theme → `theme.ts` (1.4) + frontend-design passes. ✓
- Proposed upgrades: metering check on upload (2.2/2.3), per-car baseline (2–3), live mark-leak recall (4.2/4.3), comparison (5.1). Stretch (PDF, per-window SHAP scrub) explicitly optional. ✓
- Node/dep prerequisites → Phase 0 gates (0.1–0.2). ✓
- Streamlit-stays-alive de-risk → Ground rules + 5.4. ✓

## Scope guard (do NOT)

- Do not change the 83-feature contract, models, physics, or adapters.
- Do not build a phone→laptop relay (CSV upload covers "recorded on phone").
- Do not add production auth (OAuth/email-verify/reset/RBAC).
- Do not delete the Streamlit app before 5.4 sign-off.
- Keep `.pkl` artefacts and `data/app/app.db` untracked.
```
