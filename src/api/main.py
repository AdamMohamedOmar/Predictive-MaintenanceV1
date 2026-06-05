"""FastAPI application factory for the Predictive Maintenance web app.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when the module is imported directly
# (e.g. `uvicorn src.api.main:app` from the repo root).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import CORS_ORIGINS
from src.api.db import init_db


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: create SQLite tables
        init_db()
        yield
        # Shutdown: nothing to clean up (connections pooled by SQLAlchemy)

    app = FastAPI(
        title="Predictive Maintenance API",
        description="Backend for the PM web app — wraps the XGBoost inference core.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    # ── Routers (registered incrementally as phases complete) ─────────────────
    # Phase 1:
    from src.api.routers import auth as auth_router
    app.include_router(auth_router.router, prefix="/api")

    from src.api.routers import cars as cars_router
    app.include_router(cars_router.router, prefix="/api")

    # Phase 2:
    from src.api.routers import recordings as recordings_router
    app.include_router(recordings_router.router, prefix="/api")

    # Phase 4:
    from src.api.routers import live as live_router
    app.include_router(live_router.router, prefix="/api")

    return app


app = create_app()
