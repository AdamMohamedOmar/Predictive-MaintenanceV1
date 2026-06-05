"""API configuration — paths, DB URL, JWT settings, CORS.

All configuration lives here. Other modules import from this module only;
they never reach for os.environ or Path(__file__) themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── File-system paths ─────────────────────────────────────────────────────────

# Root directory for all app-managed user data (DB + uploaded files).
# Untracked by git (regenerable / local data).
DATA_APP_DIR: Path = _REPO_ROOT / "data" / "app"

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH: Path = DATA_APP_DIR / "app.db"
DB_URL: str = f"sqlite:///{DB_PATH}"

# ── JWT ───────────────────────────────────────────────────────────────────────

# Read from env in production; use a dev-only default for local runs.
# The default is intentionally weak — it is not a secret; set PM_JWT_SECRET
# in the environment for any deployment beyond a laptop demo.
JWT_SECRET: str = os.environ.get("PM_JWT_SECRET", "dev-only-pm-jwt-secret-change-me")
JWT_ALGO: str = "HS256"
JWT_EXPIRY_DAYS: int = 7

# ── CORS ──────────────────────────────────────────────────────────────────────

# Vite dev server and the built static app.
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",   # vite dev
    "http://localhost:8000",   # uvicorn serving built dist
]
