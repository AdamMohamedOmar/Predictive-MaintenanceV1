"""Disk persistence for live ELM327 sessions.

A live Skoda run is unrepeatable evidence: rows.csv lets the §10 recall be
recomputed offline through the tested CLI path, and marks.json preserves the
mods-in / mods-out ground truth that the WebSocket used to discard.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.config import USEFUL_PIDS


class LiveSessionStore:
    """Append-only writer for one live session (rows.csv + marks.json)."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._marks: list[dict] = []
        self._alerts: list[dict] = []
        self._last_elapsed: int = -1
        self._rows_f = open(
            self.session_dir / "rows.csv", "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._rows_f, fieldnames=["elapsed_s", *USEFUL_PIDS]
        )
        self._writer.writeheader()

    def append_row(self, elapsed_s: int, row: dict) -> None:
        if elapsed_s <= self._last_elapsed:
            return  # resampler can re-deliver the same data-second; keep one
        self._last_elapsed = elapsed_s
        rec: dict = {"elapsed_s": elapsed_s}
        for pid in USEFUL_PIDS:
            v = row.get(pid)
            try:
                f = float(v)
                rec[pid] = "" if math.isnan(f) else f
            except (TypeError, ValueError):
                rec[pid] = ""
        self._writer.writerow(rec)
        self._rows_f.flush()  # a crash mid-drive must not lose the run

    def record_mark(self, state: str, elapsed_s: int) -> None:
        self._marks.append({"state": str(state), "elapsed_s": int(elapsed_s)})
        (self.session_dir / "marks.json").write_text(
            json.dumps(self._marks, indent=2)
        )

    def record_alert(self, event: dict) -> None:
        self._alerts.append(dict(event))
        (self.session_dir / "alerts.json").write_text(json.dumps(self._alerts, indent=2))

    def close(self) -> None:
        if not self._rows_f.closed:
            self._rows_f.close()
        if not (self.session_dir / "marks.json").exists():
            (self.session_dir / "marks.json").write_text("[]")
        if not (self.session_dir / "alerts.json").exists():
            (self.session_dir / "alerts.json").write_text("[]")
