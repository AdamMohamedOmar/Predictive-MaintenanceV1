"""Smoke test for the FastAPI app skeleton (Task 0.5).

Tests:
  - GET /api/health returns {"status": "ok"}

Note: The /api/_smoke development endpoint was removed in Task 2.2 once the
real recordings upload route was available.
"""

from __future__ import annotations


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
