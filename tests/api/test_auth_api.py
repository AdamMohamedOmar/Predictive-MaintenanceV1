"""Integration tests for auth + cars endpoints (Tasks 1.2 + 1.3)."""

from __future__ import annotations



# ── Helpers ───────────────────────────────────────────────────────────────────

def _signup(client, username="adam", password="pass123"):
    resp = client.post("/api/auth/signup", json={"username": username, "password": password})
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Auth tests ────────────────────────────────────────────────────────────────

def test_signup_returns_token(client):
    tok = _signup(client)
    assert isinstance(tok, str) and len(tok) > 10


def test_duplicate_signup_409(client):
    _signup(client, "adam")
    resp = client.post("/api/auth/signup", json={"username": "adam", "password": "x"})
    assert resp.status_code == 409


def test_login_correct_credentials(client):
    _signup(client, "adam", "pass123")
    resp = client.post("/api/auth/login", json={"username": "adam", "password": "pass123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password_401(client):
    _signup(client, "adam", "right")
    resp = client.post("/api/auth/login", json={"username": "adam", "password": "wrong"})
    assert resp.status_code == 401


def test_token_authorizes_cars_endpoint(client):
    tok = _signup(client)
    resp = client.get("/api/cars", headers=_auth_header(tok))
    assert resp.status_code == 200
    assert resp.json() == []   # empty garage after signup


def test_unauthenticated_cars_401(client):
    resp = client.get("/api/cars")
    assert resp.status_code == 401


# ── Cars CRUD tests ───────────────────────────────────────────────────────────

def test_create_and_list_car(client):
    tok = _signup(client)
    hdrs = _auth_header(tok)

    resp = client.post("/api/cars", json={
        "make": "Skoda", "model": "Roomster", "year": 2007, "engine_metering": "maf"
    }, headers=hdrs)
    assert resp.status_code == 201
    car = resp.json()
    assert car["make"] == "Skoda"
    assert car["engine_metering"] == "maf"
    assert car["baseline_normalizer_path"] is None

    cars = client.get("/api/cars", headers=hdrs).json()
    assert len(cars) == 1
    assert cars[0]["id"] == car["id"]


def test_get_car(client):
    tok = _signup(client)
    hdrs = _auth_header(tok)
    car_id = client.post("/api/cars", json={
        "make": "Toyota", "model": "Etios", "year": 2014
    }, headers=hdrs).json()["id"]

    resp = client.get(f"/api/cars/{car_id}", headers=hdrs)
    assert resp.status_code == 200
    assert resp.json()["model"] == "Etios"


def test_delete_car(client):
    tok = _signup(client)
    hdrs = _auth_header(tok)
    car_id = client.post("/api/cars", json={
        "make": "Ford", "model": "Fiesta", "year": 2010
    }, headers=hdrs).json()["id"]

    resp = client.delete(f"/api/cars/{car_id}", headers=hdrs)
    assert resp.status_code == 204

    resp = client.get(f"/api/cars/{car_id}", headers=hdrs)
    assert resp.status_code == 404


def test_cross_user_car_access_404(client):
    """User A cannot see User B's car."""
    tok_a = _signup(client, "alice", "pw")
    tok_b = _signup(client, "bob", "pw")

    car_id = client.post("/api/cars", json={
        "make": "Honda", "model": "Civic", "year": 2020
    }, headers=_auth_header(tok_a)).json()["id"]

    resp = client.get(f"/api/cars/{car_id}", headers=_auth_header(tok_b))
    assert resp.status_code == 404
