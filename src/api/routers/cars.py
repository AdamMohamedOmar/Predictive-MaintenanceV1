"""Cars router — garage CRUD for the authenticated user."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.auth import current_user
from src.api.config import DATA_APP_DIR
from src.api.db import get_db
from src.api.models import Car, User
from src.api.schemas import CarCreate, CarOut

router = APIRouter(prefix="/cars", tags=["cars"])


def _car_dir(user_id: int, car_id: int) -> Path:
    """Return (and create) the per-car directory for storing uploaded files."""
    d = DATA_APP_DIR / "users" / str(user_id) / "cars" / str(car_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _own_car(car_id: int, user: User, db: Session) -> Car:
    """Fetch a car owned by this user; raise 404 if not found or not owned."""
    car = db.get(Car, car_id)
    if car is None or car.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Car not found.")
    return car


@router.get("", response_model=list[CarOut])
def list_cars(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """List all cars belonging to the authenticated user."""
    return db.query(Car).filter_by(user_id=user.id).all()


@router.post("", response_model=CarOut, status_code=status.HTTP_201_CREATED)
def create_car(
    body: CarCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Add a new car to the authenticated user's garage."""
    car = Car(
        user_id=user.id,
        make=body.make,
        model=body.model,
        year=body.year,
        engine_metering=body.engine_metering,
        created_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    )
    db.add(car)
    db.commit()
    db.refresh(car)
    # Create the per-car file directory now so upload paths always exist
    _car_dir(user.id, car.id)
    return car


@router.get("/{car_id}", response_model=CarOut)
def get_car(
    car_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Fetch a single car (must be owned by the authenticated user)."""
    return _own_car(car_id, user, db)


@router.delete("/{car_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_car(
    car_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Remove a car and all associated recordings from the garage."""
    car = _own_car(car_id, user, db)
    db.delete(car)
    db.commit()
