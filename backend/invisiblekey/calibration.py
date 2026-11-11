"""Desk calibration: four image corners (TL, TR, BR, BL) → surface homography."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from .geometry import CoordinateTransformer, validate_quad

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


class DeskCalibration(BaseModel):
    image_corners: list[tuple[float, float]]
    camera_id: str = "0"
    frame_size: tuple[int, int] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("image_corners")
    @classmethod
    def _valid(cls, v):
        validate_quad(v)
        return v

    def transformer(self) -> CoordinateTransformer:
        return CoordinateTransformer(self.image_corners)


class DeskCalibrationService:
    """Step-by-step corner capture used by the calibration wizard."""

    def __init__(self) -> None:
        self.corners: list[tuple[float, float]] = []

    @property
    def next_corner(self) -> str | None:
        return CORNER_ORDER[len(self.corners)] if len(self.corners) < 4 else None

    @property
    def complete(self) -> bool:
        return len(self.corners) == 4

    def add_corner(self, x: float, y: float) -> str | None:
        if self.complete:
            raise ValueError("all four corners already set")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("corner must be inside the frame (normalized 0..1)")
        self.corners.append((float(x), float(y)))
        return self.next_corner

    def undo(self) -> None:
        if self.corners:
            self.corners.pop()

    def reset(self) -> None:
        self.corners.clear()

    def build(self, camera_id: str = "0", frame_size: tuple[int, int] | None = None) -> DeskCalibration:
        if not self.complete:
            raise ValueError(f"missing corner: {self.next_corner}")
        return DeskCalibration(image_corners=list(self.corners), camera_id=camera_id, frame_size=frame_size)
