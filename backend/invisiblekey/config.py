"""Tunable configuration (Pydantic) shared by engine, settings and API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LayoutConfig(BaseModel):
    """Keyboard rectangle inside the calibrated surface (normalized units)."""
    x: float = Field(0.0, ge=-0.5, le=1.0)
    y: float = Field(0.0, ge=-0.5, le=1.0)
    width: float = Field(1.0, gt=0.05, le=2.0)
    height: float = Field(1.0, gt=0.05, le=2.0)
    hit_tolerance: float = Field(0.35, ge=0.0, le=1.0, description="Max distance outside a key, in key units")


class TapDetectionConfig(BaseModel):
    """Rule-based tap engine. Press signal unit = hand scale (wrist→middle MCP)."""
    descend_velocity: float = Field(1.2, gt=0, description="Press velocity that starts a descent (scale/s)")
    stop_ratio: float = Field(0.35, gt=0, lt=1, description="Impact when velocity falls below ratio*peak")
    min_travel: float = Field(0.10, gt=0, description="Minimum press travel for a tap (scale)")
    noise_k: float = Field(3.0, ge=0, description="Travel must also exceed k * estimated per-finger jitter")
    noise_adapt_rate: float = Field(0.05, gt=0, le=1)
    max_descent_ms: float = Field(250, gt=0)
    lift_fraction: float = Field(0.2, gt=0, lt=1)
    rearm_fraction: float = Field(0.5, gt=0, lt=1, description="Lift needed before the finger can tap again")
    confirm_on_lift: bool = Field(False, description="Emit only after the finger lifts (fewer FP, more latency)")
    max_dwell_ms: float = Field(400, gt=0, description="With confirm_on_lift: longer contact = resting finger")
    loss_timeout_ms: float = Field(120, gt=0, description="Tracking gap that resets a finger")
    reacquire_frames: int = Field(3, ge=1)
    moving_speed: float = Field(0.4, gt=0, description="Knuckle speed (surface/s) shown as MOVING")
    filter_min_cutoff: float = Field(1.5, gt=0, description="One Euro filter min cutoff (Hz)")
    filter_beta: float = Field(2.0, ge=0)
    filter_d_cutoff: float = Field(4.0, gt=0)
    lateral_smoothing: float = Field(0.6, gt=0, le=1)


class ConfidenceConfig(BaseModel):
    accept_threshold: float = Field(0.6, ge=0, le=1)
    uncertain_threshold: float = Field(0.4, ge=0, le=1)
    velocity_ref: float = Field(2.5, gt=0)
    travel_ref: float = Field(0.22, gt=0)
    max_lateral_speed: float = Field(1.0, gt=0)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "velocity": 1.0, "deceleration": 1.0, "travel": 1.0,
        "stability": 1.0, "tracking": 1.0, "key": 1.0,
    })


class PolicyConfig(BaseModel):
    same_key_window_ms: float = Field(80, ge=0, description="Suppress same key from another finger")
    min_repeat_interval_ms: float = Field(70, ge=0, description="Debounce for the same finger+key")
