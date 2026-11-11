"""Local persistence of calibration and tuning (no cloud)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .calibration import DeskCalibration
from .config import ConfidenceConfig, LayoutConfig, PolicyConfig, TapDetectionConfig

log = logging.getLogger(__name__)


class AppSettings(BaseModel):
    camera_index: int = 0
    calibration: DeskCalibration | None = None
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    detection: TapDetectionConfig = Field(default_factory=TapDetectionConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


def default_settings_path() -> Path:
    if env := os.environ.get("INVISIBLEKEY_HOME"):
        base = Path(env)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "InvisibleKey"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "InvisibleKey"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "invisiblekey"
    return base / "settings.json"


class SettingsManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_settings_path()

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            return AppSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            backup = self.path.with_suffix(".corrupt.json")
            self.path.replace(backup)
            log.warning("invalid settings (%s); moved to %s, using defaults", exc, backup)
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic
