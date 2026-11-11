"""Record real landmark sessions to JSONL and replay them through the pipeline.

    python -m invisiblekey.recording session.jsonl --expected "hello world"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

import numpy as np

from .observations import FrameObservation, HandObservation


def frame_to_dict(f: FrameObservation) -> dict:
    return {"t": round(f.timestamp, 5), "hands": [
        {"h": h.handedness, "s": round(h.score, 4), "lm": np.round(h.landmarks, 5).tolist()} for h in f.hands]}


def frame_from_dict(d: dict) -> FrameObservation:
    return FrameObservation(d["t"], [HandObservation(h["h"], np.asarray(h["lm"], float), h["s"]) for h in d["hands"]])


class LandmarkRecorder:
    def __init__(self, path: Path, calibration_json: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = open(self.path, "w", encoding="utf-8")
        self._fh.write(json.dumps({"header": True, "calibration": calibration_json}) + "\n")
        self.frames = 0

    def write(self, frame: FrameObservation) -> None:
        self._fh.write(json.dumps(frame_to_dict(frame), separators=(",", ":")) + "\n")
        self.frames += 1

    def close(self) -> None:
        self._fh.close()


def load_recording(path: Path) -> tuple[str | None, list[FrameObservation]]:
    calibration, frames = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("header"):
                calibration = d.get("calibration")
            else:
                frames.append(frame_from_dict(d))
    return calibration, frames


def replay(path: Path, expected: str | None = None) -> dict:
    from .calibration import DeskCalibration
    from .pipeline import build_pipeline
    from .settings import SettingsManager
    from .simulation import levenshtein

    cal_json, frames = load_recording(path)
    settings = SettingsManager().load()
    if cal_json:
        settings.calibration = DeskCalibration.model_validate_json(cal_json)
    pipe = build_pipeline(settings)
    for f in frames:
        pipe.process(f)
    out = {"frames": len(frames), "typed": pipe.text, "metrics": pipe.monitor.snapshot()}
    if expected is not None:
        out["expected"] = expected
        out["char_error_rate"] = round(levenshtein(pipe.text, expected) / max(1, len(expected)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded landmark session")
    ap.add_argument("path", type=Path)
    ap.add_argument("--expected")
    a = ap.parse_args()
    print(json.dumps(replay(a.path, a.expected), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
