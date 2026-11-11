"""Download/cache the MediaPipe Hand Landmarker model (once, then fully offline)."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

from ..settings import default_settings_path

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def default_model_path() -> Path:
    return default_settings_path().parent / "models" / "hand_landmarker.task"


def is_valid_model(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith(".tflite") for n in z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def _ssl_context() -> ssl.SSLContext:
    """python.org builds on macOS ship without CA certs; certifi fixes that."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def ensure_model(path: Path | None = None, url: str = MODEL_URL) -> Path:
    path = Path(path) if path else default_model_path()
    if path.exists() and is_valid_model(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    if not is_valid_model(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is not a valid MediaPipe model: {url}")
    tmp.replace(path)
    return path


if __name__ == "__main__":
    print(ensure_model())
