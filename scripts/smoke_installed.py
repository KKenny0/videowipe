"""Smoke the installed wheel from outside the source checkout."""
from __future__ import annotations

import importlib.metadata

import cv2
import imageio.v2
import imageio_ffmpeg

from videowipe import BackendUnavailableError, WipeEngine, WipeRequest, WipeResult
from videowipe.propainter_wipe import _load_media_dependencies


def main() -> None:
    requirements = importlib.metadata.requires("videowipe") or []
    if not any(item.startswith("opencv-python-headless") for item in requirements):
        raise SystemExit("installed metadata does not require opencv-python-headless")
    if any(item.startswith("opencv-python ") for item in requirements):
        raise SystemExit("installed metadata still requires the GUI OpenCV package")

    gui_lines = [
        line.strip() for line in cv2.getBuildInformation().splitlines()
        if line.strip().startswith("GUI:")
    ]
    if not gui_lines or "NONE" not in gui_lines[0]:
        raise SystemExit(f"OpenCV is not headless: {gui_lines}")

    _load_media_dependencies()
    assert imageio.v2 is not None
    assert imageio_ffmpeg is not None

    try:
        WipeEngine()._ensure_model()
    except BackendUnavailableError as exc:
        if exc.code != "BACKEND_UNAVAILABLE":
            raise
    else:
        raise SystemExit("base install unexpectedly exposed an inference backend")

    print(
        "installed-sdk-ok",
        WipeRequest.__name__,
        WipeResult.__name__,
        gui_lines[0],
    )


if __name__ == "__main__":
    main()
