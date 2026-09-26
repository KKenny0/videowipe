"""Smoke the installed wheel from outside the source checkout."""
from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.metadata
from pathlib import Path

import cv2

from videowipe import BackendUnavailableError, WipeEngine, WipeRequest, WipeResult
from videowipe.propainter_wipe import _load_media_dependencies


def _check_headless():
    """Verify installed provider, loaded binary RECORD, and GUI capability."""
    def normalize(name):
        return name.lower().replace("_", "-").replace(".", "-")
    expected = "opencv-python-headless"
    providers = [normalize(name) for name in
                 importlib.metadata.packages_distributions().get("cv2", [])]
    variants = {"opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"}
    installed = {normalize(dist.metadata["Name"]) for dist in importlib.metadata.distributions()
                 if dist.metadata.get("Name")}
    if providers != [expected] or installed & variants:
        raise SystemExit(f"cv2 provider conflict: {providers}; competing distributions: {sorted(installed & variants)}")
    native = getattr(getattr(cv2, "_native", None), "__file__", None)
    if not native:
        native = getattr(cv2, "__file__", "")
        if not any(native.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            raise SystemExit("cannot identify loaded cv2 native binary")
    native = Path(native).resolve()
    files = importlib.metadata.distribution(expected).files or []
    record = next((entry for entry in files if Path(entry.locate()).resolve() == native), None)
    if record is None or record.hash is None:
        raise SystemExit("loaded cv2 binary has no headless RECORD hash")
    try:
        digest = hashlib.new(record.hash.mode, native.read_bytes()).digest()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot verify loaded cv2 binary: {exc}") from exc
    actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    if actual != record.hash.value:
        raise SystemExit("loaded cv2 binary differs from headless RECORD")
    gui_lines = [line.strip() for line in cv2.getBuildInformation().splitlines()
                 if line.strip().startswith("GUI:")]
    if len(gui_lines) != 1 or gui_lines[0].split(":", 1)[1].strip() != "NONE":
        raise SystemExit(f"OpenCV is not headless: {gui_lines}")
    return providers, gui_lines[0]


def main() -> None:
    requirements = importlib.metadata.requires("videowipe") or []
    if not any(item.startswith("opencv-python-headless") for item in requirements):
        raise SystemExit("installed metadata does not require opencv-python-headless")
    if any(item.startswith("opencv-python ") for item in requirements):
        raise SystemExit("installed metadata still requires the GUI OpenCV package")

    providers, gui = _check_headless()
    import imageio.v2
    import imageio_ffmpeg

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
        providers,
        gui,
    )


if __name__ == "__main__":
    main()
