"""Smoke the installed wheel from outside the source checkout."""
from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.metadata
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

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
    if len(gui_lines) != 1:
        raise SystemExit(f"cannot read OpenCV GUI build information: {gui_lines}")
    # The GUI build field describes which window toolkit the wheel links, not
    # whether display-less processing works: macOS headless wheels report
    # "GUI: COCOA" yet run without any display configuration. What the product
    # requires is that decode, DNN, composition, and encoding complete without
    # the user configuring graphical dependencies, so verify those operations
    # directly instead of asserting a build string.
    _check_headless_ops()
    return providers, gui_lines[0]


def _check_headless_ops() -> None:
    """Exercise the cv2 operations the product needs, without any display."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.mp4")
        frame = np.full((32, 48, 3), 120, np.uint8)
        frame[10:20, 12:36] = 240
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (48, 32))
        if not writer.isOpened():
            raise SystemExit("cv2 cannot open a video writer in this environment")
        for _ in range(4):
            writer.write(frame)
        writer.release()
        reader = cv2.VideoCapture(path)
        ok, decoded = reader.read()
        reader.release()
        if not ok or decoded is None:
            raise SystemExit("cv2 cannot decode video in this environment")
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
        if cv2.connectedComponents((gray > 100).astype(np.uint8))[0] < 1:
            raise SystemExit("cv2 connected components failed in this environment")
        probe_png = os.path.join(tmp, "probe.png")
        if not cv2.imwrite(probe_png, decoded):
            raise SystemExit("cv2 cannot write images in this environment")
        if cv2.imread(probe_png) is None:
            raise SystemExit("cv2 cannot read images in this environment")
    if not hasattr(cv2, "dnn") or not hasattr(cv2.dnn, "readNetFromONNX"):
        raise SystemExit("cv2 lacks the DNN module used for text detection")


def main() -> None:
    requirements = importlib.metadata.requires("videowipe") or []
    if not any(item.startswith("opencv-python-headless") for item in requirements):
        raise SystemExit("installed metadata does not require opencv-python-headless")
    if any(item.startswith("opencv-python ") for item in requirements):
        raise SystemExit("installed metadata still requires the GUI OpenCV package")

    providers, gui = _check_headless()
    # Media stack: the default task path is cv2-only (verified above). The
    # imageio/imageio-ffmpeg pair belongs to the optional propainter extra and
    # must not be a base requirement; when present it must still import cleanly.
    base_requirements = [item for item in requirements if "extra ==" not in item]
    if any(item.startswith("imageio") for item in base_requirements):
        raise SystemExit("base install unexpectedly requires imageio")
    media_extras = "present"
    try:
        import imageio.v2
        import imageio_ffmpeg

        _load_media_dependencies()
        assert imageio.v2 is not None
        assert imageio_ffmpeg is not None
    except ImportError:
        media_extras = "absent (optional propainter stack not installed)"

    # A base install ships without an inference backend; with an optional
    # backend extra installed the model must actually load.
    backend_extra = any(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "onnxruntime", "onnxruntime_gpu")
    )
    try:
        WipeEngine()._ensure_model()
    except BackendUnavailableError as exc:
        if backend_extra or exc.code != "BACKEND_UNAVAILABLE":
            raise
    else:
        if not backend_extra:
            raise SystemExit("base install unexpectedly exposed an inference backend")

    print(
        "installed-sdk-ok",
        WipeRequest.__name__,
        WipeResult.__name__,
        providers,
        gui,
        media_extras,
    )


if __name__ == "__main__":
    main()
