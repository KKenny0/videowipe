"""Auto-detect text regions in video for mask generation.

This module provides:

- :class:`TextDetector` — a protocol for pluggable text-region detectors.
  Implement this to integrate custom detection models or external APIs.
- :class:`DBNetDetector` — built-in detector that loads DBNet-family ONNX
  models via OpenCV DNN (zero extra dependencies beyond ``opencv-python``).
- :func:`detect_subtitle_mask` — high-level function that samples frames,
  runs detection, and produces a binary mask.

Quick start (mask auto-detected)::

    from videowipe import remove_text
    remove_text("video.mp4")

Custom detector::

    from videowipe.detect import detect_subtitle_mask, TextDetector, TextBox

    class MyDetector:
        def detect(self, frame):
            # call your API or model here
            return [TextBox(points=..., confidence=...)]

    mask = detect_subtitle_mask("video.mp4", detector=MyDetector())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class TextBox:
    """A detected text region.

    Attributes:
        points: ``(N, 2)`` float array of polygon vertices (pixel coords).
        confidence: Detection confidence in ``[0, 1]``.
    """

    points: np.ndarray
    confidence: float


# ── Detector protocol ────────────────────────────────────────────────────────

@runtime_checkable
class TextDetector(Protocol):
    """Protocol for pluggable text-region detectors.

    Implement this to use a custom detection model or an external API.
    The only required method is :meth:`detect`.

    Example — wrapping a remote OCR API::

        import requests

        class APIDetector:
            def __init__(self, endpoint: str, api_key: str):
                self.endpoint = endpoint
                self.api_key = api_key

            def detect(self, frame: np.ndarray) -> list[TextBox]:
                _, buf = cv2.imencode(".png", frame)
                resp = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"image": buf.tobytes()},
                )
                resp.raise_for_status()
                return [
                    TextBox(
                        points=np.array(d["polygon"], dtype=np.float32),
                        confidence=d["score"],
                    )
                    for d in resp.json()["detections"]
                ]
    """

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        """Detect text regions in a BGR ``uint8`` frame.

        Args:
            frame: ``(H, W, 3)`` BGR image.

        Returns:
            List of :class:`TextBox`.
        """
        ...


# ── Built-in DBNet detector ──────────────────────────────────────────────────

class DBNetDetector:
    """DBNet-family text detector using OpenCV DNN.

    Loads an ONNX model and performs text detection with configurable
    post-processing.  Works with DBNet models exported from PaddleOCR,
    MMOCR, or other frameworks that output a probability map.

    Tries to use ``cv2.dnn.TextDetectionModel_DB`` (OpenCV >= 4.5.4)
    for built-in post-processing; falls back to a manual implementation
    if the high-level API is unavailable.

    Args:
        weight_path: Path to the ONNX model file.
        input_size: ``(width, height)`` — both should be multiples of 32.
        bin_thresh: Threshold for binarising the probability map.
        box_thresh: Minimum mean probability inside a contour.
        unclip_ratio: Factor to expand each detected box.
        mean: Channel means for normalisation (ImageNet defaults).
        scale: Pixel-scale factor (``1/255`` normalises to ``[0, 1]``).
    """

    def __init__(
        self,
        weight_path: str,
        input_size: tuple[int, int] = (640, 640),
        bin_thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.5,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        scale: float = 1.0 / 255.0,
    ):
        self._input_w, self._input_h = input_size
        self._bin_thresh = bin_thresh
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio
        self._mean = mean
        self._scale = scale

        self._net = cv2.dnn.readNetFromONNX(weight_path)

        # Try high-level OpenCV API (>= 4.5.4)
        self._hl_model = None
        try:
            model = cv2.dnn.TextDetectionModel_DB(self._net)
            model.setInputParams(
                scale=scale, size=input_size, mean=mean, swapRB=True,
            )
            model.setBinaryThreshold(bin_thresh)
            model.setPolygonThreshold(box_thresh)
            model.setUnclipRatio(unclip_ratio)
            model.setMaxCandidates(200)
            self._hl_model = model
        except (AttributeError, cv2.error):
            pass

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        """Run text detection on a single frame."""
        if self._hl_model is not None:
            boxes = self._detect_hl(frame)
            if boxes:
                return boxes
            # High-level API returned nothing — fall back to manual post-processing
            logger.debug("High-level DB API returned 0 boxes; falling back to manual path")
        return self._detect_manual(frame)

    def _detect_hl(self, frame: np.ndarray) -> List[TextBox]:
        detections, confidences = self._hl_model.detect(frame)
        boxes: list[TextBox] = []
        for pts, conf in zip(detections, confidences):
            pts = pts.squeeze()
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            boxes.append(
                TextBox(points=pts.astype(np.float32), confidence=float(conf))
            )
        return boxes

    def _detect_manual(self, frame: np.ndarray) -> List[TextBox]:
        h, w = frame.shape[:2]
        ratio = min(self._input_w / w, self._input_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full(
            (self._input_h, self._input_w, 3), 128, dtype=np.uint8
        )
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=self._scale,
            size=(self._input_w, self._input_h),
            mean=self._mean,
            swapRB=True,
        )
        self._net.setInput(blob)
        out = self._net.forward()

        # Probability map: (1, 1, H', W')  or  (1, 2, H', W')
        prob = out[0, 0]
        # Apply sigmoid if raw logits
        if prob.max() > 1.0 or prob.min() < 0.0:
            prob = 1.0 / (1.0 + np.exp(-prob))

        prob = cv2.resize(prob, (w, h))

        binary = (prob > self._bin_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: list[TextBox] = []
        for cnt in contours:
            if len(cnt) < 4:
                continue
            rect = cv2.minAreaRect(cnt)
            bw, bh = rect[1]
            if min(bw, bh) < 3:
                continue

            cnt_mask = np.zeros(prob.shape, dtype=np.uint8)
            cv2.fillPoly(cnt_mask, [cnt], 1)
            score = float(cv2.mean(prob, cnt_mask)[0])
            if score < self._box_thresh:
                continue

            # Approximate unclip by scaling around box centre
            pts = cv2.boxPoints(rect).astype(np.float32)
            centre = pts.mean(axis=0)
            expanded = centre + (pts - centre) * self._unclip_ratio

            boxes.append(TextBox(points=expanded, confidence=score))
        return boxes


# ── Mask generation pipeline ─────────────────────────────────────────────────

def _sample_frames(video_path: str, count: int = 30) -> List[np.ndarray]:
    """Uniformly sample *count* frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot determine frame count: {video_path}")

    count = min(count, total)
    indices = sorted(set(np.linspace(0, total - 1, count, dtype=int)))

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def detect_subtitle_mask(
    video_path: str,
    detector: TextDetector | None = None,
    sample_count: int = 30,
    consistency: float = 0.6,
) -> np.ndarray:
    """Auto-detect subtitle regions and return a binary mask.

    1. Uniformly sample *sample_count* frames from the video.
    2. Run *detector* :meth:`~TextDetector.detect` on each frame.
    3. Build a per-pixel frequency map (fraction of frames with detected
       text at that location).
    4. Threshold at *consistency* and apply morphological cleanup.

    Args:
        video_path: Path to the input video.
        detector: A :class:`TextDetector`. ``None`` uses the built-in
            :class:`DBNetDetector` with auto-downloaded weights.
        sample_count: Frames to sample from the video.
        consistency: Fraction (0–1) of frames a pixel must appear in.
            Higher values reduce false positives but may miss short subtitles.

    Returns:
        ``(H, W, 1)`` uint8 array with values in ``{0, 1}`` —
        same format as :func:`~videowipe.tasks.base.read_mask`.

    Raises:
        ValueError: Cannot read the video.
        RuntimeError: Detection fails on every sampled frame.
    """
    if detector is None:
        detector = _default_detector()

    print(f"Sampling {sample_count} frames for mask detection...")
    frames = _sample_frames(video_path, sample_count)
    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    h, w = frames[0].shape[:2]

    # Accumulate detection frequency
    freq = np.zeros((h, w), dtype=np.float32)
    n_valid = 0

    for i, frame in enumerate(frames):
        try:
            boxes = detector.detect(frame)
        except Exception as exc:
            logger.warning("Detection failed on frame %d: %s", i, exc)
            continue

        frame_mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            cv2.fillPoly(frame_mask, [box.points.astype(np.int32)], 1)
        freq += frame_mask
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError(
            "Text detection failed on all sampled frames.\n"
            "Provide a mask manually with -m, or check the detection model."
        )

    freq /= n_valid
    mask = (freq >= consistency).astype(np.uint8)

    # Morphological cleanup: dilate then close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    pct = 100.0 * mask.sum() / (h * w)
    print(f"Auto-detected mask: {pct:.1f}% of frame area")

    return mask[:, :, None]


def _default_detector() -> DBNetDetector:
    """Create the default detector with auto-downloaded weights."""
    from videowipe.weights import ensure_weight

    weight = ensure_weight("ppocrv5_det_mob.onnx", version="v0.1.0")
    return DBNetDetector(weight)
