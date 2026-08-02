"""Auto-detect text regions in video for mask generation.

This module provides:

- :class:`TextDetector` — a protocol for pluggable text-region detectors.
  Implement this to integrate custom detection models or external APIs.
- :class:`DBNetDetector` — built-in detector that loads DBNet-family ONNX
  models via OpenCV DNN (zero extra dependencies beyond the default
  ``opencv-python-headless`` runtime).
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
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Protocol, runtime_checkable

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
    text: str = ""


TargetType = Literal[
    "subtitle",
    "timestamp",
    "watermark",
    "logo",
    "scene_text",
    "unknown_text",
    "region",
]


@dataclass
class CleanCandidate:
    """A candidate object that can be removed from a video."""

    id: str
    type: TargetType
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    frame_fraction: float
    reason: str
    default_remove: bool
    text_samples: list[str] = field(default_factory=list)
    presence_frames: list[int] = field(default_factory=list)
    mask: np.ndarray | None = field(default=None, repr=False)
    # Runtime-only evidence used to sharpen temporal plan segments.  It is
    # deliberately absent from to_dict()/WipePlan v1 serialization.
    temporal_sample_indices: list[int] = field(default_factory=list, repr=False)
    detector_backed: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        """Return a JSON-safe representation without the binary mask."""
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "frame_fraction": round(float(self.frame_fraction), 3),
            "reason": self.reason,
            "default_remove": self.default_remove,
            "text_samples": self.text_samples[:5],
            "presence_frames": list(self.presence_frames),
        }


@dataclass
class CleanDetectionResult:
    """Result of clean-target detection."""

    candidates: list[CleanCandidate]
    frame_shape: tuple[int, int]
    sample_indices: list[int] = field(default_factory=list)
    preview_frame: np.ndarray | None = field(default=None, repr=False)
    detector: TextDetector | None = field(default=None, repr=False)


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
        adaptive: bool = True,
    ):
        self._input_w, self._input_h = input_size
        self._bin_thresh = bin_thresh
        self._box_thresh = box_thresh
        self._unclip_ratio = unclip_ratio
        self._mean = mean
        self._scale = scale
        self._adaptive = adaptive

        self._net = cv2.dnn.readNetFromONNX(weight_path)

        # Cache for adaptive input-size computation
        self._cached_frame_dims: tuple[int, int] | None = None
        self._cached_input_size: tuple[int, int] | None = None

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

    def _adaptive_input_size(self, frame_h: int, frame_w: int) -> tuple[int, int]:
        """Compute model input size that matches the frame aspect ratio.

        Keeps the total pixel budget close to ``input_size`` area while
        choosing width/height that match the frame's aspect ratio.
        This avoids the resolution loss that occurs when a widescreen
        frame is squeezed into a square input.
        """
        dims = (frame_h, frame_w)
        if self._cached_frame_dims == dims and self._cached_input_size is not None:
            return self._cached_input_size

        aspect = frame_w / frame_h
        target_area = self._input_w * self._input_h
        new_w = int(np.sqrt(target_area * aspect))
        new_h = int(new_w / aspect)
        # Round up to multiples of 32 (required by the model)
        new_w = ((new_w + 31) // 32) * 32
        new_h = ((new_h + 31) // 32) * 32

        result = (new_w, new_h)
        self._cached_frame_dims = dims
        self._cached_input_size = result
        return result

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        """Run text detection on a single frame."""
        h, w = frame.shape[:2]
        if self._adaptive:
            input_w, input_h = self._adaptive_input_size(h, w)
        else:
            input_w, input_h = self._input_w, self._input_h

        if self._hl_model is not None:
            # Reconfigure input size when adaptive or first call
            if self._adaptive:
                self._hl_model.setInputParams(
                    scale=self._scale, size=(input_w, input_h),
                    mean=self._mean, swapRB=True,
                )
            boxes = self._detect_hl(frame)
            if boxes:
                return boxes
            # High-level API returned nothing — fall back to manual post-processing
            logger.debug("High-level DB API returned 0 boxes; falling back to manual path")
        return self._detect_manual(frame, input_w=input_w, input_h=input_h)

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

    def _detect_manual(
        self,
        frame: np.ndarray,
        input_w: int | None = None,
        input_h: int | None = None,
    ) -> List[TextBox]:
        iw = input_w or self._input_w
        ih = input_h or self._input_h
        h, w = frame.shape[:2]
        ratio = min(iw / w, ih / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((ih, iw, 3), 128, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=self._scale,
            size=(iw, ih),
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

def _sample_frames_with_indices(
    video_path: str, count: int = 30
) -> list[tuple[int, np.ndarray]]:
    """Uniformly sample *count* frames, returning ``(frame_index, frame)`` pairs.

    The frame indices are the real positions in the source video, preserved so
    callers can derive temporal evidence (which frames a candidate was present
    on). Only successfully read frames are returned, in ascending index order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot determine frame count: {video_path}")

    count = min(count, total)
    indices = sorted(set(np.linspace(0, total - 1, count, dtype=int)))

    indexed: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            indexed.append((int(idx), frame))
    cap.release()
    return indexed


def _sample_frames(video_path: str, count: int = 30) -> List[np.ndarray]:
    """Uniformly sample *count* frames from a video (drops frame indices)."""
    return [frame for _, frame in _sample_frames_with_indices(video_path, count)]


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


_TIME_RE = re.compile(
    r"(\b\d{1,2}:\d{2}(:\d{2})?\b|\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b)"
)
_WATERMARK_RE = re.compile(
    r"(@|www\.|https?://|\.com\b|\.net\b|\.tv\b|tiktok|douyin|bilibili|youtube|weibo)",
    re.IGNORECASE,
)
_INTENT_TARGETS = {
    "subtitle": ("subtitle", "subtitles", "caption", "captions", "字幕"),
    "timestamp": ("timestamp", "timecode", "date", "time", "时间戳", "日期", "时间"),
    "watermark": ("watermark", "water mark", "水印", "账号", "网址"),
    "logo": ("logo", "台标", "角标", "标志"),
    "scene_text": ("scene text", "road sign", "sign", "路牌", "招牌", "画面文字"),
}
_INTENT_ZONES = {
    "top-left": ("top left", "upper left", "左上", "左上角"),
    "top-right": ("top right", "upper right", "右上", "右上角"),
    "bottom-left": ("bottom left", "lower left", "左下", "左下角"),
    "bottom-right": ("bottom right", "lower right", "右下", "右下角"),
    "top": ("top", "upper", "顶部", "上方"),
    "bottom": ("bottom", "lower", "底部", "下方"),
    "center": ("center", "middle", "中间", "中央"),
}
_REGION_ALIASES = {
    "top-left": ("top-left", "upper-left", "left-top", "左上", "左上角"),
    "top-right": ("top-right", "upper-right", "right-top", "右上", "右上角"),
    "bottom-left": ("bottom-left", "lower-left", "left-bottom", "左下", "左下角"),
    "bottom-right": ("bottom-right", "lower-right", "right-bottom", "右下", "右下角"),
    "top": ("top", "upper", "顶部", "上方"),
    "bottom": ("bottom", "lower", "底部", "下方"),
    "center": ("center", "middle", "中间", "中央"),
}
_KEEP_WORDS = ("keep", "preserve", "保留", "不要去", "别去", "别删", "不要删")
_REMOVE_WORDS = ("remove", "clean", "erase", "delete", "去掉", "清理", "删除", "擦掉")


def normalize_target(value: str) -> str:
    """Normalize user-facing target names to candidate types."""
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sub": "subtitle",
        "subs": "subtitle",
        "caption": "subtitle",
        "captions": "subtitle",
        "text": "subtitle",
        "subtitle": "subtitle",
        "timestamp": "timestamp",
        "time": "timestamp",
        "date": "timestamp",
        "watermark": "watermark",
        "text_watermark": "watermark",
        "logo": "logo",
        "scene_text": "scene_text",
        "unknown": "unknown_text",
        "region": "region",
    }
    return aliases.get(key, key)


def normalize_region(value: str) -> str:
    """Normalize user-facing region names."""
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    for region, aliases in _REGION_ALIASES.items():
        if key in {alias.lower().replace("_", "-").replace(" ", "-") for alias in aliases}:
            return region
    return key


def infer_regions_from_text(text: str) -> list[str]:
    """Infer region names mentioned in free-form text."""
    regions = []
    for region, aliases in _REGION_ALIASES.items():
        if _mentions_any(text, aliases):
            regions.append(region)
    return regions


def infer_targets_from_text(text: str) -> list[str]:
    """Infer target types mentioned in free-form text."""
    return sorted(_mentioned_targets(text))


def _mentions_any(text: str, terms: Iterable[str]) -> bool:
    folded = text.casefold()
    return any(term.casefold() in folded for term in terms)


def _mentioned_targets(intent: str) -> set[str]:
    return {
        target
        for target, terms in _INTENT_TARGETS.items()
        if _mentions_any(intent, terms)
    }


def _mentioned_zones(intent: str) -> set[str]:
    zones = {
        zone
        for zone, terms in _INTENT_ZONES.items()
        if _mentions_any(intent, terms)
    }
    if any(zone.startswith("top-") for zone in zones):
        zones.add("top")
    if any(zone.startswith("bottom-") for zone in zones):
        zones.add("bottom")
    return zones


def _mentioned_after_words(intent: str, words: Iterable[str]) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    zones: set[str] = set()
    for word in words:
        start = intent.casefold().find(word.casefold())
        if start < 0:
            continue
        fragment = intent[start:start + 40]
        for sep in (",", ";", "，", "；", "."):
            fragment = fragment.split(sep, 1)[0]
        targets.update(_mentioned_targets(fragment))
        zones.update(_mentioned_zones(fragment))
    return targets, zones


def _candidate_matches_intent(
    candidate: CleanCandidate,
    targets: set[str],
    zones: set[str],
    intent: str,
) -> bool:
    candidate_zone = candidate.label.split(" ", 1)[0]
    if targets and normalize_target(candidate.type) not in targets:
        return False
    if zones:
        zone_matches = candidate_zone in zones
        if "top" in zones and candidate_zone.startswith("top"):
            zone_matches = True
        if "bottom" in zones and candidate_zone.startswith("bottom"):
            zone_matches = True
        if not zone_matches:
            return False
    samples = " ".join(candidate.text_samples).casefold()
    return not samples or not intent or any(
        sample.casefold() in intent.casefold()
        for sample in candidate.text_samples
        if sample
    ) or bool(targets or zones)


def _bbox(points: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    pts = points.astype(np.float32)
    x1 = max(0, int(np.floor(pts[:, 0].min())))
    y1 = max(0, int(np.floor(pts[:, 1].min())))
    x2 = min(w - 1, int(np.ceil(pts[:, 0].max())))
    y2 = min(h - 1, int(np.ceil(pts[:, 1].max())))
    return x1, y1, x2, y2


def _zone(bbox: tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    vertical = "top" if cy < h * 0.25 else "bottom" if cy > h * 0.68 else "middle"
    horizontal = "left" if cx < w * 0.33 else "right" if cx > w * 0.67 else "center"
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"



def _candidate_label(target_type: str, zone: str) -> str:
    labels = {
        "subtitle": "subtitle",
        "timestamp": "timestamp",
        "watermark": "text watermark",
        "logo": "logo",
        "scene_text": "scene text",
        "unknown_text": "unknown text",
        "region": "region",
    }
    return f"{zone} {labels.get(target_type, target_type)}"


def _region_bbox(region: str, w: int, h: int) -> tuple[int, int, int, int]:
    band_h = max(1, int(h * 0.22))
    band_w = max(1, int(w * 0.22))
    center_w = max(1, int(w * 0.36))
    center_h = max(1, int(h * 0.24))
    boxes = {
        "top-left": (0, 0, band_w - 1, band_h - 1),
        "top-right": (w - band_w, 0, w - 1, band_h - 1),
        "bottom-left": (0, h - band_h, band_w - 1, h - 1),
        "bottom-right": (w - band_w, h - band_h, w - 1, h - 1),
        "top": (0, 0, w - 1, band_h - 1),
        "bottom": (0, h - band_h, w - 1, h - 1),
        "center": (
            (w - center_w) // 2,
            (h - center_h) // 2,
            (w + center_w) // 2,
            (h + center_h) // 2,
        ),
    }
    if region not in boxes:
        raise ValueError(f"Unknown region: {region}")
    return boxes[region]


def _mask_from_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    mask[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1), 0] = 1
    return mask


def _region_candidates(
    regions: Iterable[str],
    shape: tuple[int, int],
    start_index: int = 1,
) -> list[CleanCandidate]:
    h, w = shape
    candidates: list[CleanCandidate] = []
    for offset, raw_region in enumerate(regions):
        region = normalize_region(raw_region)
        bbox = _region_bbox(region, w, h)
        candidates.append(
            CleanCandidate(
                id=f"r{start_index + offset}",
                type="region",
                label=f"{region} region",
                bbox=bbox,
                confidence=1.0,
                frame_fraction=1.0,
                reason=f"user-specified {region} region",
                default_remove=True,
                mask=_mask_from_bbox(bbox, shape),
            )
        )
    return candidates


def _largest_overlay_candidate(
    mask: np.ndarray,
    zone: str,
    target_type: str,
    reason: str,
    shape: tuple[int, int],
    candidate_id: str,
    confidence: float,
) -> CleanCandidate | None:
    h, w = shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < max(12, h * w * 0.0005):
        return None
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw > w * 0.55 or bh > h * 0.35:
        return None
    bbox = (x, y, x + bw - 1, y + bh - 1)
    dilated = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    return CleanCandidate(
        id=candidate_id,
        type=target_type,  # type: ignore[arg-type]
        label=_candidate_label(target_type, zone),
        bbox=bbox,
        confidence=confidence,
        frame_fraction=1.0,
        reason=reason,
        default_remove=False,
        mask=dilated[:, :, None],
    )


def _prepare_band_variant(crop: np.ndarray, variant: str) -> np.ndarray:
    """Prepare an image variant for band fallback detection."""
    if variant == "original":
        return crop
    if variant == "contrast":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if variant == "inverted":
        return cv2.bitwise_not(crop)
    return crop


def _iou_bbox(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """Compute IoU between two axis-aligned bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _box_overlaps_bbox(
    box: TextBox, bbox: tuple[int, int, int, int], width: int, height: int,
) -> bool:
    """Whether a detector box intersects a candidate bbox."""
    bx1, by1, bx2, by2 = _bbox(box.points, width, height)
    x1, y1, x2, y2 = bbox
    return bx1 <= x2 and bx2 >= x1 and by1 <= y2 and by2 >= y1


def _band_fallback_detect(
    frames: list[np.ndarray],
    detector: TextDetector,
    mode: str = "light",
    consistency: float = 0.4,
) -> list[CleanCandidate]:
    """Detect text in subtitle bands missed by the main frequency-map pass.

    For each band (bottom 40%, optionally top 25%), creates image variants
    and runs the detector.  Builds a per-pixel frequency map from fallback
    detections, then extracts connected-component candidates.
    """
    if not frames:
        return []

    h, w = frames[0].shape[:2]

    bands: list[tuple[str, int, int, int, int]] = []
    bands.append(("bottom", 0, int(h * 0.6), w - 1, h - 1))
    if mode == "force":
        bands.append(("top", 0, 0, w - 1, max(0, int(h * 0.25) - 1)))

    if mode == "light":
        variants = ["original", "contrast"]
    else:
        variants = ["original", "contrast", "inverted"]

    freq = np.zeros((h, w), dtype=np.float32)
    n_valid = 0

    for frame in frames:
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        for _band_name, bx1, by1, bx2, by2 in bands:
            crop = frame[by1:by2 + 1, bx1:bx2 + 1]
            if crop.size == 0:
                continue
            for variant in variants:
                prepared = _prepare_band_variant(crop, variant)
                try:
                    boxes = detector.detect(prepared)
                except Exception:
                    continue
                for box in boxes:
                    pts = box.points.copy()
                    pts[:, 0] += bx1
                    pts[:, 1] += by1
                    cv2.fillPoly(frame_mask, [pts.astype(np.int32)], 1)
        freq += frame_mask
        n_valid += 1

    if n_valid == 0:
        return []

    freq /= n_valid

    text_mask = (freq >= consistency).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(text_mask)

    candidates: list[CleanCandidate] = []
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        ys, xs = np.where(component > 0)
        if len(xs) < 10:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        zone = _zone((x1, y1, x2, y2), w, h)
        target_type, reason, default_remove = _classify_region(
            (x1, y1, x2, y2), zone, [], w, h,
        )

        region_freq = freq[y1:y2 + 1, x1:x2 + 1]
        pos_freq = region_freq[region_freq > 0]
        confidence = float(pos_freq.mean()) if len(pos_freq) > 0 else 0.0
        frame_fraction = float(region_freq.mean())

        component_mask = (labels[y1:y2 + 1, x1:x2 + 1] == label_id).astype(np.uint8)
        full_mask = np.zeros((h, w, 1), dtype=np.uint8)
        full_mask[y1:y2 + 1, x1:x2 + 1, 0] = component_mask

        candidates.append(
            CleanCandidate(
                id=f"b{len(candidates) + 1}",
                type=target_type,  # type: ignore[arg-type]
                label=_candidate_label(target_type, zone),
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                frame_fraction=frame_fraction,
                reason=f"band fallback: {reason}",
                default_remove=default_remove,
                mask=full_mask,
            )
        )

    return candidates


def _detect_fixed_logo_candidates(
    frames: list[np.ndarray],
    start_index: int = 1,
) -> list[CleanCandidate]:
    if len(frames) < 2:
        return []
    h, w = frames[0].shape[:2]
    zones = ["top-left", "top-right", "bottom-left", "bottom-right"]
    candidates: list[CleanCandidate] = []
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(first_gray, 40, 120)

    for zone in zones:
        x1, y1, x2, y2 = _region_bbox(zone, w, h)
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        crop = edge[y1:y2 + 1, x1:x2 + 1]
        if crop.mean() < 1.5:
            continue
        zone_mask[y1:y2 + 1, x1:x2 + 1] = crop > 0

        stable = True
        for frame in frames[1:min(len(frames), 6)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_crop = cv2.Canny(gray, 40, 120)[y1:y2 + 1, x1:x2 + 1] > 0
            base = crop > 0
            union = np.logical_or(base, next_crop).sum()
            overlap = np.logical_and(base, next_crop).sum()
            if union and overlap / union < 0.45:
                stable = False
                break
        if not stable:
            continue

        zone_mask = cv2.dilate(
            zone_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        candidate = _largest_overlay_candidate(
            zone_mask,
            zone,
            "logo",
            f"fixed edge graphic in {zone}",
            (h, w),
            f"l{start_index + len(candidates)}",
            confidence=0.55,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _detect_translucent_watermark_candidates(
    frames: list[np.ndarray],
    start_index: int = 1,
) -> list[CleanCandidate]:
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    contrast = cv2.absdiff(gray, blur)
    soft = ((contrast > 5) & (contrast < 45)).astype(np.uint8)
    x1, y1, x2, y2 = _region_bbox("center", w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = soft[y1:y2 + 1, x1:x2 + 1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

    candidate = _largest_overlay_candidate(
        mask,
        "center",
        "watermark",
        "possible translucent center watermark",
        (h, w),
        f"w{start_index}",
        confidence=0.45,
    )
    return [candidate] if candidate is not None else []


def _classify_region(
    bbox: tuple[int, int, int, int],
    zone: str,
    text_samples: list[str],
    w: int,
    h: int,
) -> tuple[str, str, bool]:
    """Classify a detected region by position and content patterns."""
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cy = (y1 + y2) / 2
    width_ratio = bw / float(w)
    area_ratio = (bw * bh) / float(w * h)
    text = " ".join(text_samples).strip()

    if text and _TIME_RE.search(text):
        return "timestamp", f"time-like text in {zone}", True
    if text and _WATERMARK_RE.search(text):
        return "watermark", f"watermark-like text in {zone}", True

    height_ratio = bh / float(h)

    if cy > h * 0.50 and width_ratio > 0.15:
        return "subtitle", f"wide bottom text in {zone}", True
    if cy < h * 0.30 and width_ratio > 0.15:
        return "subtitle", f"wide top text in {zone}", True

    # Wide thin text strip below top 25% → likely subtitle regardless of vertical position
    if width_ratio > 0.40 and height_ratio < 0.08 and cy > h * 0.25:
        return "subtitle", f"wide thin text in {zone}", True

    edge = x1 < w * 0.12 or x2 > w * 0.88 or y1 < h * 0.18 or y2 > h * 0.80
    if edge and area_ratio < 0.05:
        return "watermark", f"small persistent edge text in {zone}", True

    return "scene_text", f"text in {zone}", False


DETECT_MODES: dict[str, dict] = {
    "fast": {
        "sample_count": 24,
        "consistency": 0.50,
        "subtitle_fallback": "off",
    },
    "balanced": {
        "sample_count": 50,
        "consistency": 0.40,
        "subtitle_fallback": "light",
    },
    "sensitive": {
        "sample_count": 80,
        "consistency": 0.30,
        "subtitle_fallback": "force",
    },
}


def resolve_detect_params(
    mode: str,
    has_subtitle_target: bool = False,
) -> dict:
    """Resolve a detect-mode name into concrete detection parameters.

    Returns a dict with keys ``sample_count``, ``consistency``, and
    ``subtitle_fallback``.

    When *has_subtitle_target* is ``True``, ``subtitle_fallback`` is
    upgraded to ``"force"`` regardless of mode, because an explicit subtitle
    request signals that subtitle recall matters more than speed.
    """
    if mode not in DETECT_MODES:
        raise ValueError(
            f"Unknown detect mode: {mode!r}. Choose from: {list(DETECT_MODES)}"
        )
    params = dict(DETECT_MODES[mode])
    if has_subtitle_target:
        params["subtitle_fallback"] = "force"
    return params


def detect_clean_candidates(
    video_path: str,
    detector: TextDetector | None = None,
    sample_count: int = 50,
    regions: Iterable[str] | None = None,
    detect_text: bool = True,
    include_logo: bool = False,
    include_translucent_watermark: bool = False,
    consistency: float = 0.4,
    subtitle_fallback: str = "off",
    recognizer: object | None = None,
) -> CleanDetectionResult:
    """Detect removable clean targets from sampled video frames.

    Uses a frequency-map approach: builds a per-pixel frequency map of
    text detections across sampled frames, then finds connected regions
    and classifies each by position and content.

    Args:
        recognizer: Optional callable ``(image_crop) -> str | None``.
            When provided and a detected text box has no ``.text`` field,
            the crop is passed to *recognizer* to fill ``text_samples``.
    """
    if detect_text and detector is None:
        detector = _default_detector()

    indexed = _sample_frames_with_indices(video_path, sample_count)
    if not indexed:
        raise ValueError(f"No frames could be read from: {video_path}")

    all_sample_indices = [idx for idx, _ in indexed]
    frames = [frame for _, frame in indexed]
    h, w = frames[0].shape[:2]
    candidates: list[CleanCandidate] = []
    n_valid = 0
    valid_sample_indices: list[int] = []

    if detect_text and detector is not None:
        freq = np.zeros((h, w), dtype=np.float32)
        all_frame_boxes: list[list[TextBox]] = []
        best_preview_idx = 0
        best_preview_score = -1

        for i, frame in enumerate(frames):
            try:
                boxes = detector.detect(frame)
            except Exception as exc:
                logger.warning("Clean detection failed on sampled frame: %s", exc)
                continue
            n_valid += 1
            all_frame_boxes.append(boxes)
            valid_sample_indices.append(all_sample_indices[i])
            frame_mask = np.zeros((h, w), dtype=np.uint8)
            for box in boxes:
                cv2.fillPoly(frame_mask, [box.points.astype(np.int32)], 1)
            freq += frame_mask

            score = int(frame_mask.sum())
            if score > best_preview_score:
                best_preview_score = score
                best_preview_idx = i

        if n_valid == 0:
            raise RuntimeError("Target detection failed on all sampled frames.")

        freq /= n_valid

        text_mask = (freq >= consistency).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(text_mask)

        raw_regions: list[tuple[int, int, int, int, int]] = []  # (label_id, x1, y1, x2, y2)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            ys, xs = np.where(component > 0)
            if len(xs) < 10:
                continue
            raw_regions.append((label_id, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

        raw_regions.sort(key=lambda b: (b[2], b[1]))

        for label_id, x1, y1, x2, y2 in raw_regions:
            text_samples: list[str] = []
            ocr_crops: list[np.ndarray] = []
            presence_frames: list[int] = []
            for sample_pos, boxes in enumerate(all_frame_boxes):
                overlapped = False
                for box in boxes:
                    if _box_overlaps_bbox(box, (x1, y1, x2, y2), w, h):
                        overlapped = True
                        bx1, by1, bx2, by2 = _bbox(box.points, w, h)
                        if box.text:
                            text_samples.append(box.text)
                        elif recognizer is not None:
                            crop = frames[0][by1:by2 + 1, bx1:bx2 + 1]
                            if crop.size > 0:
                                ocr_crops.append(crop)
                if overlapped:
                    presence_frames.append(valid_sample_indices[sample_pos])

            if not text_samples and ocr_crops and recognizer is not None:
                for crop in ocr_crops[:3]:
                    try:
                        text = recognizer(crop)
                    except Exception:
                        text = None
                    if text:
                        text_samples.append(text)

            zone = _zone((x1, y1, x2, y2), w, h)
            target_type, reason, default_remove = _classify_region(
                (x1, y1, x2, y2), zone, text_samples, w, h,
            )

            region_freq = freq[y1:y2 + 1, x1:x2 + 1]
            pos_freq = region_freq[region_freq > 0]
            confidence = float(pos_freq.mean()) if len(pos_freq) > 0 else 0.0
            frame_fraction = float(region_freq.mean())

            component_mask = (labels[y1:y2 + 1, x1:x2 + 1] == label_id).astype(np.uint8)
            full_mask = np.zeros((h, w, 1), dtype=np.uint8)
            full_mask[y1:y2 + 1, x1:x2 + 1, 0] = component_mask

            candidates.append(
                CleanCandidate(
                    id=f"c{len(candidates) + 1}",
                    type=target_type,  # type: ignore[arg-type]
                    label=_candidate_label(target_type, zone),
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    frame_fraction=frame_fraction,
                    reason=reason,
                    default_remove=default_remove,
                    text_samples=sorted(set(text_samples))[:5],
                    presence_frames=presence_frames,
                    mask=full_mask,
                    detector_backed=True,
                )
            )

    if subtitle_fallback != "off" and detect_text and detector is not None:
        fallback_raw = _band_fallback_detect(
            frames, detector, mode=subtitle_fallback, consistency=consistency,
        )
        main_bboxes = [c.bbox for c in candidates]
        surviving = [
            fc for fc in fallback_raw
            if not any(_iou_bbox(fc.bbox, mb) > 0.3 for mb in main_bboxes)
        ]
        for idx, fc in enumerate(surviving, 1):
            fc.id = f"b{idx}"
            candidates.append(fc)

    next_index = len(candidates) + 1
    if regions:
        candidates.extend(_region_candidates(regions, (h, w), start_index=next_index))
        next_index = len(candidates) + 1
    if include_logo:
        candidates.extend(_detect_fixed_logo_candidates(frames, start_index=next_index))
        next_index = len(candidates) + 1
    if include_translucent_watermark:
        candidates.extend(
            _detect_translucent_watermark_candidates(frames, start_index=next_index)
        )

    return CleanDetectionResult(
        candidates=candidates,
        frame_shape=(h, w),
        sample_indices=valid_sample_indices,
        preview_frame=frames[best_preview_idx].copy() if detect_text and detector is not None else frames[0].copy(),
        detector=detector,
    )


def refine_temporal_presence(
    video_path: str,
    result: CleanDetectionResult,
    candidate_segments: dict[str, Iterable[object]],
    expected_frame_count: int,
    progress=None,
    check_cancelled=None,
) -> list[str]:
    """Densely recheck detector-backed remove candidates inside coarse segments.

    Detection remains sequential and each active video frame is passed to the
    detector once.  A detector error is negative evidence: keeping that frame
    is safer than carrying forward a coarse remove decision.
    """
    detector = result.detector
    candidates = {
        candidate.id: candidate
        for candidate in result.candidates
        if candidate.id in candidate_segments and getattr(
            candidate, "detector_backed", bool(candidate.presence_frames)
        )
    }
    if detector is None or not candidates:
        return []

    segments = {
        candidate_id: [
            (int(segment.start), int(segment.end))
            for segment in candidate_segments[candidate_id]
        ]
        for candidate_id in candidates
    }
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for temporal refinement: {video_path}")

    warnings: list[str] = []
    try:
        frame_index = 0
        while frame_index < expected_frame_count:
            if check_cancelled is not None:
                check_cancelled()
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Temporal refinement decoded {frame_index} frames; "
                    f"expected {expected_frame_count}"
                )
            active = [
                candidate
                for candidate_id, candidate in candidates.items()
                if any(start <= frame_index < end for start, end in segments[candidate_id])
            ]
            if active:
                for candidate in active:
                    candidate.temporal_sample_indices = sorted(set(
                        candidate.temporal_sample_indices or result.sample_indices
                    ) | {frame_index})
                    candidate.presence_frames = [
                        index for index in candidate.presence_frames if index != frame_index
                    ]
                try:
                    boxes = detector.detect(frame)
                except Exception as exc:  # safety boundary: failed frame remains keep.
                    warnings.append(
                        f"temporal refinement failed at frame {frame_index}; keeping frame: {exc}"
                    )
                else:
                    for candidate in active:
                        if any(
                            _box_overlaps_bbox(box, candidate.bbox, frame.shape[1], frame.shape[0])
                            for box in boxes
                        ):
                            candidate.presence_frames.append(frame_index)
            frame_index += 1
            if progress is not None:
                progress(frame_index, expected_frame_count)
    finally:
        cap.release()

    for candidate in candidates.values():
        candidate.presence_frames = sorted(set(candidate.presence_frames))
    return warnings


def select_clean_candidates(
    candidates: Iterable[CleanCandidate],
    targets: Iterable[str] | None = None,
    intent: str | None = None,
) -> list[CleanCandidate]:
    """Select candidates for removal by default rules or requested targets."""
    candidate_list = list(candidates)
    normalized = {normalize_target(t) for t in (targets or []) if t}
    if not normalized:
        selected = [c for c in candidate_list if c.default_remove]
    else:
        selected = [c for c in candidate_list if normalize_target(c.type) in normalized]
    if intent:
        selected = select_candidates_by_intent(candidate_list, intent, selected)
    return selected


def select_candidates_by_intent(
    candidates: Iterable[CleanCandidate],
    intent: str,
    fallback: Iterable[CleanCandidate] | None = None,
) -> list[CleanCandidate]:
    """Select candidates using conservative local intent rules."""
    candidate_list = list(candidates)
    keep_mode = _mentions_any(intent, _KEEP_WORDS)
    remove_mode = _mentions_any(intent, _REMOVE_WORDS)
    targets = _mentioned_targets(intent)
    zones = _mentioned_zones(intent)
    fallback_list = list(fallback or [])

    if keep_mode and not remove_mode:
        keep_targets = targets
        keep_zones = zones
        return [
            c for c in fallback_list
            if not _candidate_matches_intent(c, keep_targets, keep_zones, intent)
        ]

    keep_targets, keep_zones = _mentioned_after_words(intent, _KEEP_WORDS)
    remove_targets, remove_zones = _mentioned_after_words(intent, _REMOVE_WORDS)
    if keep_mode and remove_mode:
        targets = remove_targets or (targets - keep_targets)
        zones = remove_zones or zones

    if not targets and not zones:
        return fallback_list

    selected = [
        c for c in candidate_list
        if _candidate_matches_intent(c, targets, zones, intent)
    ]
    if keep_targets or keep_zones:
        selected = [
            c for c in selected
            if not _candidate_matches_intent(c, keep_targets, keep_zones, intent)
        ]
    return selected


def mask_from_candidates(
    candidates: Iterable[CleanCandidate],
    frame_shape: tuple[int, int],
    feather_radius: int = 0,
) -> np.ndarray:
    """Merge candidate masks into the mask format used by inpainting tasks.

    Candidates are merged first (``np.maximum``). When ``feather_radius > 0``,
    the *merged* mask's outer boundary is then blurred with a Gaussian so the
    inpainting blend produces a soft seam instead of a hard edge. The interior
    (mask >= 1.0) is pinned back to full opacity so thin candidates (e.g. a
    4-pixel-tall subtitle band) are not eroded inward.

    Earlier revisions feathered each bbox-only candidate independently. That
    was a no-op in practice because the clean-task detector emits a full-image
    ``mask`` for every candidate, so the bbox-only branch never ran. Feathering
    the final merged mask fixes this and also softens seams around candidates
    whose detector-provided mask already has a hard edge.

    Output dtype is ``float32`` with values in ``[0, 1]`` when
    ``feather_radius > 0``, and ``uint8`` with values in ``{0, 1}`` when
    ``feather_radius == 0`` (backwards-compatible with the eval IoU path,
    which compares against binary ground-truth masks).
    """
    h, w = frame_shape
    # Always merge candidates as float32 internally; cast down to uint8 only
    # for the legacy binary path at the end.
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for candidate in candidates:
        if candidate.mask is not None:
            mask = np.maximum(mask, candidate.mask.astype(np.float32))
            continue
        x1, y1, x2, y2 = candidate.bbox
        mask[y1:y2 + 1, x1:x2 + 1, 0] = 1.0

    if feather_radius > 0:
        # Feather the OUTER boundary of the merged mask: blur it, then pin
        # the original interior (mask >= 1.0) back to full opacity so the
        # filled region's core stays 1.0 and only the edge gets a soft
        # falloff into the surrounding frame.
        flat = mask[:, :, 0]
        k = max(3, feather_radius * 2 + 1)
        blurred = cv2.GaussianBlur(flat, (k, k), feather_radius / 2.0)
        flat = np.where(flat >= 1.0, 1.0, blurred)
        return np.clip(flat[:, :, None], 0.0, 1.0)

    return np.clip(mask, 0, 1).astype(np.uint8)


def write_clean_artifacts(
    result: CleanDetectionResult,
    selected: Iterable[CleanCandidate],
    output_dir: str,
) -> dict[str, str]:
    """Write candidate JSON and preview image artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    selected_ids = {c.id for c in selected}
    candidates_path = os.path.join(output_dir, "clean_candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "candidates": [
                    {
                        **candidate.to_dict(),
                        "selected": candidate.id in selected_ids,
                    }
                    for candidate in result.candidates
                ]
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    preview_path = os.path.join(output_dir, "clean_preview.jpg")
    if result.preview_frame is not None:
        preview = result.preview_frame.copy()
        for candidate in result.candidates:
            x1, y1, x2, y2 = candidate.bbox
            color = (0, 200, 0) if candidate.id in selected_ids else (0, 165, 255)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview,
                f"{candidate.id}:{candidate.type}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(preview_path, preview):
            raise OSError(f"Failed to write image: {preview_path}")

    return {"candidates": candidates_path, "preview": preview_path}


def _default_detector() -> DBNetDetector:
    """Create the default detector with auto-downloaded weights."""
    from videowipe.weights import ensure_weight

    weight = ensure_weight("ppocrv5_det_mob.onnx", version="v0.1.0")
    return DBNetDetector(weight)
