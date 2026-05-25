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
    mask: np.ndarray | None = field(default=None, repr=False)

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
        }


@dataclass
class CleanDetectionResult:
    """Result of clean-target detection."""

    candidates: list[CleanCandidate]
    frame_shape: tuple[int, int]
    preview_frame: np.ndarray | None = field(default=None, repr=False)


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


def _classify_text_box(
    box: TextBox,
    bbox: tuple[int, int, int, int],
    w: int,
    h: int,
) -> tuple[str, str, bool]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    text = (box.text or "").strip()
    zone = _zone(bbox, w, h)
    area_ratio = (bw * bh) / float(w * h)
    width_ratio = bw / float(w)
    edge = x1 < w * 0.12 or x2 > w * 0.88 or y1 < h * 0.18 or y2 > h * 0.75

    if text and _TIME_RE.search(text):
        return "timestamp", f"time-like text in {zone}", True
    if text and _WATERMARK_RE.search(text):
        return "watermark", f"watermark-like text in {zone}", True
    if y1 > h * 0.58 and width_ratio > 0.18:
        return "subtitle", f"wide bottom text in {zone}", True
    if y2 < h * 0.25 and width_ratio > 0.18:
        return "subtitle", f"wide top text in {zone}", True
    if edge and area_ratio < 0.04:
        return "watermark", f"small persistent edge text in {zone}", True
    return "scene_text", f"scene text in {zone}", False


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


def detect_clean_candidates(
    video_path: str,
    detector: TextDetector | None = None,
    sample_count: int = 30,
    regions: Iterable[str] | None = None,
    detect_text: bool = True,
    include_logo: bool = False,
    include_translucent_watermark: bool = False,
) -> CleanDetectionResult:
    """Detect removable clean targets from sampled video frames."""
    if detect_text and detector is None:
        detector = _default_detector()

    frames = _sample_frames(video_path, sample_count)
    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    h, w = frames[0].shape[:2]
    groups: dict[tuple[str, str], dict] = {}
    n_valid = 0

    if detect_text and detector is not None:
        for frame in frames:
            try:
                boxes = detector.detect(frame)
            except Exception as exc:
                logger.warning("Clean detection failed on sampled frame: %s", exc)
                continue
            n_valid += 1
            seen_keys = set()
            for box in boxes:
                bbox = _bbox(box.points, w, h)
                target_type, reason, default_remove = _classify_text_box(box, bbox, w, h)
                zone = _zone(bbox, w, h)
                key = (target_type, zone)
                if key not in groups:
                    groups[key] = {
                        "mask": np.zeros((h, w), dtype=np.uint8),
                        "conf": [],
                        "texts": [],
                        "seen": 0,
                        "reason": reason,
                        "default_remove": default_remove,
                    }
                cv2.fillPoly(groups[key]["mask"], [box.points.astype(np.int32)], 1)
                groups[key]["conf"].append(float(box.confidence))
                if box.text:
                    groups[key]["texts"].append(box.text)
                seen_keys.add(key)
            for key in seen_keys:
                groups[key]["seen"] += 1

    if detect_text and n_valid == 0:
        raise RuntimeError("Target detection failed on all sampled frames.")

    candidates: list[CleanCandidate] = []
    for idx, ((target_type, zone), data) in enumerate(sorted(groups.items()), 1):
        raw_mask = data["mask"].astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.dilate(raw_mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        frame_fraction = data["seen"] / float(n_valid)
        candidates.append(
            CleanCandidate(
                id=f"c{idx}",
                type=target_type,  # type: ignore[arg-type]
                label=_candidate_label(target_type, zone),
                bbox=bbox,
                confidence=float(np.mean(data["conf"])) if data["conf"] else 0.0,
                frame_fraction=frame_fraction,
                reason=data["reason"],
                default_remove=bool(data["default_remove"]),
                text_samples=sorted(set(data["texts"]))[:5],
                mask=mask[:, :, None],
            )
        )

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
        preview_frame=frames[0].copy(),
    )


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
) -> np.ndarray:
    """Merge candidate masks into the mask format used by inpainting tasks."""
    h, w = frame_shape
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    for candidate in candidates:
        if candidate.mask is not None:
            mask = np.maximum(mask, candidate.mask.astype(np.uint8))
            continue
        x1, y1, x2, y2 = candidate.bbox
        mask[y1:y2 + 1, x1:x2 + 1, 0] = 1
    return mask


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
        cv2.imwrite(preview_path, preview)

    return {"candidates": candidates_path, "preview": preview_path}


def _default_detector() -> DBNetDetector:
    """Create the default detector with auto-downloaded weights."""
    from videowipe.weights import ensure_weight

    weight = ensure_weight("ppocrv5_det_mob.onnx", version="v0.1.0")
    return DBNetDetector(weight)
