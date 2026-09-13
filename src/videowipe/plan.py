"""Serializable cleanup plans with deterministic spatial and temporal masks.

V1 stores a static mask and active intervals per target. V2 adds protection.
V3 optionally records local subtitle rectangles in disjoint half-open frame
intervals; missing spatial evidence means no removal, never a static fallback.
The NPZ stores the exact spatial union for overview/crop selection. Execution
rasterizes the active rectangles, then feathers and applies protection.
Legacy v1/v2 plans preserve their original behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from bisect import bisect_right
from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from videowipe.errors import InvalidInputError

SCHEMA_VERSION = 1
PLAN_KIND = "wipe_plan"
JSON_FILENAME = "wipe_plan.json"
MASK_FILENAME = "wipe_plan_masks.npz"

_ACTION_VALUES = ("remove", "keep", "protect")
_TOP_REGION_FRACTION = 0.30      # mirrors detect._classify_region's top-subtitle threshold
_PERSISTENT_PRESENCE = 0.80     # plan Rule 6: persistent-overlay safety default
_COARSE_GAP_SECONDS = 2.0       # plan Rule 8: temporal-coarseness warning threshold


# ── helpers ──────────────────────────────────────────────────────────────────

def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _assert_bare_name(child: str) -> None:
    """Reject anything that is not a bare basename (no separator/absolute/``..``)."""
    if not child or child in (".", "..") or os.path.isabs(child) or child != os.path.basename(child):
        raise InvalidInputError(f"unsafe plan-relative path: {child!r}")


def _child_path(parent: str, child: str) -> str:
    """Join ``parent``/``child`` and refuse any path that escapes *parent*.

    First rejects non-basename names (absolute, separators, ``.``/``..``).
    Then resolves both *parent* and the joined *child* to their real paths and
    verifies the resolved child still lives inside the resolved parent via
    :func:`os.path.commonpath`. This catches an existing sidecar symlink that
    points outside the plan directory before any read/write/hash/``np.load``.
    """
    _assert_bare_name(child)
    joined = os.path.join(parent, child)
    resolved_parent = os.path.realpath(parent)
    resolved_child = os.path.realpath(joined)
    try:
        common = os.path.commonpath([resolved_parent, resolved_child])
    except ValueError:
        common = ""
    if common != resolved_parent:
        raise InvalidInputError(
            f"unsafe plan-relative path escapes plan directory: {child!r}"
        )
    return joined


# ── data types ───────────────────────────────────────────────────────────────

@dataclass
class Source:
    """Identity of the source video the plan is bound to."""
    basename: str
    sha256: str
    width: int
    height: int
    fps: float
    frame_count: int

    def to_dict(self) -> dict:
        return {
            "basename": self.basename,
            "sha256": self.sha256,
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "frame_count": int(self.frame_count),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Source:
        return cls(
            basename=str(d["basename"]),
            sha256=str(d["sha256"]),
            width=int(d["width"]),
            height=int(d["height"]),
            fps=float(d["fps"]),
            frame_count=int(d["frame_count"]),
        )


@dataclass
class Segment:
    """Half-open frame interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        self.start = int(self.start)
        self.end = int(self.end)

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end

    def to_dict(self) -> list[int]:
        return [self.start, self.end]

    @classmethod
    def from_value(cls, value) -> Segment:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int for v in value):
            raise InvalidInputError("segments must contain integer frame pairs")
        return cls(start=value[0], end=value[1])


@dataclass
class Track:
    """One addressable candidate in a plan."""

    id: str
    type: str
    label: str
    action: str
    bbox: tuple[int, int, int, int]
    confidence: float
    presence_fraction: float
    decision_reason: str
    segments: list[Segment]
    mask_key: str
    # In-memory only; never serialized to JSON. Loaded from the sidecar NPZ.
    mask: np.ndarray | None = field(default=None, repr=False)

    spatial_segments: list[dict] | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            **({"spatial_segments": deepcopy(self.spatial_segments)} if self.spatial_segments is not None else {}),
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "action": self.action,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "presence_fraction": round(float(self.presence_fraction), 4),
            "decision_reason": self.decision_reason,
            "segments": [s.to_dict() for s in self.segments],
            "mask_key": self.mask_key,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Track:
        return cls(
            id=str(d["id"]),
            type=str(d["type"]),
            label=str(d["label"]),
            action=str(d["action"]),
            # Preserve JSON values until validate_plan() has checked that all
            # four coordinates are genuine integers. Coercing here would turn
            # a tampered 90.5 into 90 before the trust-boundary validator sees it.
            bbox=tuple(d["bbox"]),
            confidence=float(d["confidence"]),
            presence_fraction=float(d["presence_fraction"]),
            decision_reason=str(d["decision_reason"]),
            segments=[Segment.from_value(s) for s in d["segments"]],
            mask_key=str(d["mask_key"]),
            spatial_segments=d.get("spatial_segments"),
        )

    @property
    def full_video(self) -> bool:
        """True if this track is active on every frame."""
        return len(self.segments) == 1 and self.segments[0].start == 0


@dataclass
class MaskAsset:
    filename: str
    sha256: str

    def to_dict(self) -> dict:
        return {"filename": self.filename, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MaskAsset:
        return cls(filename=str(d["filename"]), sha256=str(d["sha256"]))


@dataclass
class TemporalResolution:
    """Honest statement of the plan's temporal precision."""
    max_gap_frames: int
    max_gap_seconds: float
    max_boundary_error_frames: int

    def to_dict(self) -> dict:
        return {
            "max_gap_frames": int(self.max_gap_frames),
            "max_gap_seconds": round(float(self.max_gap_seconds), 4),
            "max_boundary_error_frames": int(self.max_boundary_error_frames),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TemporalResolution:
        return cls(
            max_gap_frames=int(d["max_gap_frames"]),
            max_gap_seconds=float(d["max_gap_seconds"]),
            max_boundary_error_frames=int(d["max_boundary_error_frames"]),
        )


@dataclass
class WipePlan:
    kind: str
    schema_version: int
    source: Source
    request: dict
    temporal_resolution: TemporalResolution
    mask_asset: MaskAsset
    tracks: list[Track]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "request": dict(self.request),
            "temporal_resolution": self.temporal_resolution.to_dict(),
            "mask_asset": self.mask_asset.to_dict(),
            "tracks": [t.to_dict() for t in self.tracks],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> WipePlan:
        return cls(
            kind=str(d["kind"]),
            schema_version=d["schema_version"],
            source=Source.from_dict(d["source"]),
            request=dict(d.get("request", {})),
            temporal_resolution=TemporalResolution.from_dict(d["temporal_resolution"]),
            mask_asset=MaskAsset.from_dict(d["mask_asset"]),
            tracks=[Track.from_dict(t) for t in d["tracks"]],
            warnings=list(d.get("warnings", [])),
        )

    @property
    def remove_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.action == "remove"]

    @property
    def keep_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.action == "keep"]


# ── source / temporal helpers ────────────────────────────────────────────────

def compute_source(video_path: str) -> Source:
    """Read video identity (dimensions, fps, frame_count) and SHA-256."""
    if not os.path.isfile(video_path):
        raise InvalidInputError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InvalidInputError(f"Cannot open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Round to match tasks.base.read_frame_info (int(x + 0.5)); truncating
        # here would make the plan's frame_count one short of STTN's loop bound
        # for non-integer CAP_PROP_FRAME_COUNT, leaving the trailing frame
        # un-inpainted.
        frame_count = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5)
    finally:
        cap.release()
    return Source(
        basename=os.path.basename(video_path),
        sha256=_sha256_file(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def segments_from_presence(
    sample_indices: Sequence[int],
    presence_set: set[int],
    frame_count: int,
) -> list[Segment]:
    """Compress per-sample presence into half-open segments.

    Each frame takes the presence state of its **nearest sampled frame**; the
    switch point is the midpoint between adjacent samples (ties resolve to the
    earlier sample), so the worst-case boundary error is half the max sample
    gap. Returns sorted, non-overlapping ``[start, end)`` segments.
    """
    samples = sorted(int(s) for s in sample_indices if 0 <= int(s) < frame_count)
    if not samples or frame_count <= 0:
        return []
    present_at_sample = np.fromiter(
        (1 if s in presence_set else 0 for s in samples), dtype=np.int8
    )
    samples_arr = np.asarray(samples, dtype=np.int64)
    frames = np.arange(frame_count, dtype=np.int64)
    pos = np.searchsorted(samples_arr, frames)
    lo = np.clip(pos - 1, 0, len(samples) - 1)
    hi = np.clip(pos, 0, len(samples) - 1)
    dist_lo = frames - samples_arr[lo]
    dist_hi = samples_arr[hi] - frames
    # ties (dist_hi == dist_lo) keep the earlier sample -> midpoint switch
    choose_hi = (pos < len(samples)) & (dist_hi < dist_lo)
    nearest = np.where(choose_hi, hi, lo)
    present = present_at_sample[nearest].astype(bool)

    segments: list[Segment] = []
    start: int | None = None
    for f in range(frame_count):
        if present[f] and start is None:
            start = f
        elif not present[f] and start is not None:
            segments.append(Segment(start, f))
            start = None
    if start is not None:
        segments.append(Segment(start, frame_count))
    return segments


def _temporal_resolution(
    sample_indices: Sequence[int], frame_count: int, fps: float
) -> TemporalResolution:
    s = sorted(int(x) for x in sample_indices)
    if len(s) < 2 or frame_count <= 0:
        return TemporalResolution(0, 0.0, 0)
    gaps = [s[i] - s[i - 1] for i in range(1, len(s))]
    max_gap = max(gaps)
    seconds = (max_gap / fps) if fps > 0 else 0.0
    return TemporalResolution(
        max_gap_frames=int(max_gap),
        max_gap_seconds=seconds,
        max_boundary_error_frames=int(max_gap // 2),
    )


# ── action resolution + plan construction ────────────────────────────────────

def _resolve_action(
    candidate,
    presence_fraction: float,
    height: int,
    explicit_remove_ids: set[str],
    explicit_keep_ids: set[str],
    loaded_actions: Mapping[str, str],
) -> tuple[str, str]:
    """Return ``(action, decision_reason)`` with fixed precedence (plan Rule 7).

    (a) human/loaded plan action >
    (b) explicit selection for removal >
    (c) explicit selection for keep (everything not chosen when the user made
        a genuine decision) >
    (d) persistent top-overlay safety default (keep) >
    (e) detector default_remove.
    """
    cid = candidate.id
    if cid in loaded_actions:
        action = loaded_actions[cid]
        if action not in _ACTION_VALUES:
            raise InvalidInputError(
                f"loaded action {action!r} for track {cid} not in {_ACTION_VALUES}"
            )
        return action, f"loaded-plan:{action}"
    if cid in explicit_remove_ids:
        return "remove", "explicit-selection"
    if cid in explicit_keep_ids:
        return "keep", "explicit-keep"
    _x1, y1, _x2, y2 = candidate.bbox
    cy = (y1 + y2) / 2.0
    if cy < _TOP_REGION_FRACTION * height and presence_fraction >= _PERSISTENT_PRESENCE:
        return "keep", "safety:persistent-top-overlay"
    return ("remove" if candidate.default_remove else "keep"), "default-rule"


def _candidate_mask(candidate, frame_shape: tuple[int, int]) -> np.ndarray:
    """Concrete 2D uint8 spatial mask: the candidate's mask, or a filled bbox."""
    h, w = frame_shape
    if getattr(candidate, "mask", None) is not None:
        arr = np.asarray(candidate.mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.uint8)
    x1, y1, x2, y2 = candidate.bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = 1
    return mask


def build_wipe_plan(
    candidates: Sequence,
    sample_indices: Sequence[int],
    n_valid: int,
    source: Source,
    frame_shape: tuple[int, int],
    *,
    request: Mapping[str, Any] | None = None,
    explicit_remove_ids: set[str] | None = None,
    explicit_keep_ids: set[str] | None = None,
    loaded_actions: Mapping[str, str] | None = None,
) -> WipePlan:
    """Build a validated :class:`WipePlan` from detection outputs.

    ``candidates`` are duck-typed objects with ``id, type, label, bbox,
    confidence, default_remove, mask`` and (optionally) ``presence_frames`` —
    the real video frame indices where the candidate was present in a sampled
    frame. ``sample_indices`` are the real frame indices of the successful
    samples (same space as ``presence_frames``); ``n_valid`` is the count used
    to normalize ``presence_fraction``.
    """
    height, _width = frame_shape
    frame_count = source.frame_count
    explicit_remove_ids = set(explicit_remove_ids or set())
    explicit_keep_ids = set(explicit_keep_ids or set())
    loaded_actions = dict(loaded_actions or {})
    samples = sorted(int(x) for x in (sample_indices or []))

    tracks: list[Track] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c.id in seen:
            raise InvalidInputError(f"Duplicate candidate id: {c.id}")
        seen.add(c.id)

        presence_frames = set(getattr(c, "presence_frames", []) or [])
        candidate_samples = sorted(set(
            int(x) for x in (getattr(c, "temporal_sample_indices", []) or [])
        ))
        effective_samples = candidate_samples or samples
        effective_n_valid = len(candidate_samples) if candidate_samples else n_valid
        presence_fraction = (
            len(presence_frames) / float(effective_n_valid)
            if effective_n_valid > 0 else 0.0
        )

        if candidate_samples:
            segs = segments_from_presence(effective_samples, presence_frames, frame_count)
            seg_note = (
                f"{len(segs)} segment(s) from {len(presence_frames)}/"
                f"{effective_n_valid} samples"
            )
        elif presence_frames:
            segs = segments_from_presence(effective_samples, presence_frames, frame_count)
            seg_note = f"{len(segs)} segment(s) from {len(presence_frames)}/{effective_n_valid} samples"
        else:
            segs = [Segment(0, frame_count)] if frame_count > 0 else []
            seg_note = "full-video (no per-frame evidence)"
            warnings.append(
                f"track {c.id} ({c.label}) has no per-frame presence evidence; "
                f"using full-video segment"
            )

        action, decision = _resolve_action(
            c, presence_fraction, height, explicit_remove_ids, explicit_keep_ids, loaded_actions
        )
        tracks.append(
            Track(
                id=c.id,
                type=str(c.type),
                label=c.label,
                action=action,
                bbox=tuple(int(v) for v in c.bbox),
                confidence=float(c.confidence),
                presence_fraction=presence_fraction,
                decision_reason=f"{decision}; segments: {seg_note}",
                segments=segs,
                mask_key=c.id,
                mask=_candidate_mask(c, frame_shape),
            )
        )

    temporal = _temporal_resolution(samples, frame_count, source.fps)
    if temporal.max_gap_seconds > _COARSE_GAP_SECONDS:
        warnings.append(
            f"coarse temporal resolution: max sample gap "
            f"{temporal.max_gap_seconds:.2f}s > {_COARSE_GAP_SECONDS}s threshold; "
            f"temporal boundaries may be imprecise"
        )

    plan = WipePlan(
        kind=PLAN_KIND,
        schema_version=SCHEMA_VERSION,
        source=source,
        request=dict(request or {}),
        temporal_resolution=temporal,
        mask_asset=MaskAsset(filename=MASK_FILENAME, sha256=""),
        tracks=tracks,
        warnings=warnings,
    )
    validate_plan(plan, frame_shape=frame_shape)
    return plan


def build_refined_wipe_plan(
    video_path: str,
    result: Any,
    source: Source,
    *,
    refine: bool,
    request: Mapping[str, Any] | None = None,
    explicit_remove_ids: set[str] | None = None,
    explicit_keep_ids: set[str] | None = None,
    loaded_actions: Mapping[str, str] | None = None,
    progress: Any = None,
    check_cancelled: Any = None,
) -> WipePlan:
    """Compatibility wrapper for the planning-owned refinement flow."""
    from videowipe.planning import _finalize_result

    return _finalize_result(
        video_path,
        result,
        source,
        refine=refine,
        request=request,
        explicit_remove_ids=explicit_remove_ids,
        explicit_keep_ids=explicit_keep_ids,
        loaded_actions=loaded_actions,
        progress=progress,
        check_cancelled=check_cancelled,
    )


# ── validation ───────────────────────────────────────────────────────────────

def validate_plan(
    plan: WipePlan,
    *,
    frame_shape: tuple[int, int] | None = None,
    require_remove: bool = False,
) -> None:
    """Raise :class:`InvalidInputError` on any structural violation."""
    if plan.kind != PLAN_KIND:
        raise InvalidInputError(f"plan.kind must be {PLAN_KIND!r}, got {plan.kind!r}")
    if type(plan.schema_version) is not int or plan.schema_version not in (1, 2, 3):
        raise InvalidInputError(
            f"unsupported schema_version {plan.schema_version} (expected 1, 2 or 3)"
        )
    if plan.mask_asset.filename != MASK_FILENAME:
        raise InvalidInputError(
            f"mask_asset.filename must be {MASK_FILENAME!r}, got {plan.mask_asset.filename!r}"
        )
    _assert_bare_name(plan.mask_asset.filename)  # name-only safety check

    height = plan.source.height
    width = plan.source.width
    frame_count = plan.source.frame_count
    if width <= 0 or height <= 0 or frame_count < 0:
        raise InvalidInputError(
            f"invalid source dims: {width}x{height}@{frame_count}"
        )
    if frame_shape is not None and frame_shape != (height, width):
        raise InvalidInputError(
            f"plan source {height}x{width} disagrees with detection {frame_shape[0]}x{frame_shape[1]}"
        )

    ids: set[str] = set()
    mask_keys: set[str] = set()
    has_remove = False
    for t in plan.tracks:
        if not t.id:
            raise InvalidInputError("track id must be non-empty")
        if t.id in ids:
            raise InvalidInputError(f"duplicate track id: {t.id}")
        ids.add(t.id)
        if not t.mask_key:
            raise InvalidInputError(f"track {t.id}: mask_key must be non-empty")
        if t.mask_key in mask_keys:
            raise InvalidInputError(f"duplicate mask_key: {t.mask_key}")
        mask_keys.add(t.mask_key)
        if t.action not in _ACTION_VALUES:
            raise InvalidInputError(
                f"track {t.id}: action {t.action!r} not in {_ACTION_VALUES}"
            )
        if t.action == "protect" and plan.schema_version < 2:
            raise InvalidInputError("protect tracks require schema_version 2")
        if t.action == "protect" and (t.mask is not None and not np.any(t.mask)):
            raise InvalidInputError(f"track {t.id}: protect mask must be non-empty")
        if len(t.bbox) != 4 or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or int(v) < 0
            for v in t.bbox
        ):
            raise InvalidInputError(f"track {t.id}: bbox must be 4 non-negative ints")
        x1, y1, x2, y2 = t.bbox
        if x2 < x1 or y2 < y1:
            raise InvalidInputError(f"track {t.id}: bbox {t.bbox} is inverted")
        if x2 >= width or y2 >= height:
            raise InvalidInputError(
                f"track {t.id}: bbox {t.bbox} exceeds source {width}x{height}"
            )

        # Execution requires a precise mask on every remove track. A plan read
        # metadata-only (load_masks=False) may carry mask=None for inspection,
        # but it cannot be executed.
        if require_remove and t.action in ("remove", "protect") and t.mask is None:
            raise InvalidInputError(
                f"track {t.id}: remove track has no precise mask; cannot execute"
            )
        if t.mask is not None:
            arr = np.asarray(t.mask)
            if arr.ndim not in (2, 3):
                raise InvalidInputError(
                    f"track {t.id}: mask must be 2D/3D, got shape {arr.shape}"
                )
            if arr.ndim == 3 and arr.shape[2] != 1:
                raise InvalidInputError(
                    f"track {t.id}: 3D mask must have one channel, got shape {arr.shape}"
                )
            if (arr.shape[0], arr.shape[1]) != (height, width):
                raise InvalidInputError(
                    f"track {t.id}: mask shape {arr.shape[:2]} != source {height}x{width}"
                )
            if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
                raise InvalidInputError(
                    f"track {t.id}: mask dtype {arr.dtype} must be integer/bool"
                )

        if t.spatial_segments is not None:
            if plan.schema_version != 3 or t.action == "protect":
                raise InvalidInputError("spatial segments require schema_version 3 and a non-protect track")
            if not isinstance(t.spatial_segments, list) or len(t.spatial_segments) > frame_count:
                raise InvalidInputError("invalid spatial segments")
            end = 0
            for row in t.spatial_segments:
                if not isinstance(row, dict) or set(row) != {"start", "end", "boxes"}:
                    raise InvalidInputError("invalid spatial segment fields")
                if (type(row["start"]) is not int or type(row["end"]) is not int
                        or not end <= row["start"] < row["end"] <= frame_count):
                    raise InvalidInputError("spatial segments must be ordered, disjoint frame intervals")
                end = row["end"]
                if not isinstance(row["boxes"], list) or not 1 <= len(row["boxes"]) <= 64:
                    raise InvalidInputError("spatial segment requires 1-64 boxes")
                for box in row["boxes"]:
                    if (not isinstance(box, (list, tuple)) or len(box) != 4
                            or any(type(v) is not int for v in box)
                            or not 0 <= box[0] <= box[2] < width
                            or not 0 <= box[1] <= box[3] < height):
                        raise InvalidInputError("spatial box exceeds source or has invalid coordinates")
            if t.mask is not None and not np.array_equal(
                np.asarray(t.mask).reshape(height, width) > 0,
                _spatial_mask(t, (height, width)) > 0,
            ):
                raise InvalidInputError("spatial union disagrees with track mask")

        if t.action in ("remove", "protect") and not t.segments:
            raise InvalidInputError(
                f"track {t.id}: {t.action} track must have at least one segment"
            )
        prev_end = 0
        for seg in t.segments:
            if not (0 <= seg.start < seg.end <= frame_count):
                raise InvalidInputError(
                    f"track {t.id}: segment [{seg.start},{seg.end}) violates "
                    f"0<=start<end<={frame_count}"
                )
            if seg.start < prev_end:
                raise InvalidInputError(
                    f"track {t.id}: segments must be sorted and non-overlapping"
                )
            prev_end = seg.end
        if t.action == "remove":
            has_remove = True

    if require_remove and not has_remove:
        raise InvalidInputError("plan has no remove track; nothing to inpaint")


# ── serialization ────────────────────────────────────────────────────────────

def save_wipe_plan(plan: WipePlan, output_dir: str) -> tuple[str, str]:
    """Write ``wipe_plan.json`` + ``wipe_plan_masks.npz`` side-by-side.

    Returns ``(json_path, npz_path)``. Masks are stored as raw ``uint8``
    arrays keyed by ``track.mask_key`` (sorted for determinism). The NPZ
    SHA-256 is computed after writing and embedded in the JSON.
    """
    validate_plan(plan)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, JSON_FILENAME)
    npz_path = _child_path(output_dir, plan.mask_asset.filename)

    masks: dict[str, np.ndarray] = {}
    for t in sorted(plan.tracks, key=lambda x: x.id):
        if t.mask is None:
            raise InvalidInputError(f"track {t.id} has no mask to save")
        arr = np.asarray(t.mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        masks[t.mask_key] = (arr > 0).astype(np.uint8)
    np.savez_compressed(npz_path, **masks)

    plan.mask_asset = MaskAsset(
        filename=plan.mask_asset.filename, sha256=_sha256_file(npz_path)
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(plan.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return json_path, npz_path


def load_wipe_plan(
    plan_path: str,
    *,
    video_path: str | None = None,
    load_masks: bool = True,
) -> WipePlan:
    """Load and validate a plan from ``wipe_plan.json`` (+ sidecar NPZ).

    With ``video_path`` set, the source SHA-256/width/height/frame_count are
    re-derived and must match (executing against a different video is
    rejected); ``require_remove`` is then enforced. Masks are loaded with
    ``allow_pickle=False``.
    """
    plan_path = os.path.realpath(plan_path)
    if not os.path.isfile(plan_path):
        raise InvalidInputError(f"plan not found: {plan_path}")
    plan_dir = os.path.dirname(plan_path)
    with open(plan_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        plan = WipePlan.from_dict(data)
    except KeyError as exc:
        raise InvalidInputError(f"plan JSON missing key: {exc}") from exc

    npz_path = _child_path(plan_dir, plan.mask_asset.filename)
    if not os.path.isfile(npz_path):
        raise InvalidInputError(f"mask asset not found: {plan.mask_asset.filename}")
    actual_sha = _sha256_file(npz_path)
    if actual_sha != plan.mask_asset.sha256:
        raise InvalidInputError("mask_asset sha256 mismatch (NPZ modified or corrupted)")

    if video_path is not None:
        actual = compute_source(video_path)
        if actual.sha256 != plan.source.sha256:
            raise InvalidInputError(
                "source sha256 mismatch: plan is bound to a different video"
            )
        if (actual.width, actual.height, actual.frame_count) != (
            plan.source.width,
            plan.source.height,
            plan.source.frame_count,
        ):
            raise InvalidInputError("source width/height/frame_count mismatch")

    if load_masks:
        with np.load(npz_path, allow_pickle=False) as npz:
            available = set(npz.files)
            for t in plan.tracks:
                if t.mask_key in available:
                    t.mask = npz[t.mask_key]
                elif t.mask is None:
                    raise InvalidInputError(
                        f"mask key {t.mask_key!r} not found in NPZ"
                    )

    validate_plan(plan, require_remove=video_path is not None)
    return plan


def _spatial_mask(track, shape, frame_index=None, *, active_only=False):
    mask = np.zeros(shape, dtype=np.uint8)
    for row in track.spatial_segments:
        if frame_index is not None and not row["start"] <= frame_index < row["end"]:
            continue
        if active_only and not any(s.start < row["end"] and s.end > row["start"] for s in track.segments):
            continue
        for x1, y1, x2, y2 in row["boxes"]:
            mask[y1:y2+1, x1:x2+1] = 1
    return mask


def is_temporal(plan: WipePlan) -> bool:
    """True if any remove or protect track is not active on the entire video.

    File-based inpainters (external command, ProPainter) consume one static
    mask PNG and cannot honor temporal segments; the engine uses this to
    reject such plans instead of silently flattening them.
    """
    frame_count = plan.source.frame_count
    for t in plan.tracks:
        if t.action == "keep":
            continue
        if t.spatial_segments is not None:
            return True
        if not (len(t.segments) == 1 and t.segments[0] == Segment(0, frame_count)):
            return True
    return False


def _execution_tracks(plan: WipePlan) -> list[tuple[Track, np.ndarray]]:
    """Return remove/protect tracks with precise masks normalized to binary ``uint8``."""
    validate_plan(plan)
    prepared = []
    for track in plan.tracks:
        if track.action == "keep":
            continue
        if track.mask is None:
            raise InvalidInputError(
                f"track {track.id}: remove track has no precise mask; cannot execute"
            )
        arr = np.asarray(track.mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        normalized = (_spatial_mask(track, (plan.source.height, plan.source.width), active_only=True)
                      if track.spatial_segments is not None else (arr > 0).astype(np.uint8))
        prepared.append((track, normalized))
    return prepared


def _project_mask(
    prepared: list[tuple[Track, np.ndarray]],
    active_tracks: tuple[int, ...],
    frame_shape: tuple[int, int],
    feather_radius: int = 0,
    frame_index: int | None = None,
) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for index in active_tracks:
        if prepared[index][0].action == "remove":
            track, spatial = prepared[index]
            if frame_index is not None and track.spatial_segments is not None:
                spatial = _spatial_mask(track, frame_shape, frame_index)
            np.maximum(mask, spatial, out=mask)
    if feather_radius > 0 and mask.any():
        flat = mask.astype(np.float32)
        kernel = max(3, feather_radius * 2 + 1)
        blurred = cv2.GaussianBlur(flat, (kernel, kernel), feather_radius / 2.0)
        mask = np.clip(np.where(flat >= 1.0, 1.0, blurred), 0.0, 1.0)
    for index in active_tracks:
        if prepared[index][0].action == "protect":
            mask[prepared[index][1] != 0] = 0
    return mask


def _active_tracks(
    prepared: list[tuple[Track, np.ndarray]], frame_index: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (track, _mask) in enumerate(prepared)
        if any(segment.contains(frame_index) for segment in track.segments)
    )


def execution_masks(
    plan: WipePlan, feather_radius: int = 0
) -> tuple[np.ndarray, Callable[[int], np.ndarray] | None]:
    """Project a plan into its static mask and optional temporal frame mask.

    The static mask is ``(H, W, 1)``. Temporal plans also receive a bounded,
    one-entry cached ``frame_mask(index) -> (H, W)`` projection. Cached masks
    are immutable so one consumer cannot corrupt a later frame.
    """
    prepared = _execution_tracks(plan)
    for track, mask in prepared:
        if not mask.any():
            raise InvalidInputError(
                f"track {track.id}: remove track mask is empty; cannot execute"
            )
    frame_shape = (plan.source.height, plan.source.width)
    static_mask = _union_projection(prepared, frame_shape, feather_radius)[:, :, None]
    if not prepared or not is_temporal(plan):
        return static_mask, None

    @lru_cache(maxsize=1)
    def mask_for(active_tracks: tuple[int, ...], spatial_frame: int | None) -> np.ndarray:
        mask = _project_mask(
            prepared, active_tracks, frame_shape, feather_radius, spatial_frame
        )
        mask.setflags(write=False)
        return mask

    spatial_starts = {i: [row["start"] for row in track.spatial_segments]
                      for i, (track, _) in enumerate(prepared) if track.spatial_segments is not None}

    def frame_mask(frame_index: int) -> np.ndarray:
        active = _active_tracks(prepared, frame_index)
        boundary = None
        for i in active:
            if i not in spatial_starts:
                continue
            rows = prepared[i][0].spatial_segments
            index = bisect_right(spatial_starts[i], frame_index) - 1
            changed = (rows[index]["start"] if frame_index < rows[index]["end"]
                       else rows[index]["end"]) if index >= 0 else 0
            boundary = max(boundary or 0, changed)
        # Every frame between joint spatial boundaries has exactly the same
        # alpha. Reuse it instead of blurring a full frame on every video frame.
        return mask_for(active, boundary)

    # Private producer guarantee: these cached results never change after return.
    frame_mask._videowipe_stable_masks = True
    return static_mask, frame_mask


def predicted_mask_at(plan: WipePlan, frame_index: int) -> np.ndarray:
    """Boolean ``(H, W)`` remove-prediction mask at *frame_index*.

    Union of active remove masks minus active protection masks. This is
    the per-frame prediction a temporal plan makes; the fact-baseline evaluator
    uses it instead of replaying one static mask on every annotated frame.
    """
    prepared = _execution_tracks(plan)
    return _project_mask(
        prepared,
        _active_tracks(prepared, frame_index),
        (plan.source.height, plan.source.width),
        frame_index=frame_index,
    ).astype(bool)


def _union_projection(prepared, frame_shape, feather_radius=0):
    # Without protection the legacy union is exact and avoids repeated blurs.
    if not any(track.action == "protect" for track, _ in prepared):
        return _project_mask(prepared, tuple(range(len(prepared))), frame_shape, feather_radius)
    result = np.zeros(frame_shape, dtype=np.float32 if feather_radius else np.uint8)
    boundaries = sorted({s.start for track, _ in prepared for s in track.segments}
                        | {s.end for track, _ in prepared for s in track.segments})
    boundaries = sorted(set(boundaries) | {row[key] for track, _ in prepared
                         for row in (track.spatial_segments or []) for key in ("start", "end")})
    for frame in boundaries[:-1]:
        np.maximum(result, _project_mask(prepared, _active_tracks(prepared, frame),
                                        frame_shape, feather_radius, frame), out=result)
    return result


def remove_union_mask(plan: WipePlan) -> np.ndarray:
    """Pixels possibly removed at any time, respecting time-limited protection."""
    return _union_projection(_execution_tracks(plan),
                             (plan.source.height, plan.source.width)).astype(bool)
