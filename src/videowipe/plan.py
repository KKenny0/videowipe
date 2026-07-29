"""WipePlan v1: serializable, reviewable, deterministically-executable plan.

A :class:`WipePlan` is the intermediate representation between detection and
inpainting. Each detected candidate becomes an addressable *track* with:

- an explicit ``remove`` | ``keep`` action,
- a stable spatial mask (stored in a sidecar NPZ, not the JSON),
- half-open temporal ``[start, end)`` segments saying when it is active.

Schema v1 is a *screen-overlay* model: one stable spatial mask per track plus
time intervals. It is not a general motion-object tracker — there is no
per-frame mask propagation. That is the minimum model the current failure
facts require: it closes subtitle-gap false erasures and lets persistent top
overlays (logos, credits) be flagged ``keep`` instead of misclassified as
subtitles.

The JSON stays agent-readable; precise binary masks live in a sidecar
``wipe_plan_masks.npz``. The NPZ holds only raw ``uint8`` arrays and is loaded
with ``allow_pickle=False``; nothing picklable is ever written.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from videowipe.errors import InvalidInputError

SCHEMA_VERSION = 1
PLAN_KIND = "wipe_plan"
JSON_FILENAME = "wipe_plan.json"
MASK_FILENAME = "wipe_plan_masks.npz"

_ACTION_VALUES = ("remove", "keep")
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
        return cls(start=int(value[0]), end=int(value[1]))


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

    def to_dict(self) -> dict:
        return {
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
            bbox=tuple(int(v) for v in d["bbox"]),
            confidence=float(d["confidence"]),
            presence_fraction=float(d["presence_fraction"]),
            decision_reason=str(d["decision_reason"]),
            segments=[Segment.from_value(s) for s in d["segments"]],
            mask_key=str(d["mask_key"]),
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
            schema_version=int(d["schema_version"]),
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
        presence_fraction = (len(presence_frames) / float(n_valid)) if n_valid > 0 else 0.0

        if presence_frames:
            segs = segments_from_presence(samples, presence_frames, frame_count)
            seg_note = f"{len(segs)} segment(s) from {len(presence_frames)}/{n_valid} samples"
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
    if plan.schema_version != SCHEMA_VERSION:
        raise InvalidInputError(
            f"unsupported schema_version {plan.schema_version} (expected {SCHEMA_VERSION})"
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
    has_remove = False
    for t in plan.tracks:
        if not t.id:
            raise InvalidInputError("track id must be non-empty")
        if t.id in ids:
            raise InvalidInputError(f"duplicate track id: {t.id}")
        ids.add(t.id)
        if t.action not in _ACTION_VALUES:
            raise InvalidInputError(
                f"track {t.id}: action {t.action!r} not in {_ACTION_VALUES}"
            )
        if len(t.bbox) != 4 or any(int(v) < 0 for v in t.bbox):
            raise InvalidInputError(f"track {t.id}: bbox must be 4 non-negative ints")
        x1, y1, x2, y2 = t.bbox
        if x2 < x1 or y2 < y1:
            raise InvalidInputError(f"track {t.id}: bbox {t.bbox} is inverted")

        # Execution requires a precise mask on every remove track. A plan read
        # metadata-only (load_masks=False) may carry mask=None for inspection,
        # but it cannot be executed.
        if require_remove and t.action == "remove" and t.mask is None:
            raise InvalidInputError(
                f"track {t.id}: remove track has no precise mask; cannot execute"
            )
        if t.mask is not None:
            arr = np.asarray(t.mask)
            if arr.ndim not in (2, 3):
                raise InvalidInputError(
                    f"track {t.id}: mask must be 2D/3D, got shape {arr.shape}"
                )
            if (arr.shape[0], arr.shape[1]) != (height, width):
                raise InvalidInputError(
                    f"track {t.id}: mask shape {arr.shape[:2]} != source {height}x{width}"
                )
            if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
                raise InvalidInputError(
                    f"track {t.id}: mask dtype {arr.dtype} must be integer/bool"
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
        masks[t.mask_key] = arr.astype(np.uint8)
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


def is_temporal(plan: WipePlan) -> bool:
    """True if any remove track is not active on the entire video.

    File-based inpainters (external command, ProPainter) consume one static
    mask PNG and cannot honor temporal segments; the engine uses this to
    reject such plans instead of silently flattening them.
    """
    frame_count = plan.source.frame_count
    for t in plan.remove_tracks:
        if not (len(t.segments) == 1 and t.segments[0] == Segment(0, frame_count)):
            return True
    return False


def predicted_mask_at(plan: WipePlan, frame_index: int) -> np.ndarray:
    """Boolean ``(H, W)`` remove-prediction mask at *frame_index*.

    Union of every remove track whose segments contain *frame_index*. This is
    the per-frame prediction a temporal plan makes; the fact-baseline evaluator
    uses it instead of replaying one static mask on every annotated frame.
    """
    height, width = plan.source.height, plan.source.width
    mask = np.zeros((height, width), dtype=bool)
    for t in plan.remove_tracks:
        if any(s.start <= frame_index < s.end for s in t.segments) and t.mask is not None:
            arr = np.asarray(t.mask)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            mask |= arr.astype(bool)
    return mask


def remove_union_mask(plan: WipePlan) -> np.ndarray:
    """Boolean ``(H, W)`` spatial union of all remove tracks (time-independent)."""
    height, width = plan.source.height, plan.source.width
    mask = np.zeros((height, width), dtype=bool)
    for t in plan.remove_tracks:
        if t.mask is not None:
            arr = np.asarray(t.mask)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            mask |= arr.astype(bool)
    return mask
