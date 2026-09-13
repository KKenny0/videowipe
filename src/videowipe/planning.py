"""Internal clean planning: request -> reviewable draft -> deterministic plan."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import cv2

from videowipe.detect import (
    CleanDetectionResult,
    CleanCandidate,
    TextBox,
    _bbox,
    infer_regions_from_text,
    infer_targets_from_text,
    normalize_target,
    refine_temporal_presence,
    resolve_detect_params,
    resolve_requested_targets,
    select_clean_candidates,
)
from videowipe.plan import Source, WipePlan, build_wipe_plan, compute_source, validate_plan, _spatial_mask


@dataclass(frozen=True)
class _CleanCandidateView:
    id: str
    type: str
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    frame_fraction: float
    reason: str
    default_remove: bool
    text_samples: tuple[str, ...]
    presence_frames: tuple[int, ...]
    mask: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "frame_fraction": round(self.frame_fraction, 3),
            "reason": self.reason,
            "default_remove": self.default_remove,
            "text_samples": list(self.text_samples[:5]),
            "presence_frames": list(self.presence_frames),
        }


def _candidate_view(candidate: Any) -> _CleanCandidateView:
    mask = None
    if candidate.mask is not None:
        mask = np.asarray(candidate.mask).view()
        mask.setflags(write=False)
    return _CleanCandidateView(
        id=candidate.id,
        type=candidate.type,
        label=candidate.label,
        bbox=tuple(candidate.bbox),
        confidence=float(candidate.confidence),
        frame_fraction=float(getattr(candidate, "frame_fraction", 0.0)),
        reason=getattr(candidate, "reason", ""),
        default_remove=bool(candidate.default_remove),
        text_samples=tuple(getattr(candidate, "text_samples", ())),
        presence_frames=tuple(getattr(candidate, "presence_frames", ())),
        mask=mask,
    )


class CleanPlanDraft:
    """Minimal review view over the evidence needed to finalize a clean plan."""

    def __init__(
        self,
        video_path: str,
        result: CleanDetectionResult,
        source: Source,
        proposed_remove_ids: Iterable[str],
        resolved_request: Mapping[str, Any],
        user_directed: bool,
    ) -> None:
        self._video_path = video_path
        self._result = result
        self._source = source
        self._proposed_remove_ids = frozenset(proposed_remove_ids)
        self._resolved_request = dict(resolved_request)
        self._user_directed = user_directed

    @property
    def candidates(self) -> tuple[_CleanCandidateView, ...]:
        return tuple(
            _candidate_view(candidate) for candidate in self._result.candidates
        )

    @property
    def proposed_remove_ids(self) -> frozenset[str]:
        return self._proposed_remove_ids

    @property
    def resolved_request(self) -> dict[str, Any]:
        return dict(self._resolved_request)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return tuple(self._result.frame_shape)

    @property
    def preview_frame(self) -> np.ndarray | None:
        frame = self._result.preview_frame
        return None if frame is None else frame.copy()

    @property
    def detector_weight(self) -> tuple[str, str | None] | None:
        """Read-only detector-weight identity for formal provenance checks."""
        detector = self._result.detector
        path = getattr(detector, "_weight_path", None)
        if not path:
            return None
        return str(path), getattr(detector, "_weight_sha256", None)

    def with_remove_ids(
        self, remove_ids: Iterable[str], *, user_directed: bool = True,
    ) -> CleanPlanDraft:
        return CleanPlanDraft(
            self._video_path,
            self._result,
            self._source,
            remove_ids,
            self._resolved_request,
            user_directed,
        )

    def review_snapshot(self, output_dir: str) -> dict[str, Any]:
        """Independent public metadata; never shares detector or mask objects."""
        return {
            "source": self._source.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "default_selected_ids": sorted(self.proposed_remove_ids),
            "evidence": {
                name: str(Path(output_dir) / name)
                for name in ("clean_candidates.json", "clean_preview.jpg",
                             "clean_preview_source.jpg")
            },
        }

    def save_refinement_evidence(self, output_dir: str) -> None:
        """Private restart evidence; the public callback never exposes it."""
        evidence = {
            "sample_indices": self._result.sample_indices,
            "candidates": [dict(candidate.to_dict(),
                                detector_backed=candidate.detector_backed,
                                temporal_sample_indices=candidate.temporal_sample_indices)
                           for candidate in self._result.candidates],
            "boxes": {str(index): [dict(points=box.points.tolist(),
                                        confidence=float(box.confidence), text=box.text)
                                   for box in boxes]
                      for index, boxes in self._result.sampled_frame_boxes.items()},
        }
        (Path(output_dir) / "refinement_evidence.json").write_text(
            json.dumps(evidence), encoding="utf-8",
        )

    def for_request(
        self,
        *,
        targets: Iterable[str] = (),
        intent: str | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> CleanPlanDraft:
        """Derive another review from the same prepared detection evidence."""
        target_list = list(targets)
        effective_targets = resolve_requested_targets(target_list)
        selected = select_clean_candidates(
            self._result.candidates, targets=effective_targets, intent=intent,
        )
        resolved = dict(self._resolved_request if request is None else request)
        resolved.setdefault("intent", intent)
        resolved.setdefault("targets", list(effective_targets))
        return CleanPlanDraft(
            self._video_path,
            self._result,
            self._source,
            (candidate.id for candidate in selected),
            resolved,
            bool(target_list or intent),
        )


def build_recognizer(ocr_mode: str):
    """Build the optional OCR callable for clean detection."""
    if ocr_mode == "off":
        return None
    try:
        from videowipe.ocr import _get_engine, recognize_text

        _get_engine()
        return recognize_text
    except Exception:
        if ocr_mode == "rapidocr":
            raise RuntimeError(
                "OCR mode 'rapidocr' requested but rapidocr-onnxruntime "
                "is not installed. Install it with: pip install videowipe[ocr]"
            ) from None
        return None


def _agent_selection(agent: str, candidates: Sequence[Any], intent: str):
    from videowipe.agent import select_with_agent

    selected_ids = select_with_agent(agent, candidates, intent)
    if selected_ids is None:
        return None
    wanted = set(selected_ids)
    return [candidate for candidate in candidates if candidate.id in wanted]


def _warn_if_timestamp_unresolved(
    targets: Iterable[str], candidates: Sequence[Any], warn: Callable[[str], None],
) -> None:
    requested = {normalize_target(target) for target in targets}
    if "timestamp" in requested and not any(
        candidate.type == "timestamp" for candidate in candidates
    ):
        warn(
            "No timestamp target was confirmed. Timestamp detection requires "
            "recognized text content; use --region top-left/top-right if the "
            "current detector only finds text boxes."
        )


def prepare(
    video_path: str,
    *,
    detector: Any = None,
    targets: Iterable[str] = (),
    intent: str | None = None,
    agent: str | None = None,
    regions: Iterable[str] = (),
    detect_mode: str = "balanced",
    ocr: str = "auto",
    recognizer_builder: Callable[[str], Any] = build_recognizer,
    agent_selector: Callable[[str, Sequence[Any], str], Any] = _agent_selection,
    warn: Callable[[str], None] = print,
) -> CleanPlanDraft:
    """Interpret a request, detect candidates, and return its reviewable draft."""
    from videowipe.detect import detect_clean_candidates

    target_list = list(targets)
    region_list = list(regions)
    target_text = " ".join(target_list)
    intent_text = " ".join(part for part in (target_text, intent or "") if part)
    requested_regions = list(region_list)
    requested_regions.extend(infer_regions_from_text(intent_text))
    requested_regions = list(dict.fromkeys(requested_regions))

    intent_targets = infer_targets_from_text(intent or "")
    effective_targets = resolve_requested_targets(target_list)
    normalized_targets = {normalize_target(target) for target in effective_targets}
    if requested_regions:
        effective_targets.append("region")
        normalized_targets.add("region")
    mentioned_targets = normalized_targets | set(intent_targets)
    text_targets = {
        "subtitle", "timestamp", "watermark", "scene_text", "unknown_text",
    }
    detect_text = (
        bool(mentioned_targets & text_targets)
        or (not requested_regions and not mentioned_targets)
        or bool(intent and not mentioned_targets)
    )
    params = resolve_detect_params(
        detect_mode, has_subtitle_target="subtitle" in normalized_targets,
    )
    result = detect_clean_candidates(
        video_path,
        detector=detector,
        regions=requested_regions,
        detect_text=detect_text,
        include_logo="logo" in mentioned_targets,
        include_translucent_watermark="watermark" in mentioned_targets,
        sample_count=params["sample_count"],
        consistency=params["consistency"],
        subtitle_fallback=params["subtitle_fallback"],
        recognizer=recognizer_builder(ocr),
    )
    selected = select_clean_candidates(
        result.candidates, targets=effective_targets, intent=intent,
    )
    _warn_if_timestamp_unresolved(effective_targets, result.candidates, warn)
    if agent and intent:
        agent_selected = agent_selector(agent, result.candidates, intent)
        if agent_selected is None:
            warn("Agent selection unavailable; using local rules.")
        else:
            selected = agent_selected

    return CleanPlanDraft(
        video_path,
        result,
        compute_source(video_path),
        (candidate.id for candidate in selected),
        {
            "intent": intent,
            "targets": list(effective_targets),
            "regions": requested_regions,
            "detect_mode": detect_mode,
            "ocr": ocr,
        },
        bool(target_list or intent or region_list),
    )


def finalize(
    draft: CleanPlanDraft,
    *,
    refine: bool | None = None,
    progress: Any = None,
    check_cancelled: Any = None,
) -> WipePlan:
    """Turn a reviewed draft into its provisional -> refine -> final WipePlan."""
    selected = set(draft._proposed_remove_ids)
    if draft._user_directed:
        explicit_remove_ids = selected
        explicit_keep_ids = {candidate.id for candidate in draft._result.candidates} - selected
    else:
        explicit_remove_ids = set()
        explicit_keep_ids = set()
    should_refine = (
        draft._resolved_request.get("detect_mode") != "fast"
        if refine is None else refine
    )
    return _finalize_result(
        draft._video_path,
        draft._result,
        draft._source,
        refine=should_refine,
        request=draft._resolved_request,
        explicit_remove_ids=explicit_remove_ids,
        explicit_keep_ids=explicit_keep_ids,
        progress=progress,
        check_cancelled=check_cancelled,
    )


def _descender_extensions(frame, boxes):
    """Recover narrow glyph tails supported by contrasting lower-edge pixels.

    Conservative light glyph / dark edge evidence only; other subtitle styles
    retain their detector geometry. White tails use the existing 20% glyph margin;
    black tails retain their three-pixel search limit.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    extra = []
    for x1, y1, x2, y2 in boxes:
        top, end = y1, min(gray.shape[0], y2+1+max(3, int(np.ceil((y2-y1+1)*.2))))
        strip = gray[top:end, x1:x2+1]
        neutral = np.ptp(frame[top:end, x1:x2+1], axis=2) <= 40
        light = ((strip >= 180) & neutral).astype(np.uint8)
        dark = (strip < 16).astype(np.uint8)
        dark[:max(0, y2-8-top)] = 0
        dark[y2+4-top:] = 0
        kernel = np.ones((9,9),np.uint8)
        # A weak light band connects antialiased edges to the bright glyph;
        # it cannot qualify without a bright seed and adjacent dark contrast.
        faint = ((strip >= 128) & neutral).astype(np.uint8)
        faint &= cv2.dilate(light, np.ones((3,3), np.uint8))
        contrast = (strip <= 90).astype(np.uint8)
        candidates = ((dark, cv2.dilate(light, kernel), dark),
                      (faint, cv2.dilate(contrast, kernel), light))
        for pixels, support, seed in candidates:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(pixels)
            for label in range(1, count):
                x, y, width, height, _ = stats[label]
                bottom = top+y+height-1
                if (top+y > y2 or bottom <= y2
                        or (pixels is faint and (bottom == end-1 or x == 0 or x+width == strip.shape[1]))
                        ):
                    continue
                component = labels[y:y+height, x:x+width] == label
                if (not np.any(support[y:y+height, x:x+width][component])
                        or not np.any(seed[y:y+height, x:x+width][component])):
                    continue
                # Neighboring letters may share one outline above the edge; only
                # measure each disconnected tail below it, not the whole word.
                tail = component[y2-top+1-y:].astype(np.uint8)
                _, _, tails, _ = cv2.connectedComponentsWithStats(tail)
                for tx, ty, tw, th, _ in tails[1:]:
                    if tw <= y2-y1+1:
                        extra.append([x1+int(x+tx), y2+1+int(ty),
                                      x1+int(x+tx+tw)-1, y2+int(ty+th)])
    return sorted({tuple(box) for box in extra})


def _recover_descenders(plan, tracks, video_path, check_cancelled):
    frames = {t.id: {i: row['boxes'] for row in t.spatial_segments
                    for i in range(row['start'], row['end'])} for t in tracks}
    indices = sorted({i for rows in frames.values() for i in rows})
    if not indices:
        return
    reader = cv2.VideoCapture(video_path)
    try:
        if not reader.isOpened() or not reader.set(cv2.CAP_PROP_POS_FRAMES, indices[0]):
            raise ValueError('Cannot read subtitle outline evidence')
        for i in range(indices[0], indices[-1]+1):
            if check_cancelled is not None:
                check_cancelled()
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f'Cannot decode subtitle outline frame {i}')
            for track in tracks:
                boxes = frames[track.id].get(i)
                if boxes is None:
                    continue
                extra = _descender_extensions(frame, boxes)
                if len(boxes)+len(extra) <= 64:
                    frames[track.id][i] = boxes + [list(b) for b in extra]
    finally:
        reader.release()
    for track in tracks:
        rows = []
        for i, boxes in sorted(frames[track.id].items()):
            if rows and rows[-1]['end'] == i and rows[-1]['boxes'] == boxes:
                rows[-1]['end'] = i+1
            else:
                rows.append(dict(start=i, end=i+1, boxes=boxes))
        track.spatial_segments = rows
        track.mask = _spatial_mask(track, (plan.source.height, plan.source.width))


def _add_spatial_segments(plan, result, selected_ids=None, *, video_path=None, check_cancelled=None):
    """Reuse dense detections; absent local evidence never means a full-band wipe."""
    eligible = {c.id for c in result.candidates
                if c.type == "subtitle" and getattr(c, "detector_backed", False)}
    h, w = result.frame_shape
    evidence = getattr(result, "sampled_frame_boxes", {})
    if not evidence:
        return  # Custom mask providers need not supply detector frame evidence.
    # Midpoint interpolation can extend final intervals beyond the coarse
    # windows that were densely checked. Resolve only those unobserved frames;
    # an observed empty/error frame remains empty.
    missing = sorted({index for track in plan.remove_tracks
                      if track.id in eligible and "bbox-override" not in track.decision_reason
                      and (selected_ids is None or track.id in selected_ids)
                      for segment in track.segments for index in range(segment.start, segment.end)
                      if index not in evidence})
    if missing and video_path is not None:
        reader = cv2.VideoCapture(video_path)
        try:
            if not reader.isOpened():
                raise ValueError("Cannot open video for local subtitle boundary checks")
            for index in missing:
                if check_cancelled is not None:
                    check_cancelled()
                reader.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = reader.read()
                if not ok:
                    raise ValueError(f"Cannot decode subtitle boundary frame {index}")
                try:
                    evidence[index] = result.detector.detect(frame)
                except Exception as exc:
                    evidence[index] = []
                    plan.warnings.append(f"local subtitle detection failed at frame {index}; keeping frame: {exc}")
        finally:
            reader.release()
    evidence = sorted(evidence.items())
    for track in plan.remove_tracks:
        if (track.id not in eligible or "bbox-override" in track.decision_reason
                or (selected_ids is not None and track.id not in selected_ids)):
            continue
        x1, y1, x2, y2 = track.bbox
        local_frames = {}
        for index, boxes in evidence:
            if not any(s.contains(index) for s in track.segments):
                continue
            local = []
            for box in boxes:
                bx1, by1, bx2, by2 = _bbox(box.points, w, h)
                if x1 <= (bx1+bx2)/2 <= x2 and y1 <= (by1+by2)/2 <= y2:
                    local.append((bx1, by1, bx2, by2))
            if not local:
                continue
            margin = max(4, int(np.ceil(np.median([b[3]-b[1]+1 for b in local]) * .2)))
            local = sorted({(max(0, a-14), max(0, b-margin), min(w-1, c+14), min(h-1, d+margin))
                            for a,b,c,d in local})
            if len(local) > 64:
                raise ValueError(f"Too many local subtitle boxes at frame {index}")
            local_frames[index] = [list(box) for box in local]
        rows = []
        radius = max(0, int(plan.source.fps / 4))
        for index, boxes in local_frames.items():
            neighborhood = list(boxes)
            for direction in (-1, 1):
                for distance in range(1, radius+1):
                    adjacent = local_frames.get(index + direction * distance)
                    if adjacent is None:
                        break  # Never borrow evidence across a subtitle gap.
                    neighborhood.extend(adjacent)
            # Typography height is stable over a short window even when DBNet
            # clips accents in one frame. Width stays local to the current text.
            top = min(b[1] for b in neighborhood)
            bottom = max(b[3] for b in neighborhood)
            boxes = [[a, top, c, bottom] for a, _, c, _ in boxes]
            if rows and rows[-1]["end"] == index and rows[-1]["boxes"] == boxes:
                rows[-1]["end"] = index + 1
            else:
                rows.append({"start": index, "end": index+1, "boxes": boxes})
        track.spatial_segments = rows
        track.mask = _spatial_mask(track, (h, w))
        plan.schema_version = 3

    if video_path is not None:
        tracks = [t for t in plan.remove_tracks if t.id in eligible
                  and 'bbox-override' not in t.decision_reason
                  and (selected_ids is None or t.id in selected_ids)
                  and t.spatial_segments is not None]
        _recover_descenders(plan, tracks, video_path, check_cancelled)


def _finalize_result(
    video_path: str,
    result: CleanDetectionResult,
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
    """Build the provisional -> refine -> final plan for prepared evidence."""
    kwargs = {
        "request": request,
        "explicit_remove_ids": explicit_remove_ids,
        "explicit_keep_ids": explicit_keep_ids,
        "loaded_actions": loaded_actions,
    }
    provisional = build_wipe_plan(
        result.candidates,
        result.sample_indices,
        len(result.sample_indices),
        source,
        result.frame_shape,
        **kwargs,
    )
    if not refine:
        return provisional

    warnings = refine_temporal_presence(
        video_path,
        result,
        {track.id: track.segments for track in provisional.remove_tracks},
        source.frame_count,
        progress=progress,
        check_cancelled=check_cancelled,
    )
    plan = build_wipe_plan(
        result.candidates,
        result.sample_indices,
        len(result.sample_indices),
        source,
        result.frame_shape,
        **kwargs,
    )
    plan.warnings.extend(warnings)
    _add_spatial_segments(plan, result, video_path=video_path, check_cancelled=check_cancelled)
    validate_plan(plan, frame_shape=result.frame_shape)
    return plan


def refine_review(video_path, machine_plan, reviewed_plan, evidence_path, detector,
                  progress=None, check_cancelled=None, output_dir=None):
    """Refine newly selected keep tracks without changing reviewed masks."""
    new_ids = {track.id for track in reviewed_plan.remove_tracks} - {
        track.id for track in machine_plan.remove_tracks
    }
    if not new_ids:
        return reviewed_plan
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    masks = {track.id: track.mask for track in machine_plan.tracks}
    candidates = [CleanCandidate(**row, mask=masks[row["id"]])
                  for row in evidence["candidates"]]
    result = CleanDetectionResult(
        candidates=candidates,
        frame_shape=(machine_plan.source.height, machine_plan.source.width),
        sample_indices=evidence["sample_indices"], detector=detector,
        sampled_frame_boxes={int(index): [TextBox(np.asarray(box["points"]),
                                                box["confidence"], box["text"])
                                         for box in boxes]
                             for index, boxes in evidence["boxes"].items()},
    )
    warnings = refine_temporal_presence(
        video_path, result,
        {track.id: track.segments for track in machine_plan.tracks if track.id in new_ids},
        machine_plan.source.frame_count, progress=progress,
        check_cancelled=check_cancelled,
    )
    refined = build_wipe_plan(
        candidates, result.sample_indices, len(result.sample_indices),
        machine_plan.source, result.frame_shape, explicit_remove_ids=new_ids,
    )
    by_id = {track.id: track for track in refined.tracks}
    for track in reviewed_plan.remove_tracks:
        if track.id in new_ids:
            track.segments = by_id[track.id].segments
        if not track.segments:
            raise ValueError(f"Selected target {track.id} has no confirmed active frames")
    _add_spatial_segments(reviewed_plan, result, new_ids, video_path=video_path, check_cancelled=check_cancelled)
    reviewed_plan.warnings.extend(warnings)
    validate_plan(reviewed_plan, require_remove=True)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        CleanPlanDraft(video_path, result, machine_plan.source, new_ids, {}, True).save_refinement_evidence(output_dir)
    return reviewed_plan
