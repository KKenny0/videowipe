"""Internal clean planning: request -> reviewable draft -> deterministic plan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from videowipe.detect import (
    CleanDetectionResult,
    infer_regions_from_text,
    infer_targets_from_text,
    normalize_target,
    refine_temporal_presence,
    resolve_detect_params,
    resolve_requested_targets,
    select_clean_candidates,
)
from videowipe.plan import Source, WipePlan, build_wipe_plan, compute_source, validate_plan


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
    validate_plan(plan, frame_shape=result.frame_shape)
    return plan
