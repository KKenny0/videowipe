"""videowipe - STTN-based video inpainting tool."""

__version__ = "0.4.0"

from videowipe.engine import WipeEngine, remove_text
from videowipe.api import CancellationToken, ProgressEvent, WipeRequest, WipeResult
from videowipe.errors import (
    BackendUnavailableError,
    InvalidInputError,
    ProcessingCancelledError,
    ProcessingError,
    WipeError,
)
from videowipe.inpainters import (
    InpaintJob,
    InpaintOutcome,
    Inpainter,
    STTNInpainter,
    get_registry,
    register_inpainter,
)
from videowipe.detect import (
    CleanCandidate,
    CleanDetectionResult,
    DBNetDetector,
    TextBox,
    TextDetector,
    detect_clean_candidates,
    detect_subtitle_mask,
    infer_regions_from_text,
    infer_targets_from_text,
    mask_from_candidates,
    normalize_region,
    select_candidates_by_intent,
    select_clean_candidates,
)

__all__ = [
    "WipeEngine",
    "remove_text",
    "WipeRequest",
    "WipeResult",
    "ProgressEvent",
    "CancellationToken",
    "WipeError",
    "InvalidInputError",
    "BackendUnavailableError",
    "ProcessingCancelledError",
    "ProcessingError",
    "Inpainter",
    "InpaintJob",
    "InpaintOutcome",
    "STTNInpainter",
    "get_registry",
    "register_inpainter",
    "detect_clean_candidates",
    "detect_subtitle_mask",
    "infer_regions_from_text",
    "infer_targets_from_text",
    "normalize_region",
    "select_clean_candidates",
    "select_candidates_by_intent",
    "mask_from_candidates",
    "TextDetector",
    "DBNetDetector",
    "TextBox",
    "CleanCandidate",
    "CleanDetectionResult",
]
