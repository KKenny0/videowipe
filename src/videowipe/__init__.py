"""videowipe - STTN-based video inpainting tool."""

__version__ = "0.1.0"

from videowipe.engine import WipeEngine, remove_text
from videowipe.detect import (
    DBNetDetector,
    TextBox,
    TextDetector,
    detect_subtitle_mask,
)

__all__ = [
    "WipeEngine",
    "remove_text",
    "detect_subtitle_mask",
    "TextDetector",
    "DBNetDetector",
    "TextBox",
]
