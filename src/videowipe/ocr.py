"""Optional OCR text recognition for detected text crops.

Requires ``rapidocr-onnxruntime`` (install with ``pip install videowipe[ocr]``).
Importing this module fails gracefully when the dependency is not installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_engine = None


def _get_engine():
    """Lazily initialize the RapidOCR engine."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ImportError(
            "rapidocr-onnxruntime is not installed. "
            "Install it with: pip install videowipe[ocr]"
        ) from exc
    _engine = RapidOCR()
    return _engine


def recognize_text(image_crop: np.ndarray) -> Optional[str]:
    """Recognize text in a single image crop.

    Args:
        image_crop: BGR image as a numpy array (H, W, 3).

    Returns:
        Recognized text string, or ``None`` if no text is found.
    """
    engine = _get_engine()
    result, _ = engine(image_crop)
    if not result:
        return None
    # result is a list of [bbox, text, confidence]
    texts = [item[1] for item in result if item[1]]
    return " ".join(texts) if texts else None


def batch_recognize(crops: list[np.ndarray]) -> list[Optional[str]]:
    """Recognize text in multiple image crops.

    Args:
        crops: List of BGR image arrays.

    Returns:
        List of recognized text strings (or ``None`` per crop).
    """
    return [recognize_text(crop) for crop in crops]
