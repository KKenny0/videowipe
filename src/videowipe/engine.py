"""WipeEngine and convenience functions."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from videowipe.tasks.base import BaseTask, read_frame_info, read_mask
from videowipe.tasks.detext import DetextTask
from videowipe.tasks.delogo import DelogoTask
from videowipe.weights import ensure_onnx_weights, ensure_weight

if TYPE_CHECKING:
    from videowipe.detect import TextDetector

_TASK_CLASSES = {
    "detext": DetextTask,
    "delogo": DelogoTask,
}

_DEFAULT_WEIGHTS_PTH = {
    "detext": "detext_trial.pth",
}
_DEFAULT_WEIGHTS_ONNX = {
    "detext": "sttn",
}


class WipeEngine:
    """Reusable engine for video inpainting tasks.

    Create one engine, call process() for each video, then cleanup().

    Args:
        task: Task type, "detext" or "delogo".
        weight: Path to model weight file (.pth for PyTorch, .onnx for ONNX).
            None to auto-download the default.
        device: "auto", "cuda", or "cpu". Only used with .pth weights.
        gap: Segment length per pass. Higher = better quality, slower.
        dual: Show original video side-by-side in output.
        detector: TextDetector for auto mask generation. None to use default.
    """

    def __init__(
        self,
        task: str = "detext",
        weight: Optional[str] = None,
        device: str = "auto",
        gap: int = 200,
        dual: bool = False,
        detector: Optional[TextDetector] = None,
    ):
        if task not in _TASK_CLASSES:
            raise ValueError(f"Unknown task: {task}. Choose from: {list(_TASK_CLASSES)}")

        self.task = task
        self._weight = weight
        self._device = device
        self._detector = detector
        self._task_impl: BaseTask = _TASK_CLASSES[task](gap=gap, dual=dual)
        self._model_loaded = False

    def _ensure_model(self):
        if self._model_loaded:
            return
        weight_path = self._weight
        if weight_path is None:
            # Auto-detect: prefer ONNX if onnxruntime is available, else PyTorch
            try:
                import onnxruntime  # noqa: F401
                base = ensure_onnx_weights(_DEFAULT_WEIGHTS_ONNX[self.task])
                weight_path = base + ".onnx"
            except ImportError:
                try:
                    weight_path = ensure_weight(_DEFAULT_WEIGHTS_PTH[self.task])
                except Exception:
                    raise RuntimeError(
                        "No inference backend found. Install one of:\n"
                        "  pip install videowipe[onnx]   (lightweight, ~200MB)\n"
                        "  pip install videowipe[torch]  (full PyTorch, ~2.5GB)"
                    ) from None
        self._task_impl.load_model(weight_path, device=self._device)
        self._model_loaded = True

    def process(self, video: str, mask: str | None = None,
                output: str = "result/",
                detector: Optional[TextDetector] = None) -> str:
        """Process a single video. Returns the output file path.

        Args:
            video: Path to input video.
            mask: Path to mask image. ``None`` to auto-detect subtitle regions.
            output: Output directory.
            detector: Override the text detector for auto-mask generation.
        """
        self._ensure_model()

        os.makedirs(output, exist_ok=True)

        reader, frame_info = read_frame_info(video)

        if mask is not None:
            mask_arr = read_mask(mask)
        else:
            from videowipe.detect import detect_subtitle_mask
            det = detector or self._detector
            mask_arr = detect_subtitle_mask(video, detector=det)
            # Save auto-detected mask for inspection
            cv2.imwrite(
                os.path.join(output, "auto_mask.png"),
                (mask_arr * 255).astype(np.uint8),
            )

        return self._task_impl.process_video(reader, frame_info, mask_arr, output,
                                              video_path=video)

    def cleanup(self):
        """Release model and GPU memory."""
        self._task_impl.cleanup()
        self._model_loaded = False


def remove_text(
    video: str,
    mask: str | None = None,
    output: str = "result/",
    weight: str | None = None,
    device: str = "auto",
    gap: int = 200,
    dual: bool = False,
    detector: Optional[TextDetector] = None,
) -> str:
    """Remove hardcoded subtitles from a video. Convenience wrapper.

    Backend is auto-detected from the weight file:
      .pth → PyTorch (requires ``pip install videowipe[torch]``)
      .onnx → ONNX Runtime (requires ``pip install videowipe[onnx]``)

    If *mask* is ``None``, subtitle regions are auto-detected.
    """
    engine = WipeEngine(
        task="detext", weight=weight, device=device, gap=gap, dual=dual,
        detector=detector,
    )
    result = engine.process(video=video, mask=mask, output=output)
    engine.cleanup()
    return result
