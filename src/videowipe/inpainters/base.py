"""Inpainter protocol: the pluggable unit for video inpainting models.

An Inpainter takes an :class:`InpaintJob` (whole-video contract) and returns an
:class:`InpaintOutcome`. Two implementation shapes are supported equally:

* **Frame-based** (e.g. STTN): reads frames from ``job.reader``, runs an
  internal segment loop, and streams output via ffmpeg.
* **File-based** (e.g. an external subprocess): runs over the whole video file
  and returns the output path; it ignores ``job.reader``.

The protocol deliberately lives at the whole-video layer rather than
frames-in/frames-out, because some models (ProPainter) cannot consume an
in-memory frame stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np


@dataclass
class InpaintJob:
    """Whole-video inpainting request.

    Attributes:
        video_path: input video file path.
        mask: binary mask, shape ``(H, W, 1)``, values in ``{0, 1}``.
        output_dir: directory for the output video and artifacts.
        fps, frame_count, width, height: video metadata.
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        dual: when True, emit side-by-side original + inpainted output.
        gap: segment-length hint used by frame-based inpainters.
        output_suffix: filename suffix for the output video (e.g. ``"detext"``).
        reader: ``cv2.VideoCapture`` for frame-based inpainters; may be ``None``
            for file-based inpainters. The caller owns the reader lifecycle.
        progress: optional ``(frames_done, frames_total)`` callback.
        metrics: mutable dict (typically the benchmark ``timing`` dict) the
            inpainter writes phase timings into.
    """

    video_path: str
    mask: np.ndarray
    output_dir: str
    fps: float
    frame_count: int
    width: int
    height: int
    device: str = "auto"
    dual: bool = False
    gap: int = 200
    output_suffix: str = "detext"
    reader: object = None
    progress: Optional[Callable[[int, int], None]] = None
    metrics: dict = field(default_factory=dict)


@dataclass
class InpaintOutcome:
    """Result of an :meth:`Inpainter.inpaint` call."""

    output_path: str
    backend: str  # label written to benchmark.json


class Inpainter(Protocol):
    """Pluggable inpainting model.

    Lifecycle: :meth:`load` (engine-driven, once) -> :meth:`inpaint` per video
    -> :meth:`cleanup`. STTN keeps a loaded backend across videos; file-based
    inpainters may be effectively stateless.
    """

    name: str

    def load(self, weight_path: str, device: str = "auto") -> None:
        """Load model weights. Called once before the first inpaint()."""
        ...

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        """Inpaint one video; return the output path and a backend label."""
        ...

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        ...
