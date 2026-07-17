"""Public request, result, progress, and cancellation contracts."""
from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from videowipe.errors import ProcessingCancelledError

Pathish = Union[str, os.PathLike]


@dataclass(frozen=True)
class WipeRequest:
    """One video-cleanup request.

    Model configuration belongs to :class:`WipeEngine`; this object contains
    only per-video inputs so one loaded engine can be reused across a batch.
    """

    video: Pathish
    output_dir: Pathish = "result/"
    mask: Optional[Pathish] = None
    detector: Any = None
    targets: Sequence[str] = ()
    intent: Optional[str] = None
    agent: Optional[str] = None
    regions: Sequence[str] = ()
    preview: bool = False
    confirm: bool = False
    detect_mode: Optional[str] = None
    ocr: Optional[str] = None


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress emitted by :meth:`WipeEngine.run`."""

    phase: str
    completed: int = 0
    total: int = 0
    message: Optional[str] = None

    @property
    def fraction(self) -> Optional[float]:
        if self.total <= 0:
            return None
        return min(1.0, max(0.0, self.completed / self.total))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WipeResult:
    """Structured outcome of one SDK request."""

    output_path: str
    backend: Optional[str]
    mask_source: str
    artifacts: Sequence[str] = ()
    timings: Mapping[str, float] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    preview: bool = False

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "backend": self.backend,
            "mask_source": self.mask_source,
            "artifacts": list(self.artifacts),
            "timings": dict(self.timings),
            "warnings": list(self.warnings),
            "preview": self.preview,
        }


class CancellationToken:
    """Thread-safe cooperative cancellation flag.

    Cancellation is observed at VideoWipe phase and segment boundaries. It
    does not forcibly interrupt a single backend inference call.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise ProcessingCancelledError("Video cleanup was cancelled")


ProgressCallback = Callable[[ProgressEvent], None]
