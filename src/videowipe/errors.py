"""Stable public exceptions for the VideoWipe SDK."""
from __future__ import annotations

from typing import Optional


class WipeError(RuntimeError):
    """Base class for errors exposed by :meth:`WipeEngine.run`.

    ``cause`` preserves implementation diagnostics without making backend-
    specific exception types part of the public contract. Treat it as
    sensitive: external backends may include local paths or stderr in it.
    """

    code = "WIPE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.code = code or type(self).code
        self.retryable = retryable
        self.cause = cause


class InvalidInputError(WipeError):
    """The request cannot be processed as supplied."""

    code = "INVALID_INPUT"


class BackendUnavailableError(WipeError):
    """No usable inference backend or model is available."""

    code = "BACKEND_UNAVAILABLE"


class ProcessingCancelledError(WipeError):
    """The caller cancelled a running request at a safe boundary."""

    code = "PROCESSING_CANCELLED"


class ProcessingError(WipeError):
    """Video processing failed after the request was accepted."""

    code = "PROCESSING_FAILED"
