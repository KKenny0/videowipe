"""External model subprocess adapter.

Wraps an external inpainting command (``<command> <video> <mask>
<output_dir>``) as both a plain function (:func:`run_external`) and an
:class:`Inpainter` (:class:`ExternalInpainter`).

Invocation uses :func:`shlex.split` plus an argv list with ``shell=False``, so
shell metacharacters in the command string or the path arguments are never
interpreted by a shell — this is an injection-safety property. A command whose
executable cannot be found raises :class:`ExternalModelError`, as does a
non-zero exit or a missing output video.
"""
from __future__ import annotations

import os
import shlex
import subprocess

from videowipe.inpainters.base import InpaintJob, InpaintOutcome

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


class ExternalModelError(Exception):
    """Raised when the external model command fails."""


def run_external(command: str, video_path: str, mask_path: str,
                 output_dir: str) -> str:
    """Run an external inpainting command and return the output video path.

    The command is split with :func:`shlex.split` and invoked as an argv list
    with ``shell=False``; the three path arguments are appended verbatim. No
    shell is involved, so shell metacharacters in *command* or the paths are
    not interpreted.

    Returns the path to the output video file found in *output_dir*.
    Raises :class:`ExternalModelError` on a missing executable, a non-zero
    exit, or a missing output video.
    """
    cmd = shlex.split(command) + [video_path, mask_path, output_dir]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        raise ExternalModelError(
            f"External command not found: {cmd[0]!r}"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ExternalModelError(
            f"External command exited with code {result.returncode}"
            f"{': ' + stderr if stderr else ''}"
        )

    for name in os.listdir(output_dir):
        _, ext = os.path.splitext(name)
        if ext.lower() in _VIDEO_EXTENSIONS:
            return os.path.join(output_dir, name)

    raise ExternalModelError(
        "External command succeeded but produced no output video"
    )


class ExternalInpainter:
    """:class:`Inpainter` that shells out to an external model command.

    File-based: runs ``<command> <video> <mask> <output_dir>`` (no shell) and
    returns the produced output video path. Registered under the name
    ``"external"``. Requires ``job.mask_path`` (a mask file on disk); the
    ``mask`` ndarray field of the job is ignored.
    """

    name = "external"

    def __init__(self, command: str):
        if not command:
            raise ValueError(
                "ExternalInpainter requires a non-empty command string"
            )
        self.command = command

    def load(self, weight_path: str, device: str = "auto") -> None:
        # External models manage their own weights; nothing to preload.
        return None

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        mask_path = getattr(job, "mask_path", None)
        if not mask_path:
            raise ValueError(
                "ExternalInpainter requires job.mask_path (a mask file path)"
            )
        out_path = run_external(
            self.command, job.video_path, mask_path, job.output_dir
        )
        return InpaintOutcome(output_path=out_path, backend="external")

    def cleanup(self) -> None:
        return None
