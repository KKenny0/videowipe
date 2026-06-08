"""External model subprocess adapter."""
from __future__ import annotations

import os
import subprocess

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


class ExternalModelError(Exception):
    """Raised when the external model command fails."""


def run_external(command: str, video_path: str, mask_path: str,
                 output_dir: str) -> str:
    """Run an external inpainting command and return the output video path.

    The command is called as::

        <command> <video_path> <mask_path> <output_dir>

    Returns the path to the output video file found in *output_dir*.
    Raises ExternalModelError on non-zero exit or missing output.
    """
    full_cmd = f'{command} "{video_path}" "{mask_path}" "{output_dir}"'
    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True, check=False,
    )
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
