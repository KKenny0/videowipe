"""Inpainter package: protocol, registry, and built-in inpainters."""
import os
import shlex
import subprocess
import sys

from videowipe.inpainters.base import InpaintJob, InpaintOutcome, Inpainter
from videowipe.inpainters.registry import (
    InpainterRegistry,
    get_registry,
    register_inpainter,
)
from videowipe.external import ExternalInpainter
from videowipe.inpainters.sttn import STTNInpainter

# Register built-in inpainters. Importing this package makes "sttn" and
# "external" available to the registry.
register_inpainter("sttn", STTNInpainter)
register_inpainter("external", ExternalInpainter)


def _make_propainter_inpainter(propainter_dir=None, **_options):
    """Build an ExternalInpainter that runs the packaged ProPainter adapter.

    ProPainter is an optional external model; this factory only assembles the
    command string. The wrapper script resolves the ProPainter source directory
    from ``--propainter-dir`` / ``VIDEOWIPE_PROPINTER_DIR`` at run time. The
    instance ``name`` is overridden to ``"propainter"`` so benchmark
    output labels the model distinctly from a raw ``--external-command``.
    """
    parts = [sys.executable, "-m", "videowipe.propainter_wipe"]
    if propainter_dir:
        parts += ["--propainter-dir", propainter_dir]
    command = (
        subprocess.list2cmdline(parts)
        if os.name == "nt"
        else shlex.join(parts)
    )
    inpainter = ExternalInpainter(command=command)
    inpainter.name = "propainter"
    return inpainter


register_inpainter("propainter", _make_propainter_inpainter)

__all__ = [
    "Inpainter",
    "InpaintJob",
    "InpaintOutcome",
    "InpainterRegistry",
    "STTNInpainter",
    "ExternalInpainter",
    "get_registry",
    "register_inpainter",
]
