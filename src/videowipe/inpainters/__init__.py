"""Inpainter package: protocol, registry, and built-in STTN inpainter."""
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
