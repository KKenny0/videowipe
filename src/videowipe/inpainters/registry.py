"""Name-based registry for Inpainter implementations."""
from __future__ import annotations

from typing import Callable

from videowipe.inpainters.base import Inpainter


class InpainterRegistry:
    """Maps inpainter names to factory callables.

    Factories are called with keyword options by :meth:`create`; the built-in
    STTN factory accepts no options.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Inpainter]] = {}

    def register(self, name: str, factory: Callable[..., Inpainter]) -> None:
        if name in self._factories:
            raise ValueError(f"Inpainter already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, **options) -> Inpainter:
        if name not in self._factories:
            raise ValueError(
                f"Unknown inpainter: {name}. Available: {sorted(self._factories)}"
            )
        return self._factories[name](**options)

    def names(self) -> list[str]:
        return sorted(self._factories)


_registry = InpainterRegistry()


def register_inpainter(name: str, factory: Callable[..., Inpainter]) -> None:
    """Register an inpainter factory under *name*."""
    _registry.register(name, factory)


def get_registry() -> InpainterRegistry:
    return _registry
