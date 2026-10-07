"""Data pipeline: raw access behind the test lock, beat extraction, splits, contract, loaders.

Submodules are imported lazily so that ``import ecgaug.data`` stays cheap and free of import cycles.
"""

from __future__ import annotations

__all__ = ["load_candidate", "load_test", "load_trainval"]


def __getattr__(name: str):
    if name in __all__:
        from ecgaug.data import loaders

        return getattr(loaders, name)
    raise AttributeError(f"module 'ecgaug.data' has no attribute {name!r}")
