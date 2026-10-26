"""Generators of synthetic beats (Phase 3) and the synthetic banks they export.

``build_generator(family, settings, n_samples)`` makes a cVAE or a WGAN-GP from a config's merged
settings. Every generator has the same interface (``base.BeatGenerator``): ``fit`` on training beats
only, then ``sample(cls, n, seed)``.
"""

from __future__ import annotations

from ecgaug.generators.base import GENERATED_CLASSES, BeatGenerator, GeneratorFit, TrainOnlyError, assert_train_only
from ecgaug.generators.cvae import CVAE
from ecgaug.generators.wgan_gp import WGANGP
from ecgaug.train.loop import set_seed

FAMILIES: dict[str, type[BeatGenerator]] = {"cvae": CVAE, "wgan_gp": WGANGP}

__all__ = [
    "FAMILIES",
    "GENERATED_CLASSES",
    "BeatGenerator",
    "GeneratorFit",
    "TrainOnlyError",
    "assert_train_only",
    "build_generator",
]


def build_generator(family: str, settings: dict, n_samples: int, seed: int) -> BeatGenerator:
    """A generator of ``family`` ('cvae' or 'wgan_gp') for beats of ``n_samples`` samples.

    The initial weights come from ``seed``, so the same seed gives the same generator.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown generator family {family!r}; choose from {sorted(FAMILIES)}")
    set_seed(seed)
    return FAMILIES[family](settings, n_samples)
