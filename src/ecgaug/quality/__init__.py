"""Quality of synthetic beats: the always-on diagnostics and the four metric families.

- ``diagnostics.py``: diversity, memorization and C2ST (always computed; the phase 3 hard gates).
- ``metrics.py``: the metric families of phase2_quality_metrics, DTW, spectral, FID-analog and TSTR,
  registered in ``METRIC_FAMILIES``.
- ``sanity.py``: the sanity harness that decides which families pass (PLAN Phase 2).

A family scores a whole set of synthetic beats (contract format, classes S, V and F) against the real
training beats in a :class:`QualityContext`. The phase 2 gate fixes which families form the composite
quality rank used to break ties at the phase 3a gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityContext:
    """Everything a metric family needs besides the synthetic beats."""

    train: dict[str, np.ndarray]  # real training beats, contract format: the reference
    settings: dict  # the ``metrics`` block of configs/phase2_quality_metrics/metric_settings.yaml
    val: dict[str, np.ndarray] | None = None  # validation part (X, y, side, record, r_idx); TSTR only
    spec: object | None = None  # RepresentationSpec of the beats
    selection: str | None = None  # validation selection metric; TSTR only
    feature_net: object | None = None  # frozen feature network; FID-analog only
    device: str = "cpu"
    seed: int = 0
    cache: dict = field(default_factory=dict)  # per-context values that do not depend on the synthetic beats


@dataclass(frozen=True)
class MetricFamily:
    """A quality metric over a set of synthetic beats."""

    name: str
    evaluate: Callable[[QualityContext, dict], float]
    higher_is_better: bool


METRIC_FAMILIES: dict[str, MetricFamily] = {}


def register(family: MetricFamily) -> MetricFamily:
    METRIC_FAMILIES[family.name] = family
    return family


def composite_ranks(values: dict[str, dict[str, float]], families: list[str]) -> dict[str, float] | None:
    """Equal-weight average of per-metric ranks across candidate generators (1 = best).

    ``values`` maps a candidate to its metric values. Returns None if no family is given or a value
    is missing, because then the composite is not defined.
    """
    if not families or any(f not in METRIC_FAMILIES for f in families):
        return None
    names = sorted(values)
    if any(f not in values[n] or not np.isfinite(values[n][f]) for n in names for f in families):
        return None
    ranks = {n: [] for n in names}
    for family in families:
        sign = -1.0 if METRIC_FAMILIES[family].higher_is_better else 1.0
        ordered = sorted(names, key=lambda n: sign * values[n][family])
        for position, name in enumerate(ordered, start=1):
            ranks[name].append(position)
    return {n: float(np.mean(r)) for n, r in ranks.items()}


from ecgaug.quality import metrics as _metrics  # noqa: E402,F401  (registers the four families)
