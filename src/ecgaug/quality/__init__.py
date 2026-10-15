"""Quality of synthetic beats: the always-on diagnostics and the slot for the metric families.

The diagnostics (``diagnostics.py``: diversity, memorization, C2ST) are always computed. The metric
families (DTW, spectral, FID-analog, TSTR) belong to phase2_quality_metrics, which is not built yet;
the phase 2 gate fixes which of them form the composite quality rank. Until then ``METRIC_FAMILIES``
is empty, the composite is not available, and phase 3 ties fall back to the next tie-break.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricFamily:
    """A quality metric: ``score(real, synthetic) -> float`` for the beats of one class."""

    name: str
    score: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool


# Filled by phase2_quality_metrics; the phase 2 gate names the families that form the composite.
METRIC_FAMILIES: dict[str, MetricFamily] = {}


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
