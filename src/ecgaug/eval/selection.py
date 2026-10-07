"""Selection rules shared by every gate (PREREGISTRATION §3 and §4).

The constants here are proposals until PREREGISTRATION v1 is in force; after that they change only
through an approved DECISIONS.md entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRIMARY = "rare_macro_f1"
FALLBACK = "s_f1"

# F enters the selection metric only if validation holds at least this much F support.
MIN_F_BEATS = 100
MIN_F_RECORDS = 2

TIE_FLOOR = 0.02


def selection_metric_name(n_f_beats_val: int, n_f_records_val: int) -> str:
    """Primary metric if validation has enough F support, else the fallback (S F1, V F1 tie-break)."""
    if n_f_beats_val >= MIN_F_BEATS and n_f_records_val >= MIN_F_RECORDS:
        return PRIMARY
    return FALLBACK


def selection_value(metrics: dict, name: str) -> float:
    """The scalar that early stopping and variant ranking maximise."""
    if name == PRIMARY:
        return float(metrics["rare_macro_f1"])
    if name == FALLBACK:
        return float(metrics["per_class"]["S"]["f1"])
    raise ValueError(f"Unknown selection metric {name!r}")


def tie_break_value(metrics: dict, name: str) -> float | None:
    """Secondary value used to break ties: V F1 under the fallback, nothing under the primary metric."""
    return float(metrics["per_class"]["V"]["f1"]) if name == FALLBACK else None


def tie_margin(sd_a: float, sd_b: float, floor: float = TIE_FLOOR) -> float:
    """max(floor, pooled across-seed SD of the two variants being compared)."""
    return max(floor, math.sqrt((sd_a**2 + sd_b**2) / 2))


@dataclass(frozen=True)
class Candidate:
    """A variant's validation summary: seed-mean and across-seed SD (ddof=1) of the selection metric."""

    name: str
    mean: float
    sd: float
    complexity: float
    wall_clock_s: float = 0.0
    tie_mean: float | None = None
    tie_sd: float | None = None


def _tied_with_best(candidates: list[Candidate], value, spread) -> list[Candidate]:
    best = max(candidates, key=value)
    return [c for c in candidates if value(best) - value(c) <= tie_margin(spread(best), spread(c))]


def select_winner(candidates: list[Candidate]) -> dict:
    """Apply the pre-registered selection rule literally.

    1. Take the candidate with the highest seed-mean selection metric.
    2. Every candidate within the tie margin of it is tied with it.
    3. If tie-break values are given (fallback metric), keep the tied candidates that are within the
       tie margin of the best tie-break value among them.
    4. Among those, the lowest complexity wins, then the lowest wall-clock, then the name.
    """
    if not candidates:
        raise ValueError("No candidates")
    if any(math.isnan(c.mean) for c in candidates):
        raise ValueError("A candidate has an undefined selection metric; the rule cannot be applied")
    tied = _tied_with_best(candidates, lambda c: c.mean, lambda c: c.sd)
    steps = [{"step": "selection metric", "tied": [c.name for c in tied]}]
    if len(tied) > 1 and all(c.tie_mean is not None for c in tied):
        tied = _tied_with_best(tied, lambda c: c.tie_mean, lambda c: c.tie_sd or 0.0)
        steps.append({"step": "tie-break metric", "tied": [c.name for c in tied]})
    winner = min(tied, key=lambda c: (c.complexity, c.wall_clock_s, c.name))
    steps.append({"step": "simplicity, then wall-clock", "winner": winner.name})
    best = max(candidates, key=lambda c: c.mean)
    return {
        "winner": winner.name,
        "best_by_mean": best.name,
        "margins": {c.name: tie_margin(best.sd, c.sd) for c in candidates},
        "steps": steps,
    }
