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
       tie margin of the best tie-break value among them. An undefined (NaN) tie-break value counts as
       not given: the step is skipped and the report says so (DECISIONS A17).
    4. Among those, the lowest complexity wins, then the lowest wall-clock, then the name.
    """
    if not candidates:
        raise ValueError("No candidates")
    if any(math.isnan(c.mean) for c in candidates):
        raise ValueError("A candidate has an undefined selection metric; the rule cannot be applied")
    tied = _tied_with_best(candidates, lambda c: c.mean, lambda c: c.sd)
    steps = [{"step": "selection metric", "tied": [c.name for c in tied]}]
    undefined = [c.name for c in tied if c.tie_mean is not None and math.isnan(c.tie_mean)]
    if len(tied) > 1 and undefined:
        steps.append({"step": "tie-break metric", "skipped": f"undefined for {', '.join(undefined)}"})
    elif len(tied) > 1 and all(c.tie_mean is not None for c in tied):
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


GENERATOR_FALLBACK = (
    "No variant passes the hard gates. PLAN Phase 3 fallback: the team restricts generation to the 2-3 "
    "rarest classes or adopts variant D, and records the failure as a result."
)


@dataclass(frozen=True)
class GeneratorCandidate:
    """A generator variant's pilot summary (PLAN Phase 3 selection)."""

    name: str
    mean: float  # seed-mean pilot validation selection metric at the final checkpoint
    sd: float
    passed_hard_gates: bool
    n_parameters: int
    wall_clock_s: float = 0.0
    tie_mean: float | None = None  # V F1 under the fallback metric
    tie_sd: float | None = None
    quality_rank: float | None = None  # composite quality rank, lower is better; None until it is defined


def select_generator(candidates: list[GeneratorCandidate]) -> dict:
    """Apply the phase 3 selection rule literally (PLAN Phase 3, PREREGISTRATION §9).

    1. Hard gates: variants that fail them are out.
    2. The passing variant with the highest seed-mean pilot selection metric; every passing variant
       within the tie margin (§4) of it is tied with it.
    3. Ties are broken by the composite quality rank. Until the phase 2 gate defines the composite,
       the general tie-break of §4 is used instead (V F1 under the fallback metric), and the report
       says so.
    4. Then fewer parameters, then lower wall-clock, then the name.
    """
    if not candidates:
        raise ValueError("No candidates")
    passing = [c for c in candidates if c.passed_hard_gates]
    steps: list[dict] = [{"step": "hard gates", "passed": [c.name for c in passing], "failed": [c.name for c in candidates if not c.passed_hard_gates]}]
    if not passing:
        return {"winner": None, "best_by_mean": None, "margins": {}, "steps": steps, "fallback": GENERATOR_FALLBACK}
    if any(math.isnan(c.mean) for c in passing):
        raise ValueError("A passing candidate has an undefined pilot selection metric; the rule cannot be applied")
    tied = _tied_with_best(passing, lambda c: c.mean, lambda c: c.sd)
    steps.append({"step": "pilot selection metric", "tied": [c.name for c in tied]})
    if len(tied) > 1:
        if all(c.quality_rank is not None for c in tied):
            best_rank = min(c.quality_rank for c in tied)
            tied = [c for c in tied if c.quality_rank == best_rank]
            steps.append({"step": "composite quality rank", "tied": [c.name for c in tied]})
        elif all(c.tie_mean is not None for c in tied):
            tied = _tied_with_best(tied, lambda c: c.tie_mean, lambda c: c.tie_sd or 0.0)
            steps.append({"step": "tie-break metric (the composite quality rank is not defined yet)", "tied": [c.name for c in tied]})
    winner = min(tied, key=lambda c: (c.n_parameters, c.wall_clock_s, c.name))
    steps.append({"step": "fewer parameters, then wall-clock", "winner": winner.name})
    best = max(passing, key=lambda c: c.mean)
    return {
        "winner": winner.name,
        "best_by_mean": best.name,
        "margins": {c.name: tie_margin(best.sd, c.sd) for c in passing},
        "steps": steps,
        "fallback": None,
    }
