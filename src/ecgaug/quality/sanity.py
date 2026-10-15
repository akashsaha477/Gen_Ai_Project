"""The sanity harness of phase2_quality_metrics (PLAN Phase 2: quality interface, sanity harness, selection).

Real training beats are split into two halves, stratified by class. The split is by beat, not by
record, because F sits almost entirely in one training record. The reference half plays the real
training beats; the S, V and F beats of the held-out half play a synthetic set. A metric family is
scored on the held-out beats as they are (the real-vs-real reference) and on four degradations of
them, each at three severities:

- ``noise``: Gaussian noise with SD 0.1, 0.3 and 1.0 (beats are z-scored), z-scored again;
- ``label_shuffle``: 25%, 50% and 100% of each class's beats replaced by held-out beats of other classes;
- ``time_shift``: beats moved later by 5, 15 and 40 samples (the first value repeated);
- ``smoothing``: moving average over 3, 9 and 27 samples, z-scored again.

A family passes if (1) the unchanged held-out beats score best of all sets; (2) for every degradation
the score gets worse with severity: Spearman correlation of at least 0.8 between severity 0-3 and
badness, so at most one adjacent pair may swap; (3) it is stable: the median coefficient of
variation of the degraded scores over 3 resamples (new random halves) is below 10%.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ecgaug.data.contract import FIELDS
from ecgaug.data.records import CLASS_INDEX
from ecgaug.data.representation import zscore_per_beat
from ecgaug.generators.base import GENERATED_CLASSES
from ecgaug.quality import METRIC_FAMILIES, QualityContext

DEFAULT_LEVELS = {"noise": [0.1, 0.3, 1.0], "label_shuffle": [0.25, 0.5, 1.0], "time_shift": [5, 15, 40], "smoothing": [3, 9, 27]}


def halves(train: dict[str, np.ndarray], seed: int) -> tuple[dict, dict]:
    """Two halves of the training beats, stratified by class: (reference, held-out)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(train["y"])
    first = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        first[idx[: len(idx) // 2]] = True
    take = lambda mask: {f: np.asarray(train[f])[mask] for f in FIELDS}  # noqa: E731
    return take(first), take(~first)


def minority(part: dict) -> dict:
    mask = np.isin(np.asarray(part["y"]), [CLASS_INDEX[c] for c in GENERATED_CLASSES])
    return {f: np.asarray(part[f])[mask] for f in FIELDS}


def degrade(beats: dict, kind: str, level: float, rng: np.random.Generator, donors: dict) -> dict:
    """A degraded copy of contract-format ``beats``; ``donors`` supplies other-class beats for label shuffle."""
    out = {f: np.array(beats[f], copy=True) for f in FIELDS}
    x = out["X"].astype(np.float64)
    if kind == "noise":
        x = zscore_per_beat(x + level * rng.standard_normal(x.shape))
    elif kind == "smoothing":
        width = int(level)
        padded = np.pad(x, ((0, 0), (width // 2, width - 1 - width // 2)), mode="edge")
        x = zscore_per_beat(np.stack([np.convolve(row, np.ones(width) / width, mode="valid") for row in padded]))
    elif kind == "time_shift":
        shift = int(level)
        x = np.concatenate([np.repeat(x[:, :1], shift, axis=1), x[:, : x.shape[1] - shift]], axis=1)
    elif kind == "label_shuffle":
        y_donor = np.asarray(donors["y"])
        for cls in GENERATED_CLASSES:
            rows = np.flatnonzero(out["y"] == CLASS_INDEX[cls])
            replace = rng.choice(rows, size=int(round(level * len(rows))), replace=False)
            pool = np.flatnonzero(y_donor != CLASS_INDEX[cls])
            x[replace] = np.asarray(donors["X"], dtype=np.float64)[rng.choice(pool, size=len(replace))]
    else:
        raise ValueError(f"Unknown degradation {kind!r}")
    out["X"] = x.astype(np.float32)
    out["is_synthetic"] = np.ones(len(out["y"]), dtype=bool)
    out["source"] = np.full(len(out["y"]), f"sanity_{kind}")
    return out


def sanity_sets(heldout: dict, levels: dict, rng: np.random.Generator) -> list[dict]:
    """The reference set and every degradation at every severity."""
    beats = minority(heldout)
    sets = [{"kind": "reference", "level": 0, "severity": 0, "beats": beats}]
    for kind, values in levels.items():
        for severity, level in enumerate(values, start=1):
            sets.append({"kind": kind, "level": level, "severity": severity, "beats": degrade(beats, kind, level, rng, heldout)})
    return sets


def score_family(family: str, train: dict, settings: dict, levels: dict, resamples: int, rule: dict, **context_kwargs) -> dict:
    """Score one metric family on every sanity set for every resample and apply the pass rule."""
    metric = METRIC_FAMILIES[family]
    scores: dict[str, dict] = {}
    for r in range(resamples):
        reference, heldout = halves(train, seed=r)
        context = QualityContext(train=reference, settings=settings, seed=r, **context_kwargs)
        rng = np.random.default_rng(1000 + r)
        for item in sanity_sets(heldout, levels, rng):
            key = f"{item['kind']}:{item['level']:g}"
            entry = scores.setdefault(key, {"kind": item["kind"], "level": item["level"], "severity": item["severity"], "values": []})
            entry["values"].append(float(metric.evaluate(context, item["beats"])))
    for entry in scores.values():
        values = np.asarray(entry["values"])
        entry["mean"] = float(values.mean())
        entry["sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        entry["cv"] = float(entry["sd"] / abs(entry["mean"])) if entry["mean"] else float("inf")
    sign = -1.0 if metric.higher_is_better else 1.0
    badness = {k: sign * v["mean"] for k, v in scores.items()}
    reference_key = "reference:0"
    reference_best = all(badness[reference_key] < b for k, b in badness.items() if k != reference_key)
    spearman = {}
    for kind in levels:
        keys = [reference_key] + [k for k, v in scores.items() if v["kind"] == kind]
        rho = spearmanr([scores[k]["severity"] for k in keys], [badness[k] for k in keys]).statistic
        # rounded: with 4 points one adjacent swap gives exactly 0.8, which the rule (>= 0.8) accepts
        spearman[kind] = round(float(rho), 10) if np.isfinite(rho) else float("nan")
    cvs = [v["cv"] for k, v in scores.items() if k != reference_key]
    median_cv = float(np.median(cvs))
    checks = [
        {"check": "the unchanged held-out beats score best", "passed": bool(reference_best)},
        *[{"check": f"{kind}: worse with severity (Spearman {rho:.2f} >= {rule['spearman_min']})", "passed": bool(rho >= rule["spearman_min"])}
          for kind, rho in spearman.items()],
        {"check": f"stable over {resamples} resamples (median CV {median_cv:.3f} < {rule['cv_max']})", "passed": bool(median_cv < rule["cv_max"])},
    ]  # fmt: skip
    return {
        "family": family, "higher_is_better": metric.higher_is_better, "scores": scores, "spearman": spearman,
        "reference_best": bool(reference_best), "median_cv": median_cv, "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }  # fmt: skip
