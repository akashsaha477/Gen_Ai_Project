"""Record-level (cluster) bootstrap and multiplicity correction (the evaluation code).

The unit of resampling is the record (patient), never the beat: beats of one patient are correlated
and a beat-level bootstrap would understate uncertainty. Each replicate draws as many records as the
data set has, with replacement, and pools their beats. With several seeds, the metric is averaged
over seeds inside each replicate.

A replicate in which the metric is undefined (a class with no true beats) is NaN. Such replicates are
excluded from intervals and p-values and their fraction is reported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ecgaug.data.records import CLASSES

MetricFn = Callable[[np.ndarray, np.ndarray], float]
StatFn = Callable[[np.ndarray], np.ndarray]


def draw_record_counts(n_records: int, n: int, seed: int) -> np.ndarray:
    """Multiplicity of every record in every replicate, ``[n, n_records]``; rows sum to ``n_records``."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_records, np.full(n_records, 1.0 / n_records), size=n)


def _seed_list(probs: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        return [probs]
    return [np.asarray(p) for p in probs]


def _interval(replicates: np.ndarray, level: float) -> tuple[float, float, np.ndarray]:
    defined = replicates[~np.isnan(replicates)]
    if defined.size == 0:
        return float("nan"), float("nan"), defined
    tail = (1 - level) / 2
    low, high = np.quantile(defined, [tail, 1 - tail])
    return float(low), float(high), defined


def _summary(point: float, replicates: np.ndarray, level: float) -> dict:
    low, high, defined = _interval(replicates, level)
    return {
        "point": float(point),
        "mean": float(defined.mean()) if defined.size else float("nan"),
        "se": float(defined.std(ddof=1)) if defined.size > 1 else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "level": level,
        "n_replicates": int(replicates.size),
        "frac_undefined": float(np.isnan(replicates).mean()),
    }


def class_presence(counts: np.ndarray, class_counts_by_record: np.ndarray, records: np.ndarray) -> dict:
    """How often classes, and the record holding most of each class, are missing from replicates.

    ``class_counts_by_record`` is ``[n_records, C]`` (true beats per record and class).
    """
    totals = counts @ class_counts_by_record
    out: dict = {"class_absent_frac": {}, "dominant_record": {}}
    for i, name in enumerate(CLASSES):
        out["class_absent_frac"][name] = float((totals[:, i] == 0).mean())
        column = class_counts_by_record[:, i]
        if column.sum() == 0:
            continue
        top = int(column.argmax())
        out["dominant_record"][name] = {
            "record": int(records[top]),
            "share_of_class": float(column[top] / column.sum()),
            "absent_frac": float((counts[:, top] == 0).mean()),
        }
    return out


def cluster_bootstrap(
    metric_fn: MetricFn,
    probs: np.ndarray | Sequence[np.ndarray],
    y: np.ndarray,
    records: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    return_replicates: bool = False,
) -> dict:
    """Cluster bootstrap of ``metric_fn(probs, y)`` over records.

    ``probs`` is one ``[n_beats, C]`` array or a sequence of them (one per seed, same beats).
    Returns the point estimate, percentile interval, standard error, the fraction of undefined
    replicates, and the class and dominant-record absence fractions.
    """
    seeds = _seed_list(probs)
    y, records = np.asarray(y), np.asarray(records)
    unique, inverse = np.unique(records, return_inverse=True)
    index = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    counts = draw_record_counts(len(unique), n, seed)

    point = float(np.mean([metric_fn(p, y) for p in seeds]))
    replicates = np.empty(n)
    for b in range(n):
        idx = np.concatenate([np.tile(index[k], c) for k, c in enumerate(counts[b]) if c])
        y_b = y[idx]
        replicates[b] = np.mean([metric_fn(p[idx], y_b) for p in seeds])

    class_counts = np.zeros((len(unique), len(CLASSES)), dtype=np.int64)
    np.add.at(class_counts, (inverse, y.astype(np.int64)), 1)
    out = _summary(point, replicates, level) | class_presence(counts, class_counts, unique)
    if return_replicates:
        out["replicates"] = replicates
    return out


def cluster_bootstrap_confusion(
    stat_fn: StatFn,
    cms: np.ndarray,
    records: np.ndarray | None = None,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    return_replicates: bool = False,
) -> dict:
    """Fast cluster bootstrap for statistics that depend only on the confusion matrix.

    ``cms`` is ``[n_records, C, C]`` or ``[n_seeds, n_records, C, C]``; ``stat_fn`` maps
    ``[..., C, C]`` to ``[...]``. Draws the same record multiplicities as :func:`cluster_bootstrap`
    for the same ``seed``, so both give identical replicates for confusion-matrix metrics.
    """
    cms = np.asarray(cms, dtype=np.float64)
    if cms.ndim == 3:
        cms = cms[None]
    n_records = cms.shape[1]
    counts = draw_record_counts(n_records, n, seed)
    point = float(np.mean(stat_fn(cms.sum(axis=1))))
    replicates = stat_fn(np.einsum("br,srij->bsij", counts.astype(np.float64), cms)).mean(axis=1)
    labels = np.arange(n_records) if records is None else np.asarray(records)
    out = _summary(point, replicates, level) | class_presence(counts, cms[0].sum(axis=-1).astype(np.int64), labels)
    if return_replicates:
        out["replicates"] = replicates
    return out


def _diff_summary(point: float, replicates: np.ndarray, level: float) -> dict:
    out = _summary(point, replicates, level)
    defined = replicates[~np.isnan(replicates)]
    if defined.size:
        tail = min((defined <= 0).sum() + 1, (defined >= 0).sum() + 1)
        out["p_two_sided"] = float(min(1.0, 2 * tail / (defined.size + 1)))
    else:
        out["p_two_sided"] = float("nan")
    return out


def paired_cluster_bootstrap_diff(
    metric_fn: MetricFn,
    probs_a: np.ndarray | Sequence[np.ndarray],
    probs_b: np.ndarray | Sequence[np.ndarray],
    y: np.ndarray,
    records: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    return_replicates: bool = False,
) -> dict:
    """Paired cluster bootstrap of ``metric(a) - metric(b)``: both conditions see the same records.

    The two-sided p-value is ``min(1, 2 * min(#{d <= 0} + 1, #{d >= 0} + 1) / (B + 1))`` over the
    ``B`` defined replicates.
    """
    seeds_a, seeds_b = _seed_list(probs_a), _seed_list(probs_b)
    y, records = np.asarray(y), np.asarray(records)
    unique, inverse = np.unique(records, return_inverse=True)
    index = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    counts = draw_record_counts(len(unique), n, seed)

    def mean_metric(seeds: list[np.ndarray], idx: np.ndarray | slice, labels: np.ndarray) -> float:
        return float(np.mean([metric_fn(p[idx], labels) for p in seeds]))

    point = mean_metric(seeds_a, slice(None), y) - mean_metric(seeds_b, slice(None), y)
    replicates = np.empty(n)
    for b in range(n):
        idx = np.concatenate([np.tile(index[k], c) for k, c in enumerate(counts[b]) if c])
        y_b = y[idx]
        replicates[b] = mean_metric(seeds_a, idx, y_b) - mean_metric(seeds_b, idx, y_b)
    out = _diff_summary(point, replicates, level)
    if return_replicates:
        out["replicates"] = replicates
    return out


def paired_cluster_bootstrap_diff_confusion(
    stat_fn: StatFn,
    cms_a: np.ndarray,
    cms_b: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    return_replicates: bool = False,
) -> dict:
    """Confusion-matrix version of :func:`paired_cluster_bootstrap_diff` (same draws for the same seed)."""
    a = np.asarray(cms_a, dtype=np.float64)
    b = np.asarray(cms_b, dtype=np.float64)
    a = a[None] if a.ndim == 3 else a
    b = b[None] if b.ndim == 3 else b
    if a.shape[1] != b.shape[1]:
        raise ValueError("Both conditions must cover the same records")
    counts = draw_record_counts(a.shape[1], n, seed).astype(np.float64)
    point = float(np.mean(stat_fn(a.sum(axis=1))) - np.mean(stat_fn(b.sum(axis=1))))
    rep_a = stat_fn(np.einsum("br,srij->bsij", counts, a)).mean(axis=1)
    rep_b = stat_fn(np.einsum("br,srij->bsij", counts, b)).mean(axis=1)
    replicates = rep_a - rep_b
    out = _diff_summary(point, replicates, level)
    if return_replicates:
        out["replicates"] = replicates
    return out


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down adjustment within one family of tests.

    Returns ``{name: {"p": raw, "p_adjusted": adjusted, "reject": bool}}``. Adjusted p-values are
    ``max_{j <= i} min(1, (m - j + 1) * p_(j))`` for the i-th smallest raw p-value.
    """
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(names)
    out, running = {}, 0.0
    for rank, name in enumerate(names):
        p = float(pvalues[name])
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = {"p": p, "p_adjusted": running, "reject": bool(running < alpha)}
    return {name: out[name] for name in pvalues}
