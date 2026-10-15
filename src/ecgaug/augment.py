"""The conditions generative augmentation is compared with, and the policy that sets how many beats are added.

- Target counts: how many beats to add to each minority class (``fraction_of_n_targets``; the phase 3
  pilot policy is "minority classes up to 50% of N").
- Sources of added beats: a synthetic bank (``take_from_bank``), SMOTE (``smote_beats``) and random
  oversampling of real minority beats (``random_oversample``), all in the data-contract format.
- Class-weighted cross-entropy: ``class_weights`` with w_c = (n_max / n_c) ** gamma.

Validation and test beats are never touched: every function takes the training beats only and checks
it, and added beats are marked as training beats.
"""

from __future__ import annotations

import numpy as np

from ecgaug.data.contract import FIELDS
from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.data.representation import zscore_per_beat
from ecgaug.generators.bank import RR_FIELDS, BankError, concat_parts, synthetic_part
from ecgaug.generators.base import GENERATED_CLASSES, assert_train_only


def class_counts(y: np.ndarray) -> dict[str, int]:
    y = np.asarray(y)
    return {c: int((y == CLASS_INDEX[c]).sum()) for c in CLASSES}


def fraction_of_n_targets(counts: dict[str, int], fraction: float, classes: tuple[str, ...] = GENERATED_CLASSES) -> dict[str, int]:
    """Beats to add so each class in ``classes`` reaches ``fraction`` x the N count (0 if it already does)."""
    goal = int(round(fraction * counts["N"]))
    return {c: max(0, goal - counts[c]) for c in classes}


POLICIES = ("balance_to_n", "fraction_of_n", "multiple_of_real", "fraction_of_n_screened")


def policy_targets(policy: str, strength: float | None, counts: dict[str, int], classes: tuple[str, ...] = GENERATED_CLASSES) -> dict[str, int]:
    """Beats to add per minority class under an augmentation policy (PLAN Phase 4, phase4_augmentation_policy).

    - ``balance_to_n``: each class up to the N count (no strength);
    - ``fraction_of_n`` and ``fraction_of_n_screened``: each class up to ``strength`` x the N count;
    - ``multiple_of_real``: ``strength`` x the class's real count added.
    """
    if policy == "balance_to_n":
        return fraction_of_n_targets(counts, 1.0, classes)
    if policy in ("fraction_of_n", "fraction_of_n_screened"):
        return fraction_of_n_targets(counts, float(strength), classes)
    if policy == "multiple_of_real":
        return {c: int(round(float(strength) * counts[c])) for c in classes}
    raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")


def screened_from_bank(bank: dict[str, np.ndarray], extra: dict[str, int], train: dict[str, np.ndarray], pool_factor: float,
                       settings: dict) -> dict[str, np.ndarray]:  # fmt: skip
    """Policy D's quality screen: of the first ``pool_factor`` x ``extra[c]`` bank beats of each class,
    keep the ``extra[c]`` closest to real training beats of the class (mean DTW distance to the nearest
    real beats, ``ecgaug.quality.metrics``). If the bank holds fewer beats, the whole class is the pool."""
    from ecgaug.quality.metrics import nearest_dtw_per_beat

    parts = []
    for cls, k in extra.items():
        if k <= 0:
            continue
        idx = np.flatnonzero(np.asarray(bank["y"]) == CLASS_INDEX[cls])
        if idx.size < k:
            raise BankError(f"The bank holds {idx.size} {cls} beats; {k} are needed")
        pool = idx[: max(k, min(idx.size, int(round(pool_factor * k))))]
        real = np.asarray(train["X"], dtype=np.float64)[np.asarray(train["y"]) == CLASS_INDEX[cls]]
        distance = nearest_dtw_per_beat(np.asarray(bank["X"], dtype=np.float64)[pool], real, settings)
        keep = pool[np.sort(np.argsort(distance, kind="stable")[:k])]
        parts.append({name: np.asarray(bank[name])[keep] for name in FIELDS})
    return concat_parts(parts)


def take_from_bank(bank: dict[str, np.ndarray], extra: dict[str, int]) -> dict[str, np.ndarray]:
    """The first ``extra[c]`` beats of each class in bank order (a bank is stored in random order)."""
    parts = []
    for cls, k in extra.items():
        if k <= 0:
            continue
        idx = np.flatnonzero(np.asarray(bank["y"]) == CLASS_INDEX[cls])
        if idx.size < k:
            raise BankError(f"The bank holds {idx.size} {cls} beats; {k} are needed")
        parts.append({name: np.asarray(bank[name])[idx[:k]] for name in FIELDS})
    return concat_parts(parts)


def random_oversample(train: dict[str, np.ndarray], extra: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Copies of real minority training beats drawn with replacement (the matched-count control).

    Copies keep their record and R-peak index (they are real beats) and are marked
    ``is_synthetic=False``, ``source='random_oversampling'``.
    """
    assert_train_only(train)
    rng = np.random.default_rng(seed)
    parts = []
    for cls, k in extra.items():
        if k <= 0:
            continue
        pool = np.flatnonzero(np.asarray(train["y"]) == CLASS_INDEX[cls])
        if not pool.size:
            raise ValueError(f"No training beats of class {cls} to oversample")
        pick = rng.choice(pool, size=k, replace=True)
        part = {name: np.asarray(train[name])[pick] for name in FIELDS}
        part["is_synthetic"] = np.zeros(k, dtype=bool)
        part["source"] = np.full(k, "random_oversampling")
        parts.append(part)
    return concat_parts(parts)


def smote_beats(train: dict[str, np.ndarray], extra: dict[str, int], seed: int, k_neighbors: int = 5) -> dict[str, np.ndarray]:
    """SMOTE beats (imbalanced-learn), fitted on training beats only.

    Each beat is interpolated on the classifier's whole input: the waveform plus the three RR
    features, the RR features standardised with training statistics for the neighbour search. The
    waveform is per-beat z-scored again afterwards, as the contract requires.
    """
    from imblearn.over_sampling import SMOTE

    assert_train_only(train)
    wanted = {c: k for c, k in extra.items() if k > 0}
    if not wanted:
        return concat_parts([])
    y = np.asarray(train["y"], dtype=np.int64)
    rr = np.stack([np.asarray(train[f], dtype=np.float64) for f in RR_FIELDS], axis=1)
    rr_mean, rr_std = rr.mean(axis=0), np.maximum(rr.std(axis=0), 1e-9)
    features = np.hstack([np.asarray(train["X"], dtype=np.float64), (rr - rr_mean) / rr_std])
    smallest = min(int((y == CLASS_INDEX[c]).sum()) for c in wanted)
    if smallest < 2:
        raise ValueError("SMOTE needs at least two training beats of every class it adds to")
    strategy = {CLASS_INDEX[c]: int((y == CLASS_INDEX[c]).sum()) + k for c, k in wanted.items()}
    sampler = SMOTE(sampling_strategy=strategy, k_neighbors=min(k_neighbors, smallest - 1), random_state=seed)
    resampled, labels = sampler.fit_resample(features, y)
    new, new_y = resampled[len(y) :], labels[len(y) :]
    n_samples = train["X"].shape[1]
    parts = []
    for cls in wanted:
        rows = new[new_y == CLASS_INDEX[cls]]
        x = zscore_per_beat(rows[:, :n_samples]).astype(np.float32)
        rr_new = rows[:, n_samples:] * rr_std + rr_mean
        parts.append(synthetic_part(x, cls, {f: rr_new[:, i] for i, f in enumerate(RR_FIELDS)}, "smote"))
    return concat_parts(parts)


def with_added(train: dict[str, np.ndarray], added: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Training beats plus added beats, as one contract-format set of training beats."""
    assert_train_only(train)
    if len(added["y"]) and not (np.asarray(added["split"]) == "train").all():
        raise ValueError("Added beats must be marked as training beats")
    if len(added["y"]) and added["X"].shape[1] != train["X"].shape[1]:
        raise ValueError(f"Added beats have {added['X'].shape[1]} samples; the training beats have {train['X'].shape[1]}")
    return concat_parts([{name: np.asarray(train[name]) for name in FIELDS}, added])


def class_weights(counts: dict[str, int], gamma: float) -> np.ndarray:
    """Cross-entropy weights w_c = (n_max / n_c) ** gamma in class order N, S, V, F (PLAN Phase 3, item 4)."""
    n = np.array([counts[c] for c in CLASSES], dtype=np.float64)
    if (n <= 0).any():
        raise ValueError(f"Every class needs training beats for class weights: {counts}")
    return (n.max() / n) ** float(gamma)
