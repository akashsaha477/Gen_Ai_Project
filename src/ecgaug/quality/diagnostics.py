"""Always-on diagnostics of synthetic beats (PLAN Phase 2, item 4), used by the phase 3 hard gates.

Each compares the synthetic beats of one class with the real training beats of the same class, on the
model's input (per-beat z-scored windows), with Euclidean distances:

- **Diversity ratio:** mean pairwise distance among synthetic beats divided by the same for real
  beats. 1 means as varied as real beats; a collapsed generator gives values near 0.
- **Memorization fraction:** fraction of synthetic beats whose nearest real training beat is closer
  than the 5th percentile of the real held-out-to-train distances. Every real training beat is held
  out in turn (leave-one-out): its distance to the nearest of the other training beats of its class.
  A generator that draws new beats like the real ones scores about 0.05; one that copies training
  beats scores near 1. Leave-one-out uses every beat, so the threshold is stable even for F (about 390
  training beats); a 10% hold-out set made it so noisy that an honest generator could fail the 10%
  ceiling by chance. Beats are held out one by one rather than by record because some classes sit
  almost entirely in one training record (F), and because the generator learns from these patients.
- **C2ST accuracy:** cross-validated accuracy of a logistic regression telling real from synthetic
  beats, on the samples plus their FFT magnitudes. 0.5 means indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecgaug.data.records import CLASS_INDEX


def mean_pairwise_distance(x: np.ndarray) -> float:
    """Mean Euclidean distance over all pairs of rows (NaN with fewer than two rows)."""
    return float(pdist(np.asarray(x, dtype=np.float64)).mean()) if len(x) > 1 else float("nan")


def nearest_distances(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Distance from each row of ``query`` to its nearest row of ``pool``."""
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(pool, dtype=np.float64))
    return nn.kneighbors(np.asarray(query, dtype=np.float64))[0][:, 0]


def leave_one_out_distances(x: np.ndarray) -> np.ndarray:
    """Distance from each row to its nearest other row (each row held out in turn)."""
    x = np.asarray(x, dtype=np.float64)
    return NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x)[0][:, 1]


@dataclass
class ClassReference:
    """Real training beats of one class, prepared once per run."""

    cls: str
    pool: np.ndarray  # all training beats of the class: the nearest-neighbour pool of the memorization test
    threshold: float  # percentile of the leave-one-out nearest-neighbour distances among them
    real_diversity: float  # mean pairwise distance among real training beats (a subsample)
    real_sample: np.ndarray  # real beats for the C2ST
    n_real: int


def prepare_reference(x_class: np.ndarray, cls: str, percentile: float, max_beats: int, seed: int) -> ClassReference:
    """Fix the memorization threshold and the real diversity of one class's training beats."""
    x_class = np.asarray(x_class, dtype=np.float32)
    if len(x_class) < 4:
        raise ValueError(f"Class {cls} has {len(x_class)} training beats; the diagnostics need at least 4")
    threshold = float(np.percentile(leave_one_out_distances(x_class), percentile))
    rng = np.random.default_rng(seed)
    sample = x_class[rng.choice(len(x_class), size=min(len(x_class), max_beats), replace=False)]
    return ClassReference(cls, x_class, threshold, mean_pairwise_distance(sample), sample, len(x_class))


def c2st_accuracy(real: np.ndarray, synthetic: np.ndarray, folds: int, seed: int) -> float:
    """Cross-validated accuracy of real-vs-synthetic logistic regression (equal numbers of each)."""
    n = min(len(real), len(synthetic))
    if n < folds:
        return float("nan")

    def features(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x[:n], dtype=np.float64)
        return np.hstack([x, np.abs(np.fft.rfft(x, norm="ortho", axis=1))])

    data = np.vstack([features(real), features(synthetic)])
    labels = np.r_[np.zeros(n), np.ones(n)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(model, data, labels, cv=cv, scoring="accuracy").mean())


def score_class(synthetic: np.ndarray, ref: ClassReference, max_beats: int, folds: int, seed: int) -> dict:
    """Diversity ratio, memorization fraction and C2ST accuracy of one class's synthetic beats."""
    synthetic = np.asarray(synthetic, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(synthetic) > max_beats:
        synthetic = synthetic[rng.choice(len(synthetic), size=max_beats, replace=False)]
    if len(synthetic) < 2:
        return {"n_scored": int(len(synthetic)), "diversity_ratio": float("nan"), "memorization_fraction": float("nan"), "c2st_accuracy": float("nan")}
    return {
        "n_scored": int(len(synthetic)),
        "diversity_ratio": mean_pairwise_distance(synthetic) / ref.real_diversity,
        "memorization_fraction": float((nearest_distances(synthetic, ref.pool) < ref.threshold).mean()),
        "c2st_accuracy": c2st_accuracy(ref.real_sample, synthetic, folds, seed),
    }


def prepare_references(train: dict[str, np.ndarray], classes: tuple[str, ...], settings: dict, seed: int) -> dict[str, ClassReference]:
    """One :class:`ClassReference` per class from the training beats (``X`` and ``y``)."""
    return {
        cls: prepare_reference(train["X"][train["y"] == CLASS_INDEX[cls]], cls, settings["memorization_percentile"], settings["max_beats"], seed)
        for cls in classes
    }


def score_beats(beats: dict[str, np.ndarray], references: dict[str, ClassReference], settings: dict, seed: int) -> dict[str, dict]:
    """Diagnostics per class of contract-format synthetic ``beats`` (fields ``X`` and ``y``)."""
    return {
        cls: score_class(beats["X"][beats["y"] == CLASS_INDEX[cls]], ref, settings["max_beats"], settings["c2st_folds"], seed)
        for cls, ref in references.items()
    }
