"""The four metric families of phase2_quality_metrics (PLAN Phase 2).

| variant | family | value | better |
|---|---|---|---|
| A | ``dtw`` | mean DTW distance from each synthetic beat to its 5 nearest real training beats of the class, divided by the same for real beats (each held out in turn); averaged over S, V, F | lower |
| B | ``spectral`` | Jensen-Shannon distance between the class-mean normalised Welch spectra of real and synthetic beats; averaged over classes | lower |
| C | ``fid_analog`` | Frechet distance between Gaussians fitted to frozen feature-network features of real and synthetic beats, divided by the real-vs-real distance at the same sample size; averaged over classes | lower |
| D | ``tstr`` | validation selection metric of a classifier trained with the real minority training beats replaced by synthetic ones, divided by the same classifier trained on real beats | higher |

DTW is computed here with a banded dynamic programme vectorised over beat pairs (Sakoe-Chiba window,
squared differences, square root at the end, as in dtaidistance), because the compiled dtaidistance
library is not available everywhere the project runs. To keep it affordable the 5 nearest beats by DTW
are searched among the 20 nearest by Euclidean distance.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors

from ecgaug.data.records import CLASS_INDEX
from ecgaug.generators.base import GENERATED_CLASSES
from ecgaug.quality import MetricFamily, QualityContext, register


def _classes(synthetic: dict) -> list[str]:
    y = np.asarray(synthetic["y"])
    return [c for c in GENERATED_CLASSES if (y == CLASS_INDEX[c]).any()]


def _of(arrays: dict, cls: str) -> np.ndarray:
    return np.asarray(arrays["X"], dtype=np.float64)[np.asarray(arrays["y"]) == CLASS_INDEX[cls]]


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return x if len(x) <= n else x[rng.choice(len(x), size=n, replace=False)]


# --------------------------------------------------------------------------------------------------
# A: DTW
# --------------------------------------------------------------------------------------------------


def dtw_pairs(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """DTW distance between row i of ``a`` and row i of ``b`` for every i (equal lengths, band ``window``)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n_pairs, length = a.shape
    previous = np.full((n_pairs, length + 1), np.inf)
    previous[:, 0] = 0.0
    for i in range(1, length + 1):
        current = np.full((n_pairs, length + 1), np.inf)
        lo, hi = max(1, i - window), min(length, i + window)
        cost = (a[:, i - 1 : i] - b[:, lo - 1 : hi]) ** 2
        for j in range(lo, hi + 1):
            current[:, j] = cost[:, j - lo] + np.minimum(np.minimum(previous[:, j], current[:, j - 1]), previous[:, j - 1])
        previous = current
    return np.sqrt(previous[:, length])


def mean_nearest_dtw(query: np.ndarray, pool: np.ndarray, settings: dict, exclude_self: bool = False) -> float:
    """Mean over query beats of the mean DTW distance to their ``neighbours`` nearest pool beats."""
    k_euclid, k_dtw = int(settings["dtw_candidates"]), int(settings["dtw_neighbours"])
    window = max(1, int(round(settings["dtw_window_fraction"] * query.shape[1])))
    extra = 1 if exclude_self else 0
    index = NearestNeighbors(n_neighbors=min(len(pool), k_euclid + extra)).fit(pool).kneighbors(query)[1]
    if exclude_self:
        index = index[:, 1:]  # each query beat is in the pool; drop it (its own nearest neighbour)
    rows = np.repeat(np.arange(len(query)), index.shape[1])
    distances = dtw_pairs(query[rows], pool[index.ravel()], window).reshape(index.shape)
    return float(np.sort(distances, axis=1)[:, :k_dtw].mean())


def dtw_score(context: QualityContext, synthetic: dict) -> float:
    s, rng = context.settings, np.random.default_rng(context.seed)
    ratios = []
    for cls in _classes(synthetic):
        pool = _of(context.train, cls)
        key = ("dtw_reference", cls)
        if key not in context.cache:
            context.cache[key] = mean_nearest_dtw(_subsample(pool, s["dtw_max_beats"], rng), pool, s, exclude_self=True)
        ratios.append(mean_nearest_dtw(_subsample(_of(synthetic, cls), s["dtw_max_beats"], rng), pool, s) / context.cache[key])
    return float(np.mean(ratios))


# --------------------------------------------------------------------------------------------------
# B: spectral
# --------------------------------------------------------------------------------------------------


def mean_spectrum(x: np.ndarray, fs: float, nperseg: int) -> np.ndarray:
    """Class-mean Welch power spectrum, each beat's spectrum normalised to sum 1 first."""
    _, power = welch(x, fs=fs, nperseg=min(nperseg, x.shape[1]), axis=1)
    power = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    return power.mean(axis=0)


def spectral_score(context: QualityContext, synthetic: dict) -> float:
    fs = float(getattr(context.spec, "effective_fs", 360))
    nperseg = int(context.settings["spectral_nperseg"])
    distances = []
    for cls in _classes(synthetic):
        key = ("spectrum", cls)
        if key not in context.cache:
            context.cache[key] = mean_spectrum(_of(context.train, cls), fs, nperseg)
        distances.append(float(jensenshannon(context.cache[key], mean_spectrum(_of(synthetic, cls), fs, nperseg), base=2)))
    return float(np.mean(distances))


# --------------------------------------------------------------------------------------------------
# C: FID-analog
# --------------------------------------------------------------------------------------------------


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to the rows of ``a`` and of ``b``."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)  # near-singular covariances: the real part is used
        root = linalg.sqrtm(cov_a @ cov_b)
    root = np.real(root[0] if isinstance(root, tuple) else root)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def _features(context: QualityContext, x: np.ndarray) -> np.ndarray:
    import torch

    net = context.feature_net
    if net is None:
        raise ValueError("FID-analog needs the frozen feature network (QualityContext.feature_net)")
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            out.append(net.embed_waveform(torch.as_tensor(x[start : start + 2048], dtype=torch.float32)).numpy())
    return np.concatenate(out).astype(np.float64)


def fid_score(context: QualityContext, synthetic: dict) -> float:
    rng = np.random.default_rng(context.seed)
    ratios = []
    for cls in _classes(synthetic):
        real, synth = _of(context.train, cls), _of(synthetic, cls)
        n = min(len(real) // 2, len(synth), int(context.settings["fid_max_beats"]))
        if n < 10:
            return float("nan")
        order = rng.permutation(len(real))
        real_a, real_b = _features(context, real[order[:n]]), _features(context, real[order[n : 2 * n]])
        fake = _features(context, _subsample(synth, n, rng))
        reference = frechet_distance(real_a, real_b)
        ratios.append(frechet_distance(real_a, fake) / max(reference, 1e-12))
    return float(np.mean(ratios))


# --------------------------------------------------------------------------------------------------
# D: TSTR
# --------------------------------------------------------------------------------------------------


def _train_and_score(context: QualityContext, train: dict) -> float:
    from ecgaug.experiments.common import SIDE_FIELDS, split_part
    from ecgaug.models import build_model
    from ecgaug.train.loop import TrainConfig, fit_classifier, resolve_device, set_seed

    c = context.settings["tstr_classifier"]
    rr = bool(getattr(context.spec, "rr_features", False))
    part = split_part(train, "train", rr)
    set_seed(int(c["seed"]))
    model = build_model(c["model"], 4, len(SIDE_FIELDS) if rr else 0)
    cfg = TrainConfig(lr=float(c["lr"]), max_epochs=int(c["max_epochs"]), patience=int(c["patience"]), batch_size=int(c["batch_size"]),
                      weight_decay=float(c["weight_decay"]))  # fmt: skip
    return fit_classifier(model, part, context.val, cfg, context.selection, seed=int(c["seed"]), device=resolve_device(context.device)).best_value


def replace_minority(train: dict, synthetic: dict, seed: int) -> dict:
    """The training beats with each synthetic class's real beats swapped for as many synthetic ones."""
    from ecgaug.data.contract import FIELDS

    rng = np.random.default_rng(seed)
    y, y_syn = np.asarray(train["y"]), np.asarray(synthetic["y"])
    keep = np.ones(len(y), dtype=bool)
    parts = []
    for cls in _classes(synthetic):
        real = y == CLASS_INDEX[cls]
        keep &= ~real
        pool = np.flatnonzero(y_syn == CLASS_INDEX[cls])
        pick = rng.choice(pool, size=int(real.sum()), replace=len(pool) < real.sum())
        parts.append({f: np.asarray(synthetic[f])[pick] for f in FIELDS})
    out = {f: np.concatenate([np.asarray(train[f])[keep], *[p[f] for p in parts]]) for f in FIELDS}
    out["split"] = np.full(len(out["y"]), "train", dtype="<U5")
    return out


def tstr_score(context: QualityContext, synthetic: dict) -> float:
    if "trtr" not in context.cache:
        context.cache["trtr"] = _train_and_score(context, context.train)
    if context.cache["trtr"] <= 0:
        return float("nan")
    return _train_and_score(context, replace_minority(context.train, synthetic, context.seed)) / context.cache["trtr"]


register(MetricFamily("dtw", dtw_score, higher_is_better=False))
register(MetricFamily("spectral", spectral_score, higher_is_better=False))
register(MetricFamily("fid_analog", fid_score, higher_is_better=False))
register(MetricFamily("tstr", tstr_score, higher_is_better=True))
