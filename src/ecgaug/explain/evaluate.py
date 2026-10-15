"""How good a saliency map is (PLAN Phase 5, phase5_explainability evaluation).

- **Faithfulness (deletion):** the top-k% most salient samples of a beat (by absolute saliency) are
  replaced by linear interpolation between the samples that remain, for k = 0, 10, ..., 50%; the drop
  in the probability of the predicted class is integrated over k. The same is done in random order
  (averaged over a few orders). The faithfulness gain is the saliency AUC minus the random AUC: above 0
  means the map points at samples the model relies on.
- **Stability:** Spearman correlation between the saliency maps of the same beats from models trained
  with different seeds, averaged over beats and seed pairs.
- **Physiological alignment (descriptive):** share of absolute saliency in fixed R-relative windows,
  P [-250, -100] ms, QRS [-50, 50] ms, ST [50, 150] ms, T [150, 400] ms, each clipped to the beat window.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr

SEGMENTS_MS = {"P": (-250, -100), "QRS": (-50, 50), "ST": (50, 150), "T": (150, 400)}


def interpolate_out(x: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Copy of beat ``x`` with the samples in ``remove`` replaced by linear interpolation of the others."""
    keep = np.setdiff1d(np.arange(len(x)), remove)
    out = x.copy()
    if keep.size == 0:
        out[:] = 0.0
    elif remove.size:
        out[remove] = np.interp(remove, keep, x[keep])
    return out


def _probabilities(model, x: np.ndarray, side: np.ndarray | None, target: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32), None if side is None else torch.as_tensor(side, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()[np.arange(len(target)), target]


def deletion_auc(model, x: np.ndarray, side: np.ndarray | None, orders: np.ndarray, fractions: list[float]) -> np.ndarray:
    """Area under the drop in predicted-class probability as the first k% of ``orders`` are removed, per beat."""
    model.cpu().eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32), None if side is None else torch.as_tensor(side, dtype=torch.float32))
    target = logits.argmax(dim=1).numpy()
    base = torch.softmax(logits, dim=1).numpy()[np.arange(len(target)), target]
    drops = []
    for fraction in fractions:
        k = int(round(fraction * x.shape[1]))
        removed = np.stack([interpolate_out(x[i], orders[i, :k]) for i in range(len(x))]).astype(np.float32)
        drops.append(base - _probabilities(model, removed, side, target))
    return np.trapezoid(np.stack(drops, axis=1), x=np.asarray(fractions), axis=1)


def faithfulness(model, x: np.ndarray, side: np.ndarray | None, saliency: np.ndarray, fractions: list[float], n_random: int, seed: int) -> dict:
    """Deletion AUC by saliency, by random order, and their difference (the faithfulness gain), per beat."""
    by_saliency = deletion_auc(model, x, side, np.argsort(-np.abs(saliency), axis=1), fractions)
    rng = np.random.default_rng(seed)
    random = np.mean([deletion_auc(model, x, side, np.stack([rng.permutation(x.shape[1]) for _ in range(len(x))]), fractions)
                      for _ in range(n_random)], axis=0)  # fmt: skip
    return {"auc_saliency": by_saliency, "auc_random": random, "gain": by_saliency - random}


def stability(maps: list[np.ndarray]) -> float:
    """Mean Spearman correlation of the same beats' saliency across models (all pairs)."""
    values = []
    for a, b in combinations(maps, 2):
        for row_a, row_b in zip(np.abs(a), np.abs(b)):
            rho = spearmanr(row_a, row_b).statistic
            if np.isfinite(rho):
                values.append(rho)
    return float(np.mean(values)) if values else float("nan")


def segment_masses(saliency: np.ndarray, pre_samples: int, fs: float) -> dict[str, np.ndarray]:
    """Share of absolute saliency in each R-relative window, per beat (windows clipped to the beat)."""
    magnitude = np.abs(saliency)
    total = np.maximum(magnitude.sum(axis=1), 1e-12)
    ms = (np.arange(saliency.shape[1]) - pre_samples) / fs * 1000
    return {name: magnitude[:, (ms >= lo) & (ms < hi)].sum(axis=1) / total for name, (lo, hi) in SEGMENTS_MS.items()}
