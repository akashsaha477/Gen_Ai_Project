"""Classification metrics (Module 2).

All F1, precision and recall values come from argmax decisions; no threshold is tuned. Functions
that take a confusion matrix accept leading batch dimensions (``[..., C, C]``, rows = true class,
columns = predicted class) so the bootstrap can evaluate thousands of replicates at once.

Undefined values are NaN, never 0: a class with no true beats has no recall or F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecgaug.data.records import CLASS_INDEX, CLASSES

_N, _S, _V, _F = (CLASS_INDEX[c] for c in ("N", "S", "V", "F"))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Counts ``[n_classes, n_classes]`` with rows = true class and columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def confusion_by_record(
    y_true: np.ndarray, y_pred: np.ndarray, records: np.ndarray, n_classes: int = len(CLASSES)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-record confusion matrices: ``(unique_records, cms[n_records, C, C])``."""
    unique, inverse = np.unique(np.asarray(records), return_inverse=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    size = n_classes * n_classes
    flat = np.bincount(inverse * size + y_true * n_classes + y_pred, minlength=len(unique) * size)
    return unique, flat.reshape(len(unique), n_classes, n_classes)


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def recall_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class recall ``[..., C]``; NaN where a class has no true beats."""
    cm = np.asarray(cm, dtype=np.float64)
    return _divide(np.diagonal(cm, axis1=-2, axis2=-1), cm.sum(axis=-1))


def precision_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class precision ``[..., C]``; NaN where a class is never predicted."""
    cm = np.asarray(cm, dtype=np.float64)
    return _divide(np.diagonal(cm, axis1=-2, axis2=-1), cm.sum(axis=-2))


def f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 = 2TP / (2TP + FP + FN), ``[..., C]``; NaN where a class has no true beats."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    f1 = _divide(2 * tp, support + predicted)
    return np.where(support > 0, f1, np.nan)


def rare_macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """The primary metric: mean F1 of S and F. NaN if either class has no true beats."""
    f1 = f1_from_cm(cm)
    return (f1[..., _S] + f1[..., _F]) / 2


def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Mean F1 over N, S, V and F. NaN if any class has no true beats."""
    return f1_from_cm(cm).mean(axis=-1)


def aami_ppv_v_from_cm(cm: np.ndarray) -> np.ndarray:
    """AAMI-style positive predictivity for V: fusion beats called V are not counted as false positives.

    +P_V = TP_V / (TP_V + N->V + S->V). The convention follows the AAMI recommended practice as used
    by de Chazal et al. (2004); verify against the source before quoting it in the paper.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm[..., _V, _V]
    return _divide(tp, tp + cm[..., _N, _V] + cm[..., _S, _V])


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """All confusion-matrix metrics for a single ``[C, C]`` matrix, as plain floats."""
    cm = np.asarray(cm)
    recall, precision, f1 = recall_from_cm(cm), precision_from_cm(cm), f1_from_cm(cm)
    out: dict = {
        "n_beats": int(cm.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
        "rare_macro_f1": float(rare_macro_f1_from_cm(cm)),
        "macro_f1": float(macro_f1_from_cm(cm)),
        "n_recall": float(recall[_N]),
        "aami": {
            "se_S": float(recall[_S]),
            "ppv_S": float(precision[_S]),  # with Q excluded, the AAMI +P for S is the plain precision
            "se_V": float(recall[_V]),
            "ppv_V": float(aami_ppv_v_from_cm(cm)),
        },
        "per_class": {},
        "confusion_matrix": cm.astype(int).tolist(),
    }
    for i, name in enumerate(CLASSES):
        out["per_class"][name] = {
            "support": int(cm[i].sum()),
            "predicted": int(cm[:, i].sum()),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
    return out


def pr_auc_one_vs_rest(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """One-vs-rest average precision per class; NaN for a class with no true beats."""
    probs, y = np.asarray(probs), np.asarray(y)
    out = {}
    for i, name in enumerate(CLASSES):
        positives = y == i
        out[name] = float(average_precision_score(positives, probs[:, i])) if positives.any() else float("nan")
    return out


def evaluate(probs: np.ndarray, y: np.ndarray, records: np.ndarray | None = None) -> dict:
    """Evaluate class probabilities ``[n, C]`` against labels ``y``.

    Returns per-class precision, recall and F1, one-vs-rest PR-AUC, macro averages, AAMI-style Se and
    +P for S and V, and the confusion matrix. With ``records``, also the per-record confusion
    matrices that the cluster bootstrap resamples.
    """
    probs = np.asarray(probs)
    y = np.asarray(y)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES) or probs.shape[0] != y.shape[0]:
        raise ValueError(f"probs {probs.shape} and y {y.shape} do not match [n, {len(CLASSES)}] and [n]")
    y_pred = probs.argmax(axis=1)
    out = metrics_from_confusion(confusion_matrix(y, y_pred))
    pr_auc = pr_auc_one_vs_rest(probs, y)
    out["pr_auc"] = pr_auc
    out["macro_pr_auc"] = float(np.mean(list(pr_auc.values())))
    out["rare_macro_pr_auc"] = float((pr_auc["S"] + pr_auc["F"]) / 2)
    if records is not None:
        unique, cms = confusion_by_record(y, y_pred, records)
        out["per_record"] = {int(r): cm.astype(int).tolist() for r, cm in zip(unique, cms)}
    return out


def multilabel_auroc(probs: np.ndarray, y: np.ndarray, names: tuple[str, ...] | list[str]) -> dict:
    """Per-label AUROC for multi-label targets ``y`` [n, K] and scores ``probs`` [n, K].

    A label that is all 0 or all 1 has no AUROC (NaN). ``macro`` is the mean over the defined labels,
    NaN if none is defined.
    """
    probs, y = np.asarray(probs), np.asarray(y) > 0.5
    per_label = {}
    for i, name in enumerate(names):
        column = y[:, i]
        per_label[name] = float(roc_auc_score(column, probs[:, i])) if column.any() and not column.all() else float("nan")
    defined = [v for v in per_label.values() if not np.isnan(v)]
    return {"per_label": per_label, "macro": float(np.mean(defined)) if defined else float("nan"), "n_defined": len(defined)}
