"""Evaluation utilities: metrics, record-level bootstrap, selection rules, plotting helpers."""

from ecgaug.eval.bootstrap import (
    cluster_bootstrap,
    cluster_bootstrap_confusion,
    holm,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_confusion,
)
from ecgaug.eval.metrics import evaluate

__all__ = [
    "cluster_bootstrap",
    "cluster_bootstrap_confusion",
    "evaluate",
    "holm",
    "paired_cluster_bootstrap_diff",
    "paired_cluster_bootstrap_diff_confusion",
]
