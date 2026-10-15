"""Pieces shared by the experiment runners: preconditions, data parts, and the fine-tuning grid."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ecgaug import runs
from ecgaug.data.counts import count_record
from ecgaug.data.records import CLASSES, DS1
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, load_approved_split
from ecgaug.eval.selection import tie_break_value
from ecgaug.models import build_model, count_parameters
from ecgaug.train.loop import TrainConfig, fit_classifier, set_seed

SIDE_FIELDS = ("rr_prev", "rr_next", "rr_ratio")
BUDGET_KEYS = ("seeds", "lr_grid", "max_epochs", "patience", "batch_size", "weight_decay")

# Called after a model is built for a seed. Loads weights into it and returns what the run's
# config should record about where they came from.
InitFn = Callable[[nn.Module, int], dict]


class PreconditionError(RuntimeError):
    """A real run was requested before its preconditions were met."""


def check_preconditions(allow_dirty: bool = False) -> Split:
    """Return the approved split, or raise :class:`PreconditionError` explaining what is missing."""
    counts = pd.DataFrame([count_record(r) for r in DS1])
    try:
        split = load_approved_split(counts)
    except (FileNotFoundError, ValueError) as err:
        raise PreconditionError(str(err)) from err
    if not allow_dirty and runs.git_dirty():
        raise PreconditionError("The git tree has uncommitted changes; a run must map to one commit.")
    return split


def split_part(arrays: dict[str, np.ndarray], name: str, with_side: bool) -> dict[str, np.ndarray]:
    """The beats of one split of a contract-format dataset, as the training loop expects them."""
    mask = arrays["split"] == name
    part = {"X": arrays["X"][mask], "y": arrays["y"][mask], "record": arrays["record"][mask]}
    part["r_idx"] = arrays["r_idx"][mask]
    part["side"] = np.stack([arrays[f][mask] for f in SIDE_FIELDS], axis=1) if with_side else None
    return part


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


@contextmanager
def temporary_root():
    """Point ECGAUG_ROOT at a throwaway directory for the duration of a smoke run."""
    previous = os.environ.get("ECGAUG_ROOT")
    with tempfile.TemporaryDirectory(prefix="ecgaug-smoke-") as tmp:
        os.environ["ECGAUG_ROOT"] = tmp
        try:
            yield Path(tmp)
        finally:
            if previous is None:
                os.environ.pop("ECGAUG_ROOT", None)
            else:
                os.environ["ECGAUG_ROOT"] = previous


def train_grid(
    cfg: dict,
    split: Split,
    spec: RepresentationSpec,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    phase: int,
    module: int,
    label: str,
    device: torch.device,
    dataset: dict,
    group: tuple[str, ...],
    init: InitFn | None = None,
) -> tuple[dict[str, dict], int]:
    """Train the classifier for every learning rate and seed of the budget and log every run.

    Returns the grid ``{tag: {lr, per_seed, mean, sd, ...}}`` and the model's parameter count.
    With ``init``, each freshly built model gets weights loaded before training (fine-tuning).
    Runs are saved under ``runs/<group...>/``.
    """
    selection = split.selection_metric
    grid: dict[str, dict] = {}
    n_params = 0
    for lr in cfg["lr_grid"]:
        tag = f"lr{lr:.0e}"
        per_seed = []
        for seed in cfg["seeds"]:
            run_id = runs.make_run_id(phase, module, label, seed, tag=tag)
            set_seed(seed)
            model = build_model(cfg["model"], len(CLASSES), len(SIDE_FIELDS) if spec.rr_features else 0)
            n_params = count_parameters(model)
            origin = init(model, seed) if init else None
            train_cfg = TrainConfig(
                lr=float(lr),
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                batch_size=cfg["batch_size"],
                weight_decay=cfg["weight_decay"],
            )
            fit = fit_classifier(model, train, val, train_cfg, selection, seed=seed, device=device)
            metrics = {
                **fit.best_metrics,
                "selection_metric": selection,
                "selection_value": fit.best_value,
                "best_epoch": fit.best_epoch,
                "epochs_run": fit.epochs_run,
                "stopped_early": fit.stopped_early,
            }
            env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
            run_config = {
                **cfg,
                "lr": float(lr),
                "seed": seed,
                "run_id": run_id,
                "split": {"name": split.name, "val": list(split.val), "selection_metric": selection},
                "dataset": dataset,
            }
            if origin is not None:
                run_config["init"] = origin
            preds = {"probs": fit.val_probs, "y": val["y"], "record": val["record"], "r_idx": val["r_idx"]}
            runs.write_run(run_id, run_config, env, fit.history, metrics, preds, fit.state_dict, group=group)
            runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, group=group))
            per_seed.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "selection_value": fit.best_value,
                    "tie_break_value": tie_break_value(fit.best_metrics, selection),
                    "rare_macro_f1": fit.best_metrics["rare_macro_f1"],
                    "f1": {c: fit.best_metrics["per_class"][c]["f1"] for c in CLASSES},
                    "pr_auc": fit.best_metrics["pr_auc"],
                    "n_recall": fit.best_metrics["n_recall"],
                    "best_epoch": fit.best_epoch,
                    "epochs_run": fit.epochs_run,
                    "wall_clock_s": fit.wall_clock_s,
                }
            )
        mean, sd = mean_sd([r["selection_value"] for r in per_seed])
        entry = {"lr": float(lr), "per_seed": per_seed, "mean": mean, "sd": sd}
        if per_seed[0]["tie_break_value"] is not None:
            entry["tie_break_mean"], entry["tie_break_sd"] = mean_sd([r["tie_break_value"] for r in per_seed])
        entry["wall_clock_s"] = float(sum(r["wall_clock_s"] for r in per_seed))
        grid[tag] = entry
    return grid, n_params


def best_grid_point(grid: dict[str, dict]) -> dict:
    """The learning rate with the higher seed-mean selection metric; the first one wins exact ties."""
    return grid[max(grid, key=lambda tag: grid[tag]["mean"])]
