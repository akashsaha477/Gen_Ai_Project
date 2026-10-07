"""Module 1 runner: compare input representations with the probe CNN.

Usage:
    python -m ecgaug.experiments.m1 --config configs/m1/A.yaml            # the variant experiment
    python -m ecgaug.experiments.m1 --config configs/m1/A.yaml --check    # build DS1 inputs, no training
    python -m ecgaug.experiments.m1 --config configs/m1/A.yaml --smoke    # fake data, end to end

A real run needs three things first: PREREGISTRATION.md in force, an approved validation split and
a clean git tree. ``--check`` reads DS1 only and trains nothing. ``--smoke`` never touches real data
or the repository's results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ecgaug import paths, runs
from ecgaug.data.contract import build_arrays, build_candidate, content_sha256
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb
from ecgaug.data.loaders import load_candidate
from ecgaug.data.records import CLASSES, DS1
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, load_approved_split, selection_metric_for
from ecgaug.eval.selection import selection_value, tie_break_value
from ecgaug.models.probe import ProbeCNN, count_parameters
from ecgaug.train.loop import TrainConfig, fit_classifier, resolve_device, set_seed

MODULE = 1
PHASE = 1
VARIANT_KEYS = {"base", "variant", "description", "representation"}
SIDE_FIELDS = ("rr_prev", "rr_next", "rr_ratio")


class PreconditionError(RuntimeError):
    """A real variant run was requested before its preconditions were met."""


def load_config(path: Path) -> dict:
    """Merge a variant config with the shared budget it points to.

    A variant file may define only its letter, description and representation. Everything else
    (seeds, learning-rate grid, epochs, patience, model) comes from the shared file, so budgets
    cannot drift between variants (CLAUDE.md rule 7).
    """
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared budget file, not in a variant")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    return {
        **shared,
        "variant": str(variant["variant"]),
        "description": variant.get("description", ""),
        "representation": variant["representation"],
    }


def prereg_status() -> str:
    """The ``Status:`` line of docs/PREREGISTRATION.md ('DRAFT', 'IN FORCE', or 'MISSING')."""
    path = paths.docs_dir() / "PREREGISTRATION.md"
    if not path.is_file():
        return "MISSING"
    match = re.search(r"^\*{0,2}Status:\*{0,2}\s*(.+?)\s*$", path.read_text(), flags=re.MULTILINE)
    return match.group(1).strip() if match else "MISSING"


def check_preconditions(allow_dirty: bool = False) -> Split:
    """Return the approved split, or raise :class:`PreconditionError` explaining what is missing."""
    status = prereg_status()
    if not status.upper().startswith("IN FORCE"):
        raise PreconditionError(
            f"docs/PREREGISTRATION.md has status '{status}'. Module 1 variants may run only after "
            "v1 is approved, committed and marked 'IN FORCE'."
        )
    counts = pd.DataFrame([count_record(r) for r in DS1])
    try:
        split = load_approved_split(counts)
    except (FileNotFoundError, ValueError) as err:
        raise PreconditionError(str(err)) from err
    if not allow_dirty and runs.git_dirty():
        raise PreconditionError("The git tree has uncommitted changes; a run must map to one commit.")
    return split


def _part(arrays: dict[str, np.ndarray], name: str, with_side: bool) -> dict[str, np.ndarray]:
    mask = arrays["split"] == name
    part = {"X": arrays["X"][mask], "y": arrays["y"][mask], "record": arrays["record"][mask]}
    part["r_idx"] = arrays["r_idx"][mask]
    part["side"] = np.stack([arrays[f][mask] for f in SIDE_FIELDS], axis=1) if with_side else None
    return part


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def run_variant(cfg: dict, split: Split, device_name: str | None = None, require_all_ds1: bool = True) -> dict:
    """Train the probe for every learning rate and seed, log every run, and write the result file."""
    spec = RepresentationSpec(**cfg["representation"])
    variant = cfg["variant"]
    selection = split.selection_metric
    device = resolve_device(device_name or cfg.get("device", "cpu"))

    dataset_dir = build_candidate(spec, split, require_all_ds1=require_all_ds1)
    arrays = load_candidate(spec.name)
    dataset_hash = content_sha256(arrays)
    train, val = _part(arrays, "train", spec.rr_features), _part(arrays, "val", spec.rr_features)

    grid: dict[str, dict] = {}
    n_params = 0
    for lr in cfg["lr_grid"]:
        tag = f"lr{lr:.0e}"
        per_seed = []
        for seed in cfg["seeds"]:
            run_id = runs.make_run_id(PHASE, MODULE, variant, seed, tag=tag)
            set_seed(seed)
            model = ProbeCNN(
                n_classes=len(CLASSES),
                n_side=len(SIDE_FIELDS) if spec.rr_features else 0,
                channels=tuple(cfg["model"]["channels"]),
                kernel=cfg["model"]["kernel"],
                side_hidden=cfg["model"]["side_hidden"],
            )
            n_params = count_parameters(model)
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
                "dataset": {"path": str(dataset_dir), "content_sha256": dataset_hash},
            }
            preds = {"probs": fit.val_probs, "y": val["y"], "record": val["record"], "r_idx": val["r_idx"]}
            runs.write_run(run_id, run_config, env, fit.history, metrics, preds, fit.state_dict)
            runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s))
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
        mean, sd = _mean_sd([r["selection_value"] for r in per_seed])
        entry = {"lr": float(lr), "per_seed": per_seed, "mean": mean, "sd": sd}
        if per_seed[0]["tie_break_value"] is not None:
            entry["tie_break_mean"], entry["tie_break_sd"] = _mean_sd([r["tie_break_value"] for r in per_seed])
        entry["wall_clock_s"] = float(sum(r["wall_clock_s"] for r in per_seed))
        grid[tag] = entry

    # The learning rate with the higher seed-mean selection metric is used; the first one wins exact ties.
    chosen_tag = max(grid, key=lambda t: grid[t]["mean"])
    chosen = grid[chosen_tag]
    result = {
        "module": MODULE,
        "variant": variant,
        "description": cfg.get("description", ""),
        "representation": spec.to_dict(),
        "complexity_inputs_per_beat": spec.n_inputs,
        "n_parameters": n_params,
        "split": {"name": split.name, "val": list(split.val)},
        "selection_metric": selection,
        "seeds": list(cfg["seeds"]),
        "chosen_lr": chosen["lr"],
        "mean": chosen["mean"],
        "sd": chosen["sd"],
        "tie_break_mean": chosen.get("tie_break_mean"),
        "tie_break_sd": chosen.get("tie_break_sd"),
        "per_seed": chosen["per_seed"],
        "run_ids": [r["run_id"] for r in chosen["per_seed"]],
        "lr_grid": grid,
        "wall_clock_s": float(sum(e["wall_clock_s"] for e in grid.values())),
        "budget": {k: cfg[k] for k in ("seeds", "lr_grid", "max_epochs", "patience", "batch_size", "weight_decay")},
        "budget_deviations": [],
        "device": str(device),
        "git_sha": runs.git_sha(),
        "dataset_content_sha256": dataset_hash,
    }
    out_dir = paths.results_dir() / "validation" / f"m{MODULE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    notes = out_dir / f"{variant}_notes.md"
    if not notes.exists():
        notes.write_text(
            f"# Module {MODULE} variant {variant}: notes\n\n"
            "What worked:\n\nWhat failed:\n\nSurprises:\n\nSanity checks (shapes, NaNs, class coverage, "
            "learning curves, early-stop epochs):\n\nFigure paths:\n"
        )
    return result


def representation_check(cfg: dict) -> dict:
    """Build the variant's inputs for all DS1 records without a split and without training."""
    spec = RepresentationSpec(**cfg["representation"])
    arrays, report = build_arrays(list(DS1), spec, {r: "none" for r in DS1})
    x = arrays["X"]
    side = np.stack([arrays[f] for f in SIDE_FIELDS], axis=1)
    return {
        "variant": cfg["variant"],
        "representation": spec.to_dict(),
        "inputs_per_beat": spec.n_inputs,
        "X_shape": list(x.shape),
        "X_dtype": str(x.dtype),
        "non_finite_X": int((~np.isfinite(x)).sum()),
        "non_finite_rr": int((~np.isfinite(side)).sum()),
        "per_beat_mean_abs_max": float(np.abs(x.mean(axis=1)).max()),
        "per_beat_std_range": [float(x.std(axis=1).min()), float(x.std(axis=1).max())],
        "kept": {c: int(sum(r["kept"][c] for r in report)) for c in CLASSES},
        "annotated": {c: int(sum(r["annotated"][c] for r in report)) for c in CLASSES},
        "dropped_unusable": {c: int(sum(r["dropped_unusable"][c] for r in report)) for c in CLASSES},
        "q_beats_excluded": int(sum(r["q_beats_excluded"] for r in report)),
        "unmapped_qrs_excluded": int(sum(r["unmapped_qrs_excluded"] for r in report)),
        "beat_set_sha256": content_sha256({"record": arrays["record"], "r_idx": arrays["r_idx"], "y": arrays["y"]}),
        "content_sha256": content_sha256(arrays),
    }


@contextmanager
def _temporary_root():
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


def smoke(cfg: dict) -> dict:
    """Run the whole variant pipeline on fake records in a temporary root (one seed, two epochs)."""
    train_records, val_records = (101, 106, 108, 114), (118, 201)
    with _temporary_root() as root:
        write_fake_mitdb(paths.raw_dir(), train_records + val_records, seconds=120.0)
        counts = pd.DataFrame([count_record(r) for r in train_records + val_records])
        split = Split(name="smoke", train=train_records, val=val_records)
        check_split(split, require_all_ds1=False)
        split = Split(
            name="smoke", train=train_records, val=val_records, selection_metric=selection_metric_for(counts, split)
        )
        smoke_cfg = {**cfg, "seeds": [0], "lr_grid": [1e-3], "max_epochs": 2}
        result = run_variant(smoke_cfg, split, device_name="cpu", require_all_ds1=False)
        run_dir = root / "runs" / result["run_ids"][0]
        produced = sorted(p.name for p in run_dir.iterdir())
        expected = {"config.yaml", "env.json", "train_log.csv", "metrics_val.json", "preds_val.npz", "checkpoints"}
        missing = sorted(expected - set(produced))
        if missing:
            raise RuntimeError(f"Smoke run did not produce {missing}")
        return {
            "variant": result["variant"],
            "inputs_per_beat": result["complexity_inputs_per_beat"],
            "n_parameters": result["n_parameters"],
            "selection_metric": result["selection_metric"],
            "run_files": produced,
            "result_file": f"results/validation/m{MODULE}/{result['variant']}.json",
            "note": "fake data; the numbers of a smoke run mean nothing and are discarded",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Module 1: input-representation variants with the probe CNN.")
    parser.add_argument("--config", required=True, help="configs/m1/<A-D>.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: from the shared config)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    mode.add_argument("--check", action="store_true", help="build DS1 inputs and report shapes; no training")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    if args.check:
        report = representation_check(cfg)
        out = paths.results_dir() / "data" / f"m1_representation_check_{cfg['variant']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["non_finite_X"] == 0 and report["non_finite_rr"] == 0 else 1

    try:
        split = check_preconditions()
    except PreconditionError as err:
        print(f"Refusing to run variant {cfg['variant']}: {err}")
        return 2
    result = run_variant(cfg, split, device_name=args.device)
    print(
        f"variant {result['variant']}: {result['selection_metric']} = {result['mean']:.4f} "
        f"± {result['sd']:.4f} over seeds {result['seeds']} (lr {result['chosen_lr']:g}); "
        f"runs: {', '.join(result['run_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
