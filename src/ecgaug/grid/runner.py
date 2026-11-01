"""Grid runner (PLAN Phase 3, item 5): conditions x strengths x architectures x seeds from one YAML file.

    python -m ecgaug.grid.runner --config configs/phase4_full_grid/full_grid.yaml --dry-run   # list the cells and their state
    python -m ecgaug.grid.runner --config configs/phase4_full_grid/full_grid.yaml [--preview] [--device cuda]

Each cell trains one classifier on the training beats plus what its condition adds (or with the
class-weighted loss); validation stays 100% real. Conditions:

- ``none``: real training beats only.
- ``class_weighted``: cross-entropy weights (n_max / n_c) ** gamma, one cell per gamma.
- ``random_oversampling``, ``smote`` and any condition with a ``bank`` (the generators): the policy sets
  how many beats each minority class gets at each strength (``ecgaug.augment.policy_targets``). A bank
  is named directly (``phase3_cvae-A``) or as frozen at the phase 3 gate (``phase3_gate/phase3_cvae``).
  Classifier seed s uses bank seed s (PLAN Phase 4). The quality screen of policy
  ``fraction_of_n_screened`` applies to generator banks only.

A cell is identified by a hash of everything that defines it: condition, strength, architecture,
seed, classifier settings, policy, and the hashes of the dataset and of the bank. A cell already done
is skipped, so an interrupted grid resumes where it stopped. Every cell is a run (``runs/`` and
``results/runs_index.csv``) and has its own small ledger file, ``results/<experiment>/<grid>/cells/<hash>.json``,
so grid parts run in different Kaggle sessions merge in git without conflicts.

Official grids need the phase 1 gate (frozen dataset), the phase 2 gate for ``cnn`` and ``rnn``, and the
phase 3 gate for banks, whose hashes are checked against the manifest. ``--preview`` relaxes that and
marks the runs provisional. The test set is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import augment, paths, runs, workflow
from ecgaug.data.records import CLASSES
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, mean_sd, split_part
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators.bank import BankError, concat_parts, load_bank
from ecgaug.generators.base import training_beats
from ecgaug.models import build_model
from ecgaug.names import PHASE2_GATE, PHASE3_GATE, phase_of
from ecgaug.train.loop import TrainConfig, fit_classifier, resolve_device, set_seed

GRID_KEYS = {"experiment", "name", "policy", "seeds", "architectures", "conditions"}
PLAIN_CONDITIONS = {"none", "class_weighted", "random_oversampling", "smote"}
QUALITY_SETTINGS = Path("configs") / "phase2_quality_metrics" / "metric_settings.yaml"


@dataclass(frozen=True)
class Cell:
    condition: str
    strength: float | None
    architecture: str
    seed: int

    @property
    def tag(self) -> str:
        """Run-ID tag, e.g. ``smote_0.25``, ``class_weighted_0.5`` or ``none``."""
        return self.condition if self.strength is None else f"{self.condition}_{self.strength:g}"

    @property
    def label(self) -> str:
        """Run-ID variant: the architecture name with only letters and digits."""
        return re.sub(r"[^A-Za-z0-9]", "", self.architecture)


def policy_of(grid: dict) -> dict:
    """The grid's policy as a dict (a file may give just the kind)."""
    return grid["policy"] if isinstance(grid["policy"], dict) else {"kind": grid["policy"]}


def load_grid(path: Path) -> dict:
    grid = yaml.safe_load(Path(path).read_text())
    if set(grid) != GRID_KEYS:
        raise ValueError(f"{path}: a grid file has exactly the keys {sorted(GRID_KEYS)}")
    phase_of(grid["experiment"])
    policy = policy_of(grid)
    if policy["kind"] not in augment.POLICIES:
        raise ValueError(f"{path}: unknown policy {policy['kind']!r}; choose from {augment.POLICIES}")
    for name, spec in grid["conditions"].items():
        spec = spec or {}
        if name not in PLAIN_CONDITIONS and "bank" not in spec:
            raise ValueError(f"{path}: condition {name!r} is neither {sorted(PLAIN_CONDITIONS)} nor a generator bank")
    return {**grid, "policy": policy, "file": str(path)}


def cells(grid: dict) -> list[Cell]:
    """Every cell of the grid, in a fixed order: condition, strength, architecture, seed."""
    out = []
    for name, spec in grid["conditions"].items():
        spec = spec or {}
        if name == "none":
            strengths = [None]
        elif name == "class_weighted":
            strengths = spec.get("gammas")
        elif policy_of(grid)["kind"] == "balance_to_n":
            strengths = [None]
        else:
            strengths = spec.get("strengths")
        if not strengths:
            raise ValueError(f"Condition {name!r} needs {'gammas' if name == 'class_weighted' else 'strengths'}")
        for strength in strengths:
            for architecture in grid["architectures"]:
                for seed in grid["seeds"]:
                    out.append(Cell(name, None if strength is None else float(strength), architecture, int(seed)))
    return out


def classifier_settings(grid: dict, architecture: str, preview: bool) -> dict:
    """Model and training settings of an architecture: given in the grid file, or the phase 2 gate's choice."""
    spec = grid["architectures"][architecture]
    if spec == "phase2_gate":
        chosen = (workflow.manifest().get(PHASE2_GATE) or {}).get(architecture)
        if not chosen:
            raise PreconditionError(f"The phase 2 gate has not chosen the {architecture} yet")
        return chosen
    if not preview:
        raise PreconditionError(f"Architecture {architecture} is set in the grid file; official grids use the phase 2 gate's classifiers")
    return spec


def bank_for(grid: dict, condition: str, seed: int, preview: bool) -> tuple[dict, str, str]:
    """A condition's bank for one seed: (arrays, SHA-256, generator id), checked against the frozen hashes."""
    name = grid["conditions"][condition]["bank"]
    frozen = workflow.manifest().get(PHASE3_GATE) or {}
    if name.startswith("phase3_gate/"):
        experiment = name.split("/", 1)[1]
        if experiment not in frozen:
            raise PreconditionError(f"The phase 3 gate has not frozen a bank for {experiment}")
        generator_id = frozen[experiment]["generator_id"]
    else:
        generator_id = name
    expected = next((e["banks"][str(seed)]["sha256"] for e in frozen.values() if isinstance(e, dict) and e.get("generator_id") == generator_id), None)
    if expected is None and not preview:
        raise PreconditionError(f"Bank {generator_id} is not frozen at the phase 3 gate")
    arrays = load_bank(generator_id, seed, expected_sha256=expected)
    meta = json.loads((paths.repo_root() / "data" / "synthetic" / generator_id / "bank.json").read_text())
    return arrays, meta["seeds"][str(seed)]["sha256"], generator_id


def cell_sha(cell: Cell, grid: dict, classifier: dict, data: DataSource, bank_sha: str | None) -> str:
    key = {"cell": asdict(cell), "policy": policy_of(grid), "classifier": classifier, "dataset": data.info.get("content_sha256"),
           "bank": bank_sha, "condition_settings": grid["conditions"][cell.condition] or {}}  # fmt: skip
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def cells_dir(grid: dict) -> Path:
    return paths.results_for(grid["experiment"]) / grid["name"] / "cells"


def read_ledger(grid: dict) -> dict[str, dict]:
    folder = cells_dir(grid)
    if not folder.is_dir():
        return {}
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return {row["cell_sha"]: row for row in rows}


def write_cell(grid: dict, row: dict) -> Path:
    folder = cells_dir(grid)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{row['cell_sha'][:16]}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    return path


def is_done(row: dict | None) -> bool:
    return bool(row) and row["status"] == "done" and runs.find_run_dir(row["run_id"]) is not None


def quality_settings() -> dict:
    """DTW settings for the quality screen: the ones fixed at the phase 2 gate, else the config's."""
    fixed = (workflow.manifest().get(PHASE2_GATE) or {}).get("quality_settings")
    path = paths.repo_root() / QUALITY_SETTINGS
    return fixed or yaml.safe_load((path if path.is_file() else paths.source_root() / QUALITY_SETTINGS).read_text())["metrics"]


def added_beats(cell: Cell, grid: dict, train: dict, bank: dict | None) -> tuple[dict, np.ndarray | None]:
    """What a cell adds to the training beats, and its class weights (or None)."""
    counts = augment.class_counts(train["y"])
    spec = grid["conditions"][cell.condition] or {}
    policy = policy_of(grid)
    if cell.condition == "none":
        return concat_parts([]), None
    if cell.condition == "class_weighted":
        return concat_parts([]), augment.class_weights(counts, cell.strength)
    extra = augment.policy_targets(policy["kind"], cell.strength, counts)
    if cell.condition == "random_oversampling":
        return augment.random_oversample(train, extra, cell.seed), None
    if cell.condition == "smote":
        return augment.smote_beats(train, extra, cell.seed, int(spec.get("k_neighbors", 5))), None
    if policy["kind"] == "fraction_of_n_screened":
        return augment.screened_from_bank(bank, extra, train, float(policy.get("pool_factor", 2.0)), quality_settings()), None
    return augment.take_from_bank(bank, extra), None


def run_cell(cell: Cell, grid: dict, split, data: DataSource, classifier: dict, device: torch.device, sha: str,
             bank: dict | None, bank_id: str | None, provisional: list[str], run_root: tuple[str, ...] = ()) -> dict:  # fmt: skip
    """Train and log one cell under ``runs/<run_root...>/``; return its ledger row."""
    train = training_beats(data.arrays)
    val = split_part(data.arrays, "val", data.spec.rr_features)
    added, weights = added_beats(cell, grid, train, bank)
    part = split_part(augment.with_added(train, added), "train", data.spec.rr_features)
    set_seed(cell.seed)
    model = build_model(classifier["model"], len(CLASSES), len(SIDE_FIELDS) if data.spec.rr_features else 0)
    train_cfg = TrainConfig(lr=float(classifier["lr"]), max_epochs=int(classifier["max_epochs"]), patience=int(classifier["patience"]),
                            batch_size=int(classifier["batch_size"]), weight_decay=float(classifier["weight_decay"]),
                            min_epochs=int(classifier.get("min_epochs", 0)))  # fmt: skip
    fit = fit_classifier(model, part, val, train_cfg, split.selection_metric, seed=cell.seed, device=device, class_weights=weights)
    variant = grid.get("variant")  # an experiment's variant letter; the full grid has none and names runs by architecture
    run_id = runs.make_run_id(grid["experiment"], variant or cell.label, cell.seed, tag=cell.tag if variant is None else f"{cell.label}.{cell.tag}")
    group = (*run_root, grid["experiment"], grid["name"], cell.label)
    metrics = {**fit.best_metrics, "selection_metric": split.selection_metric, "selection_value": fit.best_value,
               "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run, "stopped_early": fit.stopped_early}  # fmt: skip
    n_added = {c: int((np.asarray(added["y"]) == i).sum()) for i, c in enumerate(CLASSES)}
    config = {
        "grid": grid["file"], "grid_name": grid["name"], "cell": asdict(cell), "cell_sha": sha, "policy": policy_of(grid),
        "condition_settings": grid["conditions"][cell.condition] or {}, "bank": bank_id, "classifier": classifier,
        "class_weights": None if weights is None else weights.tolist(), "n_added": n_added, "run_id": run_id,
        "split": {"name": split.name, "val": list(split.val), "selection_metric": split.selection_metric},
        "dataset": data.info, "provisional": bool(provisional), "provisional_reasons": provisional,
    }  # fmt: skip
    env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
    preds = {"probs": fit.val_probs, "y": val["y"], "record": val["record"], "r_idx": val["r_idx"]}
    runs.write_run(run_id, config, env, fit.history, metrics, preds, fit.state_dict, group=group)
    runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, group=group))
    return {
        "cell_sha": sha, "cell": asdict(cell), "condition": cell.condition, "strength": cell.strength,
        "architecture": cell.architecture, "seed": cell.seed, "run_id": run_id, "status": "done", "bank": bank_id,
        "selection_metric": split.selection_metric, "selection_value": fit.best_value,
        "tie_break_value": fit.best_metrics["per_class"]["V"]["f1"], "rare_macro_f1": fit.best_metrics["rare_macro_f1"],
        "n_recall": fit.best_metrics["n_recall"], "n_added": n_added, "wall_clock_s": round(fit.wall_clock_s, 1),
        "provisional": bool(provisional),
    }  # fmt: skip


def run_grid(grid: dict, split, data: DataSource, device: torch.device, preview: bool, dry_run: bool = False, provisional: list[str] | None = None) -> dict:
    """Run every cell that is not done yet (or, with ``dry_run``, only report them)."""
    provisional = list(provisional or [])
    ledger = read_ledger(grid)
    report = {"done_before": [], "ran": [], "planned": []}
    banks: dict[tuple[str, int], tuple[dict, str, str]] = {}
    for cell in cells(grid):
        classifier = classifier_settings(grid, cell.architecture, preview)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition not in PLAIN_CONDITIONS:
            key = (cell.condition, cell.seed)
            if key not in banks:
                banks[key] = bank_for(grid, cell.condition, cell.seed, preview)
            bank, bank_sha, bank_id = banks[key]
        sha = cell_sha(cell, grid, classifier, data, bank_sha)
        if is_done(ledger.get(sha)):
            report["done_before"].append(ledger[sha]["run_id"])
            continue
        if dry_run:
            report["planned"].append(asdict(cell))
            continue
        ledger[sha] = run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, provisional)
        write_cell(grid, ledger[sha])  # after every cell, so an interruption loses at most one cell
        report["ran"].append(ledger[sha]["run_id"])
    return report


def summarize(grid: dict) -> dict[tuple, dict]:
    """Seed means per (condition, strength, architecture) from the ledger: what the phase 4 gate reads."""
    groups: dict[tuple, list[dict]] = {}
    for row in read_ledger(grid).values():
        if row["status"] == "done":
            groups.setdefault((row["condition"], row["strength"], row["architecture"]), []).append(row)
    out = {}
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["seed"])
        mean, sd = mean_sd([r["selection_value"] for r in rows])
        tie_mean, tie_sd = mean_sd([r["tie_break_value"] for r in rows])
        out[key] = {"seeds": [r["seed"] for r in rows], "run_ids": [r["run_id"] for r in rows], "mean": mean, "sd": sd,
                    "tie_break_mean": tie_mean, "tie_break_sd": tie_sd, "n_recall": float(np.mean([r["n_recall"] for r in rows])),
                    "provisional": any(r.get("provisional") for r in rows), "bank": rows[0].get("bank")}  # fmt: skip
    return out


def smoke(grid: dict) -> dict:
    """The grid's first two plain conditions on fake records, one seed, one epoch, in a temporary root."""
    from ecgaug.experiments.data_source import smoke_data

    small = {"model": {"name": "probe_cnn", "channels": [8, 16, 16], "kernel": 5, "side_hidden": 8}, "lr": 1e-3, "max_epochs": 1,
             "patience": 5, "batch_size": 256, "weight_decay": 0.01}  # fmt: skip
    conditions = {k: v for k, v in grid["conditions"].items() if k in PLAIN_CONDITIONS}
    test_grid = {**grid, "name": "smoke", "seeds": [grid["seeds"][0]], "architectures": {"probe_cnn": small}, "conditions": conditions}
    with smoke_data() as (split, data):
        report = run_grid(test_grid, split, data, torch.device("cpu"), preview=True, provisional=["smoke run"])
        again = run_grid(test_grid, split, data, torch.device("cpu"), preview=True)
        return {"cells": len(report["ran"]), "skipped_on_rerun": len(again["done_before"]), "note": "fake data; the numbers mean nothing"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run (or list) the cells of a grid of conditions, strengths, architectures and seeds.")
    parser.add_argument("--config", required=True, help="a grid file, e.g. configs/phase4_full_grid/full_grid.yaml")
    parser.add_argument("--dry-run", action="store_true", help="list the cells and whether each is done; train nothing")
    parser.add_argument("--preview", action="store_true", help="run before the gates the grid needs; runs are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="the grid's plain conditions on fake data, then a rerun that skips them")
    args = parser.parse_args(argv)
    grid = load_grid(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(grid), indent=2))
        return 0
    try:
        if not args.preview and workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first, or add --preview.")
        split = check_preconditions(allow_dirty=args.dry_run)
        data = load_data(split, args.preview, args.representation)
        report = run_grid(grid, split, data, resolve_device(args.device or "auto"), args.preview, args.dry_run,
                          [data.provisional] if data.provisional else [])  # fmt: skip
    except (PreconditionError, BankError) as err:
        print(f"Refusing to run grid {grid['name']}: {err}")
        return 2
    total = len(cells(grid))
    print(f"grid {grid['name']}: {total} cells; {len(report['done_before'])} already done, {len(report['ran'])} run now"
          + (f", {len(report['planned'])} to run" if args.dry_run else ""))  # fmt: skip
    for cell in report["planned"]:
        print("  to run:", cell)
    print(f"ledger: {cells_dir(grid).relative_to(paths.repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
