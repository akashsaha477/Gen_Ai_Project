"""Grid runner (PLAN Phase 3, item 5): conditions x strengths x architectures x seeds from one YAML file.

    python -m ecgaug.grid.runner --config configs/phase4_full_grid/dry_run_baselines.yaml --dry-run   # list the cells and their state
    python -m ecgaug.grid.runner --config configs/phase4_full_grid/dry_run_baselines.yaml [--preview] [--device cuda]

Each cell trains one classifier on the training beats plus what its condition adds (or with the
class-weighted loss); validation stays 100% real. Conditions:

- ``none``: real training beats only.
- ``class_weighted``: cross-entropy weights (n_max / n_c) ** gamma, one cell per gamma.
- ``random_oversampling``, ``smote`` and any condition with a ``bank`` (the generators): each minority
  class is brought up to ``strength`` x the training N count (policy ``fraction_of_n``), one cell per
  strength. Classifier seed s uses bank seed s (PLAN Phase 4).

A cell is identified by a hash of everything that defines it: condition, strength, architecture,
seed, classifier settings, policy, and the hashes of the dataset and of the bank. A cell already done
is skipped, so an interrupted grid resumes where it stopped. Every cell is a run (``runs/`` and
``results/runs_index.csv``), and ``results/<experiment>/<grid name>.csv`` lists every cell with its run.

Official grids need the phase 1 gate (frozen dataset), the phase 2 gate for ``cnn`` and ``rnn``, and the
phase 3 gate for banks, whose hashes are checked against the manifest. ``--preview`` relaxes that and
marks the runs provisional. The test set is never read.
"""

from __future__ import annotations

import argparse
import csv
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
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, split_part
from ecgaug.experiments.generator_runs import DataSource, load_data
from ecgaug.generators.bank import BankError, concat_parts, load_bank
from ecgaug.generators.base import training_beats
from ecgaug.models import build_model
from ecgaug.names import PHASE2_GATE, PHASE3_GATE, phase_of
from ecgaug.train.loop import TrainConfig, fit_classifier, resolve_device, set_seed

GRID_KEYS = {"experiment", "name", "policy", "seeds", "architectures", "conditions"}
PLAIN_CONDITIONS = {"none", "class_weighted", "random_oversampling", "smote"}
LEDGER_COLUMNS = [
    "cell_sha", "condition", "strength", "architecture", "seed", "run_id", "status", "selection_metric",
    "selection_value", "rare_macro_f1", "n_added", "wall_clock_s",
]  # fmt: skip


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


def load_grid(path: Path) -> dict:
    grid = yaml.safe_load(Path(path).read_text())
    if set(grid) != GRID_KEYS:
        raise ValueError(f"{path}: a grid file has exactly the keys {sorted(GRID_KEYS)}")
    phase_of(grid["experiment"])
    if grid["policy"] != "fraction_of_n":
        raise ValueError(f"{path}: unknown policy {grid['policy']!r} (the policy is fixed in PREREGISTRATION v3)")
    for name, spec in grid["conditions"].items():
        spec = spec or {}
        if name not in PLAIN_CONDITIONS and "bank" not in spec:
            raise ValueError(f"{path}: condition {name!r} is neither {sorted(PLAIN_CONDITIONS)} nor a generator bank")
    return {**grid, "file": str(path)}


def cells(grid: dict) -> list[Cell]:
    """Every cell of the grid, in a fixed order: condition, strength, architecture, seed."""
    out = []
    for name, spec in grid["conditions"].items():
        spec = spec or {}
        strengths = [None] if name == "none" else spec.get("gammas") if name == "class_weighted" else spec.get("strengths")
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


def bank_for(grid: dict, condition: str, seed: int, preview: bool) -> tuple[dict, str]:
    """A condition's bank for one seed, checked against the frozen hashes (or bank.json in a preview)."""
    generator_id = grid["conditions"][condition]["bank"]
    frozen = workflow.manifest().get(PHASE3_GATE) or {}
    expected = None
    for entry in frozen.values():
        if isinstance(entry, dict) and entry.get("generator_id") == generator_id:
            expected = entry["banks"][str(seed)]["sha256"]
    if expected is None and not preview:
        raise PreconditionError(f"Bank {generator_id} is not frozen at the phase 3 gate")
    arrays = load_bank(generator_id, seed, expected_sha256=expected)
    return arrays, expected or json.loads((paths.repo_root() / "data" / "synthetic" / generator_id / "bank.json").read_text())["seeds"][str(seed)]["sha256"]


def cell_sha(cell: Cell, grid: dict, classifier: dict, data: DataSource, bank_sha: str | None) -> str:
    key = {"cell": asdict(cell), "policy": grid["policy"], "classifier": classifier, "dataset": data.info.get("content_sha256"),
           "bank": bank_sha, "condition_settings": grid["conditions"][cell.condition] or {}}  # fmt: skip
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def ledger_path(grid: dict) -> Path:
    return paths.results_for(grid["experiment"]) / f"{grid['name']}.csv"


def read_ledger(grid: dict) -> dict[str, dict]:
    path = ledger_path(grid)
    if not path.is_file():
        return {}
    with path.open(newline="") as fh:
        return {row["cell_sha"]: row for row in csv.DictReader(fh)}


def write_ledger(grid: dict, rows: dict[str, dict]) -> None:
    path = ledger_path(grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


def is_done(row: dict | None) -> bool:
    return bool(row) and row["status"] == "done" and runs.find_run_dir(row["run_id"]) is not None


def run_cell(cell: Cell, grid: dict, split, data: DataSource, classifier: dict, device: torch.device, sha: str,
             bank: dict | None, provisional: list[str]) -> dict:  # fmt: skip
    """Train and log one cell; return its ledger row."""
    train = training_beats(data.arrays)
    val = split_part(data.arrays, "val", data.spec.rr_features)
    counts = augment.class_counts(train["y"])
    weights, added = None, concat_parts([])
    spec = grid["conditions"][cell.condition] or {}
    if cell.condition == "class_weighted":
        weights = augment.class_weights(counts, cell.strength)
    elif cell.condition != "none":
        extra = augment.fraction_of_n_targets(counts, cell.strength)
        if cell.condition == "random_oversampling":
            added = augment.random_oversample(train, extra, cell.seed)
        elif cell.condition == "smote":
            added = augment.smote_beats(train, extra, cell.seed, int(spec.get("k_neighbors", 5)))
        else:
            added = augment.take_from_bank(bank, extra)
    part = split_part(augment.with_added(train, added), "train", data.spec.rr_features)
    set_seed(cell.seed)
    model = build_model(classifier["model"], len(CLASSES), len(SIDE_FIELDS) if data.spec.rr_features else 0)
    train_cfg = TrainConfig(lr=float(classifier["lr"]), max_epochs=int(classifier["max_epochs"]), patience=int(classifier["patience"]),
                            batch_size=int(classifier["batch_size"]), weight_decay=float(classifier["weight_decay"]))  # fmt: skip
    fit = fit_classifier(model, part, val, train_cfg, split.selection_metric, seed=cell.seed, device=device, class_weights=weights)
    run_id = runs.make_run_id(grid["experiment"], cell.label, cell.seed, tag=cell.tag)
    group = (grid["experiment"], grid["name"], cell.label)
    metrics = {**fit.best_metrics, "selection_metric": split.selection_metric, "selection_value": fit.best_value,
               "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run, "stopped_early": fit.stopped_early}  # fmt: skip
    config = {
        "grid": grid["file"], "grid_name": grid["name"], "cell": asdict(cell), "cell_sha": sha, "policy": grid["policy"],
        "condition_settings": spec, "classifier": classifier, "class_weights": None if weights is None else weights.tolist(),
        "n_added": {c: int((added["y"] == i).sum()) for i, c in enumerate(CLASSES)}, "run_id": run_id,
        "split": {"name": split.name, "val": list(split.val), "selection_metric": split.selection_metric},
        "dataset": data.info, "provisional": bool(provisional), "provisional_reasons": provisional,
    }  # fmt: skip
    env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
    preds = {"probs": fit.val_probs, "y": val["y"], "record": val["record"], "r_idx": val["r_idx"]}
    runs.write_run(run_id, config, env, fit.history, metrics, preds, fit.state_dict, group=group)
    runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, group=group))
    return {
        "cell_sha": sha, "condition": cell.condition, "strength": "" if cell.strength is None else cell.strength,
        "architecture": cell.architecture, "seed": cell.seed, "run_id": run_id, "status": "done",
        "selection_metric": split.selection_metric, "selection_value": fit.best_value, "rare_macro_f1": fit.best_metrics["rare_macro_f1"],
        "n_added": int(len(added["y"])), "wall_clock_s": round(fit.wall_clock_s, 1),
    }  # fmt: skip


def run_grid(grid: dict, split, data: DataSource, device: torch.device, preview: bool, dry_run: bool = False, provisional: list[str] | None = None) -> dict:
    """Run every cell that is not done yet (or, with ``dry_run``, only report them)."""
    provisional = list(provisional or [])
    ledger = read_ledger(grid)
    report = {"done_before": [], "ran": [], "planned": []}
    banks: dict[tuple[str, int], tuple[dict, str]] = {}
    for cell in cells(grid):
        classifier = classifier_settings(grid, cell.architecture, preview)
        bank, bank_sha = None, None
        if cell.condition not in PLAIN_CONDITIONS:
            key = (cell.condition, cell.seed)
            if key not in banks:
                banks[key] = bank_for(grid, cell.condition, cell.seed, preview)
            bank, bank_sha = banks[key]
        sha = cell_sha(cell, grid, classifier, data, bank_sha)
        if is_done(ledger.get(sha)):
            report["done_before"].append(ledger[sha]["run_id"])
            continue
        if dry_run:
            report["planned"].append(asdict(cell))
            continue
        ledger[sha] = run_cell(cell, grid, split, data, classifier, device, sha, bank, provisional)
        write_ledger(grid, ledger)  # after every cell, so an interruption loses at most one cell
        report["ran"].append(ledger[sha]["run_id"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run (or list) the cells of a grid of conditions, strengths, architectures and seeds.")
    parser.add_argument("--config", required=True, help="a grid file, e.g. configs/phase4_full_grid/dry_run_baselines.yaml")
    parser.add_argument("--dry-run", action="store_true", help="list the cells and whether each is done; train nothing")
    parser.add_argument("--preview", action="store_true", help="run before the gates the grid needs; runs are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: auto)")
    args = parser.parse_args(argv)
    grid = load_grid(Path(args.config))
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
    print(f"ledger: {ledger_path(grid).relative_to(paths.repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
