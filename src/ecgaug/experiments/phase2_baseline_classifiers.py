"""Phase 2 experiment phase2_baseline_classifiers: four classifier architectures on real, imbalanced data.

Usage:
    python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/A_shallow_cnn.yaml
    python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/A_shallow_cnn.yaml --smoke
    add --preview [--representation C] to run before the phase 1 gate (results are marked provisional)

Each variant is trained for every seed and learning rate of the shared budget, with unweighted
cross-entropy and no reweighting or augmentation, on the training records of the frozen dataset:
exactly like a phase1_input_representation variant, except that the architecture differs. At the
phase 2 gate the selection rule picks the ``cnn`` from A and B and the ``rnn`` from C and D (ties go to
fewer parameters). The ``cnn`` becomes the pilot classifier of phase 3; both enter the phase 4 grid.

Official runs need the phase 1 gate (the frozen dataset), the approved validation split and a clean
git tree. The test set is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ecgaug import figures, paths, runs, workflow
from ecgaug.experiments.common import PreconditionError, best_grid_point, check_preconditions, split_part, train_grid
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.names import PHASE2_BASELINE_CLASSIFIERS
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE2_BASELINE_CLASSIFIERS
VARIANT_KEYS = {"base", "variant", "description", "role", "model"}
SHARED_KEYS = {"experiment", "seeds", "lr_grid", "max_epochs", "patience", "min_epochs", "batch_size", "weight_decay", "device"}
BUDGET = ("seeds", "lr_grid", "max_epochs", "patience", "min_epochs", "batch_size", "weight_decay")
ROLES = ("cnn", "rnn")


def load_config(path: Path) -> dict:
    """Merge a variant (letter, description, role, model) with the shared budget it points to."""
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared budget file, not in a variant")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    if set(shared) != SHARED_KEYS:
        raise ValueError(f"{path.parent / variant['base']}: expected exactly the keys {sorted(SHARED_KEYS)}")
    if variant["role"] not in ROLES:
        raise ValueError(f"{path}: role must be one of {ROLES}")
    return {
        **shared,
        "variant": str(variant["variant"]),
        "description": variant.get("description", ""),
        "role": variant["role"],
        "model": variant["model"],
        "config_name": path.stem,
    }


def run_variant(cfg: dict, split, data: DataSource, device_name: str | None = None, reasons: list[str] | None = None) -> dict:
    """Train every seed and learning rate, log every run, and write ``results/<experiment>/<letter>.json``."""
    reasons = list(reasons or [])
    device = resolve_device(device_name or cfg.get("device", "auto"))
    train = split_part(data.arrays, "train", data.spec.rr_features)
    val = split_part(data.arrays, "val", data.spec.rr_features)
    grid, n_params = train_grid(
        cfg, split, data.spec, train, val, experiment=EXPERIMENT, label=cfg["variant"], device=device,
        dataset=data.info, group=(EXPERIMENT, cfg["config_name"]),
    )  # fmt: skip
    chosen = best_grid_point(grid)
    result = {
        "experiment": EXPERIMENT, "variant": cfg["variant"], "description": cfg["description"], "role": cfg["role"],
        "model": cfg["model"], "n_parameters": n_params, "provisional": bool(reasons), "provisional_reasons": reasons,
        "representation": data.spec.to_dict(), "dataset": data.info,
        "split": {"name": split.name, "val": list(split.val)}, "selection_metric": split.selection_metric,
        "seeds": list(cfg["seeds"]), "chosen_lr": chosen["lr"], "mean": chosen["mean"], "sd": chosen["sd"],
        "tie_break_mean": chosen.get("tie_break_mean"), "tie_break_sd": chosen.get("tie_break_sd"),
        "per_seed": chosen["per_seed"], "run_ids": [r["run_id"] for r in chosen["per_seed"]], "lr_grid": grid,
        "wall_clock_s": float(sum(e["wall_clock_s"] for e in grid.values())),
        "budget": {k: cfg[k] for k in BUDGET}, "budget_deviations": [], "device": str(device), "git_sha": runs.git_sha(),
        "dataset_content_sha256": data.info.get("content_sha256"),
    }  # fmt: skip
    out_dir = paths.results_for(EXPERIMENT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    notes = out_dir / f"{cfg['variant']}_notes.md"
    if not notes.exists():
        notes.write_text(
            f"# {EXPERIMENT}, variant {cfg['variant']}: notes\n\n"
            "What worked:\n\nWhat failed:\n\nSurprises:\n\nSanity checks (learning curves, early-stop epochs, "
            "confusion matrix):\n\nFigure paths:\n"
        )
    figures.safe_for_result(out_dir / f"{cfg['variant']}.json")
    return result


def smoke(cfg: dict) -> dict:
    """The whole variant on fake records in a temporary root (one seed, one learning rate, two epochs)."""
    with smoke_data() as (split, data):
        result = run_variant({**cfg, "seeds": [0], "lr_grid": [1e-3], "max_epochs": 2}, split, data, "cpu", [data.provisional])
        run_dir = runs.find_run_dir(result["run_ids"][0])
        produced = sorted(p.name for p in run_dir.iterdir())
        missing = sorted({"config.yaml", "env.json", "train_log.csv", "metrics_val.json", "preds_val.npz", "checkpoints"} - set(produced))
        if missing:
            raise RuntimeError(f"Smoke run did not produce {missing}")
        return {
            "variant": result["variant"], "role": result["role"], "n_parameters": result["n_parameters"],
            "selection_metric": result["selection_metric"], "run_files": produced,
            "result_file": f"results/{EXPERIMENT}/{result['variant']}.json",
            "note": "fake data; the numbers of a smoke run mean nothing and are discarded",
        }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase2_baseline_classifiers: baseline classifier variants on real data.")
    parser.add_argument("--config", required=True, help="configs/phase2_baseline_classifiers/<variant>.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: from the shared budget)")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    parser.add_argument("--preview", action="store_true", help="run before the phase 1 gate; results are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    if args.representation and not args.preview:
        print("--representation is for previews only; official runs use the dataset frozen at the phase 1 gate")
        return 2
    try:
        if not args.preview and workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first (python -m ecgaug.workflow status), or add --preview.")
        split = check_preconditions()
        data = load_data(split, args.preview, args.representation)
    except PreconditionError as err:
        print(f"Refusing to run {EXPERIMENT} variant {cfg['variant']}: {err}")
        return 2
    result = run_variant(cfg, split, data, args.device, [data.provisional] if data.provisional else [])
    print(
        f"variant {result['variant']} ({result['role']}, {result['n_parameters']:,} parameters): {result['selection_metric']} = "
        f"{result['mean']:.4f} ± {result['sd']:.4f} (lr {result['chosen_lr']:g}); runs: {', '.join(result['run_ids'])}"
        + (" — PROVISIONAL: " + "; ".join(result["provisional_reasons"]) if result["provisional"] else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
