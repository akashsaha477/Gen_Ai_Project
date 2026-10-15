"""Phase 1 experiment phase1_input_representation: compare input representations with the probe CNN.

Usage:
    python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml            # the variant experiment
    python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml --check    # build DS1 inputs, no training
    python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml --smoke    # fake data, end to end

A real run needs two things first: an approved validation split (configs/validation_split/approved.yaml)
and a clean git tree. ``--check`` reads DS1 only and trains nothing. ``--smoke`` never touches real data
or the repository's results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ecgaug import figures, paths, runs
from ecgaug.names import PHASE1_DATA, PHASE1_INPUT_REPRESENTATION
from ecgaug.data.contract import build_arrays, build_candidate, content_sha256
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb
from ecgaug.data.loaders import load_candidate
from ecgaug.data.records import CLASSES, DS1
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, selection_metric_for
from ecgaug.experiments.common import (
    BUDGET_KEYS,
    SIDE_FIELDS,
    PreconditionError,
    best_grid_point,
    check_preconditions,
    split_part,
    temporary_root,
    train_grid,
)
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE1_INPUT_REPRESENTATION
VARIANT_KEYS = {"base", "variant", "description", "representation"}


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
        "config_name": path.stem,
    }


def run_variant(cfg: dict, split: Split, device_name: str | None = None, require_all_ds1: bool = True) -> dict:
    """Train the probe for every learning rate and seed, log every run, and write the result file."""
    spec = RepresentationSpec(**cfg["representation"])
    variant = cfg["variant"]
    selection = split.selection_metric
    device = resolve_device(device_name or cfg.get("device", "cpu"))

    dataset_dir = build_candidate(spec, split, require_all_ds1=require_all_ds1)
    arrays = load_candidate(spec.name)
    dataset_hash = content_sha256(arrays)
    train, val = split_part(arrays, "train", spec.rr_features), split_part(arrays, "val", spec.rr_features)
    grid, n_params = train_grid(
        cfg, split, spec, train, val, experiment=EXPERIMENT, label=variant, device=device,
        dataset={"path": str(dataset_dir), "content_sha256": dataset_hash},
        group=(EXPERIMENT, cfg.get("config_name", variant)),
    )  # fmt: skip
    chosen = best_grid_point(grid)
    result = {
        "experiment": EXPERIMENT,
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
        "budget": {k: cfg[k] for k in BUDGET_KEYS},
        "budget_deviations": [],
        "device": str(device),
        "git_sha": runs.git_sha(),
        "dataset_content_sha256": dataset_hash,
    }
    out_dir = paths.results_for(EXPERIMENT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    figures.safe_for_result(out_dir / f"{variant}.json")
    notes = out_dir / f"{variant}_notes.md"
    if not notes.exists():
        notes.write_text(
            f"# {EXPERIMENT}, variant {variant}: notes\n\n"
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


def smoke(cfg: dict) -> dict:
    """Run the whole variant pipeline on fake records in a temporary root (one seed, two epochs)."""
    train_records, val_records = (101, 106, 108, 114), (118, 201)
    with temporary_root() as root:
        write_fake_mitdb(paths.raw_dir(), train_records + val_records, seconds=120.0)
        counts = pd.DataFrame([count_record(r) for r in train_records + val_records])
        split = Split(name="smoke", train=train_records, val=val_records)
        check_split(split, require_all_ds1=False)
        split = Split(
            name="smoke", train=train_records, val=val_records, selection_metric=selection_metric_for(counts, split)
        )
        smoke_cfg = {**cfg, "seeds": [0], "lr_grid": [1e-3], "max_epochs": 2}
        result = run_variant(smoke_cfg, split, device_name="cpu", require_all_ds1=False)
        run_dir = runs.find_run_dir(result["run_ids"][0])
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
            "result_file": f"results/{EXPERIMENT}/{result['variant']}.json",
            "note": "fake data; the numbers of a smoke run mean nothing and are discarded",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase1_input_representation: input-representation variants with the probe CNN.")
    parser.add_argument("--config", required=True, help="configs/phase1_input_representation/<variant>.yaml")
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
        out = paths.results_for(PHASE1_DATA) / f"representation_check_{cfg['variant']}.json"
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
