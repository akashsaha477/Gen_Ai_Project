"""phase2_ptbxl_pretraining experiment: pretrain on PTB-XL beats, fine-tune on MIT-BIH, compare with from scratch.

Usage:
    python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --ptbxl-root <PTB-XL folder>
    python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --ptbxl-download
    python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --smoke   # fake data, end to end

One command does everything, so a single Kaggle session is enough:

1. PTB-XL preprocessing (``ecgaug.data.ptbxl``), unless a complete build is already on disk.
2. Pretraining, once per seed: the classifier with a 5-output head learns the diagnostic superclasses
   of each beat's recording (multi-label, binary cross-entropy, early stopping on PTB-XL fold 9).
3. Fine-tuning on the MIT-BIH training records with the usual budget, twice: from the pretrained
   weights (head replaced) and from scratch. Same seeds, learning rates, epochs and early stopping.
4. The selection rule decides whether pretraining wins; it has to beat from-scratch by more than the
   tie margin, because from-scratch is the simpler option.

The representation and the fine-tuning budget are read from the phase1_input_representation config the variant points
to, so they cannot drift from phase1_input_representation. A real run needs the approved validation split and a clean
git tree. The test set is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ecgaug import figures, paths, runs, workflow
from ecgaug.names import PHASE2_PTBXL_PRETRAINING
from ecgaug.data import ptbxl, raw
from ecgaug.data.contract import build_candidate, content_sha256
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb, write_fake_ptbxl
from ecgaug.data.loaders import load_candidate
from ecgaug.data.records import CLASS_INDEX
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, selection_metric_for
from ecgaug.eval.selection import Candidate, select_winner
from ecgaug.experiments import phase1_input_representation as m1
from ecgaug.experiments.common import (
    BUDGET_KEYS,
    SIDE_FIELDS,
    PreconditionError,
    best_grid_point,
    check_preconditions,
    mean_sd,
    split_part,
    temporary_root,
    train_grid,
)
from ecgaug.models import build_model, load_encoder
from ecgaug.train.loop import TrainConfig, fit_multilabel, resolve_device, set_seed

VARIANT_KEYS = {"base", "variant", "description", "representation_from"}
SHARED_KEYS = {"experiment", "pretrain", "min_epochs"}  # min_epochs: fine-tuning only (DECISIONS A15)
PRETRAIN_KEYS = {"lr", "max_epochs", "patience", "batch_size", "weight_decay"}
ARMS = ("scratch", "pretrained")
EXPERIMENT = PHASE2_PTBXL_PRETRAINING


def load_config(path: Path) -> dict:
    """Merge a pretraining variant with its phase1_input_representation config and the shared pretraining settings.

    The representation, model and fine-tuning budget come from the phase1_input_representation config named by
    ``representation_from``. The shared file may set only the phase, the module and the ``pretrain``
    block, so the fine-tuning budget is the phase1_input_representation budget by construction.
    """
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: unexpected keys {sorted(extra)}")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    if set(shared) - SHARED_KEYS or set(shared["pretrain"]) != PRETRAIN_KEYS:
        raise ValueError(
            f"{path.parent / variant['base']} may define only {sorted(SHARED_KEYS)}, with pretrain keys "
            f"{sorted(PRETRAIN_KEYS)}; the fine-tuning budget comes from phase1_input_representation"
        )
    base = m1.load_config((path.parent / variant["representation_from"]).resolve())
    if str(variant["variant"]) != base["variant"]:
        raise ValueError(f"{path}: variant {variant['variant']} points at the phase1_input_representation config of {base['variant']}")
    return {
        **base,
        **shared,
        "variant": str(variant["variant"]),
        "description": variant.get("description", base.get("description", "")),
        "representation_source": str(variant["representation_from"]),
        "config_name": path.stem,
    }


def ptbxl_part(arrays: dict[str, np.ndarray], spec: RepresentationSpec, split: str) -> dict[str, np.ndarray]:
    """PTB-XL beats of one split in the form the training loop expects."""
    part = ptbxl.beat_arrays(arrays, spec, split)
    if not spec.rr_features:
        part["side"] = None
    return part


def pretrain(cfg: dict, spec: RepresentationSpec, train: dict, val: dict, device) -> dict[int, dict]:
    """Pretrain once per seed; log each run. Returns ``{seed: {run_id, state_dict, macro_auroc, ...}}``."""
    settings = cfg["pretrain"]
    train_cfg = TrainConfig(
        lr=float(settings["lr"]),
        max_epochs=settings["max_epochs"],
        patience=settings["patience"],
        batch_size=settings["batch_size"],
        weight_decay=settings["weight_decay"],
    )
    out = {}
    for seed in cfg["seeds"]:
        run_id = runs.make_run_id(EXPERIMENT, f"pretrain{cfg['variant']}", seed)
        set_seed(seed)
        model = build_model(cfg["model"], len(ptbxl.SUPERCLASSES), len(SIDE_FIELDS) if spec.rr_features else 0)
        fit = fit_multilabel(model, train, val, train_cfg, ptbxl.SUPERCLASSES, seed=seed, device=device)
        metrics = {
            **fit.best_metrics,
            "selection_metric": "ptbxl_val_macro_auroc",
            "selection_value": fit.best_value,
            "best_epoch": fit.best_epoch,
            "epochs_run": fit.epochs_run,
            "stopped_early": fit.stopped_early,
            "n_train_beats": int(len(train["y"])),
            "n_val_beats": int(len(val["y"])),
        }
        env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
        run_config = {**cfg, "seed": seed, "run_id": run_id, "stage": "pretrain", "labels": list(ptbxl.SUPERCLASSES)}
        preds = {"probs": fit.val_probs, "y": val["y"], "ecg_id": val["ecg_id"]}
        group = (EXPERIMENT, cfg.get("config_name", cfg["variant"]), "pretrain")
        runs.write_run(run_id, run_config, env, fit.history, metrics, preds, fit.state_dict, group=group)
        runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, group=group))
        out[seed] = {
            "run_id": run_id,
            "state_dict": fit.state_dict,
            "macro_auroc": fit.best_value,
            "auroc": fit.best_metrics["per_label"],
            "best_epoch": fit.best_epoch,
            "epochs_run": fit.epochs_run,
            "wall_clock_s": fit.wall_clock_s,
        }
    return out


def run(cfg: dict, split: Split, ptbxl_arrays: dict, device_name: str | None = None, require_all_ds1: bool = True) -> dict:
    """Pretrain, fine-tune both arms, apply the selection rule and write the result file."""
    spec = RepresentationSpec(**cfg["representation"])
    variant = cfg["variant"]
    device = resolve_device(device_name or cfg.get("device", "cpu"))

    dataset_dir = build_candidate(spec, split, require_all_ds1=require_all_ds1)
    arrays = load_candidate(spec.name)
    dataset = {"path": str(dataset_dir), "content_sha256": content_sha256(arrays)}
    train, val = split_part(arrays, "train", spec.rr_features), split_part(arrays, "val", spec.rr_features)

    pre_train, pre_val = ptbxl_part(ptbxl_arrays, spec, "train"), ptbxl_part(ptbxl_arrays, spec, "val")
    normal = pre_train["y"][:, ptbxl.SUPERCLASSES.index("NORM")] > 0.5
    alignment = ptbxl.alignment_report(pre_train["X"][normal], train["X"][train["y"] == CLASS_INDEX["N"]], spec)

    pretrained = pretrain(cfg, spec, pre_train, pre_val, device)

    def from_pretrained(model, seed: int) -> dict:
        copied = load_encoder(model, pretrained[seed]["state_dict"])
        return {"kind": "ptbxl_pretrained", "run_id": pretrained[seed]["run_id"], "tensors_copied": len(copied)}

    arms = {}
    for arm, kind, init in (("scratch", "scratch", None), ("pretrained", "finetune", from_pretrained)):
        grid, n_params = train_grid(
            cfg, split, spec, train, val, experiment=EXPERIMENT, label=f"{kind}{variant}",
            device=device, dataset=dataset, init=init, group=(EXPERIMENT, cfg.get("config_name", variant), kind),
        )  # fmt: skip
        chosen = best_grid_point(grid)
        arms[arm] = {
            "chosen_lr": chosen["lr"],
            "mean": chosen["mean"],
            "sd": chosen["sd"],
            "tie_break_mean": chosen.get("tie_break_mean"),
            "tie_break_sd": chosen.get("tie_break_sd"),
            "per_seed": chosen["per_seed"],
            "run_ids": [r["run_id"] for r in chosen["per_seed"]],
            "lr_grid": grid,
            "wall_clock_s": float(sum(e["wall_clock_s"] for e in grid.values())),
            "n_parameters": n_params,
        }
    pretrain_seconds = float(sum(p["wall_clock_s"] for p in pretrained.values()))
    arms["pretrained"]["wall_clock_s"] += pretrain_seconds

    # From scratch is the simpler option, so pretraining has to win by more than the tie margin.
    decision = select_winner(
        [
            Candidate(
                arm, arms[arm]["mean"], arms[arm]["sd"], complexity=ARMS.index(arm),
                wall_clock_s=arms[arm]["wall_clock_s"], tie_mean=arms[arm]["tie_break_mean"],
                tie_sd=arms[arm]["tie_break_sd"],
            )  # fmt: skip
            for arm in ARMS
        ]
    )
    auroc_mean, auroc_sd = mean_sd([p["macro_auroc"] for p in pretrained.values()])
    winner = workflow.phase1_winner()
    official = winner == variant
    ptbxl_report = ptbxl.build_report() or {}
    result = {
        "experiment": EXPERIMENT,
        "variant": variant,
        "description": cfg.get("description", ""),
        "representation": spec.to_dict(),
        "provisional": not official,
        "provisional_reason": None if official else (
            "preview: the phase 1 gate has not been approved yet" if winner is None
            else f"preview: the phase 1 gate chose variant {winner}, not {variant}"
        ),
        "split": {"name": split.name, "val": list(split.val)},
        "selection_metric": split.selection_metric,
        "seeds": list(cfg["seeds"]),
        "winner": decision["winner"],
        "decision": decision,
        "difference_pretrained_minus_scratch": arms["pretrained"]["mean"] - arms["scratch"]["mean"],
        "arms": arms,
        "pretraining": {
            "settings": cfg["pretrain"],
            "labels": list(ptbxl.SUPERCLASSES),
            "val_macro_auroc_mean": auroc_mean,
            "val_macro_auroc_sd": auroc_sd,
            "per_seed": [{"seed": s, **{k: v for k, v in p.items() if k != "state_dict"}} for s, p in pretrained.items()],
            "n_train_beats": int(len(pre_train["y"])),
            "n_val_beats": int(len(pre_val["y"])),
            "wall_clock_s": pretrain_seconds,
        },
        "ptbxl_data": {
            "content_sha256": ptbxl_report.get("content_sha256"),
            "recordings_kept": ptbxl_report.get("recordings_kept"),
            "beats_kept": ptbxl_report.get("beats_kept"),
            "partial_run": ptbxl_report.get("partial_run"),
        },
        "alignment_with_mitbih_train": alignment,
        "budget": {**{k: cfg[k] for k in BUDGET_KEYS}, "min_epochs": int(cfg.get("min_epochs", 0))},
        "budget_deviations": [],
        "device": str(device),
        "git_sha": runs.git_sha(),
        "dataset_content_sha256": dataset["content_sha256"],
    }
    out_dir = paths.results_for(EXPERIMENT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    figures.safe_for_result(out_dir / f"{variant}.json")
    notes = out_dir / f"{variant}_notes.md"
    if not notes.exists():
        notes.write_text(
            f"# phase2_ptbxl_pretraining experiment, representation {variant}: notes\n\n"
            "What worked:\n\nWhat failed:\n\nSurprises:\n\nSanity checks (pretraining AUROC per label, "
            "alignment with MIT-BIH, learning curves, early-stop epochs):\n\nFigure paths:\n"
        )
    return result


def ensure_ptbxl(root: str | None, workers: int, verify: bool, download: bool = False) -> dict[str, np.ndarray]:
    """The complete preprocessed PTB-XL arrays, building them if they are not on disk.

    The raw files come from ``root`` (an attached copy) or, with ``download``, from PhysioNet.
    """
    if not ptbxl.is_full_build():
        if not root and not download:
            found = "only a partial (sample) build" if ptbxl.build_report() else "no build"
            raise PreconditionError(
                f"PTB-XL is not preprocessed here ({found} on disk). Pass --ptbxl-root with the folder "
                "that holds ptbxl_database.csv and records500/, or --ptbxl-download to fetch it from PhysioNet."
            )
        try:
            if root:
                source = ptbxl.find_root(Path(root))
            else:
                print("Downloading PTB-XL 1.0.3 from PhysioNet (1.8 GB)...", flush=True)
                source = raw.download_ptbxl(paths.repo_root() / "data" / "raw" / "ptbxl_physionet")
        except (FileNotFoundError, OSError) as err:
            raise PreconditionError(str(err)) from err
        try:
            report = ptbxl.prepare(source, workers=workers, verify=verify)
        except FileNotFoundError as err:
            raise PreconditionError(
                f"The PTB-XL copy at {source} is incomplete: {err}. The pipeline needs the 500 Hz files "
                "(records500/). Attach another copy or use --ptbxl-download."
            ) from err
        print(
            f"PTB-XL preprocessed: {report['recordings_kept']} recordings, {report['beats_kept']} beats; "
            f"excluded {report['excluded_by_metadata']} and {report['excluded_by_signal']}"
        )
        if report.get("partial_run"):
            raise PreconditionError("The PTB-XL folder is incomplete: some recording files are missing.")
    return ptbxl.load()


def smoke(cfg: dict) -> dict:
    """Run the whole arm on fake MIT-BIH and fake PTB-XL records in a temporary root."""
    train_records, val_records = (101, 106, 108, 114), (118, 201)
    with temporary_root() as root:
        write_fake_mitdb(paths.raw_dir(), train_records + val_records, seconds=120.0)
        write_fake_ptbxl(root / "ptbxl_raw", n_extra=40)
        ptbxl.prepare(root / "ptbxl_raw", verify=False)
        counts = pd.DataFrame([count_record(r) for r in train_records + val_records])
        split = Split(name="smoke", train=train_records, val=val_records)
        check_split(split, require_all_ds1=False)
        split = Split(
            name="smoke", train=train_records, val=val_records, selection_metric=selection_metric_for(counts, split)
        )
        smoke_cfg = {**cfg, "seeds": [0], "lr_grid": [1e-3], "max_epochs": 2}
        smoke_cfg["pretrain"] = {**cfg["pretrain"], "max_epochs": 2}
        result = run(smoke_cfg, split, ptbxl.load(), device_name="cpu", require_all_ds1=False)
        run_ids = [result["pretraining"]["per_seed"][0]["run_id"], *(result["arms"][a]["run_ids"][0] for a in ARMS)]
        missing = [r for r in run_ids if not (runs.find_run_dir(r) / "checkpoints" / "best.pt").is_file()]
        if missing or not (paths.results_for(EXPERIMENT) / f"{result['variant']}.json").is_file():
            raise RuntimeError(f"Smoke run is incomplete (missing runs: {missing})")
        return {
            "variant": result["variant"],
            "runs": run_ids,
            "selection_metric": result["selection_metric"],
            "winner_field_present": result["winner"] in ARMS,
            "pretrained_init": yaml.safe_load((runs.find_run_dir(run_ids[2]) / "config.yaml").read_text())["init"],
            "run_folders": sorted({str(runs.find_run_dir(r).parent.relative_to(paths.runs_dir())) for r in run_ids}),
            "result_file": f"results/phase2_ptbxl_pretraining/{result['variant']}.json",
            "note": "fake data; the numbers of a smoke run mean nothing and are discarded",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTB-XL pretraining, MIT-BIH fine-tuning, comparison with from scratch.")
    parser.add_argument("--config", required=True, help="configs/phase2_ptbxl_pretraining/<variant>.yaml")
    parser.add_argument("--ptbxl-root", default=None, help="folder holding ptbxl_database.csv, or any folder above it")
    parser.add_argument("--ptbxl-download", action="store_true", help="fetch PTB-XL from PhysioNet if no copy is given")
    parser.add_argument("--workers", type=int, default=4, help="worker processes for PTB-XL preprocessing")
    parser.add_argument("--skip-verify", action="store_true", help="do not compare PTB-XL files with PhysioNet checksums")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: from the shared config)")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    parser.add_argument(
        "--preview", action="store_true",
        help="run before the phase 1 gate, or for a variant that the phase 1 gate did not choose; the result is marked provisional",
    )  # fmt: skip
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    try:
        winner = workflow.phase1_winner()
        if not args.preview and winner is None:
            raise PreconditionError(
                "Phase 1 is not finished: the phase 1 gate has not chosen the input variant yet. Run the phase1_input_representation "
                "variants and the phase 1 gate first (python -m ecgaug.workflow status), or add --preview."
            )
        if not args.preview and winner != cfg["variant"]:
            right = workflow.variant_configs()[winner].name
            raise PreconditionError(
                f"The phase 1 gate chose variant {winner}; run configs/phase2_ptbxl_pretraining/{right}, or add --preview."
            )
        split = check_preconditions()
        ptbxl_arrays = ensure_ptbxl(args.ptbxl_root, args.workers, not args.skip_verify, download=args.ptbxl_download)
    except PreconditionError as err:
        print(f"Refusing to run the phase2_ptbxl_pretraining experiment for {cfg['variant']}: {err}")
        return 2
    result = run(cfg, split, ptbxl_arrays, device_name=args.device)
    metric = result["selection_metric"]
    for arm in ARMS:
        a = result["arms"][arm]
        print(f"{arm:>10}: {metric} = {a['mean']:.4f} ± {a['sd']:.4f} (lr {a['chosen_lr']:g}); runs: {', '.join(a['run_ids'])}")
    print(
        f"pretraining AUROC on PTB-XL fold 9: {result['pretraining']['val_macro_auroc_mean']:.4f}; "
        f"difference pretrained - scratch: {result['difference_pretrained_minus_scratch']:+.4f}; "
        f"winner by the selection rule: {result['winner']}"
        + (f" ({result['provisional_reason']})" if result["provisional"] else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
