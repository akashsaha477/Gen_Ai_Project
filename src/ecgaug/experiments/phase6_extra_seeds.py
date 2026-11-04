"""phase6_extra_seeds: seeds 3 and 4 for the conditions of the final evaluation (PREREGISTRATION §10 v4, P3; post hoc).

    python -m ecgaug.experiments.phase6_extra_seeds --config configs/phase6_extra_seeds/extra_seeds.yaml --plan
    python -m ecgaug.experiments.phase6_extra_seeds --config configs/phase6_extra_seeds/extra_seeds.yaml --part train [--device cuda]
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_extra_seeds --config configs/phase6_extra_seeds/extra_seeds.yaml --part test
    python -m ecgaug.experiments.phase6_extra_seeds --config configs/phase6_extra_seeds/extra_seeds.yaml --smoke

Nothing is chosen here. ``--part train`` (validation data only; Kaggle) trains the generator variants frozen at
the phase 3 gate at their full budget on the frozen DS1 training beats for the new seeds, exports their banks to
``data/synthetic/phase6_extra_seeds/`` (the frozen banks are never touched), and trains the 12 conditions of the
phase 4 gate (both architectures, the gate's strengths) on them; classifier seed s uses bank seed s. ``--part
test`` (local, after the final evaluation, one logged ``posthoc`` access) predicts DS2 with the new checkpoints,
takes seeds 0-2 from the final evaluation's saved predictions, and repeats the pre-registered families on 5 seeds
in ``results/phase6_extra_seeds/summary.json``, beside the 3-seed verdicts. The primary results stay the 3-seed ones.

The generators are trained without the phase 3 pilot scoring at intermediate checkpoints (it informs nothing at a
fixed setting), so a seed's random stream differs from what a phase 3 run of that seed would have drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError, logged_accesses
from ecgaug.data.contract import content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.loaders import load_test_posthoc
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, load_run_classifier
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.experiments.phase4_leakage_audit import generator_spec
from ecgaug.generators import build_generator
from ecgaug.generators.bank import bank_counts, sample_bank
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import FINAL_EVAL, PHASE3_GATE, PHASE4_GATE, PHASE6_EXTRA_SEEDS
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE6_EXTRA_SEEDS
CONFIG_KEYS = {"experiment", "seeds"}
GENERATIVE = {"cvae": "phase3_cvae", "wgan_gp": "phase3_wgan_gp"}
GRID_NAME = "extra_seeds"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if set(cfg["seeds"]) & {0, 1, 2}:
        raise ValueError(f"{path}: seeds 0-2 are the final evaluation's; only new seeds are trained here")
    return {**cfg, "file": str(path)}


def frozen_context() -> dict:
    """The phase 4 gate's conditions and strengths and the phase 3 gate's generator ids."""
    gate = workflow.manifest().get(PHASE4_GATE)
    frozen = workflow.manifest().get(PHASE3_GATE) or {}
    if not gate or not all(e in frozen for e in GENERATIVE.values()):
        raise PreconditionError("The phase 3 and phase 4 gates must be approved")
    full = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    return {"gate": gate, "full_grid": full, "generators": {c: frozen[e]["generator_id"] for c, e in GENERATIVE.items()}}


def bank_dir(generator_id: str) -> Path:
    return paths.repo_root() / "data" / "synthetic" / EXPERIMENT / generator_id


def train_generator(condition: str, spec: dict, seed: int, data: DataSource, device: torch.device, val_records: tuple = (),
                    steps: int | None = None) -> dict:  # fmt: skip
    """One frozen generator variant at its full budget for a new seed; its bank goes to ``bank_dir``."""
    folder = bank_dir(spec["generator_id"])
    path = folder / f"seed{seed}.npz"
    if path.is_file():
        return {"condition": condition, "seed": seed, "skipped": "bank exists"}
    steps = int(steps or spec["steps"])
    train = training_beats(data.arrays)
    generator = build_generator(spec["family"], spec["settings"], data.spec.n_samples, seed)
    fit = generator.fit(train, steps, seed, device, val_records=val_records, fractions=(1.0,), log_every=spec["log_every"])
    run_id = runs.make_run_id(EXPERIMENT, condition.replace("_", ""), seed, tag="generator")
    entry = None
    if not fit.diverged:
        bank = sample_bank(generator, train, bank_counts(train["y"]), seed=seed, source=f"{EXPERIMENT}/{spec['generator_id']}")
        folder.mkdir(parents=True, exist_ok=True)
        save_npz_deterministic(path, bank)
        entry = {"file": path.name, "sha256": sha256_file(path), "content_sha256": content_sha256(bank), "run_id": run_id, "steps": fit.steps_done}
        meta_path = folder / "bank.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"generator_id": f"{EXPERIMENT}/{spec['generator_id']}", "seeds": {}}
        meta["seeds"][str(seed)] = {**entry, "trained_on": data.info["source"]}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    metrics = {"selection_metric": "", "selection_value": float("nan"), "diverged": fit.diverged, "divergence": fit.divergence, "steps_done": fit.steps_done}
    config = {"experiment": EXPERIMENT, "condition": condition, **{k: spec[k] for k in ("generator_id", "family", "setting", "settings")},
              "steps": steps, "seed": seed, "dataset": data.info, "bank": entry}  # fmt: skip
    env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
    group = (EXPERIMENT, "generators")
    runs.write_run(run_id, config, env, fit.history, metrics, {}, generator.state_dict(), group=group)
    runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, status="diverged" if fit.diverged else "done", group=group))
    return {"condition": condition, "seed": seed, "run_id": run_id, "diverged": fit.diverged, "bank": entry}


def load_extra_bank(generator_id: str, seed: int) -> tuple[dict, str]:
    """A bank of this experiment, after checking the file against its ``bank.json``."""
    folder = bank_dir(generator_id)
    path, meta_path = folder / f"seed{seed}.npz", folder / "bank.json"
    entry = (json.loads(meta_path.read_text())["seeds"] if meta_path.is_file() else {}).get(str(seed))
    if entry is None or not path.is_file():
        raise PreconditionError(f"Bank {generator_id} seed {seed} of {EXPERIMENT} is missing: train the generators first")
    if sha256_file(path) != entry["sha256"]:
        raise PreconditionError(f"{path} changed since it was exported")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}, entry["sha256"]


def extra_grid(cfg: dict, ctx: dict) -> dict:
    """The full grid's policy and condition settings, with this experiment's banks and the new seeds."""
    full = ctx["full_grid"]
    conditions = {c: dict(full["conditions"][c] or {}) for c in full["conditions"]}
    for condition, generator_id in ctx["generators"].items():
        conditions[condition]["bank"] = f"{EXPERIMENT}/{generator_id}"
    return {**full, "experiment": EXPERIMENT, "name": GRID_NAME, "seeds": list(cfg["seeds"]), "conditions": conditions, "file": cfg["file"]}


def run_classifiers(cfg: dict, ctx: dict, data: DataSource, split, device: torch.device) -> list[dict]:
    """Every (architecture, condition) of the phase 4 gate at its strength, for the new seeds; cells resume."""
    grid = extra_grid(cfg, ctx)
    ledger = runner.read_ledger(grid)
    rows = []
    for arch, conditions in ctx["gate"]["chosen"].items():
        classifier = runner.classifier_settings(grid, arch, preview=ctx.get("smoke", False))
        for condition, chosen in conditions.items():
            for seed in cfg["seeds"]:
                cell = runner.Cell(condition, chosen["strength"], arch, int(seed))
                bank, bank_sha, bank_id = None, None, None
                if condition in ctx["generators"]:
                    bank, bank_sha = load_extra_bank(ctx["generators"][condition], int(seed))
                    bank_id = grid["conditions"][condition]["bank"]
                sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
                if runner.is_done(ledger.get(sha)):
                    rows.append(ledger[sha])
                    continue
                row = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, [])
                runner.write_cell(grid, row)
                ledger[sha] = row
                rows.append(row)
    return rows


def _done_cells(cfg: dict, ctx: dict) -> dict[tuple[str, str, int], str]:
    rows = runner.read_ledger(extra_grid(cfg, ctx)).values()
    return {(r["architecture"], r["condition"], int(r["seed"])): r["run_id"] for r in rows if runner.is_done(r)}


def test_report(cfg: dict, ctx: dict) -> dict:
    """5-seed DS2 statistics (one logged posthoc access) beside the final evaluation's 3-seed verdicts."""
    from ecgaug import final_eval

    if not logged_accesses("evaluation"):
        raise PreconditionError("Post-hoc analyses run only after the final evaluation")
    gate = ctx["gate"]
    cells = _done_cells(cfg, ctx)
    missing = [(a, c, s) for a, cs in gate["chosen"].items() for c in cs for s in cfg["seeds"] if (a, c, int(s)) not in cells]
    if missing:
        raise PreconditionError(f"{len(missing)} cells of the new seeds are not trained yet, e.g. {missing[:3]}")
    new_ids = sorted(cells.values())
    test = load_test_posthoc("v1", f"{EXPERIMENT}: DS2 predictions of seeds {cfg['seeds']}", run_ids=f"{len(new_ids)} new checkpoints (results/{EXPERIMENT}/summary.json)")
    rr = bool(json.loads((paths.processed_dir() / "v1" / "contract.json").read_text())["representation"]["rr_features"])
    x = torch.from_numpy(test["X"])
    side = torch.from_numpy(np.stack([test[f] for f in SIDE_FIELDS], axis=1)) if rr else None
    preds: dict[tuple[str, str], list[np.ndarray]] = {}
    seeds: dict[str, list[int]] = {}
    for arch, conditions in gate["chosen"].items():
        for condition, chosen in conditions.items():
            key = (arch, condition)
            for run_id in chosen["run_ids"]:  # seeds 0-2: the final evaluation's saved predictions
                with np.load(runs.find_run_dir(run_id) / "preds_test.npz") as saved:
                    if not (np.array_equal(saved["y"], test["y"]) and np.array_equal(saved["r_idx"], test["r_idx"])):
                        raise PreconditionError(f"{run_id}: saved test predictions do not match the frozen test beats")
                    preds.setdefault(key, []).append(saved["probs"])
            for seed in cfg["seeds"]:
                run_id = cells[(arch, condition, int(seed))]
                probs = predict_proba(load_run_classifier(run_id, rr), x, side, torch.device("cpu"))
                np.savez_compressed(runs.find_run_dir(run_id) / "preds_test.npz", probs=probs, y=test["y"], record=test["record"], r_idx=test["r_idx"])
                preds[key].append(probs)
            seeds[f"{arch}/{condition}"] = [0, 1, 2, *map(int, cfg["seeds"])]
    stats = final_eval.statistics(preds, test["y"], test["record"], gate)
    three = json.loads((paths.results_for(FINAL_EVAL) / "summary.json").read_text())
    verdicts = {family: {k: {"three_seeds": three["families"][family][k]["reject_holm"], "five_seeds": v["reject_holm"]} for k, v in items.items()}
                for family, items in stats["families"].items()}  # fmt: skip
    changed = sorted(f"{family}: {k}" for family, items in verdicts.items() for k, v in items.items() if v["three_seeds"] != v["five_seeds"])
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v4, P3", "seeds": seeds, "new_run_ids": new_ids,
               "conditions": stats["conditions"], "families": stats["families"], "verdicts": verdicts, "changed": changed,
               "sensitivity": stats["sensitivity"], "git_sha": runs.git_sha()}  # fmt: skip
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def smoke(cfg: dict) -> dict:
    """Generators and classifiers for one new seed on fake records in a temporary root; no gate and no DS2."""
    configs = paths.source_root() / "configs"
    specs = {c: generator_spec(f"{e}-A", configs) for c, e in GENERATIVE.items()}
    small = {**cfg, "seeds": cfg["seeds"][:1]}
    classifier = {"model": {"name": "probe_cnn", "channels": [8, 16, 16], "kernel": 5, "side_hidden": 8}, "lr": 1e-3, "max_epochs": 1,
                  "patience": 5, "batch_size": 256, "weight_decay": 0.01}  # fmt: skip
    with smoke_data() as (split, data):
        device = torch.device("cpu")
        generators = [train_generator(c, specs[c], int(small["seeds"][0]), data, device, steps=4) for c in specs]
        full = {"policy": {"kind": "fraction_of_n"}, "conditions": {"none": {}, "cvae": {}, "wgan_gp": {}}, "architectures": {"probe_cnn": classifier}}
        gate = {"chosen": {"probe_cnn": {"none": {"strength": None}, "cvae": {"strength": 0.25}, "wgan_gp": {"strength": 0.25}}}}
        ctx = {"gate": gate, "full_grid": full, "generators": {c: s["generator_id"] for c, s in specs.items()}, "smoke": True}
        rows = run_classifiers(small, ctx, data, split, device)
        again = run_classifiers(small, ctx, data, split, device)
        return {"generators": generators, "cells": len(rows), "skipped_on_rerun": sum(a["run_id"] == b["run_id"] for a, b in zip(rows, again)),
                "note": "fake data; the numbers mean nothing"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["train", "generators", "classifiers", "test"], default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan", action="store_true", help="what would run; reads no test data")
    parser.add_argument("--smoke", action="store_true", help="end to end on fake data in a temporary root")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2, default=str))
        return 0
    try:
        ctx = frozen_context()
        if args.plan:
            cells = [(a, c, chosen["strength"], s) for a, cs in ctx["gate"]["chosen"].items() for c, chosen in cs.items() for s in cfg["seeds"]]
            print(json.dumps({"generators": {c: generator_spec(g) for c, g in ctx["generators"].items()}, "cells": cells}, indent=2, default=str))
            return 0
        if args.part in ("train", "generators", "classifiers"):
            split = check_preconditions()
            data = load_data(split, preview=False)
            device = resolve_device(args.device)
            if args.part in ("train", "generators"):
                for condition, generator_id in ctx["generators"].items():
                    for seed in cfg["seeds"]:
                        print(json.dumps(train_generator(condition, generator_spec(generator_id), int(seed), data, device, split.val), default=str))
            if args.part in ("train", "classifiers"):
                print(f"{len(run_classifiers(cfg, ctx, data, split, device))} cells done")
        if args.part == "test":
            summary = test_report(cfg, ctx)
            for family, items in summary["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:32s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            print("changed verdicts: " + (", ".join(summary["changed"]) or "none"))
            print(f"written: results/{EXPERIMENT}/summary.json")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
