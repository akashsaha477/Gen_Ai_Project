"""phase4_leakage_audit: the same pipeline under the common intra-patient protocol (PLAN Phase 4, optional;
DECISIONS A4, A16; paper contribution C2, Table 5).

    python -m ecgaug.experiments.phase4_leakage_audit --config configs/phase4_leakage_audit/audit.yaml --plan
    ALLOW_TEST_AUDIT=1 python -m ecgaug.experiments.phase4_leakage_audit --config configs/phase4_leakage_audit/audit.yaml [--part generators|classifiers|report] [--device cuda]
    python -m ecgaug.experiments.phase4_leakage_audit --config configs/phase4_leakage_audit/audit.yaml --smoke

Only the split changes. The beats of DS1 and DS2 are pooled, and each class's beats are assigned at
random to train, validation and test, with as many beats of each class in each part as the
inter-patient protocol has; so a patient's beats land in all three parts. Everything else is the frozen
pipeline: the representation (phase 1 gate), the cnn and its settings (phase 2 gate), the generator
variants frozen at the phase 3 gate, retrained at their full budget on the intra-patient training
beats, and the augmentation policy of the phase 4a gate at medium strength (the middle of the full
grid's strengths). Conditions none, cVAE and WGAN-GP; the grid seeds; classifier seed s uses bank seed s.

It reads DS2, so it runs only after the final evaluation (an evaluation entry in
``docs/TEST_ACCESS_LOG.md`` and ``results/final_eval/summary.json``), with ``ALLOW_TEST_AUDIT=1``; each
invocation that reads DS2 is logged as an audit access. Runs go to ``runs/audit/``, banks to
``data/synthetic/audit/``; the main pipeline refuses both. No decision of any kind uses this experiment.

Report (``--part report``): per condition, the intra-patient test rare-class macro-F1 (mean and SD over
seeds) beside the inter-patient test result of the final evaluation, and each side's gain over none,
in ``results/phase4_leakage_audit/summary.json``; ``python -m ecgaug.paper`` builds the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError, log_test_access, logged_accesses, require_test_access
from ecgaug.data.contract import build_arrays, content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.loaders import frozen_test_sha256, load_test_for_audit
from ecgaug.data.records import CLASSES, DS2
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split
from ecgaug.eval import metrics as M
from ecgaug.experiments import generator_runs as gr
from ecgaug.experiments.common import PreconditionError, check_preconditions, load_run_classifier, mean_sd, split_part
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.generators import build_generator
from ecgaug.generators.bank import bank_counts, concat_parts, sample_bank
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import FINAL_EVAL, PHASE1_GATE, PHASE3_GATE, PHASE4_LEAKAGE_AUDIT
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE4_LEAKAGE_AUDIT
CONFIG_KEYS = {"experiment", "split_seed", "seeds", "architecture", "conditions", "strength"}
GENERATIVE = {"cvae": "phase3_cvae", "wgan_gp": "phase3_wgan_gp"}  # condition -> the phase 3 experiment of its generator
RUN_ROOT = ("audit",)
GRID_NAME = "audit"
SMOKE_CLASSIFIER = {"model": {"name": "probe_cnn", "channels": [8, 16, 16], "kernel": 5, "side_hidden": 8}, "lr": 1e-3, "max_epochs": 1,
                    "patience": 5, "batch_size": 256, "weight_decay": 0.01}  # fmt: skip


# --------------------------------------------------------------------------------------------------
# config, gates and the split
# --------------------------------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS:
        raise ValueError(f"{path}: expected exactly the keys {sorted(CONFIG_KEYS)}")
    if cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: experiment must be {EXPERIMENT}")
    if "none" not in cfg["conditions"] or set(cfg["conditions"]) - {"none", *GENERATIVE}:
        raise ValueError(f"{path}: conditions are none plus any of {sorted(GENERATIVE)}")
    if cfg["strength"] != "medium":
        raise ValueError(f"{path}: the strength is 'medium' (PLAN Phase 4)")
    return {**cfg, "file": str(path)}


def final_eval_done() -> bool:
    return bool(logged_accesses("evaluation")) and (paths.results_for(FINAL_EVAL) / "summary.json").is_file()


def medium_strength(grid: dict, condition: str) -> float | None:
    """The middle of a condition's strengths in the full grid (the upper of two; None for balance_to_n)."""
    if runner.policy_of(grid)["kind"] == "balance_to_n":
        return None
    strengths = sorted(float(s) for s in (grid["conditions"][condition] or {})["strengths"])
    return strengths[len(strengths) // 2]


def frozen_context(cfg: dict) -> dict:
    """What the audit takes from the approved gates; raises PreconditionError when something is missing."""
    if not final_eval_done():
        raise PreconditionError("The leakage audit reads DS2, so it runs only after the final evaluation (PLAN Phase 4, DECISIONS A4)")
    grid_path = paths.repo_root() / workflow.FULL_GRID_FILE
    if not grid_path.is_file():
        raise PreconditionError(f"{workflow.FULL_GRID_FILE} is missing: the phase 4a gate writes it")
    full = runner.load_grid(grid_path)
    classifier = runner.classifier_settings({"architectures": {cfg["architecture"]: "phase2_gate"}}, cfg["architecture"], preview=False)
    frozen = workflow.manifest().get(PHASE3_GATE) or {}
    generators = {}
    for condition in cfg["conditions"]:
        if condition in GENERATIVE:
            if GENERATIVE[condition] not in frozen:
                raise PreconditionError(f"The phase 3 gate has not frozen a {condition} generator")
            generators[condition] = frozen[GENERATIVE[condition]]["generator_id"]
    return {"policy": full["policy"], "strengths": {c: medium_strength(full, c) for c in generators}, "classifier": classifier,
            "generators": generators, "full_grid": str(workflow.FULL_GRID_FILE)}  # fmt: skip


def intra_patient_split(trainval: dict, test: dict, seed: int) -> dict:
    """DS1 and DS2 beats pooled, each class's beats assigned at random to train, val and test.

    Each part keeps the number of beats of each class it has under the inter-patient protocol, so only
    the separation of patients changes. The same seed gives the same split.
    """
    pooled = concat_parts([trainval, test])
    original, y = np.asarray(pooled["split"]), np.asarray(pooled["y"])
    rng = np.random.default_rng(seed)
    split = np.full(len(y), "test", dtype="<U5")
    for i in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(y == i))
        n_train = int(((original == "train") & (y == i)).sum())
        n_val = int(((original == "val") & (y == i)).sum())
        split[idx[:n_train]], split[idx[n_train : n_train + n_val]] = "train", "val"
    return {**pooled, "split": split}


def describe_split(pooled: dict, seed: int) -> dict:
    """Beats per class and part, and how many records have beats in both training and test (the leak)."""
    split, y, record = np.asarray(pooled["split"]), np.asarray(pooled["y"]), np.asarray(pooled["record"])
    shared = set(record[split == "train"].tolist()) & set(record[split == "test"].tolist())
    return {
        "name": "intra_patient_random", "seed": seed, "records": len(set(record.tolist())), "records_in_train_and_test": len(shared),
        "counts": {part: {c: int(((split == part) & (y == i)).sum()) for i, c in enumerate(CLASSES)} for part in ("train", "val", "test")},
    }  # fmt: skip


def make_source(frozen: DataSource, test: dict, seed: int) -> tuple[DataSource, dict, dict]:
    """The intra-patient training and validation beats as a DataSource, its test beats, and the split's description."""
    pooled = intra_patient_split(frozen.arrays, test, seed)
    held_out = np.asarray(pooled["split"]) == "test"
    trainval = {k: np.asarray(v)[~held_out] for k, v in pooled.items()}
    held = {k: np.asarray(v)[held_out] for k, v in pooled.items()}
    info = describe_split(pooled, seed)
    source = {"source": f"intra-patient random split (seed {seed}) of the frozen DS1 and DS2 beats", "representation": frozen.spec.name,
              "content_sha256": content_sha256(trainval), "frozen": False}  # fmt: skip
    return DataSource(trainval, frozen.spec, source), held, info


def test_arrays(spec: RepresentationSpec, purpose: str) -> dict:
    """The DS2 beats of the frozen dataset, read as an audit access and checked against the frozen hash.

    From ``data/processed/v1`` where it exists, otherwise rebuilt from the raw DS2 records (Kaggle).
    """
    require_test_access("audit")
    if not final_eval_done():
        raise PreconditionError("The leakage audit reads DS2 only after the final evaluation")
    expected = frozen_test_sha256(workflow.manifest()[PHASE1_GATE]["files"])
    arrays = load_test_for_audit("v1", f"{EXPERIMENT}: {purpose}")
    if arrays is None:
        arrays, _ = build_arrays(sorted(DS2), spec, {r: "test" for r in DS2}, access="audit")
        log_test_access("audit", f"{EXPERIMENT}: {purpose} (rebuilt from the raw DS2 records)")
    if content_sha256(arrays) != expected:
        raise PreconditionError("The DS2 beats do not match the frozen test set (results/manifest.json)")
    return arrays


# --------------------------------------------------------------------------------------------------
# generators: the frozen variants retrained on the intra-patient training beats
# --------------------------------------------------------------------------------------------------


def generator_spec(generator_id: str, configs: Path | None = None) -> dict:
    """The variant config, setting and full budget behind a frozen bank id, e.g. ``phase3_cvae-B.beta0.25``."""
    experiment, rest = generator_id.split("-", 1)
    letter, _, setting = rest.partition(".")
    (path,) = sorted(((configs or paths.configs_dir()) / experiment).glob(f"{letter}_*.yaml"))
    cfg = gr.load_config(path)
    setting = setting or gr.DEFAULT_SETTING
    generator = dict(gr.settings_grid(cfg))[setting]
    return {"generator_id": generator_id, "family": cfg["family"], "setting": setting, "settings": gr.generator_settings(cfg, generator),
            "steps": int(cfg["training"]["final_steps"]), "log_every": int(cfg["training"]["log_every"])}  # fmt: skip


def audit_bank_dir(generator_id: str) -> Path:
    return paths.repo_root() / "data" / "synthetic" / "audit" / generator_id


def train_generator(condition: str, spec: dict, seed: int, data: DataSource, device: torch.device, steps: int | None = None) -> dict:
    """Train one frozen generator variant on the intra-patient training beats, export its bank, log the run."""
    folder = audit_bank_dir(spec["generator_id"])
    path = folder / f"seed{seed}.npz"
    if path.is_file():
        return {"condition": condition, "seed": seed, "skipped": "bank exists"}
    steps = int(steps or spec["steps"])
    train = training_beats(data.arrays)
    generator = build_generator(spec["family"], spec["settings"], data.spec.n_samples, seed)
    fit = generator.fit(train, steps, seed, device, fractions=(1.0,), log_every=spec["log_every"])
    run_id = runs.make_run_id(EXPERIMENT, condition.replace("_", ""), seed, tag="generator")
    entry = None
    if not fit.diverged:
        bank = sample_bank(generator, train, bank_counts(train["y"]), seed=seed, source=f"audit/{spec['generator_id']}")
        folder.mkdir(parents=True, exist_ok=True)
        save_npz_deterministic(path, bank)
        entry = {"file": path.name, "sha256": sha256_file(path), "content_sha256": content_sha256(bank), "run_id": run_id, "steps": fit.steps_done}
        meta_path = folder / "bank.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"generator_id": f"audit/{spec['generator_id']}", "seeds": {}}
        meta["seeds"][str(seed)] = {**entry, "trained_on": data.info["source"]}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    metrics = {"selection_metric": "", "selection_value": float("nan"), "diverged": fit.diverged, "divergence": fit.divergence, "steps_done": fit.steps_done}
    config = {"experiment": EXPERIMENT, "condition": condition, **{k: spec[k] for k in ("generator_id", "family", "setting", "settings")},
              "steps": steps, "seed": seed, "dataset": data.info, "bank": entry}  # fmt: skip
    env = {**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}
    group = (*RUN_ROOT, EXPERIMENT, "generators")
    runs.write_run(run_id, config, env, fit.history, metrics, {}, generator.state_dict(), group=group)
    runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, status="diverged" if fit.diverged else "done", group=group))
    return {"condition": condition, "seed": seed, "run_id": run_id, "diverged": fit.diverged, "bank": entry}


def load_audit_bank(generator_id: str, seed: int) -> tuple[dict, str]:
    """An audit bank's arrays and SHA-256, after checking the file against its ``bank.json``."""
    folder = audit_bank_dir(generator_id)
    path, meta_path = folder / f"seed{seed}.npz", folder / "bank.json"
    entry = (json.loads(meta_path.read_text())["seeds"] if meta_path.is_file() else {}).get(str(seed))
    if entry is None or not path.is_file():
        raise PreconditionError(f"Audit bank {generator_id} seed {seed} is missing: run --part generators first")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise PreconditionError(f"{path} changed since it was exported")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}, actual


# --------------------------------------------------------------------------------------------------
# classifiers, evaluated on the intra-patient test beats
# --------------------------------------------------------------------------------------------------


def audit_grid(cfg: dict, ctx: dict) -> dict:
    conditions: dict = {"none": {}}
    for condition, generator_id in ctx["generators"].items():
        strength = ctx["strengths"][condition]
        conditions[condition] = {"bank": f"audit/{generator_id}", **({"strengths": [strength]} if strength is not None else {})}
    return {"experiment": EXPERIMENT, "name": GRID_NAME, "policy": ctx["policy"], "seeds": list(cfg["seeds"]),
            "architectures": {cfg["architecture"]: "phase2_gate"}, "conditions": conditions, "file": cfg["file"]}  # fmt: skip


def evaluate_on_test(run_id: str, part: dict, rr: bool) -> dict:
    """The run's classifier on the intra-patient test beats; probabilities and metrics are kept in its folder."""
    model = load_run_classifier(run_id, rr)
    side = torch.from_numpy(part["side"]) if rr else None
    probs = predict_proba(model, torch.from_numpy(part["X"]), side, torch.device("cpu"))
    m = M.evaluate(probs, part["y"])
    run_dir = runs.find_run_dir(run_id)
    np.savez_compressed(run_dir / "preds_audit.npz", probs=probs, y=part["y"], record=part["record"], r_idx=part["r_idx"])
    (run_dir / "metrics_audit.json").write_text(json.dumps(m, indent=2) + "\n")
    return {"rare_macro_f1": m["rare_macro_f1"], "macro_f1": m["macro_f1"], "n_recall": m["n_recall"], "n_beats": m["n_beats"],
            "f1": {c: m["per_class"][c]["f1"] for c in CLASSES}}  # fmt: skip


def run_classifiers(cfg: dict, ctx: dict, data: DataSource, test: dict, split: Split, device: torch.device) -> list[dict]:
    """Train and evaluate every audit cell not done yet (cells resume like the grid's); return the ledger rows."""
    grid = audit_grid(cfg, ctx)
    ledger = runner.read_ledger(grid)
    part = split_part(test, "test", data.spec.rr_features)
    rows = []
    for condition in grid["conditions"]:
        strength = ctx["strengths"].get(condition)
        for seed in cfg["seeds"]:
            cell = runner.Cell(condition, strength, cfg["architecture"], int(seed))
            bank, bank_sha, bank_id = None, None, None
            if condition in GENERATIVE:
                bank, bank_sha = load_audit_bank(ctx["generators"][condition], int(seed))
                bank_id = grid["conditions"][condition]["bank"]
            sha = runner.cell_sha(cell, grid, ctx["classifier"], data, bank_sha)
            if runner.is_done(ledger.get(sha)):
                rows.append(ledger[sha])
                continue
            row = runner.run_cell(cell, grid, split, data, ctx["classifier"], device, sha, bank, bank_id, [], run_root=RUN_ROOT)
            row["audit_test"] = evaluate_on_test(row["run_id"], part, data.spec.rr_features)
            runner.write_cell(grid, row)
            ledger[sha] = row
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------


def report(cfg: dict) -> dict:
    """Intra-patient (this experiment) beside inter-patient (the final evaluation) results; written to summary.json."""
    out = paths.results_for(EXPERIMENT)
    rows = [r for r in runner.read_ledger({"experiment": EXPERIMENT, "name": GRID_NAME}).values() if r.get("audit_test")]
    final_path = paths.results_for(FINAL_EVAL) / "summary.json"
    final = json.loads(final_path.read_text()) if final_path.is_file() else None
    arch = cfg["architecture"]
    by_seed = {(r["condition"], r["seed"]): r for r in rows}
    summary: dict = {"experiment": EXPERIMENT, "architecture": arch, "metric": "rare_macro_f1",
                     "note": "Never used for any decision (PLAN Phase 4). Intra-patient: mean and SD over seeds; inter-patient: the final evaluation, 95% record-level CI.",
                     "split": json.loads((out / "split.json").read_text()) if (out / "split.json").is_file() else None,
                     "conditions": {}, "gains": {}, "run_ids": sorted(r["run_id"] for r in rows)}  # fmt: skip
    for condition in cfg["conditions"]:
        mine = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["seed"])
        values = [r["audit_test"]["rare_macro_f1"] for r in mine]
        intra = None
        if values:
            mean, sd = mean_sd(values)
            intra = {"mean": mean, "sd": sd, "per_seed": values, "seeds": [r["seed"] for r in mine], "strength": mine[0]["strength"],
                     "run_ids": [r["run_id"] for r in mine]}  # fmt: skip
        entry = (final or {}).get("conditions", {}).get(f"{arch}/{condition}")
        inter = {**{k: entry["rare_macro_f1"][k] for k in ("point", "ci_low", "ci_high")}, "strength": entry["strength"]} if entry else None
        summary["conditions"][condition] = {"intra": intra, "inter": inter}
        if condition == "none":
            continue
        paired = [by_seed[(condition, s)]["audit_test"]["rare_macro_f1"] - by_seed[("none", s)]["audit_test"]["rare_macro_f1"]
                  for s in cfg["seeds"] if (condition, s) in by_seed and ("none", s) in by_seed]  # fmt: skip
        comparison = ((final or {}).get("families", {}).get("confirmatory", {})).get(f"{arch}: {condition} vs none")
        summary["gains"][condition] = {
            "intra": dict(zip(("mean", "sd"), mean_sd(paired)), per_seed=paired) if paired else None,
            "inter": {k: comparison[k] for k in ("point", "ci_low", "ci_high")} if comparison else None,
        }
    none = summary["conditions"].get("none", {})
    if none.get("intra") and none.get("inter"):
        summary["inflation_none"] = none["intra"]["mean"] - none["inter"]["point"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------------------------------
# smoke run and command line
# --------------------------------------------------------------------------------------------------


def smoke(cfg: dict) -> dict:
    """Generators, classifiers and the report on fake records in a temporary root: no DS2 and no gate is read."""
    configs = paths.source_root() / "configs"  # the committed configs, whatever the root
    specs = {c: generator_spec(f"{experiment}-A", configs) for c, experiment in GENERATIVE.items() if c in cfg["conditions"]}
    small = {**cfg, "seeds": cfg["seeds"][:1]}
    with smoke_data() as (split, frozen):
        val = np.asarray(frozen.arrays["split"]) == "val"
        test = {k: np.asarray(v)[val] for k, v in frozen.arrays.items()}  # fake: the validation beats stand in for DS2
        test["split"] = np.full(int(val.sum()), "test", dtype="<U5")
        data, held, info = make_source(frozen, test, cfg["split_seed"])
        ctx = {"policy": {"kind": "fraction_of_n"}, "strengths": {c: 0.25 for c in specs}, "classifier": SMOKE_CLASSIFIER,
               "generators": {c: s["generator_id"] for c, s in specs.items()}}  # fmt: skip
        device = torch.device("cpu")
        generators = [train_generator(c, specs[c], seed, data, device, steps=4) for c in specs for seed in small["seeds"]]
        rows = run_classifiers(small, ctx, data, held, split, device)
        again = run_classifiers(small, ctx, data, held, split, device)
        out = paths.results_for(EXPERIMENT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "split.json").write_text(json.dumps(info, indent=2) + "\n")
        summary = report(small)
        audit_runs = [p.name for p in (paths.runs_dir() / "audit").rglob(f"{EXPERIMENT}-*") if p.is_dir()]
        return {"generators": generators, "cells": len(rows), "skipped_on_rerun": sum(a["run_id"] == b["run_id"] for a, b in zip(rows, again)),
                "runs_under_audit": len(audit_runs), "conditions": sorted(summary["conditions"]), "split": info,
                "note": "fake data; the numbers mean nothing"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["all", "generators", "classifiers", "report"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan", action="store_true", help="what would run; reads no test data")
    parser.add_argument("--smoke", action="store_true", help="end to end on fake data in a temporary root")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2, default=str))
        return 0
    try:
        ctx = frozen_context(cfg)
        if args.plan:
            cells = [(c, ctx["strengths"].get(c), s) for c in ("none", *ctx["generators"]) for s in cfg["seeds"]]
            print(json.dumps({"context": ctx, "generators": {c: generator_spec(g) for c, g in ctx["generators"].items()}, "cells": cells}, indent=2, default=str))
            return 0
        if args.part in ("all", "generators", "classifiers"):
            split = check_preconditions()
            frozen = load_data(split, preview=False)
            data, test, info = make_source(frozen, test_arrays(frozen.spec, f"--part {args.part}"), cfg["split_seed"])
            device = resolve_device(args.device)
            out = paths.results_for(EXPERIMENT)
            out.mkdir(parents=True, exist_ok=True)
            (out / "split.json").write_text(json.dumps(info, indent=2) + "\n")
            if args.part in ("all", "generators"):
                for condition, generator_id in ctx["generators"].items():
                    spec = generator_spec(generator_id)
                    for seed in cfg["seeds"]:
                        print(json.dumps(train_generator(condition, spec, int(seed), data, device), default=str))
            if args.part in ("all", "classifiers"):
                audit_split = Split(name="intra_patient_random", train=(), val=(), selection_metric=split.selection_metric)
                rows = run_classifiers(cfg, ctx, data, test, audit_split, device)
                print(f"{len(rows)} audit cells done")
        if args.part in ("all", "report"):
            summary = report(cfg)
            for condition, c in summary["conditions"].items():
                intra = f"{c['intra']['mean']:.3f} ± {c['intra']['sd']:.3f}" if c["intra"] else "not run"
                inter = f"{c['inter']['point']:.3f}" if c["inter"] else "missing"
                print(f"  {condition:8s} intra-patient {intra}   inter-patient {inter}")
            print(f"written: results/{EXPERIMENT}/summary.json")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
