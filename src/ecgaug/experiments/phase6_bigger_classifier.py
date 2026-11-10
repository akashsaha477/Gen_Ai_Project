"""phase6_bigger_classifier: phase 2's ResNet1D in P18's pipeline (PREREGISTRATION §10 v21, P22; post hoc).

    python -m ecgaug.experiments.phase6_bigger_classifier --config configs/phase6_bigger_classifier/bigger_classifier.yaml --part run --beats-dir DIR [--device cuda]
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_bigger_classifier --config ... --part test      # one posthoc access

Phase 2's ResNet1D (variant B: a stem convolution, four residual stages, global average pooling, the RR side input;
6.7 times the cnn's parameters) lost to the cnn at phase 2 and has not run since. P22 trains it on P13's data with
phase 2's learning rate for it and the budget every phase 2 classifier shared, for no augmentation, the cVAE and the
WGAN-GP (P12's banks in P13's form, bank seed s mod 3), seeds 0-9. ``--part run`` trains the 30 runs, chooses Huang's
stage and P16's veto for each ResNet ensemble on P14's patients by P20's rule and compares it with P18's cnn ensemble for
the same condition on P19's patients; ``--part test`` reads DS2 once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.experiments import ensemble_compare as ec
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_BIGGER_CLASSIFIER
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_BIGGER_CLASSIFIER
RULE = "PREREGISTRATION §10 v21"
CONFIG_KEYS = {"experiment", "seed_ensemble_config", "classifier_variant", "conditions"}
VARIANT = "B"  # phase 2's ResNet1D
ARCHITECTURE = "resnet"
CONDITIONS = ("none", "cvae", "wgan_gp")
SEEDS = tuple(range(10))
GRID_NAME = "grid"
BUDGET_KEYS = ("max_epochs", "patience", "min_epochs", "batch_size", "weight_decay")
PAIRS = {f"{ARCHITECTURE}/{c}": f"{ec.BASE}{c}" for c in CONDITIONS}  # the ResNet's ensemble vs P18's cnn ensemble
WITHIN = {c: (f"{ARCHITECTURE}/{c}", f"{ARCHITECTURE}/none") for c in CONDITIONS if c != "none"}  # augmentation within the ResNet


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if cfg["classifier_variant"] != VARIANT or tuple(cfg["conditions"]) != CONDITIONS:
        raise ValueError(f"{path}: classifier_variant and conditions must be those of PREREGISTRATION v21")
    se_cfg = se.load_config(paths.repo_root() / cfg["seed_ensemble_config"])
    return {**cfg, "file": str(path), "se": se_cfg, "rr": se_cfg["rr"]}


def classifier_settings() -> dict:
    """Phase 2's variant B with the learning rate phase 2 chose for it and the budget it shared with the cnn (the phase 2
    gate's record of the cnn); refuses if B's recorded budget differs from the cnn's."""
    b = json.loads((paths.results_for("phase2_baseline_classifiers") / f"{VARIANT}.json").read_text())
    cnn = (workflow.manifest().get(runner.PHASE2_GATE) or {}).get("cnn")
    if not cnn:
        raise PreconditionError("The phase 2 gate's cnn settings are missing")
    if b["budget_deviations"] or any(b["budget"][k] != cnn[k] for k in BUDGET_KEYS):
        raise PreconditionError("Phase 2's ResNet1D did not run under the cnn's budget")
    return {"variant": VARIANT, "config": "configs/phase2_baseline_classifiers/B_resnet1d.yaml", "model": dict(b["model"]),
            "lr": float(b["chosen_lr"]), **{k: cnn[k] for k in BUDGET_KEYS}, "n_parameters_phase2": int(b["n_parameters"])}  # fmt: skip


def grid(cfg: dict) -> dict:
    """P13's grid (data, policy, carried strengths) with the ResNet as its one architecture and three conditions, seeds 0-9."""
    base = rr.record_grid(cfg["rr"])
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "seeds": list(SEEDS), "architectures": {ARCHITECTURE: classifier_settings()},
            "conditions": {c: base["conditions"][c] for c in CONDITIONS}, "file": cfg["file"]}  # fmt: skip


def run_grid(cfg: dict, device: torch.device) -> dict:
    """The 30 runs on P13's data, banks hash-checked (P18's ``load_bank``); cells already done are skipped."""
    data, before, _ = rr.build(cfg["rr"])
    split = check_preconditions()
    g = grid(cfg)
    classifier = g["architectures"][ARCHITECTURE]
    ledger = runner.read_ledger(g)
    train_before, train_after = training_beats(before.arrays), training_beats(data.arrays)
    loaded: dict[tuple[str, int], tuple[dict, str]] = {}
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(g):
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in se.GENERATIVE:
            key = (cell.condition, se.bank_seed(cell.seed))
            if key not in loaded:
                loaded = {k: v for k, v in loaded.items() if k[0] == cell.condition}
                loaded[key] = se.load_bank(cfg["se"], cell.condition, cell.seed, train_before, train_after)
            bank, bank_sha = loaded[key]
            bank_id = g["conditions"][cell.condition]["bank"]
        sha = runner.cell_sha(cell, g, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, g, split, data, classifier, device, sha, bank, bank_id, [])
        runner.write_cell(g, ledger[sha])
        report["ran"] += 1
    return report


def members(cfg: dict) -> dict:
    """Each condition's ten runs in seed order, with their validation S, V and F F1 (each run's kept epoch)."""
    rows: dict[tuple[str, int], list[dict]] = {}
    for r in runner.read_ledger(grid(cfg)).values():
        if r["status"] == "done":
            rows.setdefault((r["condition"], int(r["seed"])), []).append(r)
    out, validation = {}, {}
    for c in CONDITIONS:
        found = [rows.get((c, s), []) for s in SEEDS]
        if any(len(f) != 1 for f in found):
            raise PreconditionError(f"{c}: expected one done run per seed 0-9")
        name = f"{ARCHITECTURE}/{c}"
        out[name] = [f[0]["run_id"] for f in found]
        validation[name] = {"f1_S": [float(f[0]["selection_value"]) for f in found], "f1_V": [float(f[0]["tie_break_value"]) for f in found],
                            "f1_F": [2 * float(f[0]["rare_macro_f1"]) - float(f[0]["selection_value"]) for f in found]}  # fmt: skip
    return {"members": out, "validation": validation}


def cnn_validation() -> dict | None:
    """P18's cnn runs' validation S, V and F F1 per condition (their stored metrics; None where a run folder is not here,
    e.g. on Kaggle, where P18's runs come as checkpoints only). Descriptive, validation patients only."""
    from ecgaug.experiments import phase6_fresh_patients as fp

    out = {}
    for c in CONDITIONS:
        values: dict[str, list[float]] = {"f1_S": [], "f1_V": [], "f1_F": []}
        for run_id in fp.frozen_members()[c]:
            folder = runs.find_run_dir(run_id)
            if folder is None or not (folder / "metrics_val.json").is_file():
                return None
            m = json.loads((folder / "metrics_val.json").read_text())
            for k in values:
                values[k].append(float(m["per_class"][k.split("_")[1]]["f1"]))
        out[f"{ec.BASE}{c}"] = values
    return out


def _write(name: str, payload: dict) -> Path:
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path


def _read(name: str, hint: str) -> dict:
    path = paths.results_for(EXPERIMENT) / name
    if not path.is_file():
        raise PreconditionError(f"{name} is missing: {hint}")
    return json.loads(path.read_text())


def run(cfg: dict, beats_dir: Path, device: torch.device) -> dict:
    """Train the 30 runs, write ``members.json``, then choose on P14's patients and score P19's."""
    report = run_grid(cfg, device)
    m = members(cfg)
    _write("members.json", {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "classifier": classifier_settings(), **m,
                            "report": report, "git_sha": runs.git_sha()})  # fmt: skip
    choice, summary = ec.run_patients(EXPERIMENT, RULE, m["members"], PAIRS, beats_dir, device, WITHIN)
    _write("choice.json", choice)
    _write("summary.json", summary)
    return {"report": report, "choice": choice, "summary": summary}


def test_report(cfg: dict) -> dict:
    choice = _read("choice.json", "run --part run and ingest it first")
    m = _read("members.json", "run --part run and ingest it first")["members"]
    out = ec.ds2_report(EXPERIMENT, RULE, cfg["se"], choice, m, PAIRS, WITHIN)
    _write("test_summary.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=("run", "test"), required=True)
    parser.add_argument("--beats-dir", type=Path, default=rt.default_beats_dir(), help="P20's built beats (run)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.part == "run":
            out = run(cfg, args.beats_dir, resolve_device(args.device))
            s = out["summary"]
            print(json.dumps({"report": out["report"], "chosen": out["choice"]["chosen"],
                              "secondary": {k: {"point": v["point"], "p_holm": v["p_holm"]} for k, v in s["families"]["secondary"].items()}}, indent=2, default=float))  # fmt: skip
        else:
            out = test_report(cfg)
            print(json.dumps({k: {"point": v["point"], "p_holm": v["p_holm"]} for k, v in out["families"]["confirmatory"].items()}, indent=2, default=float))
    except (PreconditionError, LockedTestDataError) as exc:
        print(f"refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
