"""Phase 4 experiment phase4_augmentation_policy: how many synthetic beats to add, and which (PLAN Phase 4).

Usage:
    python -m ecgaug.experiments.phase4_augmentation_policy --config configs/phase4_augmentation_policy/B_fraction_of_n.yaml
    python -m ecgaug.experiments.phase4_augmentation_policy --config ... --smoke
    add --preview [--bank cvae=phase3_cvae-A --bank wgan_gp=phase3_wgan_gp-B] to run before the gates

| variant | policy |
|---|---|
| A | each minority class up to the training N count |
| B | each minority class up to a fixed fraction of N |
| C | a fixed multiple of each minority class's real count added |
| D | as B, quality-screened: of twice as many bank beats, keep those closest to real beats by DTW |

The pilot (Days 16-17) trains the ``cnn`` with seed 0 once per frozen bank (cVAE and WGAN-GP) at the
variant's pilot strength, plus one reference run without augmentation that every variant shares. A
variant's score is the mean over the two banks of the validation selection metric. At the phase 4a
gate, variants that lower validation N recall by more than 0.01 against the reference are out; the
best of the rest wins, and ties within the 0.02 margin go to the simpler policy. The winner's three
pre-registered strengths then define the full grid. Runs go through the grid runner, so they are
logged and resumable like grid cells.

Official runs need the phase 1, phase 2 and phase 3 gates. The test set is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.generators.bank import BankError
from ecgaug.grid import runner as grid_runner
from ecgaug.names import PHASE2_GATE, PHASE3_GATE, PHASE4_AUGMENTATION_POLICY
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE4_AUGMENTATION_POLICY
VARIANT_KEYS = {"base", "variant", "description", "policy", "pilot_strength", "grid_strengths", "complexity"}
SHARED_KEYS = {"experiment", "classifier", "seed", "banks", "max_n_recall_drop", "preview_classifier"}
REFERENCE_GRID = "reference_none"


def load_config(path: Path) -> dict:
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared settings file, not in a variant")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    if set(shared) != SHARED_KEYS:
        raise ValueError(f"{path.parent / variant['base']}: expected exactly the keys {sorted(SHARED_KEYS)}")
    return {**shared, **{k: variant[k] for k in VARIANT_KEYS - {"base"}}, "variant": str(variant["variant"]), "config_name": path.stem}


def classifier_spec(cfg: dict, preview: bool) -> tuple[object, str | None]:
    """'phase2_gate' (the grid runner reads the cnn from the manifest), or the preview stand-in."""
    if (workflow.manifest().get(PHASE2_GATE) or {}).get(cfg["classifier"]):
        return "phase2_gate", None
    if not preview:
        raise PreconditionError("Phase 2 is not finished: the phase 2 gate has not chosen the cnn. Finish phase 2 first, or add --preview.")
    return cfg["preview_classifier"], "the phase 2 gate has not chosen the cnn: the classifier is the phase1_input_representation probe CNN"


def pilot_grids(cfg: dict, classifier: object, banks: dict[str, str]) -> tuple[dict, dict]:
    """The variant's pilot as a grid (one cell per bank) and the shared reference grid (no augmentation)."""
    base = {"experiment": EXPERIMENT, "seeds": [int(cfg["seed"])], "architectures": {cfg["classifier"]: classifier}}
    strengths = [] if cfg["pilot_strength"] is None else [float(cfg["pilot_strength"])]
    pilot = {**base, "name": cfg["config_name"], "variant": cfg["variant"], "policy": cfg["policy"], "file": f"configs/{EXPERIMENT}/{cfg['config_name']}.yaml",
             "conditions": {family: {"bank": bank, "strengths": strengths} for family, bank in banks.items()}}  # fmt: skip
    reference = {**base, "name": REFERENCE_GRID, "policy": {"kind": "fraction_of_n"}, "file": "reference", "conditions": {"none": {}}}
    return pilot, reference


def run_variant(cfg: dict, split, data: DataSource, device_name: str | None, preview: bool, reasons: list[str],
                banks: dict[str, str] | None = None) -> dict:  # fmt: skip
    """Run the reference (once) and the variant's pilot cells; write ``results/<experiment>/<letter>.json``."""
    device = resolve_device(device_name or "auto")
    classifier, why = classifier_spec(cfg, preview)
    reasons = list(reasons) + ([why] if why else [])
    banks = banks or dict(cfg["banks"])
    pilot, reference = pilot_grids(cfg, classifier, banks)
    grid_runner.run_grid(reference, split, data, device, preview, provisional=reasons)
    grid_runner.run_grid(pilot, split, data, device, preview, provisional=reasons)
    arch = cfg["classifier"]
    ref = grid_runner.summarize(reference)[("none", None, arch)]
    summary = grid_runner.summarize(pilot)
    strength = None if cfg["policy"]["kind"] == "balance_to_n" else float(cfg["pilot_strength"])
    per_bank = {}
    for family in banks:
        cell = summary[(family, strength, arch)]
        per_bank[family] = {"bank": cell["bank"], "selection_value": cell["mean"], "tie_break_value": cell["tie_break_mean"],
                            "n_recall": cell["n_recall"], "n_recall_drop": ref["n_recall"] - cell["n_recall"], "run_ids": cell["run_ids"]}  # fmt: skip
    mean = float(np.mean([b["selection_value"] for b in per_bank.values()]))
    drop = float(np.mean([b["n_recall_drop"] for b in per_bank.values()]))
    result = {
        "experiment": EXPERIMENT, "variant": cfg["variant"], "description": cfg["description"], "policy": cfg["policy"],
        "pilot_strength": cfg["pilot_strength"], "grid_strengths": cfg["grid_strengths"], "complexity": cfg["complexity"],
        "classifier": cfg["classifier"], "seed": cfg["seed"], "per_bank": per_bank, "mean": mean, "sd": 0.0,
        "tie_break_mean": float(np.mean([b["tie_break_value"] for b in per_bank.values()])), "tie_break_sd": 0.0,
        "reference": {"run_ids": ref["run_ids"], "selection_value": ref["mean"], "n_recall": ref["n_recall"]},
        "constraint": {"max_n_recall_drop": cfg["max_n_recall_drop"], "n_recall_drop": drop, "passed": bool(drop <= cfg["max_n_recall_drop"])},
        "note": "one seed: the tie margin is the 0.02 floor", "provisional": bool(reasons), "provisional_reasons": reasons,
        "split": {"name": split.name, "val": list(split.val)}, "selection_metric": split.selection_metric, "dataset": data.info,
        "git_sha": runs.git_sha(),
    }  # fmt: skip
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def smoke(cfg: dict) -> dict:
    """The variant on fake records with a fake bank (training beats repeated), in a temporary root."""
    from ecgaug import augment
    from ecgaug.generators.bank import bank_counts, export_bank
    from ecgaug.generators.base import training_beats

    small = {**cfg, "preview_classifier": {**cfg["preview_classifier"], "max_epochs": 1}}
    with smoke_data() as (split, data):
        train = training_beats(data.arrays)
        fake = augment.random_oversample(train, bank_counts(train["y"]), seed=0)  # sized like a real bank
        fake["is_synthetic"][:] = True
        for name in ("smoke_cvae", "smoke_wgan_gp"):
            export_bank(fake, name, 0, {"train_only_check": "passed"})
        result = run_variant(small, split, data, "cpu", True, [data.provisional], {"cvae": "smoke_cvae", "wgan_gp": "smoke_wgan_gp"})
        return {"variant": result["variant"], "policy": result["policy"]["kind"], "banks": sorted(result["per_bank"]),
                "run_ids": sorted(r for b in result["per_bank"].values() for r in b["run_ids"]),
                "constraint_evaluated": "passed" in result["constraint"], "note": "fake data; the numbers mean nothing"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase4_augmentation_policy: pilot an augmentation policy with both banks.")
    parser.add_argument("--config", required=True, help="configs/phase4_augmentation_policy/<variant>.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    parser.add_argument("--preview", action="store_true", help="run before the gates; results are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter")
    parser.add_argument("--bank", action="append", default=[], metavar="FAMILY=GENERATOR_ID", help="preview only: a bank to use, e.g. cvae=phase3_cvae-A")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    if (args.representation or args.bank) and not args.preview:
        print("--representation and --bank are for previews only")
        return 2
    try:
        if not args.preview:
            if workflow.phase1_winner() is None:
                raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first, or add --preview.")
            if not workflow.manifest().get(PHASE3_GATE):
                raise PreconditionError("Phase 3 is not finished: the phase 3 gate has not frozen the banks, or add --preview.")
        split = check_preconditions()
        data = load_data(split, args.preview, args.representation)
        banks = dict(item.split("=", 1) for item in args.bank) or None
        result = run_variant(cfg, split, data, args.device, args.preview, [data.provisional] if data.provisional else [], banks)
    except (PreconditionError, BankError) as err:
        print(f"Refusing to run {EXPERIMENT} variant {cfg['variant']}: {err}")
        return 2
    c = result["constraint"]
    print(f"variant {result['variant']} ({result['policy']['kind']}): {result['selection_metric']} {result['mean']:.4f} (mean of both banks); "
          f"N recall drop {c['n_recall_drop']:+.4f} ({'within' if c['passed'] else 'OVER'} the {c['max_n_recall_drop']} limit)")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
