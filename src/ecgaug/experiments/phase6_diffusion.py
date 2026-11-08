"""phase6_diffusion: a class-conditional diffusion model as the generator (PREREGISTRATION §10 v14, P15; post hoc).

    python -m ecgaug.experiments.phase6_diffusion --config configs/phase6_diffusion/diffusion.yaml --part train [--device cuda]
    python -m ecgaug.experiments.phase6_diffusion --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_diffusion --config ... --part test        # one posthoc access

The study's generators are a cVAE and a WGAN-GP; the current standard for synthetic ECG is a diffusion model (SSSD-ECG,
Alcaraz and Strodthoff 2023; conditional diffusion for imbalanced ECG classes, Kwak and Jung 2024). P15 trains a
class-conditional diffusion model (``ecgaug.generators.diffusion``) on P13's training beats in the form the cnn sees them
(P11's patients; beat minus the patient's template, divided by one scale as in P12), with the phase 3 step budget, batch
and seeds; its banks take their RR features from real training beats of the class in P13's form; the P7 cnn is trained on
P13's data with the bank at the phase 4 strengths, and the strength is chosen on validation. Nothing is chosen on DS2.
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
from ecgaug.data.records import CLASS_INDEX, DS2
from ecgaug.eval import metrics as M
from ecgaug.experiments import phase6_augmentation_mechanism as mech
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators import build_generator
from ecgaug.generators import bank as banks
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_DIFFUSION
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_DIFFUSION
FAMILY = "diffusion"
CONFIG_KEYS = {"experiment", "record_rr_config", "generator", "training", "classes", "seeds", "strengths", "mechanism_beats", "rule_threshold"}
GRID_NAME = "full_grid"
BEFORE = ("none", "class_weighted", "random_oversampling", "cvae")  # P13's conditions it is compared with


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if sorted(cfg["strengths"]) != sorted(rg.strengths()):
        raise ValueError(f"{path}: strengths must be the phase 4 strengths {rg.strengths()}")
    return {**cfg, "file": str(path), "rr": rr.load_config(paths.repo_root() / cfg["record_rr_config"])}


def generator_id() -> str:
    return f"{EXPERIMENT}/diffusion"


def settings(cfg: dict) -> dict:
    return {**cfg["training"], **cfg["generator"], "classes": list(cfg["classes"])}


def generators_path() -> Path:
    return paths.results_for(EXPERIMENT) / "diffusion_generators.json"


def generators_record() -> dict:
    path = generators_path()
    if not path.is_file():
        raise PreconditionError("Train the generators first: --part train")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# the generators (P13's training beats only) and their banks
# --------------------------------------------------------------------------------------------------


def train_generators(cfg: dict, data: DataSource, device: torch.device) -> dict:
    """One diffusion model per seed on the training beats (divided by one scale), and one bank per seed."""
    arrays = data.arrays
    train = training_beats(arrays)
    val_records = sorted(set(np.asarray(arrays["record"])[np.asarray(arrays["split"]) == "val"].tolist()))
    scale = rg.scale_of(train["X"])
    part = {**train, "X": (np.asarray(train["X"], dtype=np.float64) / scale).astype(np.float32)}
    s = settings(cfg)
    steps = int(cfg["training"]["final_steps"])
    counts = rg.bank_counts(train["y"], max(cfg["strengths"]))
    gid = generator_id()
    record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v14", "family": FAMILY, "settings": s, "generator_id": gid, "scale": scale,
              "steps": steps, "bank_counts": counts, "data_sha256": data.info["content_sha256"], "device": str(device), "seeds": {}}  # fmt: skip
    for seed in cfg["seeds"]:
        run_id = runs.make_run_id(EXPERIMENT, "generator", int(seed), tag=FAMILY)
        run_dir = paths.runs_dir() / EXPERIMENT / "generators" / FAMILY / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "family": FAMILY, "settings": s, "steps": steps,
                                                             "seed": int(seed), "scale": scale, "data_sha256": data.info["content_sha256"]}))  # fmt: skip
        generator = build_generator(FAMILY, s, data.spec.n_samples, int(seed))
        fit = generator.fit(part, steps, int(seed), device, val_records=val_records, fractions=(1.0,), log_every=int(cfg["training"]["log_every"]))
        rg._write_log(run_dir / "train_log.csv", fit.history or [{"step": 0}])
        (run_dir / "env.json").write_text(json.dumps({**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}, indent=2) + "\n")
        entry = {"run_id": run_id, "steps_done": fit.steps_done, "diverged": fit.diverged, "divergence": fit.divergence,
                 "generator_seconds": fit.generator_seconds, "n_parameters": generator.n_parameters()}  # fmt: skip
        if not fit.diverged:
            checkpoint = run_dir / "checkpoints" / "final.pt"
            torch.save(generator.state_dict(), checkpoint)
            bank = rg.sample_relative_bank(generator, train, counts, int(seed), scale, source=gid)  # RR from real training beats of the class
            exported = banks.export_bank(bank, gid, int(seed), {"run_id": run_id, "scale": scale, "step": fit.steps_done,
                                                                 "checkpoint_sha256": banks.sha256_file(checkpoint), "dataset": data.info})  # fmt: skip
            entry.update(checkpoint_sha256=exported["checkpoint_sha256"], bank_sha256=exported["sha256"], bank_counts=exported["counts"])
        record["seeds"][str(seed)] = entry
        (run_dir / "metrics_val.json").write_text(json.dumps(entry, indent=2, default=float) + "\n")
    meta = banks.bank_dir(gid) / banks.BANK_FILE
    if meta.is_file():
        m = json.loads(meta.read_text())
        m["normalization"] = "P13's form (beat minus the patient's template); generator trained on beats / scale, samples multiplied back by scale"
        meta.write_text(json.dumps(m, indent=2) + "\n")
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    generators_path().write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def load_diffusion_bank(seed: int) -> tuple[dict, str]:
    entry = generators_record()["seeds"].get(str(seed))
    if entry is None or "bank_sha256" not in entry:
        raise PreconditionError(f"No diffusion bank for seed {seed} (did its generator diverge?)")
    return banks.load_bank(generator_id(), seed, expected_sha256=entry["bank_sha256"]), entry["bank_sha256"]


# --------------------------------------------------------------------------------------------------
# the grid and the selection (validation only)
# --------------------------------------------------------------------------------------------------


def diffusion_grid(cfg: dict) -> dict:
    """P13's grid (the P7 cnn, settings, seeds, unscreened policy) with the diffusion bank at the phase 4 strengths."""
    base = rr.record_grid(cfg["rr"])
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "conditions": {FAMILY: {"bank": generator_id(), "strengths": list(cfg["strengths"])}},
            "file": cfg["file"]}  # fmt: skip


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    split = check_preconditions()
    grid = diffusion_grid(cfg)
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha = load_diffusion_bank(cell.seed)
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, generator_id(), [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def select(cfg: dict) -> dict:
    grid = diffusion_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v14 (the phase 4 gate rule)", "chosen": chosen, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# whose beats the synthetic beats resemble (training data only; P8's measure)
# --------------------------------------------------------------------------------------------------


def mechanism(cfg: dict, data: DataSource) -> dict:
    """For S and F: the patient of the nearest real training beat of 'mechanism_beats' synthetic beats from the seed-0 bank,
    beside the real beats' own spread and P12's cVAE bank (the generator P13 used)."""
    train = training_beats(data.arrays)
    units = rg.patient_units(train["record"])
    diffusion, _ = load_diffusion_bank(0)
    cvae, _ = rg.load_relative_bank(cfg["rr"]["rg"], "cvae", 0)
    n = int(cfg["mechanism_beats"])
    out = {"experiment": EXPERIMENT, "unit": "patient", "beats_per_class": n, "classes": {}}
    for cls in ("S", "F"):
        mask = np.asarray(train["y"]) == CLASS_INDEX[cls]
        reference, owners = np.asarray(train["X"])[mask], units[mask]
        entry = {"real": mech._entry(owners)}
        for name, bank in (("diffusion", diffusion), ("cvae_p12", cvae)):
            idx = np.flatnonzero(np.asarray(bank["y"]) == CLASS_INDEX[cls])[:n]
            entry[name] = mech._entry(owners[mech.nearest(reference, np.asarray(bank["X"])[idx])])
        out["classes"][cls] = entry
    (paths.results_for(EXPERIMENT) / "mechanism.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Confirmatory: the diffusion model at its chosen strength vs P13's cnn without
    augmentation and vs P13's cVAE. Secondary: vs P13's class weighting and random oversampling. Descriptive: Huang et al.'s
    SVEB stage on top. P13's scores are recomputed and checked first."""
    from ecgaug import final_eval

    new_sel, old_sel = load_selection()["chosen"]["cnn"], rr.load_selection()["chosen"]["cnn"]
    half = cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    v3 = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    test = rr.normalize(v3, rr.mean_rr_by_record(np.asarray(v3["record"]), access="posthoc"))
    y, records = test["y"], test["record"]
    probs = pt._predict(new_sel[FAMILY]["run_ids"], test)
    unique, new = final_eval.per_record_cms(probs, y, records)
    _, new_rule = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], cfg["rule_threshold"]) for p in probs], y, records)
    old, old_rule = {}, {}
    for c in BEFORE:
        p_old = pt._predict(old_sel[c]["run_ids"], test)
        _, old[c] = final_eval.per_record_cms(p_old, y, records)
        _, old_rule[c] = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], cfg["rule_threshold"]) for p in p_old], y, records)
    reproduced = {c: rr._reproduced(old[c], paths.results_for(rr.EXPERIMENT) / "summary.json", f"cnn/{c}") for c in BEFORE}
    off = {c: d for c, d in reproduced.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"P13's scores do not reproduce: {off}")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v14", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": float(cfg["rule_threshold"]),
               "conditions": {f"cnn/{FAMILY}": ref.condition_summary(new, unique) | {"strength": new_sel[FAMILY]["strength"]}},
               "before": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in old.items()},
               "sveb_rule": {f"cnn/{FAMILY}": ref.condition_summary(new_rule, unique), **{f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in old_rule.items()}},
               "families": {"confirmatory": mp._family({f"cnn: diffusion vs {c} (P13)": (new, old[c]) for c in ("none", "cvae")}),
                            "secondary": mp._family({f"cnn: diffusion vs {c} (P13)": (new, old[c]) for c in ("class_weighted", "random_oversampling")})},
               "note": "the SVEB-rule rows are descriptive (no test)",
               "reproduction_difference": reproduced, "run_ids": list(new_sel[FAMILY]["run_ids"]), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["train", "select", "test"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "train":
            data, _, _ = rr.build(cfg["rr"])  # P13's data, hash-checked against its data.json
            device = resolve_device(args.device)
            record = train_generators(cfg, data, device)
            print(json.dumps({s: {k: e[k] for k in ("steps_done", "diverged", "generator_seconds")} for s, e in record["seeds"].items()}, default=float))
            print(json.dumps(run_grid(cfg, data, device)))
            m = mechanism(cfg, data)
            for cls, e in m["classes"].items():
                print(cls, {k: (round(v["effective_records"], 1), round(v["top_share"], 2)) for k, v in e.items()})
        elif args.part == "select":
            s = select(cfg)
            print({c: (v["strength"], round(v["mean"], 3)) for c, v in s["chosen"]["cnn"].items()})
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:40s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
