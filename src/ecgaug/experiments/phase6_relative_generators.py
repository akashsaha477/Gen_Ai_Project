"""phase6_relative_generators: the generators trained on what the P7 cnn sees, from P11's patients (PREREGISTRATION §10 v11, P12; post hoc).

    python -m ecgaug.experiments.phase6_relative_generators --config configs/phase6_relative_generators/relative_generators.yaml --part train --family cvae [--device cuda]
    python -m ecgaug.experiments.phase6_relative_generators --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_relative_generators --config ... --part test      # one posthoc access

P7 and P11 augmented with banks of the phase 3 generators, which learned ordinary beats of DS1's training records; their
beats were made patient-relative afterwards by subtracting the template of the real beat whose RR values they carry. P12
trains the same generators (the variants chosen at the phase 3a gate, their settings and step budget unchanged) on P11's
training beats in the form the cnn sees them (dataset v3: the beat minus the patient's template), divided by one scale (the
SD of the training beats) so that they work at the scale they were tuned for. Banks come from the final checkpoint,
multiplied back by the scale and not z-scored per beat. The P7 cnn is trained on P11's data with each bank at the phase 4
strengths, unscreened (as P9), and the strength is chosen on P11's validation by the phase 4 gate rule. ``--part train``
runs one family end to end (generators, banks, its grid cells and the nearest-patient check), so the two families can run
in two sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import augment, paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.records import CLASS_INDEX, DS2
from ecgaug.eval import metrics as M
from ecgaug.experiments import generator_runs as gr
from ecgaug.experiments import phase6_augmentation_mechanism as mech
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators import GENERATED_CLASSES, build_generator
from ecgaug.generators import bank as banks
from ecgaug.generators.base import BeatGenerator, training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_RELATIVE_GENERATORS
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_RELATIVE_GENERATORS
CONFIG_KEYS = {"experiment", "more_patients_config", "generators", "mechanism_beats"}
FAMILIES = ("cvae", "wgan_gp")
GRID_NAME = "full_grid"
MECHANISM_CLASSES = ("S", "F")
INCART_PATIENT_OFFSET = 1000  # INCART patients become units 1001-1032; no MIT-BIH (100-234) or SVDB (800-894) record is that high


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if set(cfg["generators"]) != set(FAMILIES):
        raise ValueError(f"{path}: generators must name exactly {FAMILIES}")
    root = paths.repo_root()
    gen = {family: gr.load_config(root / p) for family, p in cfg["generators"].items()}
    for family, g in gen.items():
        if g["family"] != family:
            raise ValueError(f"{cfg['generators'][family]} trains {g['family']}, not {family}")
    return {**cfg, "file": str(path), "mp": mp.load_config(root / cfg["more_patients_config"]), "gen": gen}


# --------------------------------------------------------------------------------------------------
# generators and banks (P11's training beats only)
# --------------------------------------------------------------------------------------------------


def scale_of(x: np.ndarray) -> float:
    """The generators' input scale: the SD of every sample of the training beats (no labels used)."""
    return float(np.std(np.asarray(x, dtype=np.float64)))


def strengths() -> list[float]:
    """The phase 4 strengths, as the full grid file holds them."""
    full = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    return [float(s) for s in full["conditions"]["cvae"]["strengths"]]


def bank_counts(train_y: np.ndarray, largest: float) -> dict[str, int]:
    """Beats of each class the unscreened policy adds at the largest strength: the most any cell takes from a bank."""
    return augment.fraction_of_n_targets(augment.class_counts(train_y), largest)


def generator_id(cfg: dict, family: str) -> str:
    """E.g. ``phase6_relative_generators/phase3_cvae-D`` (a folder under data/synthetic/)."""
    g = cfg["gen"][family]
    return f"{EXPERIMENT}/{g['experiment']}-{g['variant']}"


def sample_relative_bank(generator: BeatGenerator, train: dict, counts: dict[str, int], seed: int, scale: float, source: str) -> dict:
    """``counts[c]`` beats of each class, multiplied back by ``scale`` and NOT z-scored per beat (that would undo the
    patient-relative form); RR features copied from real training beats of the class, as in phase 3."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in GENERATED_CLASSES:
        n = int(counts.get(cls, 0))
        if n <= 0:
            continue
        x = (generator.sample(cls, n, seed=seed * 100 + CLASS_INDEX[cls]).astype(np.float64) * scale).astype(np.float32)
        parts.append(banks.synthetic_part(x, cls, banks.rr_from_training(train, cls, n, rng), source))
    return banks.concat_parts(parts)


def generators_path(family: str) -> Path:
    return paths.results_for(EXPERIMENT) / f"{family}_generators.json"


def generators_record(family: str) -> dict:
    path = generators_path(family)
    if not path.is_file():
        raise PreconditionError(f"No {family} generators yet: --part train --family {family}")
    return json.loads(path.read_text())


def _write_log(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_generators(cfg: dict, family: str, data: DataSource, device: torch.device) -> dict:
    """Train the family's generator for every final seed on P11's training beats and export one bank per seed."""
    g = cfg["gen"][family]
    arrays = data.arrays
    train = training_beats(arrays)
    val_records = sorted(set(np.asarray(arrays["record"])[np.asarray(arrays["split"]) == "val"].tolist()))
    scale = scale_of(train["X"])
    part = {**train, "X": (np.asarray(train["X"], dtype=np.float64) / scale).astype(np.float32)}
    settings = gr.generator_settings(g, g["generator"])
    steps = int(g["training"]["final_steps"])
    counts = bank_counts(train["y"], max(strengths()))
    gid = generator_id(cfg, family)
    record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v11", "family": family, "generator_config": cfg["generators"][family],
              "variant": g["variant"], "generator_id": gid, "scale": scale, "steps": steps, "bank_counts": counts,
              "data_sha256": data.info["content_sha256"], "device": str(device), "seeds": {}}  # fmt: skip
    for seed in g["final_seeds"]:
        run_id = runs.make_run_id(EXPERIMENT, "generator", int(seed), tag=family)  # e.g. phase6_relative_generators-generator.cvae-s0-...
        run_dir = paths.runs_dir() / EXPERIMENT / "generators" / family / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "family": family, "settings": settings, "steps": steps,
                                                             "seed": int(seed), "scale": scale, "data_sha256": data.info["content_sha256"]}))  # fmt: skip
        generator = build_generator(g["family"], settings, data.spec.n_samples, int(seed))
        fit = generator.fit(part, steps, int(seed), device, val_records=val_records, fractions=(1.0,), log_every=int(g["training"]["log_every"]))
        _write_log(run_dir / "train_log.csv", fit.history or [{"step": 0}])
        (run_dir / "env.json").write_text(json.dumps({**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}, indent=2) + "\n")
        entry = {"run_id": run_id, "steps_done": fit.steps_done, "diverged": fit.diverged, "divergence": fit.divergence,
                 "generator_seconds": fit.generator_seconds, "n_parameters": generator.n_parameters()}  # fmt: skip
        if not fit.diverged:
            checkpoint = run_dir / "checkpoints" / "final.pt"
            torch.save(generator.state_dict(), checkpoint)
            bank = sample_relative_bank(generator, train, counts, int(seed), scale, source=gid)
            exported = banks.export_bank(bank, gid, int(seed), {"run_id": run_id, "scale": scale, "step": fit.steps_done,
                                                                 "checkpoint_sha256": banks.sha256_file(checkpoint), "dataset": data.info})  # fmt: skip
            entry.update(checkpoint_sha256=exported["checkpoint_sha256"], bank_sha256=exported["sha256"], bank_counts=exported["counts"])
        record["seeds"][str(seed)] = entry
        (run_dir / "metrics_val.json").write_text(json.dumps(entry, indent=2, default=float) + "\n")  # the generator's record (ingest needs the file)
    meta = banks.bank_dir(gid) / banks.BANK_FILE
    if meta.is_file():  # the bank format note: these beats are patient-relative, not z-scored per beat
        m = json.loads(meta.read_text())
        m["normalization"] = "dataset v3 (beat minus the patient's template); generator trained on beats / scale, samples multiplied back by scale"
        meta.write_text(json.dumps(m, indent=2) + "\n")
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    generators_path(family).write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def load_relative_bank(cfg: dict, family: str, seed: int) -> tuple[dict, str]:
    entry = generators_record(family)["seeds"].get(str(seed))
    if entry is None or "bank_sha256" not in entry:
        raise PreconditionError(f"No {family} bank for seed {seed} (did its generator diverge?)")
    return banks.load_bank(generator_id(cfg, family), seed, expected_sha256=entry["bank_sha256"]), entry["bank_sha256"]


# --------------------------------------------------------------------------------------------------
# the grid and the selection (validation only)
# --------------------------------------------------------------------------------------------------


def relative_grid(cfg: dict) -> dict:
    """P11's grid (the P7 cnn, its settings and seeds) with the P12 banks at the phase 4 strengths, unscreened."""
    base = pt.v3_grid(cfg["mp"]["pt"])
    conditions = {family: {"bank": generator_id(cfg, family), "strengths": strengths()} for family in FAMILIES}
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "architectures": {"cnn": base["architectures"]["cnn"]},
            "policy": {"kind": "fraction_of_n"}, "conditions": conditions, "file": cfg["file"]}  # fmt: skip


def run_grid(cfg: dict, data: DataSource, device: torch.device, family: str) -> dict:
    """The grid cells of one family (the whole grid is the one object, so cell hashes never depend on the session)."""
    split = check_preconditions()
    grid = relative_grid(cfg)
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        if cell.condition != family:
            continue
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha = load_relative_bank(cfg, family, cell.seed)
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, grid["conditions"][family]["bank"], [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def select(cfg: dict) -> dict:
    grid = relative_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v11 (the phase 4 gate rule)", "chosen": chosen, "git_sha": runs.git_sha()}
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


def patient_units(records: np.ndarray, incart: dict[int, int] | None = None) -> np.ndarray:
    """One unit per patient: INCART records (numbered below 100) map to their patient (``incart``, by default from the
    headers); MIT-BIH and SVDB records stand for their patient (as in P11's split)."""
    if incart is None:
        from ecgaug.experiments import phase6_external_validation as ext

        incart = ext.patient_of_record("incartdb")
    r = np.asarray(records).astype(np.int64)
    return np.array([INCART_PATIENT_OFFSET + incart[int(v)] if v < 100 else int(v) for v in r], dtype=np.int64)


def mechanism(cfg: dict, family: str, data: DataSource) -> dict:
    """For S and F: the patient of the nearest real training beat (Euclidean, v3 form) of synthetic beats from the P12 bank
    and from the bank P11 used (seed 0, the first ``mechanism_beats`` of the class), beside the real beats' own shares."""
    train = training_beats(data.arrays)
    units = patient_units(train["record"])
    p12, _ = load_relative_bank(cfg, family, 0)
    p11, _ = pt.load_v3_bank(family, 0)
    out = {"experiment": EXPERIMENT, "family": family, "unit": "patient", "beats_per_class": int(cfg["mechanism_beats"]), "classes": {}}
    for cls in MECHANISM_CLASSES:
        mask = np.asarray(train["y"]) == CLASS_INDEX[cls]
        reference, owners = np.asarray(train["X"])[mask], units[mask]
        entry = {"real": mech._entry(owners), "real_nearest_other": mech._entry(owners[mech.nearest(reference, reference, exclude_self=True)], own=owners)}
        for name, bank in (("p12", p12), ("p11", p11)):
            idx = np.flatnonzero(np.asarray(bank["y"]) == CLASS_INDEX[cls])[: int(cfg["mechanism_beats"])]
            entry[name] = mech._entry(owners[mech.nearest(reference, np.asarray(bank["X"])[idx])])
        out["classes"][cls] = entry
    (paths.results_for(EXPERIMENT) / f"{family}_mechanism.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def test_report(cfg: dict) -> dict:
    """DS2 in v3 form, one posthoc access. Confirmatory: each P12 generator (at its chosen strength) against the same
    generator in P11 (the banks of the phase 3 generators). Secondary: each P12 generator against P11's cnn without
    augmentation and against P11's random oversampling (copies of real beats). P11's scores are recomputed and checked."""
    from ecgaug import final_eval

    selection, before = load_selection()["chosen"]["cnn"], mp.load_selection()["chosen"]["cnn"]
    half = cfg["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    test = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    y, records = test["y"], test["record"]
    new, old, unique = {}, {}, None
    for family in FAMILIES:
        unique, new[family] = final_eval.per_record_cms(pt._predict(selection[family]["run_ids"], test), y, records)
    for condition in ("none", "random_oversampling", *FAMILIES):
        _, old[condition] = final_eval.per_record_cms(pt._predict(before[condition]["run_ids"], test), y, records)
    p11 = json.loads((paths.results_for(mp.EXPERIMENT) / "summary.json").read_text())
    reproduced = float(np.mean(M.rare_macro_f1_from_cm(old["none"].sum(axis=1)))) - p11["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
    if abs(reproduced) > 1e-9:
        raise PreconditionError(f"P11's cnn without augmentation scores {reproduced:+.2e} away from results/{mp.EXPERIMENT}/summary.json")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v11", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED,
               "conditions": {f"cnn/{f}": ref.condition_summary(cm, unique) | {"strength": selection[f]["strength"]} for f, cm in new.items()},
               "families": {"confirmatory": mp._family({f"cnn: {f}, P12 vs P11 generators": (new[f], old[f]) for f in FAMILIES}),
                            "secondary": mp._family({**{f"cnn: {f} (P12) vs none (P11)": (new[f], old["none"]) for f in FAMILIES},
                                                     **{f"cnn: {f} (P12) vs random oversampling (P11)": (new[f], old["random_oversampling"]) for f in FAMILIES}})},
               "p11_none_reproduction_difference": reproduced, "run_ids": [r for f in FAMILIES for r in selection[f]["run_ids"]],
               "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["train", "select", "test"], required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "train":
            if args.family is None:
                raise PreconditionError("--part train needs --family cvae or --family wgan_gp")
            data, _ = mp.build(cfg["mp"])  # P11's data, hash-checked against its data.json
            device = resolve_device(args.device)
            record = train_generators(cfg, args.family, data, device)
            print(json.dumps({s: {k: e[k] for k in ("steps_done", "diverged", "generator_seconds")} for s, e in record["seeds"].items()}, default=float))
            print(json.dumps(run_grid(cfg, data, device, args.family)))
            m = mechanism(cfg, args.family, data)
            for cls, e in m["classes"].items():
                print(cls, {k: (round(v["effective_records"], 1), round(v["top_share"], 2)) for k, v in e.items()})
        elif args.part == "select":
            s = select(cfg)
            print({f: (v["strength"], round(v["mean"], 3)) for f, v in s["chosen"]["cnn"].items()})
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:44s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
