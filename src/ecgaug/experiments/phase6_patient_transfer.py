"""phase6_patient_transfer: rare-class beats moved onto other patients (PREREGISTRATION §10 v8, P9; post hoc).

    python -m ecgaug.experiments.phase6_patient_transfer --config configs/phase6_patient_transfer/patient_transfer.yaml --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_patient_transfer --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_patient_transfer --config ... --part test   # one posthoc access
    python -m ecgaug.experiments.phase6_patient_transfer --config ... --part external [--database incartdb]

P8 found that the networks learn the shapes of the few training patients and that the generators reproduce those same
patients. Patient transfer writes a rare-class beat as its patient's template plus a deviation and puts the deviation
on the template of another training patient: x' = zscore(t_q + (x - t_p)), q != p, with rr_prev and rr_next scaled by the
ratio of the two patients' median RR (rr_ratio is unchanged). The deviations come from real training beats
(``transfer_real``) or from the frozen cVAE and WGAN-GP banks, each synthetic beat taking the template of the real beat
whose RR values it carries (``transfer_cvae``, ``transfer_wgan_gp``). The target patient is drawn uniformly among the
training records, and the target template is that of a random N beat of the patient. Everything else is v5's dataset
v2 and the phase 4a gate's grid, with the unscreened policy.
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
from ecgaug.data.contract import content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.records import CLASS_INDEX, CLASSES, DS2
from ecgaug.data.representation import zscore_per_beat
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators.bank import load_bank
from ecgaug.grid import runner
from ecgaug.names import PHASE6_PATIENT_TRANSFER
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_PATIENT_TRANSFER
CONFIG_KEYS = {"experiment", "half_window_s", "bank_size_per_class"}
SOURCES = {"transfer_real": None, "transfer_cvae": "cvae", "transfer_wgan_gp": "wgan_gp"}
GENERATIVE_TRANSFER = ("transfer_cvae", "transfer_wgan_gp")
RARE = ("S", "V", "F")
GRID_NAME = "full_grid"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path)}


# --------------------------------------------------------------------------------------------------
# the transfer
# --------------------------------------------------------------------------------------------------


def transfer(deviation: np.ndarray, rr: np.ndarray, source_median: np.ndarray, target_template: np.ndarray, target_median: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beats ``zscore(target_template + deviation)`` and RR features ``(rr_prev, rr_next)`` scaled by the ratio of median RR;
    ``rr`` holds (rr_prev, rr_next, rr_ratio) per row, and the ratio column is returned unchanged."""
    x = zscore_per_beat(np.asarray(target_template, dtype=np.float64) + np.asarray(deviation, dtype=np.float64)).astype(np.float32)
    scale = (np.asarray(target_median, dtype=np.float64) / np.asarray(source_median, dtype=np.float64))[:, None]
    out = np.asarray(rr, dtype=np.float64).copy()
    out[:, :2] = out[:, :2] * scale
    return x, out.astype(np.float32)


def draw_targets(rng: np.random.Generator, source_records: np.ndarray, train_records: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """For each source beat, the index of a training N beat of a different record, the record drawn uniformly."""
    n_mask = np.asarray(train_y) == CLASS_INDEX["N"]
    by_record = {int(r): np.flatnonzero(n_mask & (np.asarray(train_records) == r)) for r in np.unique(train_records)}
    by_record = {r: idx for r, idx in by_record.items() if idx.size}
    records = np.array(sorted(by_record))
    out = np.empty(len(source_records), dtype=np.int64)
    for i, p in enumerate(np.asarray(source_records)):
        choices = records[records != int(p)]
        q = int(rng.choice(choices))
        out[i] = int(rng.choice(by_record[q]))
    return out


def matched_real_beats(bank: dict, v1_train: dict) -> np.ndarray:
    """For each synthetic beat, the index of the real training beat of its class whose v1 RR values it carries (v5's rule)."""
    lookup: dict[tuple, int] = {}
    for i in range(len(v1_train["y"])):
        lookup.setdefault((int(v1_train["y"][i]), *(np.float32(v1_train[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio"))), i)
    out = np.empty(len(bank["y"]), dtype=np.int64)
    for i in range(len(bank["y"])):
        key = (int(bank["y"][i]), *(np.float32(bank[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio")))
        if key not in lookup:
            raise PreconditionError(f"Synthetic beat {i} carries RR values of no real training beat of its class")
        out[i] = lookup[key]
    return out


def build_bank(name: str, seed: int, size: int, v1_train: dict, v2_train: dict, templates: np.ndarray, frozen: dict | None) -> dict:
    """``size`` transferred beats per rare class, in random order, for one condition and seed."""
    rng = np.random.default_rng(1000 + seed)
    median = np.asarray(v2_train["rr_prev"], dtype=np.float64) / np.asarray(v2_train["rr_ratio"], dtype=np.float64)
    parts = []
    for cls in RARE:
        i = CLASS_INDEX[cls]
        if frozen is None:  # real deviations of the class
            pool = np.flatnonzero(np.asarray(v2_train["y"]) == i)
            pick = pool[rng.integers(0, pool.size, size)]
            deviation = np.asarray(v2_train["X"])[pick] - templates[pick]
            source = pick
        else:  # generated deviations: each synthetic beat minus the template of its matched real beat
            mask = np.asarray(frozen["y"]) == i
            x_syn = np.asarray(frozen["X"])[mask]
            match = matched_real_beats({k: np.asarray(v)[mask] for k, v in frozen.items()}, v1_train)
            chosen = rng.integers(0, x_syn.shape[0], size)
            deviation = x_syn[chosen] - templates[match[chosen]]
            source = match[chosen]
        target = draw_targets(rng, np.asarray(v2_train["record"])[source], v2_train["record"], v2_train["y"])
        rr = np.stack([np.asarray(v2_train[f])[source] for f in ("rr_prev", "rr_next", "rr_ratio")], axis=1)
        x, rr_new = transfer(deviation, rr, median[source], templates[target], median[target])
        parts.append({"X": x, "y": np.full(size, i, dtype=np.int8), "record": np.full(size, -1, dtype=np.int16), "split": np.full(size, "train", dtype="<U5"),
                      "r_idx": np.full(size, -1, dtype=np.int32), "rr_prev": rr_new[:, 0], "rr_next": rr_new[:, 1], "rr_ratio": rr_new[:, 2],
                      "is_synthetic": np.ones(size, dtype=bool), "source": np.full(size, f"{EXPERIMENT}-{name}")})  # fmt: skip
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = rng.permutation(len(bank["y"]))
    return {k: v[order] for k, v in bank.items()}


# --------------------------------------------------------------------------------------------------
# data: v2 beats (v5) and the transfer banks (DS1 only)
# --------------------------------------------------------------------------------------------------


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def bank_folder(name: str) -> Path:
    return paths.repo_root() / "data" / "synthetic" / EXPERIMENT / name


def build(cfg: dict) -> tuple[DataSource, dict]:
    """v2 training and validation beats (checked against v5) and the transfer banks, saved and checked against data.json."""
    split = check_preconditions()
    v1 = load_data(split, preview=False)
    v2_arrays = pt._v2_arrays(v1, cfg["half_window_s"])
    sha = content_sha256(v2_arrays)
    if sha != v5.data_record()["trainval_sha256"]:
        raise PreconditionError("The v2 beats do not match results/phase6_rr_baseline/data.json")
    train_mask = (np.asarray(v1.arrays["split"]) == "train") & ~np.asarray(v1.arrays["is_synthetic"])
    templates = pt.mitbih_templates(v1.arrays, v1.spec, cfg["half_window_s"])[train_mask]
    v1_train, v2_train = pt._train(v1.arrays), pt._train(v2_arrays)
    generators = v5.frozen_generators()
    record = data_record()
    banks = {}
    for name, generator in SOURCES.items():
        folder = bank_folder(name)
        folder.mkdir(parents=True, exist_ok=True)
        seeds = {}
        for seed in ("0", "1", "2"):
            frozen = None
            if generator is not None:
                entry = generators[generator]
                frozen = load_bank(entry["generator_id"], int(seed), expected_sha256=entry["banks"][seed]["sha256"])
            bank = build_bank(name, int(seed), int(cfg["bank_size_per_class"]), v1_train, v2_train, templates, frozen)
            save_npz_deterministic(folder / f"seed{seed}.npz", bank)
            seeds[seed] = {"sha256": sha256_file(folder / f"seed{seed}.npz")}
            if record is not None and record["banks"][name]["seeds"][seed]["sha256"] != seeds[seed]["sha256"]:
                raise PreconditionError(f"Transfer bank {name} seed {seed} does not match data.json")
        (folder / "bank.json").write_text(json.dumps({"generator_id": f"{EXPERIMENT}/{name}", "rule": "PREREGISTRATION §10 v8", "seeds": seeds}, indent=2) + "\n")
        banks[name] = {"seeds": seeds, "deviations_from": generator or "real training beats"}
    if record is None:
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v8", "half_window_s": cfg["half_window_s"], "trainval_sha256": sha,
                  "bank_size_per_class": cfg["bank_size_per_class"], "banks": banks, "git_sha": runs.git_sha()}  # fmt: skip
        out = paths.results_for(EXPERIMENT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    info = {"source": "data/processed/v2 (v5), with patient-transfer banks", "representation": v1.spec.name, "content_sha256": sha, "frozen": False, "version": "v2"}
    return DataSource(v2_arrays, v1.spec, info), record


def load_transfer_bank(name: str, seed: int) -> tuple[dict, str]:
    path = bank_folder(name) / f"seed{seed}.npz"
    sha = sha256_file(path)
    if sha != data_record()["banks"][name]["seeds"][str(seed)]["sha256"]:
        raise PreconditionError(f"{path} does not match data.json")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}, sha


# --------------------------------------------------------------------------------------------------
# the grid and the selection (validation only)
# --------------------------------------------------------------------------------------------------


def transfer_grid(cfg: dict) -> dict:
    """The phase 4a gate's grid (architectures, seeds, strengths) with no augmentation and the three transfer conditions, unscreened."""
    full = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    strengths = list(full["conditions"]["cvae"]["strengths"])
    conditions = {"none": {}, **{name: {"bank": f"{EXPERIMENT}/{name}", "strengths": strengths} for name in SOURCES}}
    return {**full, "experiment": EXPERIMENT, "name": GRID_NAME, "policy": {"kind": "fraction_of_n"}, "conditions": conditions, "file": cfg["file"]}


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    split = check_preconditions()
    grid = transfer_grid(cfg)
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in SOURCES:
            bank, bank_sha = load_transfer_bank(cell.condition, cell.seed)
            bank_id = grid["conditions"][cell.condition]["bank"]
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def select(cfg: dict) -> dict:
    grid = transfer_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v8 (the phase 4 gate rule)", "chosen": chosen, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


def families(selection: dict) -> dict[str, list[tuple[str, str, str]]]:
    """Confirmatory: each generative transfer condition vs none, per architecture. Secondary: transfer_real vs none, and the better
    generative transfer condition on validation (seed-mean selection value) vs transfer_real."""
    confirmatory, secondary = [], []
    for arch, conds in selection["chosen"].items():
        confirmatory += [(arch, c, "none") for c in GENERATIVE_TRANSFER]
        best = max(GENERATIVE_TRANSFER, key=lambda c: conds[c]["mean"])
        secondary += [(arch, "transfer_real", "none"), (arch, best, "transfer_real")]
    return {"confirmatory": confirmatory, "secondary": secondary}


def _family(cms: dict, items: list[tuple[str, str, str]], stat) -> dict:
    out = {f"{arch}: {a} vs {b}": paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, a)], cms[(arch, b)], n=ref.N_BOOT, seed=ref.BOOT_SEED) for arch, a, b in items}
    adjusted = holm({k: v["p_two_sided"] for k, v in out.items()})
    for key, value in out.items():
        value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"])
    return out


# --------------------------------------------------------------------------------------------------
# DS2 and the external databases
# --------------------------------------------------------------------------------------------------


def test_report(cfg: dict) -> dict:
    """DS2 with v2 features: each condition, the two families with Holm, and paired differences against v5's conditions (one posthoc access)."""
    from ecgaug import final_eval

    selection = load_selection()
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, cfg["half_window_s"], access="posthoc").items()})
    test = v5.substitute(v1_test, ratios)
    y, records = test["y"], test["record"]
    cms, unique = {}, None
    for arch, conds in selection["chosen"].items():
        for condition, chosen in conds.items():
            unique, cms[(arch, condition)] = final_eval.per_record_cms(pt._predict(chosen["run_ids"], test), y, records)
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": selection["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    fam = families(selection)
    stat = M.rare_macro_f1_from_cm
    out_families = {name: _family(cms, items, stat) for name, items in fam.items()}
    v5_chosen = v5.load_selection()["chosen"]
    v5_summary = json.loads((paths.results_for("phase6_rr_baseline") / "summary.json").read_text())
    versus, reproduced = {}, {}
    for arch in selection["chosen"]:
        for old in ("none", "cvae", "wgan_gp"):
            _, old_cm = final_eval.per_record_cms(pt._predict(v5_chosen[arch][old]["run_ids"], test), y, records)
            if old == "none":
                reproduced[arch] = float(np.mean(stat(old_cm.sum(axis=1)))) - v5_summary["conditions"][f"{arch}/none"]["rare_macro_f1"]["point"]
            for condition in selection["chosen"][arch]:
                versus[f"{arch}/{condition} vs v2 {old}"] = paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, condition)], old_cm, n=ref.N_BOOT, seed=ref.BOOT_SEED)
    run_ids = [r for a in selection["chosen"].values() for c in a.values() for r in c["run_ids"]]
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v8", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "conditions": conditions, "families": out_families, "versus_v2": versus,
               "v2_none_reproduction_difference": reproduced, "run_ids": run_ids, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def external_report(cfg: dict, db: str) -> dict:
    from ecgaug import final_eval
    from ecgaug.experiments import phase6_external_validation as ext

    selection = load_selection()
    arrays, data = ext.load_database(db)
    arrays = v5.substitute(arrays, v5.external_ratios(db, arrays, cfg["half_window_s"]))
    y, records = arrays["y"], arrays["record"]
    cms, unique = {}, None
    for arch, conds in selection["chosen"].items():
        for condition, chosen in conds.items():
            unique, cms[(arch, condition)] = final_eval.per_record_cms(pt._predict(chosen["run_ids"], arrays), y, records)
    metric = data["primary_metric"]
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": selection["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v8 (P2 rules)", "database": db, "role": data["role"],
               "primary_metric": metric, "beats": data["beats"], "records": len(unique), "conditions": conditions,
               "confirmatory": _family(cms, families(selection)["confirmatory"], final_eval.STATISTICS[metric]), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["data", "grid", "select", "test", "external"], required=True)
    parser.add_argument("--database", choices=["incartdb", "svdb", "all"], default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "data":
            _, record = build(cfg)
            print({k: v["seeds"]["0"]["sha256"][:12] for k, v in record["banks"].items()})
        elif args.part == "grid":
            data, _ = build(cfg)
            print(json.dumps(run_grid(cfg, data, resolve_device(args.device)), indent=2))
        elif args.part == "select":
            s = select(cfg)
            for arch, conditions in s["chosen"].items():
                print(arch, {c: (v["strength"], round(v["mean"], 3)) for c, v in conditions.items()})
        elif args.part == "test":
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:36s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            for key, c in s["conditions"].items():
                print(f"  {key:28s} rare-class macro-F1 {c['rare_macro_f1']['point']:.3f}  S F1 {c['f1_S']['point']:.3f}  F F1 {c['f1_F']['point']:.3f}")
            print("  v2 none reproduced (difference):", s["v2_none_reproduction_difference"])
        else:
            for db in (["incartdb", "svdb"] if args.database == "all" else [args.database]):
                s = external_report(cfg, db)
                for key, c in s["confirmatory"].items():
                    print(f"  {db} {key:36s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
