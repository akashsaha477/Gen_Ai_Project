"""phase6_rr_baseline: the full grid again with a patient-baseline RR ratio (PREREGISTRATION §10 v5, P5; post hoc).

    python -m ecgaug.experiments.phase6_rr_baseline --config configs/phase6_rr_baseline/rr_baseline.yaml --part data
    ALLOW_TEST_BUILD=1 python -m ecgaug.experiments.phase6_rr_baseline --config ... --part test-ratios   # construction access
    python -m ecgaug.experiments.phase6_rr_baseline --config ... --part grid [--device cuda]               # Kaggle
    python -m ecgaug.experiments.phase6_rr_baseline --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_rr_baseline --config ... --part test        # one posthoc access
    python -m ecgaug.experiments.phase6_rr_baseline --config ... --part external

Dataset v2 is the frozen v1 dataset with one field changed: ``rr_ratio`` = ``rr_prev`` divided by the median RR
interval of the beat's own record within ``half_window_s`` seconds (interval midpoints), instead of the mean of the
ten preceding intervals. ``--part data`` builds it from the frozen v1 arrays and the DS1 annotations, remaps the RR
ratio of the frozen synthetic banks (each synthetic beat takes the v2 ratio of the real training beat whose v1 RR
values it carries) and records every hash in ``results/phase6_rr_baseline/data.json``; on Kaggle the same code
rebuilds both and refuses unless the hashes match. ``--part grid`` runs the phase 4a gate's full grid unchanged on
v2, ``--part select`` applies the phase 4 gate rule with the same code, ``--part test`` evaluates DS2 once and
``--part external`` scores INCART and SVDB. Nothing here changes a frozen artifact or a reported result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.access import LockedTestDataError, log_test_access, logged_accesses
from ecgaug.data.beats import extract_beats
from ecgaug.data.contract import content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.loaders import load_test_posthoc
from ecgaug.data.records import CLASSES, DS2, FS, QRS_SYMBOLS
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, load_run_classifier
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators.bank import load_bank
from ecgaug.grid import runner
from ecgaug.names import PHASE3_GATE, PHASE4_GATE, PHASE6_RR_BASELINE
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE6_RR_BASELINE
VERSION = "v2"
CONFIG_KEYS = {"experiment", "half_window_s"}
GENERATIVE = {"cvae": "phase3_cvae", "wgan_gp": "phase3_wgan_gp"}
GRID_NAME = "full_grid"
TEST_RATIOS = "test_rr_ratio.npz"
REFERENCE_KEYS = tuple(ref.REFERENCES)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path)}


# --------------------------------------------------------------------------------------------------
# the feature
# --------------------------------------------------------------------------------------------------


def baseline_ratio(qrs_samples: np.ndarray, fs: float, half_window_s: float) -> np.ndarray:
    """``rr_prev`` over the median RR interval whose midpoint lies within ``half_window_s`` of the beat.

    One value per QRS annotation (NaN for the first, which has no ``rr_prev``). Times come from the
    annotation samples only; no label is used.
    """
    t = np.asarray(qrs_samples, dtype=np.float64) / float(fs)
    out = np.full(t.size, np.nan)
    if t.size < 2:
        return out
    rr = np.diff(t)
    mid = (t[1:] + t[:-1]) / 2
    lo = np.searchsorted(mid, t - half_window_s, side="left")
    hi = np.searchsorted(mid, t + half_window_s, side="right")
    for i in range(1, t.size):
        out[i] = rr[i - 1] / np.median(rr[lo[i] : hi[i]])  # the window always holds the beat's own interval
    return out


def record_ratios(record: int, half_window_s: float, access: str | None = None) -> dict[int, float]:
    """``{r_idx: v2 rr_ratio}`` for every QRS annotation of an MIT-BIH record."""
    beats = extract_beats(record, access)
    r = beats["r_idx"].to_numpy()
    return dict(zip(r.tolist(), baseline_ratio(r, FS, half_window_s).tolist()))


def substitute(arrays: dict, ratios: dict[tuple[int, int], float]) -> dict:
    """The arrays with ``rr_ratio`` replaced by the v2 value of each (record, r_idx); refuses a beat without one."""
    keys = list(zip(np.asarray(arrays["record"]).tolist(), np.asarray(arrays["r_idx"]).tolist()))
    missing = [k for k in keys if k not in ratios or not np.isfinite(ratios[k])]
    if missing:
        raise PreconditionError(f"{len(missing)} beats have no v2 ratio, e.g. {missing[:3]}")
    return {**arrays, "rr_ratio": np.array([ratios[k] for k in keys], dtype=np.float32)}


# --------------------------------------------------------------------------------------------------
# data: v2 training and validation beats, v2 banks (DS1 only)
# --------------------------------------------------------------------------------------------------


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def v2_dir() -> Path:
    return paths.processed_dir() / VERSION


def bank_folder(generator_id: str) -> Path:
    return paths.repo_root() / "data" / "synthetic" / EXPERIMENT / generator_id


def frozen_generators() -> dict[str, dict]:
    frozen = workflow.manifest().get(PHASE3_GATE) or {}
    if not all(e in frozen for e in GENERATIVE.values()):
        raise PreconditionError("The phase 3 gate must be approved")
    return {c: frozen[e] for c, e in GENERATIVE.items()}


def remap_bank(bank: dict, v1_train: dict, v2_train: dict) -> dict:
    """A bank with each synthetic beat's ``rr_ratio`` replaced by the v2 value of the real training beat of its class
    whose v1 (rr_prev, rr_next, rr_ratio) it carries (the first in dataset order on ties)."""
    lookup: dict[tuple, float] = {}
    for i in range(len(v1_train["y"])):
        key = (int(v1_train["y"][i]), *(np.float32(v1_train[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio")))
        lookup.setdefault(key, float(v2_train["rr_ratio"][i]))
    out = np.empty(len(bank["y"]), dtype=np.float32)
    for i in range(len(bank["y"])):
        key = (int(bank["y"][i]), *(np.float32(bank[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio")))
        if key not in lookup:
            raise PreconditionError(f"Synthetic beat {i} carries RR values of no real training beat of its class")
        out[i] = lookup[key]
    return {**bank, "rr_ratio": out}


def _train(arrays: dict) -> dict:
    mask = (np.asarray(arrays["split"]) == "train") & ~np.asarray(arrays["is_synthetic"])
    return {k: np.asarray(v)[mask] for k, v in arrays.items()}


def build(cfg: dict) -> tuple[DataSource, dict]:
    """v2 training and validation beats and v2 banks, built and saved, checked against ``data.json`` when it exists."""
    split = check_preconditions()
    v1 = load_data(split, preview=False)
    ratios = {}
    for record in sorted(set(np.asarray(v1.arrays["record"]).tolist())):
        ratios.update({(record, r): v for r, v in record_ratios(record, cfg["half_window_s"]).items()})
    arrays = substitute(v1.arrays, ratios)
    for name in ("X", "y", "record", "split", "r_idx", "rr_prev", "rr_next", "is_synthetic", "source"):
        if not np.array_equal(np.asarray(arrays[name]), np.asarray(v1.arrays[name])):
            raise PreconditionError(f"v2 field {name} differs from v1")
    sha = content_sha256(arrays)
    record = data_record()
    if record is not None and record["trainval_sha256"] != sha:
        raise PreconditionError(f"v2 training data {sha[:12]}... does not match data.json ({record['trainval_sha256'][:12]}...)")
    v2_dir().mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(v2_dir() / "trainval.npz", arrays)
    v1_train, v2_train = _train(v1.arrays), _train(arrays)
    banks = {}
    for condition, entry in frozen_generators().items():
        gid = entry["generator_id"]
        folder = bank_folder(gid)
        folder.mkdir(parents=True, exist_ok=True)
        seeds = {}
        for seed, frozen in sorted(entry["banks"].items()):
            remapped = remap_bank(load_bank(gid, int(seed), expected_sha256=frozen["sha256"]), v1_train, v2_train)
            save_npz_deterministic(folder / f"seed{seed}.npz", remapped)
            seeds[seed] = {"sha256": sha256_file(folder / f"seed{seed}.npz"), "frozen_sha256": frozen["sha256"]}
            if record is not None and record["banks"][condition]["seeds"][seed]["sha256"] != seeds[seed]["sha256"]:
                raise PreconditionError(f"Remapped bank {gid} seed {seed} does not match data.json")
        (folder / "bank.json").write_text(json.dumps({"generator_id": gid, "rule": "PREREGISTRATION §10 v5", "seeds": seeds}, indent=2) + "\n")
        banks[condition] = {"generator_id": gid, "seeds": seeds}
    info = {"source": f"data/processed/{VERSION} (v1 with the baseline RR ratio, PREREGISTRATION v5)", "representation": v1.spec.name,
            "content_sha256": sha, "frozen": False, "version": VERSION}  # fmt: skip
    if record is None:
        train, val = arrays["split"] == "train", arrays["split"] == "val"
        diagnostic = {part: {c: {"n": int(((arrays["y"] == i) & m).sum()),
                                  "below_0.9_v1": float(np.mean(v1.arrays["rr_ratio"][(arrays["y"] == i) & m] < 0.9)),
                                  "below_0.9_v2": float(np.mean(arrays["rr_ratio"][(arrays["y"] == i) & m] < 0.9))}
                             for i, c in enumerate(CLASSES)} for part, m in (("train", train), ("val", val))}  # fmt: skip
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v5", "half_window_s": cfg["half_window_s"], "v1_sha256": v1.info["content_sha256"],
                  "trainval_sha256": sha, "banks": banks, "diagnostic": diagnostic, "git_sha": runs.git_sha()}  # fmt: skip
        out = paths.results_for(EXPERIMENT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    return DataSource(arrays, v1.spec, info), record


def build_test_ratios(cfg: dict) -> dict:
    """v2 ratios of every QRS annotation of the DS2 records, under one logged construction access; no labels saved."""
    rec, idx, val = [], [], []
    for record in DS2:
        for r, v in record_ratios(record, cfg["half_window_s"], access="construction").items():
            rec.append(record), idx.append(r), val.append(v)
    arrays = {"record": np.array(rec, dtype=np.int16), "r_idx": np.array(idx, dtype=np.int32), "rr_ratio": np.array(val, dtype=np.float32)}
    v2_dir().mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(v2_dir() / TEST_RATIOS, arrays)
    log_test_access("construction", f"{EXPERIMENT}: v2 RR ratios of DS2 from annotation times (no labels, no model)", run_ids="")
    record = data_record()
    record["test_ratios_sha256"] = sha256_file(v2_dir() / TEST_RATIOS)
    (paths.results_for(EXPERIMENT) / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"beats": len(val), "sha256": record["test_ratios_sha256"]}


def load_v2() -> DataSource:
    """The saved v2 training and validation beats, checked against ``data.json``."""
    record = data_record()
    if record is None:
        raise PreconditionError("Build v2 first: --part data")
    with np.load(v2_dir() / "trainval.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    sha = content_sha256(arrays)
    if sha != record["trainval_sha256"]:
        raise PreconditionError("data/processed/v2/trainval.npz changed since it was built")
    from ecgaug.experiments.phase6_external_validation import frozen_spec

    spec = frozen_spec()
    return DataSource(arrays, spec, {"source": f"data/processed/{VERSION}", "representation": spec.name, "content_sha256": sha, "frozen": False, "version": VERSION})


def load_v2_bank(condition: str, seed: int) -> tuple[dict, str]:
    entry = data_record()["banks"][condition]
    path = bank_folder(entry["generator_id"]) / f"seed{seed}.npz"
    sha = sha256_file(path)
    if sha != entry["seeds"][str(seed)]["sha256"]:
        raise PreconditionError(f"{path} does not match data.json")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}, sha


# --------------------------------------------------------------------------------------------------
# the grid and the selection (validation only)
# --------------------------------------------------------------------------------------------------


def v2_grid(cfg: dict) -> dict:
    """The phase 4a gate's full grid with this experiment's name and banks; nothing else changes."""
    full = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    conditions = {c: dict(full["conditions"][c] or {}) for c in full["conditions"]}
    for condition in GENERATIVE:
        conditions[condition]["bank"] = f"{EXPERIMENT}/{data_record()['banks'][condition]['generator_id']}"
    return {**full, "experiment": EXPERIMENT, "name": GRID_NAME, "conditions": conditions, "file": cfg["file"]}


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    split = check_preconditions()
    grid = v2_grid(cfg)
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in GENERATIVE:
            bank, bank_sha = load_v2_bank(cell.condition, cell.seed)
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
    """The phase 4 gate rule, with the same code, on the v2 grid's ledger; written before any DS2 read."""
    grid = v2_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if runner.is_done(r)}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, best = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v5 (the phase 4 gate rule)", "chosen": chosen, "best_generative": best, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# references, DS2 and the external databases
# --------------------------------------------------------------------------------------------------


def fit_references(data: DataSource) -> dict:
    """The two P1 references refitted on the v2 training beats (one run each)."""
    path = paths.results_for(EXPERIMENT) / "references.json"
    if path.is_file():
        return json.loads(path.read_text())
    train, val = _train(data.arrays), {k: np.asarray(v)[np.asarray(data.arrays["split"]) == "val"] for k, v in data.arrays.items()}
    out = {}
    for reference in REFERENCE_KEYS:
        model = ref.fit_reference(train, reference)
        probs = ref.predict(model, val, reference)
        m = M.evaluate(probs, val["y"])
        run_id = runs.make_run_id(EXPERIMENT, reference.replace("_", ""), 0)
        run_dir = paths.runs_dir() / EXPERIMENT / "references" / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        blob = pickle.dumps(model)
        (run_dir / "checkpoints" / "model.pkl").write_bytes(blob)
        sha = hashlib.sha256(blob).hexdigest()
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "reference": reference, "dataset": f"data/processed/{VERSION}", "run_id": run_id}, sort_keys=False))
        (run_dir / "env.json").write_text(json.dumps({**runs.capture_env("cpu"), "wall_clock_s": 0.0, "checkpoint": {"path": "checkpoints/model.pkl", "sha256": sha}}, indent=2) + "\n")
        (run_dir / "train_log.csv").write_text("step\n0\n")
        (run_dir / "metrics_val.json").write_text(json.dumps({**m, "selection_metric": "", "selection_value": float("nan")}, indent=2, default=float) + "\n")
        np.savez_compressed(run_dir / "preds_val.npz", probs=probs, y=val["y"], record=val["record"], r_idx=val["r_idx"])
        runs.upsert_runs_index(runs.index_row(run_id, {**m, "selection_metric": "", "selection_value": float("nan")}, 0.0, group=(EXPERIMENT, "references")))
        out[reference] = {"run_id": run_id, "sha256": sha, "validation_rare_macro_f1": m["rare_macro_f1"], "validation_f1_S": m["per_class"]["S"]["f1"]}
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_references() -> dict:
    fitted = json.loads((paths.results_for(EXPERIMENT) / "references.json").read_text())
    models = {}
    for reference, entry in fitted.items():
        blob = (runs.find_run_dir(entry["run_id"]) / "checkpoints" / "model.pkl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise PreconditionError(f"{entry['run_id']}: the fitted model changed")
        models[reference] = pickle.loads(blob)  # noqa: S301 (our own file, hash checked)
    return models


def _predict_all(selection: dict, arrays: dict) -> dict[tuple[str, str], list[np.ndarray]]:
    x = torch.from_numpy(np.asarray(arrays["X"]))
    side = torch.from_numpy(np.stack([arrays[f] for f in SIDE_FIELDS], axis=1))
    preds: dict[tuple[str, str], list[np.ndarray]] = {}
    for arch, conditions in selection["chosen"].items():
        for condition, chosen in conditions.items():
            preds[(arch, condition)] = [predict_proba(load_run_classifier(run_id, True), x, side, torch.device("cpu")) for run_id in chosen["run_ids"]]
    return preds


def test_report() -> dict:
    """DS2 with v2 features: the §7 analysis, v2 minus v1 per condition, and the v2 references (one posthoc access)."""
    from ecgaug import final_eval

    if not logged_accesses("evaluation"):
        raise PreconditionError("Post-hoc analyses run only after the final evaluation")
    selection, record = load_selection(), data_record()
    path = v2_dir() / TEST_RATIOS
    if "test_ratios_sha256" not in record or sha256_file(path) != record["test_ratios_sha256"]:
        raise PreconditionError("Build the DS2 ratios first (--part test-ratios), or they changed since")
    run_ids = [r for a in selection["chosen"].values() for c in a.values() for r in c["run_ids"]]
    test = load_test_posthoc("v1", f"{EXPERIMENT}: DS2 predictions with the v2 RR ratio", run_ids=f"{len(run_ids)} checkpoints (results/{EXPERIMENT}/selection.json)")
    with np.load(path, allow_pickle=False) as f:
        ratios = {(int(r), int(i)): float(v) for r, i, v in zip(f["record"], f["r_idx"], f["rr_ratio"])}
    test = substitute(test, ratios)
    y, records = test["y"], test["record"]
    preds = _predict_all(selection, test)
    for (arch, condition), probs in preds.items():
        for run_id, p in zip(selection["chosen"][arch][condition]["run_ids"], probs):
            np.savez_compressed(runs.find_run_dir(run_id) / "preds_test.npz", probs=p, y=y, record=records, r_idx=test["r_idx"])
    stats = final_eval.statistics(preds, y, records, selection)
    v1_gate = workflow.manifest()[PHASE4_GATE]
    versus_v1 = {}
    for (arch, condition), probs in preds.items():
        old = []
        for run_id in v1_gate["chosen"][arch][condition]["run_ids"]:
            with np.load(runs.find_run_dir(run_id) / "preds_test.npz") as saved:
                if not np.array_equal(saved["r_idx"], test["r_idx"]):
                    raise PreconditionError(f"{run_id}: saved test predictions do not match the frozen test beats")
                old.append(saved["probs"])
        unique, new_cm = final_eval.per_record_cms(probs, y, records)
        _, old_cm = final_eval.per_record_cms(old, y, records)
        versus_v1[f"{arch}/{condition}"] = paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, new_cm, old_cm, n=ref.N_BOOT, seed=ref.BOOT_SEED)
    references = {}
    for reference, model in load_references().items():
        unique, cm = final_eval.per_record_cms([ref.predict(model, test, reference)], y, records)
        references[reference] = ref.condition_summary(cm, unique)
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v5", **{k: stats[k] for k in ("records", "n_boot", "boot_seed", "conditions", "families", "sensitivity")},
               "versus_v1": versus_v1, "references": references, "run_ids": run_ids, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def external_ratios(db: str, arrays: dict, half_window_s: float) -> dict[tuple[int, int], float]:
    """v2 ratios of an external database's beats, from its annotations at the native rate, keyed by (record, r_idx at 360 Hz)."""
    from ecgaug.experiments import phase6_external_validation as ext

    wanted = set(np.asarray(arrays["record"]).tolist())
    out = {}
    for name in ext.record_names(db):
        number = ext.record_number(name)
        if number not in wanted:
            continue
        samples, symbols, _, fs, _ = raw.read_external(db, name, ext.DATABASES[db]["lead"])
        r_native = samples[np.isin(symbols, sorted(QRS_SYMBOLS))]
        r = np.rint(r_native * FS / fs).astype(np.int64)
        out.update({(number, int(i)): float(v) for i, v in zip(r, baseline_ratio(r_native, fs, half_window_s))})
    return out


def external_report(cfg: dict, db: str) -> dict:
    """INCART or SVDB with v2 features: the v2 checkpoints and references, the confirmatory family with Holm (P2 rules)."""
    from ecgaug import final_eval
    from ecgaug.experiments import phase6_external_validation as ext

    selection = load_selection()
    arrays, data = ext.load_database(db)
    arrays = substitute(arrays, external_ratios(db, arrays, cfg["half_window_s"]))
    y, records = arrays["y"], ext.clusters(db, arrays)  # bootstrap over patients, not records
    cms = {}
    for key, probs in _predict_all(selection, arrays).items():
        unique, cms[key] = final_eval.per_record_cms(probs, y, records)
    metric = data["primary_metric"]
    stat = final_eval.STATISTICS[metric]
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": selection["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    references = {}
    for reference, model in load_references().items():
        _, cm = final_eval.per_record_cms([ref.predict(model, arrays, reference)], y, records)
        references[reference] = ref.condition_summary(cm, unique)
    family = {f"{arch}: {a} vs {b}": paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, a)], cms[(arch, b)], n=ref.N_BOOT, seed=ref.BOOT_SEED)
              for arch, a, b in final_eval.comparisons(selection)["confirmatory"]}  # fmt: skip
    adjusted = holm({k: v["p_two_sided"] for k, v in family.items()})
    for key, value in family.items():
        value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"])
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v5 (P2 rules)", "database": db, "role": data["role"],
               "primary_metric": metric, "beats": data["beats"], "records": int(len(np.unique(arrays["record"]))), "patients": len(unique), "clustered_by": "patient", "conditions": conditions, "references": references,
               "confirmatory": family, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["data", "test-ratios", "grid", "select", "references", "test", "external"], required=True)
    parser.add_argument("--database", choices=["incartdb", "svdb", "all"], default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "data":
            _, record = build(cfg)
            d = record["diagnostic"]["train"]["S"]
            print(f"v2 built: {record['trainval_sha256'][:12]}; training S beats with ratio < 0.9: v1 {d['below_0.9_v1']:.2f}, v2 {d['below_0.9_v2']:.2f}")
        elif args.part == "test-ratios":
            print(build_test_ratios(cfg))
        elif args.part == "grid":
            data, _ = build(cfg)  # rebuilt and hash-checked (Kaggle has no v2 on disk)
            print(run_grid(cfg, data, resolve_device(args.device)))
        elif args.part == "select":
            s = select(cfg)
            for arch, conditions in s["chosen"].items():
                print(arch, {c: v["strength"] for c, v in conditions.items()}, "best generative:", s["best_generative"].get(arch))
        elif args.part == "references":
            print(fit_references(load_v2()))
        elif args.part == "test":
            s = test_report()
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:32s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            for key, c in s["conditions"].items():
                d = s["versus_v1"][key]
                print(f"  {key:28s} {c['rare_macro_f1']['point']:.3f}  v2 - v1 {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
        else:
            for db in (["incartdb", "svdb"] if args.database == "all" else [args.database]):
                s = external_report(cfg, db)
                for key, c in s["confirmatory"].items():
                    print(f"  {db} {key:28s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
