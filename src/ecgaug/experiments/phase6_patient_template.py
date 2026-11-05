"""phase6_patient_template: patient-relative beats, the full grid again (PREREGISTRATION §10 v6, P6; post hoc).

    python -m ecgaug.experiments.phase6_patient_template --config configs/phase6_patient_template/patient_template.yaml --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_patient_template --config ... --part diagnostic       # validation only, CPU
    python -m ecgaug.experiments.phase6_patient_template --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_patient_template --config ... --part test   # one posthoc access
    python -m ecgaug.experiments.phase6_patient_template --config ... --part external

Dataset v3 is dataset v2 (the patient-baseline RR ratio of v5) with one more change: each beat window ``X`` is
replaced by its difference from the patient's local median beat, the element-wise median of the z-scored windows of
every QRS annotation of the same record whose R peak lies within ``half_window_s`` seconds (the beat itself
included). The template uses annotation times and the signal only, never a label, so it is computed the same way
for training, validation, test and external records. A synthetic beat takes the template (and the v2 ratio) of the
real training beat of its class whose v1 RR values it carries, as in v5.

``--part grid`` first runs the cells without augmentation; the adoption rule (fixed in the pre-registration) then
compares them with the v5 cells on validation, and the rest of the grid runs only if v3 is adopted. Everything after
that repeats v5 with the same code: the phase 4 gate rule on validation, one DS2 read, INCART and SVDB.
``--part diagnostic`` describes on validation why the networks lose S beats: the train-versus-validation gap of the
frozen v1, v2 and v3 networks, per-record S recall, and fixed linear models on three feature sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.access import LockedTestDataError, log_test_access, logged_accesses, require_test_access
from ecgaug.data.beats import extract_beats
from ecgaug.data.contract import build_arrays, content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.loaders import frozen_test_sha256, test_arrays_present
from ecgaug.data.records import CLASSES, DS2, FS, QRS_SYMBOLS
from ecgaug.data.representation import RepresentationSpec, transform_record
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, load_run_classifier, mean_sd, split_part
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators.bank import load_bank
from ecgaug.grid import runner
from ecgaug.names import PHASE1_GATE, PHASE4_GATE, PHASE6_PATIENT_TEMPLATE
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE6_PATIENT_TEMPLATE
VERSION = "v3"
CONFIG_KEYS = {"experiment", "half_window_s", "adoption_margin"}
GENERATIVE = v5.GENERATIVE
GRID_NAME = "full_grid"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path)}


# --------------------------------------------------------------------------------------------------
# the template
# --------------------------------------------------------------------------------------------------


def local_templates(windows: np.ndarray, r_all: np.ndarray, r_wanted: np.ndarray, fs: float, half_window_s: float) -> np.ndarray:
    """For each R peak in ``r_wanted``: the element-wise median of the ``windows`` (one per R peak in ``r_all``,
    sorted) whose R peak lies within ``half_window_s`` seconds of it. Every wanted peak must be in ``r_all``."""
    r_all, r_wanted = np.asarray(r_all, dtype=np.int64), np.asarray(r_wanted, dtype=np.int64)
    if r_all.size > 1 and not np.all(np.diff(r_all) > 0):
        raise ValueError("R peaks must be strictly increasing")
    if not np.isin(r_wanted, r_all).all():
        raise ValueError("Every beat needs its own window among the template windows")
    h = int(round(half_window_s * fs))
    lo = np.searchsorted(r_all, r_wanted - h, side="left")
    hi = np.searchsorted(r_all, r_wanted + h, side="right")
    return np.stack([np.median(windows[a:b], axis=0) for a, b in zip(lo, hi)]).astype(np.float32) if r_wanted.size else np.zeros((0, windows.shape[1]), np.float32)


def signal_templates(signal: np.ndarray, r_qrs: np.ndarray, r_wanted: np.ndarray, spec: RepresentationSpec, half_window_s: float) -> np.ndarray:
    """Templates of the wanted beats from a signal at 360 Hz and all its QRS annotations (no labels)."""
    pre, post = spec.window_samples
    r = np.asarray(r_qrs, dtype=np.int64)
    r = r[(r - pre >= 0) & (r + post <= np.asarray(signal).size)]
    return local_templates(transform_record(signal, r, spec), r, r_wanted, FS, half_window_s)


def record_templates(record: int, r_wanted: np.ndarray, spec: RepresentationSpec, half_window_s: float, access: str | None = None) -> np.ndarray:
    """Templates of the wanted beats of an MIT-BIH record, from every QRS annotation of the record."""
    beats = extract_beats(record, access)
    return signal_templates(raw.read_signal(record, access), beats["r_idx"].to_numpy(), r_wanted, spec, half_window_s)


def relative(arrays: dict, templates: np.ndarray) -> dict:
    """The arrays with ``X`` replaced by ``X`` minus its template."""
    return {**arrays, "X": (np.asarray(arrays["X"]) - templates).astype(np.float32)}


def mitbih_templates(arrays: dict, spec: RepresentationSpec, half_window_s: float, access: str | None = None) -> np.ndarray:
    out = np.empty_like(np.asarray(arrays["X"]), dtype=np.float32)
    records = np.asarray(arrays["record"])
    for record in sorted(set(records.tolist())):
        idx = np.where(records == record)[0]
        order = idx[np.argsort(np.asarray(arrays["r_idx"])[idx])]
        out[order] = record_templates(record, np.asarray(arrays["r_idx"])[order], spec, half_window_s, access)
    return out


def remap_bank(bank: dict, v1_train: dict, v3_train: dict, templates: np.ndarray) -> dict:
    """A bank in v3: each synthetic beat takes the v2 ratio and the template of the real training beat of its class whose
    v1 (rr_prev, rr_next, rr_ratio) it carries (the first in dataset order on ties, the v5 rule)."""
    lookup: dict[tuple, int] = {}
    for i in range(len(v1_train["y"])):
        lookup.setdefault((int(v1_train["y"][i]), *(np.float32(v1_train[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio"))), i)
    match = np.empty(len(bank["y"]), dtype=np.int64)
    for i in range(len(bank["y"])):
        key = (int(bank["y"][i]), *(np.float32(bank[f][i]).tobytes() for f in ("rr_prev", "rr_next", "rr_ratio")))
        if key not in lookup:
            raise PreconditionError(f"Synthetic beat {i} carries RR values of no real training beat of its class")
        match[i] = lookup[key]
    return {**bank, "X": (np.asarray(bank["X"]) - templates[match]).astype(np.float32), "rr_ratio": np.asarray(v3_train["rr_ratio"])[match].astype(np.float32)}


# --------------------------------------------------------------------------------------------------
# data: v3 training and validation beats, v3 banks (DS1 only)
# --------------------------------------------------------------------------------------------------


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def bank_folder(generator_id: str) -> Path:
    return paths.repo_root() / "data" / "synthetic" / EXPERIMENT / generator_id


def _train(arrays: dict) -> dict:
    mask = (np.asarray(arrays["split"]) == "train") & ~np.asarray(arrays["is_synthetic"])
    return {k: np.asarray(v)[mask] for k, v in arrays.items()}


def _v2_arrays(v1: DataSource, half_window_s: float) -> dict:
    ratios = {}
    for record in sorted(set(np.asarray(v1.arrays["record"]).tolist())):
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half_window_s).items()})
    return v5.substitute(v1.arrays, ratios)


def build(cfg: dict) -> tuple[DataSource, dict]:
    """v3 training and validation beats and v3 banks, built and saved; checked against ``data.json`` when it exists."""
    split = check_preconditions()
    v1 = load_data(split, preview=False)
    v2 = _v2_arrays(v1, cfg["half_window_s"])
    if content_sha256(v2) != v5.data_record()["trainval_sha256"]:
        raise PreconditionError("The v2 beats do not match results/phase6_rr_baseline/data.json")
    templates = mitbih_templates(v1.arrays, v1.spec, cfg["half_window_s"])
    arrays = relative(v2, templates)
    sha = content_sha256(arrays)
    record = data_record()
    if record is not None and record["trainval_sha256"] != sha:
        raise PreconditionError(f"v3 training data {sha[:12]}... does not match data.json ({record['trainval_sha256'][:12]}...)")
    out_dir = paths.processed_dir() / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(out_dir / "trainval.npz", arrays)
    train_mask = (np.asarray(v1.arrays["split"]) == "train") & ~np.asarray(v1.arrays["is_synthetic"])
    v1_train, v3_train, train_templates = _train(v1.arrays), _train(arrays), templates[train_mask]
    banks = {}
    for condition, entry in v5.frozen_generators().items():
        gid = entry["generator_id"]
        folder = bank_folder(gid)
        folder.mkdir(parents=True, exist_ok=True)
        seeds = {}
        for seed, frozen in sorted(entry["banks"].items()):
            remapped = remap_bank(load_bank(gid, int(seed), expected_sha256=frozen["sha256"]), v1_train, v3_train, train_templates)
            save_npz_deterministic(folder / f"seed{seed}.npz", remapped)
            seeds[seed] = {"sha256": sha256_file(folder / f"seed{seed}.npz"), "frozen_sha256": frozen["sha256"]}
            if record is not None and record["banks"][condition]["seeds"][seed]["sha256"] != seeds[seed]["sha256"]:
                raise PreconditionError(f"v3 bank {gid} seed {seed} does not match data.json")
        (folder / "bank.json").write_text(json.dumps({"generator_id": gid, "rule": "PREREGISTRATION §10 v6", "seeds": seeds}, indent=2) + "\n")
        banks[condition] = {"generator_id": gid, "seeds": seeds}
    info = {"source": f"data/processed/{VERSION} (v2 with patient-relative beats, PREREGISTRATION v6)", "representation": v1.spec.name,
            "content_sha256": sha, "frozen": False, "version": VERSION}  # fmt: skip
    if record is None:
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v6", "half_window_s": cfg["half_window_s"], "v1_sha256": v1.info["content_sha256"],
                  "v2_sha256": content_sha256(v2), "trainval_sha256": sha, "banks": banks, "git_sha": runs.git_sha()}  # fmt: skip
        out = paths.results_for(EXPERIMENT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    return DataSource(arrays, v1.spec, info), record


def load_v3_bank(condition: str, seed: int) -> tuple[dict, str]:
    entry = data_record()["banks"][condition]
    path = bank_folder(entry["generator_id"]) / f"seed{seed}.npz"
    sha = sha256_file(path)
    if sha != entry["seeds"][str(seed)]["sha256"]:
        raise PreconditionError(f"{path} does not match data.json")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}, sha


# --------------------------------------------------------------------------------------------------
# the grid, the adoption rule and the selection (validation only)
# --------------------------------------------------------------------------------------------------


def v3_grid(cfg: dict) -> dict:
    """The phase 4a gate's full grid with this experiment's name and banks; nothing else changes."""
    full = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    conditions = {c: dict(full["conditions"][c] or {}) for c in full["conditions"]}
    for condition in GENERATIVE:
        conditions[condition]["bank"] = f"{EXPERIMENT}/{data_record()['banks'][condition]['generator_id']}"
    return {**full, "experiment": EXPERIMENT, "name": GRID_NAME, "conditions": conditions, "file": cfg["file"]}


def _run_cells(grid: dict, data: DataSource, device: torch.device, keep) -> dict:
    split = check_preconditions()
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        if not keep(cell):
            continue
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in GENERATIVE:
            bank, bank_sha = load_v3_bank(cell.condition, cell.seed)
            bank_id = grid["conditions"][cell.condition]["bank"]
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def _none_rows(grid: dict) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for row in runner.read_ledger(grid).values():
        if row["condition"] == "none" and row["status"] == "done":  # the committed ledger; run folders are in git
            rows.setdefault(row["architecture"], []).append(row)
    return rows


def adoption(cfg: dict, v3_rows: dict[str, list[dict]], v2_rows: dict[str, list[dict]]) -> dict:
    """The adoption rule (PREREGISTRATION §10 v6): v3 is adopted if the mean over the architectures of the difference in
    seed-mean validation selection value (no augmentation, v3 minus v2) exceeds max(adoption_margin, pooled seed SD)."""
    per_arch, variances = {}, []
    for arch in sorted(v2_rows):
        new = [r["selection_value"] for r in sorted(v3_rows[arch], key=lambda r: r["seed"])]
        old = [r["selection_value"] for r in sorted(v2_rows[arch], key=lambda r: r["seed"])]
        if len(new) != len(old):
            raise PreconditionError(f"{arch}: {len(new)} v3 seeds against {len(old)} v2 seeds")
        (m_new, sd_new), (m_old, sd_old) = mean_sd(new), mean_sd(old)
        variances += [sd_new**2, sd_old**2]
        per_arch[arch] = {"v3_mean": m_new, "v3_sd": sd_new, "v2_mean": m_old, "v2_sd": sd_old, "difference": m_new - m_old,
                          "v3_run_ids": [r["run_id"] for r in v3_rows[arch]], "v2_run_ids": [r["run_id"] for r in v2_rows[arch]]}  # fmt: skip
    mean_difference = float(np.mean([v["difference"] for v in per_arch.values()]))
    margin = max(float(cfg["adoption_margin"]), float(np.sqrt(np.mean(variances))))
    return {"rule": "PREREGISTRATION §10 v6 (adoption)", "per_architecture": per_arch, "mean_difference": mean_difference, "margin": margin,
            "adopted": mean_difference > margin}  # fmt: skip


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    """The cells without augmentation, the adoption rule, then (only if adopted) every other cell."""
    grid = v3_grid(cfg)
    report = {"none": _run_cells(grid, data, device, lambda c: c.condition == "none")}
    v2_rows = _none_rows(v5.v2_grid({"file": "configs/phase6_rr_baseline/rr_baseline.yaml"}))
    decision = adoption(cfg, _none_rows(grid), v2_rows)
    (paths.results_for(EXPERIMENT) / "adoption.json").write_text(json.dumps({**decision, "git_sha": runs.git_sha()}, indent=2, default=float) + "\n")
    report["adoption"] = {k: decision[k] for k in ("mean_difference", "margin", "adopted")}
    if decision["adopted"]:
        report["rest"] = _run_cells(grid, data, device, lambda c: c.condition != "none")
    return report


def load_adoption() -> dict:
    path = paths.results_for(EXPERIMENT) / "adoption.json"
    if not path.is_file():
        raise PreconditionError("Run the grid first: --part grid")
    return json.loads(path.read_text())


def select(cfg: dict) -> dict:
    """The phase 4 gate rule, with the same code, on the v3 grid's ledger; written before any DS2 read."""
    if not load_adoption()["adopted"]:
        raise PreconditionError("v3 was not adopted on validation: the grid stops after the cells without augmentation")
    grid = v3_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if runner.is_done(r)}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, best = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v6 (the phase 4 gate rule)", "chosen": chosen, "best_generative": best, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# the diagnostic (validation only)
# --------------------------------------------------------------------------------------------------

FEATURE_SETS = ("rr", "rr_raw", "rr_relative")


def linear_features(arrays: dict, name: str) -> np.ndarray:
    """RR features (v2), alone or with every 10th sample of the beat (raw) or of the patient-relative beat."""
    rr = np.stack([np.asarray(arrays[f], dtype=np.float64) for f in SIDE_FIELDS], axis=1)
    if name == "rr":
        return rr
    return np.concatenate([rr, np.asarray(arrays["X"], dtype=np.float64)[:, :: ref.MORPH_STEP]], axis=1)


def _class_report(probs: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score

    m = M.evaluate(probs, y)
    return {"f1": {c: m["per_class"][c]["f1"] for c in CLASSES}, "rare_macro_f1": m["rare_macro_f1"],
            "pr_auc": {c: float(average_precision_score(y == i, probs[:, i])) if (y == i).any() else float("nan") for i, c in enumerate(CLASSES)}}  # fmt: skip


def network_gap(run_ids: list[str], data: DataSource) -> dict:
    """Seed means of per-class PR-AUC and F1 on the training and validation beats, and per-record S recall on validation."""
    parts = {name: split_part(data.arrays, name, True) for name in ("train", "val")}
    out: dict = {"run_ids": run_ids}
    per_seed: dict[str, list[dict]] = {"train": [], "val": []}
    recall: dict[int, list[float]] = {}
    for run_id in run_ids:
        model = load_run_classifier(run_id, True)
        for name, part in parts.items():
            probs = predict_proba(model, torch.from_numpy(part["X"]), torch.from_numpy(part["side"]), torch.device("cpu"))
            per_seed[name].append(_class_report(probs, part["y"]))
            if name == "val":
                pred = probs.argmax(axis=1)
                for record in sorted(set(part["record"][part["y"] == 1].tolist())):
                    mask = (part["record"] == record) & (part["y"] == 1)
                    recall.setdefault(int(record), []).append(float((pred[mask] == 1).mean()))
    for name, reports in per_seed.items():
        out[name] = {"pr_auc": {c: float(np.mean([r["pr_auc"][c] for r in reports])) for c in CLASSES},
                     "f1": {c: float(np.mean([r["f1"][c] for r in reports])) for c in CLASSES}}  # fmt: skip
    counts = {int(r): int(((parts["val"]["record"] == r) & (parts["val"]["y"] == 1)).sum()) for r in recall}
    out["val_s_recall_by_record"] = {str(r): {"n": counts[r], "recall": float(np.mean(v))} for r, v in recall.items()}
    return out


def diagnostic(cfg: dict) -> dict:
    """Why the networks lose S beats, on DS1 only: the gap of the v1, v2 and v3 networks, and linear models on fixed features."""
    split = check_preconditions()
    v1 = load_data(split, preview=False)
    v2 = DataSource(_v2_arrays(v1, cfg["half_window_s"]), v1.spec, {**v1.info, "version": "v2"})
    v3, _ = build(cfg)
    gate = workflow.manifest()[PHASE4_GATE]["chosen"]
    v5_rows = _none_rows(v5.v2_grid({"file": "configs/phase6_rr_baseline/rr_baseline.yaml"}))
    v3_rows = _none_rows(v3_grid(cfg))
    networks = {}
    for arch in sorted(gate):
        networks[arch] = {"v1": network_gap(gate[arch]["none"]["run_ids"], v1),
                          "v2": network_gap([r["run_id"] for r in sorted(v5_rows[arch], key=lambda r: r["seed"])], v2)}  # fmt: skip
        if arch in v3_rows:
            networks[arch]["v3"] = network_gap([r["run_id"] for r in sorted(v3_rows[arch], key=lambda r: r["seed"])], v3)
    linear = {}
    for name, source in (("rr", v2), ("rr_raw", v2), ("rr_relative", v3)):
        train = _train(source.arrays)
        val = {k: np.asarray(v)[np.asarray(source.arrays["split"]) == "val"] for k, v in source.arrays.items()}
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd", priors=np.full(len(CLASSES), 1 / len(CLASSES))))
        model.fit(linear_features(train, name), train["y"])
        linear[name] = _class_report(model.predict_proba(linear_features(val, name)), val["y"])
    summary = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v6 (diagnostic, validation only)", "networks": networks, "linear": linear,
               "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "diagnostic.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


# --------------------------------------------------------------------------------------------------
# DS2 and the external databases
# --------------------------------------------------------------------------------------------------


def _predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
    x = torch.from_numpy(np.asarray(arrays["X"], dtype=np.float32))
    side = torch.from_numpy(np.stack([np.asarray(arrays[f], dtype=np.float32) for f in SIDE_FIELDS], axis=1))
    return [predict_proba(load_run_classifier(run_id, True), x, side, torch.device("cpu")) for run_id in run_ids]


def test_arrays(spec: RepresentationSpec) -> dict:
    """The frozen DS2 beats, read under the posthoc access: from data/processed/v1 when present, else rebuilt from the raw
    DS2 records, and checked against the frozen hash either way."""
    from ecgaug.data.loaders import load_test_posthoc

    require_test_access("posthoc")
    if not logged_accesses("evaluation"):
        raise PreconditionError("Post-hoc analyses run only after the final evaluation")
    purpose = f"{EXPERIMENT}: DS2 predictions with v3 beats (and the v1, v2 checkpoints for the paired differences)"
    if test_arrays_present("v1"):
        arrays = load_test_posthoc("v1", purpose, run_ids=f"results/{EXPERIMENT}/selection.json")
    else:
        arrays, _ = build_arrays(sorted(DS2), spec, {r: "test" for r in DS2}, access="posthoc")
        log_test_access("posthoc", f"{purpose} (rebuilt from the raw DS2 records)", run_ids=f"results/{EXPERIMENT}/selection.json")
    if content_sha256(arrays) != frozen_test_sha256(workflow.manifest()[PHASE1_GATE]["files"]):
        raise PreconditionError("The DS2 beats do not match the frozen test set (results/manifest.json)")
    return arrays


def test_report(cfg: dict) -> dict:
    """DS2 with v3 beats: the §7 analysis, v3 minus v1 and v3 minus v2 per condition (one posthoc access)."""
    from ecgaug import final_eval

    selection = load_selection()
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = test_arrays(spec)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, cfg["half_window_s"], access="posthoc").items()})
    v2_test = v5.substitute(v1_test, ratios)
    v3_test = relative(v2_test, mitbih_templates(v1_test, spec, cfg["half_window_s"], access="posthoc"))
    y, records = v1_test["y"], v1_test["record"]
    preds = {(a, c): _predict(chosen["run_ids"], v3_test) for a, conds in selection["chosen"].items() for c, chosen in conds.items()}
    stats = final_eval.statistics(preds, y, records, selection)
    final = json.loads((paths.results_for("final_eval") / "summary.json").read_text())
    earlier = {"v1": (workflow.manifest()[PHASE4_GATE]["chosen"], v1_test), "v2": (v5.load_selection()["chosen"], v2_test)}
    versus, reproduced = {}, {}
    for version, (chosen, arrays) in earlier.items():
        for (arch, condition), probs in preds.items():
            old = _predict(chosen[arch][condition]["run_ids"], arrays)
            unique, new_cm = final_eval.per_record_cms(probs, y, records)
            _, old_cm = final_eval.per_record_cms(old, y, records)
            key = f"{arch}/{condition}"
            versus.setdefault(version, {})[key] = paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, new_cm, old_cm, n=ref.N_BOOT, seed=ref.BOOT_SEED)
            if version == "v1":  # the recomputed v1 scores must equal the final evaluation's
                reproduced[key] = abs(float(np.mean(M.rare_macro_f1_from_cm(old_cm.sum(axis=1)))) - final["conditions"][key]["rare_macro_f1"]["point"]) < 1e-9
    if not all(reproduced.values()):
        raise PreconditionError(f"Recomputed v1 test scores differ from results/final_eval: {[k for k, v in reproduced.items() if not v]}")
    run_ids = [r for a in selection["chosen"].values() for c in a.values() for r in c["run_ids"]]
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v6",
               **{k: stats[k] for k in ("records", "n_boot", "boot_seed", "conditions", "families", "sensitivity")},
               "versus_v1": versus["v1"], "versus_v2": versus["v2"], "v1_reproduced": True, "run_ids": run_ids, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def external_templates(db: str, arrays: dict, spec: RepresentationSpec, half_window_s: float) -> np.ndarray:
    """Templates of an external database's beats, from its signal resampled to 360 Hz and all its QRS annotations."""
    from ecgaug.experiments import phase6_external_validation as ext

    out = np.empty_like(np.asarray(arrays["X"]), dtype=np.float32)
    records = np.asarray(arrays["record"])
    wanted = set(records.tolist())
    for name in ext.record_names(db):
        number = ext.record_number(name)
        if number not in wanted:
            continue
        samples, symbols, lead_signal, fs, _ = raw.read_external(db, name, ext.DATABASES[db]["lead"])
        signal = ext.resample_to_fs(np.nan_to_num(lead_signal), fs)
        r = np.rint(samples[np.isin(symbols, sorted(QRS_SYMBOLS))] * FS / fs).astype(np.int64)
        idx = np.where(records == number)[0]
        order = idx[np.argsort(np.asarray(arrays["r_idx"])[idx])]
        out[order] = signal_templates(signal, r, np.asarray(arrays["r_idx"])[order], spec, half_window_s)
    return out


def external_report(cfg: dict, db: str) -> dict:
    """INCART or SVDB with v3 beats: the v3 checkpoints, the confirmatory family with Holm (P2 rules)."""
    from ecgaug import final_eval
    from ecgaug.experiments import phase6_external_validation as ext

    selection = load_selection()
    spec = load_data(check_preconditions(), preview=False).spec
    arrays, data = ext.load_database(db)
    arrays = v5.substitute(arrays, v5.external_ratios(db, arrays, cfg["half_window_s"]))
    arrays = relative(arrays, external_templates(db, arrays, spec, cfg["half_window_s"]))
    y, records = arrays["y"], arrays["record"]
    cms = {}
    for arch, conds in selection["chosen"].items():
        for condition, chosen in conds.items():
            unique, cms[(arch, condition)] = final_eval.per_record_cms(_predict(chosen["run_ids"], arrays), y, records)
    metric = data["primary_metric"]
    stat = final_eval.STATISTICS[metric]
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": selection["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    family = {f"{arch}: {a} vs {b}": paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, a)], cms[(arch, b)], n=ref.N_BOOT, seed=ref.BOOT_SEED)
              for arch, a, b in final_eval.comparisons(selection)["confirmatory"]}  # fmt: skip
    adjusted = holm({k: v["p_two_sided"] for k, v in family.items()})
    for key, value in family.items():
        value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"])
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v6 (P2 rules)", "database": db, "role": data["role"],
               "primary_metric": metric, "beats": data["beats"], "records": len(unique), "conditions": conditions, "confirmatory": family,
               "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["data", "grid", "diagnostic", "select", "test", "external"], required=True)
    parser.add_argument("--database", choices=["incartdb", "svdb", "all"], default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "data":
            _, record = build(cfg)
            print(f"v3 built: {record['trainval_sha256'][:12]}")
        elif args.part == "grid":
            data, _ = build(cfg)
            print(json.dumps(run_grid(cfg, data, resolve_device(args.device)), indent=2, default=float))
        elif args.part == "diagnostic":
            s = diagnostic(cfg)
            for arch, versions in s["networks"].items():
                for version, g in versions.items():
                    print(f"  {arch} {version}: S PR-AUC train {g['train']['pr_auc']['S']:.3f} val {g['val']['pr_auc']['S']:.3f}; "
                          f"V PR-AUC train {g['train']['pr_auc']['V']:.3f} val {g['val']['pr_auc']['V']:.3f}")  # fmt: skip
            for name, r in s["linear"].items():
                print(f"  LDA {name:12s} val S F1 {r['f1']['S']:.3f}, S PR-AUC {r['pr_auc']['S']:.3f}, V PR-AUC {r['pr_auc']['V']:.3f}")
        elif args.part == "select":
            s = select(cfg)
            for arch, conditions in s["chosen"].items():
                print(arch, {c: v["strength"] for c, v in conditions.items()}, "best generative:", s["best_generative"].get(arch))
        elif args.part == "test":
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:32s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            for key, c in s["conditions"].items():
                d1, d2 = s["versus_v1"][key], s["versus_v2"][key]
                print(f"  {key:28s} {c['rare_macro_f1']['point']:.3f}  v3 - v1 {d1['point']:+.3f} [{d1['ci_low']:+.3f}, {d1['ci_high']:+.3f}]  v3 - v2 {d2['point']:+.3f}")
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
