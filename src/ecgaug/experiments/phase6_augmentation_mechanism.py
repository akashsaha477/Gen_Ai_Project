"""phase6_augmentation_mechanism: why augmentation does not carry over to new patients (PREREGISTRATION §10 v7, P8; post hoc).

    python -m ecgaug.experiments.phase6_augmentation_mechanism --config configs/phase6_augmentation_mechanism/mechanism.yaml --part attribution
    python -m ecgaug.experiments.phase6_augmentation_mechanism --config ... --part gap

DS1 only, descriptive, nothing chosen. ``attribution``: each synthetic beat of the frozen banks (and each beat the policy
adds at the phase 4 gate's strengths) is attributed to the training record of its nearest real training beat of the
same class (Euclidean distance between z-scored windows), beside the same attribution for real training beats (their
nearest other real beat) and for validation beats; per class, the share of each record and the effective number of
records, exp(entropy). ``gap``: for every condition of the final evaluation, the seed-mean S and V PR-AUC of the frozen
networks on the real training beats and on the validation beats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data.records import CLASSES, CLASS_INDEX
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.generators.bank import load_bank
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE4_GATE, PHASE6_AUGMENTATION_MECHANISM

EXPERIMENT = PHASE6_AUGMENTATION_MECHANISM
CONFIG_KEYS = {"experiment"}
RARE = ("S", "V", "F")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path)}


def nearest(reference: np.ndarray, queries: np.ndarray, exclude_self: bool = False, batch: int = 4096) -> np.ndarray:
    """Index of the nearest ``reference`` row (Euclidean) for every query row; with ``exclude_self`` the queries are the
    reference rows themselves and a row is never its own neighbour."""
    ref = torch.from_numpy(np.asarray(reference, dtype=np.float32))
    out = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), batch):
        q = torch.from_numpy(np.asarray(queries[start : start + batch], dtype=np.float32))
        d = torch.cdist(q, ref)
        if exclude_self:
            rows = torch.arange(len(q))
            d[rows, rows + start] = float("inf")
        out[start : start + len(q)] = d.argmin(dim=1).numpy()
    return out


def record_shares(records: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(np.asarray(records), return_counts=True)
    return {str(int(v)): float(c / counts.sum()) for v, c in sorted(zip(values, counts), key=lambda vc: -vc[1])}


def effective_records(shares: dict[str, float]) -> float:
    """exp(entropy) of the record shares: 1 when one record holds everything, n for n equal records."""
    p = np.array([v for v in shares.values() if v > 0])
    return float(np.exp(-(p * np.log(p)).sum())) if p.size else float("nan")


def _entry(records: np.ndarray, own: np.ndarray | None = None) -> dict:
    shares = record_shares(records)
    entry = {"n": int(len(records)), "shares": shares, "effective_records": effective_records(shares), "top_record": next(iter(shares), None),
             "top_share": next(iter(shares.values()), float("nan"))}  # fmt: skip
    if own is not None:
        entry["same_record"] = float(np.mean(np.asarray(records) == np.asarray(own)))
    return entry


def attribution(cfg: dict) -> dict:
    split = check_preconditions()
    data = load_data(split, preview=False)
    train = training_beats(data.arrays)
    val_mask = np.asarray(data.arrays["split"]) == "val"
    val = {k: np.asarray(v)[val_mask] for k, v in data.arrays.items()}
    gate = workflow.manifest()[PHASE4_GATE]["chosen"]
    grid = runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    out: dict = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v7 (P8 attribution)", "distance": "euclidean, z-scored windows", "classes": {}}
    for cls in RARE:
        i = CLASS_INDEX[cls]
        mask = train["y"] == i
        real_x, real_rec = train["X"][mask], train["record"][mask]
        entry = {"real_training": _entry(real_rec)}
        if mask.sum() > 1:
            entry["real_nearest_other"] = _entry(real_rec[nearest(real_x, real_x, exclude_self=True)], own=real_rec)
        vmask = val["y"] == i
        if vmask.any():
            entry["validation_nearest"] = _entry(real_rec[nearest(real_x, val["X"][vmask])])
        out["classes"][cls] = entry
    for condition, frozen in v5.frozen_generators().items():
        gid = frozen["generator_id"]
        for seed, bank_entry in sorted(frozen["banks"].items()):
            bank = load_bank(gid, int(seed), expected_sha256=bank_entry["sha256"])
            added = {}
            for arch, conds in gate.items():
                cell = runner.Cell(condition, conds[condition]["strength"], arch, int(seed))
                added[arch], _ = runner.added_beats(cell, grid, train, bank)
            for cls in RARE:
                i = CLASS_INDEX[cls]
                mask = train["y"] == i
                real_x, real_rec = train["X"][mask], train["record"][mask]
                parts = {"bank": bank["X"][np.asarray(bank["y"]) == i]}
                parts.update({f"added_{arch}": a["X"][np.asarray(a["y"]) == i] for arch, a in added.items()})
                for name, x in parts.items():
                    if len(x):
                        out["classes"][cls].setdefault(f"{condition}_{name}", {})[str(seed)] = _entry(real_rec[nearest(real_x, x)])
    out["git_sha"] = runs.git_sha()
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "attribution.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def gap(cfg: dict) -> dict:
    split = check_preconditions()
    data = load_data(split, preview=False)
    gate = workflow.manifest()[PHASE4_GATE]["chosen"]
    out: dict = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v7 (P8 gap)", "networks": {}}
    for arch, conds in gate.items():
        for condition, chosen in conds.items():
            g = pt.network_gap(chosen["run_ids"], data)
            g["gap"] = {c: g["train"]["pr_auc"][c] - g["val"]["pr_auc"][c] for c in CLASSES}
            g["strength"] = chosen["strength"]
            out["networks"].setdefault(arch, {})[condition] = g
    out["git_sha"] = runs.git_sha()
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gap.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["attribution", "gap", "all"], required=True)
    parser.add_argument("--device", default="cpu", help="ignored: everything here runs on the CPU")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part in ("attribution", "all"):
            s = attribution(cfg)
            for cls, entry in s["classes"].items():
                line = [f"real {entry['real_training']['effective_records']:.2f} eff. records (top {entry['real_training']['top_record']} {entry['real_training']['top_share']:.2f})"]
                for key, seeds in entry.items():
                    if key.startswith(("cvae", "wgan_gp")):
                        e = seeds.get("0")
                        if e:
                            line.append(f"{key} s0 {e['effective_records']:.2f} (top {e['top_record']} {e['top_share']:.2f})")
                print(f"  {cls}: " + "; ".join(line))
        if args.part in ("gap", "all"):
            s = gap(cfg)
            for arch, conds in s["networks"].items():
                for condition, g in conds.items():
                    print(f"  {arch} {condition:20s} S PR-AUC train {g['train']['pr_auc']['S']:.3f} val {g['val']['pr_auc']['S']:.3f} gap {g['gap']['S']:+.3f}")
    except PreconditionError as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
