"""phase6_more_patients: the P7 cnn trained on more patients (PREREGISTRATION §10 v10, P11; post hoc).

    python -m ecgaug.experiments.phase6_more_patients --config configs/phase6_more_patients/more_patients.yaml --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_more_patients --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_more_patients --config ... --part test      # one posthoc access

Training and validation beats: DS1 in v3 form (v6's dataset, hash-checked) plus every record of INCART and SVDB in v3 form,
keeping every S, V and F beat and at most ``n_cap_per_record`` N beats per external record. External patients (INCART's
headers name them; SVDB records stand for patients) are sorted and every ``validation_every``-th goes to validation. The
P7 cnn's settings and strengths are reused unchanged; SMOTE is left out. DS2 stays the only test set.
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
from ecgaug.data.contract import content_sha256
from ecgaug.data.records import CLASS_INDEX, CLASSES, DS2
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_relative_cnn as p7
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.grid import runner
from ecgaug.names import PHASE6_MORE_PATIENTS
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_MORE_PATIENTS
CONFIG_KEYS = {"experiment", "patient_template_config", "relative_cnn_config", "n_cap_per_record", "validation_every", "cap_seed"}
DATABASES = ("incartdb", "svdb")
CONDITIONS = ("none", "class_weighted", "random_oversampling", "cvae", "wgan_gp")
GRID_NAME = "full_grid"
FIELDS = ("X", "y", "record", "split", "r_idx", "rr_prev", "rr_next", "rr_ratio", "is_synthetic", "source")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    root = paths.repo_root()
    return {**cfg, "file": str(path), "pt": pt.load_config(root / cfg["patient_template_config"]), "p7": p7.load_config(root / cfg["relative_cnn_config"])}


# --------------------------------------------------------------------------------------------------
# data: DS1 v3 plus the external patients (DS1 and the external databases only)
# --------------------------------------------------------------------------------------------------


def validation_units(units: np.ndarray, every: int) -> set[int]:
    """Every ``every``-th unit (patient or record) in sorted order goes to validation: positions every, 2 * every, ..."""
    ordered = sorted(set(np.asarray(units).tolist()))
    return set(ordered[every - 1 :: every])


def cap_normal(y: np.ndarray, records: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """A mask keeping every non-N beat and at most ``cap`` N beats per record (chosen at random with ``seed``), in the original order."""
    rng = np.random.default_rng(seed)
    keep = np.asarray(y) != CLASS_INDEX["N"]
    for record in np.unique(records):
        n = np.flatnonzero((np.asarray(records) == record) & (np.asarray(y) == CLASS_INDEX["N"]))
        keep[n if n.size <= cap else np.sort(rng.choice(n, size=cap, replace=False))] = True
    return keep


def external_v3(db: str, spec, half_window_s: float) -> dict:
    """Every beat of an external database in v3 form (v2 ratio from the annotations, patient-relative beat)."""
    from ecgaug.experiments import phase6_external_validation as ext

    arrays, _ = ext.load_database(db)
    arrays = v5.substitute(arrays, v5.external_ratios(db, arrays, half_window_s))
    return pt.relative(arrays, pt.external_templates(db, arrays, spec, half_window_s))


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def build(cfg: dict) -> tuple[DataSource, dict]:
    from ecgaug.experiments import phase6_external_validation as ext

    v3, _ = pt.build(cfg["pt"])  # DS1 in v3 form and v6's banks, hash-checked against v6's data.json
    half = cfg["pt"]["half_window_s"]
    parts, report = [{k: np.asarray(v3.arrays[k]) for k in FIELDS}], {}
    for db in DATABASES:
        arrays = external_v3(db, v3.spec, half)
        units = ext.clusters(db, arrays)
        val_units = validation_units(units, int(cfg["validation_every"]))
        keep = cap_normal(arrays["y"], arrays["record"], int(cfg["n_cap_per_record"]), int(cfg["cap_seed"]))
        part = {k: np.asarray(arrays[k])[keep] for k in FIELDS}
        part["split"] = np.where(np.isin(units[keep], sorted(val_units)), "val", "train").astype("<U5")
        parts.append(part)
        report[db] = {"validation_units": sorted(int(u) for u in val_units), "units": int(len(np.unique(units))),
                      "beats": {s: {c: int(((part["split"] == s) & (part["y"] == i)).sum()) for i, c in enumerate(CLASSES)} for s in ("train", "val")}}  # fmt: skip
    combined = {k: np.concatenate([np.asarray(p[k]).astype(parts[0][k].dtype) if k not in ("split", "source") else np.asarray(p[k]).astype(str) for p in parts]) for k in FIELDS}
    sha = content_sha256(combined)
    record = data_record()
    if record is not None and record["trainval_sha256"] != sha:
        raise PreconditionError(f"P11 training data {sha[:12]}... does not match data.json ({record['trainval_sha256'][:12]}...)")
    if record is None:
        ds1 = {s: {c: int(((parts[0]["split"] == s) & (parts[0]["y"] == i)).sum()) for i, c in enumerate(CLASSES)} for s in ("train", "val")}
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v10", "trainval_sha256": sha, "v3_sha256": v3.info["content_sha256"],
                  "n_cap_per_record": cfg["n_cap_per_record"], "validation_every": cfg["validation_every"], "ds1": ds1, "external": report,
                  "git_sha": runs.git_sha()}  # fmt: skip
        out = paths.results_for(EXPERIMENT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    info = {"source": "DS1 v3 plus INCART and SVDB patients (PREREGISTRATION v10)", "representation": v3.spec.name, "content_sha256": sha, "frozen": False, "version": "v3+external"}
    return DataSource(combined, v3.spec, info), record


# --------------------------------------------------------------------------------------------------
# the grid and the (trivial) selection
# --------------------------------------------------------------------------------------------------


def more_grid(cfg: dict) -> dict:
    """v6's grid for the cnn with P7's chosen strengths only (one per condition), under this experiment's name."""
    base = pt.v3_grid(cfg["pt"])
    chosen = p7.load_selection()["chosen"]["cnn"]
    conditions = {"none": {}, "class_weighted": {"gammas": [chosen["class_weighted"]["strength"]]},
                  "random_oversampling": {"strengths": [chosen["random_oversampling"]["strength"]]},
                  **{g: {"bank": base["conditions"][g]["bank"], "strengths": [chosen[g]["strength"]]} for g in ("cvae", "wgan_gp")}}  # fmt: skip
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "architectures": {"cnn": base["architectures"]["cnn"]}, "conditions": conditions, "file": cfg["file"]}


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    return pt._run_cells(more_grid(cfg), data, device, lambda cell: True)


def select(cfg: dict) -> dict:
    grid = more_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v10 (fixed strengths)", "chosen": chosen, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def _family(items: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    out = {k: paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, a, b, n=ref.N_BOOT, seed=ref.BOOT_SEED) for k, (a, b) in items.items()}
    adjusted = holm({k: v["p_two_sided"] for k, v in out.items()})
    for k, v in out.items():
        v.update(p_holm=adjusted[k]["p_adjusted"], reject_holm=adjusted[k]["reject"])
    return out


def test_report(cfg: dict) -> dict:
    from ecgaug import final_eval

    selection, before = load_selection()["chosen"]["cnn"], p7.load_selection()["chosen"]["cnn"]
    half = cfg["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    test = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    y, records = test["y"], test["record"]
    cms, old, unique = {}, {}, None
    for condition in CONDITIONS:
        unique, cms[condition] = final_eval.per_record_cms(pt._predict(selection[condition]["run_ids"], test), y, records)
        _, old[condition] = final_eval.per_record_cms(pt._predict(before[condition]["run_ids"], test), y, records)
    p7_summary = json.loads((paths.results_for("phase6_relative_cnn") / "summary.json").read_text())
    reproduced = float(np.mean(M.rare_macro_f1_from_cm(old["none"].sum(axis=1)))) - p7_summary["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v10", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED,
               "conditions": {f"cnn/{c}": ref.condition_summary(cm, unique) | {"strength": selection[c]["strength"]} for c, cm in cms.items()},
               "families": {"confirmatory": _family({"cnn: none, more patients vs DS1 only (P7)": (cms["none"], old["none"])}),
                            "secondary": _family({f"cnn: {c} vs none (more patients)": (cms[c], cms["none"]) for c in CONDITIONS if c != "none"})},
               "versus_p7": {f"cnn/{c}": paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, cms[c], old[c], n=ref.N_BOOT, seed=ref.BOOT_SEED) for c in CONDITIONS},
               "p7_none_reproduction_difference": reproduced, "run_ids": [r for c in CONDITIONS for r in selection[c]["run_ids"]], "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["data", "grid", "select", "test"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "data":
            _, record = build(cfg)
            print(json.dumps({"ds1": record["ds1"], "external": {db: e["beats"] for db, e in record["external"].items()}}, indent=1))
        elif args.part == "grid":
            data, record = build(cfg)
            print(json.dumps({db: e["beats"] for db, e in record["external"].items()}))
            print(json.dumps(run_grid(cfg, data, resolve_device(args.device)), indent=2))
        elif args.part == "select":
            s = select(cfg)
            print({c: (v["strength"], round(v["mean"], 3)) for c, v in s["chosen"]["cnn"].items()})
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:44s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            for key, c in s["conditions"].items():
                d = s["versus_p7"][key]
                print(f"  {key:26s} rare-class macro-F1 {c['rare_macro_f1']['point']:.3f}  S F1 {c['f1_S']['point']:.3f}  F F1 {c['f1_F']['point']:.3f}  vs P7 {d['point']:+.3f}")
            print("  P7 none reproduced (difference):", s["p7_none_reproduction_difference"])
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
