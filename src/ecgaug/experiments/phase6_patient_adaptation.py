"""phase6_patient_adaptation: per-patient batch-normalization statistics at test time (PREREGISTRATION §10 v9, P10; post hoc).

    python -m ecgaug.experiments.phase6_patient_adaptation --config configs/phase6_patient_adaptation/patient_adaptation.yaml --part validation
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_patient_adaptation --config ... --part test      # one posthoc access
    python -m ecgaug.experiments.phase6_patient_adaptation --config ... --part external [--database incartdb]

Adaptive batch normalization (Li et al., 2016) on frozen checkpoints: for each record, a copy of the network recomputes the
statistics of every BatchNorm1d layer from that record's beats (no labels) and classifies them with those statistics.
Nothing is retrained. Versions: the final evaluation's checkpoints (v1 inputs, cnn and rnn) and P7's (cnn, v3 inputs).
``--part validation`` applies the adoption rule on the validation records and writes ``adoption.json``; ``--part test``
reads DS2 once, only for the adopted pairs; ``--part external`` scores INCART and SVDB.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.records import CLASSES, DS2
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_relative_cnn as p7
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, load_run_classifier, mean_sd
from ecgaug.experiments.data_source import load_data
from ecgaug.names import PHASE4_GATE, PHASE6_PATIENT_ADAPTATION

EXPERIMENT = PHASE6_PATIENT_ADAPTATION
CONFIG_KEYS = {"experiment", "half_window_s", "adoption_margin"}
GENERATIVE = ("cvae", "wgan_gp")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path)}


# --------------------------------------------------------------------------------------------------
# adaptation
# --------------------------------------------------------------------------------------------------


def adapted_predict(model: nn.Module, x: np.ndarray, side: np.ndarray, records: np.ndarray) -> np.ndarray:
    """Probabilities with every BatchNorm1d layer normalizing by the statistics of the beat's own record.

    Each record is scored by a fresh copy of ``model``: its batch-norm running statistics are reset, recomputed in one
    forward pass over all of the record's beats (cumulative average of one batch = that batch's statistics), and the
    record is then classified in evaluation mode. ``model`` itself is not changed; no labels are used."""
    out = np.zeros((len(x), len(CLASSES)), dtype=np.float32)
    records = np.asarray(records)
    for record in np.unique(records):
        idx = np.flatnonzero(records == record)
        m = copy.deepcopy(model).eval()
        layers = [layer for layer in m.modules() if isinstance(layer, nn.BatchNorm1d)]
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        xb = torch.from_numpy(np.asarray(x[idx], dtype=np.float32))
        sb = torch.from_numpy(np.asarray(side[idx], dtype=np.float32))
        with torch.no_grad():
            m(xb, sb)
            m.eval()
            out[idx] = torch.softmax(m(xb, sb), dim=1).numpy()
    return out


def _side(arrays: dict) -> np.ndarray:
    return np.stack([np.asarray(arrays[f], dtype=np.float32) for f in SIDE_FIELDS], axis=1)


def predict(run_ids: list[str], arrays: dict, adapt: bool) -> list[np.ndarray]:
    if not adapt:
        return pt._predict(run_ids, arrays)
    x, side = np.asarray(arrays["X"], dtype=np.float32), _side(arrays)
    return [adapted_predict(load_run_classifier(run_id, True), x, side, arrays["record"]) for run_id in run_ids]


def versions(cfg: dict) -> dict[str, dict]:
    """The checkpoints each version scores, and the input they take."""
    return {"v1": {"chosen": workflow.manifest()[PHASE4_GATE]["chosen"], "input": "v1"},
            "p7": {"chosen": p7.load_selection()["chosen"], "input": "v3"}}  # fmt: skip


def _subset(arrays: dict, mask: np.ndarray) -> dict:
    return {k: np.asarray(v)[mask] for k, v in arrays.items()}


# --------------------------------------------------------------------------------------------------
# adoption on validation
# --------------------------------------------------------------------------------------------------


def validation(cfg: dict) -> dict:
    split = check_preconditions()
    data = load_data(split, preview=False)
    val_mask = np.asarray(data.arrays["split"]) == "val"
    v1_val = _subset(data.arrays, val_mask)
    v2_val = _subset(pt._v2_arrays(data, cfg["half_window_s"]), val_mask)
    inputs = {"v1": v1_val, "v3": pt.relative(v2_val, pt.mitbih_templates(v2_val, data.spec, cfg["half_window_s"]))}
    y = v1_val["y"]
    pairs = {}
    for version, spec in versions(cfg).items():
        arrays = inputs[spec["input"]]
        for arch, conds in spec["chosen"].items():
            run_ids = conds["none"]["run_ids"]
            scores = {}
            for adapt in (False, True):
                metrics = [M.evaluate(p, y) for p in predict(run_ids, arrays, adapt)]
                scores["adapted" if adapt else "plain"] = {"f1_S": [m["per_class"]["S"]["f1"] for m in metrics],
                                                           "rare_macro_f1": [m["rare_macro_f1"] for m in metrics]}  # fmt: skip
            (m_p, sd_p), (m_a, sd_a) = mean_sd(scores["plain"]["f1_S"]), mean_sd(scores["adapted"]["f1_S"])
            margin = max(float(cfg["adoption_margin"]), float(np.sqrt((sd_p**2 + sd_a**2) / 2)))
            pairs[f"{version}/{arch}"] = {"run_ids": run_ids, "plain_mean": m_p, "plain_sd": sd_p, "adapted_mean": m_a, "adapted_sd": sd_a,
                                          "difference": m_a - m_p, "margin": margin, "adopted": (m_a - m_p) > margin, "per_seed": scores}  # fmt: skip
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v9 (adoption)", "pairs": pairs, "git_sha": runs.git_sha()}
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adoption.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_adoption() -> dict:
    path = paths.results_for(EXPERIMENT) / "adoption.json"
    if not path.is_file():
        raise PreconditionError("Apply the adoption rule first: --part validation")
    return json.loads(path.read_text())


def adopted_pairs() -> list[str]:
    return [key for key, entry in load_adoption()["pairs"].items() if entry["adopted"]]


# --------------------------------------------------------------------------------------------------
# DS2 and the external databases
# --------------------------------------------------------------------------------------------------


def _families(cms: dict, plain_none: dict, adopted: list[str], stat) -> dict:
    families = {"adaptation": {f"{key}: none adapted vs plain": paired_cluster_bootstrap_diff_confusion(stat, cms[(key, "none")], plain_none[key], n=ref.N_BOOT, seed=ref.BOOT_SEED)
                               for key in adopted},
                "augmentation_under_adaptation": {f"{key}: {g} vs none": paired_cluster_bootstrap_diff_confusion(stat, cms[(key, g)], cms[(key, "none")], n=ref.N_BOOT, seed=ref.BOOT_SEED)
                                                  for key in adopted for g in GENERATIVE if (key, g) in cms}}  # fmt: skip
    for items in families.values():
        adjusted = holm({k: v["p_two_sided"] for k, v in items.items()})
        for k, v in items.items():
            v.update(p_holm=adjusted[k]["p_adjusted"], reject_holm=adjusted[k]["reject"])
    return families


def _score(cfg: dict, inputs: dict, adopted: list[str], stat) -> tuple[dict, dict, dict, np.ndarray]:
    from ecgaug import final_eval

    cms, plain_none, unique = {}, {}, None
    vs = versions(cfg)
    for key in adopted:
        version, arch = key.split("/")
        arrays = inputs[vs[version]["input"]]
        y, records = arrays["y"], arrays["record"]
        for condition, chosen in vs[version]["chosen"][arch].items():
            unique, cms[(key, condition)] = final_eval.per_record_cms(predict(chosen["run_ids"], arrays, True), y, records)
        _, plain_none[key] = final_eval.per_record_cms(predict(vs[version]["chosen"][arch]["none"]["run_ids"], arrays, False), y, records)
    conditions = {f"{key}/{c}": ref.condition_summary(cm, unique) for (key, c), cm in cms.items()}
    return cms, plain_none, conditions, unique


def test_report(cfg: dict) -> dict:
    adopted = adopted_pairs()
    if not adopted:
        raise PreconditionError("No pair was adopted on validation: DS2 is not read (PREREGISTRATION v9)")
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    inputs = {"v1": v1_test}
    if any(key.startswith("p7/") for key in adopted):
        ratios = {}
        for record in DS2:
            ratios.update({(record, r): v for r, v in v5.record_ratios(record, cfg["half_window_s"], access="posthoc").items()})
        v2_test = v5.substitute(v1_test, ratios)
        inputs["v3"] = pt.relative(v2_test, pt.mitbih_templates(v1_test, spec, cfg["half_window_s"], access="posthoc"))
    stat = M.rare_macro_f1_from_cm
    cms, plain_none, conditions, unique = _score(cfg, inputs, adopted, stat)
    earlier = {"v1": json.loads((paths.results_for("final_eval") / "summary.json").read_text()),
               "p7": json.loads((paths.results_for("phase6_relative_cnn") / "summary.json").read_text())}  # fmt: skip
    reproduced = {key: float(np.mean(stat(plain_none[key].sum(axis=1)))) - earlier[key.split("/")[0]]["conditions"][f"{key.split('/')[1]}/none"]["rare_macro_f1"]["point"]
                  for key in adopted}  # fmt: skip
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v9", "adopted": adopted, "records": [int(r) for r in unique],
               "n_boot": ref.N_BOOT, "boot_seed": ref.BOOT_SEED, "conditions": conditions, "families": _families(cms, plain_none, adopted, stat),
               "plain_none_reproduction_difference": reproduced, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def external_report(cfg: dict, db: str) -> dict:
    from ecgaug import final_eval
    from ecgaug.experiments import phase6_external_validation as ext

    adopted = adopted_pairs()
    if not adopted:
        raise PreconditionError("No pair was adopted on validation")
    spec = load_data(check_preconditions(), preview=False).spec
    arrays, data = ext.load_database(db)
    inputs = {"v1": arrays}
    if any(key.startswith("p7/") for key in adopted):
        v2 = v5.substitute(arrays, v5.external_ratios(db, arrays, cfg["half_window_s"]))
        inputs["v3"] = pt.relative(v2, pt.external_templates(db, v2, spec, cfg["half_window_s"]))
    metric = data["primary_metric"]
    stat = final_eval.STATISTICS[metric]
    cms, plain_none, conditions, unique = _score(cfg, inputs, adopted, stat)
    plain_conditions = {f"{key}/none": ref.condition_summary(cm, unique) for key, cm in plain_none.items()}
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v9 (P2 rules)", "database": db, "primary_metric": metric,
               "adopted": adopted, "records": len(unique), "conditions": conditions, "plain_none": plain_conditions,
               "families": _families(cms, plain_none, adopted, stat), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["validation", "test", "external"], required=True)
    parser.add_argument("--database", choices=["incartdb", "svdb", "all"], default="all")
    parser.add_argument("--device", default="cpu", help="ignored: everything here runs on the CPU")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "validation":
            s = validation(cfg)
            for key, e in s["pairs"].items():
                print(f"  {key:8s} validation S F1 plain {e['plain_mean']:.3f} adapted {e['adapted_mean']:.3f} "
                      f"difference {e['difference']:+.3f} margin {e['margin']:.3f} adopted {e['adopted']}")  # fmt: skip
        elif args.part == "test":
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:30s} {name:34s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            for key, c in s["conditions"].items():
                print(f"  {key:24s} rare-class macro-F1 {c['rare_macro_f1']['point']:.3f}  S F1 {c['f1_S']['point']:.3f}  F F1 {c['f1_F']['point']:.3f}")
            print("  plain none reproduced (difference):", s["plain_none_reproduction_difference"])
        else:
            for db in (["incartdb", "svdb"] if args.database == "all" else [args.database]):
                s = external_report(cfg, db)
                for family, items in s["families"].items():
                    for name, c in items.items():
                        print(f"  {db} {family:30s} {name:34s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
