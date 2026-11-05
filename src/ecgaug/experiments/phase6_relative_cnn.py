"""phase6_relative_cnn: the cnn alone with patient-relative beats, the full comparison (PREREGISTRATION §10 v7, P7; post hoc).

    python -m ecgaug.experiments.phase6_relative_cnn --config configs/phase6_relative_cnn/relative_cnn.yaml --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_relative_cnn --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_relative_cnn --config ... --part test      # one posthoc access
    python -m ecgaug.experiments.phase6_relative_cnn --config ... --part external [--database incartdb]

v6's adoption rule averaged over the cnn and the rnn and did not adopt dataset v3; on validation v3 helped the cnn and
hurt the rnn. P7 takes the cnn with v3 inputs (chosen on validation after v6's result, and reported as such) and runs the
phase 4a gate's full grid for the cnn alone on dataset v3 and v6's banks (48 runs), then repeats v6's selection, test
and external parts with the same code. Nothing here changes a frozen artifact or an earlier result.
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
from ecgaug.data.records import DS2
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.grid import runner
from ecgaug.names import PHASE4_GATE, PHASE6_RELATIVE_CNN
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_RELATIVE_CNN
ARCHITECTURE = "cnn"
CONFIG_KEYS = {"experiment", "patient_template_config"}
GRID_NAME = "full_grid"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    return {**cfg, "file": str(path), "pt": pt.load_config(paths.repo_root() / cfg["patient_template_config"])}


def cnn_grid(cfg: dict) -> dict:
    """v6's grid (the phase 4a gate's full grid on v3 with v6's banks) for the cnn alone, under this experiment's name."""
    grid = pt.v3_grid(cfg["pt"])
    return {**grid, "experiment": EXPERIMENT, "name": GRID_NAME, "architectures": {ARCHITECTURE: grid["architectures"][ARCHITECTURE]}, "file": cfg["file"]}


def run_grid(cfg: dict, data: DataSource, device: torch.device) -> dict:
    return pt._run_cells(cnn_grid(cfg), data, device, lambda cell: True)


def select(cfg: dict) -> dict:
    """The phase 4 gate rule, with the same code, on the 48 cells; written before any DS2 read."""
    grid = cnn_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, best = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v7 (the phase 4 gate rule)", "chosen": chosen, "best_generative": best, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Apply the selection first: --part select")
    return json.loads(path.read_text())


def test_report(cfg: dict) -> dict:
    """DS2 with v3 beats for the cnn: the §7 analysis, P7 minus v1 and P7 minus v2 per condition (one posthoc access)."""
    from ecgaug import final_eval

    selection = load_selection()
    half = cfg["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    v2_test = v5.substitute(v1_test, ratios)
    v3_test = pt.relative(v2_test, pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    y, records = v1_test["y"], v1_test["record"]
    preds = {(a, c): pt._predict(chosen["run_ids"], v3_test) for a, conds in selection["chosen"].items() for c, chosen in conds.items()}
    stats = final_eval.statistics(preds, y, records, selection)
    final = json.loads((paths.results_for("final_eval") / "summary.json").read_text())
    earlier = {"v1": (workflow.manifest()[PHASE4_GATE]["chosen"], v1_test), "v2": (v5.load_selection()["chosen"], v2_test)}
    versus, reproduced = {}, {}
    for version, (chosen, arrays) in earlier.items():
        for (arch, condition), probs in preds.items():
            old = pt._predict(chosen[arch][condition]["run_ids"], arrays)
            _, new_cm = final_eval.per_record_cms(probs, y, records)
            _, old_cm = final_eval.per_record_cms(old, y, records)
            key = f"{arch}/{condition}"
            versus.setdefault(version, {})[key] = paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, new_cm, old_cm, n=ref.N_BOOT, seed=ref.BOOT_SEED)
            if version == "v1":
                reproduced[key] = abs(float(np.mean(M.rare_macro_f1_from_cm(old_cm.sum(axis=1)))) - final["conditions"][key]["rare_macro_f1"]["point"]) < 1e-9
    if not all(reproduced.values()):
        raise PreconditionError(f"Recomputed v1 test scores differ from results/final_eval: {[k for k, v in reproduced.items() if not v]}")
    run_ids = [r for a in selection["chosen"].values() for c in a.values() for r in c["run_ids"]]
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v7",
               **{k: stats[k] for k in ("records", "n_boot", "boot_seed", "conditions", "families", "sensitivity")},
               "versus_v1": versus["v1"], "versus_v2": versus["v2"], "v1_reproduced": True, "run_ids": run_ids, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def external_report(cfg: dict, db: str) -> dict:
    """INCART or SVDB with v3 beats: the 18 selected cnn checkpoints, the confirmatory family with Holm (P2 rules)."""
    from ecgaug import final_eval
    from ecgaug.experiments import phase6_external_validation as ext

    selection = load_selection()
    half = cfg["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    arrays, data = ext.load_database(db)
    arrays = v5.substitute(arrays, v5.external_ratios(db, arrays, half))
    arrays = pt.relative(arrays, pt.external_templates(db, arrays, spec, half))
    y, records = arrays["y"], ext.clusters(db, arrays)  # bootstrap over patients, not records
    cms = {}
    for arch, conds in selection["chosen"].items():
        for condition, chosen in conds.items():
            unique, cms[(arch, condition)] = final_eval.per_record_cms(pt._predict(chosen["run_ids"], arrays), y, records)
    metric = data["primary_metric"]
    stat = final_eval.STATISTICS[metric]
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": selection["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    family = {f"{arch}: {a} vs {b}": paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, a)], cms[(arch, b)], n=ref.N_BOOT, seed=ref.BOOT_SEED)
              for arch, a, b in final_eval.comparisons(selection)["confirmatory"]}  # fmt: skip
    adjusted = holm({k: v["p_two_sided"] for k, v in family.items()})
    for key, value in family.items():
        value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"])
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v7 (P2 rules)", "database": db, "role": data["role"],
               "primary_metric": metric, "beats": data["beats"], "records": int(len(np.unique(arrays["record"]))), "patients": len(unique), "clustered_by": "patient", "conditions": conditions, "confirmatory": family,
               "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["grid", "select", "test", "external"], required=True)
    parser.add_argument("--database", choices=["incartdb", "svdb", "all"], default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "grid":
            data, _ = pt.build(cfg["pt"])  # dataset v3 and v6's banks, rebuilt and hash-checked
            print(json.dumps(run_grid(cfg, data, resolve_device(args.device)), indent=2))
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
                print(f"  {key:28s} {c['rare_macro_f1']['point']:.3f}  P7 - v1 {d1['point']:+.3f} [{d1['ci_low']:+.3f}, {d1['ci_high']:+.3f}]  P7 - v2 {d2['point']:+.3f}")
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
