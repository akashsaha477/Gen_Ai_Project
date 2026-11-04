"""phase6_audit_check: two checks of the leakage audit (PREREGISTRATION §10 v4, P4; post hoc).

    ALLOW_TEST_AUDIT=1 python -m ecgaug.experiments.phase6_audit_check [--device auto]

(a) The audit's own split (split seed 0) and banks, with WGAN-GP at the strength the phase 4 gate chose for the cnn
(the audit used the grid's middle strength). (b) No augmentation on two more random beat-level splits (split seeds 1
and 2), to see whether the inflation depends on the split. Seeds 0-2, the cnn of the phase 2 gate, the audit's
policy. Each DS2 read is an audit access; runs go under ``runs/audit/``; nothing feeds a decision. Results:
``results/phase6_audit_check/summary.json``.
"""

from __future__ import annotations

import argparse
import json

import torch

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.splits import Split
from ecgaug.experiments import phase4_leakage_audit as audit
from ecgaug.experiments.common import PreconditionError, check_preconditions, mean_sd, split_part
from ecgaug.experiments.data_source import load_data
from ecgaug.grid import runner
from ecgaug.names import FINAL_EVAL, PHASE4_GATE, PHASE6_AUDIT_CHECK
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_AUDIT_CHECK
SEEDS = (0, 1, 2)
EXTRA_SPLITS = (1, 2)
ARCH = "cnn"


def checks() -> list[dict]:
    """The pre-registered cells: (split seed, condition, strength)."""
    chosen = workflow.manifest()[PHASE4_GATE]["chosen"][ARCH]["wgan_gp"]["strength"]
    return [{"split_seed": 0, "condition": "wgan_gp", "strength": chosen}, *({"split_seed": s, "condition": "none", "strength": None} for s in EXTRA_SPLITS)]


def run(device: torch.device) -> dict:
    cfg = {"architecture": ARCH, "conditions": ["none", "wgan_gp"], "seeds": list(SEEDS), "strength": "medium", "split_seed": 0}
    ctx = audit.frozen_context(cfg)
    split = check_preconditions()
    frozen = load_data(split, preview=False)
    test = audit.test_arrays(frozen.spec, f"{EXPERIMENT}: checks of the audit")
    audit_split = Split(name="intra_patient_random", train=(), val=(), selection_metric=split.selection_metric)
    results = []
    for check in checks():
        data, held, info = audit.make_source(frozen, test, check["split_seed"])
        part = split_part(held, "test", data.spec.rr_features)
        grid = {"experiment": EXPERIMENT, "name": f"split{check['split_seed']}_{check['condition']}", "policy": ctx["policy"], "seeds": list(SEEDS),
                "architectures": {ARCH: "phase2_gate"}, "file": "src/ecgaug/experiments/phase6_audit_check.py",
                "conditions": {"none": {}} if check["condition"] == "none" else {"wgan_gp": {"bank": f"audit/{ctx['generators']['wgan_gp']}", "strengths": [check["strength"]]}}}  # fmt: skip
        ledger = runner.read_ledger(grid)
        rows = []
        for seed in SEEDS:
            cell = runner.Cell(check["condition"], check["strength"], ARCH, seed)
            bank, bank_sha, bank_id = None, None, None
            if check["condition"] == "wgan_gp":
                bank, bank_sha = audit.load_audit_bank(ctx["generators"]["wgan_gp"], seed)
                bank_id = grid["conditions"]["wgan_gp"]["bank"]
            sha = runner.cell_sha(cell, grid, ctx["classifier"], data, bank_sha)
            if runner.is_done(ledger.get(sha)):
                rows.append(ledger[sha])
                continue
            row = runner.run_cell(cell, grid, audit_split, data, ctx["classifier"], device, sha, bank, bank_id, [], run_root=audit.RUN_ROOT)
            row["audit_test"] = audit.evaluate_on_test(row["run_id"], part, data.spec.rr_features)
            runner.write_cell(grid, row)
            rows.append(row)
        values = [r["audit_test"]["rare_macro_f1"] for r in rows]
        mean, sd = mean_sd(values)
        results.append({**check, "split": info, "intra": {"mean": mean, "sd": sd, "per_seed": values, "run_ids": [r["run_id"] for r in rows]}})
    final = json.loads((paths.results_for(FINAL_EVAL) / "summary.json").read_text())
    audit_summary = json.loads((paths.results_for(audit.EXPERIMENT) / "summary.json").read_text())
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v4, P4", "architecture": ARCH, "checks": results,
               "audit_split0": {c: v["intra"] for c, v in audit_summary["conditions"].items()},
               "inter_patient": {c: final["conditions"][f"{ARCH}/{c}"]["rare_macro_f1"] for c in ("none", "wgan_gp")}, "git_sha": runs.git_sha()}  # fmt: skip
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        summary = run(resolve_device(args.device))
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    for c in summary["checks"]:
        print(f"split {c['split_seed']} {c['condition']} (strength {c['strength']}): intra-patient {c['intra']['mean']:.3f} ± {c['intra']['sd']:.3f}")
    print(f"written: results/{EXPERIMENT}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
