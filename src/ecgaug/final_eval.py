"""The single test-set evaluation (/final-eval; CLAUDE.md rule 2, PREREGISTRATION §7-8).

    python -m ecgaug.final_eval --plan                                    # what would be evaluated; reads no test data
    ALLOW_TEST=1 python -m ecgaug.final_eval --confirm "run final eval"   # the one evaluation

It runs only after the phase 4 gate is approved, and only once: it refuses if
``docs/TEST_ACCESS_LOG.md`` already holds an ``evaluation`` entry, if ``ALLOW_TEST=1`` is not set, if the
confirmation phrase is not given exactly, or if any frozen checkpoint is missing or changed. Nothing is
trained, selected or tuned here: the checkpoints, strengths and the "best generative" condition were
fixed on validation at the phase 4 gate.

Pre-registered analysis (PREREGISTRATION §7), per architecture:

- every condition: rare-class macro-F1 (primary) and the secondary metrics, with record-level cluster
  bootstrap intervals (10,000 replicates over the 22 DS2 records, seeds averaged inside each replicate);
- confirmatory family (Holm over 4 tests): cVAE vs none, WGAN-GP vs none;
- secondary family (Holm over 8 tests): cVAE vs WGAN-GP, best generative vs SMOTE, vs class-weighted, vs
  random oversampling;
- paired differences use the same resampled records for both conditions; replicates in which S or F
  has no true beats are undefined, excluded and counted; two-sided percentile-bootstrap p-values;
- sensitivity: everything again without record 202 (same subject as DS1 record 201), leave-one-record-out
  point estimates for the records holding S or F beats, and per-record results.

Outputs: ``results/final_eval/`` (summary.json and CSV tables), ``tables/final_eval_*.tex``,
``figures/final_eval/`` and ``paper/generated/numbers_final_eval.tex``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.loaders import load_test
from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import cluster_bootstrap_confusion, holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments.common import SIDE_FIELDS, load_run_classifier
from ecgaug.names import FINAL_EVAL, PHASE4_GATE
from ecgaug.train.loop import predict_proba

CONFIRMATION = "run final eval"
N_BOOT = 10_000
BOOT_SEED = 0
SAME_SUBJECT_RECORD = 202  # DS2 record of the subject of DS1 record 201 (DECISIONS A3)
GENERATIVE = ("cvae", "wgan_gp")
SECONDARY_BASELINES = ("smote", "class_weighted", "random_oversampling")

_i = CLASS_INDEX
STATISTICS = {
    "rare_macro_f1": M.rare_macro_f1_from_cm,
    "macro_f1": M.macro_f1_from_cm,
    "f1_S": lambda cm: M.f1_from_cm(cm)[..., _i["S"]],
    "f1_V": lambda cm: M.f1_from_cm(cm)[..., _i["V"]],
    "f1_F": lambda cm: M.f1_from_cm(cm)[..., _i["F"]],
    "n_recall": lambda cm: M.recall_from_cm(cm)[..., _i["N"]],
    "se_S": lambda cm: M.recall_from_cm(cm)[..., _i["S"]],
    "ppv_S": lambda cm: M.precision_from_cm(cm)[..., _i["S"]],
    "se_V": lambda cm: M.recall_from_cm(cm)[..., _i["V"]],
    "ppv_V": M.aami_ppv_v_from_cm,
}


class FinalEvalError(RuntimeError):
    """The final evaluation cannot run; the message says why."""


def prior_evaluations() -> list[str]:
    """Rows of ``docs/TEST_ACCESS_LOG.md`` whose kind is ``evaluation``."""
    log = paths.docs_dir() / "TEST_ACCESS_LOG.md"
    if not log.is_file():
        return []
    return [line for line in log.read_text().splitlines() if line.startswith("|") and line.split("|")[2].strip() == "evaluation"]


def comparisons(gate: dict) -> dict[str, list[tuple[str, str, str]]]:
    """The pre-registered comparisons per family: (architecture, condition a, condition b)."""
    confirmatory, secondary = [], []
    for arch, conditions in gate["chosen"].items():
        for generative in GENERATIVE:
            if generative in conditions and "none" in conditions:
                confirmatory.append((arch, generative, "none"))
        if all(g in conditions for g in GENERATIVE):
            secondary.append((arch, "cvae", "wgan_gp"))
        best = gate["best_generative"].get(arch)
        for baseline in SECONDARY_BASELINES:
            if best and baseline in conditions:
                secondary.append((arch, best, baseline))
    return {"confirmatory": confirmatory, "secondary": secondary}


def plan() -> dict:
    """What the evaluation will use: checkpoints with their hashes and state, and the comparisons."""
    gate = workflow.manifest().get(PHASE4_GATE)
    if not gate:
        raise FinalEvalError("The phase 4 gate is not approved; the final evaluation runs only after it (PLAN Phase 4)")
    checkpoints = []
    for arch, conditions in gate["chosen"].items():
        for condition, c in conditions.items():
            for run_id in c["run_ids"]:
                run_dir = runs.find_run_dir(run_id)
                path = run_dir / "checkpoints" / "best.pt" if run_dir else None
                expected = c["checkpoints"].get(run_id)
                actual = runs._sha256(path) if path and path.is_file() else None
                checkpoints.append({"architecture": arch, "condition": condition, "strength": c["strength"], "run_id": run_id,
                                    "path": str(path) if path else None, "sha256": expected, "present": actual is not None,
                                    "hash_matches": actual is not None and actual == expected})  # fmt: skip
    return {"gate": gate, "checkpoints": checkpoints, "comparisons": comparisons(gate), "prior_evaluations": prior_evaluations()}


def per_record_cms(probs: list[np.ndarray], y: np.ndarray, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(records, cms[n_seeds, n_records, C, C])`` from one probability array per seed."""
    out, unique = [], None
    for p in probs:
        unique, cms = M.confusion_by_record(y, np.asarray(p).argmax(axis=1), records)
        out.append(cms)
    return unique, np.stack(out)


def _drop(cms: np.ndarray, records: np.ndarray, record: int) -> np.ndarray:
    return cms[:, records != record]


def statistics(preds: dict[tuple[str, str], list[np.ndarray]], y: np.ndarray, records: np.ndarray, gate: dict,
               n_boot: int = N_BOOT, boot_seed: int = BOOT_SEED) -> dict:  # fmt: skip
    """The pre-registered analysis from test predictions (one probability array per seed per condition)."""
    cms = {}
    unique = None
    for key, probs in preds.items():
        unique, cms[key] = per_record_cms(probs, y, records)
    out: dict = {"records": unique.tolist(), "n_boot": n_boot, "boot_seed": boot_seed, "conditions": {}, "families": {}, "sensitivity": {}}
    for (arch, condition), c in cms.items():
        entry = {name: cluster_bootstrap_confusion(fn, c, unique, n=n_boot, seed=boot_seed) for name, fn in STATISTICS.items()}
        pr = [M.pr_auc_one_vs_rest(p, y) for p in preds[(arch, condition)]]
        entry["pr_auc"] = {cls: float(np.mean([r[cls] for r in pr])) for cls in CLASSES}
        entry["strength"] = gate["chosen"][arch][condition]["strength"]
        entry["n_seeds"] = int(c.shape[0])
        out["conditions"][f"{arch}/{condition}"] = entry

    def compare(family: list[tuple[str, str, str]], drop: int | None = None) -> dict:
        results = {}
        for arch, a, b in family:
            ca, cb = cms[(arch, a)], cms[(arch, b)]
            if drop is not None:
                ca, cb = _drop(ca, unique, drop), _drop(cb, unique, drop)
            results[f"{arch}: {a} vs {b}"] = paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, ca, cb, n=n_boot, seed=boot_seed)
        adjusted = holm({k: v["p_two_sided"] for k, v in results.items()})
        for key, value in results.items():
            value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"])
        return results

    family_lists = comparisons(gate)
    for family, items in family_lists.items():
        out["families"][family] = compare(items)
    out["sensitivity"]["without_record_202"] = {family: compare(items, drop=SAME_SUBJECT_RECORD) for family, items in family_lists.items()}
    loro = {}
    rare_records = [int(r) for r, cm in zip(unique, next(iter(cms.values()))[0]) if cm[_i["S"]].sum() or cm[_i["F"]].sum()]
    for record in rare_records:
        loro[str(record)] = {
            f"{arch}: {a} vs {b}": float(np.nanmean(M.rare_macro_f1_from_cm(_drop(cms[(arch, a)], unique, record).sum(axis=1)))
                                         - np.nanmean(M.rare_macro_f1_from_cm(_drop(cms[(arch, b)], unique, record).sum(axis=1))))
            for arch, a, b in family_lists["confirmatory"]
        }  # fmt: skip
    out["sensitivity"]["leave_one_record_out"] = loro
    per_record = {}
    for (arch, condition), c in cms.items():
        mean_cm = c.mean(axis=0)
        per_record[f"{arch}/{condition}"] = {
            str(int(r)): {cls: {"support": int(round(cm[_i[cls]].sum())), "recall": float(cm[_i[cls], _i[cls]] / cm[_i[cls]].sum()) if cm[_i[cls]].sum() else None}
                          for cls in ("S", "F")}
            for r, cm in zip(unique, mean_cm)
        }  # fmt: skip
    out["per_record"] = per_record
    return out


def run(confirm: str, n_boot: int = N_BOOT) -> dict:
    """The final evaluation. Every check comes before the first test byte is read."""
    info = plan()
    gate = info["gate"]
    if info["prior_evaluations"]:
        raise FinalEvalError("docs/TEST_ACCESS_LOG.md already holds an evaluation entry: the final evaluation runs once (PREREGISTRATION §8)")
    if confirm != CONFIRMATION:
        raise FinalEvalError(f'Confirm with exactly --confirm "{CONFIRMATION}"')
    if os.environ.get("ALLOW_TEST") != "1":
        raise FinalEvalError("ALLOW_TEST=1 is not set for this process")
    bad = [c["run_id"] for c in info["checkpoints"] if not c["hash_matches"]]
    if bad:
        raise FinalEvalError(f"Checkpoints missing or changed since the phase 4 gate: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
    contract = json.loads((paths.processed_dir() / "v1" / "contract.json").read_text())
    rr = bool(contract["representation"]["rr_features"])
    run_ids = [c["run_id"] for c in info["checkpoints"]]
    test = load_test("v1", run_ids=f"{len(run_ids)} grid checkpoints (results/final_eval/summary.json)", approved_by=", ".join(gate["approved_by"]))
    x = torch.from_numpy(test["X"])
    side = torch.from_numpy(np.stack([test[f] for f in SIDE_FIELDS], axis=1)) if rr else None
    preds: dict[tuple[str, str], list[np.ndarray]] = {}
    for c in info["checkpoints"]:
        probs = predict_proba(load_run_classifier(c["run_id"], rr), x, side, torch.device("cpu"))
        np.savez_compressed(runs.find_run_dir(c["run_id"]) / "preds_test.npz", probs=probs, y=test["y"], record=test["record"], r_idx=test["r_idx"],
                            rr_prev=test["rr_prev"], rr_ratio=test["rr_ratio"])  # RR saved so error analysis never reads the test set again
        preds.setdefault((c["architecture"], c["condition"]), []).append(probs)
    stats = statistics(preds, test["y"], test["record"], gate, n_boot=n_boot)
    stats.update({"run_ids": run_ids, "git_sha": runs.git_sha(), "phase4_gate": {"approved_by": gate["approved_by"], "best_generative": gate["best_generative"]}})
    write_outputs(stats)
    return stats


# --------------------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _macro_name(*parts: str) -> str:
    """A LaTeX macro name from parts: letters only, CamelCase (digits spelled out)."""
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    text = "".join(p.replace("_", " ").title().replace(" ", "") for p in parts)
    return "".join(words.get(ch, ch) for ch in text if ch.isalnum())


def write_outputs(stats: dict) -> None:
    out = paths.results_for(FINAL_EVAL)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(stats, indent=2, default=float) + "\n")
    with (out / "conditions.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["architecture", "condition", "strength", "metric", "point", "ci_low", "ci_high", "frac_undefined"])
        for key, entry in stats["conditions"].items():
            arch, condition = key.split("/")
            for name in STATISTICS:
                s = entry[name]
                writer.writerow([arch, condition, entry["strength"], name, s["point"], s["ci_low"], s["ci_high"], s["frac_undefined"]])
    with (out / "comparisons.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "comparison", "difference", "ci_low", "ci_high", "p_two_sided", "p_holm", "reject_holm", "frac_undefined"])
        for family, results in stats["families"].items():
            for name, r in results.items():
                writer.writerow([family, name, r["point"], r["ci_low"], r["ci_high"], r["p_two_sided"], r["p_holm"], r["reject_holm"], r["frac_undefined"]])
    tables = paths.repo_root() / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Architecture & Condition & Strength & Rare macro-F1 [95\\% CI] & N recall \\\\", "\\midrule"]
    for key, entry in stats["conditions"].items():
        arch, condition = key.split("/")
        r = entry["rare_macro_f1"]
        lines.append(f"{arch} & {condition.replace('_', ' ')} & {entry['strength'] if entry['strength'] is not None else '--'} & "
                     f"{_fmt(r['point'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] & {_fmt(entry['n_recall']['point'])} \\\\")  # fmt: skip
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "final_eval_conditions.tex").write_text("\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Family & Comparison & $\\Delta$ rare macro-F1 [95\\% CI] & $p$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    for family, results in stats["families"].items():
        for name, r in results.items():
            lines.append(f"{family} & {name.replace('_', ' ')} & {_fmt(r['point'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] & "
                         f"{_fmt(r['p_two_sided'], 4)} & {_fmt(r['p_holm'], 4)} \\\\")  # fmt: skip
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "final_eval_comparisons.tex").write_text("\n".join(lines) + "\n")
    macros = ["% Written by ecgaug.final_eval; do not edit."]
    for key, entry in stats["conditions"].items():
        arch, condition = key.split("/")
        r = entry["rare_macro_f1"]
        for suffix, value in (("", r["point"]), ("Low", r["ci_low"]), ("High", r["ci_high"])):
            macros.append(f"\\newcommand{{\\{_macro_name('test', arch, condition, 'rare', suffix)}}}{{{_fmt(value)}}}")
    for family, results in stats["families"].items():
        for name, r in results.items():
            arch, pair = name.split(": ")
            a, b = pair.split(" vs ")
            base = _macro_name("test", arch, a, "vs", b)
            for suffix, value, digits in (("Diff", r["point"], 3), ("Low", r["ci_low"], 3), ("High", r["ci_high"], 3), ("PHolm", r["p_holm"], 4)):
                macros.append(f"\\newcommand{{\\{base}{suffix}}}{{{_fmt(value, digits)}}}")
    generated = paths.repo_root() / "paper" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "numbers_final_eval.tex").write_text("\n".join(macros) + "\n")
    from ecgaug import figures

    figures.safe_call(figures.final_eval_figures, stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The single final evaluation on the test set (after the phase 4 gate).")
    parser.add_argument("--plan", action="store_true", help="list the checkpoints and comparisons; reads no test data")
    parser.add_argument("--confirm", default="", help=f'the exact phrase "{CONFIRMATION}"')
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.plan:
            info = plan()
            print(json.dumps({k: v for k, v in info.items() if k != "gate"}, indent=2))
            missing = [c["run_id"] for c in info["checkpoints"] if not c["hash_matches"]]
            print(f"{len(info['checkpoints'])} checkpoints, {len(missing)} missing or changed; prior evaluations: {len(info['prior_evaluations'])}")
            return 0 if not missing else 1
        stats = run(args.confirm, n_boot=args.n_boot)
    except (FinalEvalError, LockedTestDataError) as err:
        print(f"Final evaluation refused: {err}")
        return 2
    for family, results in stats["families"].items():
        print(f"{family} family:")
        for name, r in results.items():
            print(f"  {name}: {_fmt(r['point'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}], p {_fmt(r['p_two_sided'], 4)}, "
                  f"Holm {_fmt(r['p_holm'], 4)}{' (rejects H0)' if r['reject_holm'] else ''}")  # fmt: skip
    print("Written: results/final_eval/, tables/final_eval_*.tex, figures/final_eval/, paper/generated/numbers_final_eval.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
