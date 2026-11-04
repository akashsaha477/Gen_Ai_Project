"""The project's step-by-step workflow: where we are, what to run next, and the gates.

    python -m ecgaug.workflow status                     # checklist of every step and the next action
    python -m ecgaug.workflow phase1-gate                # compare variants A-D and write the phase 1 gate report
    ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "Name One" "Name Two"
    python -m ecgaug.workflow phase2-gate [--approve ...]    # choose the cnn and the rnn, fix the quality composite
    python -m ecgaug.workflow phase3a-gate [--approve ...]   # choose one generator per family from the pilots
    python -m ecgaug.workflow phase3-gate [--approve ...]    # check and freeze the synthetic banks
    python -m ecgaug.workflow phase4a-gate [--approve ...]   # choose the augmentation policy; approval writes the full grid
    python -m ecgaug.workflow phase4-gate [--approve ...]    # strengths per condition on validation; then /final-eval
    python -m ecgaug.workflow phase5-gate [--approve ...]    # explanation method per architecture, note variant, demo

Approving the phase 1 gate freezes the dataset of the winning variant (``data/processed/v1``, test part
included; a logged construction access) and records it in ``results/manifest.json``. Later steps read
the winner from there, so nobody has to remember which variant won. The phase 3 gates work the same
way: 3a records the chosen generator of each family, 3 records the SHA-256 of every bank.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import yaml

from ecgaug import paths, runs
from ecgaug.names import (
    PHASE1_DATA,
    PHASE1_GATE,
    PHASE1_INPUT_REPRESENTATION,
    PHASE2_BASELINE_CLASSIFIERS,
    PHASE2_GATE,
    PHASE2_PTBXL_PRETRAINING,
    PHASE2_QUALITY_METRICS,
    PHASE3_CVAE,
    PHASE3_GATE,
    PHASE3_WGAN_GP,
    PHASE3A_GATE,
    PHASE4_AUGMENTATION_POLICY,
    PHASE4_FULL_GRID,
    PHASE4_LEAKAGE_AUDIT,
    PHASE4_QUALITY_VS_GAIN,
    PHASE4_GATE,
    PHASE4A_GATE,
    PHASE5_CLINICAL_NOTES,
    PHASE5_EXPLAINABILITY,
    PHASE5_GATE,
    FINAL_EVAL,
    PHASES,
)
from ecgaug.eval.selection import Candidate, GeneratorCandidate, select_generator, select_winner

PRETRAIN_RESULTS = PHASE2_PTBXL_PRETRAINING
GENERATOR_EXPERIMENTS = (PHASE3_CVAE, PHASE3_WGAN_GP)
DAY_ONE = _dt.date(2026, 10, 7)
LAST_DAY = {1: 3, 2: 8, 3: 14, 4: 21, 5: 26, 6: 30}  # last planned day of each phase


class GateError(RuntimeError):
    """A gate cannot be run or approved yet; the message says why."""


def variant_configs() -> dict[str, Path]:
    """phase1_input_representation variant letter -> config file, e.g. {"A": .../A_short_window.yaml}."""
    folder = paths.configs_dir() / PHASE1_INPUT_REPRESENTATION
    return {p.name[0]: p for p in sorted(folder.glob("[A-Z]_*.yaml"))}


def phase1_results() -> dict[str, dict | None]:
    """The phase1_input_representation result file of each variant, or None where the variant has not been run."""
    folder = paths.results_for(PHASE1_INPUT_REPRESENTATION)
    out = {}
    for letter in variant_configs():
        path = folder / f"{letter}.json"
        out[letter] = json.loads(path.read_text()) if path.is_file() else None
    return out


def manifest() -> dict:
    path = paths.results_dir() / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def phase1_winner() -> str | None:
    """The variant letter chosen and approved at the phase 1 gate, or None before that."""
    entry = manifest().get(PHASE1_GATE)
    return entry["winner"] if entry else None


def evaluate_phase1_gate() -> dict:
    """Check the four phase1_input_representation results and apply the selection rule. Raises GateError if not ready."""
    results = phase1_results()
    missing = [letter for letter, r in results.items() if r is None]
    if missing:
        raise GateError(f"phase1_input_representation variants not run yet: {', '.join(missing)}")
    checks = []

    def same(field: str, label: str) -> None:
        values = {letter: json.dumps(r[field], sort_keys=True) for letter, r in results.items()}
        ok = len(set(values.values())) == 1
        checks.append({"check": label, "passed": ok, "detail": "identical" if ok else values})

    same("split", "same validation split for every variant")
    same("selection_metric", "same selection metric")
    same("budget", "same budget (seeds, learning rates, epochs, patience)")
    deviations = {letter: r["budget_deviations"] for letter, r in results.items() if r["budget_deviations"]}
    checks.append({"check": "no budget deviations reported", "passed": not deviations, "detail": deviations or "none"})
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise GateError(f"Phase 1 gate checks failed: {'; '.join(failed)}. See the result files.")

    candidates = [
        Candidate(
            name=letter, mean=r["mean"], sd=r["sd"], complexity=r["complexity_inputs_per_beat"],
            wall_clock_s=r["wall_clock_s"], tie_mean=r.get("tie_break_mean"), tie_sd=r.get("tie_break_sd"),
        )  # fmt: skip
        for letter, r in results.items()
    ]
    decision = select_winner(candidates)
    return {"results": results, "checks": checks, "decision": decision, "winner": decision["winner"]}


def _step_text(step: dict) -> str:
    """One step of a selection decision, as the gate reports print it."""
    if "passed" in step:
        return f"{step['step']}: passed {', '.join(step['passed']) or 'none'}; failed {', '.join(step['failed']) or 'none'}"
    if "skipped" in step:
        return f"{step['step']}: skipped ({step['skipped']})"
    if "tied" in step:
        return f"{step['step']}: tied with the best: {', '.join(step['tied'])}"
    return f"{step['step']}: **{step['winner']}**"


def phase1_gate_markdown(gate: dict) -> str:
    results, decision = gate["results"], gate["decision"]
    metric = next(iter(results.values()))["selection_metric"]
    lines = [
        "# Phase 1 gate: phase1_input_representation",
        "",
        f"Generated by `python -m ecgaug.workflow phase1-gate` on {_dt.date.today()}. Selection metric: `{metric}`",
        "(validation, mean over seeds). Rule: `docs/PREREGISTRATION.md` §4–5, applied by",
        "`ecgaug.eval.selection.select_winner`.",
        "",
        "## Checks",
        "",
        *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in gate["checks"]],
        "",
        "## Results",
        "",
        "| variant | inputs per beat | mean | SD | per seed | V F1 (tie-break) | margin to best | run time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for letter, r in results.items():
        seeds = ", ".join(f"{s['selection_value']:.4f}" for s in r["per_seed"])
        tie = f"{r['tie_break_mean']:.4f}" if r.get("tie_break_mean") is not None else "-"
        lines.append(
            f"| {letter} | {r['complexity_inputs_per_beat']} | {r['mean']:.4f} | {r['sd']:.4f} | {seeds} | {tie} | "
            f"{decision['margins'][letter]:.4f} | {r['wall_clock_s']:.0f} |"
        )
    lines += ["", "## Decision", ""]
    lines += [f"- {_step_text(step)}" for step in decision["steps"]]
    lines += [
        "",
        f"**Winner: variant {gate['winner']}** (`{variant_configs()[gate['winner']].name}`).",
        "",
        "Status: PENDING APPROVAL. After the team agrees, approve with",
        '`ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "Name" "Name"`.',
        "",
    ]
    return "\n".join(lines)


def write_phase1_gate_report() -> dict:
    """Evaluate the phase 1 gate and write ``results/phase1_gate/report.md`` and ``report.json``."""
    gate = evaluate_phase1_gate()
    out = paths.results_for(PHASE1_GATE)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(phase1_gate_markdown(gate))
    summary = {k: gate[k] for k in ("checks", "decision", "winner")}
    summary["run_ids"] = {letter: r["run_ids"] for letter, r in gate["results"].items()}
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    from ecgaug import figures

    try:
        figures.phase1_gate_figure()
    except Exception as err:  # noqa: BLE001 (the report is written; the figure is extra)
        print(f"WARNING: the phase 1 gate figure was not made: {err}")
    return gate


def approve_phase1_gate(approvers: list[str]) -> dict:
    """Freeze the winner's dataset as ``data/processed/v1`` and record the phase 1 gate in the manifest.

    Needs ``ALLOW_TEST_BUILD=1``, because building the test file reads DS2 (a logged construction
    access; no model sees it).
    """
    from ecgaug.data.contract import build_frozen
    from ecgaug.data.representation import RepresentationSpec
    from ecgaug.experiments.common import check_preconditions
    from ecgaug.experiments.phase1_input_representation import load_config

    if not approvers:
        raise GateError("Name at least one approver")
    if phase1_winner():
        raise GateError(f"The phase 1 gate is already approved (winner {phase1_winner()})")
    report = paths.results_for(PHASE1_GATE) / "report.json"
    if not report.is_file():
        raise GateError("Write the phase 1 gate report first: python -m ecgaug.workflow phase1-gate")
    gate = evaluate_phase1_gate()
    if gate["winner"] != json.loads(report.read_text())["winner"]:
        raise GateError("The results changed since the report was written; run `phase1-gate` again and re-read it")
    split = check_preconditions()
    config = variant_configs()[gate["winner"]]
    spec = RepresentationSpec(**load_config(config)["representation"])
    out = build_frozen("v1", spec, split)
    contract = json.loads((out / "contract.json").read_text())
    entry = {
        "winner": gate["winner"],
        "config": str(config.relative_to(paths.repo_root())),
        "approved_by": list(approvers),
        "approved_on": str(_dt.date.today()),
        "git_sha": runs.git_sha(),
        "dataset": "data/processed/v1",
        "files": contract["files"],
        "counts": contract["counts"],
    }
    path = paths.results_dir() / "manifest.json"
    path.write_text(json.dumps({**manifest(), PHASE1_GATE: entry}, indent=2) + "\n")
    decisions = paths.docs_dir() / "DECISIONS.md"
    with decisions.open("a") as fh:
        fh.write(
            f"\n## the phase 1 gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
            f"phase1_input_representation: winner {gate['winner']} ({config.stem}) — evidence: results/phase1_gate/report.md — "
            "rule: PREREGISTRATION §4–5\n"
            f"Frozen artifacts (hashes): data/processed/v1, see results/manifest.json\n"
        )
    return entry


# --------------------------------------------------------------------------------------------------
# phase 2: the cnn and the rnn, the quality composite
# --------------------------------------------------------------------------------------------------

PROPOSED_V2 = {
    "class_weight_gammas": [0.5, 0.75, 1.0],
    "hard_gates": "configs/phase3_*/generator_settings.yaml (diversity_min_ratio, memorization_max_fraction)",
    "seeds": "3 seeds unless the measured run times fit the full grid at 5 seeds within the Kaggle quota",
}


def evaluate_phase2_gate() -> dict:
    """Apply the phase 2 rules: cnn from {A, B} and rnn from {C, D}; the composite is the passing families."""
    baselines = experiment_results(PHASE2_BASELINE_CLASSIFIERS)
    quality_results = experiment_results(PHASE2_QUALITY_METRICS)
    for name, results in ((PHASE2_BASELINE_CLASSIFIERS, baselines), (PHASE2_QUALITY_METRICS, quality_results)):
        missing = [letter for letter, r in results.items() if r is None]
        if not results or missing:
            raise GateError(f"{name} variants not run yet: {', '.join(missing) or 'all'}")
    checks = [
        _same(baselines, lambda r: r["split"], "baselines: same validation split"),
        _same(baselines, lambda r: r["selection_metric"], "baselines: same selection metric"),
        _same(baselines, lambda r: r["budget"], "baselines: same budget (seeds, learning rates, epochs, patience)"),
        _same(baselines, lambda r: r["dataset_content_sha256"], "baselines: same training data"),
        _same(quality_results, lambda r: r["settings"], "quality metrics: same harness settings"),
        _same(quality_results, lambda r: r["dataset"].get("content_sha256"), "quality metrics: same training data"),
    ]
    deviations = {letter: r["budget_deviations"] for letter, r in baselines.items() if r["budget_deviations"]}
    checks.append({"check": "baselines: no budget deviations reported", "passed": not deviations, "detail": deviations or "none"})
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise GateError(f"Phase 2 gate checks failed: {'; '.join(failed)}. See the result files.")
    roles = {}
    for role in ("cnn", "rnn"):
        members = {letter: r for letter, r in baselines.items() if r["role"] == role}
        candidates = [
            Candidate(name=letter, mean=r["mean"], sd=r["sd"], complexity=r["n_parameters"], wall_clock_s=r["wall_clock_s"],
                      tie_mean=r.get("tie_break_mean"), tie_sd=r.get("tie_break_sd"))  # fmt: skip
            for letter, r in members.items()
        ]
        roles[role] = select_winner(candidates)
    composite = [r["family"] for r in quality_results.values() if r["passed"]]
    fid = next((r for r in quality_results.values() if r["family"] == "fid_analog"), None)
    winner = phase1_winner()
    pretrain_path = paths.results_for(PHASE2_PTBXL_PRETRAINING) / f"{winner}.json" if winner else None
    pretraining = json.loads(pretrain_path.read_text()) if pretrain_path and pretrain_path.is_file() else None
    provisional = sorted(f"{n} {l}" for n, results in ((PHASE2_BASELINE_CLASSIFIERS, baselines), (PHASE2_QUALITY_METRICS, quality_results))
                         for l, r in results.items() if r.get("provisional"))  # fmt: skip
    return {
        "baselines": {"results": baselines, "cnn": roles["cnn"], "rnn": roles["rnn"]},
        "quality": {"results": quality_results, "composite": composite, "feature_net": fid["feature_net"] if fid and "fid_analog" in composite else None,
                    "settings": next(iter(quality_results.values()))["settings"]["metrics"]},
        "pretraining": None if pretraining is None else {"winner": pretraining["winner"], "provisional": pretraining.get("provisional", True),
                                                         "difference": pretraining["difference_pretrained_minus_scratch"]},
        "checks": checks, "provisional": provisional, "proposed_v2": PROPOSED_V2,
    }  # fmt: skip


def phase2_gate_markdown(gate: dict) -> str:
    lines = [
        "# Phase 2 gate: baseline classifiers and quality metrics",
        "",
        f"Generated by `python -m ecgaug.workflow phase2-gate` on {_dt.date.today()}. Rules: PLAN Phase 2 and",
        "PREREGISTRATION §4: the `cnn` is the better of A and B, the `rnn` the better of C and D, by the seed-mean",
        "validation selection metric; ties within max(0.02, pooled SD) go to fewer parameters. The composite quality",
        "rank is the equal-weight average of the ranks of the metric families that pass the sanity harness.",
        "",
    ]
    if gate["provisional"]:
        lines += [f"**PREVIEW: provisional results ({', '.join(gate['provisional'])}); this report cannot be approved.**", ""]
    lines += ["## Checks", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in gate["checks"]], ""]
    lines += ["## Baseline classifiers", "", "| variant | role | parameters | mean | SD | per seed | V F1 (tie-break) | lr | run time (s) |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for letter, r in gate["baselines"]["results"].items():
        seeds = ", ".join(f"{s['selection_value']:.4f}" for s in r["per_seed"])
        tie = f"{r['tie_break_mean']:.4f}" if r.get("tie_break_mean") is not None else "-"
        lines.append(f"| {letter} | {r['role']} | {r['n_parameters']:,} | {r['mean']:.4f} | {r['sd']:.4f} | {seeds} | {tie} | {r['chosen_lr']:g} | {r['wall_clock_s']:.0f} |")
    for role in ("cnn", "rnn"):
        decision = gate["baselines"][role]
        steps = "; ".join(_step_text(s).replace("**", "") for s in decision["steps"])
        lines += ["", f"**{role}: variant {decision['winner']}** ({steps})"]
    lines += ["", "## Quality metrics", "", "| variant | family | passes | reference best | Spearman per degradation | median CV |", "|---|---|---|---|---|---|"]
    for letter, r in gate["quality"]["results"].items():
        rho = ", ".join(f"{k} {v:.2f}" for k, v in r["spearman"].items())
        lines.append(f"| {letter} | {r['family']} | {'yes' if r['passed'] else 'no'} | {'yes' if r['reference_best'] else 'no'} | {rho} | {r['median_cv']:.3f} |")
    composite = gate["quality"]["composite"]
    lines += ["", f"**Composite quality rank: {', '.join(composite) if composite else 'none (no family passes; phase 3 ties fall back to V F1)'}**", ""]
    pre = gate["pretraining"]
    lines += ["## phase2_ptbxl_pretraining", "", "Not run for the phase 1 winner yet." if pre is None else
              f"Winner {pre['winner']} (pretrained minus scratch {pre['difference']:+.4f}){' — provisional' if pre['provisional'] else ''}. "
              "Pretraining joins the phase 4 grid only if it wins.", ""]  # fmt: skip
    lines += ["## To fix in PREREGISTRATION v2 before any official phase 3 run", "",
              *[f"- {k.replace('_', ' ')}: {v}" for k, v in gate["proposed_v2"].items()], "",
              "Status: PENDING APPROVAL. After the team agrees, approve with",
              '`python -m ecgaug.workflow phase2-gate --approve "Name" "Name"`.', ""]  # fmt: skip
    return "\n".join(lines)


def write_phase2_gate_report() -> dict:
    gate = evaluate_phase2_gate()
    out = paths.results_for(PHASE2_GATE)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(phase2_gate_markdown(gate))
    summary = {"cnn": gate["baselines"]["cnn"], "rnn": gate["baselines"]["rnn"], "composite": gate["quality"]["composite"], "provisional": gate["provisional"]}
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    from ecgaug import figures

    figures.safe_call(figures.phase2_gate_figure, gate)
    return gate


def _checkpoint_hashes(run_ids: list[str]) -> dict[str, str | None]:
    out = {}
    for run_id in run_ids:
        run_dir = runs.find_run_dir(run_id)
        env = json.loads((run_dir / "env.json").read_text()) if run_dir else {}
        out[run_id] = (env.get("checkpoint") or {}).get("sha256")
    return out


def approve_phase2_gate(approvers: list[str]) -> dict:
    """Record the cnn, the rnn, the quality composite and the feature network in the manifest and DECISIONS.md."""
    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE2_GATE):
        raise GateError("The phase 2 gate is already approved")
    report = paths.results_for(PHASE2_GATE) / "report.json"
    if not report.is_file():
        raise GateError("Write the phase 2 gate report first: python -m ecgaug.workflow phase2-gate")
    gate = evaluate_phase2_gate()
    if gate["provisional"]:
        raise GateError(f"Provisional results ({', '.join(gate['provisional'])}); rerun them officially after the phase 1 gate")
    written = json.loads(report.read_text())
    if written["cnn"]["winner"] != gate["baselines"]["cnn"]["winner"] or written["rnn"]["winner"] != gate["baselines"]["rnn"]["winner"]:
        raise GateError("The results changed since the report was written; run `phase2-gate` again and re-read it")
    entry: dict = {}
    for role in ("cnn", "rnn"):
        letter = gate["baselines"][role]["winner"]
        r = gate["baselines"]["results"][letter]
        entry[role] = {
            "variant": letter, "config": str(experiment_configs(PHASE2_BASELINE_CLASSIFIERS)[letter].relative_to(paths.repo_root())),
            "model": r["model"], "lr": r["chosen_lr"], **{k: r["budget"][k] for k in ("max_epochs", "patience", "min_epochs", "batch_size", "weight_decay")},
            "run_ids": r["run_ids"], "checkpoints": _checkpoint_hashes(r["run_ids"]),
        }  # fmt: skip
    entry.update({
        "class_weight_gammas": PROPOSED_V2["class_weight_gammas"],
        "quality_composite": gate["quality"]["composite"], "quality_settings": gate["quality"]["settings"],
        "feature_net": gate["quality"]["feature_net"], "approved_by": list(approvers), "approved_on": str(_dt.date.today()),
        "git_sha": runs.git_sha(),
    })  # fmt: skip
    path = paths.results_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**manifest(), PHASE2_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(
            f"\n## the phase 2 gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
            f"phase2_baseline_classifiers: cnn = variant {entry['cnn']['variant']}, rnn = variant {entry['rnn']['variant']}; "
            f"phase2_quality_metrics: composite = {', '.join(entry['quality_composite']) or 'none'} — evidence: results/phase2_gate/report.md — "
            "rule: PLAN Phase 2, PREREGISTRATION §4\nFrozen artifacts (hashes): checkpoints and feature network, see results/manifest.json\n"
        )
    return entry


# --------------------------------------------------------------------------------------------------
# phase 3: the 3a gate (one generator per family) and the 3 gate (banks frozen)
# --------------------------------------------------------------------------------------------------


def experiment_configs(experiment: str) -> dict[str, Path]:
    """Variant letter -> config file of any experiment."""
    folder = paths.configs_dir() / experiment
    return {p.name[0]: p for p in sorted(folder.glob("[A-Z]_*.yaml"))}


def experiment_results(experiment: str) -> dict[str, dict | None]:
    """The result file of each variant of an experiment, or None where the variant has not run."""
    folder = paths.results_for(experiment)
    out = {}
    for letter in experiment_configs(experiment):
        path = folder / f"{letter}.json"
        out[letter] = json.loads(path.read_text()) if path.is_file() else None
    return out


def phase3a_choice(experiment: str) -> dict | None:
    """The generator variant approved at the phase 3a gate for ``experiment``, or None."""
    return (manifest().get(PHASE3A_GATE) or {}).get(experiment)


def _same(results: dict[str, dict], get, label: str) -> dict:
    values = {letter: json.dumps(get(r), sort_keys=True) for letter, r in results.items()}
    ok = len(set(values.values())) == 1
    return {"check": label, "passed": ok, "detail": "identical" if ok else values}


def evaluate_phase3a_gate() -> dict:
    """Check every generator pilot and apply the phase 3 selection rule per family. Raises GateError if not ready."""
    from ecgaug.quality import composite_ranks

    composite = (manifest().get(PHASE2_GATE) or {}).get("quality_composite") or []
    families = {}
    for experiment in GENERATOR_EXPERIMENTS:
        results = experiment_results(experiment)
        missing = [letter for letter, r in results.items() if r is None]
        if not results or missing:
            raise GateError(f"{experiment} variants not piloted yet: {', '.join(missing) or 'all'}")
        checks = [
            _same(results, lambda r: r["split"], "same validation split for every variant"),
            _same(results, lambda r: r["selection_metric"], "same selection metric"),
            _same(results, lambda r: r["budget"], "same budget (steps, seeds, checkpoints, pilot policy)"),
            _same(results, lambda r: r["pilot"]["classifier"], "same pilot classifier and settings"),
            _same(results, lambda r: r["hard_gate_thresholds"], "same hard-gate thresholds"),
            _same(results, lambda r: r["dataset"].get("content_sha256"), "same training data"),
        ]
        deviations = {letter: r["budget_deviations"] for letter, r in results.items() if r["budget_deviations"]}
        checks.append({"check": "no budget deviations reported", "passed": not deviations, "detail": deviations or "none"})
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            raise GateError(f"{experiment}: phase 3a gate checks failed: {'; '.join(failed)}. See the result files.")
        ranks = composite_ranks({letter: r["quality_mean"] for letter, r in results.items()}, composite)
        candidates = [
            GeneratorCandidate(
                name=letter, mean=r["mean"], sd=r["sd"], passed_hard_gates=r["hard_gates"]["passed"],
                n_parameters=r["n_parameters"], wall_clock_s=r["wall_clock_s"], tie_mean=r.get("tie_break_mean"),
                tie_sd=r.get("tie_break_sd"), quality_rank=ranks[letter] if ranks else None,
            )  # fmt: skip
            for letter, r in results.items()
        ]
        decision = select_generator(candidates)
        families[experiment] = {
            "results": results, "checks": checks, "decision": decision, "winner": decision["winner"],
            "setting": results[decision["winner"]]["chosen_setting"] if decision["winner"] else None,
            "provisional": sorted(letter for letter, r in results.items() if r.get("provisional")),
            "quality_ranks": ranks,
        }  # fmt: skip
    return {"families": families}


def _diagnostic_extremes(r: dict) -> tuple[float, float, float]:
    per_seed = [s for s in r["per_seed"] if s.get("diagnostics")]
    values = {name: [s["diagnostics"][c][name] for s in per_seed for c in s["diagnostics"]] for name in ("diversity_ratio", "memorization_fraction", "c2st_accuracy")}
    lowest = min(values["diversity_ratio"]) if values["diversity_ratio"] else float("nan")
    highest = max(values["memorization_fraction"]) if values["memorization_fraction"] else float("nan")
    c2st = max(values["c2st_accuracy"]) if values["c2st_accuracy"] else float("nan")
    return lowest, highest, c2st


def phase3a_gate_markdown(gate: dict) -> str:
    lines = [
        "# Phase 3a gate: one generator per family",
        "",
        f"Generated by `python -m ecgaug.workflow phase3a-gate` on {_dt.date.today()}. Rule: PLAN Phase 3 selection and",
        "PREREGISTRATION §9, applied by `ecgaug.eval.selection.select_generator`: hard gates (no divergence on",
        "any pilot seed, diversity ratio at least the floor, memorization fraction at most the ceiling, for every",
        "generated class and seed), then the pilot selection metric at the final checkpoint (mean over pilot",
        "seeds), ties within max(0.02, pooled SD) broken by the composite quality rank, then fewer parameters.",
        "",
    ]
    provisional = {e: f["provisional"] for e, f in gate["families"].items() if f["provisional"]}
    if provisional:
        reasons = sorted({reason for f in gate["families"].values() for r in f["results"].values() for reason in r.get("provisional_reasons", [])})
        lines += ["**PREVIEW: these pilots are provisional, so this report cannot be approved.** Reasons:", "", *[f"- {r}" for r in reasons], ""]
    for experiment, family in gate["families"].items():
        decision = family["decision"]
        lines += [f"## {experiment}", "", "### Checks", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in family["checks"]], ""]
        lines += [
            "### Results",
            "",
            "| variant | setting | pilot mean | SD | per seed | V F1 (tie-break) | diversity (lowest) | memorization (highest) | C2ST (highest) | hard gates | parameters | run time (s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for letter, r in family["results"].items():
            seeds = ", ".join(f"{s['selection_value']:.4f}" for s in r["per_seed"])
            tie = f"{r['tie_break_mean']:.4f}" if r.get("tie_break_mean") is not None else "-"
            lowest, highest, c2st = _diagnostic_extremes(r)
            gates = "pass" if r["hard_gates"]["passed"] else "fail: " + "; ".join(c["check"] for c in r["hard_gates"]["checks"] if not c["passed"])
            lines.append(
                f"| {letter} | {r['chosen_setting']} | {r['mean']:.4f} | {r['sd']:.4f} | {seeds} | {tie} | {lowest:.3f} | "
                f"{highest:.3f} | {c2st:.3f} | {gates} | {r['n_parameters']:,} | {r['wall_clock_s']:.0f} |"
            )
        other = [(letter, name, s) for letter, r in family["results"].items() if len(r["settings"]) > 1 for name, s in r["settings"].items()]
        if other:
            lines += ["", "Every setting of a variant with several (both are reported; the variant uses its best passing one):", ""]
            lines += [f"- {letter} {name}: {s['mean']:.4f} ± {s['sd']:.4f}, hard gates {'pass' if s['hard_gates']['passed'] else 'fail'}" for letter, name, s in other]
        lines += ["", "### Decision", ""]
        lines += [f"- {_step_text(step)}" for step in decision["steps"]]
        if family["winner"]:
            config = experiment_configs(experiment)[family["winner"]].name
            lines += ["", f"**Winner: variant {family['winner']}** (`{config}`, setting {family['setting']}).", ""]
        else:
            lines += ["", f"**No winner.** {decision['fallback']}", ""]
    lines += [
        "Status: PENDING APPROVAL. After the team agrees, approve with",
        '`python -m ecgaug.workflow phase3a-gate --approve "Name" "Name"`; then run each winner with `--final`.',
        "",
    ]
    return "\n".join(lines)


def write_phase3a_gate_report() -> dict:
    """Evaluate the phase 3a gate and write ``results/phase3a_gate/report.md`` and ``report.json``."""
    gate = evaluate_phase3a_gate()
    out = paths.results_for(PHASE3A_GATE)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(phase3a_gate_markdown(gate))
    summary = {
        experiment: {k: f[k] for k in ("checks", "decision", "winner", "setting", "provisional")} | {"run_ids": {l: r["run_ids"] for l, r in f["results"].items()}}
        for experiment, f in gate["families"].items()
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    from ecgaug import figures

    figures.safe_call(figures.phase3a_gate_figure, gate)
    return gate


def approve_phase3a_gate(approvers: list[str]) -> dict:
    """Record the chosen generator of each family in the manifest and in DECISIONS.md."""
    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE3A_GATE):
        raise GateError("The phase 3a gate is already approved")
    report = paths.results_for(PHASE3A_GATE) / "report.json"
    if not report.is_file():
        raise GateError("Write the phase 3a gate report first: python -m ecgaug.workflow phase3a-gate")
    gate = evaluate_phase3a_gate()
    written = json.loads(report.read_text())
    for experiment, family in gate["families"].items():
        if family["provisional"]:
            raise GateError(f"{experiment}: pilots {', '.join(family['provisional'])} are previews; rerun them officially after the phase 1 and phase 2 gates")
        if family["winner"] is None:
            raise GateError(f"{experiment}: no variant passes the hard gates. Record the team's fallback (PLAN Phase 3) in DECISIONS.md first.")
        if written[experiment]["winner"] != family["winner"]:
            raise GateError("The results changed since the report was written; run `phase3a-gate` again and re-read it")
    entry = {
        experiment: {
            "winner": f["winner"], "setting": f["setting"], "config": str(experiment_configs(experiment)[f["winner"]].relative_to(paths.repo_root())),
            "generator_id": f"{experiment}-{f['winner']}" + ("" if f["setting"] == "default" else f".{f['setting']}"),
        }
        for experiment, f in gate["families"].items()
    }  # fmt: skip
    entry.update({"approved_by": list(approvers), "approved_on": str(_dt.date.today()), "git_sha": runs.git_sha()})
    path = paths.results_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**manifest(), PHASE3A_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(
            f"\n## the phase 3a gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
            + "".join(f"{e}: winner {entry[e]['winner']} (setting {entry[e]['setting']}) — " for e in GENERATOR_EXPERIMENTS)
            + "evidence: results/phase3a_gate/report.md — rule: PLAN Phase 3 selection, PREREGISTRATION §9\n"
        )
    return entry


def evaluate_phase3_gate() -> dict:
    """Check the final runs and banks of both chosen generators. Raises GateError if not ready."""
    from ecgaug.experiments.generator_runs import load_config
    from ecgaug.generators.bank import BankError, bank_dir, load_bank

    chosen = manifest().get(PHASE3A_GATE)
    if not chosen:
        raise GateError("Approve the phase 3a gate first: python -m ecgaug.workflow phase3a-gate")
    families = {}
    for experiment in GENERATOR_EXPERIMENTS:
        choice = chosen[experiment]
        path = paths.results_for(experiment) / f"{choice['winner']}_final.json"
        if not path.is_file():
            raise GateError(f"{experiment}: the final runs of variant {choice['winner']} have not been ingested ({path.name})")
        final = json.loads(path.read_text())
        seeds = load_config(paths.repo_root() / choice["config"])["final_seeds"]
        checks = [
            {"check": "official run (not a preview)", "passed": not final["provisional"]},
            {"check": "the setting approved at the phase 3a gate", "passed": final["setting"] == choice["setting"]},
            {"check": f"every grid seed {seeds}", "passed": sorted(final["seeds"]) == sorted(seeds)},
            {"check": "no seed diverged", "passed": not final["diverged_seeds"]},
        ]
        banks = {}
        for seed in seeds:
            try:
                arrays = load_bank(choice["generator_id"], seed)
                meta = json.loads((bank_dir(choice["generator_id"]) / "bank.json").read_text())["seeds"][str(seed)]
                ok = meta.get("train_only_check") == "passed" and bool(arrays["is_synthetic"].all())
                banks[str(seed)] = {"file": f"data/synthetic/{choice['generator_id']}/seed{seed}.npz", "sha256": meta["sha256"],
                                    "content_sha256": meta["content_sha256"], "counts": meta["counts"], "diagnostics": meta.get("diagnostics")}  # fmt: skip
                checks.append({"check": f"bank seed {seed}: present, hash matches, train-only check passed", "passed": ok})
            except (BankError, KeyError, FileNotFoundError) as err:
                checks.append({"check": f"bank seed {seed}: {err}", "passed": False})
        families[experiment] = {"choice": choice, "final": final, "checks": checks, "banks": banks}
    return {"families": families, "passed": all(c["passed"] for f in families.values() for c in f["checks"])}


def phase3_gate_markdown(gate: dict) -> str:
    lines = ["# Phase 3 gate: synthetic banks", "", f"Generated by `python -m ecgaug.workflow phase3-gate` on {_dt.date.today()}.", ""]
    for experiment, family in gate["families"].items():
        lines += [f"## {experiment}: {family['choice']['generator_id']}", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in family["checks"]], ""]
        if family["banks"]:
            lines += ["| seed | beats S / V / F | diversity (lowest) | memorization (highest) | SHA-256 |", "|---|---|---|---|---|"]
            for seed, bank in family["banks"].items():
                diag = bank.get("diagnostics") or {}
                lowest = min((d["diversity_ratio"] for d in diag.values()), default=float("nan"))
                highest = max((d["memorization_fraction"] for d in diag.values()), default=float("nan"))
                counts = " / ".join(f"{bank['counts'][c]:,}" for c in ("S", "V", "F"))
                lines.append(f"| {seed} | {counts} | {lowest:.3f} | {highest:.3f} | `{bank['sha256'][:12]}...` |")
            lines.append("")
    lines += ["Status: " + ("ready; approve with `python -m ecgaug.workflow phase3-gate --approve \"Name\" \"Name\"`." if gate["passed"] else "NOT READY: fix the failed checks above."), ""]
    return "\n".join(lines)


def approve_phase3_gate(approvers: list[str]) -> dict:
    """Freeze the banks: record their hashes in the manifest and the decision in DECISIONS.md."""
    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE3_GATE):
        raise GateError("The phase 3 gate is already approved")
    gate = evaluate_phase3_gate()
    if not gate["passed"]:
        raise GateError("Phase 3 gate checks failed; run `python -m ecgaug.workflow phase3-gate` and read the report")
    entry = {e: {"generator_id": f["choice"]["generator_id"], "banks": f["banks"]} for e, f in gate["families"].items()}
    entry.update({"approved_by": list(approvers), "approved_on": str(_dt.date.today()), "git_sha": runs.git_sha()})
    path = paths.results_dir() / "manifest.json"
    path.write_text(json.dumps({**manifest(), PHASE3_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(
            f"\n## the phase 3 gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
            f"Frozen banks: {', '.join(entry[e]['generator_id'] for e in GENERATOR_EXPERIMENTS)} (hashes in results/manifest.json) — "
            "evidence: results/phase3_gate/report.md\n"
        )
    return entry


# --------------------------------------------------------------------------------------------------
# phase 4: the 4a gate (augmentation policy) and the 4 gate (strengths and checkpoints for the final evaluation)
# --------------------------------------------------------------------------------------------------

FULL_GRID_FILE = Path("configs") / PHASE4_FULL_GRID / "full_grid.yaml"
N_RECALL_REPORT_LIMIT = 0.01  # the phase 4 gate report flags a chosen strength past this; only the policy pilot enforces it


def evaluate_phase4a_gate() -> dict:
    """Policies that keep N recall within the limit compete on the pilot score; ties go to the simpler policy."""
    results = experiment_results(PHASE4_AUGMENTATION_POLICY)
    missing = [letter for letter, r in results.items() if r is None]
    if not results or missing:
        raise GateError(f"{PHASE4_AUGMENTATION_POLICY} variants not run yet: {', '.join(missing) or 'all'}")
    checks = [
        _same(results, lambda r: r["reference"]["run_ids"], "one shared reference run without augmentation"),
        _same(results, lambda r: sorted(b["bank"] for b in r["per_bank"].values()), "the same two banks"),
        _same(results, lambda r: [r["classifier"], r["seeds"]], "the same classifier and seeds"),
        _same(results, lambda r: r["dataset"].get("content_sha256"), "the same training data"),
    ]
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise GateError(f"Phase 4a gate checks failed: {'; '.join(failed)}")
    eligible = {letter: r for letter, r in results.items() if r["constraint"]["passed"]}
    if eligible:
        decision = select_winner([
            Candidate(name=letter, mean=r["mean"], sd=r["sd"], complexity=r["complexity"], tie_mean=r["tie_break_mean"], tie_sd=r["tie_break_sd"])
            for letter, r in eligible.items()
        ])  # fmt: skip
    else:
        decision = {"winner": None, "steps": [], "fallback": "No policy keeps N recall within the limit; the team decides and records it in DECISIONS.md."}
    return {"results": results, "checks": checks, "decision": decision, "winner": decision["winner"],
            "excluded": {l: r["constraint"]["n_recall_drop"] for l, r in results.items() if not r["constraint"]["passed"]},
            "provisional": sorted(l for l, r in results.items() if r.get("provisional"))}  # fmt: skip


def phase4a_gate_markdown(gate: dict) -> str:
    lines = ["# Phase 4a gate: augmentation policy", "",
             f"Generated by `python -m ecgaug.workflow phase4a-gate` on {_dt.date.today()}. Rule: PLAN Phase 4, DECISIONS A19: the validation",
             "selection metric (cnn, mean over the seeds of the mean over the two banks), with validation N recall at most 0.01 below the",
             "run without augmentation, on average over the banks; ties within max(0.02, pooled SD) go to V F1, then to the simpler policy",
             "(complexity A 1, B 2, C 2, D 3). The per-bank columns are for reading only and do not change the rule.", ""]  # fmt: skip
    if gate["provisional"]:
        lines += [f"**PREVIEW: provisional results ({', '.join(gate['provisional'])}); this report cannot be approved.**", ""]
    lines += ["## Checks", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in gate["checks"]], "",
              "## Results", "", "| variant | policy | pilot strength | mean | SD | per bank | N recall drop | per bank | V F1 drop per bank | within limit | grid strengths |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    over = []
    for letter, r in gate["results"].items():
        banks = ", ".join(f"{k} {v['selection_value']:.4f}" for k, v in r["per_bank"].items())
        n_drops = ", ".join(f"{k} {v['n_recall_drop']:+.4f}" for k, v in r["per_bank"].items())
        v_drops = ", ".join(f"{k} {v['tie_break_drop']:+.4f}" for k, v in r["per_bank"].items())
        lines.append(f"| {letter} | {r['policy']['kind']} | {r['pilot_strength']} | {r['mean']:.4f} | {r['sd']:.4f} | {banks} | {r['constraint']['n_recall_drop']:+.4f} | "
                     f"{n_drops} | {v_drops} | {'yes' if r['constraint']['passed'] else 'no'} | {r['grid_strengths']} |")  # fmt: skip
        over += [f"{letter} with {k} ({v['n_recall_drop']:+.4f})" for k, v in r["per_bank"].items() if v["n_recall_drop"] > r["constraint"]["max_n_recall_drop"]]
    if over:
        lines += ["", f"**Read before approving:** one bank alone is over the N recall limit, hidden by the average: {', '.join(over)}."]
    lines += ["", "## Decision", ""]
    if gate["winner"]:
        lines += [f"**Winner: variant {gate['winner']}** ({gate['results'][gate['winner']]['policy']['kind']}). Approval writes `{FULL_GRID_FILE}`."]
    else:
        lines += [f"**No winner.** {gate['decision']['fallback']}"]
    lines += ["", "Status: PENDING APPROVAL. Approve with `python -m ecgaug.workflow phase4a-gate --approve \"Name\" \"Name\"`.", ""]
    return "\n".join(lines)


def full_grid_yaml(policy: dict, strengths: list | None, gammas: list, seeds: list) -> str:
    """The full grid file for the approved policy (PLAN Phase 4: six conditions, cnn and rnn, the grid seeds)."""
    with_strengths = {} if strengths is None else {"strengths": strengths}
    grid = {
        "experiment": PHASE4_FULL_GRID, "name": "full_grid", "policy": policy, "seeds": seeds,
        "architectures": {"cnn": "phase2_gate", "rnn": "phase2_gate"},
        "conditions": {
            "none": {}, "class_weighted": {"gammas": gammas}, "random_oversampling": dict(with_strengths),
            "smote": {**with_strengths, "k_neighbors": 5}, "cvae": {"bank": "phase3_gate/phase3_cvae", **with_strengths},
            "wgan_gp": {"bank": "phase3_gate/phase3_wgan_gp", **with_strengths},
        },
    }  # fmt: skip
    header = (f"# Written by `python -m ecgaug.workflow phase4a-gate --approve` on {_dt.date.today()} from the approved policy.\n"
              "# Do not edit: the grid is what the phase 4a gate decided. Cells already run are skipped on a rerun.\n")  # fmt: skip
    return header + yaml.safe_dump(grid, sort_keys=False)


def approve_phase4a_gate(approvers: list[str]) -> dict:
    """Record the policy and write the full grid file from it."""
    from ecgaug.experiments.generator_runs import load_config as generator_config

    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE4A_GATE):
        raise GateError("The phase 4a gate is already approved")
    gate = evaluate_phase4a_gate()
    if gate["provisional"]:
        raise GateError(f"Provisional results ({', '.join(gate['provisional'])}); rerun them officially after the phase 3 gate")
    if gate["winner"] is None:
        raise GateError("No policy qualifies; record the team's decision in DECISIONS.md first")
    winner = gate["results"][gate["winner"]]
    path = paths.repo_root() / FULL_GRID_FILE
    if path.exists():
        raise GateError(f"{FULL_GRID_FILE} exists already; it is never overwritten")
    gammas = (manifest().get(PHASE2_GATE) or {}).get("class_weight_gammas", PROPOSED_V2["class_weight_gammas"])
    seeds = generator_config(experiment_configs(PHASE3_CVAE)["A"])["final_seeds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full_grid_yaml(winner["policy"], winner["grid_strengths"], gammas, seeds))
    entry = {"winner": gate["winner"], "config": str(experiment_configs(PHASE4_AUGMENTATION_POLICY)[gate["winner"]].relative_to(paths.repo_root())),
             "policy": winner["policy"], "strengths": winner["grid_strengths"], "full_grid": str(FULL_GRID_FILE),
             "approved_by": list(approvers), "approved_on": str(_dt.date.today()), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_dir() / "manifest.json").write_text(json.dumps({**manifest(), PHASE4A_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(f"\n## the phase 4a gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
                 f"{PHASE4_AUGMENTATION_POLICY}: winner {gate['winner']} ({winner['policy']['kind']}, strengths {winner['grid_strengths']}) — "
                 f"evidence: results/phase4a_gate/report.md — rule: PLAN Phase 4\nFull grid: {FULL_GRID_FILE}\n")  # fmt: skip
    return entry


def evaluate_phase4_gate() -> dict:
    """Strength per condition and architecture on validation; best generative condition per architecture."""
    from ecgaug.grid import runner as grid_runner

    if not manifest().get(PHASE4A_GATE):
        raise GateError("Approve the phase 4a gate first: python -m ecgaug.workflow phase4a-gate")
    grid = grid_runner.load_grid(paths.repo_root() / FULL_GRID_FILE)
    expected = grid_runner.cells(grid)
    rows = {(r["condition"], r["strength"], r["architecture"], r["seed"]): r for r in grid_runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in rows]
    provisional = sorted({r["run_id"] for r in rows.values() if r.get("provisional")})
    checks = [
        {"check": f"every cell of the grid is done ({len(expected) - len(missing)} of {len(expected)})", "passed": not missing},
        {"check": "no provisional (preview) cells", "passed": not provisional},
        {"check": "every cell's run folder is present", "passed": all(runs.find_run_dir(r["run_id"]) for r in rows.values())},
    ]
    summary = grid_runner.summarize(grid)
    chosen: dict = {}
    best_generative: dict = {}
    for arch in grid["architectures"]:
        chosen[arch] = {}
        for condition in grid["conditions"]:
            options = {key[1]: value for key, value in summary.items() if key[0] == condition and key[2] == arch}
            if not options:
                continue
            decision = select_winner([
                Candidate(name=str(strength), mean=v["mean"], sd=v["sd"], complexity=strength or 0.0, tie_mean=v["tie_break_mean"], tie_sd=v["tie_break_sd"])
                for strength, v in options.items()
            ])  # fmt: skip
            strength = next(s for s in options if str(s) == decision["winner"])
            chosen[arch][condition] = {"strength": strength, **options[strength], "decision": decision,
                                       "all_strengths": {str(s): {"mean": v["mean"], "sd": v["sd"]} for s, v in options.items()}}  # fmt: skip
        generative = [c for c in grid["conditions"] if "bank" in (grid["conditions"][c] or {}) and c in chosen[arch]]
        if generative:
            best_generative[arch] = select_winner([
                Candidate(name=c, mean=chosen[arch][c]["mean"], sd=chosen[arch][c]["sd"], complexity=0.0,
                          tie_mean=chosen[arch][c]["tie_break_mean"], tie_sd=chosen[arch][c]["tie_break_sd"]) for c in generative
            ])["winner"]  # fmt: skip
    return {"grid": grid, "checks": checks, "missing": [c.tag + f" {c.architecture} s{c.seed}" for c in missing],
            "chosen": chosen, "best_generative": best_generative, "passed": all(c["passed"] for c in checks)}  # fmt: skip


def phase4_gate_markdown(gate: dict) -> str:
    lines = ["# Phase 4 gate: validation summary of the full grid", "",
             f"Generated by `python -m ecgaug.workflow phase4-gate` on {_dt.date.today()}. Each condition enters the final",
             "evaluation at the strength with the best seed-mean validation selection metric (ties within max(0.02, pooled SD)",
             "go to the V F1 tie-break, then the lower strength; 'selection steps' lists who was tied at each step). 'Best generative'",
             "is the generative condition with the better validation metric,",
             "fixed here, before the test set is read (PAPER_TRACK, A5).", "",
             "## Checks", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in gate["checks"]], ""]  # fmt: skip
    if gate["missing"]:
        lines += ["Missing cells: " + ", ".join(gate["missing"][:20]) + (" ..." if len(gate["missing"]) > 20 else ""), ""]
    for arch, conditions in gate["chosen"].items():
        lines += [f"## {arch}", "", "| condition | chosen strength | mean | SD | V F1 | N recall | every strength | selection steps |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for condition, c in conditions.items():
            every = "; ".join(f"{s}: {v['mean']:.4f}" for s, v in c["all_strengths"].items())
            steps = " → ".join(f"{st['step']}: {', '.join(st['tied'])}" for st in c["decision"]["steps"] if "tied" in st)
            lines.append(f"| {condition} | {c['strength']} | {c['mean']:.4f} | {c['sd']:.4f} | {c['tie_break_mean']:.4f} | {c['n_recall']:.4f} | {every} | {steps} |")
        none = conditions.get("none")
        low = [condition for condition, c in conditions.items() if none and none["n_recall"] - c["n_recall"] > N_RECALL_REPORT_LIMIT]
        if low:
            lines += ["", f"**Read before approving:** N recall is more than {N_RECALL_REPORT_LIMIT} below `none` for: {', '.join(low)}."]
        if arch in gate["best_generative"]:
            lines += ["", f"Best generative condition ({arch}): **{gate['best_generative'][arch]}**"]
        lines.append("")
    lines += ["Status: " + ("PENDING APPROVAL. Approve with `python -m ecgaug.workflow phase4-gate --approve \"Name\" \"Name\"`, then run /final-eval."
                            if gate["passed"] else "NOT READY: fix the failed checks above."), ""]  # fmt: skip
    return "\n".join(lines)


def approve_phase4_gate(approvers: list[str]) -> dict:
    """Fix the strengths, the best generative condition and the checkpoints the final evaluation will test."""
    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE4_GATE):
        raise GateError("The phase 4 gate is already approved")
    gate = evaluate_phase4_gate()
    if not gate["passed"]:
        raise GateError("Phase 4 gate checks failed; run `python -m ecgaug.workflow phase4-gate` and read the report")
    chosen = {arch: {cond: {"strength": c["strength"], "run_ids": c["run_ids"], "checkpoints": _checkpoint_hashes(c["run_ids"]),
                            "validation_mean": c["mean"], "bank": c.get("bank")} for cond, c in conditions.items()}
              for arch, conditions in gate["chosen"].items()}  # fmt: skip
    entry = {"grid": str(FULL_GRID_FILE), "chosen": chosen, "best_generative": gate["best_generative"],
             "approved_by": list(approvers), "approved_on": str(_dt.date.today()), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_dir() / "manifest.json").write_text(json.dumps({**manifest(), PHASE4_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(f"\n## the phase 4 gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
                 "Strengths chosen on validation and best generative condition per architecture: "
                 + "; ".join(f"{a}: {g}" for a, g in entry["best_generative"].items())
                 + " — evidence: results/phase4_gate/report.md — rule: PLAN Phase 4, PAPER_TRACK A5\n"
                 "Frozen artifacts (hashes): the grid checkpoints in results/manifest.json. Next: /final-eval.\n")  # fmt: skip
    return entry


def _phase4_lines() -> tuple[list[str], str | None]:
    """Checklist lines for phase 4 and its first unfinished item."""
    lines, first = [], None

    def item(done: bool, text: str, todo: str) -> None:
        nonlocal first
        lines.append(_check(done, text, todo))
        if not done and first is None:
            first = f"{text}: {todo}"

    configs = experiment_configs(PHASE4_AUGMENTATION_POLICY)
    for letter, r in experiment_results(PHASE4_AUGMENTATION_POLICY).items():
        text = f"{PHASE4_AUGMENTATION_POLICY} variant {letter} ({configs[letter].stem})"
        if r:
            text += f": {r['selection_metric']} {r['mean']:.4f}, N recall drop {r['constraint']['n_recall_drop']:+.4f}"
            text += " (preview, not official)" if r.get("provisional") else ""
        item(bool(r) and not r.get("provisional"), text, f"kaggle/{PHASE4_AUGMENTATION_POLICY}.ipynb with VARIANTS = ['A', 'B', 'C', 'D'] in one session")
    item((paths.results_for(PHASE4A_GATE) / "report.md").is_file(), "phase 4a gate: report written", "python -m ecgaug.workflow phase4a-gate")
    item(bool(manifest().get(PHASE4A_GATE)), "phase 4a gate: approved, full grid written", 'python -m ecgaug.workflow phase4a-gate --approve "<names>"')
    grid_done = False
    if (paths.repo_root() / FULL_GRID_FILE).is_file():
        from ecgaug.grid import runner as grid_runner

        grid = grid_runner.load_grid(paths.repo_root() / FULL_GRID_FILE)
        total, done = len(grid_runner.cells(grid)), sum(r["status"] == "done" for r in grid_runner.read_ledger(grid).values())
        grid_done = done >= total
        item(grid_done, f"full grid: {done} of {total} cells done", f"kaggle/{PHASE4_FULL_GRID}.ipynb (resumable)")
    else:
        item(False, "full grid", "written when the phase 4a gate is approved")
    item((paths.results_for(PHASE4_GATE) / "report.md").is_file(), "phase 4 gate: validation summary written", "python -m ecgaug.workflow phase4-gate")
    item(bool(manifest().get(PHASE4_GATE)), "phase 4 gate: approved, checkpoints fixed", 'python -m ecgaug.workflow phase4-gate --approve "<names>"')
    item((paths.results_for(FINAL_EVAL) / "summary.json").is_file(), "final evaluation on the test set (once)", "/final-eval (python -m ecgaug.final_eval)")
    rq3 = paths.results_for(PHASE4_QUALITY_VS_GAIN) / "summary.json"  # analyses: listed, never the next action
    rq3_done = rq3.is_file() and not json.loads(rq3.read_text())["provisional"]
    lines.append(_check(rq3_done, "RQ3: quality metrics against the downstream gain", "python -m ecgaug.eval.rq3 (after the phase 3 runs)"))
    lines.append(_check((paths.results_for(PHASE4_LEAKAGE_AUDIT) / "summary.json").is_file(), "leakage audit (optional, after the final evaluation)",
                        f"kaggle/{PHASE4_LEAKAGE_AUDIT}.ipynb, then python -m ecgaug.experiments.{PHASE4_LEAKAGE_AUDIT} --config configs/{PHASE4_LEAKAGE_AUDIT}/audit.yaml --part report"))  # fmt: skip
    return lines, first


# --------------------------------------------------------------------------------------------------
# phase 5: explanation method per architecture, note variant, demo
# --------------------------------------------------------------------------------------------------


def note_ratings() -> dict[str, float]:
    """Mean blinded readability rating per note variant, if the team has filled the rating sheet."""
    folder = paths.results_for(PHASE5_CLINICAL_NOTES) / "rating"
    ratings, key = folder / "ratings.csv", folder / "key.csv"
    if not (ratings.is_file() and key.is_file()):
        return {}
    import csv as _csv

    with key.open(newline="") as fh:
        variant_of = {row["item"]: row["variant"] for row in _csv.DictReader(fh)}
    scores: dict[str, list[float]] = {}
    with ratings.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            values = [float(row[k]) for k in ("clarity_1_5", "accuracy_of_emphasis_1_5", "brevity_1_5") if row.get(k)]
            if values and row["item"] in variant_of:
                scores.setdefault(variant_of[row["item"]], []).append(sum(values) / len(values))
    return {v: sum(s) / len(s) for v, s in scores.items()}


def saliency_choice(explain: dict | None = None) -> dict:
    """Saliency method per architecture by the pre-registered rule (faithfulness, ties: stability, then cost).

    Empty while any phase5_explainability variant has no result. The phase 5 gate applies the same rule; the
    notes cases use it before the gate so that the notes describe the method the gate will approve.
    """
    explain = experiment_results(PHASE5_EXPLAINABILITY) if explain is None else explain
    if not explain or any(r is None for r in explain.values()):
        return {}
    methods = {}
    for arch in sorted({a for r in explain.values() for a in r["per_architecture"]}):
        candidates = [
            Candidate(name=letter, mean=r["per_architecture"][arch]["mean_gain"], sd=r["per_architecture"][arch]["sd_gain"], complexity=r["cost"],
                      tie_mean=r["per_architecture"][arch]["mean_stability"], tie_sd=0.0)  # fmt: skip
            for letter, r in explain.items() if r["per_architecture"].get(arch, {}).get("applicable")
        ]
        decision = select_winner(candidates)
        methods[arch] = {"variant": decision["winner"], "method": explain[decision["winner"]]["method"], "decision": decision}
    return methods


def evaluate_phase5_gate() -> dict:
    """Saliency method per architecture by faithfulness (ties: stability); note variant by the hard gate, coverage and ratings."""
    explain = experiment_results(PHASE5_EXPLAINABILITY)
    notes = experiment_results(PHASE5_CLINICAL_NOTES)
    for name, results in ((PHASE5_EXPLAINABILITY, explain), (PHASE5_CLINICAL_NOTES, notes)):
        missing = [letter for letter, r in results.items() if r is None]
        if not results or missing:
            raise GateError(f"{name} variants not run yet: {', '.join(missing) or 'all'}")
    methods = saliency_choice(explain)
    ratings = note_ratings()
    template_ok = bool(notes.get("A")) and notes["A"].get("passes_hard_gate", False)
    eligible = [l for l, r in notes.items() if l != "A" and not r.get("not_run") and r.get("passes_hard_gate")]
    ranked = sorted(eligible, key=lambda l: (-notes[l]["coverage"], -ratings.get(l, 0.0), notes[l]["latency_s_mean"], notes[l]["output_tokens"]))
    notes_variant = ranked[0] if ranked else "A"
    from ecgaug import demo

    try:
        demo_check = {"passed": demo.smoke()["unsupported_claims"] == 0, "detail": "python -m ecgaug.demo --smoke"}
    except Exception as err:  # noqa: BLE001 (reported in the gate)
        demo_check = {"passed": False, "detail": str(err)}
    checks = [
        {"check": "every explainability variant scored", "passed": True},
        {"check": "the template note (A) has zero unsupported claims (offline fallback)", "passed": template_ok},
        {"check": f"chosen note variant {notes_variant} has zero unsupported claims", "passed": notes_variant == "A" and template_ok or notes_variant in eligible},
        {"check": "the demo runs (smoke)", "passed": demo_check["passed"]},
        {"check": "no provisional results", "passed": not any(r.get("provisional") for r in (*explain.values(), *notes.values()))},
    ]  # fmt: skip
    return {"explain": explain, "notes": notes, "methods": methods, "notes_variant": notes_variant, "ratings": ratings,
            "checks": checks, "passed": all(c["passed"] for c in checks)}  # fmt: skip


def phase5_gate_markdown(gate: dict) -> str:
    lines = ["# Phase 5 gate: explanations, notes, demo", "", f"Generated by `python -m ecgaug.workflow phase5-gate` on {_dt.date.today()}.", "",
             "## Checks", "", *[f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in gate["checks"]], "", "## Saliency method per architecture", "",
             "| variant | method | " + " | ".join(f"{a} gain | {a} stability" for a in gate["methods"]) + " |",
             "|---|---|" + "---|---|" * len(gate["methods"])]  # fmt: skip
    for letter, r in gate["explain"].items():
        cells = []
        for arch in gate["methods"]:
            e = r["per_architecture"].get(arch, {})
            cells += [f"{e['mean_gain']:+.4f}", f"{e['mean_stability']:.3f}".replace("nan", "undefined")] if e.get("applicable") else ["n/a", "n/a"]
        lines.append(f"| {letter} | {r['method']} | " + " | ".join(cells) + " |")
    for arch, m in gate["methods"].items():
        lines += ["", *[f"- {arch}, {_step_text(step)}" for step in m["decision"]["steps"]], f"**{arch}: variant {m['variant']} ({m['method']})**"]
    lines += ["", "## Notes", "",
              "| variant | ran | unsupported claims | coverage | fell back | mean latency (s) | output tokens | rating |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for letter, r in gate["notes"].items():
        if r.get("not_run"):
            lines.append(f"| {letter} | no ({r['reason']}) | | | | | | |")
        else:
            lines.append(f"| {letter} | yes | {r['unsupported_claims']} | {r['coverage']:.2f} | {r['fell_back_to_template']} | {r['latency_s_mean']:.2f} | "
                         f"{r['output_tokens']} | {gate['ratings'].get(letter, '-')} |")  # fmt: skip
    lines += ["", f"**Note variant: {gate['notes_variant']}**", "",
              "Status: " + ("PENDING APPROVAL. Approve with `python -m ecgaug.workflow phase5-gate --approve \"Name\" \"Name\"`, then freeze the code: `git tag v1.0`."
                            if gate["passed"] else "NOT READY: fix the failed checks above."), ""]  # fmt: skip
    return "\n".join(lines)


def approve_phase5_gate(approvers: list[str]) -> dict:
    if not approvers:
        raise GateError("Name at least one approver")
    if manifest().get(PHASE5_GATE):
        raise GateError("The phase 5 gate is already approved")
    gate = evaluate_phase5_gate()
    if not gate["passed"]:
        raise GateError("Phase 5 gate checks failed; run `python -m ecgaug.workflow phase5-gate` and read the report")
    entry = {"explainability": {a: m["method"] for a, m in gate["methods"].items()}, "notes_variant": gate["notes_variant"],
             "approved_by": list(approvers), "approved_on": str(_dt.date.today()), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_dir() / "manifest.json").write_text(json.dumps({**manifest(), PHASE5_GATE: entry}, indent=2) + "\n")
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    with (paths.docs_dir() / "DECISIONS.md").open("a") as fh:
        fh.write(f"\n## the phase 5 gate — {entry['approved_on']} — status: APPROVED by {', '.join(approvers)}\n"
                 f"Saliency: {entry['explainability']}; notes: variant {entry['notes_variant']} — evidence: results/phase5_gate/report.md — "
                 "rule: PLAN Phase 5\nNext: code freeze (git tag v1.0) and phase 6.\n")  # fmt: skip
    return entry


def _phase3_lines() -> tuple[list[str], str | None]:
    """Checklist lines for phase 3 and its first unfinished item."""
    lines, first = [], None

    def item(done: bool, text: str, todo: str) -> None:
        nonlocal first
        lines.append(_check(done, text, todo))
        if not done and first is None:
            first = f"{text}: {todo}"

    for experiment in GENERATOR_EXPERIMENTS:
        configs = experiment_configs(experiment)
        for letter, r in experiment_results(experiment).items():
            text = f"{experiment} variant {letter} ({configs[letter].stem}) pilot"
            if r:
                text += f": {r['selection_metric']} {r['mean']:.4f} ± {r['sd']:.4f}, hard gates {'pass' if r['hard_gates']['passed'] else 'fail'}"
                text += " (preview, not official)" if r.get("provisional") else ""
            item(bool(r) and not r.get("provisional"), text, f"kaggle/{experiment}.ipynb with VARIANTS = ['{letter}']")
    report = (paths.results_for(PHASE3A_GATE) / "report.md").is_file()
    chosen = manifest().get(PHASE3A_GATE)
    item(report, "phase 3a gate: report written", "python -m ecgaug.workflow phase3a-gate")
    item(bool(chosen), "phase 3a gate: approved, one generator per family", 'python -m ecgaug.workflow phase3a-gate --approve "<names>"')
    for experiment in GENERATOR_EXPERIMENTS:
        choice = (chosen or {}).get(experiment)
        final = paths.results_for(experiment) / f"{choice['winner']}_final.json" if choice else None
        done = bool(final and final.is_file() and not json.loads(final.read_text())["provisional"])
        label = f"{experiment} final runs and banks" + (f" (variant {choice['winner']})" if choice else "")
        item(done, label, f"kaggle/{experiment}.ipynb with MODE = 'final'")
    item(bool(manifest().get(PHASE3_GATE)), "phase 3 gate: banks frozen", 'python -m ecgaug.workflow phase3-gate --approve "<names>"')
    return lines, first


def _check(done: bool, text: str, todo: str = "") -> str:
    return f"  [{'x' if done else ' '}] {text}" + (f"  ->  {todo}" if todo and not done else "")


def status(today: _dt.date | None = None) -> tuple[list[str], str]:
    """Checklist lines for every step, and the single next action."""
    today = today or _dt.date.today()
    day = (today - DAY_ONE).days + 1
    planned = next((n for n, last in LAST_DAY.items() if day <= last), 6)
    lines = [f"Day {day} of 30. By plan we should be in phase {planned} ({PHASES[planned]})."]

    raw = paths.results_for(PHASE1_DATA) / "raw_checksums.json"
    data_ok = raw.is_file() and not json.loads(raw.read_text())["problems"]
    split_ok = (paths.configs_dir() / "validation_split" / "approved.yaml").is_file()
    results = phase1_results()
    gate_report = (paths.results_for(PHASE1_GATE) / "report.md").is_file()
    winner = phase1_winner()

    lines += ["", f"Phase 1  {PHASES[1]}"]
    lines.append(_check(data_ok, "MIT-BIH downloaded and verified", "python -m ecgaug.data.raw"))
    lines.append(_check(split_ok, "validation split approved", "write configs/validation_split/approved.yaml"))
    for letter, r in results.items():
        text = f"{PHASE1_INPUT_REPRESENTATION} variant {letter} ({variant_configs()[letter].stem})"
        if r:
            text += f": {r['selection_metric']} {r['mean']:.4f} ± {r['sd']:.4f}"
        config = variant_configs()[letter].relative_to(paths.repo_root())
        lines.append(_check(r is not None, text, f"python -m ecgaug.experiments.phase1_input_representation --config {config}"))
    lines.append(_check(gate_report, "phase 1 gate: report written", "python -m ecgaug.workflow phase1-gate"))
    lines.append(_check(winner is not None, "phase 1 gate: approved, dataset frozen" + (f" (winner {winner})" if winner else ""),
                        'ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "<names>"'))  # fmt: skip

    lines += ["", f"Phase 2  {PHASES[2]}"]
    folder = paths.results_for(PRETRAIN_RESULTS)
    pretrain = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("[A-Z].json"))} if folder.is_dir() else {}
    official = winner is not None and winner in pretrain and not pretrain[winner].get("provisional", True)
    previews = sorted(k for k, v in pretrain.items() if v.get("provisional", True))
    text = f"{PRETRAIN_RESULTS} for the phase 1 winner"
    if previews:
        text += f" (preview runs, not official: {', '.join(previews)})"
    lines.append(_check(official, text, "python -m ecgaug.experiments.phase2_ptbxl_pretraining --config <winner's config>"))
    phase2_items = []
    for experiment in (PHASE2_BASELINE_CLASSIFIERS, PHASE2_QUALITY_METRICS):
        configs = experiment_configs(experiment)
        for letter, r in experiment_results(experiment).items():
            text = f"{experiment} variant {letter} ({configs[letter].stem})"
            if r and experiment == PHASE2_BASELINE_CLASSIFIERS:
                text += f": {r['selection_metric']} {r['mean']:.4f} ± {r['sd']:.4f}"
            elif r:
                text += f": {'passes' if r['passed'] else 'fails'} the sanity harness"
            text += " (preview, not official)" if r and r.get("provisional") else ""
            done = bool(r) and not r.get("provisional")
            lines.append(_check(done, text, f"kaggle/{experiment}.ipynb with VARIANTS = ['{letter}']"))
            phase2_items.append((done, f"Run {experiment} variant {letter} (kaggle/{experiment}.ipynb)"))
    phase2_report = (paths.results_for(PHASE2_GATE) / "report.md").is_file()
    phase2_gate = bool(manifest().get(PHASE2_GATE))
    lines.append(_check(phase2_report, "phase 2 gate: report written", "python -m ecgaug.workflow phase2-gate"))
    lines.append(_check(phase2_gate, "phase 2 gate: approved (cnn, rnn, quality composite)", 'python -m ecgaug.workflow phase2-gate --approve "<names>"'))
    phase2_items += [(phase2_report, "Write the phase 2 gate report: python -m ecgaug.workflow phase2-gate"),
                     (phase2_gate, 'Team reads results/phase2_gate/report.md, then: python -m ecgaug.workflow phase2-gate --approve "<names>"')]  # fmt: skip
    phase3, phase3_next = _phase3_lines()
    phase4, phase4_next = _phase4_lines()
    phase5 = []
    for experiment in (PHASE5_EXPLAINABILITY, PHASE5_CLINICAL_NOTES):
        configs = experiment_configs(experiment)
        for letter, r in experiment_results(experiment).items():
            phase5.append(_check(bool(r) and not r.get("not_run"), f"{experiment} variant {letter} ({configs[letter].stem})", f"python -m ecgaug.experiments.{experiment} --config {configs[letter].relative_to(paths.repo_root())}"))
    phase5.append(_check(bool(manifest().get(PHASE5_GATE)), "phase 5 gate: approved (method, notes, demo)", "python -m ecgaug.workflow phase5-gate"))
    lines += ["", f"Phase 3  {PHASES[3]}", *phase3, "", f"Phase 4  {PHASES[4]}", *phase4, "", f"Phase 5  {PHASES[5]}", *phase5,
              "", f"Phase 6  {PHASES[6]}: python -m ecgaug.paper (tables and numbers for the draft in paper/)"]  # fmt: skip

    if not data_ok:
        nxt = "Download and verify MIT-BIH: python -m ecgaug.data.raw"
    elif not split_ok:
        nxt = "Approve a validation split (configs/validation_split/approved.yaml)"
    elif any(r is None for r in results.values()):
        todo = [letter for letter, r in results.items() if r is None]
        nxt = f"Run phase1_input_representation variants {', '.join(todo)} (kaggle/phase1_input_representation.ipynb, VARIANTS = {todo})"
    elif not gate_report:
        nxt = "Write the phase 1 gate report: python -m ecgaug.workflow phase1-gate"
    elif winner is None:
        nxt = 'Team reads results/phase1_gate/report.md, then: ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "<names>"'
    elif not phase2_gate:
        nxt = next(text for done, text in phase2_items if not done)
        if not official:
            nxt += f" (and {PRETRAIN_RESULTS} for the phase 1 winner, variant {winner})"
    else:
        nxt = phase3_next or phase4_next or "Phase 4 is done: phase 5 (phase5_explainability, phase5_clinical_notes)"
    lines += ["", f"NEXT: {nxt}"]
    return lines, nxt


def _phase3_command(command: str, approve: list[str] | None) -> int:
    """The phase 2, 3a and 3 gates (they need no test access)."""
    try:
        if command == "phase2-gate":
            if approve:
                entry = approve_phase2_gate(approve)
                print(f"Phase 2 gate approved by {', '.join(entry['approved_by'])}: cnn {entry['cnn']['variant']}, "
                      f"rnn {entry['rnn']['variant']}, composite {', '.join(entry['quality_composite']) or 'none'}.")  # fmt: skip
            else:
                gate = write_phase2_gate_report()
                print(phase2_gate_markdown(gate))
                print("Written to results/phase2_gate/report.md. Approve only after the team has read it.")
        elif command in ("phase4a-gate", "phase4-gate", "phase5-gate"):
            folder, evaluate, markdown, approve_fn = {
                "phase4a-gate": (PHASE4A_GATE, evaluate_phase4a_gate, phase4a_gate_markdown, approve_phase4a_gate),
                "phase4-gate": (PHASE4_GATE, evaluate_phase4_gate, phase4_gate_markdown, approve_phase4_gate),
                "phase5-gate": (PHASE5_GATE, evaluate_phase5_gate, phase5_gate_markdown, approve_phase5_gate),
            }[command]
            if approve:
                approve_fn(approve)
                print(f"{command} approved by {', '.join(approve)}; recorded in results/manifest.json and docs/DECISIONS.md.")
            else:
                gate = evaluate()
                out = paths.results_for(folder)
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.md").write_text(markdown(gate))
                print(markdown(gate))
                if command == "phase4-gate":
                    from ecgaug import figures

                    figures.safe_call(figures.phase4_strength_figure, gate)
        elif command == "phase3a-gate":
            if approve:
                entry = approve_phase3a_gate(approve)
                print(f"Phase 3a gate approved by {', '.join(entry['approved_by'])}: "
                      + "; ".join(f"{e} variant {entry[e]['winner']} ({entry[e]['setting']})" for e in GENERATOR_EXPERIMENTS))  # fmt: skip
                print("Next: run each winner with --final (kaggle notebooks, MODE = 'final').")
            else:
                gate = write_phase3a_gate_report()
                print(phase3a_gate_markdown(gate))
                print("Written to results/phase3a_gate/report.md. Approve only after the team has read it.")
        else:
            if approve:
                entry = approve_phase3_gate(approve)
                print(f"Phase 3 gate approved by {', '.join(entry['approved_by'])}; bank hashes are in results/manifest.json.")
            else:
                gate = evaluate_phase3_gate()
                out = paths.results_for(PHASE3_GATE)
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.md").write_text(phase3_gate_markdown(gate))
                print(phase3_gate_markdown(gate))
    except GateError as err:
        print(f"{command}: {err}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project workflow: status and the gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="checklist of every step and the next action")
    gate = sub.add_parser("phase1-gate", help="compare phase1_input_representation variants; with --approve, freeze the winner's dataset")
    gate.add_argument("--approve", nargs="+", metavar="NAME", help="approvers' names (needs ALLOW_TEST_BUILD=1)")
    for name, text in (("phase2-gate", "choose the cnn and the rnn and fix the quality composite; with --approve, record them"),
                       ("phase3a-gate", "choose one generator per family from the pilots; with --approve, record it"),
                       ("phase3-gate", "check the final banks; with --approve, freeze them"),
                       ("phase4a-gate", "choose the augmentation policy; with --approve, record it and write the full grid"),
                       ("phase4-gate", "validation summary of the full grid; with --approve, fix what the final evaluation tests"),
                       ("phase5-gate", "saliency method per architecture, note variant, demo; with --approve, record them")):  # fmt: skip
        sub.add_parser(name, help=text).add_argument("--approve", nargs="+", metavar="NAME", help="approvers' names")
    args = parser.parse_args(argv)

    if args.command == "status":
        lines, _ = status()
        print("\n".join(lines))
        return 0
    if args.command in ("phase2-gate", "phase3a-gate", "phase3-gate", "phase4a-gate", "phase4-gate", "phase5-gate"):
        return _phase3_command(args.command, args.approve)
    try:
        if args.approve:
            entry = approve_phase1_gate(args.approve)
            print(f"Phase 1 gate approved by {', '.join(entry['approved_by'])}: variant {entry['winner']} ({entry['config']}).")
            print("Dataset frozen in data/processed/v1; hashes in results/manifest.json.")
            print("Commit results/ and docs/, then tag: git tag gate-1 && git push --tags")
        else:
            gate = write_phase1_gate_report()
            print(phase1_gate_markdown(gate))
            print("Written to results/phase1_gate/report.md. Approve only after the team has read it.")
    except GateError as err:
        print(f"Phase 1 gate: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
