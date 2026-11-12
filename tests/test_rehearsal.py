"""Dress rehearsal: every phase, gate, the final evaluation, phase 5, RQ3, the audit and the paper, in order.

    ECGAUG_REHEARSAL=1 uv run pytest tests/test_rehearsal.py -s

Runs the real command-line entry points one after another on fake records, in a temporary root, with
the budgets of a copy of the configs shrunk (epochs, steps, seeds) and the generator hard gates
relaxed so that untrained generators pass. It checks that each stage accepts what the previous one
wrote; the numbers mean nothing. Skipped unless ECGAUG_REHEARSAL=1 (several minutes on a CPU).
"""

import json
import os
import shutil

import pytest
import yaml
from conftest import REPO

from ecgaug import paths, runs, workflow

pytestmark = pytest.mark.skipif(os.environ.get("ECGAUG_REHEARSAL") != "1", reason="set ECGAUG_REHEARSAL=1 to run the dress rehearsal")

APPROVERS = ["Rehearsal"]


def _edit(relative: str, change) -> None:
    path = paths.configs_dir() / relative
    cfg = yaml.safe_load(path.read_text())
    change(cfg)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _shrink() -> None:
    def classifier(c: dict) -> None:
        c.update(max_epochs=2, patience=1, **({"min_epochs": 1} if "min_epochs" in c else {}))

    def budget(c: dict) -> None:
        c.update(seeds=[0, 1])
        classifier(c)

    _edit("phase1_input_representation/shared_budget.yaml", budget)
    _edit("phase2_baseline_classifiers/shared_budget.yaml", budget)

    def quality(c: dict) -> None:
        c["resamples"] = 2
        c["metrics"].update(dtw_max_beats=20, fid_max_beats=100)
        c["metrics"]["tstr_classifier"].update(max_epochs=1)
        c["feature_net"].update(max_epochs=1)

    _edit("phase2_quality_metrics/metric_settings.yaml", quality)
    for family in ("phase3_cvae", "phase3_wgan_gp"):

        def generator(c: dict) -> None:
            c["training"].update(pilot_steps=8, final_steps=8, batch_size=16, log_every=2)
            c.update(final_seeds=[0, 1])
            c["diagnostics"].update(max_beats=40, c2st_folds=2)
            c["hard_gates"].update(diversity_min_ratio=0.0, memorization_max_fraction=1.0)  # untrained generators must pass
            classifier(c["pilot"]["preview_classifier"])

        _edit(f"{family}/generator_settings.yaml", generator)

    def explain(c: dict) -> None:
        c.update(beats_per_class=12, synthetic_beats_per_class=12, random_orders=2)
        c["method_settings"].update(ig_steps=4, smoothgrad_samples=3)

    _edit("phase4_augmentation_policy/pilot_settings.yaml", lambda c: c.update(seeds=[0, 1]))  # the banks exist for final_seeds only
    _edit("phase5_explainability/shared_settings.yaml", explain)
    _edit("phase5_clinical_notes/shared_settings.yaml", lambda c: c.update(n_cases=12, min_low_confidence=2))
    _edit("phase5_clinical_notes/shared_settings.yaml", lambda c: c["method_settings"].update(ig_steps=4))


def _cli(main, *argv: str) -> None:
    code = main(list(argv))
    assert code in (0, None), f"{main.__module__} {' '.join(argv)} exited with {code}"


def _configs(experiment: str) -> list[str]:
    return [str(p) for p in workflow.experiment_configs(experiment).values()]


def test_dress_rehearsal(full_fake_root, monkeypatch, capsys):
    from test_workflow import _approve_split

    from ecgaug import final_eval, paper
    from ecgaug.eval import rq3
    from ecgaug.experiments import (
        phase1_input_representation,
        phase2_baseline_classifiers,
        phase2_quality_metrics,
        phase3_cvae,
        phase3_wgan_gp,
        phase4_augmentation_policy,
        phase4_leakage_audit,
        phase5_clinical_notes,
        phase5_explainability,
    )
    from ecgaug.grid import runner

    shutil.copytree(REPO / "configs", paths.configs_dir(), dirs_exist_ok=True)
    (paths.repo_root() / workflow.FULL_GRID_FILE).unlink()  # the repository holds the real one; the phase 4a gate below writes it again
    _shrink()
    _approve_split()
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    gate = workflow.main

    # phase 1
    from ecgaug.data import counts

    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    counts.main(["--include-test"])  # exits 1 here: fake records do not have the literature's counts
    monkeypatch.delenv("ALLOW_TEST_BUILD")
    assert (paths.results_for("phase1_data") / "record_counts.csv").is_file()
    for config in _configs("phase1_input_representation"):
        _cli(phase1_input_representation.main, "--config", config, "--device", "cpu")
    _cli(gate, "phase1-gate")
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    _cli(gate, "phase1-gate", "--approve", *APPROVERS)
    monkeypatch.delenv("ALLOW_TEST_BUILD")

    # phase 2 (phase2_ptbxl_pretraining needs PTB-XL and is optional for the gate)
    for config in _configs("phase2_baseline_classifiers"):
        _cli(phase2_baseline_classifiers.main, "--config", config, "--device", "cpu")
    for config in _configs("phase2_quality_metrics"):
        _cli(phase2_quality_metrics.main, "--config", config, "--device", "cpu")
    _cli(gate, "phase2-gate")
    _cli(gate, "phase2-gate", "--approve", *APPROVERS)

    # phase 3
    for module, experiment in ((phase3_cvae, "phase3_cvae"), (phase3_wgan_gp, "phase3_wgan_gp")):
        for config in _configs(experiment):
            _cli(module.main, "--config", config, "--device", "cpu")
    _cli(gate, "phase3a-gate")
    _cli(gate, "phase3a-gate", "--approve", *APPROVERS)
    for module, experiment in ((phase3_cvae, "phase3_cvae"), (phase3_wgan_gp, "phase3_wgan_gp")):
        letter = workflow.manifest()["phase3a_gate"][experiment]["winner"]
        _cli(module.main, "--config", str(workflow.experiment_configs(experiment)[letter]), "--final", "--device", "cpu")
    _cli(gate, "phase3-gate")
    _cli(gate, "phase3-gate", "--approve", *APPROVERS)

    # phase 4
    for config in _configs("phase4_augmentation_policy"):
        _cli(phase4_augmentation_policy.main, "--config", config, "--device", "cpu")
    _cli(gate, "phase4a-gate")
    _cli(gate, "phase4a-gate", "--approve", *APPROVERS)
    _cli(runner.main, "--config", str(paths.repo_root() / workflow.FULL_GRID_FILE), "--device", "cpu")
    _cli(gate, "phase4-gate")
    _cli(gate, "phase4-gate", "--approve", *APPROVERS)
    _cli(final_eval.main, "--plan")
    monkeypatch.setenv("ALLOW_TEST", "1")
    _cli(final_eval.main, "--confirm", final_eval.CONFIRMATION, "--n-boot", "200")
    monkeypatch.delenv("ALLOW_TEST")
    assert final_eval.main(["--confirm", final_eval.CONFIRMATION]) != 0  # only once

    # the two analyses
    _cli(rq3.main, "--n-boot", "100")
    monkeypatch.setenv("ALLOW_TEST_AUDIT", "1")
    audit_config = str(paths.configs_dir() / "phase4_leakage_audit" / "audit.yaml")
    _edit("phase4_leakage_audit/audit.yaml", lambda c: c.update(seeds=[0, 1]))
    _cli(phase4_leakage_audit.main, "--config", audit_config, "--device", "cpu")
    monkeypatch.delenv("ALLOW_TEST_AUDIT")

    # phase 5
    for config in _configs("phase5_explainability"):
        _cli(phase5_explainability.main, "--config", config)
    for config in _configs("phase5_clinical_notes"):
        _cli(phase5_clinical_notes.main, "--config", config)
    from ecgaug import demo
    from ecgaug.report import error_analysis

    _cli(error_analysis.main)
    _cli(error_analysis.main, "--test")
    _cli(demo.main, "--record", "118", "--beat", "3")
    _cli(gate, "phase5-gate")
    _cli(gate, "phase5-gate", "--approve", *APPROVERS)

    # phase 6
    report = paper.build()
    capsys.readouterr()
    assert report["final_evaluation"]
    preregistered = {"table_dataset", "table_input_representation", "table_baselines", "table_quality_metrics", "table_generators", "table_policy",
                     "table_quality_vs_gain", "table_leakage_audit", "table_explainability", "final_eval_conditions.tex", "final_eval_comparisons.tex",
                     "results_rq1", "results_rq2", "results_rq3", "results_rq4", "results_notes", "results_concentration"}  # fmt: skip
    left = sorted(preregistered & set(report["missing"]))
    assert left == [], f"paper tables still waiting for results: {left}"
    # Not rehearsed: the follow-up analyses P1-P23 (their texts stay marked missing, never invented), the LLM note variants
    # (table_notes) and the report's example, which needs a confident correct S and V case from two-epoch models.
    others = set(report["missing"]) - preregistered - {"table_notes", "report_example"}
    assert all("posthoc" in m or m.startswith("results_") for m in others), f"missing beyond the follow-up analyses: {others}"
    lines, nxt = workflow.status()
    print("\n".join(lines))
    print(json.dumps(report, indent=2))
