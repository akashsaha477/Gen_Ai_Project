"""Phase 4: augmentation policies, the quality screen, the phase 4a and 4 gates, and the final evaluation."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import REPO, variant_config, variant_letters
from ecgaug import augment, final_eval, paths, runs, workflow
from ecgaug.data.counts import count_record
from ecgaug.data.records import CLASS_INDEX, DS1
from ecgaug.experiments import phase4_augmentation_policy as policy
from ecgaug.experiments.data_source import smoke_data
from ecgaug.generators.bank import bank_counts, export_bank
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner as grid_runner

from test_generators import training_part

POLICIES = REPO / "configs" / "phase4_augmentation_policy"
QUALITY = {"dtw_candidates": 5, "dtw_neighbours": 2, "dtw_window_fraction": 0.1}


def test_policy_targets_hand_computed():
    counts = {"N": 1000, "S": 100, "V": 600, "F": 10}
    assert augment.policy_targets("balance_to_n", None, counts) == {"S": 900, "V": 400, "F": 990}
    assert augment.policy_targets("fraction_of_n", 0.5, counts) == {"S": 400, "V": 0, "F": 490}
    assert augment.policy_targets("fraction_of_n_screened", 0.25, counts) == {"S": 150, "V": 0, "F": 240}
    assert augment.policy_targets("multiple_of_real", 2.0, counts) == {"S": 200, "V": 1200, "F": 20}
    with pytest.raises(ValueError):
        augment.policy_targets("everything", 1.0, counts)


def test_quality_screen_keeps_the_beats_closest_to_real_ones():
    train = training_part(n=200)
    rng = np.random.default_rng(0)
    real_s = train["X"][train["y"] == CLASS_INDEX["S"]]
    clean = real_s[:10] + 0.01 * rng.standard_normal((10, real_s.shape[1]))
    noisy = real_s[:10] + 2.0 * rng.standard_normal((10, real_s.shape[1]))
    bank = augment.random_oversample(train, {"S": 20}, seed=0)
    bank["X"] = np.concatenate([noisy, clean]).astype(np.float32)  # bank order: the noisy ones first
    kept = augment.screened_from_bank(bank, {"S": 10}, train, pool_factor=2.0, settings=QUALITY)
    np.testing.assert_array_equal(kept["X"], clean.astype(np.float32))


def test_policy_variants_are_the_ones_in_the_plan():
    assert variant_letters(POLICIES) == list("ABCD")
    cfgs = {letter: policy.load_config(variant_config(POLICIES, letter)) for letter in "ABCD"}
    assert [cfgs[l]["policy"]["kind"] for l in "ABCD"] == ["balance_to_n", "fraction_of_n", "multiple_of_real", "fraction_of_n_screened"]
    assert cfgs["B"]["grid_strengths"] == [0.1, 0.25, 0.5] and cfgs["C"]["grid_strengths"] == [1.0, 2.0, 4.0]
    assert [cfgs[l]["complexity"] for l in "ABCD"] == [1, 2, 2, 3] and cfgs["A"]["max_n_recall_drop"] == 0.01


def test_policy_smoke_run():
    out = policy.smoke(policy.load_config(variant_config(POLICIES, "C")))
    assert out["banks"] == ["cvae", "wgan_gp"] and out["constraint_evaluated"]
    assert all(r.startswith("phase4_augmentation_policy-C.cnn.") for r in out["run_ids"])  # the variant letter keeps B and D runs apart


@pytest.fixture
def project(tmp_root):
    shutil.copytree(REPO / "configs", paths.configs_dir(), ignore=shutil.ignore_patterns("full_grid.yaml"))  # written by the 4a gate
    return tmp_root


def _manifest(**entries):
    path = paths.results_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**current, **entries}))


def _policy_result(letter, mean, drop, complexity, provisional=False, sd=0.0, bank_drops=(0.0, 0.0)):
    cfg = policy.load_config(variant_config(paths.configs_dir() / "phase4_augmentation_policy", letter))
    result = {
        "experiment": "phase4_augmentation_policy", "variant": letter, "policy": cfg["policy"], "pilot_strength": cfg["pilot_strength"],
        "grid_strengths": cfg["grid_strengths"], "complexity": complexity, "classifier": "cnn", "seeds": [0, 1, 2], "mean": mean, "sd": sd,
        "tie_break_mean": 0.5, "tie_break_sd": 0.0, "per_bank": {"cvae": {"bank": "b1", "selection_value": mean, "n_recall_drop": bank_drops[0], "tie_break_drop": 0.1},
                                                       "wgan_gp": {"bank": "b2", "selection_value": mean, "n_recall_drop": bank_drops[1], "tie_break_drop": 0.0}},
        "reference": {"run_ids": ["ref"], "selection_value": 0.2, "n_recall": 0.97}, "dataset": {"content_sha256": "d"},
        "constraint": {"max_n_recall_drop": 0.01, "n_recall_drop": drop, "passed": drop <= 0.01}, "provisional": provisional,
        "selection_metric": "s_f1",
    }  # fmt: skip
    folder = paths.results_for("phase4_augmentation_policy")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def test_phase4a_gate_applies_the_n_recall_limit_and_writes_the_full_grid(project):
    for letter, mean, drop, complexity in (("A", 0.50, 0.03, 1), ("B", 0.40, 0.005, 2), ("C", 0.39, 0.0, 2), ("D", 0.30, 0.0, 3)):
        _policy_result(letter, mean, drop, complexity)
    gate = workflow.evaluate_phase4a_gate()
    assert gate["winner"] == "B" and gate["excluded"] == {"A": 0.03}  # A is best but lowers N recall too much; B and C tie, B first by name
    entry = workflow.approve_phase4a_gate(["Ann"])
    grid = grid_runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    assert entry["policy"]["kind"] == "fraction_of_n" and grid["seeds"] == [0, 1, 2]
    assert set(grid["conditions"]) == {"none", "class_weighted", "random_oversampling", "smote", "cvae", "wgan_gp"}
    assert grid["conditions"]["cvae"] == {"bank": "phase3_gate/phase3_cvae", "strengths": [0.1, 0.25, 0.5]}
    assert len(grid_runner.cells(grid)) == (1 + 5 * 3) * 2 * 3  # 96 runs at 3 seeds (PLAN Phase 4)
    with pytest.raises(workflow.GateError, match="already approved"):
        workflow.approve_phase4a_gate(["Ann"])


def test_phase4a_gate_uses_the_seed_spread_as_tie_margin(project):
    # 0.05 apart: not tied at the 0.02 floor, tied once the seeds spread by 0.08, so the simpler policy wins
    for sd, winner in ((0.0, "B"), (0.08, "A")):
        for letter, mean, complexity in (("A", 0.40, 1), ("B", 0.45, 2), ("C", 0.30, 2), ("D", 0.30, 3)):
            _policy_result(letter, mean, 0.0, complexity, sd=sd)
        assert workflow.evaluate_phase4a_gate()["winner"] == winner


def test_phase4a_report_flags_a_bank_the_average_hides(project):
    for letter in "ABCD":
        _policy_result(letter, 0.4, 0.002, 1, bank_drops=(-0.011, 0.015) if letter == "B" else (0.0, 0.0))
    gate = workflow.evaluate_phase4a_gate()
    assert "B" not in gate["excluded"]  # the rule is the average over the banks
    assert "B with wgan_gp (+0.0150)" in workflow.phase4a_gate_markdown(gate)


def test_policy_pilot_runs_every_seed():
    cfg = policy.load_config(variant_config(POLICIES, "B"))
    assert cfg["seeds"] == [0, 1, 2]
    pilot, reference = policy.pilot_grids(cfg, "phase2_gate", {"cvae": "b1", "wgan_gp": "b2"})
    assert len(grid_runner.cells(pilot)) == 6 and len(grid_runner.cells(reference)) == 3


def test_policy_score_is_the_seed_mean_of_the_bank_mean():
    cfg = policy.load_config(variant_config(POLICIES, "C"))
    small = {**cfg, "seeds": [0, 1], "preview_classifier": {**cfg["preview_classifier"], "max_epochs": 1}}
    with smoke_data() as (split, data):
        train = training_beats(data.arrays)
        fake = augment.random_oversample(train, bank_counts(train["y"]), seed=0)
        fake["is_synthetic"][:] = True
        for name in ("b1", "b2"):
            for seed in (0, 1):
                export_bank(fake, name, seed, {"train_only_check": "passed"})
        result = policy.run_variant(small, split, data, "cpu", True, ["test"], {"cvae": "b1", "wgan_gp": "b2"})
    per_seed = np.mean([result["per_bank"][k]["per_seed"] for k in ("cvae", "wgan_gp")], axis=0)
    assert result["mean"] == pytest.approx(per_seed.mean()) and result["sd"] == pytest.approx(per_seed.std(ddof=1))
    assert len(result["reference"]["run_ids"]) == 2


def test_phase4a_gate_refuses_previews(project):
    for letter in "ABCD":
        _policy_result(letter, 0.3, 0.0, 1, provisional=True)
    with pytest.raises(workflow.GateError, match="Provisional"):
        workflow.approve_phase4a_gate(["Ann"])


def _ledger_row(grid, condition, strength, arch, seed, value, run_id=None):
    sha = hashlib.sha256(f"{condition}|{strength}|{arch}|{seed}".encode()).hexdigest()
    row = {"cell_sha": sha, "condition": condition, "strength": strength, "architecture": arch,
           "seed": seed, "run_id": run_id or f"phase4_full_grid-{arch}.{condition}-s{seed}-abcdef0-202610220000", "status": "done",
           "selection_value": value, "tie_break_value": 0.5, "n_recall": 0.97, "bank": None, "provisional": False}  # fmt: skip
    grid_runner.write_cell(grid, row)


def test_phase4_gate_chooses_strengths_on_validation(project, monkeypatch):
    _manifest(phase4a_gate={"winner": "B"})
    (paths.repo_root() / workflow.FULL_GRID_FILE).write_text(workflow.full_grid_yaml({"kind": "fraction_of_n"}, [0.1, 0.25, 0.5], [0.5, 0.75, 1.0], [0, 1]))
    grid = grid_runner.load_grid(paths.repo_root() / workflow.FULL_GRID_FILE)
    monkeypatch.setattr(runs, "find_run_dir", lambda run_id: paths.repo_root())  # every run folder "exists"
    gate = workflow.evaluate_phase4_gate()
    assert not gate["passed"] and len(gate["missing"]) == len(grid_runner.cells(grid))
    value = {"none": 0.20, "class_weighted": 0.25, "random_oversampling": 0.22, "smote": 0.24, "cvae": 0.30, "wgan_gp": 0.28}
    for cell in grid_runner.cells(grid):
        bonus = 0.05 if cell.strength == 0.25 else 0.0
        _ledger_row(grid, cell.condition, cell.strength, cell.architecture, cell.seed, value[cell.condition] + bonus + 0.001 * cell.seed)
    gate = workflow.evaluate_phase4_gate()
    assert gate["passed"]
    assert gate["chosen"]["cnn"]["cvae"]["strength"] == 0.25 and gate["chosen"]["cnn"]["none"]["strength"] is None
    assert gate["best_generative"] == {"cnn": "cvae", "rnn": "cvae"}
    monkeypatch.setattr(workflow, "_checkpoint_hashes", lambda run_ids: {r: "0" * 64 for r in run_ids})
    entry = workflow.approve_phase4_gate(["Ann"])
    assert entry["chosen"]["rnn"]["smote"]["strength"] == 0.25 and "phase 4 gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()


def _fake_preds(n_records=6, beats=60, seeds=2, boost=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0] * 40 + [1] * 10 + [2] * 6 + [3] * 4), n_records)
    records = np.repeat(np.array([200, 202, 210, 213, 222, 232][:n_records]), beats)
    probs = []
    for s in range(seeds):
        logits = rng.standard_normal((len(y), 4))
        logits[np.arange(len(y)), y] += 1.0 + boost
        probs.append(torch.softmax(torch.as_tensor(logits), 1).numpy())
    return probs, y, records


def test_statistics_follow_the_preregistered_plan():
    probs_none, y, records = _fake_preds(seed=1)
    probs_cvae, _, _ = _fake_preds(boost=1.5, seed=2)
    probs_wgan, _, _ = _fake_preds(boost=0.5, seed=3)
    probs_smote, _, _ = _fake_preds(boost=0.2, seed=4)
    preds = {("cnn", "none"): probs_none, ("cnn", "cvae"): probs_cvae, ("cnn", "wgan_gp"): probs_wgan, ("cnn", "smote"): probs_smote}
    gate = {"chosen": {"cnn": {c: {"strength": 0.25} for c in ("none", "cvae", "wgan_gp", "smote")}}, "best_generative": {"cnn": "cvae"}}
    stats = final_eval.statistics(preds, y, records, gate, n_boot=300)
    confirmatory = stats["families"]["confirmatory"]
    assert set(confirmatory) == {"cnn: cvae vs none", "cnn: wgan_gp vs none"}
    assert set(stats["families"]["secondary"]) == {"cnn: cvae vs wgan_gp", "cnn: cvae vs smote"}
    strong = confirmatory["cnn: cvae vs none"]
    assert strong["point"] > 0 and strong["ci_low"] > 0 and strong["p_holm"] >= strong["p_two_sided"]
    assert 202 not in [r for r in stats["records"] if False] and "without_record_202" in stats["sensitivity"]
    assert set(stats["sensitivity"]["leave_one_record_out"]) == {"200", "202", "210", "213", "222", "232"}
    assert stats["conditions"]["cnn/cvae"]["rare_macro_f1"]["n_replicates"] == 300
    assert stats["per_record"]["cnn/none"]["213"]["F"]["support"] == 4


def _frozen_project(full_fake_root, monkeypatch):
    """A temporary project with the phase 1 gate approved (fake data frozen as v1, test part included)."""
    from test_workflow import _approve_split, _result

    shutil.copytree(REPO / "configs", paths.configs_dir(), dirs_exist_ok=True)
    _approve_split()
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    for letter in "ABCD":
        _result(letter, {"A": 0.05, "B": 0.08, "C": 0.40, "D": 0.03}[letter])
    workflow.write_phase1_gate_report()
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    workflow.approve_phase1_gate(["Ann"])
    monkeypatch.delenv("ALLOW_TEST_BUILD")


def test_final_evaluation_runs_once_after_the_phase4_gate(full_fake_root, monkeypatch, capsys):
    from ecgaug.experiments.data_source import load_data
    from ecgaug.data.splits import load_approved_split

    _frozen_project(full_fake_root, monkeypatch)
    with pytest.raises(final_eval.FinalEvalError, match="phase 4 gate is not approved"):
        final_eval.plan()
    split = load_approved_split(pd.DataFrame([count_record(r) for r in DS1]))
    data = load_data(split, preview=False)
    train = training_beats(data.arrays)
    fake = augment.random_oversample(train, bank_counts(train["y"]), seed=0)
    fake["is_synthetic"][:] = True
    for gid in ("phase3_cvae-A", "phase3_wgan_gp-B"):
        for seed in (0, 1):
            export_bank(fake, gid, seed, {"train_only_check": "passed"})
    small = {"model": {"name": "probe_cnn", "channels": [4, 8, 8], "kernel": 5, "side_hidden": 4}, "lr": 1e-3, "max_epochs": 1,
             "patience": 5, "batch_size": 256, "weight_decay": 0.01}  # fmt: skip
    grid = {"experiment": "phase4_full_grid", "name": "fake_full_grid", "policy": {"kind": "fraction_of_n"}, "seeds": [0, 1],
            "architectures": {"cnn": small}, "file": "fake",
            "conditions": {"none": {}, "class_weighted": {"gammas": [0.5]}, "random_oversampling": {"strengths": [0.3]},
                           "smote": {"strengths": [0.3]}, "cvae": {"bank": "phase3_cvae-A", "strengths": [0.3]},
                           "wgan_gp": {"bank": "phase3_wgan_gp-B", "strengths": [0.3]}}}  # fmt: skip
    grid_runner.run_grid(grid, split, data, torch.device("cpu"), preview=True)
    chosen = {"cnn": {}}
    for (condition, strength, arch), value in grid_runner.summarize(grid).items():
        chosen[arch][condition] = {"strength": strength, "run_ids": value["run_ids"], "checkpoints": workflow._checkpoint_hashes(value["run_ids"])}
    _manifest(phase4_gate={"chosen": chosen, "best_generative": {"cnn": "cvae"}, "approved_by": ["Ann", "Ben"]})

    info = final_eval.plan()
    assert len(info["checkpoints"]) == 12 and all(c["hash_matches"] for c in info["checkpoints"])
    assert len(info["comparisons"]["confirmatory"]) == 2 and len(info["comparisons"]["secondary"]) == 4
    log = paths.docs_dir() / "TEST_ACCESS_LOG.md"
    before = log.read_text()
    with pytest.raises(final_eval.FinalEvalError, match="Confirm with exactly"):
        final_eval.run("yes")
    with pytest.raises(final_eval.FinalEvalError, match="ALLOW_TEST=1"):
        final_eval.run(final_eval.CONFIRMATION)
    assert log.read_text() == before  # nothing was read

    monkeypatch.setenv("ALLOW_TEST", "1")
    stats = final_eval.run(final_eval.CONFIRMATION, n_boot=200)
    assert set(stats["families"]) == {"confirmatory", "secondary"} and "| evaluation |" in log.read_text()
    assert all((runs.find_run_dir(r) / "preds_test.npz").is_file() for r in stats["run_ids"])
    assert (paths.results_for("final_eval") / "summary.json").is_file() and (paths.repo_root() / "tables" / "final_eval_comparisons.tex").is_file()
    numbers = (paths.repo_root() / "paper" / "generated" / "numbers_final_eval.tex").read_text()
    assert "\\newcommand{\\TestCnnCvaeVsNoneDiff}" in numbers
    with pytest.raises(final_eval.FinalEvalError, match="already holds an evaluation entry"):
        final_eval.run(final_eval.CONFIRMATION, n_boot=200)
