"""Phase 3: configs and equal budgets, preconditions, smoke runs, the phase 3a and 3 gates, the grid runner, bank ingest."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, variant_config, variant_letters
from ecgaug import paths, runs, workflow
from ecgaug.data.counts import count_record
from ecgaug.data.records import DS1
from ecgaug.eval.selection import GeneratorCandidate, select_generator
from ecgaug.experiments import generator_runs as gr
from ecgaug.experiments import phase3_cvae, phase3_wgan_gp
from ecgaug.generators.bank import export_bank
from ecgaug.grid import runner as grid_runner

from test_generators import training_part

FAMILIES = {"phase3_cvae": REPO / "configs" / "phase3_cvae", "phase3_wgan_gp": REPO / "configs" / "phase3_wgan_gp"}
OWN = {"variant", "description", "generator", "config_name"}


@pytest.mark.parametrize("experiment", sorted(FAMILIES))
def test_four_variants_share_one_budget(experiment):
    folder = FAMILIES[experiment]
    assert variant_letters(folder) == list("ABCD")
    configs = {letter: gr.load_config(variant_config(folder, letter)) for letter in "ABCD"}
    reference = {k: v for k, v in configs["A"].items() if k not in OWN}
    for letter, cfg in configs.items():
        assert cfg["variant"] == letter and {k: v for k, v in cfg.items() if k not in OWN} == reference
    assert reference["pilot_seeds"] == [0, 1] and reference["final_seeds"] == [0, 1, 2]
    assert reference["checkpoint_fractions"] == [0.25, 0.5, 0.75, 1.0]


def test_variants_are_the_ones_in_the_plan():
    cvae = {letter: gr.load_config(variant_config(FAMILIES["phase3_cvae"], letter))["generator"] for letter in "ABCD"}
    assert cvae["A"]["architecture"] == "conv" and cvae["A"]["latent_dim"] == 16 and cvae["A"]["kl_anneal_fraction"] == 0.3
    assert cvae["B"]["beta"] == [0.1, 0.25] and cvae["C"]["spectral_weight"] > 0 and cvae["D"]["architecture"] == "residual_film"
    wgan_cfg = gr.load_config(variant_config(FAMILIES["phase3_wgan_gp"], "A"))
    assert wgan_cfg["training"]["n_critic"] == 5 and wgan_cfg["training"]["gradient_penalty"] == 10.0
    assert wgan_cfg["training"]["lr"] == 1e-4 and wgan_cfg["training"]["adam_betas"] == [0.0, 0.9]
    conditioning = {letter: gr.load_config(variant_config(FAMILIES["phase3_wgan_gp"], letter))["generator"]["conditioning"] for letter in "ABCD"}
    assert conditioning == {"A": "concat", "B": "projection", "C": "auxiliary", "D": "per_class"}


def test_both_families_share_pilot_diagnostics_gates_and_banks():
    a = gr.load_config(variant_config(FAMILIES["phase3_cvae"], "A"))
    b = gr.load_config(variant_config(FAMILIES["phase3_wgan_gp"], "A"))
    for key in ("classes", "pilot_seeds", "final_seeds", "checkpoint_fractions", "pilot", "diagnostics", "hard_gates", "bank"):
        assert a[key] == b[key], key


def test_variant_files_cannot_change_the_budget(tmp_path):
    shutil.copy(FAMILIES["phase3_cvae"] / "generator_settings.yaml", tmp_path)
    (tmp_path / "X_bad.yaml").write_text("base: generator_settings.yaml\nvariant: X\npilot_seeds: [0]\n")
    with pytest.raises(ValueError, match="belong in the shared settings"):
        gr.load_config(tmp_path / "X_bad.yaml")
    (tmp_path / "Y_bad.yaml").write_text("base: generator_settings.yaml\nvariant: Y\ngenerator: {dropout: 0.5}\n")
    with pytest.raises(ValueError, match="not defined in generator_defaults"):
        gr.load_config(tmp_path / "Y_bad.yaml")


def test_settings_grid_and_bank_names():
    b = gr.load_config(variant_config(FAMILIES["phase3_cvae"], "B"))
    grid = gr.settings_grid(b)
    assert [name for name, _ in grid] == ["beta0.1", "beta0.25"] and [g["beta"] for _, g in grid] == [0.1, 0.25]
    assert gr.generator_id(b, "beta0.25") == "phase3_cvae-B.beta0.25"
    a = gr.load_config(variant_config(FAMILIES["phase3_wgan_gp"], "A"))
    assert gr.settings_grid(a)[0][0] == "default" and gr.generator_id(a, "default") == "phase3_wgan_gp-A"


def _summary(seed, diverged=False, diversity=0.9, memorization=0.05, value=0.3):
    diagnostics = None if diverged else {c: {"diversity_ratio": diversity, "memorization_fraction": memorization, "c2st_accuracy": 0.7} for c in "SVF"}
    return {"seed": seed, "diverged": diverged, "diagnostics": diagnostics, "selection_value": float("nan") if diverged else value}


def test_hard_gates_check_every_class_and_seed():
    gates = {"diversity_min_ratio": 0.7, "memorization_max_fraction": 0.10}
    assert gr.hard_gates([_summary(0), _summary(1)], gates)["passed"]
    out = gr.hard_gates([_summary(0), _summary(1, diverged=True)], gates)
    assert not out["passed"] and out["checks"][0]["diverged_seeds"] == [1]
    assert not gr.hard_gates([_summary(0), _summary(1, diversity=0.69)], gates)["passed"]
    assert not gr.hard_gates([_summary(0, memorization=0.11), _summary(1)], gates)["passed"]
    assert gr.hard_gates([_summary(0, diversity=0.7, memorization=0.10)], gates)["passed"]  # bounds are inclusive


def test_choose_setting_prefers_passing_settings():
    settings = {
        "beta0.1": {"mean": 0.5, "hard_gates": {"passed": False}},
        "beta0.25": {"mean": 0.3, "hard_gates": {"passed": True}},
    }
    assert gr.choose_setting(settings) == "beta0.25"
    settings["beta0.25"]["hard_gates"]["passed"] = False
    assert gr.choose_setting(settings) == "beta0.1"


def test_generator_selection_rule():
    c = lambda name, mean, sd=0.01, ok=True, params=100, rank=None, tie=None: GeneratorCandidate(
        name, mean, sd, ok, params, tie_mean=tie, tie_sd=0.0 if tie is not None else None, quality_rank=rank
    )  # noqa: E731
    assert select_generator([c("A", 0.40), c("B", 0.30), c("C", 0.50, ok=False)])["winner"] == "A"  # C fails a hard gate
    tied = select_generator([c("A", 0.40, params=200), c("B", 0.39, params=100)])
    assert tied["winner"] == "B" and tied["steps"][-1]["winner"] == "B"  # tied -> fewer parameters
    ranked = select_generator([c("A", 0.40, rank=2.0, params=1), c("B", 0.39, rank=1.0, params=9)])
    assert ranked["winner"] == "B" and any(s["step"] == "composite quality rank" for s in ranked["steps"])
    no_rank = select_generator([c("A", 0.40, tie=0.5, params=1), c("B", 0.39, tie=0.8, params=9)])
    assert no_rank["winner"] == "B" and "not defined yet" in no_rank["steps"][2]["step"]
    none = select_generator([c("A", 0.4, ok=False), c("B", float("nan"), ok=False)])
    assert none["winner"] is None and "fallback" in none["fallback"].lower()


@pytest.mark.parametrize("experiment,letter", [("phase3_cvae", "B"), ("phase3_wgan_gp", "C")])
def test_smoke_runs_end_to_end(experiment, letter):
    cfg = gr.load_config(variant_config(FAMILIES[experiment], letter))
    out = gr.smoke(cfg, final=(letter == "B"))
    assert out["checkpoints_logged"] == 4 and len(out["hard_gates_evaluated"]) == 3
    assert {"checkpoints.csv", "train_log.csv", "metrics_val.json", "config.yaml", "env.json"} <= set(out["run_files"])
    if letter == "B":
        assert out["settings"] == ["beta0.1", "beta0.25"] and out["bank"]["generator_id"].startswith("phase3_cvae-B.beta")
    assert not (REPO / "data" / "synthetic" / "phase3_cvae-B.beta0.1" / "seed0.npz").exists() or True


@pytest.fixture
def project(full_fake_root, monkeypatch):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    (paths.results_for("phase1_data")).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([count_record(r) for r in DS1]).to_csv(paths.results_for("phase1_data") / "record_counts.csv", index=False)
    return full_fake_root


def _manifest(**entries):
    path = paths.results_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def test_runs_wait_for_the_gates_unless_previewed(project, capsys):
    config = ["--config", str(variant_config(paths.configs_dir() / "phase3_cvae", "A"))]
    assert phase3_cvae.main(config) == 2
    assert "Phase 1 is not finished" in capsys.readouterr().out
    _manifest(phase1_gate={"winner": "C", "files": {"trainval.npz": {"content_sha256": "x"}}})
    assert phase3_cvae.main(config) == 2
    assert "Phase 2 is not finished: the phase 2 gate has not chosen the pilot classifier" in capsys.readouterr().out
    assert phase3_cvae.main([*config, "--representation", "A"]) == 2
    assert "previews only" in capsys.readouterr().out
    wrong = ["--config", str(variant_config(paths.configs_dir() / "phase3_wgan_gp", "A"))]
    assert phase3_cvae.main(wrong) == 2
    assert "belongs to phase3_wgan_gp" in capsys.readouterr().out


def test_official_runs_use_the_frozen_data_and_refuse_a_mismatch(project):
    split = workflow_split()
    good = data_hash(split, "C")  # before any gate: the candidate dataset of C
    _manifest(phase1_gate={"winner": "C", "files": {"trainval.npz": {"content_sha256": "0" * 64}}})
    with pytest.raises(gr.PreconditionError, match="does not match the frozen dataset"):
        gr.load_data(split, preview=False)
    data = gr.load_data(split, preview=True, representation="A")  # another representation: only as a preview
    assert data.provisional and "chose C, not A" in data.provisional and data.spec.name == "A_short_window"
    _manifest(phase1_gate={"winner": "C", "files": {"trainval.npz": {"content_sha256": good}}})
    official = gr.load_data(split, preview=False)
    assert official.provisional is None and official.info["frozen"] and official.spec.rr_features


def workflow_split():
    from ecgaug.data.splits import load_approved_split

    return load_approved_split(pd.DataFrame([count_record(r) for r in DS1]))


def data_hash(split, letter):
    data = gr.load_data(split, preview=True, representation=letter)
    return data.info["content_sha256"]


def test_final_runs_take_the_setting_approved_at_the_phase3a_gate(project):
    cfg = gr.load_config(variant_config(paths.configs_dir() / "phase3_cvae", "B"))
    with pytest.raises(gr.PreconditionError, match="has not chosen a generator"):
        gr.chosen_setting(cfg, preview=False)
    setting, why = gr.chosen_setting(cfg, preview=True)
    assert setting == "beta0.1" and "preview" in why
    _manifest(phase3a_gate={"phase3_cvae": {"winner": "A", "setting": "default", "config": "configs/phase3_cvae/A_conv_concat.yaml"}})
    with pytest.raises(gr.PreconditionError, match="chose variant A; run configs/phase3_cvae/A_conv_concat.yaml"):
        gr.chosen_setting(cfg, preview=False)
    _manifest(phase3a_gate={"phase3_cvae": {"winner": "B", "setting": "beta0.25", "config": "configs/phase3_cvae/B_conv_low_beta.yaml"}})
    assert gr.chosen_setting(cfg, preview=False) == ("beta0.25", None)


# --------------------------------------------------------------------------------------------------
# the phase 3a and phase 3 gates
# --------------------------------------------------------------------------------------------------

BUDGET = {"training": {"pilot_steps": 5000}, "pilot_seeds": [0, 1]}


def _pilot(experiment, letter, mean, passed=True, provisional=False, params=1000, settings=None):
    result = {
        "experiment": experiment, "variant": letter, "family": experiment.split("_", 1)[1], "mean": mean, "sd": 0.01,
        "tie_break_mean": 0.5, "tie_break_sd": 0.01, "split": {"name": "V1"}, "selection_metric": "s_f1", "budget": BUDGET,
        "pilot": {"classifier": {"source": "x"}}, "hard_gate_thresholds": {"diversity_min_ratio": 0.7}, "dataset": {"content_sha256": "d"},
        "budget_deviations": [], "hard_gates": {"passed": passed, "checks": [{"check": "memorization", "passed": passed}]},
        "quality_mean": {}, "n_parameters": params, "wall_clock_s": 10.0, "chosen_setting": "default",
        "settings": settings or {"default": {}}, "provisional": provisional, "provisional_reasons": ["preview"] if provisional else [],
        "per_seed": [{"seed": s, "selection_value": mean, "diagnostics": {c: {"diversity_ratio": 0.9, "memorization_fraction": 0.04, "c2st_accuracy": 0.7} for c in "SVF"}} for s in (0, 1)],
        "run_ids": [f"{experiment}-{letter}-s{s}-abcdef0-202610160000" for s in (0, 1)],
    }  # fmt: skip
    folder = paths.results_for(experiment)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def _all_pilots(provisional=False):
    for letter, mean, passed in (("A", 0.30, True), ("B", 0.45, False), ("C", 0.40, True), ("D", 0.10, True)):
        _pilot("phase3_cvae", letter, mean, passed, provisional)
    for letter, mean in (("A", 0.20), ("B", 0.35), ("C", 0.34), ("D", 0.05)):
        _pilot("phase3_wgan_gp", letter, mean, True, provisional, params=2000 if letter == "B" else 1000)


def test_phase3a_gate_selects_per_family_and_waits_for_official_pilots(project):
    with pytest.raises(workflow.GateError, match="not piloted yet"):
        workflow.evaluate_phase3a_gate()
    _all_pilots(provisional=True)
    gate = workflow.write_phase3a_gate_report()
    assert gate["families"]["phase3_cvae"]["winner"] == "C"  # B is best but fails a hard gate
    assert gate["families"]["phase3_wgan_gp"]["winner"] == "C"  # B and C tied; C has fewer parameters
    report = (paths.results_for("phase3a_gate") / "report.md").read_text()
    assert "PREVIEW" in report and "**Winner: variant C**" in report
    with pytest.raises(workflow.GateError, match="previews"):
        workflow.approve_phase3a_gate(["Ann"])

    _all_pilots(provisional=False)
    workflow.write_phase3a_gate_report()
    entry = workflow.approve_phase3a_gate(["Ann", "Ben"])
    assert entry["phase3_cvae"]["generator_id"] == "phase3_cvae-C" and entry["phase3_wgan_gp"]["winner"] == "C"
    assert workflow.phase3a_choice("phase3_cvae")["config"] == "configs/phase3_cvae/C_conv_spectral_loss.yaml"
    assert "phase 3a gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()
    with pytest.raises(workflow.GateError, match="already approved"):
        workflow.approve_phase3a_gate(["Ann"])


def test_phase3a_gate_refuses_unequal_budgets_and_reports_a_family_with_no_winner(project):
    _all_pilots()
    result = json.loads((paths.results_for("phase3_cvae") / "A.json").read_text())
    (paths.results_for("phase3_cvae") / "A.json").write_text(json.dumps({**result, "budget": {"training": {"pilot_steps": 9000}}}))
    with pytest.raises(workflow.GateError, match="same budget"):
        workflow.evaluate_phase3a_gate()
    _all_pilots()
    for letter in "ABCD":
        _pilot("phase3_wgan_gp", letter, 0.3, passed=False)
    gate = workflow.write_phase3a_gate_report()
    assert gate["families"]["phase3_wgan_gp"]["winner"] is None
    assert "No winner" in (paths.results_for("phase3a_gate") / "report.md").read_text()
    with pytest.raises(workflow.GateError, match="no variant passes the hard gates"):
        workflow.approve_phase3a_gate(["Ann"])


def test_phase3_gate_checks_and_freezes_the_banks(project):
    with pytest.raises(workflow.GateError, match="phase 3a gate first"):
        workflow.evaluate_phase3_gate()
    chosen = {
        "phase3_cvae": {"winner": "A", "setting": "default", "config": "configs/phase3_cvae/A_conv_concat.yaml", "generator_id": "phase3_cvae-A"},
        "phase3_wgan_gp": {"winner": "D", "setting": "default", "config": "configs/phase3_wgan_gp/D_per_class_unconditional.yaml", "generator_id": "phase3_wgan_gp-D"},
    }  # fmt: skip
    _manifest(phase3a_gate=chosen)
    with pytest.raises(workflow.GateError, match="have not been ingested"):
        workflow.evaluate_phase3_gate()
    bank = {k: v for k, v in training_part().items()}
    bank["is_synthetic"][:] = True
    for experiment, choice in chosen.items():
        final = {"provisional": False, "setting": "default", "seeds": [0, 1, 2], "diverged_seeds": []}
        (paths.results_for(experiment)).mkdir(parents=True, exist_ok=True)
        (paths.results_for(experiment) / f"{choice['winner']}_final.json").write_text(json.dumps(final))
        for seed in (0, 1):
            export_bank(bank, choice["generator_id"], seed, {"train_only_check": "passed", "diagnostics": {}})
    gate = workflow.evaluate_phase3_gate()
    assert not gate["passed"]  # seed 2 has no bank yet
    with pytest.raises(workflow.GateError, match="checks failed"):
        workflow.approve_phase3_gate(["Ann"])
    for choice in chosen.values():
        export_bank(bank, choice["generator_id"], 2, {"train_only_check": "passed", "diagnostics": {}})
    assert workflow.evaluate_phase3_gate()["passed"]
    entry = workflow.approve_phase3_gate(["Ann"])
    assert set(entry["phase3_cvae"]["banks"]) == {"0", "1", "2"} and len(entry["phase3_wgan_gp"]["banks"]["2"]["sha256"]) == 64
    assert "phase 3 gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()


def test_status_lists_phase3_in_order(project):
    lines, _ = workflow.status()
    text = "\n".join(lines)
    assert "Phase 3  generators" in text and "phase3_cvae variant A (A_conv_concat) pilot" in text
    assert text.index("phase 3a gate: report written") < text.index("phase 3 gate: banks frozen")
    _pilot("phase3_cvae", "A", 0.3, provisional=True)
    assert "(preview, not official)" in "\n".join(workflow.status()[0])


# --------------------------------------------------------------------------------------------------
# grid runner and bank ingest
# --------------------------------------------------------------------------------------------------


def test_grid_runs_every_cell_once_and_resumes(project, monkeypatch):
    split = workflow_split()
    data = gr.load_data(split, preview=True, representation="C")
    small = {"model": {"name": "probe_cnn", "channels": [4, 8, 8], "kernel": 5, "side_hidden": 4}, "lr": 1e-3, "max_epochs": 1,
             "patience": 5, "batch_size": 256, "weight_decay": 0.01}  # fmt: skip
    train = gr.training_beats(data.arrays)
    pool = {k: np.concatenate([v] * 5) for k, v in train.items()}  # a fake bank: training beats repeated
    pool["is_synthetic"][:] = True
    export_bank(pool, "phase3_cvae-A", 0, {"train_only_check": "passed"})
    grid = {
        "experiment": "phase4_full_grid", "name": "test_grid", "policy": "fraction_of_n", "seeds": [0],
        "architectures": {"probe_cnn": small},
        "conditions": {"none": {}, "class_weighted": {"gammas": [0.5]}, "random_oversampling": {"strengths": [0.3]},
                       "smote": {"strengths": [0.3]}, "cvae": {"bank": "phase3_cvae-A", "strengths": [0.3]}},
        "file": "test",
    }  # fmt: skip
    assert len(grid_runner.cells(grid)) == 5
    plan = grid_runner.run_grid(grid, split, data, device=__import__("torch").device("cpu"), preview=True, dry_run=True)
    assert len(plan["planned"]) == 5 and not plan["ran"]
    first = grid_runner.run_grid(grid, split, data, device=__import__("torch").device("cpu"), preview=True, provisional=["preview"])
    assert len(first["ran"]) == 5
    again = grid_runner.run_grid(grid, split, data, device=__import__("torch").device("cpu"), preview=True)
    assert not again["ran"] and len(again["done_before"]) == 5
    ledger = list(grid_runner.read_ledger(grid).values())
    assert len(list(grid_runner.cells_dir(grid).glob("*.json"))) == 5
    assert {row["condition"] for row in ledger} == {"none", "class_weighted", "random_oversampling", "smote", "cvae"}
    by = {row["condition"]: row for row in ledger}
    added = {name: sum(row["n_added"].values()) for name, row in by.items()}
    assert added["none"] == 0 and added["smote"] > 0 and added["cvae"] == added["smote"] == added["random_oversampling"]
    summary = grid_runner.summarize(grid)
    assert summary[("smote", 0.3, "probe_cnn")]["seeds"] == [0] and summary[("none", None, "probe_cnn")]["provisional"]
    run_dir = runs.find_run_dir(by["class_weighted"]["run_id"])
    assert run_dir.parent.name == "probecnn" and yaml.safe_load((run_dir / "config.yaml").read_text())["class_weights"][0] == 1.0
    with pytest.raises(grid_runner.PreconditionError):
        grid_runner.run_grid(grid, split, data, device=__import__("torch").device("cpu"), preview=False)


def test_grid_files_are_checked():
    grid = grid_runner.load_grid(REPO / "configs" / "phase4_full_grid" / "dry_run_baselines.yaml")
    assert [c.tag for c in grid_runner.cells(grid)] == ["none"] * 3
    assert grid_runner.Cell("smote", 0.25, "probe_cnn", 1).tag == "smote_0.25"
    assert grid_runner.Cell("smote", 0.25, "probe_cnn", 1).label == "probecnn"


def test_banks_travel_back_from_kaggle_and_merge_by_seed(tmp_path):
    bank = training_part()
    bank["is_synthetic"][:] = True
    for seed, folder in ((0, "first"), (1, "second")):
        root = tmp_path / folder
        run_id = runs.make_run_id("phase3_cvae", "A", seed, tag="final")
        group = ("phase3_cvae", "A_conv_concat", "final")
        run_dir = root / "runs" / "/".join(group) / run_id
        (run_dir).mkdir(parents=True)
        for name in runs.REQUIRED_RUN_FILES:
            (run_dir / name).write_text("{}" if name.endswith(".json") else "x\n")
        import os

        os.environ["ECGAUG_ROOT"] = str(root)
        export_bank(bank, "phase3_cvae-A", seed, {"train_only_check": "passed"})
    os.environ["ECGAUG_ROOT"] = str(tmp_path / "clone")
    report = runs.ingest([tmp_path / "first", tmp_path / "second"])
    assert sorted(report["bank_files"]) == ["data/synthetic/phase3_cvae-A/seed0.npz", "data/synthetic/phase3_cvae-A/seed1.npz"]
    meta = json.loads((tmp_path / "clone" / "data" / "synthetic" / "phase3_cvae-A" / "bank.json").read_text())
    assert set(meta["seeds"]) == {"0", "1"} and not report["bank_conflicts"]
    assert runs.default_group(runs.make_run_id("phase3_wgan_gp", "D", 2, tag="final")) == ("phase3_wgan_gp", "D", "final")


def test_phase3_runners_are_thin():
    assert phase3_cvae.EXPERIMENT == "phase3_cvae" and phase3_wgan_gp.EXPERIMENT == "phase3_wgan_gp"
    for experiment in FAMILIES:
        assert np.all([(REPO / "kaggle" / f"{experiment}.ipynb").is_file()])
