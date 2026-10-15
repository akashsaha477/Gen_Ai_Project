"""The workflow: status checklist, phase 1 gate report and approval, and the phase order for pretraining."""

import datetime as dt
import json
import shutil

import pandas as pd
import pytest
import yaml

from conftest import REPO, variant_config
from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.counts import count_record
from ecgaug.data.records import DS1
from ecgaug.data.splits import Split, selection_metric_for
from ecgaug.experiments import phase2_ptbxl_pretraining as pretrain

BUDGET = {"seeds": [0, 1, 2], "lr_grid": [0.001, 0.0003], "max_epochs": 40, "patience": 5, "batch_size": 256, "weight_decay": 0.01}
INPUTS = {"A": 180, "B": 360, "C": 183, "D": 63}


@pytest.fixture
def project(tmp_root):
    """The real configs inside the temporary root."""
    shutil.copytree(REPO / "configs", paths.configs_dir())
    return tmp_root


def _result(letter, mean, sd=0.0, tie=0.3, **override):
    result = {
        "variant": letter, "mean": mean, "sd": sd, "tie_break_mean": tie, "tie_break_sd": 0.0,
        "complexity_inputs_per_beat": INPUTS[letter], "wall_clock_s": 60.0, "selection_metric": "s_f1",
        "split": {"name": "V1", "val": [115, 118, 201, 205, 223]}, "budget": BUDGET, "budget_deviations": [],
        "per_seed": [{"selection_value": mean, "seed": s} for s in (0, 1, 2)], "run_ids": [f"run-{letter}-{s}" for s in (0, 1, 2)],
    }  # fmt: skip
    result.update(override)
    folder = paths.results_for("phase1_input_representation")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def test_gate1_needs_all_four_variants(project):
    _result("A", 0.2)
    with pytest.raises(workflow.GateError, match="not run yet: B, C, D"):
        workflow.evaluate_phase1_gate()


def test_gate1_applies_the_rule_and_writes_the_report(project):
    for letter, mean in {"A": 0.05, "B": 0.08, "C": 0.40, "D": 0.03}.items():
        _result(letter, mean)
    gate = workflow.write_phase1_gate_report()
    assert gate["winner"] == "C"
    report = (paths.results_for("phase1_gate") / "report.md").read_text()
    assert "Winner: variant C" in report and "PENDING APPROVAL" in report and "| C | 183 | 0.4000 |" in report
    assert json.loads((paths.results_for("phase1_gate") / "report.json").read_text())["winner"] == "C"


def test_gate1_tie_goes_to_the_simpler_variant(project):
    for letter, mean in {"A": 0.395, "B": 0.10, "C": 0.40, "D": 0.20}.items():
        _result(letter, mean, tie=0.3)
    assert workflow.evaluate_phase1_gate()["winner"] == "A"  # 0.005 apart: tied; A has 180 inputs, C 183


def test_gate1_refuses_unequal_budgets_or_splits(project):
    for letter in "ABC":
        _result(letter, 0.2)
    _result("D", 0.3, budget={**BUDGET, "max_epochs": 100})
    with pytest.raises(workflow.GateError, match="same budget"):
        workflow.evaluate_phase1_gate()
    _result("D", 0.3, split={"name": "V2", "val": [208]})
    with pytest.raises(workflow.GateError, match="same validation split"):
        workflow.evaluate_phase1_gate()


def _approve_split():
    counts = pd.DataFrame([count_record(r) for r in DS1])
    val = [115, 118, 201, 205, 223]
    metric = selection_metric_for(counts, Split(name="x", train=tuple(r for r in DS1 if r not in val), val=tuple(val)))
    path = paths.configs_dir() / "validation_split" / "approved.yaml"
    path.write_text(yaml.safe_dump({"name": "V1", "val": val, "selection_metric": metric, "approved_by": ["A"]}))


def test_approval_freezes_the_winner_and_records_it(full_fake_root, monkeypatch):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    _approve_split()
    for letter, mean in {"A": 0.05, "B": 0.08, "C": 0.40, "D": 0.03}.items():
        _result(letter, mean)
    with pytest.raises(workflow.GateError, match="report first"):
        workflow.approve_phase1_gate(["Ann"])
    workflow.write_phase1_gate_report()
    with pytest.raises(LockedTestDataError):  # freezing builds test.npz, which needs the construction flag
        workflow.approve_phase1_gate(["Ann"])
    assert workflow.phase1_winner() is None

    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    entry = workflow.approve_phase1_gate(["Ann", "Ben"])
    assert entry["winner"] == "C" and entry["approved_by"] == ["Ann", "Ben"]
    assert entry["config"] == "configs/phase1_input_representation/C_short_window_rr_features.yaml"
    assert workflow.phase1_winner() == "C"
    assert (paths.processed_dir() / "v1" / "test.npz").is_file() and set(entry["files"]) == {"trainval.npz", "test.npz"}
    assert "phase 1 gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()
    assert "| construction |" in (paths.docs_dir() / "TEST_ACCESS_LOG.md").read_text()
    with pytest.raises(workflow.GateError, match="already approved"):
        workflow.approve_phase1_gate(["Ann"])


def test_status_names_the_next_step_in_order(project):
    day9 = dt.date(2026, 10, 15)
    lines, nxt = workflow.status(day9)
    assert lines[0].startswith("Day 9 of 30") and "python -m ecgaug.data.raw" in nxt
    (paths.results_for("phase1_data")).mkdir(parents=True)
    (paths.results_for("phase1_data") / "raw_checksums.json").write_text(json.dumps({"problems": []}))
    (paths.configs_dir() / "validation_split" / "approved.yaml").unlink()
    assert "validation split" in workflow.status(day9)[1]
    _approve_split_file = paths.configs_dir() / "validation_split" / "approved.yaml"
    _approve_split_file.write_text("name: V1\n")
    assert workflow.status(day9)[1].startswith("Run phase1_input_representation variants A, B, C, D")
    for letter in "ABCD":
        _result(letter, {"A": 0.05, "B": 0.08, "C": 0.40, "D": 0.03}[letter])
    assert "phase1-gate" in workflow.status(day9)[1] and "--approve" not in workflow.status(day9)[1]
    workflow.write_phase1_gate_report()
    assert "--approve" in workflow.status(day9)[1]
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase1_gate": {"winner": "C"}}))
    lines, nxt = workflow.status(day9)
    assert "Run phase2_ptbxl_pretraining for the phase 1 winner, variant C" in nxt
    assert any("[x] phase 1 gate: approved" in line and "winner C" in line for line in lines)


def test_pretraining_waits_for_gate1_unless_previewed(full_fake_root, capsys, monkeypatch):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    _approve_split()
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    config = ["--config", str(variant_config(paths.configs_dir() / "phase2_ptbxl_pretraining", "A"))]
    assert pretrain.main(config) == 2
    assert "Phase 1 is not finished" in capsys.readouterr().out

    (paths.results_dir()).mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase1_gate": {"winner": "C"}}))
    assert pretrain.main(config) == 2
    assert "phase 1 gate chose variant C; run configs/phase2_ptbxl_pretraining/C_short_window_rr_features.yaml" in capsys.readouterr().out

    assert pretrain.main([*config, "--preview"]) == 2  # passes the phase check, stops at the next one
    out = capsys.readouterr().out
    assert "Phase 1" not in out and "phase 1 gate chose" not in out and "PTB-XL is not preprocessed" in out
