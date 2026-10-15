"""The Module 1 runner: equal budgets, preconditions for real runs, and end-to-end smoke runs."""

import json
import os

import pytest
import yaml

from conftest import REPO, variant_config, variant_letters
from ecgaug import paths
from ecgaug.experiments import input_representation as m1

CONFIG_DIR = REPO / "configs" / "m1_input_representation"


def _config(letter):
    return variant_config(CONFIG_DIR, letter)


def _output_files():
    """Every file under the repository's output directories, with its modification time."""
    return {
        (str(p), p.stat().st_mtime_ns)
        for name in ("runs", "results", "data/processed")
        for p in (REPO / name).rglob("*")
        if p.is_file()
    }


def test_four_variants_share_one_budget():
    configs = {letter: m1.load_config(_config(letter)) for letter in "ABCD"}
    assert variant_letters(CONFIG_DIR) == list("ABCD")
    varying = {"variant", "description", "representation", "config_name"}
    reference = {k: v for k, v in configs["A"].items() if k not in varying}
    assert reference["seeds"] == [0, 1, 2] and reference["lr_grid"] == [1e-3, 3e-4] and reference["patience"] == 5
    for letter, cfg in configs.items():
        assert cfg["variant"] == letter
        assert {k: v for k, v in cfg.items() if k not in varying} == reference


def test_a_variant_file_cannot_override_the_shared_budget(tmp_path):
    (tmp_path / "shared_budget.yaml").write_text((CONFIG_DIR / "shared_budget.yaml").read_text())
    variant = yaml.safe_load(_config("A").read_text())
    variant["seeds"] = [0, 1, 2, 3, 4]
    (tmp_path / "A.yaml").write_text(yaml.safe_dump(variant))
    with pytest.raises(ValueError, match="shared budget"):
        m1.load_config(tmp_path / "A.yaml")


def test_real_run_is_refused_without_an_approved_split(full_fake_root, capsys):
    assert m1.main(["--config", str(_config("A"))]) == 2
    assert "approved" in capsys.readouterr().out
    assert not paths.runs_dir().exists() and not (paths.results_dir() / "validation").exists()


@pytest.mark.parametrize("letter", "ABCD")
def test_smoke_run_executes_the_whole_pipeline(letter, capsys, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")  # smoke manages its own temporary root and must restore the environment
    before = _output_files()
    assert m1.main(["--config", str(_config(letter)), "--smoke"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["variant"] == letter
    assert report["inputs_per_beat"] == {"A": 180, "B": 360, "C": 183, "D": 63}[letter]
    assert set(report["run_files"]) == {
        "config.yaml", "env.json", "train_log.csv", "metrics_val.json", "preds_val.npz", "checkpoints",
    }  # fmt: skip
    assert "ECGAUG_ROOT" not in os.environ
    assert _output_files() == before, "a smoke run must not write into the repository"
