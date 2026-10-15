"""The phase2_ptbxl_pretraining experiment: multi-label training, weight transfer, config, preconditions, end to end."""

import csv
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import REPO, variant_config, variant_letters
from ecgaug import paths, runs
from ecgaug.data import ptbxl
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb, write_fake_ptbxl
from ecgaug.data.records import DS1
from ecgaug.data.splits import Split, selection_metric_for
from ecgaug.eval.metrics import multilabel_auroc
from ecgaug.experiments import phase1_input_representation as m1
from ecgaug.experiments import phase2_ptbxl_pretraining as pretrain
from ecgaug.experiments.common import BUDGET_KEYS
from ecgaug.models import build_model, load_encoder
from ecgaug.train.loop import TrainConfig, fit_multilabel, set_seed

CONFIGS = REPO / "configs" / "phase2_ptbxl_pretraining"
M1_CONFIGS = REPO / "configs" / "phase1_input_representation"
MODEL = {"name": "probe_cnn", "channels": [16, 32, 64], "kernel": 7, "side_hidden": 16}


def test_multilabel_auroc_hand_computed():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    probs = np.array([[0.9, 0.2], [0.8, 0.6], [0.7, 0.1], [0.1, 0.3]])
    out = multilabel_auroc(probs, y, ("a", "b"))
    # label a: positives score 0.9 and 0.7, negatives 0.8 and 0.1 -> 3 of 4 pairs ordered correctly
    assert out["per_label"]["a"] == pytest.approx(0.75)
    assert np.isnan(out["per_label"]["b"])  # no positive example
    assert out["macro"] == pytest.approx(0.75) and out["n_defined"] == 1
    assert np.isnan(multilabel_auroc(probs, np.zeros_like(y), ("a", "b"))["macro"])


def _multilabel(n=400, length=180, seed=0):
    """Two independent labels, each marked by a bump at its own position."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
    x = 0.2 * rng.standard_normal((n, length))
    x[:, 40:60] += 2.0 * y[:, :1]
    x[:, 110:130] += 2.0 * y[:, 1:]
    return {"X": x.astype(np.float32), "y": y}


def _fit(seed=0, val=None):
    set_seed(seed)
    model = build_model(MODEL, n_classes=2)
    cfg = TrainConfig(lr=3e-3, max_epochs=5, patience=5, batch_size=64)
    return fit_multilabel(model, _multilabel(seed=1), val or _multilabel(seed=2), cfg, ("a", "b"), seed=seed)


def test_multilabel_training_learns_and_is_deterministic():
    a, b = _fit(seed=3), _fit(seed=3)
    assert a.best_value > 0.97 and a.best_metrics["n_defined"] == 2
    assert a.val_probs.shape == (400, 2) and 0 <= a.val_probs.min() and a.val_probs.max() <= 1
    np.testing.assert_array_equal(a.val_probs, b.val_probs)
    assert "val_auroc_a" in a.history[0] and "val_loss" in a.history[0]


def test_multilabel_training_refuses_constant_validation_labels():
    val = _multilabel(seed=2)
    val["y"][:] = 0
    with pytest.raises(ValueError, match="undefined"):
        _fit(val=val)


def test_encoder_transfer_copies_everything_but_the_head():
    torch.manual_seed(0)
    source = build_model(MODEL, n_classes=5, n_side=3)
    torch.manual_seed(1)
    target = build_model(MODEL, n_classes=4, n_side=3)
    head_before = target.head.weight.detach().clone()
    copied = load_encoder(target, source.state_dict())
    assert copied and not any(name.startswith("head.") for name in copied)
    for name, tensor in source.state_dict().items():
        if not name.startswith("head."):
            torch.testing.assert_close(target.state_dict()[name], tensor)
    torch.testing.assert_close(target.head.weight, head_before)  # the new 4-class head is untouched
    assert target.head.out_features == 4 and source.head.out_features == 5


def test_encoder_transfer_refuses_a_different_architecture():
    source = build_model(MODEL, n_classes=5)
    with pytest.raises(ValueError, match="do not fit"):
        load_encoder(build_model({**MODEL, "channels": [8, 16, 32]}, n_classes=4), source.state_dict())
    with pytest.raises(ValueError, match="do not fit"):
        load_encoder(build_model(MODEL, n_classes=4, n_side=3), source.state_dict())  # side branch missing
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "transformer"}, n_classes=4)


def test_config_takes_representation_and_budget_from_module_1():
    assert variant_letters(CONFIGS) == list("ABCD")
    for letter in "ABCD":
        cfg = pretrain.load_config(variant_config(CONFIGS, letter))
        base = m1.load_config(variant_config(M1_CONFIGS, letter))
        assert cfg["variant"] == letter and cfg["representation"] == base["representation"]
        assert {k: cfg[k] for k in BUDGET_KEYS} == {k: base[k] for k in BUDGET_KEYS}
        assert cfg["model"] == base["model"]
        assert cfg["experiment"] == "phase2_ptbxl_pretraining"
        assert set(cfg["pretrain"]) == pretrain.PRETRAIN_KEYS


def test_config_cannot_change_the_fine_tuning_budget_or_point_at_another_variant(tmp_path):
    (tmp_path / "phase1_input_representation").mkdir()
    for name in ("shared_budget.yaml", "A_short_window.yaml", "B_long_window_bandpass.yaml"):
        (tmp_path / "phase1_input_representation" / name).write_text((M1_CONFIGS / name).read_text())
    (tmp_path / "pretrain").mkdir()
    shared = yaml.safe_load((CONFIGS / "pretraining_settings.yaml").read_text())
    variant = {
        "base": "pretraining_settings.yaml",
        "variant": "A",
        "representation_from": "../phase1_input_representation/A_short_window.yaml",
    }

    def write(shared_cfg, variant_cfg):
        (tmp_path / "pretrain" / "pretraining_settings.yaml").write_text(yaml.safe_dump(shared_cfg))
        (tmp_path / "pretrain" / "A.yaml").write_text(yaml.safe_dump(variant_cfg))
        return pretrain.load_config(tmp_path / "pretrain" / "A.yaml")

    assert write(shared, variant)["variant"] == "A"
    with pytest.raises(ValueError, match="fine-tuning budget comes from phase1_input_representation"):
        write({**shared, "seeds": [0, 1, 2, 3, 4]}, variant)
    with pytest.raises(ValueError, match="points at the phase1_input_representation config of B"):
        write(shared, {**variant, "representation_from": "../phase1_input_representation/B_long_window_bandpass.yaml"})
    with pytest.raises(ValueError, match="unexpected keys"):
        write(shared, {**variant, "max_epochs": 100})


def _fake_world(tmp_root, n_extra=40):
    """Fake MIT-BIH and fake preprocessed PTB-XL in the temporary root, with a matching split."""
    train_records, val_records = (101, 106, 108, 114), (118, 201)
    write_fake_mitdb(paths.raw_dir(), train_records + val_records, seconds=120.0)
    write_fake_ptbxl(tmp_root / "ptbxl_raw", n_extra=n_extra)
    ptbxl.prepare(tmp_root / "ptbxl_raw", verify=False)
    counts = pd.DataFrame([count_record(r) for r in train_records + val_records])
    split = Split(name="t", train=train_records, val=val_records)
    return Split(name="t", train=train_records, val=val_records, selection_metric=selection_metric_for(counts, split))


def test_arm_end_to_end_on_fake_data(tmp_root):
    split = _fake_world(tmp_root)
    cfg = pretrain.load_config(variant_config(CONFIGS, "C"))
    cfg = {**cfg, "seeds": [0, 1], "lr_grid": [1e-3], "max_epochs": 2, "pretrain": {**cfg["pretrain"], "max_epochs": 2}}
    result = pretrain.run(cfg, split, ptbxl.load(), device_name="cpu", require_all_ds1=False)

    assert result["winner"] in pretrain.ARMS and result["provisional"] is True
    assert set(result["arms"]) == {"scratch", "pretrained"}
    pre = {p["seed"]: p["run_id"] for p in result["pretraining"]["per_seed"]}
    assert set(pre) == {0, 1} and len(set(pre.values())) == 2
    assert all(0.0 <= p["macro_auroc"] <= 1.0 for p in result["pretraining"]["per_seed"])
    for arm in pretrain.ARMS:
        runs_of_arm = result["arms"][arm]["per_seed"]
        assert [r["seed"] for r in runs_of_arm] == [0, 1]
        for r in runs_of_arm:
            config = yaml.safe_load((runs.find_run_dir(r["run_id"]) / "config.yaml").read_text())
            if arm == "pretrained":  # each seed starts from the pretraining run of the same seed
                assert config["init"]["run_id"] == pre[r["seed"]] and config["init"]["tensors_copied"] > 0
            else:
                assert "init" not in config
    assert result["arms"]["pretrained"]["wall_clock_s"] > result["pretraining"]["wall_clock_s"]
    assert result["difference_pretrained_minus_scratch"] == pytest.approx(
        result["arms"]["pretrained"]["mean"] - result["arms"]["scratch"]["mean"]
    )
    assert result["decision"]["steps"][-1]["winner"] == result["winner"]
    assert result["alignment_with_mitbih_train"]["n_ptbxl_beats"] > 0
    assert result["budget"] == {k: cfg[k] for k in BUDGET_KEYS} and result["budget_deviations"] == []

    figure_dir = paths.repo_root() / "figures" / "phase2_ptbxl_pretraining" / "C"
    assert sorted(f.name for f in figure_dir.glob("*.png")) == [
        "01_pretraining_curves.png", "02_finetune_vs_scratch_curves.png",
        "03_confusion_matrices.png", "04_f1_per_class_before_after_pretraining.png",
    ]  # fmt: skip
    written = json.loads((paths.results_for("phase2_ptbxl_pretraining") / "C.json").read_text())
    assert written["winner"] == result["winner"]
    with (paths.results_dir() / "runs_index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert sorted({r["variant"] for r in rows}) == ["finetuneC", "pretrainC", "scratchC"] and len(rows) == 6
    assert {r["phase"] for r in rows} == {"2"} and {r["experiment"] for r in rows} == {"phase2_ptbxl_pretraining"}
    # one folder per experiment, variant and kind of run
    assert {r["folder"] for r in rows} == {
        "phase2_ptbxl_pretraining/C_short_window_rr_features/pretrain",
        "phase2_ptbxl_pretraining/C_short_window_rr_features/scratch",
        "phase2_ptbxl_pretraining/C_short_window_rr_features/finetune",
    }
    for r in rows:
        assert (paths.runs_dir() / r["folder"] / r["run_id"] / "config.yaml").is_file()


def test_scratch_arm_matches_a_module_1_run_exactly(tmp_root):
    """Same data, seed, budget and model: the from-scratch arm is a phase1_input_representation variant run."""
    split = _fake_world(tmp_root)
    cfg = pretrain.load_config(variant_config(CONFIGS, "A"))
    small = {"seeds": [0], "lr_grid": [1e-3], "max_epochs": 2}
    result = pretrain.run({**cfg, **small, "pretrain": {**cfg["pretrain"], "max_epochs": 1}}, split, ptbxl.load(), "cpu", False)
    reference = m1.run_variant({**m1.load_config(variant_config(M1_CONFIGS, "A")), **small}, split, "cpu", False)
    assert result["arms"]["scratch"]["mean"] == reference["mean"]
    assert result["arms"]["scratch"]["per_seed"][0]["f1"] == reference["per_seed"][0]["f1"]


@pytest.mark.parametrize("letter", ["A", "C"])
def test_smoke_run_from_the_cli(letter, capsys, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    before = {(str(p), p.stat().st_mtime_ns) for d in ("runs", "results", "data/processed") for p in (REPO / d).rglob("*") if p.is_file()}
    assert pretrain.main(["--config", str(variant_config(CONFIGS, letter)), "--smoke"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["variant"] == letter and len(report["runs"]) == 3
    assert [f.split("/")[-1] for f in report["run_folders"]] == ["finetune", "pretrain", "scratch"]
    assert report["pretrained_init"]["kind"] == "ptbxl_pretrained" and report["pretrained_init"]["run_id"] == report["runs"][0]
    assert "ECGAUG_ROOT" not in os.environ
    after = {(str(p), p.stat().st_mtime_ns) for d in ("runs", "results", "data/processed") for p in (REPO / d).rglob("*") if p.is_file()}
    assert after == before, "a smoke run must not write into the repository"


def _approve(full_fake_root):
    counts = pd.DataFrame([count_record(r) for r in DS1])
    val = [115, 118, 201, 205, 223]
    metric = selection_metric_for(counts, Split(name="x", train=tuple(r for r in DS1 if r not in val), val=tuple(val)))
    path = paths.configs_dir() / "validation_split" / "approved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": "V1", "val": val, "selection_metric": metric, "approved_by": ["A"]}))


def test_real_run_is_refused_without_split_or_ptbxl(full_fake_root, capsys, monkeypatch):
    monkeypatch.setattr(runs, "git_dirty", lambda: False)
    config = ["--config", str(variant_config(CONFIGS, "A")), "--preview"]
    assert pretrain.main(config) == 2
    assert "approved" in capsys.readouterr().out

    _approve(full_fake_root)
    assert pretrain.main(config) == 2
    assert "no build on disk" in capsys.readouterr().out

    assert pretrain.main([*config, "--ptbxl-root", str(full_fake_root[0] / "nowhere")]) == 2
    assert "no ptbxl_database.csv" in capsys.readouterr().out

    write_fake_ptbxl(full_fake_root[0] / "ptbxl_raw", n_extra=4)
    ptbxl.prepare(full_fake_root[0] / "ptbxl_raw", limit=6, verify=False)  # a sample build, like on a laptop
    assert pretrain.main(config) == 2
    assert "partial" in capsys.readouterr().out
    assert not paths.runs_dir().exists()


def test_ptbxl_is_built_once_and_then_reused(tmp_root, monkeypatch):
    source = tmp_root / "attached" / "some-kaggle-dataset"
    write_fake_ptbxl(source, n_extra=4)
    calls = []
    original = ptbxl.prepare
    monkeypatch.setattr(ptbxl, "prepare", lambda *a, **k: calls.append(a) or original(*a, **k))
    first = pretrain.ensure_ptbxl(str(tmp_root / "attached"), workers=1, verify=False)  # root is found below the folder
    second = pretrain.ensure_ptbxl(None, workers=1, verify=False)  # already built: no root needed
    assert len(calls) == 1 and calls[0][0] == source
    np.testing.assert_array_equal(first["ecg_id"], second["ecg_id"])


def test_pretraining_code_never_asks_for_test_data():
    for name in ("experiments/phase2_ptbxl_pretraining.py", "experiments/common.py", "data/checks.py"):
        text = (REPO / "src" / "ecgaug" / name).read_text()
        assert "load_test" not in text and "ALLOW_TEST" not in text and 'access="' not in text


def test_a_ptbxl_copy_without_the_500hz_files_is_refused_clearly(tmp_root):
    import shutil

    source = tmp_root / "attached" / "low-rate-only"
    write_fake_ptbxl(source, n_extra=2)
    shutil.rmtree(source / "records500")
    with pytest.raises(pretrain.PreconditionError, match="records500"):
        pretrain.ensure_ptbxl(str(tmp_root / "attached"), workers=1, verify=False)
    assert not ptbxl.is_full_build()


def test_gpu_determinism_setting_is_in_place():
    import ecgaug.train.loop  # noqa: F401

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
