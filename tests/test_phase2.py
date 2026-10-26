"""Phase 2: baseline architectures, min_epochs, the quality metric families, the sanity harness, the phase 2 gate."""

import json
import shutil

import numpy as np
import pytest
import torch
import yaml

from conftest import REPO, variant_config, variant_letters
from ecgaug import paths, quality, workflow
from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import phase2_baseline_classifiers as baselines
from ecgaug.experiments import phase2_quality_metrics as qm
from ecgaug.experiments.generator_runs import QualitySetup
from ecgaug.models import build_model, count_parameters, load_encoder
from ecgaug.quality import QualityContext
from ecgaug.quality.feature_net import ensure_feature_net
from ecgaug.quality.metrics import dtw_pairs, frechet_distance, replace_minority
from ecgaug.quality.sanity import degrade, halves, score_family
from ecgaug.train import loop

from test_generators import training_part

BASELINES = REPO / "configs" / "phase2_baseline_classifiers"
QUALITY = REPO / "configs" / "phase2_quality_metrics"
MODELS = [yaml.safe_load(variant_config(BASELINES, letter).read_text())["model"] for letter in "ABCD"]


@pytest.mark.parametrize("model_cfg", MODELS, ids=["shallow_cnn", "resnet1d", "bilstm", "cnn_bilstm"])
def test_baseline_architectures(model_cfg):
    for length in (63, 180, 360):
        for n_side in (0, 3):
            model = build_model(model_cfg, 4, n_side).eval()
            x, side = torch.randn(3, length), (torch.randn(3, 3) if n_side else None)
            assert model(x, side).shape == (3, 4) and model.embed(x, side).shape[0] == 3
    with pytest.raises(ValueError, match="side input"):
        build_model(model_cfg, 4, 3)(torch.randn(2, 180))
    with pytest.raises(ValueError, match="no side input"):
        build_model(model_cfg, 4, 0)(torch.randn(2, 180), torch.randn(2, 3))
    source, target = build_model(model_cfg, 5, 3), build_model(model_cfg, 4, 3)
    copied = load_encoder(target, source.state_dict())
    assert copied and not any(k.startswith("head.") for k in copied)
    if model_cfg["name"] in ("shallow_cnn", "resnet1d", "cnn_bilstm"):
        assert isinstance(target.last_conv, torch.nn.Conv1d)
    with pytest.raises(ValueError, match="does not take"):
        build_model({**model_cfg, "dropout": 0.1}, 4)


def test_baseline_variants_share_one_budget_and_have_roles():
    assert variant_letters(BASELINES) == list("ABCD")
    configs = {letter: baselines.load_config(variant_config(BASELINES, letter)) for letter in "ABCD"}
    own = {"variant", "description", "role", "model", "config_name"}
    reference = {k: v for k, v in configs["A"].items() if k not in own}
    assert all({k: v for k, v in c.items() if k not in own} == reference for c in configs.values())
    assert [configs[letter]["role"] for letter in "ABCD"] == ["cnn", "cnn", "rnn", "rnn"]
    assert reference["seeds"] == [0, 1, 2] and reference["lr_grid"] == [1e-3, 3e-4] and reference["min_epochs"] == 10
    sizes = [count_parameters(build_model(configs[letter]["model"], 4, 3)) for letter in "ABCD"]
    assert sizes[0] < sizes[1]  # the plain CNN is the simpler of the two cnn candidates


def test_baseline_smoke_run():
    out = baselines.smoke(baselines.load_config(variant_config(BASELINES, "D")))
    assert out["role"] == "rnn" and "preds_val.npz" in out["run_files"]


def test_min_epochs_delays_early_stopping(monkeypatch):
    monkeypatch.setattr(loop, "selection_value", lambda metrics, name: 0.0)  # never improves after epoch 1
    part = {"X": np.random.default_rng(0).standard_normal((64, 63)).astype(np.float32), "y": np.arange(64) % 4}
    model = build_model({"name": "probe_cnn", "channels": [4, 4, 4], "kernel": 3, "side_hidden": 2}, 4)
    plain = loop.fit_classifier(model, part, part, loop.TrainConfig(lr=1e-3, max_epochs=20, patience=2, batch_size=32), "s_f1", 0)
    model = build_model({"name": "probe_cnn", "channels": [4, 4, 4], "kernel": 3, "side_hidden": 2}, 4)
    delayed = loop.fit_classifier(model, part, part, loop.TrainConfig(lr=1e-3, max_epochs=20, patience=2, batch_size=32, min_epochs=8), "s_f1", 0)
    assert plain.epochs_run == 3 and delayed.epochs_run == 8 and plain.best_epoch == delayed.best_epoch == 1


def test_dtw_matches_a_reference_implementation():
    from dtaidistance import dtw

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((6, 40)), rng.standard_normal((6, 40))
    ours = dtw_pairs(a, b, window=5)
    reference = [dtw.distance(a[i], b[i], window=6, use_c=False) for i in range(6)]  # dtaidistance's window is exclusive
    np.testing.assert_allclose(ours, reference, rtol=1e-10)
    np.testing.assert_allclose(dtw_pairs(a, a, 3), 0.0)
    assert dtw_pairs(np.array([[0.0, 1.0, 2.0]]), np.array([[0.0, 0.0, 1.0]]), 1)[0] == pytest.approx(1.0)


def test_frechet_distance_hand_computed():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5000, 2))
    assert frechet_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert frechet_distance(a, a + np.array([3.0, 4.0])) == pytest.approx(25.0, rel=1e-6)


def _context(**kwargs):
    settings = yaml.safe_load((QUALITY / "metric_settings.yaml").read_text())["metrics"]
    settings = {**settings, "dtw_max_beats": 30, "tstr_classifier": {**settings["tstr_classifier"], "max_epochs": 2}}
    reference, heldout = halves(training_part(n=400), seed=0)
    return QualityContext(train=reference, settings=settings, **kwargs), heldout


def test_metric_families_rank_clean_beats_above_degraded_ones():
    rng = np.random.default_rng(0)
    for name in ("dtw", "spectral"):
        context, heldout = _context()
        clean = {k: v for k, v in heldout.items()}
        noisy = degrade(clean, "noise", 1.0, rng, heldout)
        family = quality.METRIC_FAMILIES[name]
        assert family.evaluate(context, clean) < family.evaluate(context, noisy), name


def test_tstr_replaces_minority_beats_and_scores_relative_to_real():
    part = training_part(n=400)
    synthetic = {k: v.copy() for k, v in part.items()}
    synthetic = {k: v[synthetic["y"] == CLASS_INDEX["F"]] for k, v in synthetic.items()}
    mixed = replace_minority(part, synthetic, seed=0)
    assert np.bincount(mixed["y"]).tolist() == np.bincount(part["y"]).tolist() and (mixed["split"] == "train").all()
    val = {"X": part["X"], "y": part["y"], "side": None, "record": part["record"], "r_idx": part["r_idx"]}
    context, heldout = _context(val=val, selection="s_f1")
    value = quality.METRIC_FAMILIES["tstr"].evaluate(context, {k: v for k, v in heldout.items()})
    assert np.isnan(value) or value >= 0


def test_feature_net_is_saved_once_and_hash_checked():
    part = training_part(n=200)
    split_part = {"X": part["X"], "y": part["y"], "side": None, "record": part["record"], "r_idx": part["r_idx"]}
    settings = {**yaml.safe_load((QUALITY / "metric_settings.yaml").read_text())["feature_net"], "max_epochs": 1}
    net, meta = ensure_feature_net(split_part, split_part, settings, "s_f1", "a" * 64)
    assert meta["path"].endswith("feature_net_aaaaaaaaaaaa.pt") and len(meta["sha256"]) == 64
    assert net.embed_waveform(torch.randn(2, 63)).shape == (2, 64)
    again, meta2 = ensure_feature_net(split_part, split_part, settings, "s_f1", "a" * 64)
    assert meta2 == meta
    path = paths.repo_root() / meta["path"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match"):
        ensure_feature_net(split_part, split_part, settings, "s_f1", "a" * 64)
    context, heldout = _context(feature_net=again)
    assert np.isfinite(quality.METRIC_FAMILIES["fid_analog"].evaluate(context, heldout))


def test_sanity_harness_structure_and_rule():
    context, _ = _context()
    report = score_family("spectral", training_part(n=400), context.settings,
                          {"noise": [0.1, 0.3, 1.0], "smoothing": [3, 9, 27]}, 2, {"spearman_min": 0.8, "cv_max": 0.10})  # fmt: skip
    assert set(report["scores"]) == {"reference:0", "noise:0.1", "noise:0.3", "noise:1", "smoothing:3", "smoothing:9", "smoothing:27"}
    assert len(report["checks"]) == 4 and report["spearman"]["noise"] == 1.0
    assert all(len(v["values"]) == 2 for v in report["scores"].values())


@pytest.mark.parametrize("letter", list("ABCD"))
def test_quality_smoke_runs(letter):
    out = qm.smoke(qm.load_config(variant_config(QUALITY, letter)))
    assert out["sets_scored"] == 13 and len(out["checks"]) == 6


def test_quality_configs():
    assert variant_letters(QUALITY) == list("ABCD")
    families = [qm.load_config(variant_config(QUALITY, letter))["family"] for letter in "ABCD"]
    assert families == ["dtw", "spectral", "fid_analog", "tstr"] and set(families) == set(quality.METRIC_FAMILIES)


# --------------------------------------------------------------------------------------------------
# the phase 2 gate
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_root):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    return tmp_root


def _baseline(letter, mean, role, params, provisional=False):
    result = {
        "experiment": "phase2_baseline_classifiers", "variant": letter, "role": role, "model": {"name": "x"}, "n_parameters": params,
        "mean": mean, "sd": 0.01, "tie_break_mean": 0.5, "tie_break_sd": 0.01, "chosen_lr": 0.001, "wall_clock_s": 1.0,
        "split": {"name": "V1"}, "selection_metric": "s_f1", "dataset_content_sha256": "d", "budget_deviations": [],
        "budget": {"seeds": [0, 1, 2], "max_epochs": 40, "patience": 5, "min_epochs": 0, "batch_size": 256, "weight_decay": 0.01},
        "per_seed": [{"seed": s, "selection_value": mean} for s in (0, 1, 2)], "run_ids": [f"phase2_baseline_classifiers-{letter}-s{s}" for s in (0, 1, 2)],
        "provisional": provisional,
    }  # fmt: skip
    folder = paths.results_for("phase2_baseline_classifiers")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def _quality(letter, family, passed, provisional=False):
    result = {"experiment": "phase2_quality_metrics", "variant": letter, "family": family, "passed": passed, "reference_best": passed,
              "spearman": {"noise": 1.0}, "median_cv": 0.05, "settings": {"metrics": {"dtw_max_beats": 200}}, "dataset": {"content_sha256": "d"},
              "feature_net": {"path": "results/phase2_quality_metrics/feature_net_x.pt", "sha256": "f" * 64, "settings": {}} if family == "fid_analog" else None,
              "provisional": provisional}  # fmt: skip
    folder = paths.results_for("phase2_quality_metrics")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def _all(provisional=False):
    for letter, mean, role, params in (("A", 0.30, "cnn", 70_000), ("B", 0.31, "cnn", 490_000), ("C", 0.20, "rnn", 130_000), ("D", 0.28, "rnn", 80_000)):
        _baseline(letter, mean, role, params, provisional)
    for letter, family, passed in (("A", "dtw", True), ("B", "spectral", True), ("C", "fid_analog", True), ("D", "tstr", False)):
        _quality(letter, family, passed, provisional)


def test_phase2_gate_picks_cnn_rnn_and_composite(project):
    with pytest.raises(workflow.GateError, match="not run yet"):
        workflow.evaluate_phase2_gate()
    _all(provisional=True)
    gate = workflow.write_phase2_gate_report()
    assert gate["baselines"]["cnn"]["winner"] == "A"  # A and B tied within the margin; A has fewer parameters
    assert gate["baselines"]["rnn"]["winner"] == "D"
    assert gate["quality"]["composite"] == ["dtw", "spectral", "fid_analog"] and gate["quality"]["feature_net"]["sha256"] == "f" * 64
    assert "PREVIEW" in (paths.results_for("phase2_gate") / "report.md").read_text()
    with pytest.raises(workflow.GateError, match="Provisional"):
        workflow.approve_phase2_gate(["Ann"])
    _all()
    workflow.write_phase2_gate_report()
    entry = workflow.approve_phase2_gate(["Ann"])
    assert entry["cnn"]["variant"] == "A" and entry["cnn"]["lr"] == 0.001 and entry["cnn"]["max_epochs"] == 40
    assert entry["cnn"]["config"] == "configs/phase2_baseline_classifiers/A_shallow_cnn.yaml"
    assert entry["quality_composite"] == ["dtw", "spectral", "fid_analog"] and "phase 2 gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()
    with pytest.raises(workflow.GateError, match="already approved"):
        workflow.approve_phase2_gate(["Ann"])


def test_phase3_reads_the_composite_from_the_phase2_gate(project, monkeypatch):
    from ecgaug.experiments import generator_runs as gr

    (paths.results_dir() / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase2_gate": {"quality_composite": ["dtw", "spectral"], "quality_settings": {"x": 1}}}))
    setup, why = gr.quality_families(preview=False)
    assert setup.families == ["dtw", "spectral"] and setup.settings == {"x": 1} and setup.feature_net is None and why is None


def test_pilot_logs_the_composite_metrics_at_every_checkpoint():
    from ecgaug.experiments import generator_runs as gr
    from ecgaug.experiments.data_source import smoke_data

    cfg = gr.load_config(variant_config(REPO / "configs" / "phase3_cvae", "A"))
    small = {**cfg, "pilot_seeds": [0], "training": {**cfg["training"], "pilot_steps": 4, "batch_size": 16, "log_every": 2},
             "diagnostics": {**cfg["diagnostics"], "max_beats": 40, "c2st_folds": 2}}  # fmt: skip
    settings = {**yaml.safe_load((QUALITY / "metric_settings.yaml").read_text())["metrics"], "dtw_max_beats": 10}
    classifier = {**cfg["pilot"]["preview_classifier"], "max_epochs": 1}
    with smoke_data() as (split, data):
        result = gr.run_pilot(small, split, data, classifier, QualitySetup(["dtw", "spectral"], settings), torch.device("cpu"), ["smoke"])
        assert set(result["per_seed"][0]["quality"]) == {"dtw", "spectral"} and set(result["quality_mean"]) == {"dtw", "spectral"}
        from ecgaug import runs

        rows = (runs.find_run_dir(result["run_ids"][0]) / "checkpoints.csv").read_text()
        assert "quality_dtw" in rows and "quality_spectral" in rows
