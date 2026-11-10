"""P22: the config fixes the registered values; the ResNet gets phase 2's settings for it and the cnn's budget; the grid
is P13's with the ResNet and three conditions for seeds 0-9; members are read from the ledger in seed order."""

import json
from pathlib import Path

import pytest

from ecgaug import paths
from ecgaug.experiments import phase6_bigger_classifier as bc
from ecgaug.experiments.common import PreconditionError
from ecgaug.grid import runner
from ecgaug.models import build_model
from ecgaug.paths import repo_root

CONFIG = "configs/phase6_bigger_classifier/bigger_classifier.yaml"


def test_config_fixes_the_registered_values(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = bc.load_config(repo_root() / CONFIG)
    assert cfg["se"]["experiment"] == "phase6_seed_ensemble"
    for old, new in (("classifier_variant: B", "classifier_variant: C"), ("[none, cvae, wgan_gp]", "[none, cvae]")):
        bad = tmp_path / "bad.yaml"
        bad.write_text((repo_root() / CONFIG).read_text().replace(old, new).replace("configs/", str(repo_root() / "configs") + "/"))
        with pytest.raises(ValueError):
            bc.load_config(bad)


def test_the_resnet_gets_phase_2s_settings_and_the_cnns_budget(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    s = bc.classifier_settings()
    assert s["model"]["name"] == "resnet1d" and s["lr"] == 0.001 and s["variant"] == "B"
    assert (s["max_epochs"], s["patience"], s["min_epochs"], s["batch_size"], s["weight_decay"]) == (40, 5, 10, 256, 0.01)
    resnet = sum(p.numel() for p in build_model(s["model"], 4, 3).parameters())
    cnn = sum(p.numel() for p in build_model({"name": "shallow_cnn", "channels": [32, 64, 128], "kernel": 7, "side_hidden": 16}, 4, 3).parameters())
    assert resnet > 5 * cnn
    b = json.loads((paths.results_for("phase2_baseline_classifiers") / "B.json").read_text())
    monkeypatch.setattr(bc.json, "loads", lambda text, _real=json.loads: {**_real(text), "budget_deviations": ["patience"]} if '"variant": "B"' in text else _real(text))
    with pytest.raises(PreconditionError):
        bc.classifier_settings()
    assert b["n_parameters"] == s["n_parameters_phase2"]


def test_the_grid_is_p13s_with_the_resnet_and_three_conditions(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = bc.load_config(repo_root() / CONFIG)
    g = bc.grid(cfg)
    cells = runner.cells(g)
    assert list(g["architectures"]) == ["resnet"] and tuple(g["conditions"]) == bc.CONDITIONS and len(cells) == 30
    assert g["conditions"]["cvae"]["strengths"] == [0.1] and g["conditions"]["wgan_gp"]["strengths"] == [0.25]
    assert {c.seed for c in cells} == set(range(10)) and g["experiment"] == bc.EXPERIMENT
    assert bc.PAIRS == {"resnet/none": "cnn/none", "resnet/cvae": "cnn/cvae", "resnet/wgan_gp": "cnn/wgan_gp"}


def test_members_come_from_the_ledger_in_seed_order(monkeypatch):
    rows = {}
    for c in bc.CONDITIONS:
        for s in reversed(range(10)):
            rows[f"{c}{s}"] = {"status": "done", "condition": c, "seed": s, "run_id": f"{c}-s{s}", "selection_value": 0.5,
                               "tie_break_value": 0.8, "rare_macro_f1": 0.3}  # fmt: skip
    monkeypatch.setattr(runner, "read_ledger", lambda grid: rows)
    monkeypatch.setattr(bc, "grid", lambda cfg: {})
    m = bc.members({})
    assert m["members"]["resnet/cvae"] == [f"cvae-s{s}" for s in range(10)]
    assert m["validation"]["resnet/none"]["f1_F"][0] == pytest.approx(0.1) and m["validation"]["resnet/none"]["f1_V"][0] == 0.8
    del rows["wgan_gp3"]
    with pytest.raises(PreconditionError):
        bc.members({})


def test_figures_and_paper_text_come_from_the_result_files(monkeypatch):
    from ecgaug import figures, paper_text

    assert "\\missing{P22: --part run" in paper_text.posthoc_bigger_classifier()
    validation = {f"resnet/{c}": {k: [0.5 + 0.01 * s for s in range(10)] for k in ("f1_S", "f1_V", "f1_F")} for c in bc.CONDITIONS}
    members = {"members": {k: [f"{k}-{s}" for s in range(10)] for k in validation}, "validation": validation, "classifier": {"n_parameters_phase2": 488362}}
    bc._write("members.json", members)
    cnn = {k.replace("resnet", "cnn"): v for k, v in validation.items()}
    monkeypatch.setattr(bc, "cnn_validation", lambda: cnn)
    assert "\\missing{P22: the scoring session" in paper_text.posthoc_bigger_classifier()
    e = {"point": 0.4, "ci_low": 0.3, "ci_high": 0.5}
    block = {c: {"f1_S": e, "f1_F": e, "rare_macro_f1": e} for c in (*bc.PAIRS, *bc.PAIRS.values())}
    comp = {"point": -0.02, "ci_low": -0.05, "ci_high": 0.01, "p_holm": 0.5, "reject_holm": False}
    names = [f"{c} vs {b}, chosen settings" for c, b in bc.PAIRS.items()]
    summary = {"patients": 977, "conditions": {"chosen": block, "ensemble_alone": block}, "families": {"secondary": {n: comp for n in names}}}
    bc._write("summary.json", summary)
    assert "\\missing{P22: --part test" in paper_text.posthoc_bigger_classifier()
    test = {"conditions": {"chosen": block, "ensemble_alone": block}, "families": {"confirmatory": {n: comp for n in names}}}
    bc._write("test_summary.json", test)
    out = figures.bigger_classifier_validation_figure(members, cnn) + figures.ensemble_pairs_figure(summary, test, bc.PAIRS, bc.EXPERIMENT, "t", "c")
    assert len(out) == 2 and all(p.is_file() for p in out)
    text = paper_text.posthoc_bigger_classifier()
    assert "\\missing" not in text and "the ResNet with the cVAE against the cnn with the cVAE" in text and "for the cnn" in text
    assert "fig:p22-val" in text and "fig:p22}" in text and text.count("{") == text.count("}") and "_" not in text
