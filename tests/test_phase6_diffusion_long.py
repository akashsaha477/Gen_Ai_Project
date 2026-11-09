"""P17: the config may differ from P15's only in the step budget; the generators, banks and grid carry P17's own names; the
DS2 report on fakes checks P13's, P15's and P16's scores first."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_diffusion as dif
from ecgaug.experiments import phase6_diffusion_long as long
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root

TINY = {"timesteps": 50, "sample_steps": 5, "guidance": 1.5, "p_uncond": 0.1, "ema_decay": 0.9, "channels": [8, 16, 16], "kernel": 5,
        "embed_dim": 16, "groups": 4, "weight_decay": 0.0, "clip_x0": 10.0}  # fmt: skip
CONFIG = "configs/phase6_diffusion_long/diffusion_long.yaml"


def test_config_differs_from_p15_only_in_a_longer_budget(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = long.load_config(repo_root() / CONFIG)
    assert cfg["training"]["final_steps"] == 100000 and cfg["p15"]["training"]["final_steps"] == 20000
    assert cfg["generator"] == cfg["p15"]["generator"] and cfg["seeds"] == [0, 1, 2]
    text = (repo_root() / CONFIG).read_text()
    for old, new in (("guidance: 1.5", "guidance: 3.0"), ("final_steps: 100000", "final_steps: 20000"), ("batch_size: 128", "batch_size: 512")):
        bad = tmp_path / "bad.yaml"
        bad.write_text(text.replace(old, new).replace("configs/", str(repo_root() / "configs") + "/"))
        with pytest.raises(ValueError):
            long.load_config(bad)


def test_generators_bank_and_grid_carry_p17s_names():
    from types import SimpleNamespace

    from ecgaug import paths, runs

    rng = np.random.default_rng(0)
    n, length = 64, 32
    y = np.array([0] * 40 + [1] * 8 + [2] * 8 + [3] * 8, dtype=np.int8)
    split = np.array(["train"] * n)
    split[::8] = "val"
    arrays = {"X": (0.3 * rng.standard_normal((n, length))).astype(np.float32), "y": y, "record": np.where(split == "val", 900, 100).astype(np.int16),
              "split": split, "r_idx": np.arange(n, dtype=np.int32), "rr_prev": np.full(n, 0.7, np.float32), "rr_next": np.full(n, 0.9, np.float32),
              "rr_ratio": np.ones(n, np.float32), "is_synthetic": np.zeros(n, bool), "source": np.array(["real"] * n)}  # fmt: skip
    data = SimpleNamespace(arrays=arrays, spec=SimpleNamespace(n_samples=length), info={"content_sha256": "fake"})
    cfg = {"generator": TINY, "training": {"batch_size": 16, "lr": 1e-3, "final_steps": 2, "log_every": 1}, "classes": ["N", "S", "V", "F"],
           "seeds": [0], "strengths": [0.1, 0.25, 0.5]}  # fmt: skip
    record = dif.train_generators(cfg, data, "cpu", experiment=long.EXPERIMENT, rule=long.RULE)
    assert record["experiment"] == long.EXPERIMENT and record["rule"] == long.RULE and record["generator_id"] == "phase6_diffusion_long/diffusion"
    assert (paths.results_for(long.EXPERIMENT) / "diffusion_generators.json").is_file() and not dif.generators_path().is_file()
    run_dir = runs.find_run_dir(record["seeds"]["0"]["run_id"])
    assert run_dir is not None and long.EXPERIMENT in run_dir.parts
    bank, _ = dif.load_diffusion_bank(0, long.EXPERIMENT)
    assert np.asarray(bank["X"]).shape[1] == length


def test_grid_is_p15s_with_p17s_bank(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = long.load_config(repo_root() / CONFIG)
    grid, p15 = dif.diffusion_grid(cfg, long.EXPERIMENT), dif.diffusion_grid(cfg["p15"])
    assert grid["experiment"] == long.EXPERIMENT and grid["conditions"]["diffusion"]["bank"] == "phase6_diffusion_long/diffusion"
    assert {k: v for k, v in grid.items() if k not in ("experiment", "conditions", "file")} == {k: v for k, v in p15.items() if k not in ("experiment", "conditions", "file")}


def test_test_report_compares_with_p15_and_p13_and_checks_earlier_scores_first(monkeypatch):
    from types import SimpleNamespace

    from ecgaug import final_eval, paths
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments import phase6_rr_baseline as v5

    rng = np.random.default_rng(0)
    records = np.repeat([100, 103, 105, 111], 40)
    y = np.tile(np.repeat([0, 1, 2, 3], 10), 4).astype(np.int8)
    test = {"X": np.zeros((len(y), 4), np.float32), "y": y, "record": records, "r_idx": np.arange(len(y)), "rr_prev": rng.uniform(0.5, 1.2, len(y)).astype(np.float32),
            "rr_next": np.full(len(y), 0.9, np.float32), "rr_ratio": np.ones(len(y), np.float32)}  # fmt: skip
    p_af = np.where(records == 103, 0.99, 0.01)

    def fake_predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
        out = []
        for run_id in run_ids:
            g = np.random.default_rng(sum(map(ord, run_id)))
            out.append(np.eye(4)[np.where(g.random(len(arrays["y"])) < 0.7, arrays["y"], g.integers(0, 4, len(arrays["y"])))])
        return out

    for name, fn in (("check_preconditions", lambda: None), ("load_data", lambda *a, **k: SimpleNamespace(spec=None))):
        monkeypatch.setattr(long, name, fn)
    monkeypatch.setattr(pt, "test_arrays", lambda spec, experiment: test)
    monkeypatch.setattr(v5, "record_ratios", lambda record, half, access: {})
    monkeypatch.setattr(v5, "substitute", lambda arrays, ratios: arrays)
    monkeypatch.setattr(pt, "mitbih_templates", lambda arrays, spec, half, access: None)
    monkeypatch.setattr(pt, "relative", lambda arrays, templates: arrays)
    monkeypatch.setattr(pt, "_predict", fake_predict)
    monkeypatch.setattr(rr, "mean_rr_by_record", lambda recs, access=None: {int(r): 1.0 for r in np.unique(recs)})
    monkeypatch.setattr(af, "load_detector", lambda: {})
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    monkeypatch.setattr(af, "ds2_af_probability", lambda detector, recs, r_idx: p_af)

    def chosen(prefix: str, conditions: tuple) -> dict:
        return {"chosen": {"cnn": {c: {"strength": 0.25, "run_ids": [f"{prefix}-{c}-s{s}" for s in range(3)]} for c in conditions}}}

    for experiment, conditions in ((long.EXPERIMENT, ("diffusion",)), (dif.EXPERIMENT, ("diffusion",)), (rr.EXPERIMENT, rr.CONDITIONS)):
        paths.results_for(experiment).mkdir(parents=True, exist_ok=True)
        (paths.results_for(experiment) / "selection.json").write_text(json.dumps(chosen(experiment, conditions)))
    ids = {**{c: chosen(rr.EXPERIMENT, rr.CONDITIONS)["chosen"]["cnn"][c]["run_ids"] for c in long.BEFORE}, "diffusion": chosen(dif.EXPERIMENT, ("diffusion",))["chosen"]["cnn"]["diffusion"]["run_ids"]}

    def point(rows: list[np.ndarray]) -> dict:
        cms = final_eval.per_record_cms(rows, y, records)[1]
        return {"rare_macro_f1": {"point": long._rare(cms)}}

    def write_summaries(shift: float) -> None:
        plain = {c: fake_predict(r, test) for c, r in ids.items()}
        (paths.results_for(rr.EXPERIMENT) / "summary.json").write_text(json.dumps({"conditions": {f"cnn/{c}": point(plain[c]) for c in long.BEFORE}}))
        (paths.results_for(dif.EXPERIMENT) / "summary.json").write_text(json.dumps({"conditions": {"cnn/diffusion": point(plain["diffusion"])}}))
        rule = {f"cnn/{c}": point([rr.sveb_rule(p, test["rr_prev"], 0.8) for p in plain[c]]) for c in ids}
        veto = {f"cnn/{c}": point([af.veto_rule(p, test["rr_prev"], p_af, 0.8, 0.95, "rule") for p in plain[c]]) for c in ids}
        veto["cnn/cvae"]["rare_macro_f1"]["point"] += shift
        paths.results_for(af.EXPERIMENT).mkdir(parents=True, exist_ok=True)
        (paths.results_for(af.EXPERIMENT) / "summary.json").write_text(json.dumps({"rule_alone": rule, "with_af_veto": veto}))

    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "rule_threshold": 0.8}
    write_summaries(0.01)
    with pytest.raises(PreconditionError):
        long.test_report(cfg)
    write_summaries(0.0)
    s = long.test_report(cfg)
    assert set(s["conditions"]) == {"cnn/diffusion_long"} and set(s["before"]) == {"cnn/diffusion", *(f"cnn/{c}" for c in long.BEFORE)}
    assert set(s["families"]["confirmatory"]) == {"cnn: diffusion_long vs diffusion (P15)", "cnn: diffusion_long vs none (P13)"}
    assert set(s["families"]["secondary"]) == {f"cnn: diffusion_long vs {c} (P13)" for c in ("cvae", "class_weighted", "random_oversampling")}
    assert set(s["sveb_rule"]) == set(s["with_af_veto"]) == {"cnn/diffusion_long", "cnn/diffusion", *(f"cnn/{c}" for c in long.BEFORE)}
    assert s["veto_form"] == "rule" and all(abs(d) < 1e-9 for d in s["reproduction_difference"].values())
