"""P15: the config, the grid, the generators end to end on fake beats (bank, record, run folder) and the DS2 report on fakes."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.experiments import phase6_diffusion as dif
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root

TINY = {"timesteps": 50, "sample_steps": 5, "guidance": 1.5, "p_uncond": 0.1, "ema_decay": 0.9, "channels": [8, 16, 16], "kernel": 5,
        "embed_dim": 16, "groups": 4, "weight_decay": 0.0, "clip_x0": 10.0}  # fmt: skip


def test_config_needs_the_phase4_strengths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = dif.load_config(repo_root() / "configs/phase6_diffusion/diffusion.yaml")
    assert sorted(cfg["strengths"]) == sorted(rg.strengths()) and cfg["training"]["final_steps"] == 20000 and cfg["seeds"] == [0, 1, 2]
    bad = tmp_path / "bad.yaml"
    text = (repo_root() / "configs/phase6_diffusion/diffusion.yaml").read_text().replace("strengths: [0.1, 0.25, 0.5]", "strengths: [0.1, 0.3]")
    bad.write_text(text)
    with pytest.raises(ValueError):
        dif.load_config(bad)


def test_grid_is_p13s_cnn_with_the_diffusion_bank_at_the_phase4_strengths(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = dif.load_config(repo_root() / "configs/phase6_diffusion/diffusion.yaml")
    grid, base = dif.diffusion_grid(cfg), rr.record_grid(cfg["rr"])
    assert grid["conditions"] == {"diffusion": {"bank": "phase6_diffusion/diffusion", "strengths": [0.1, 0.25, 0.5]}}
    assert grid["architectures"] == base["architectures"] and grid["policy"] == {"kind": "fraction_of_n"} and grid["seeds"] == base["seeds"]


def test_generators_end_to_end_on_fake_beats():
    """Two steps of the real diffusion model: a checkpoint, a bank at the generator's scale with RR from training beats, the record."""
    from types import SimpleNamespace

    from ecgaug import runs
    from ecgaug.generators import bank as banks

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
    record = dif.train_generators(cfg, data, "cpu")
    entry = record["seeds"]["0"]
    assert not entry["diverged"] and entry["steps_done"] == 2
    assert entry["bank_counts"] == rg.bank_counts(y[split == "train"], 0.5)
    bank, _ = dif.load_diffusion_bank(0)
    assert np.asarray(bank["X"]).shape[1] == length and np.isfinite(np.asarray(bank["X"])).all() and np.asarray(bank["is_synthetic"]).all()
    assert np.allclose(np.asarray(bank["rr_prev"]), 0.7) and np.allclose(np.asarray(bank["rr_next"]), 0.9)  # copied from P13-form training beats
    assert record["scale"] == pytest.approx(rg.scale_of(arrays["X"][split == "train"]))
    meta = json.loads((banks.bank_dir(dif.generator_id()) / banks.BANK_FILE).read_text())
    assert "scale" in meta["normalization"]
    run_dir = runs.find_run_dir(entry["run_id"])
    assert all((run_dir / f).is_file() for f in runs.REQUIRED_RUN_FILES) and (run_dir / "checkpoints" / "final.pt").is_file()


def test_test_report_compares_the_diffusion_model_with_p13_and_checks_p13_first(monkeypatch):
    from types import SimpleNamespace

    from ecgaug import final_eval, paths
    from ecgaug.eval import metrics as M
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments import phase6_rr_baseline as v5

    records = np.repeat([100, 103, 105, 111], 40)
    y = np.tile(np.repeat([0, 1, 2, 3], 10), 4).astype(np.int8)
    test = {"X": np.zeros((len(y), 4), np.float32), "y": y, "record": records, "rr_prev": np.full(len(y), 0.7, np.float32),
            "rr_next": np.full(len(y), 0.9, np.float32), "rr_ratio": np.ones(len(y), np.float32)}  # fmt: skip

    def fake_predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
        out = []
        for run_id in run_ids:
            rng = np.random.default_rng(sum(map(ord, run_id)))
            out.append(np.eye(4)[np.where(rng.random(len(arrays["y"])) < 0.7, arrays["y"], rng.integers(0, 4, len(arrays["y"])))])
        return out

    monkeypatch.setattr(dif, "check_preconditions", lambda: None)
    monkeypatch.setattr(dif, "load_data", lambda *a, **k: SimpleNamespace(spec=None))
    monkeypatch.setattr(pt, "test_arrays", lambda spec, experiment: test)
    monkeypatch.setattr(v5, "record_ratios", lambda record, half, access: {})
    monkeypatch.setattr(v5, "substitute", lambda arrays, ratios: arrays)
    monkeypatch.setattr(pt, "mitbih_templates", lambda arrays, spec, half, access: None)
    monkeypatch.setattr(pt, "relative", lambda arrays, templates: arrays)
    monkeypatch.setattr(pt, "_predict", fake_predict)
    monkeypatch.setattr(rr, "mean_rr_by_record", lambda recs, access=None: {int(r): 0.5 for r in np.unique(recs)})

    def chosen(experiment: str, conditions: tuple) -> dict:
        return {"chosen": {"cnn": {c: {"strength": 0.25, "run_ids": [f"{experiment}-cnn.{c}-s{s}" for s in range(3)]} for c in conditions}}}

    for name, conditions in ((dif.EXPERIMENT, ("diffusion",)), (rr.EXPERIMENT, rr.CONDITIONS)):
        paths.results_for(name).mkdir(parents=True, exist_ok=True)
        (paths.results_for(name) / "selection.json").write_text(json.dumps(chosen(name, conditions)))

    def p13_summary(shift: float) -> None:
        body = {"conditions": {}}
        for c in rr.CONDITIONS:
            cms = final_eval.per_record_cms(fake_predict([f"{rr.EXPERIMENT}-cnn.{c}-s{s}" for s in range(3)], test), y, records)[1]
            body["conditions"][f"cnn/{c}"] = {"rare_macro_f1": {"point": float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1)))) + shift}}
        (paths.results_for(rr.EXPERIMENT) / "summary.json").write_text(json.dumps(body))

    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "rule_threshold": 0.8}
    p13_summary(0.01)
    with pytest.raises(PreconditionError):
        dif.test_report(cfg)
    p13_summary(0.0)
    s = dif.test_report(cfg)
    assert set(s["conditions"]) == {"cnn/diffusion"} and set(s["before"]) == {f"cnn/{c}" for c in dif.BEFORE}
    assert set(s["families"]["confirmatory"]) == {"cnn: diffusion vs none (P13)", "cnn: diffusion vs cvae (P13)"}
    assert set(s["families"]["secondary"]) == {"cnn: diffusion vs class_weighted (P13)", "cnn: diffusion vs random_oversampling (P13)"}
    assert set(s["sveb_rule"]) == {"cnn/diffusion", *(f"cnn/{c}" for c in dif.BEFORE)}
    assert all("p_holm" in c for f in s["families"].values() for c in f.values())
