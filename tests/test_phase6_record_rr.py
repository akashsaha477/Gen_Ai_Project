"""P13: RR intervals over the recording's mean RR, Huang's SVEB rule, the bank RR mapping, the grid and the DS2 report (hand-made cases)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root


def test_record_mean_rr_is_the_mean_of_every_interval_pauses_included():
    assert rr.record_mean_rr(np.array([0, 360, 1080, 1440]), 360) == pytest.approx((1 + 2 + 1) / 3)
    with pytest.raises(ValueError):
        rr.record_mean_rr(np.array([5]), 360)


def test_normalize_divides_rr_prev_and_rr_next_by_their_recordings_mean_and_keeps_the_local_ratio():
    arrays = {"record": np.array([100, 100, 801]), "rr_prev": np.array([0.5, 1.0, 0.4], np.float32),
              "rr_next": np.array([1.0, 1.0, 0.8], np.float32), "rr_ratio": np.array([0.6, 1.1, 0.9], np.float32)}  # fmt: skip
    out = rr.normalize(arrays, {100: 1.0, 801: 0.5})
    assert np.allclose(out["rr_prev"], [0.5, 1.0, 0.8]) and np.allclose(out["rr_next"], [1.0, 1.0, 1.6])
    assert out["rr_ratio"] is arrays["rr_ratio"]
    with pytest.raises(PreconditionError):
        rr.normalize(arrays, {100: 1.0})


def test_an_s_dominant_record_escapes_the_local_ratios_but_not_the_recording_mean():
    """S beats in runs at 0.8 s without a pause, each run ended by a 2.4 s pause before an N beat (the pattern of SVDB 865)."""
    fs, t, symbols = 360, [0.0], ["N"]
    for _ in range(40):
        for _ in range(6):
            t.append(t[-1] + 0.8)
            symbols.append("A")
        t.append(t[-1] + 2.4)
        symbols.append("N")
    table = rr.ratio_table(np.rint(np.array(t) * fs).astype(np.int64), np.array(symbols), fs, half_window_s=150.0)
    assert rr._rule(table, "v2", 0.8)["se"] < 0.05  # the local median is the S beats' own interval
    assert rr._rule(table, "record", 0.8)["se"] > 0.95  # mean RR (6 x 0.8 + 2.4) / 7 = 1.03 s, so 0.8 s reads as premature
    assert rr._rule(table, "v1", 0.8)["se"] < rr._rule(table, "record", 0.8)["se"]


def test_sveb_rule_turns_premature_n_calls_into_s_and_leaves_every_other_call():
    probs = np.eye(4)[[0, 0, 2, 3, 1, 0]]
    ratio = np.array([0.7, 0.9, 0.5, 0.5, 0.95, 0.79])
    out = rr.sveb_rule(probs, ratio, 0.8).argmax(axis=1)
    assert out.tolist() == [1, 0, 2, 3, 1, 1]


def _train() -> dict:
    y = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int8)
    n = len(y)
    return {"X": np.zeros((n, 4), np.float32), "y": y, "record": np.arange(n, dtype=np.int16), "split": np.array(["train"] * n),
            "r_idx": np.arange(n, dtype=np.int32), "rr_prev": np.arange(n, dtype=np.float32) + 0.5, "rr_next": np.arange(n, dtype=np.float32) + 0.25,
            "rr_ratio": np.arange(n, dtype=np.float32) + 0.75, "is_synthetic": np.zeros(n, bool), "source": np.array(["real"] * n)}  # fmt: skip


class ConstantGenerator:
    def sample(self, cls: str, n: int, seed: int) -> np.ndarray:
        return np.full((n, 4), CLASS_INDEX[cls], dtype=np.float32)


def test_bank_rr_comes_from_the_same_real_beat_in_the_new_form_and_a_changed_bank_is_refused():
    before = _train()
    after = {**before, "rr_prev": before["rr_prev"] * 2, "rr_next": before["rr_next"] * 3}
    counts = {"S": 5, "V": 3, "F": 4}
    bank = rg.sample_relative_bank(ConstantGenerator(), before, counts, seed=7, scale=1.0, source="p12")  # P12's own export
    out = rr.v4_bank(bank, before, after, counts, seed=7)
    assert np.allclose(out["rr_prev"], np.asarray(bank["rr_prev"]) * 2) and np.allclose(out["rr_next"], np.asarray(bank["rr_next"]) * 3)
    assert np.array_equal(out["rr_ratio"], bank["rr_ratio"]) and np.array_equal(out["X"], bank["X"])
    with pytest.raises(PreconditionError):
        rr.v4_bank(bank, before, after, counts, seed=8)  # other draws: the stored values do not match
    tampered = {**bank, "rr_prev": np.asarray(bank["rr_prev"]) + 1}
    with pytest.raises(PreconditionError):
        rr.v4_bank(tampered, before, after, counts, seed=7)


def test_grid_carries_p11s_and_p12s_strengths_for_the_cnn_unscreened(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")  # read-only: the real configs and P11's and P12's committed selections
    cfg = rr.load_config(repo_root() / "configs/phase6_record_rr/record_rr.yaml")
    grid = rr.record_grid(cfg)
    assert list(grid["architectures"]) == ["cnn"] and grid["policy"] == {"kind": "fraction_of_n"} and list(grid["conditions"]) == list(rr.CONDITIONS)
    s = rr.strengths()
    assert grid["conditions"]["class_weighted"] == {"gammas": [s["class_weighted"]]}
    assert grid["conditions"]["random_oversampling"] == {"strengths": [s["random_oversampling"]]}
    assert {f: grid["conditions"][f]["bank"] for f in rr.GENERATIVE} == {f: rg.generator_id(cfg["rg"], f) for f in rr.GENERATIVE}
    assert all(grid["conditions"][f]["strengths"] == [s[f]] for f in rr.GENERATIVE)


def test_config_rejects_a_threshold_outside_zero_one(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: phase6_record_rr\nrelative_generators_config: x\nrule_threshold: 1.2\ncheck: {}\n")
    with pytest.raises(ValueError):
        rr.load_config(bad)


def test_test_report_compares_each_condition_with_before_and_checks_p11_and_p12(monkeypatch):
    """The DS2 read on fake beats and fake predictions: both families, the SVEB rule and the reproduction guard."""
    from types import SimpleNamespace

    from ecgaug import final_eval, paths
    from ecgaug.eval import metrics as M
    from ecgaug.experiments import phase6_more_patients as mp
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments import phase6_rr_baseline as v5

    records = np.repeat([100, 103, 105, 111], 40)
    y = np.tile(np.repeat([0, 1, 2, 3], 10), 4).astype(np.int8)
    test = {"X": np.zeros((len(y), 4), np.float32), "y": y, "record": records, "rr_prev": np.full(len(y), 0.7, np.float32),
            "rr_next": np.full(len(y), 0.9, np.float32), "rr_ratio": np.ones(len(y), np.float32)}  # fmt: skip

    def fake_predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
        out = []
        for run_id in run_ids:  # each run keeps 70% of the labels and guesses the rest; the new form shifts the guesses
            rng = np.random.default_rng(sum(map(ord, run_id)) + int(float(np.asarray(arrays["rr_prev"])[0]) * 100))
            out.append(np.eye(4)[np.where(rng.random(len(arrays["y"])) < 0.7, arrays["y"], rng.integers(0, 4, len(arrays["y"])))])
        return out

    monkeypatch.setattr(rr, "check_preconditions", lambda: None)
    monkeypatch.setattr(rr, "load_data", lambda *a, **k: SimpleNamespace(spec=None))
    monkeypatch.setattr(pt, "test_arrays", lambda spec, experiment: test)
    monkeypatch.setattr(v5, "record_ratios", lambda record, half, access: {})
    monkeypatch.setattr(v5, "substitute", lambda arrays, ratios: arrays)
    monkeypatch.setattr(pt, "mitbih_templates", lambda arrays, spec, half, access: None)
    monkeypatch.setattr(pt, "relative", lambda arrays, templates: arrays)
    monkeypatch.setattr(pt, "_predict", fake_predict)
    monkeypatch.setattr(rr, "mean_rr_by_record", lambda recs, access=None: {int(r): 0.5 for r in np.unique(recs)})

    def chosen(experiment: str, conditions: tuple, strength: float) -> dict:
        return {"chosen": {"cnn": {c: {"strength": strength, "run_ids": [f"{experiment}-cnn.{c}-s{s}" for s in range(3)]} for c in conditions}}}

    selections = {rr.EXPERIMENT: chosen(rr.EXPERIMENT, rr.CONDITIONS, 0.25), mp.EXPERIMENT: chosen(mp.EXPERIMENT, rr.CONDITIONS, 0.1),
                  rg.EXPERIMENT: chosen(rg.EXPERIMENT, rr.GENERATIVE, 0.5)}  # fmt: skip
    for name, selection in selections.items():
        paths.results_for(name).mkdir(parents=True, exist_ok=True)
        (paths.results_for(name) / "selection.json").write_text(json.dumps(selection))

    def score(condition: str) -> float:
        sel = selections[rg.EXPERIMENT if condition in rr.GENERATIVE else mp.EXPERIMENT]["chosen"]["cnn"][condition]["run_ids"]
        cms = final_eval.per_record_cms(fake_predict(sel, test), y, records)[1]
        return float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1))))

    def write_before(shift: float) -> None:
        for name, conditions in ((mp.EXPERIMENT, ("none", "class_weighted", "random_oversampling")), (rg.EXPERIMENT, rr.GENERATIVE)):
            body = {"conditions": {f"cnn/{c}": {"rare_macro_f1": {"point": score(c) + shift}} for c in conditions}}
            (paths.results_for(name) / "summary.json").write_text(json.dumps(body))

    cfg = {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}, "rule_threshold": 0.8}
    write_before(0.01)
    with pytest.raises(PreconditionError):  # P11's and P12's scores must reproduce before anything is compared
        rr.test_report(cfg)
    write_before(0.0)
    s = rr.test_report(cfg)
    assert set(s["conditions"]) == set(s["before"]) == set(s["sveb_rule"]) == {f"cnn/{c}" for c in rr.CONDITIONS}
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, recording-mean RR vs before" for c in rr.CONDITIONS}
    assert len(s["families"]["secondary"]) == 6 and all("p_holm" in c for f in s["families"].values() for c in f.values())
    assert all(abs(d) < 1e-12 for d in s["reproduction_difference"].values())
    for c in rr.CONDITIONS:
        assert s["families"]["confirmatory"][f"cnn: {c}, recording-mean RR vs before"]["point"] == pytest.approx(
            s["conditions"][f"cnn/{c}"]["rare_macro_f1"]["point"] - s["before"][f"cnn/{c}"]["rare_macro_f1"]["point"])
    # every fake test beat has rr_prev 0.7 / 0.5 = 1.4 over its recording's mean, so the rule changes nothing
    assert s["sveb_rule"]["cnn/none"]["rare_macro_f1"]["point"] == pytest.approx(s["conditions"]["cnn/none"]["rare_macro_f1"]["point"])


def test_kaggle_keeps_p12s_nested_banks_apart_from_the_flat_phase3_banks_of_the_same_name(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kaggle_run", Path(__file__).resolve().parents[1] / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)
    layouts = {"ecgaug-banks/phase3_cvae-D": b"phase 3", "ecgaug-p12-banks/data/synthetic/phase6_relative_generators/phase3_cvae-D": b"P12"}
    for folder, content in layouts.items():
        bank = tmp_path / "input" / folder
        bank.mkdir(parents=True)
        (bank / "bank.json").write_text("{}")
        (bank / "seed0.npz").write_bytes(content)
    copied = kaggle_run.setup_artifacts(str(tmp_path / "input"), repo=tmp_path / "repo")
    synthetic = tmp_path / "repo" / "data" / "synthetic"
    assert (synthetic / "phase3_cvae-D" / "seed0.npz").read_bytes() == b"phase 3"
    assert (synthetic / "phase6_relative_generators" / "phase3_cvae-D" / "seed0.npz").read_bytes() == b"P12"
    assert "data/synthetic/phase6_relative_generators/phase3_cvae-D/seed0.npz" in copied
