"""P21: the simulator scores its own beats zero and the fit recovers them; the Euler loss adds a real template and scores
only S, V and F; it is off by default and moves the generators when on; whole beats are matched by (record, r_idx); the
fusion model recovers alpha and the fusion bank changes only F waveforms; the lambda rule; the stop rules; scoring on
P19's patients and the DS2 read reproduce P20's baselines."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ecgaug import paths
from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_physics_prior as pp
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.generators import build_generator
from ecgaug.generators import physics as ph
from ecgaug.paths import repo_root
from test_generators import CVAE, WGAN, training_part

CONFIG = "configs/phase6_physics_prior/physics_prior.yaml"


def _eta(scale: float = 8.0, baseline: float = 0.1) -> torch.Tensor:
    f = torch.float64
    return ph.join_eta(torch.tensor([ph.MCSHARRY_POSITIONS], dtype=f), torch.tensor([ph.MCSHARRY_HEIGHTS], dtype=f) * scale,
                       torch.tensor([ph.MCSHARRY_WIDTHS], dtype=f), torch.tensor([baseline], dtype=f))  # fmt: skip


def test_config_fixes_the_registered_values(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = pp.load_config(repo_root() / CONFIG)
    assert cfg["se"]["experiment"] == se.EXPERIMENT and set(cfg["rg"]["gen"]) == {"cvae", "wgan_gp"}
    for old, new in (("[0.1, 1.0, 10.0]", "[0.1, 1.0, 100.0]"), ("per_class: 5000", "per_class: 500")):
        bad = tmp_path / "bad.yaml"
        bad.write_text((repo_root() / CONFIG).read_text().replace(old, new).replace("configs/", str(repo_root() / "configs") + "/"))
        with pytest.raises(ValueError):
            pp.load_config(bad)


def test_the_simulator_scores_its_own_beat_zero_and_the_fit_recovers_a_shifted_one():
    omega = torch.tensor([2 * math.pi / 0.8], dtype=torch.float64)
    eta = _eta()
    beat = ph.simulate(eta, omega, torch.zeros(1, dtype=torch.float64), 180)
    assert float(ph.euler_distance(beat, eta, omega)) < 1e-20 and float(ph.euler_distance(beat + 0.1 * torch.sin(torch.arange(180.0)), eta, omega)) > 1
    shifted = eta.clone()
    shifted[0, :5] += torch.tensor([0.1, -0.05, 0.02, 0.05, -0.2], dtype=torch.float64)
    target = ph.simulate(shifted, omega, torch.zeros(1, dtype=torch.float64), 180)
    f = ph.fit(target.numpy(), omega.numpy(), iterations=300)
    assert f["r2"][0] > 0.999 and f["eta"].shape == (1, ph.N_ETA)
    d = ph.class_distribution(np.vstack([f["eta"], f["eta"]]))
    assert np.allclose(d["sd"], 0) and ph.kappa(target.numpy(), omega.numpy(), d) == pytest.approx(float(ph.euler_distance(target, torch.tensor(f["eta"]), omega)))
    assert ph.omega_of(np.array([0.4]), np.array([0.5]))[0] == pytest.approx(2 * math.pi / 0.8)


def _prior(length: int = 63, n: int = 6, scale: float = 2.0) -> ph.EulerPrior:
    g = np.random.default_rng(0)
    pools = {CLASS_INDEX["S"]: (g.standard_normal((n, length)).astype(np.float32), np.full(n, 2 * math.pi, np.float32))}
    physics = {CLASS_INDEX["S"]: {"mean": _eta()[0].tolist(), "sd": [0.01] * ph.N_ETA, "kappa": 5.0}}
    return ph.EulerPrior.build(pools, physics, scale)


def test_the_euler_loss_adds_a_real_template_and_scores_only_its_classes():
    prior = _prior()
    beats = torch.randn(4, 63)
    assert prior(beats, torch.zeros(4, dtype=torch.long)) is None  # N beats: no loss
    labels = torch.tensor([1, 0, 1, 2])
    torch.manual_seed(5)
    got = prior(beats, labels)
    torch.manual_seed(5)
    e = prior.classes[1]
    pick = torch.randint(e["templates"].shape[0], (2,))
    eta = e["mean"] + e["sd"] * torch.randn(2, ph.N_ETA)
    want = (ph.euler_distance(beats[labels == 1] * 2.0 + e["templates"][pick], eta, e["omega"][pick]) / 5.0).mean()
    assert float(got) == pytest.approx(float(want))


@pytest.mark.parametrize("family,settings", [("cvae", CVAE), ("wgan_gp", WGAN)], ids=["cvae", "wgan"])
def test_the_prior_is_off_by_default_and_moves_the_generator_when_on(family, settings):
    part = training_part()
    plain = build_generator(family, settings, 63, seed=3)
    plain.fit(part, steps=3, seed=3, log_every=1)
    zero = build_generator(family, settings, 63, seed=3)
    zero.prior, zero.prior_weight = _prior(), 0.0
    zero.fit(part, steps=3, seed=3, log_every=1)
    np.testing.assert_array_equal(plain.sample("S", 4, seed=1), zero.sample("S", 4, seed=1))
    on = build_generator(family, settings, 63, seed=3)
    on.prior, on.prior_weight = _prior(), 1.0
    fit = on.fit(part, steps=3, seed=3, log_every=1)
    assert all("prior" in row and np.isfinite(row["prior"]) for row in fit.history)
    assert not np.array_equal(plain.sample("S", 4, seed=1), on.sample("S", 4, seed=1))


def test_whole_beats_are_matched_by_record_and_r_idx():
    a = {"X": np.arange(12, dtype=np.float32).reshape(3, 4), "record": np.array([101, 101, 1]), "r_idx": np.array([5, 9, 5])}
    b = {"X": -np.ones((1, 4), np.float32), "record": np.array([800]), "r_idx": np.array([5])}
    train = {"X": np.zeros((3, 4), np.float32), "record": np.array([800, 1, 101]), "r_idx": np.array([5, 5, 9])}
    out = pp.match_whole(train, [a, b])
    np.testing.assert_array_equal(out, np.vstack([b["X"][0], a["X"][2], a["X"][1]]))
    with pytest.raises(PreconditionError):
        pp.match_whole({**train, "r_idx": np.array([5, 5, 7])}, [a, b])
    with pytest.raises(PreconditionError):
        pp.match_whole(train, [a, a])
    z = np.random.default_rng(0).standard_normal((5, 180))
    assert pp.zscore_share((z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True)) == 1.0 and pp.zscore_share(z) < 1


def _fusion_train(alpha: float = 0.4, noise: float = 0.02) -> dict:
    g = np.random.default_rng(1)
    rows, y, records = [], [], []
    for r in (101, 800, 1, 2):
        v = np.sin(np.linspace(0, 3, 180)) * (1 + 0.1 * r % 3)
        for _ in range(8):
            rows.append(v + noise * g.standard_normal(180))
            y.append(2)
            records.append(r)
        for _ in range(20):
            rows.append(g.uniform(alpha - 0.1, alpha + 0.1) * v + noise * g.standard_normal(180))
            y.append(3)
            records.append(r)
    return {"X": np.array(rows, np.float32), "y": np.array(y, np.int8), "record": np.array(records, np.int16)}


def test_the_fusion_model_recovers_alpha_and_its_bank_changes_only_f_waveforms():
    s, ex = pp.fusion_fit(_fusion_train())
    assert s["passed"] and s["f_beats"] == 80 and len(s["recordings"]) == 4 and s["median_r2"] > 0.9
    assert 0.3 < s["alpha_low"] < 0.4 < s["alpha_high"] < 0.5 and ex["fusion"]["n"] == 20
    assert not pp.fusion_fit(_fusion_train(noise=2.0))[0]["passed"]  # noise: the model explains too little
    g = np.random.default_rng(2)
    y = np.array([1, 3, 2, 3, 3], np.int8)
    bank = {"X": g.standard_normal((5, 180)).astype(np.float32), "y": y, "rr_prev": g.uniform(size=5).astype(np.float32),
            "source": np.array(["cvae"] * 5)}  # fmt: skip
    v_pool = g.standard_normal((7, 180)).astype(np.float32)
    out = pp.fusion_bank(bank, v_pool, 0.3, 0.5, seed=4)
    keep = y != 3
    np.testing.assert_array_equal(out["X"][keep], bank["X"][keep])
    np.testing.assert_array_equal(out["rr_prev"], bank["rr_prev"])
    ratios = [np.linalg.lstsq(v_pool.T, out["X"][i], rcond=None)[0] for i in np.flatnonzero(~keep)]
    assert all(np.isclose(np.abs(r).sum(), r.max(), atol=1e-4) and 0.3 <= r.max() <= 0.5 for r in ratios)
    assert set(out["source"][~keep]) == {"phase6_physics_prior/fusion"} and np.array_equal(out["X"], pp.fusion_bank(bank, v_pool, 0.3, 0.5, 4)["X"])


def test_the_lambda_rule_takes_the_best_mean_and_ties_to_the_smaller():
    def rows(means: dict[float, float]) -> dict:
        return {(pp.selection_condition("cvae", v), s): [{"selection_value": m + 0.01 * (s - 1), "run_id": f"r{v}-{s}"}] for v, m in means.items() for s in pp.SELECTION_SEEDS}

    c = pp.choose_lambda(rows({0.0: 0.9, 0.1: 0.5, 1.0: 0.7, 10.0: 0.6}), "cvae")
    assert c["chosen"] == 1.0 and c["table"]["0"]["mean"] == pytest.approx(0.9) and c["table"]["1"]["run_ids"][0] == "r1.0-0"
    assert pp.choose_lambda(rows({0.0: 0.1, 0.1: 0.7, 1.0: 0.7, 10.0: 0.6}), "cvae")["chosen"] == 0.1
    broken = rows({0.0: 0.1, 0.1: 0.7, 1.0: 0.7, 10.0: 0.6})
    del broken[(pp.selection_condition("cvae", 10.0), 2)]
    with pytest.raises(PreconditionError):
        pp.choose_lambda(broken, "cvae")
    assert pp.generator_id("wgan_gp", 0.1) == "phase6_physics_prior/wgan_gp-lambda0.1"


def test_the_stop_rules_refuse_the_conditions_they_govern():
    with pytest.raises(PreconditionError):
        pp.load_physics("physics")
    folder = paths.results_for(pp.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "physics.json").write_text(json.dumps({"physics_passed": False, "fusion": {"passed": True}}))
    with pytest.raises(PreconditionError):
        pp.load_physics("physics")
    assert pp.load_physics("fusion")["fusion"]["passed"]
    (folder / "physics.json").write_text(json.dumps({"physics_passed": True, "fusion": {"passed": False}}))
    assert pp.load_physics("physics")["physics_passed"]
    with pytest.raises(PreconditionError):
        pp.load_physics("fusion")


def _arrays(n_patients: int = 20, beats: int = 50, s_ratio: float = 0.62, n_ratio: float = 0.9) -> dict:
    y = np.tile(np.r_[np.zeros(beats - 8, np.int8), np.full(5, 1, np.int8), np.full(3, 2, np.int8)], n_patients)
    rr_prev = np.where(y == 1, s_ratio, n_ratio).astype(np.float32)
    return {"y": y, "record": np.repeat(10_000 + np.arange(n_patients), beats).astype(np.int16), "r_idx": np.arange(y.size, dtype=np.int32),
            "rr_prev": rr_prev, "p_af": np.zeros(y.size, np.float32)}  # fmt: skip


def _probs(a: dict) -> dict[str, np.ndarray]:
    """The baselines miss every S beat (only the stage finds them); the new conditions find half of them."""
    y = np.asarray(a["y"])
    miss = np.where((y == 2)[:, None], np.eye(4)[2], np.eye(4)[0]) * 0.9 + 0.025
    half = miss.copy()
    half[np.flatnonzero(y == 1)[::2]] = np.eye(4)[1] * 0.9 + 0.025
    return {"cvae": miss, "wgan_gp": miss, **{c: half for c in pp.NEW}}


def _fake_p20(a: dict, probs: dict) -> dict:
    p20 = rt.choose({b: probs[b] for b in ("cvae", "wgan_gp")}, a)
    folder = paths.results_for(rt.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "choice.json").write_text(json.dumps(p20))
    y, rr_prev, p_af = np.asarray(a["y"]), np.asarray(a["rr_prev"]), np.asarray(a["p_af"])
    s = rt.chosen_settings(p20)
    chosen = {f"cnn/{b}": {"f1_S": {"point": rt.pooled_f1_s(rt.decide(probs[b], rr_prev, p_af, s[b], "rule"), y)}} for b in s}
    (folder / "summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    return p20


def test_scoring_on_p19s_patients_compares_with_p20s_baselines_and_checks_their_values(monkeypatch):
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    probs = _probs(a)
    _fake_p20(a, probs)
    choice = pp.choose(probs, a)
    assert set(choice["chosen"]) == set(pp.NEW) and choice["experiment"] == pp.EXPERIMENT
    s = pp.score(choice, probs, a)
    assert set(s["families"]["secondary"]) == {f"cnn: {c} vs {b}, chosen settings" for c, b in pp.NEW.items()}
    assert all(abs(d) < 1e-12 for d in s["reproduction_difference"].values()) and s["patients"] == 20
    assert s["conditions"]["chosen"]["cnn/cvae_physics"]["f1_S"]["point"] == pytest.approx(1.0)
    path = paths.results_for(rt.EXPERIMENT) / "summary.json"
    bad = json.loads(path.read_text())
    bad["conditions"]["chosen"]["cnn/wgan_gp"]["f1_S"]["point"] -= 0.01
    path.write_text(json.dumps(bad))
    with pytest.raises(PreconditionError):
        pp.score(choice, probs, a)


def test_the_ds2_read_checks_p20s_values_and_compares_with_the_baselines(monkeypatch):
    from ecgaug import final_eval

    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    y = np.asarray(a["y"]).copy()
    y[np.flatnonzero(y == 2)[::3]] = 3
    a = {**a, "y": y}
    probs = _probs(a)
    p20 = _fake_p20(a, probs)
    choice = pp.choose(probs, a)
    pp._write("choice.json", choice)
    members = {c: [f"{c}-s{s}" for s in range(10)] for c in probs}
    monkeypatch.setattr(pp, "new_members", lambda: {c: members[c] for c in pp.NEW})
    monkeypatch.setattr(pp, "baseline_members", lambda: {b: members[b] for b in ("cvae", "wgan_gp")})
    test = {"y": y, "record": (100 + np.asarray(a["record"]) - 10_000).astype(np.int16), "rr_prev": a["rr_prev"], "r_idx": a["r_idx"]}
    monkeypatch.setattr(se, "ds2_inputs", lambda cfg, experiment: (test, a["p_af"]))
    monkeypatch.setattr(pp.pt, "_predict", lambda ids, t: [probs[ids[0].rsplit("-", 1)[0]]] * len(ids))
    chosen = {}
    for b, s in rt.chosen_settings(p20).items():
        onehot = np.eye(4, dtype=np.float32)[rt.decide(probs[b], a["rr_prev"], a["p_af"], s, "rule")]
        chosen[f"cnn/{b}"] = {"rare_macro_f1": {"point": se._rare(final_eval.per_record_cms([onehot], y, test["record"])[1])}}
    (paths.results_for(rt.EXPERIMENT) / "test_summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    s = pp.test_report({"se": {}})
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c} vs {b}, chosen settings" for c, b in pp.NEW.items()}
    assert all(d == 0 for d in s["reproduction_difference"].values()) and len(s["records"]) == 20
    assert set(s["conditions"]) == {"chosen", "ensemble_alone"} and "cnn/cvae_fusion" in s["without_202"]["chosen"]
    chosen["cnn/cvae"]["rare_macro_f1"]["point"] += 0.01
    (paths.results_for(rt.EXPERIMENT) / "test_summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    with pytest.raises(PreconditionError):
        pp.test_report({"se": {}})


def test_figures_and_paper_text_come_from_the_result_files():
    from ecgaug import figures, paper_text

    assert "\\missing{P21: --part check" in paper_text.posthoc_physics_prior()
    wave = np.sin(np.linspace(0, 6, 180)).tolist()
    ex = {"real": wave, "fitted": wave, "r2": 0.99}
    physics = {"r2_stop": 0.5, "physics_passed": True,
               "classes": {c: {"beats": 5000, "median_r2": 0.99} for c in ("S", "V", "F")},
               "fusion": {"f_beats": 400, "recordings": [208, 800, 1], "median_r2": 0.8, "alpha_low": 0.3, "alpha_high": 0.7, "passed": True,
                          "alpha_quantiles": {"q10": 0.3, "q25": 0.4, "q50": 0.5, "q75": 0.6, "q90": 0.7}},
               "examples": {**{c: {"q10": ex, "q50": ex, "q90": ex} for c in ("S", "V", "F")},
                            "fusion": {"record": 208, "n": 300, "real": wave, "fitted": wave, "vbar": wave, "alpha": 0.5}}}  # fmt: skip
    folder = paths.results_for(pp.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "physics.json").write_text(json.dumps(physics))
    assert "\\missing{P21: --part train" in paper_text.posthoc_physics_prior()
    bank = {c: {"euler": 1.5, "median_corr": 0.9, "nearest_patients": 20.0} for c in ("S", "V", "F")}
    for f in ("cvae", "wgan_gp"):
        table = {k: {"mean": 0.7, "values": [0.69, 0.7, 0.71], "run_ids": ["a", "b", "c"]} for k in ("0", "0.1", "1", "10")}
        d = {"lambda_choice": {"table": table}, "chosen_lambda": 1.0,
             "describe": {"real": bank, "p12_seed0": bank, **{f"lambda{k}_seed0": bank for k in ("0.1", "1", "10")}}}  # fmt: skip
        (folder / f"{f}.json").write_text(json.dumps(d))
    (folder / "fusion.json").write_text(json.dumps({"nearest_patients_F": {"fusion_seed0": 30.0, "p12_seed0": 3.4}}))
    e = {"point": 0.4, "ci_low": 0.3, "ci_high": 0.5}
    block = {f"cnn/{c}": {"f1_S": e, "f1_F": e, "rare_macro_f1": e} for c in (*pp.NEW, "cvae", "wgan_gp")}
    comp = {"point": 0.02, "ci_low": -0.01, "ci_high": 0.05, "p_holm": 0.5, "reject_holm": False}
    names = [f"cnn: {c} vs {b}, chosen settings" for c, b in pp.NEW.items()]
    summary = {"patients": 977, "conditions": {"chosen": block, "ensemble_alone": block}, "families": {"secondary": {n: comp for n in names}}}
    (folder / "summary.json").write_text(json.dumps(summary))
    assert "\\missing{P21: --part test" in paper_text.posthoc_physics_prior()
    test = {"conditions": {"chosen": block, "ensemble_alone": block}, "families": {"confirmatory": {n: comp for n in names}}}
    (folder / "test_summary.json").write_text(json.dumps(test))
    out = figures.physics_check_figure(physics) + figures.physics_banks_figure({f: json.loads((folder / f"{f}.json").read_text()) for f in ("cvae", "wgan_gp")},
                                                                               json.loads((folder / "fusion.json").read_text())) + figures.physics_results_figure(summary, test)  # fmt: skip
    assert len(out) == 3 and all(p.is_file() for p in out)
    text = paper_text.posthoc_physics_prior()
    assert "\\missing" not in text and "WGAN-GP with the Euler loss" in text and "cite{golany2020simgans}" in text and text.count("{") == text.count("}")
    assert "fig:p21-fit" in text and "fig:p21-banks" in text and "fig:p21}" in text and "_" not in text.replace("\\_", "")
