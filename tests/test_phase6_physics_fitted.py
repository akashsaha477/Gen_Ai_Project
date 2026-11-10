"""P23: the fitted simulator explains its own beats fully and solves heights exactly; the fitted physics loss adds a real
template, scores only S, V and F and moves the generators when on; the stop rule; the shared comparison of new ensembles
with P18's (P22 and P23) chooses on P14's patients, checks P20's values and reads DS2 once."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ecgaug import paths
from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import ensemble_compare as ec
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_physics_fitted as pf
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.generators import build_generator
from ecgaug.generators import physics as ph
from ecgaug.paths import repo_root
from test_generators import CVAE, WGAN, training_part

CONFIG = "configs/phase6_physics_fitted/physics_fitted.yaml"


def test_config_fixes_the_registered_values(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = pf.load_config(repo_root() / CONFIG)
    assert cfg["p21"]["experiment"] == "phase6_physics_prior" and cfg["se"]["experiment"] == se.EXPERIMENT
    bad = tmp_path / "bad.yaml"
    bad.write_text((repo_root() / CONFIG).read_text().replace("[0.1, 1.0, 10.0]", "[1.0, 10.0]").replace("configs/", str(repo_root() / "configs") + "/"))
    with pytest.raises(ValueError):
        pf.load_config(bad)


def _eta(scale: float = 8.0, baseline: float = 0.1) -> torch.Tensor:
    f = torch.float64
    return ph.join_eta(torch.tensor([ph.MCSHARRY_POSITIONS], dtype=f), torch.tensor([ph.MCSHARRY_HEIGHTS], dtype=f) * scale,
                       torch.tensor([ph.MCSHARRY_WIDTHS], dtype=f), torch.tensor([baseline], dtype=f))  # fmt: skip


def test_the_fitted_simulator_explains_its_own_beats_and_solves_heights_exactly():
    omega = torch.tensor([2 * math.pi / 0.8, 2 * math.pi / 0.8], dtype=torch.float64)
    beats = torch.cat([ph.simulate(_eta(), omega[:1], torch.tensor([0.3], dtype=torch.float64), 180),
                       ph.simulate(_eta(scale=-3.0, baseline=-0.4), omega[:1], torch.tensor([-0.2], dtype=torch.float64), 180)])  # fmt: skip
    positions, widths = ph.class_shape({"mean": _eta()[0].tolist()})
    r = ph.simulation_fit(beats, omega, positions, widths)
    assert torch.all(r["distance"] < 1e-6) and torch.all(r["r2"] > 1 - 1e-6)  # any heights and baseline: solved exactly (a 1e-9 ridge)
    rough = beats + 0.05 * torch.sin(torch.arange(180.0, dtype=torch.float64) * 2.5)
    assert torch.all(ph.simulation_fit(rough, omega, positions, widths)["distance"] > 0.1)
    x = rough.clone().requires_grad_(True)
    ph.simulation_fit(x, omega, positions, widths)["distance"].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def _prior(length: int = 63, n: int = 6, scale: float = 2.0) -> ph.FittedPrior:
    g = np.random.default_rng(0)
    pools = {CLASS_INDEX["S"]: (g.standard_normal((n, length)).astype(np.float32), np.full(n, 2 * math.pi, np.float32))}
    shapes = {CLASS_INDEX["S"]: {"mean": _eta()[0].tolist(), "kappa": 5.0}}
    return ph.FittedPrior.build(pools, shapes, scale)


def test_the_fitted_loss_adds_a_real_template_and_scores_only_its_classes():
    prior = _prior()
    beats = torch.randn(4, 63)
    assert prior(beats, torch.zeros(4, dtype=torch.long)) is None  # N beats: no loss
    labels = torch.tensor([1, 0, 1, 2])
    torch.manual_seed(5)
    got = prior(beats, labels)
    torch.manual_seed(5)
    e = prior.classes[1]
    pick = torch.randint(e["templates"].shape[0], (2,))
    want = (ph.simulation_fit(beats[labels == 1] * 2.0 + e["templates"][pick], e["omega"][pick], e["positions"], e["widths"])["distance"] / 5.0).mean()
    assert got.dtype == beats.dtype and float(got) == pytest.approx(float(want), rel=1e-5)


@pytest.mark.parametrize("family,settings", [("cvae", CVAE), ("wgan_gp", WGAN)], ids=["cvae", "wgan"])
def test_the_fitted_loss_moves_the_generator_when_on(family, settings):
    part = training_part()
    plain = build_generator(family, settings, 63, seed=3)
    plain.fit(part, steps=3, seed=3, log_every=1)
    on = build_generator(family, settings, 63, seed=3)
    on.prior, on.prior_weight = _prior(), 1.0
    fit = on.fit(part, steps=3, seed=3, log_every=1)
    assert all("prior" in row and np.isfinite(row["prior"]) for row in fit.history)
    assert not np.array_equal(plain.sample("S", 4, seed=1), on.sample("S", 4, seed=1))


def test_the_fit_samples_are_p21s_draw_and_the_stop_rule_refuses():
    g = np.random.default_rng(3)
    y = np.repeat(np.array([1, 2, 3, 0], np.int8), [7, 9, 4, 5])
    train = {"y": y, "rr_prev": g.uniform(0.6, 1.0, y.size), "rr_ratio": g.uniform(0.9, 1.1, y.size)}
    whole = g.standard_normal((y.size, 10))
    out = pf.fit_samples(train, whole)
    rng = np.random.default_rng(0)
    for cls in ("S", "V", "F"):  # P21's check: one generator, classes in this order, at most 5,000 each
        pool = np.flatnonzero(y == CLASS_INDEX[cls])
        pick = np.sort(rng.choice(pool, size=pool.size, replace=False))
        np.testing.assert_array_equal(out[cls][0], whole[pick])
    with pytest.raises(PreconditionError):
        pf.load_physics()
    pf._write("physics.json", {"physics_passed": False})
    with pytest.raises(PreconditionError):
        pf.load_physics()
    with pytest.raises(PreconditionError):
        pf.new_members()
    assert pf.roughness(np.array([[0.0, 1.0, 3.0]])) == pytest.approx(2.5)


# --------------------------------------------------------------------------------------------------
# the shared comparison (ensemble_compare), with P23's pairs
# --------------------------------------------------------------------------------------------------


def _arrays(n_patients: int = 20, beats: int = 50) -> dict:
    y = np.tile(np.r_[np.zeros(beats - 8, np.int8), np.full(5, 1, np.int8), np.full(3, 2, np.int8)], n_patients)
    rr_prev = np.where(y == 1, 0.62, 0.9).astype(np.float32)
    return {"y": y, "record": np.repeat(10_000 + np.arange(n_patients), beats).astype(np.int16), "r_idx": np.arange(y.size, dtype=np.int32),
            "rr_prev": rr_prev, "p_af": np.zeros(y.size, np.float32)}  # fmt: skip


def _probs(a: dict) -> dict[str, np.ndarray]:
    """The baselines miss every S beat (only the stage finds them); the new conditions find half of them."""
    y = np.asarray(a["y"])
    miss = np.where((y == 2)[:, None], np.eye(4)[2], np.eye(4)[0]) * 0.9 + 0.025
    half = miss.copy()
    half[np.flatnonzero(y == 1)[::2]] = np.eye(4)[1] * 0.9 + 0.025
    return {"cnn/cvae": miss, "cnn/wgan_gp": miss, **{c: half for c in pf.PAIRS}}


def _fake_p20(a: dict, probs: dict) -> dict:
    p20 = rt.choose({ec.base_name(b): probs[b] for b in ("cnn/cvae", "cnn/wgan_gp")}, a)
    folder = paths.results_for(rt.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "choice.json").write_text(json.dumps(p20))
    y, rr_prev, p_af = np.asarray(a["y"]), np.asarray(a["rr_prev"]), np.asarray(a["p_af"])
    s = rt.chosen_settings(p20)
    chosen = {f"cnn/{b}": {"f1_S": {"point": rt.pooled_f1_s(rt.decide(probs[f'cnn/{b}'], rr_prev, p_af, s[b], "rule"), y)}} for b in s}
    (folder / "summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    return p20


def test_scoring_on_p19s_patients_compares_with_p20s_baselines_and_checks_their_values(monkeypatch):
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    probs = _probs(a)
    _fake_p20(a, probs)
    choice = ec.choose(pf.EXPERIMENT, pf.RULE, probs, a, list(pf.PAIRS))
    assert set(choice["chosen"]) == {"cnn/cvae_fitted", "cnn/wgan_gp_fitted"} and choice["experiment"] == pf.EXPERIMENT
    within = {"x": ("cnn/cvae_fitted", "cnn/cvae")}
    s = ec.score_patients(pf.EXPERIMENT, pf.RULE, choice, probs, a, pf.PAIRS, within)
    assert set(s["families"]["secondary"]) == {f"{c} vs {b}, chosen settings" for c, b in pf.PAIRS.items()} and "within" in s
    assert all(abs(d) < 1e-12 for d in s["reproduction_difference"].values()) and s["patients"] == 20
    assert s["conditions"]["chosen"]["cnn/cvae_fitted"]["f1_S"]["point"] == pytest.approx(1.0)
    assert s["baseline_settings"]["cnn/cvae"] == rt.label(rt.chosen_settings(rt.load_choice())["cvae"])
    path = paths.results_for(rt.EXPERIMENT) / "summary.json"
    bad = json.loads(path.read_text())
    bad["conditions"]["chosen"]["cnn/wgan_gp"]["f1_S"]["point"] -= 0.01
    path.write_text(json.dumps(bad))
    with pytest.raises(PreconditionError):
        ec.score_patients(pf.EXPERIMENT, pf.RULE, choice, probs, a, pf.PAIRS)


def test_the_ds2_read_checks_p20s_values_and_compares_with_the_baselines(monkeypatch):
    from ecgaug import final_eval

    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    y = np.asarray(a["y"]).copy()
    y[np.flatnonzero(y == 2)[::3]] = 3
    a = {**a, "y": y}
    probs = _probs(a)
    p20 = _fake_p20(a, probs)
    choice = ec.choose(pf.EXPERIMENT, pf.RULE, probs, a, list(pf.PAIRS))
    pf._write("choice.json", choice)
    members = {c: [f"{c}-s{s}" for s in range(10)] for c in probs}
    monkeypatch.setattr(pf, "new_members", lambda: {c: members[c] for c in pf.PAIRS})
    monkeypatch.setattr(ec, "baseline_members", lambda pairs: {b: members[b] for b in ("cnn/cvae", "cnn/wgan_gp")})
    test = {"y": y, "record": (100 + np.asarray(a["record"]) - 10_000).astype(np.int16), "rr_prev": a["rr_prev"], "r_idx": a["r_idx"]}
    monkeypatch.setattr(se, "ds2_inputs", lambda cfg, experiment: (test, a["p_af"]))
    monkeypatch.setattr(ec.pt, "_predict", lambda ids, t: [probs[ids[0].rsplit("-", 1)[0]]] * len(ids))
    chosen = {}
    for b, s in rt.chosen_settings(p20).items():
        onehot = np.eye(4, dtype=np.float32)[rt.decide(probs[f"cnn/{b}"], a["rr_prev"], a["p_af"], s, "rule")]
        chosen[f"cnn/{b}"] = {"rare_macro_f1": {"point": se._rare(final_eval.per_record_cms([onehot], y, test["record"])[1])}}
    (paths.results_for(rt.EXPERIMENT) / "test_summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    s = pf.test_report({"se": {}})
    assert set(s["families"]["confirmatory"]) == {f"{c} vs {b}, chosen settings" for c, b in pf.PAIRS.items()}
    assert all(d == 0 for d in s["reproduction_difference"].values()) and len(s["records"]) == 20
    assert "cnn/cvae_fitted" in s["without_202"]["chosen"] and s["note"].startswith("confirmatory family (Holm, 2)")
    chosen["cnn/cvae"]["rare_macro_f1"]["point"] += 0.01
    (paths.results_for(rt.EXPERIMENT) / "test_summary.json").write_text(json.dumps({"conditions": {"chosen": chosen}}))
    with pytest.raises(PreconditionError):
        pf.test_report({"se": {}})


def test_figures_and_paper_text_come_from_the_result_files(monkeypatch):
    from ecgaug import figures, paper_text

    assert "\\missing{P23: --part check" in paper_text.posthoc_physics_fitted()
    q = {"q10": 0.4, "q25": 0.6, "q50": 0.8, "q75": 0.9, "q90": 0.95}
    entry = {c: {"physics": 1.2, "roughness": 1.1, "median_corr": 0.9, "nearest_patients": 20.0} for c in ("S", "V", "F")}
    check = {"r2_stop": 0.5, "physics_passed": True, "classes": {c: {"beats": 5000, "median_r2": 0.8, "r2_quantiles": q, "p21_median_r2": 0.99, "kappa": 3.0} for c in ("S", "V", "F")},
             "describe": {"real": entry, "templates_alone": {c: {"physics": 0.5} for c in ("S", "V", "F")}, "p12_cvae_seed0": entry, "p21_wgan_gp_seed0": entry}}  # fmt: skip
    pf._write("physics.json", check)
    assert "\\missing{P23: --part train" in paper_text.posthoc_physics_fitted()
    for f in ("cvae", "wgan_gp"):
        table = {k: {"mean": 0.7, "values": [0.69, 0.7, 0.71], "run_ids": ["a", "b", "c"]} for k in ("0", "0.1", "1", "10")}
        describe = {"real": entry, "templates_alone": {c: {"physics": 0.5} for c in ("S", "V", "F")}, "p12_seed0": entry, "p21_seed0": entry,
                    **{f"lambda{k}_seed0": entry for k in ("0.1", "1", "10")}}  # fmt: skip
        pf._write(f"{f}.json", {"lambda_choice": {"table": table}, "chosen_lambda": 1.0, "describe": describe})
    assert "\\missing{P23: the scoring session" in paper_text.posthoc_physics_fitted()
    e = {"point": 0.4, "ci_low": 0.3, "ci_high": 0.5}
    block = {c: {"f1_S": e, "f1_F": e, "rare_macro_f1": e} for c in (*pf.PAIRS, *pf.PAIRS.values())}
    comp = {"point": 0.02, "ci_low": -0.01, "ci_high": 0.05, "p_holm": 0.5, "reject_holm": False}
    names = [f"{c} vs {b}, chosen settings" for c, b in pf.PAIRS.items()]
    summary = {"patients": 977, "conditions": {"chosen": block, "ensemble_alone": block}, "families": {"secondary": {n: comp for n in names}}}
    pf._write("summary.json", summary)
    assert "\\missing{P23: --part test" in paper_text.posthoc_physics_fitted()
    test = {"conditions": {"chosen": block, "ensemble_alone": block}, "families": {"confirmatory": {n: comp for n in names}}}
    pf._write("test_summary.json", test)
    fams = {f: json.loads((paths.results_for(pf.EXPERIMENT) / f"{f}.json").read_text()) for f in ("cvae", "wgan_gp")}
    out = (figures.physics_fitted_check_figure(check) + figures.physics_fitted_banks_figure(fams)
           + figures.ensemble_pairs_figure(summary, test, pf.PAIRS, pf.EXPERIMENT, "t", "c"))  # fmt: skip
    assert len(out) == 3 and all(p.is_file() for p in out)
    text = paper_text.posthoc_physics_fitted()
    assert "\\missing" not in text and "the WGAN-GP with the fitted loss against the cnn with the WGAN-GP" in text
    assert "fig:p23-check" in text and "fig:p23-banks" in text and "fig:p23}" in text and text.count("{") == text.count("}") and "_" not in text
    pf._write("physics.json", {**check, "physics_passed": False})
    assert "neither condition was run" in paper_text.posthoc_physics_fitted()
