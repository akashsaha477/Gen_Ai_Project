"""P20: the config fixes the grid; the choice takes the best pooled S F1 on the tuning patients with the registered tie
order and is written before the evaluation patients are loaded; scoring compares with P19's setting and the ensemble alone
and checks P19's own value; the evaluation beats must reproduce P19's; the DS2 read checks P18's values."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug import paths
from ecgaug.data.contract import content_sha256
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_fresh_patients as fp
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root
from test_phase6_fresh_patients import _members

CONFIG = "configs/phase6_rule_tuning/rule_tuning.yaml"


def test_config_fixes_the_grid(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = rt.load_config(repo_root() / CONFIG)
    assert cfg["fresh"]["patients"] == 1000 and cfg["af"]["rule_threshold"] == 0.8
    for old, new in (("0.90]", "0.95]"), ("[0.50,", "[0.40,")):
        bad = tmp_path / "bad.yaml"
        bad.write_text((repo_root() / CONFIG).read_text().replace(old, new).replace("configs/", str(repo_root() / "configs") + "/"))
        with pytest.raises(ValueError):
            rt.load_config(bad)


def test_fifty_settings_and_the_tie_order():
    s = rt.settings()
    assert len(s) == len(set(s)) == 50 and s[0] == (None, None)
    order = sorted(s, key=rt.preference)
    assert order[:4] == [(None, None), (0.8, None), (0.75, None), (0.85, None)]
    assert order.index((0.8, 0.95)) < order.index((0.8, 0.5)) < order.index((0.75, 0.95))
    assert rt.label((None, None)) == "ensemble alone" and rt.label((0.7, 0.9)) == "stage 0.70, veto 0.90"


def test_pooled_s_f1_matches_the_bootstrap_statistic():
    g = np.random.default_rng(0)
    y, pred = g.integers(0, 3, 500), g.integers(0, 4, 500)
    from ecgaug.eval import metrics as M

    cm = M.confusion_by_record(y, pred, g.integers(0, 7, 500))[1]
    assert rt.pooled_f1_s(pred, y) == pytest.approx(float(fp._stat("f1_S")(cm.sum(axis=0)[None])[0]))


def _arrays(n_patients: int = 20, beats: int = 50, s_ratio: float = 0.62, n_ratio: float = 1.0, seed: int = 0) -> dict:
    """N beats at ``n_ratio`` and S beats at ``s_ratio`` of the mean RR; ten patients' worth of p_af are high in half the N beats."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.r_[np.zeros(beats - 8, np.int8), np.full(5, 1, np.int8), np.full(3, 2, np.int8)], n_patients)
    rr_prev = np.where(y == 1, s_ratio, n_ratio).astype(np.float32)
    return {"X": rng.standard_normal((y.size, 4)).astype(np.float32), "y": y, "record": np.repeat(10_000 + np.arange(n_patients), beats).astype(np.int16),
            "r_idx": np.arange(y.size, dtype=np.int32), "rr_prev": rr_prev, "rr_next": np.ones(y.size, np.float32), "rr_ratio": np.ones(y.size, np.float32),
            "p_af": np.zeros(y.size, np.float32)}  # fmt: skip


def _cnn_misses_s(a: dict) -> dict[str, np.ndarray]:
    """Every condition calls every non-V beat N (so only the stage can find S beats)."""
    y = np.asarray(a["y"])
    p = np.where((y == 2)[:, None], np.eye(4)[2], np.eye(4)[0]) * 0.9 + 0.025
    return {c: p for c in se.CONDITIONS}


def test_choice_takes_the_best_pooled_s_f1_and_breaks_ties_by_the_registered_order(monkeypatch):
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    c = rt.choose(_cnn_misses_s(a), a)  # every t >= 0.65 finds every S beat and nothing else: Huang's 0.80 wins the tie
    assert {v["label"] for v in c["chosen"].values()} == {"stage 0.80"} and c["chosen"]["cvae"]["s_f1"] == pytest.approx(1.0)
    a = _arrays(n_ratio=0.78)  # now every N beat is premature at t >= 0.80: the nearest lower threshold wins
    c = rt.choose(_cnn_misses_s(a), a)
    assert c["chosen"]["none"]["label"] == "stage 0.75" and len(c["tuning_surface"]["none"]) == 50
    a["p_af"][:] = np.where(np.asarray(a["y"]) == 0, 0.97, 0.0)  # the veto removes the N beats' conversions at t 0.80+
    c = rt.choose(_cnn_misses_s(a), a)
    assert c["chosen"]["none"]["label"] == "stage 0.75"  # still tied with 0.80 + veto; no veto comes first
    perfect = {k: np.eye(4)[np.asarray(a["y"])] for k in se.CONDITIONS}
    assert rt.choose(perfect, a)["chosen"]["cvae"]["label"] == "ensemble alone"
    monkeypatch.setattr(af, "chosen_design", lambda: (0.9, "rule"))
    with pytest.raises(PreconditionError):
        rt.choose(perfect, a)


def _fake_p19(a: dict, probs: dict) -> None:
    folder = paths.results_for(fp.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    y, rr_prev, p_af = np.asarray(a["y"]), np.asarray(a["rr_prev"]), np.asarray(a["p_af"])
    points = {f"cnn/{c}": {"f1_S": {"point": rt.pooled_f1_s(rt.decide(p, rr_prev, p_af, rt.FIXED, "rule"), y)}} for c, p in probs.items()}
    (folder / "summary.json").write_text(json.dumps({"ensemble": {"with_af_veto": points}}))


def test_score_compares_with_p19s_setting_and_the_ensemble_alone(monkeypatch):
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays(n_ratio=0.9)
    probs = _cnn_misses_s(a)
    probs["cvae"] = np.eye(4)[np.asarray(a["y"])] * 0.9 + 0.025  # the cVAE alone is perfect
    choice = rt.choose(probs, a)
    assert choice["chosen"]["cvae"]["label"] == "ensemble alone" and choice["chosen"]["none"]["label"] == "stage 0.80"
    _fake_p19(a, probs)
    s = rt.score(choice, probs, a)
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, chosen vs P19's setting" for c in se.CONDITIONS}
    assert set(s["families"]["secondary"]) == {f"cnn: {c}, chosen vs ensemble alone" for c in se.CONDITIONS}
    same = s["families"]["secondary"]["cnn: cvae, chosen vs ensemble alone"]
    assert same["point"] == 0 and same["p_two_sided"] == 1 and len(s["versus_cvae"]) == 5
    assert s["conditions"]["chosen"]["cnn/none"]["f1_S"]["point"] == pytest.approx(1.0) and s["patients"] == 20
    assert all(abs(d) < 1e-12 for d in s["reproduction_difference"].values()) and len(s["evaluation_surface"]["cvae"]) == 50
    bad = json.loads((paths.results_for(fp.EXPERIMENT) / "summary.json").read_text())
    bad["ensemble"]["with_af_veto"]["cnn/none"]["f1_S"]["point"] -= 0.01
    (paths.results_for(fp.EXPERIMENT) / "summary.json").write_text(json.dumps(bad))
    with pytest.raises(PreconditionError):
        rt.score(choice, probs, a)


def test_build_needs_p19s_beats_and_load_checks_the_hash(tmp_path: Path, monkeypatch):
    evaluation, tuning = _arrays(seed=1), _arrays(seed=2)
    p19_patients = fp.sample_patients(fp.N_FRESH, fp.FRESH_SEED, fp.used_patients())
    folder = paths.results_for(fp.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.json").write_text(json.dumps({"patients": p19_patients, "content_sha256": content_sha256(evaluation)}))

    def fake_set(patients, spec, half, workers=None):
        arrays = evaluation if patients == p19_patients else tuning
        return arrays, {"segments_used": 20, "beats": {}, "patients_with": {}, "s_dominant_patients": 0, "content_sha256": content_sha256(arrays)}

    monkeypatch.setattr(fp, "build_set", fake_set)
    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 60.0}}}}}
    record = rt.build(cfg, None, tmp_path / "beats")
    assert record["evaluation_matches_p19"] and record["tuning"]["patients"] == fp.used_patients()
    assert np.array_equal(rt.load_set(tmp_path, "tuning")["y"], tuning["y"])  # found below the given folder
    np.savez(tmp_path / "beats" / "tuning.npz", **evaluation)
    with pytest.raises(PreconditionError):
        rt.load_set(tmp_path / "beats", "tuning")
    (folder / "data.json").write_text(json.dumps({"patients": p19_patients, "content_sha256": "0" * 64}))
    with pytest.raises(PreconditionError):
        rt.build(cfg, None, tmp_path / "other")


def test_the_choice_is_written_before_the_evaluation_patients_are_loaded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    a = _arrays()
    probs = {name: _cnn_misses_s(a) for name in rt.SETS}
    _fake_p19(a, probs["evaluation"])
    seen = []

    def fake_load(beats_dir, name):
        seen.append((name, (paths.results_for(rt.EXPERIMENT) / "choice.json").is_file()))
        return a

    monkeypatch.setattr(rt, "load_set", fake_load)
    rt._save_probs(probs, tmp_path)
    loaded = rt._load_probs(tmp_path)
    assert all(np.array_equal(loaded[n][c], probs[n][c]) for n in rt.SETS for c in se.CONDITIONS)
    choice, summary = rt.choose_and_score(loaded, tmp_path)
    assert seen == [("tuning", False), ("evaluation", True)] and summary["chosen"] == choice["chosen"]


def test_ds2_read_checks_p18s_values_and_compares_with_p18s_setting(monkeypatch):
    from ecgaug import final_eval

    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    m = _members()
    a = _arrays(n_ratio=0.9)
    y = np.asarray(a["y"]).copy()
    y[np.flatnonzero(y == 2)[::3]] = 3  # some F beats, so rare-class macro-F1 is defined
    test = {"y": y, "record": (100 + np.asarray(a["record"]) - 10_000).astype(np.int16), "rr_prev": a["rr_prev"], "r_idx": a["r_idx"]}
    monkeypatch.setattr(se, "ds2_inputs", lambda cfg, experiment: (test, a["p_af"]))
    probs = {c: np.eye(4)[np.where(y == 1, 0, y)] * 0.9 + 0.025 for c in se.CONDITIONS}  # every S beat called N
    monkeypatch.setattr(rt.pt, "_predict", lambda ids, t: [probs[ids[0].split("-")[0]]] * len(ids))
    choice = rt.choose(probs, {**a, "y": y})
    folder = paths.results_for(rt.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "choice.json").write_text(json.dumps(choice))
    p18 = {}
    for c in m:
        onehot = np.eye(4, dtype=np.float32)[rt.decide(probs[c], a["rr_prev"], a["p_af"], rt.FIXED, "rule")]
        p18[f"cnn/{c}"] = {"rare_macro_f1": {"point": se._rare(final_eval.per_record_cms([onehot], y, test["record"])[1])}}
    (paths.results_for(se.EXPERIMENT) / "summary.json").write_text(json.dumps({"ensemble": {"with_af_veto": p18}}))
    s = rt.test_report({})
    assert set(s["families"]["ds2"]) == {f"cnn: {c}, chosen vs P18's setting" for c in se.CONDITIONS}
    assert all(d == 0 for d in s["reproduction_difference"].values()) and len(s["records"]) == 20
    assert s["conditions"]["chosen"]["cnn/cvae"]["f1_S"]["point"] == pytest.approx(1.0) and set(s["without_202"]) == {"chosen", "p18_setting"}
    p18["cnn/none"]["rare_macro_f1"]["point"] += 0.01
    (paths.results_for(se.EXPERIMENT) / "summary.json").write_text(json.dumps({"ensemble": {"with_af_veto": p18}}))
    with pytest.raises(PreconditionError):
        rt.test_report({})


def test_figures_and_paper_text_come_from_the_result_files():
    from ecgaug import figures, paper_text

    assert "\\missing{P20: --part build" in paper_text.posthoc_rule_tuning() and paper_text._tuning_sentence() is None
    surface = {c: {rt.label(s): 0.3 for s in rt.settings()} for c in se.CONDITIONS}
    chosen = {c: {"rule_threshold": 0.7, "veto_threshold": None, "label": "stage 0.70", "s_f1": 0.3} for c in se.CONDITIONS}
    choice = {"chosen": chosen, "tuning_surface": surface, "patients": 388, "beats": {"N": 1, "S": 15_000, "V": 1, "F": 0}}
    e = {"point": 0.4, "ci_low": 0.3, "ci_high": 0.5}
    block = {f"cnn/{c}": {"f1_S": e, "f1_V": e, "f1_F": e, "rare_macro_f1": e} for c in se.CONDITIONS}
    comp = {"point": 0.02, "ci_low": -0.01, "ci_high": 0.05, "p_holm": 0.5, "reject_holm": False}
    summary = {"patients": 977, "chosen": chosen, "evaluation_surface": surface,
               "conditions": {"chosen": block, "p19_setting": block, "ensemble_alone": block},
               "families": {"confirmatory": {f"cnn: {c}, chosen vs P19's setting": comp for c in se.CONDITIONS},
                            "secondary": {f"cnn: {c}, chosen vs ensemble alone": comp for c in se.CONDITIONS}}}  # fmt: skip
    test = {"conditions": {"chosen": block, "p18_setting": block, "ensemble_alone": block},
            "families": {"ds2": {f"cnn: {c}, chosen vs P18's setting": comp for c in se.CONDITIONS}}}
    folder = paths.results_for(rt.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    for name, d in (("choice.json", choice), ("summary.json", summary)):
        (folder / name).write_text(json.dumps(d))
    assert "\\missing{P20: --part test" in paper_text.posthoc_rule_tuning()
    (folder / "test_summary.json").write_text(json.dumps(test))
    out = figures.rule_tuning_choice_figure(choice, summary) + figures.rule_tuning_results_figure(summary, test)
    assert len(out) == 2 and all(p.is_file() for p in out)
    text = paper_text.posthoc_rule_tuning()
    assert "\\missing" not in text and "cVAE: stage 0.70" in text and "On P19's 977 patients" in text and text.count("{") == text.count("}")
    assert "fig:p20-choice" in text and "fig:p20}" in text and "sec:rule-tuning" in paper_text._tuning_sentence(with_section=True)
    assert paper_text._best_test_score() >= 0.4
