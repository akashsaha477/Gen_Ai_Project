"""P16: the irregularity features (regular, AF-like, sinus with isolated premature beats), the rhythm labels, the veto and
the DS2 report on fakes."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments.common import PreconditionError


def _times(rr_s: np.ndarray, fs: int = 360) -> np.ndarray:
    return np.rint(np.concatenate([[0.0], np.cumsum(rr_s)]) * fs).astype(np.int64)


def test_features_separate_af_from_regular_and_from_isolated_premature_beats():
    rng = np.random.default_rng(0)
    regular = af.irregularity_features(_times(np.full(300, 0.8) + 0.005 * rng.standard_normal(300)), 360)
    fib = af.irregularity_features(_times(rng.uniform(0.4, 1.1, 300)), 360)
    sinus = np.full(300, 0.8)
    sinus[::10] = 0.5  # an isolated premature beat every 10 beats ...
    sinus[1::10] = 1.1  # ... and its compensatory pause
    ectopic = af.irregularity_features(_times(sinus + 0.005 * rng.standard_normal(300)), 360)
    mid = slice(50, 250)
    assert np.nanmedian(fib[mid, 0]) > 5 * np.nanmedian(regular[mid, 0])
    assert np.nanmedian(ectopic[mid, 0]) < 0.3 * np.nanmedian(fib[mid, 0])  # the premature beats and their pauses are filtered out
    assert np.isnan(af.irregularity_features(_times(np.full(10, 0.8)), 360)).all()  # too few intervals


def test_rhythm_labels_follow_the_plus_annotations():
    samples = np.array([0, 100, 200, 300, 400])
    symbols = np.array(["+", "N", "+", "N", "N"])
    aux = ["(N", "", "(AFIB", "", ""]
    assert af.rhythm_at(samples, symbols, aux, np.array([50, 250, 450])).tolist() == ["(N", "(AFIB", "(AFIB"]


def test_veto_undoes_only_the_rules_own_conversions_inside_af():
    probs = np.eye(4)[[0, 0, 1, 0, 2]]
    ratio = np.array([0.7, 0.7, 0.7, 0.9, 0.5])
    p_af = np.array([0.99, 0.10, 0.99, 0.99, 0.99])
    out = af.veto_rule(probs, ratio, p_af, 0.8, 0.95).argmax(axis=1)
    assert out.tolist() == [0, 1, 1, 0, 2]  # rule's S in AF vetoed; rule's S outside AF kept; the cnn's own S and V untouched
    every = af.veto_rule(probs, ratio, p_af, 0.8, 0.95, form="all").argmax(axis=1)
    assert every.tolist() == [0, 1, 0, 0, 2]  # form "all": the cnn's own S in AF is undone too; V untouched
    with pytest.raises(ValueError):
        af.veto_rule(probs, ratio, p_af, 0.8, 0.95, form="some")


def test_validation_check_scores_each_form_and_keeps_the_narrower_one_unless_the_other_wins(tmp_path, monkeypatch):
    at = np.array([100, 200, 300, 400, 500, 600])
    feats = (at, np.array([np.nan, 0.7, 0.7, 0.9, 0.7, 0.7]), np.array([0.0, 0.99, 0.10, 0.99, 0.99, 0.99]))
    monkeypatch.setattr(af, "_record_beats", lambda record, detector: feats)
    monkeypatch.setattr(af, "_conditions", lambda: {"none": ["run-a"]})
    monkeypatch.setattr(af.runs, "find_run_dir", lambda run_id: tmp_path / run_id)
    (tmp_path / "run-a").mkdir()
    preds = {"probs": np.eye(4)[[0, 0, 0, 1, 1]], "y": np.array([0, 1, 0, 1, 0], np.int8), "record": np.full(5, 900, np.int16), "r_idx": at[1:]}
    np.savez(tmp_path / "run-a" / "preds_val.npz", **preds)
    v = af.validation_check({"rule_threshold": 0.8}, detector={}, af_t=0.95)
    f1 = {k: e["f1"][0] for k, e in v["conditions"]["none"].items()}
    assert f1 == pytest.approx({"cnn": 0.5, "rule": 2 / 3, "veto_rule": 0.8, "veto_all": 2 / 3})
    assert v["chosen_form"] == "rule" and v["records"] == [900]
    np.savez(tmp_path / "run-a" / "preds_val.npz", **{**preds, "r_idx": at[1:] + 1})  # beats off the annotations
    with pytest.raises(PreconditionError):
        af.validation_check({"rule_threshold": 0.8}, detector={}, af_t=0.95)


def test_af_probability_uses_the_stored_coefficients():
    detector = {"settings": {"half_window_beats": 32, "premature_ratio": 0.85, "local_median_beats": 10}, "coef": [1.0, 0, 0, 0, 0], "intercept": 2.0}
    rng = np.random.default_rng(1)
    p_reg = af.af_probability(detector, _times(np.full(200, 0.8) + 0.005 * rng.standard_normal(200)), 360)
    p_fib = af.af_probability(detector, _times(rng.uniform(0.4, 1.1, 200)), 360)
    assert p_reg.shape == (201,) and np.median(p_fib[50:150]) > np.median(p_reg[50:150])


def test_test_report_compares_with_and_without_the_veto_and_checks_p13s_rule_scores(monkeypatch):
    from types import SimpleNamespace

    from ecgaug import final_eval, paths
    from ecgaug.data import raw
    from ecgaug.eval import metrics as M
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments import phase6_rr_baseline as v5

    rng = np.random.default_rng(0)
    records = np.repeat([100, 103], 120)
    y = np.tile(np.repeat([0, 1, 2, 3], 30), 2).astype(np.int8)
    qrs = {r: _times(rng.uniform(0.5, 1.0, 130)) for r in (100, 103)}
    r_idx = np.concatenate([qrs[r][5:125] for r in (100, 103)])
    test = {"X": np.zeros((y.size, 4), np.float32), "y": y, "record": records, "r_idx": r_idx, "rr_prev": rng.uniform(0.6, 1.2, y.size).astype(np.float32),
            "rr_next": np.ones(y.size, np.float32), "rr_ratio": np.ones(y.size, np.float32)}  # fmt: skip

    def fake_predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
        out = []
        for run_id in run_ids:
            g = np.random.default_rng(sum(map(ord, run_id)))
            out.append(np.eye(4)[np.where(g.random(len(arrays["y"])) < 0.7, arrays["y"], g.integers(0, 4, len(arrays["y"])))])
        return out

    monkeypatch.setattr(af, "check_preconditions", lambda: None)
    monkeypatch.setattr(af, "load_data", lambda *a, **k: SimpleNamespace(spec=None))
    monkeypatch.setattr(pt, "test_arrays", lambda spec, experiment: test)
    monkeypatch.setattr(v5, "record_ratios", lambda record, half, access: {})
    monkeypatch.setattr(v5, "substitute", lambda arrays, ratios: arrays)
    monkeypatch.setattr(pt, "mitbih_templates", lambda arrays, spec, half, access: None)
    monkeypatch.setattr(pt, "relative", lambda arrays, templates: arrays)
    monkeypatch.setattr(pt, "_predict", fake_predict)
    monkeypatch.setattr(rr, "mean_rr_by_record", lambda recs, access=None: {int(r): 1.0 for r in np.unique(recs)})
    monkeypatch.setattr(raw, "read_annotations", lambda record, access=None: (qrs[int(record)], np.array(["N"] * qrs[int(record)].size)))

    folder = paths.results_for(af.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "af_detector.json").write_text(json.dumps({"settings": {"half_window_beats": 32, "premature_ratio": 0.85, "local_median_beats": 10},
                                                         "coef": [3.0, 0, 0, 0, 0], "intercept": 4.0}))  # fmt: skip
    (folder / "design_check.json").write_text(json.dumps({"chosen_threshold": 0.95}))
    with pytest.raises(PreconditionError):  # no form chosen yet
        af.test_report({"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "rule_threshold": 0.8})
    (folder / "design_check.json").write_text(json.dumps({"chosen_threshold": 0.95, "chosen_form": "rule"}))
    conditions = rr.CONDITIONS
    sel = {"chosen": {"cnn": {c: {"strength": 0.1, "run_ids": [f"p13-{c}-s{s}" for s in range(3)]} for c in conditions}}}
    paths.results_for(rr.EXPERIMENT).mkdir(parents=True, exist_ok=True)
    (paths.results_for(rr.EXPERIMENT) / "selection.json").write_text(json.dumps(sel))

    def p13_summary(shift: float) -> None:
        body = {"sveb_rule": {}}
        for c in conditions:
            probs = fake_predict(sel["chosen"]["cnn"][c]["run_ids"], test)
            cms = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], 0.8) for p in probs], y, records)[1]
            body["sveb_rule"][f"cnn/{c}"] = {"rare_macro_f1": {"point": float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1)))) + shift}}
        (paths.results_for(rr.EXPERIMENT) / "summary.json").write_text(json.dumps(body))

    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "rule_threshold": 0.8}
    p13_summary(0.01)
    with pytest.raises(PreconditionError):
        af.test_report(cfg)
    p13_summary(0.0)
    s = af.test_report(cfg)
    assert set(s["rule_alone"]) == set(s["with_af_veto"]) == {f"cnn/{c}" for c in conditions}
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, rule with AF veto vs rule alone" for c in conditions}
    assert s["af_threshold"] == 0.95 and s["veto_form"] == "rule" and set(s["af_share_flagged"]) == {"100", "103"}
    assert set(s["without_202"]) == {"rule_alone", "with_af_veto"} and set(s["without_202"]["with_af_veto"]) == set(s["with_af_veto"])
    from ecgaug.experiments import phase6_diffusion as dif

    dif_ids = [f"p15-diffusion-s{s}" for s in range(3)]
    paths.results_for(dif.EXPERIMENT).mkdir(parents=True, exist_ok=True)
    (paths.results_for(dif.EXPERIMENT) / "selection.json").write_text(json.dumps({"chosen": {"cnn": {"diffusion": {"strength": 0.25, "run_ids": dif_ids}}}}))
    with pytest.raises(PreconditionError):  # P15's own test has not run yet
        af.test_report(cfg)
    cms = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], 0.8) for p in fake_predict(dif_ids, test)], y, records)[1]
    point = float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1))))
    (paths.results_for(dif.EXPERIMENT) / "summary.json").write_text(json.dumps({"sveb_rule": {"cnn/diffusion": {"rare_macro_f1": {"point": point}}}}))
    s = af.test_report(cfg)
    assert "cnn/diffusion" in s["with_af_veto"] and len(s["families"]["confirmatory"]) == len(conditions) + 1
    assert abs(s["reproduction_difference"]["diffusion"]) < 1e-9
    bad = dict(qrs)
    bad[100] = qrs[100] + 1  # the test beats no longer sit on the record's QRS annotations
    monkeypatch.setattr(raw, "read_annotations", lambda record, access=None: (bad[int(record)], np.array(["N"] * bad[int(record)].size)))
    with pytest.raises(PreconditionError):
        af.test_report(cfg)
