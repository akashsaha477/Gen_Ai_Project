"""P18: seeds 3-9 cycle over the three frozen banks; the ensemble averages probabilities before the argmax, the SVEB stage
and the AF veto; the grid is P13's plus P15's diffusion condition; the members are ten runs per condition in seed order;
the DS2 report on fakes checks P13's, P15's and P16's scores first."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug import paths
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_diffusion as dif
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.grid import runner
from ecgaug.paths import repo_root

CONFIG = "configs/phase6_seed_ensemble/seed_ensemble.yaml"


def test_config_requires_seeds_3_to_9(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = se.load_config(repo_root() / CONFIG)
    assert cfg["new_seeds"] == [3, 4, 5, 6, 7, 8, 9] and cfg["af"]["rule_threshold"] == 0.8 and "rg" in cfg["rr"]
    bad = tmp_path / "bad.yaml"
    bad.write_text((repo_root() / CONFIG).read_text().replace("[3, 4, 5, 6, 7, 8, 9]", "[3, 4, 5]").replace("configs/", str(repo_root() / "configs") + "/"))
    with pytest.raises(ValueError):
        se.load_config(bad)


def test_new_seeds_cycle_over_the_three_banks():
    assert [se.bank_seed(s) for s in range(10)] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    with pytest.raises(ValueError):
        se.bank_seed(-1)


def test_ensemble_averages_probabilities_before_the_argmax():
    a = np.array([[0.5, 0.4, 0.1, 0.0], [0.9, 0.1, 0.0, 0.0]])
    b = np.array([[0.1, 0.45, 0.45, 0.0], [0.8, 0.0, 0.2, 0.0]])
    mean = se.ensemble([a, b])
    assert np.allclose(mean, (a + b) / 2) and mean.argmax(axis=1).tolist() == [1, 0]
    assert a.argmax(axis=1)[0] == 0  # one run alone calls the first beat N; the average calls it S
    with pytest.raises(ValueError):
        se.ensemble([])
    with pytest.raises(ValueError):
        se.ensemble([np.zeros(4)])


def test_decisions_put_the_stage_and_then_the_veto_on_the_average():
    probs = np.array([[0.6, 0.3, 0.1, 0.0], [0.6, 0.3, 0.1, 0.0], [0.2, 0.1, 0.7, 0.0]])
    rr_prev = np.array([0.7, 0.7, 0.7])  # premature against the recording's mean
    p_af = np.array([0.0, 0.99, 0.99])  # the second and third beats lie in atrial fibrillation
    d = se.decisions(probs, rr_prev, p_af, 0.8, 0.95, "rule")
    assert d["cnn"].argmax(axis=1).tolist() == [0, 0, 2]
    assert d["rule"].argmax(axis=1).tolist() == [1, 1, 2]  # N calls become S; the V call stays
    assert d["veto"].argmax(axis=1).tolist() == [1, 0, 2]  # the stage's S call inside AF is undone


def test_grid_is_p13s_plus_p15s_diffusion_condition_for_seeds_3_to_9(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = se.load_config(repo_root() / CONFIG)
    grid, p13 = se.ensemble_grid(cfg), rr.record_grid(cfg["rr"])
    assert grid["experiment"] == se.EXPERIMENT and grid["seeds"] == list(se.NEW_SEEDS) and tuple(grid["conditions"]) == se.CONDITIONS
    assert {k: v for k, v in grid.items() if k not in ("experiment", "conditions", "seeds", "file")} == {
        k: v for k, v in p13.items() if k not in ("experiment", "conditions", "seeds", "file")}
    assert {c: grid["conditions"][c] for c in rr.CONDITIONS} == p13["conditions"]
    assert grid["conditions"]["diffusion"] == {"bank": "phase6_diffusion/diffusion", "strengths": [se.strengths()["diffusion"]]}
    cells = runner.cells(grid)
    assert len(cells) == 42 and {c.seed for c in cells} == set(se.NEW_SEEDS)
    assert all(c.strength == se.strengths()[c.condition] for c in cells)


def _selections(seeds=(0, 1, 2)) -> dict[str, list[str]]:
    """Fake P13 and P15 selections in the temporary root; the run IDs of each condition."""
    out = {}
    for experiment, conditions in ((rr.EXPERIMENT, rr.CONDITIONS), (dif.EXPERIMENT, (dif.FAMILY,))):
        chosen = {c: {"strength": None if c == "none" else 0.25, "seeds": list(seeds), "run_ids": [f"{experiment}-{c}-s{s}" for s in seeds]} for c in conditions}
        paths.results_for(experiment).mkdir(parents=True, exist_ok=True)
        (paths.results_for(experiment) / "selection.json").write_text(json.dumps({"chosen": {"cnn": chosen}}))
        out.update({c: v["run_ids"] for c, v in chosen.items()})
    return out


def test_members_are_ten_runs_in_seed_order_and_refuse_gaps(monkeypatch):
    existing = _selections()
    monkeypatch.setattr(se, "ensemble_grid", lambda cfg: {"experiment": se.EXPERIMENT, "name": se.GRID_NAME})
    grid = {"experiment": se.EXPERIMENT, "name": se.GRID_NAME}
    for c in se.CONDITIONS:
        for s in se.NEW_SEEDS:
            if (c, s) != ("cvae", 7):
                runner.write_cell(grid, {"cell_sha": f"c{se.CONDITIONS.index(c)}s{s:02d}".ljust(16, "0"), "condition": c, "seed": s, "strength": None if c == "none" else 0.25,
                                         "run_id": f"{se.EXPERIMENT}-{c}-s{s}", "status": "done"})  # fmt: skip
    with pytest.raises(PreconditionError):
        se.members({})
    runner.write_cell(grid, {"cell_sha": "c3s07".ljust(16, "0"), "condition": "cvae", "seed": 7, "strength": 0.25, "run_id": f"{se.EXPERIMENT}-cvae-s7", "status": "done"})
    m = se.members({})
    assert tuple(m) == se.CONDITIONS and all(len(ids) == 10 for ids in m.values())
    assert m["cvae"][:3] == existing["cvae"] and m["cvae"][3:] == [f"{se.EXPERIMENT}-cvae-s{s}" for s in se.NEW_SEEDS]
    runner.write_cell(grid, {"cell_sha": "c0s03b".ljust(16, "0"), "condition": "none", "seed": 3, "strength": None, "run_id": "again", "status": "done"})
    with pytest.raises(PreconditionError):  # two runs for one condition and seed
        se.members({})


def test_validation_check_scores_single_runs_and_both_ensembles(monkeypatch):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2, 3], 25).astype(np.int8)
    beats = (y, np.repeat([115, 118], 50).astype(np.int64), np.arange(100, dtype=np.int64))
    ids = {c: [f"{c}-s{s}" for s in range(10)] for c in se.CONDITIONS}
    monkeypatch.setattr(se, "members", lambda cfg: ids)
    monkeypatch.setattr(se, "_validation_predictions", lambda run_ids: ([rng.dirichlet(np.ones(4), size=100) for _ in run_ids], beats))
    monkeypatch.setattr(se, "_validation_features", lambda rec, r_idx, detector, cache: (np.full(100, 0.9), np.zeros(100)))
    monkeypatch.setattr(af, "load_detector", lambda: {})
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    out = se.validation_check({"af": {"rule_threshold": 0.8}})
    assert out["members"] == ids and set(out["validation"]) == set(se.CONDITIONS)
    entry = out["validation"]["cvae"]["veto"]["S"]
    assert len(entry["single"]) == 10 and 0.0 <= entry["three"] <= 1.0 and 0.0 <= entry["ten"] <= 1.0
    assert set(out["mean_s_f1"]) == set(se.KINDS) and se.load_members()["members"] == ids


def test_test_report_checks_earlier_scores_then_compares_the_ensembles(monkeypatch):
    from types import SimpleNamespace

    from ecgaug import final_eval
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments import phase6_rr_baseline as v5

    rng = np.random.default_rng(0)
    records = np.repeat([100, 103, 202, 213], 40)
    y = np.tile(np.repeat([0, 1, 2, 3], 10), 4).astype(np.int8)
    test = {"X": np.zeros((len(y), 4), np.float32), "y": y, "record": records, "r_idx": np.arange(len(y)), "rr_prev": rng.uniform(0.5, 1.2, len(y)).astype(np.float32),
            "rr_next": np.full(len(y), 0.9, np.float32), "rr_ratio": np.ones(len(y), np.float32)}  # fmt: skip
    p_af = np.where(records == 103, 0.99, 0.01)

    def fake_predict(run_ids: list[str], arrays: dict) -> list[np.ndarray]:
        out = []
        for run_id in run_ids:
            g = np.random.default_rng(sum(map(ord, run_id)))
            out.append(np.eye(4)[np.where(g.random(len(arrays["y"])) < 0.6, arrays["y"], g.integers(0, 4, len(arrays["y"])))])
        return out

    for name, fn in (("check_preconditions", lambda: None), ("load_data", lambda *a, **k: SimpleNamespace(spec=None))):
        monkeypatch.setattr(se, name, fn)
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

    existing = _selections()
    ids = {c: [*existing[c], *(f"{se.EXPERIMENT}-{c}-s{s}" for s in se.NEW_SEEDS)] for c in se.CONDITIONS}
    paths.results_for(se.EXPERIMENT).mkdir(parents=True, exist_ok=True)
    (paths.results_for(se.EXPERIMENT) / "members.json").write_text(json.dumps({"members": ids}))

    def point(rows: list[np.ndarray]) -> dict:
        return {"rare_macro_f1": {"point": se._rare(final_eval.per_record_cms(rows, y, records)[1])}}

    def write_summaries(shift: float) -> None:
        plain = {c: fake_predict(existing[c], test) for c in se.CONDITIONS}
        (paths.results_for(rr.EXPERIMENT) / "summary.json").write_text(json.dumps({"conditions": {f"cnn/{c}": point(plain[c]) for c in rr.CONDITIONS}}))
        (paths.results_for(dif.EXPERIMENT) / "summary.json").write_text(json.dumps({"conditions": {"cnn/diffusion": point(plain["diffusion"])}}))
        rule = {f"cnn/{c}": point([rr.sveb_rule(p, test["rr_prev"], 0.8) for p in plain[c]]) for c in se.CONDITIONS}
        veto = {f"cnn/{c}": point([af.veto_rule(p, test["rr_prev"], p_af, 0.8, 0.95, "rule") for p in plain[c]]) for c in se.CONDITIONS}
        veto["cnn/cvae"]["rare_macro_f1"]["point"] += shift
        paths.results_for(af.EXPERIMENT).mkdir(parents=True, exist_ok=True)
        (paths.results_for(af.EXPERIMENT) / "summary.json").write_text(json.dumps({"rule_alone": rule, "with_af_veto": veto}))

    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "af": {"rule_threshold": 0.8}}
    write_summaries(0.01)
    with pytest.raises(PreconditionError):
        se.test_report(cfg)
    write_summaries(0.0)
    s = se.test_report(cfg)
    keys = {f"cnn/{c}" for c in se.CONDITIONS}
    assert set(s["ensemble"]) == {"with_af_veto", "rule_alone", "cnn_alone"} and all(set(v) == keys for v in s["ensemble"].values())
    assert set(s["p16"]["with_af_veto"]) == set(s["single_runs"]["with_af_veto"]) == set(s["three_run_ensemble"]["with_af_veto"]) == keys
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, ten-run ensemble vs single runs (P16)" for c in se.CONDITIONS}
    assert set(s["families"]["secondary"]) == {f"cnn: cvae vs {c}, ten-run ensembles" for c in se.CONDITIONS if c != "cvae"}
    assert all(len(v) == 10 for v in s["per_run_points"].values()) and set(s["without_202"]["ensemble"]) == set(s["without_202"]["p16"]) == keys
    assert all(abs(d) < 1e-9 for d in s["reproduction_difference"].values()) and s["seeds"] == list(range(10))
    p16 = s["p16"]["with_af_veto"]["cnn/cvae"]["rare_macro_f1"]["point"]
    assert p16 == pytest.approx(float(np.mean(s["per_run_points"]["cnn/cvae"][:3])))
