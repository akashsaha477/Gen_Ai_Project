"""P19: the config fixes 1,000 patients and seed 1; the draw avoids P14's patients; a segment is built with every beat and
its AF probability; scoring streams P18's frozen runs, averages before the decision and compares on S F1."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug import paths
from ecgaug.data.records import CLASS_INDEX
from ecgaug.data.representation import RepresentationSpec
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_fresh_patients as fp
from ecgaug.experiments import phase6_icentia as ic
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root
from test_phase6_icentia import _write_record

CONFIG = "configs/phase6_fresh_patients/fresh_patients.yaml"
SPEC = RepresentationSpec(name="t", window_pre_s=0.25, window_post_s=0.25)


def test_config_fixes_the_number_of_patients_and_the_seed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    cfg = fp.load_config(repo_root() / CONFIG)
    assert cfg["patients"] == 1000 and cfg["patient_seed"] == 1 and cfg["af"]["rule_threshold"] == 0.8
    for old, new in (("patients: 1000", "patients: 500"), ("patient_seed: 1", "patient_seed: 2")):
        bad = tmp_path / "bad.yaml"
        bad.write_text((repo_root() / CONFIG).read_text().replace(old, new).replace("configs/", str(repo_root() / "configs") + "/"))
        with pytest.raises(ValueError):
            fp.load_config(bad)


def test_the_draw_avoids_p14s_patients_and_is_fixed_by_its_seed():
    used = fp.used_patients()
    assert used == ic.sample_patients(400, 0)
    a = fp.sample_patients(1000, 1, used)
    assert a == fp.sample_patients(1000, 1, used) and a == sorted(a) and len(set(a)) == 1000
    assert not set(a) & set(used) and 0 <= a[0] and a[-1] < ic.N_PATIENTS
    with pytest.raises(ValueError):
        fp.sample_patients(10, 0, list(range(ic.N_PATIENTS - 5)))


def test_a_segment_keeps_every_beat_and_gets_its_af_probability():
    from ecgaug.data import raw

    _write_record(raw.external_dir(ic.DB), "p00002_s00")
    detector = {"settings": {"half_window_beats": 32, "premature_ratio": 0.85, "local_median_beats": 10},
                "coef": [0.0] * len(af.FEATURES), "intercept": 0.0}  # fmt: skip
    arrays, report = fp.build_one("p00002_s00", SPEC, 60.0, detector)
    assert arrays is not None and "excluded" not in report and set(arrays) == set(fp.FIELDS)
    y = arrays["y"]
    assert int((y == CLASS_INDEX["N"]).sum()) == report["usable"]["N"] > 20  # no cap, unlike P14's training build
    assert ((arrays["p_af"] == 0.0) | (arrays["p_af"] == 0.5)).all() and (arrays["p_af"] == 0.5).any()  # zero weights: 0.5 where a window exists
    assert arrays["X"].shape == (y.size, 180) and np.isfinite(arrays["X"]).all()


def test_build_reports_failures_and_writes_counts_without_any_model(monkeypatch):
    def fake_download(patients: list[int], dest: Path, **_) -> list[str]:
        for p in patients[:3]:
            _write_record(dest, ic.segment_name(p))
        return [ic.segment_name(p) for p in patients[:3]]

    monkeypatch.setattr(ic, "download", fake_download)
    monkeypatch.setattr(fp, "N_FRESH", 5)
    monkeypatch.setattr(af, "load_detector", lambda: {"settings": {"half_window_beats": 32, "premature_ratio": 0.85, "local_median_beats": 10},
                                                       "coef": [0.0] * len(af.FEATURES), "intercept": 0.0})  # fmt: skip
    arrays, record = fp.build({"rr": {"rg": {"mp": {"pt": {"half_window_s": 60.0}}}}}, SPEC, workers=1)
    assert record["segments_used"] == 3 and len(record["excluded"]) == 2 and record["patients_with"]["S"] == 3
    assert len(np.unique(arrays["record"])) == 3 and json.loads((paths.results_for(fp.EXPERIMENT) / "data.json").read_text())["beats"]["S"] > 0


def _members() -> dict[str, list[str]]:
    for experiment, conditions in ((se.rr.EXPERIMENT, se.rr.CONDITIONS), (se.dif.EXPERIMENT, (se.dif.FAMILY,))):
        chosen = {c: {"strength": None if c == "none" else 0.25, "seeds": [0, 1, 2], "run_ids": [f"{c}-s{s}" for s in range(3)]} for c in conditions}
        paths.results_for(experiment).mkdir(parents=True, exist_ok=True)
        (paths.results_for(experiment) / "selection.json").write_text(json.dumps({"chosen": {"cnn": chosen}}))
    m = {c: [f"{c}-s{s}" for s in range(10)] for c in se.CONDITIONS}
    paths.results_for(se.EXPERIMENT).mkdir(parents=True, exist_ok=True)
    (paths.results_for(se.EXPERIMENT) / "members.json").write_text(json.dumps({"members": m}))
    return m


def _arrays(n_patients: int = 30, beats: int = 60) -> dict:
    rng = np.random.default_rng(0)
    y = np.tile(np.r_[np.zeros(beats - 10, np.int8), np.full(6, 1, np.int8), np.full(4, 2, np.int8)], n_patients)
    return {"X": np.zeros((y.size, 4), np.float32), "y": y, "record": np.repeat(10_000 + np.arange(n_patients), beats).astype(np.int16),
            "r_idx": np.arange(y.size), "rr_prev": np.where(y == 1, 0.6, 1.0).astype(np.float32) + 0.05 * rng.standard_normal(y.size).astype(np.float32),
            "rr_next": np.ones(y.size, np.float32), "rr_ratio": np.ones(y.size, np.float32), "p_af": np.zeros(y.size, np.float32)}  # fmt: skip


def test_score_streams_frozen_runs_and_compares_on_s_f1(monkeypatch):
    m = _members()
    arrays = _arrays()
    monkeypatch.setattr(af, "chosen_design", lambda: (0.95, "rule"))
    seen = []

    def fake_predict(run_id: str, a: dict) -> np.ndarray:
        seen.append(run_id)
        g = np.random.default_rng(sum(map(ord, run_id)))
        y = a["y"]
        return np.eye(4)[np.where(g.random(y.size) < 0.7, y, g.integers(0, 3, y.size))] * 0.8 + 0.05

    s = fp.score({"af": {"rule_threshold": 0.8}}, arrays, fake_predict)
    assert seen == [i for ids in m.values() for i in ids]  # every frozen run once, in order
    keys = {f"cnn/{c}" for c in se.CONDITIONS}
    assert set(s["ensemble"]["with_af_veto"]) == keys and set(s["ensemble"]["with_af_veto"]["cnn/cvae"]) == set(fp.STATS)
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, ensemble vs P16" for c in se.CONDITIONS}
    assert set(s["families"]["secondary"]) == {f"cnn: cvae vs {c}, ensembles" for c in se.CONDITIONS if c != "cvae"}
    assert all("p_holm" in v for f in s["families"].values() for v in f.values()) and s["patients"] == 30
    assert all(len(v) == 10 for v in s["per_run_points"].values()) and s["beats"]["F"] == 0
    p16 = s["p16"]["with_af_veto"]["cnn/none"]["f1_S"]["point"]
    assert p16 == pytest.approx(float(np.mean(s["per_run_points"]["cnn/none"][:3])))


def test_members_must_be_p18s(monkeypatch):
    m = _members()
    m["cvae"] = m["cvae"][:9]
    (paths.results_for(se.EXPERIMENT) / "members.json").write_text(json.dumps({"members": m}))
    with pytest.raises(PreconditionError):
        fp.frozen_members()


def _fake_summary() -> dict:
    keys = [f"cnn/{c}" for c in se.CONDITIONS]
    entry = {s: {"point": 0.5, "ci_low": 0.4, "ci_high": 0.6} for s in fp.STATS}
    comparison = {"point": 0.02, "ci_low": -0.01, "ci_high": 0.05, "p_holm": 0.5, "reject_holm": False}
    return {"patients": 990, "beats": {"N": 3_700_000, "S": 40_000, "V": 19_000, "F": 0},
            "data": {"patients_with": {"S": 900, "V": 800}, "s_dominant_patients": 2, "largest_s_share": 0.08},
            "ensemble": {k: {c: entry for c in keys} for k in ("with_af_veto", "rule_alone", "cnn_alone")},
            "families": {"confirmatory": {f"cnn: {c}, ensemble vs P16": comparison for c in se.CONDITIONS},
                         "secondary": {f"cnn: cvae vs {c}, ensembles": comparison for c in se.CONDITIONS if c != "cvae"}},
            "stages": {f"cnn/{c}: {k} vs cnn alone": comparison for c in se.CONDITIONS for k in ("rule", "veto")}}  # fmt: skip


def test_figure_and_paper_text_come_from_the_summary():
    from ecgaug import figures, paper_text

    assert "\\missing{P19: --part run on Kaggle}" in paper_text.posthoc_fresh_patients() and paper_text._fresh_sentence() is None
    s = _fake_summary()
    folder = paths.results_for(fp.EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(s))
    out = figures.fresh_patients_figure(s, None)
    assert len(out) == 1 and out[0].is_file()
    text = paper_text.posthoc_fresh_patients()
    assert "\\missing" not in text and "990 segments were usable" in text and "2 with more S than N beats" in text
    assert text.count("{") == text.count("}") and "Figure~\\ref{fig:p19}" in text
    assert "On 990 Icentia11k patients" in paper_text._fresh_sentence() and "sec:fresh-patients" in paper_text._fresh_sentence(with_section=True)
