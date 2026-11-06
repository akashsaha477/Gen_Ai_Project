"""P14: the Icentia11k sample, file layout and record numbers, one synthetic record end to end, the grid and the DS2 report."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.data.records import CLASS_INDEX
from ecgaug.data.representation import RepresentationSpec
from ecgaug.experiments import phase6_icentia as ic
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments.common import PreconditionError
from ecgaug.paths import repo_root

SPEC = RepresentationSpec(name="t", window_pre_s=0.25, window_post_s=0.25)


def test_the_sample_is_fixed_by_its_seed_distinct_sorted_and_in_range():
    a, b = ic.sample_patients(400, 0), ic.sample_patients(400, 0)
    assert a == b and a == sorted(a) and len(set(a)) == 400 and 0 <= a[0] and a[-1] < ic.N_PATIENTS
    assert ic.sample_patients(400, 1) != a


def test_files_follow_physionets_layout_and_records_get_numbers_clear_of_the_other_databases():
    assert ic.segment_url(9000, "hea").endswith("/p09/p09000/p09000_s00.hea")
    assert ic.segment_url(2, "atr").endswith("/p00/p00002/p00002_s00.atr")
    assert ic.record_number("p00002_s00") == 10_002 and ic.record_number("p10999_s00") == 20_999 <= np.iinfo(np.int16).max


def _write_record(folder: Path, name: str, fs: int = 250) -> tuple[np.ndarray, list[str]]:
    import wfdb

    t, symbols = [1.0], ["N"]
    for i in range(1, 700):
        kind = "S" if i % 10 == 5 else "V" if i % 17 == 3 else "Q" if i % 50 == 7 else "N"
        t.append(t[-1] + {"S": 0.5, "V": 0.55, "Q": 0.8, "N": 0.8}[kind])
        symbols.append(kind)
    samples = np.rint(np.array(t) * fs).astype(np.int64)
    n = int(samples[-1] + 2 * fs)
    rng = np.random.default_rng(0)
    signal = 0.02 * rng.standard_normal(n)
    grid = np.arange(n)
    for s, kind in zip(samples, symbols):
        width = 12 if kind == "V" else 4
        signal += (1.5 if kind == "V" else 1.0) * np.exp(-0.5 * ((grid - s) / width) ** 2)
    folder.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ecg"], p_signal=signal[:, None], fmt=["16"], write_dir=str(folder))
    ann_samples = np.concatenate([[samples[0] - 10], samples])
    ann_symbols = ["+"] + symbols
    wfdb.wrann(name, "atr", sample=ann_samples, symbol=ann_symbols, aux_note=["(N"] + [""] * len(symbols), write_dir=str(folder))
    return samples, symbols


def test_one_record_is_built_in_p13s_form_with_the_n_cap_applied_before_the_templates():
    from ecgaug.data import raw

    samples, symbols = _write_record(raw.external_dir(ic.DB), "p00002_s00")
    arrays, report = ic.build_record("p00002_s00", SPEC, cap=20, cap_seed=0, half_window_s=60.0)
    assert arrays is not None and "excluded" not in report
    y = arrays["y"]
    assert int((y == CLASS_INDEX["N"]).sum()) == 20  # the cap
    assert int((y == CLASS_INDEX["S"]).sum()) == report["usable"]["S"] > 0 and int((y == CLASS_INDEX["V"]).sum()) == report["usable"]["V"] > 0
    assert set(np.unique(arrays["record"]).tolist()) == {10_002} and arrays["X"].shape[1] == 180 and np.isfinite(arrays["X"]).all()
    t = np.asarray(samples, dtype=np.float64) / 250
    mean = float(np.mean(np.diff(t)))  # every QRS annotation, the Q beats included
    assert report["mean_rr_s"] == pytest.approx(mean)
    i = np.searchsorted(np.rint(np.asarray(samples) * 360 / 250).astype(np.int64), arrays["r_idx"])  # each kept beat's annotation
    assert np.allclose(arrays["rr_prev"], (t[i] - t[i - 1]) / mean, atol=1e-5) and np.allclose(arrays["rr_next"], (t[i + 1] - t[i]) / mean, atol=1e-5)
    s = y == CLASS_INDEX["S"]
    assert (arrays["rr_ratio"][s] < 0.8).all()  # v2: premature against the local median
    assert np.all(np.diff(arrays["r_idx"]) > 0)  # still in time order


def test_grid_keeps_four_of_p13s_conditions_at_their_strengths(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")  # read-only: the real configs and the committed selections
    cfg = ic.load_config(repo_root() / "configs/phase6_icentia/icentia.yaml")
    grid, base = ic.icentia_grid(cfg), rr.record_grid(cfg["rr"])
    assert list(grid["conditions"]) == list(ic.CONDITIONS) and "wgan_gp" not in grid["conditions"]
    assert all(grid["conditions"][c] == base["conditions"][c] for c in ic.CONDITIONS)
    assert grid["experiment"] == ic.EXPERIMENT and grid["policy"] == base["policy"] and grid["file"].endswith("icentia.yaml")


def test_config_rejects_more_patients_than_the_database_has(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: phase6_icentia\nrecord_rr_config: x\npatients: 20000\npatient_seed: 0\nn_cap_per_record: 600\ncap_seed: 0\n"
                   "validation_every: 5\nrule_threshold: 0.8\n")  # fmt: skip
    with pytest.raises(ValueError):
        ic.load_config(bad)


def test_test_report_compares_each_condition_with_p13_and_checks_p13_first(monkeypatch):
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

    monkeypatch.setattr(ic, "check_preconditions", lambda: None)
    monkeypatch.setattr(ic, "load_data", lambda *a, **k: SimpleNamespace(spec=None))
    monkeypatch.setattr(pt, "test_arrays", lambda spec, experiment: test)
    monkeypatch.setattr(v5, "record_ratios", lambda record, half, access: {})
    monkeypatch.setattr(v5, "substitute", lambda arrays, ratios: arrays)
    monkeypatch.setattr(pt, "mitbih_templates", lambda arrays, spec, half, access: None)
    monkeypatch.setattr(pt, "relative", lambda arrays, templates: arrays)
    monkeypatch.setattr(pt, "_predict", fake_predict)
    monkeypatch.setattr(rr, "mean_rr_by_record", lambda recs, access=None: {int(r): 0.5 for r in np.unique(recs)})

    def chosen(experiment: str, conditions: tuple) -> dict:
        return {"chosen": {"cnn": {c: {"strength": 0.1, "run_ids": [f"{experiment}-cnn.{c}-s{s}" for s in range(3)]} for c in conditions}}}

    for name in (ic.EXPERIMENT, rr.EXPERIMENT):
        paths.results_for(name).mkdir(parents=True, exist_ok=True)
        (paths.results_for(name) / "selection.json").write_text(json.dumps(chosen(name, rr.CONDITIONS)))

    def p13_summary(shift: float) -> None:
        body = {"conditions": {}}
        for c in rr.CONDITIONS:
            cms = final_eval.per_record_cms(fake_predict([f"{rr.EXPERIMENT}-cnn.{c}-s{s}" for s in range(3)], test), y, records)[1]
            body["conditions"][f"cnn/{c}"] = {"rare_macro_f1": {"point": float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1)))) + shift}}
        (paths.results_for(rr.EXPERIMENT) / "summary.json").write_text(json.dumps(body))

    cfg = {"rr": {"rg": {"mp": {"pt": {"half_window_s": 150.0}}}}, "rule_threshold": 0.8}
    p13_summary(0.01)
    with pytest.raises(PreconditionError):
        ic.test_report(cfg)
    p13_summary(0.0)
    s = ic.test_report(cfg)
    assert set(s["conditions"]) == set(s["before"]) == set(s["sveb_rule"]) == {f"cnn/{c}" for c in ic.CONDITIONS}
    assert set(s["families"]["confirmatory"]) == {f"cnn: {c}, with Icentia11k patients vs P13" for c in ic.CONDITIONS}
    assert set(s["families"]["secondary"]) == {f"cnn: cvae vs {c} (with Icentia11k patients)" for c in ("none", "random_oversampling", "class_weighted")}
    assert all("p_holm" in c for f in s["families"].values() for c in f.values())


def test_the_sampled_patients_are_built_split_every_fifth_and_their_files_hashed(monkeypatch):
    from ecgaug.data import raw

    def fake_download(patients: list[int], dest: Path, **_) -> list[str]:
        for p in patients:
            _write_record(dest, ic.segment_name(p))
        return [ic.segment_name(p) for p in patients]

    monkeypatch.setattr(ic, "download", fake_download)
    cfg = {"patients": 6, "patient_seed": 3, "n_cap_per_record": 10, "cap_seed": 0, "validation_every": 5,
           "rr": {"rg": {"mp": {"pt": {"half_window_s": 60.0}}}}}  # fmt: skip
    arrays, report = ic.build_icentia(cfg, SPEC)
    patients = ic.sample_patients(6, 3)
    assert report["patients"] == patients and sorted(np.unique(arrays["record"]).tolist()) == [ic.RECORD_OFFSET + p for p in patients]
    units = sorted(ic.RECORD_OFFSET + p for p in patients)
    assert report["validation_units"] == units[4::5]  # P11's rule: the 5th, 10th, ... in sorted order
    val = np.isin(arrays["record"], report["validation_units"])
    assert (arrays["split"][val] == "val").all() and (arrays["split"][~val] == "train").all()
    assert len(report["files_sha256"]) == 18 and all(len(h) == 64 for h in report["files_sha256"].values())
    assert set(arrays) == set(ic.mp.FIELDS) and (raw.external_dir(ic.DB) / f"{ic.segment_name(patients[0])}.dat").is_file()
