"""RQ3: quality metrics against the downstream gain, from the phase 3 checkpoints (ecgaug.eval.rq3)."""

import csv
import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from ecgaug import paths, paper
from ecgaug.eval import rq3

STAMP = "abcdef0-202610200000"


def _run(experiment: str, variant: str, seed: int, rows: list[dict]) -> str:
    run_id = f"{experiment}-{variant}-s{seed}-{STAMP}"
    folder = paths.runs_dir() / experiment / variant / "pilot" / run_id
    folder.mkdir(parents=True)
    with (folder / "checkpoints.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_id


def _rows(values: list[float], dtw: list[float], tstr: list[float]) -> list[dict]:
    out = []
    for k, (v, d, t) in enumerate(zip(values, dtw, tstr), start=1):
        row = {"checkpoint": k, "step": 100 * k, "pilot_selection_metric": "s_f1", "pilot_selection_value": v,
               "quality_dtw": d, "quality_tstr": t}  # fmt: skip
        for name in rq3.DIAGNOSTICS:
            row.update({f"{name}_{c}": 0.1 * k for c in rq3.GENERATED})
        out.append(row)
    return out


def _result(experiment: str, family: str, variant: str, run_ids: list[str], provisional: bool = False) -> None:
    folder = paths.results_for(experiment)
    folder.mkdir(parents=True, exist_ok=True)
    result = {"experiment": experiment, "family": family, "variant": variant, "mode": "pilot", "provisional": provisional,
              "settings": {"default": {"run_ids": run_ids}}}  # fmt: skip
    (folder / f"{variant}.json").write_text(json.dumps(result))


def _reference(value: float) -> None:
    run_id = f"phase2_baseline_classifiers-A.lr1e-03-s0-{STAMP}"
    folder = paths.runs_dir() / "phase2_baseline_classifiers" / "A_shallow_cnn" / run_id
    folder.mkdir(parents=True)
    (folder / "metrics_val.json").write_text(json.dumps({"selection_value": value}))
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase2_gate": {"cnn": {"run_ids": [run_id]}}}))


@pytest.fixture
def checkpoints():
    rng = np.random.default_rng(0)
    cvae = [_run("phase3_cvae", "A", s, _rows(list(rng.random(4)), list(rng.random(4)), list(rng.random(4)))) for s in (0, 1)]
    wgan = [_run("phase3_wgan_gp", "B", s, _rows(list(rng.random(4)), list(rng.random(4)), list(rng.random(4)))) for s in (0, 1)]
    _result("phase3_cvae", "cvae", "A", cvae)
    _result("phase3_wgan_gp", "wgan_gp", "B", wgan)
    return cvae + wgan


def test_rho_is_spearman_over_all_checkpoints_and_the_reference_only_shifts_the_gain(checkpoints):
    without = rq3.analyse(n_boot=50)
    pooled = [r for r in without["rows"]]
    expected = spearmanr([r["quality_dtw"] for r in pooled], [r["pilot_selection_value"] for r in pooled]).statistic
    assert without["metrics"]["quality_dtw"]["rho"] == pytest.approx(expected)
    assert without["n_runs"] == 4 and without["n_checkpoints"] == 16 and without["reference"] is None
    _reference(0.25)
    with_ref = rq3.analyse(n_boot=50)
    assert with_ref["metrics"]["quality_dtw"]["rho"] == pytest.approx(expected)
    assert all(r["gain"] == pytest.approx(r["pilot_selection_value"] - 0.25) for r in with_ref["rows"])


def test_tstr_is_analysed_apart_and_diagnostics_are_class_means(checkpoints):
    summary = rq3.analyse(n_boot=20)
    assert summary["metrics"]["quality_tstr"]["group"] == "tstr"
    assert summary["metrics"]["quality_dtw"]["group"] == "fidelity and diversity"
    assert {"diversity_ratio_mean", "memorization_fraction_mean", "c2st_accuracy_mean"} <= set(summary["metrics"])
    assert summary["rows"][1]["diversity_ratio_mean"] == pytest.approx(0.2)


def test_the_interval_resamples_runs_not_checkpoints():
    ids = [_run("phase3_cvae", "A", s, _rows([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4], [4, 3, 2, 1])) for s in range(3)]
    _result("phase3_cvae", "cvae", "A", ids)
    m = rq3.analyse(n_boot=200)["metrics"]
    assert m["quality_dtw"]["rho"] == pytest.approx(1.0) and m["quality_dtw"]["n_runs"] == 3
    assert m["quality_dtw"]["ci_low"] == pytest.approx(1.0) and m["quality_dtw"]["excludes_zero"] is True
    assert m["quality_tstr"]["ci_high"] == pytest.approx(-1.0)


def test_provisional_results_are_left_out_unless_previewed():
    ids = [_run("phase3_cvae", "A", s, _rows([0.1, 0.2], [1, 2], [2, 1])) for s in (0, 1)]
    _result("phase3_cvae", "cvae", "A", ids, provisional=True)
    with pytest.raises(FileNotFoundError):
        rq3.analyse(n_boot=10)
    summary = rq3.analyse(preview=True, n_boot=10)
    assert summary["provisional"] is True and summary["n_runs"] == 2


def test_outputs_table_and_figure(checkpoints):
    summary = rq3.analyse(n_boot=20)
    rq3.write_outputs(summary)
    assert (paths.results_for("phase4_quality_vs_gain") / "checkpoints.csv").is_file()
    assert (paths.figures_for("phase4_quality_vs_gain") / "quality_vs_gain.png").is_file()
    tex, _ = paper.quality_vs_gain_table()
    assert "\\missing" not in tex and "dtw" in tex and "tstr" in tex
