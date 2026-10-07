"""Run IDs, run directories, the runs index and ingest."""

import csv
import datetime as dt
import json
import shutil

import numpy as np
import pytest
import torch

from ecgaug import paths, runs

NOW = dt.datetime(2026, 10, 8, 15, 30)
METRICS = {"rare_macro_f1": 0.41, "selection_metric": "s_f1", "selection_value": 0.52}


def _write(run_id, root=None, metrics=METRICS):
    history = [{"epoch": 1, "train_loss": 0.5}, {"epoch": 2, "train_loss": 0.4}]
    preds = {"probs": np.full((3, 4), 0.25), "y": np.array([0, 1, 2])}
    env = {"python": "3.11", "wall_clock_s": 12.3}
    return runs.write_run(run_id, {"seed": 0}, env, history, metrics, preds, {"w": torch.zeros(2)})


def test_run_id_format_round_trips():
    run_id = runs.make_run_id(1, 1, "A", 0, tag="lr1e-03", now=NOW)
    assert runs.RUN_ID_RE.match(run_id) and run_id.endswith("-202610081530")
    parts = runs.parse_run_id(run_id)
    assert (parts["phase"], parts["module"], parts["variant"], parts["tag"], parts["seed"]) == (1, 1, "A", "lr1e-03", 0)
    assert parts["sha"] == runs.git_sha(short=True) and len(parts["sha"]) == 7
    plain = runs.parse_run_id(runs.make_run_id(3, 5, "C", 2, now=NOW))
    assert plain["tag"] is None and plain["variant"] == "C" and plain["seed"] == 2
    with pytest.raises(ValueError):
        runs.parse_run_id("not-a-run-id")


def test_run_directory_has_the_logged_files_and_is_never_overwritten():
    run_id = runs.make_run_id(1, 1, "A", 0, now=NOW)
    run_dir = _write(run_id)
    names = {p.name for p in run_dir.iterdir()}
    assert names == {"config.yaml", "env.json", "train_log.csv", "metrics_val.json", "preds_val.npz", "checkpoints"}
    env = json.loads((run_dir / "env.json").read_text())
    assert len(env["checkpoint"]["sha256"]) == 64 and (run_dir / env["checkpoint"]["path"]).is_file()
    with (run_dir / "train_log.csv").open() as fh:
        assert [row["epoch"] for row in csv.DictReader(fh)] == ["1", "2"]
    with pytest.raises(FileExistsError):
        _write(run_id)


def test_runs_index_upserts_by_run_id():
    first = runs.make_run_id(1, 1, "A", 0, tag="lr1e-03", now=NOW)
    second = runs.make_run_id(1, 1, "B", 1, now=NOW)
    runs.upsert_runs_index(runs.index_row(first, METRICS, 10.0))
    runs.upsert_runs_index(runs.index_row(second, METRICS, 20.0))
    index = runs.upsert_runs_index(runs.index_row(first, {**METRICS, "rare_macro_f1": 0.9}, 11.0))
    with index.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["run_id"] for r in rows] == sorted([first, second])
    by_id = {r["run_id"]: r for r in rows}
    assert by_id[first]["rare_macro_f1"] == "0.9" and by_id[first]["tag"] == "lr1e-03" and by_id[first]["seed"] == "0"
    assert by_id[second]["variant"] == "B" and by_id[second]["status"] == "done"
    assert list(rows[0]) == runs.RUNS_INDEX_COLUMNS


def test_ingest_from_a_zip_then_skip_and_conflict(tmp_path, monkeypatch):
    run_id = runs.make_run_id(2, 3, "B", 1, now=NOW)
    _write(run_id)
    archive = shutil.make_archive(str(tmp_path / "kaggle_output"), "zip", paths.runs_dir())
    shutil.rmtree(paths.runs_dir())

    report = runs.ingest([archive])
    assert report["ingested"] == [run_id] and not report["conflicts"]
    assert (paths.runs_dir() / run_id / "checkpoints" / "best.pt").is_file()
    with (paths.results_dir() / "runs_index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["run_id"] == run_id and rows[0]["wall_clock_s"] == "12.3" and rows[0]["selection_value"] == "0.52"

    assert runs.ingest([archive])["skipped_identical"] == [run_id]
    (paths.runs_dir() / run_id / "metrics_val.json").write_text(json.dumps({"rare_macro_f1": 0.99}))
    assert runs.ingest([archive])["conflicts"] == [run_id]
    assert json.loads((paths.runs_dir() / run_id / "metrics_val.json").read_text()) == {"rare_macro_f1": 0.99}


def test_ingest_refuses_runs_that_carry_test_predictions(tmp_path):
    run_id = runs.make_run_id(4, 7, "A", 0, now=NOW)
    source = tmp_path / "incoming"
    shutil.copytree(_write(run_id), source / run_id)
    shutil.rmtree(paths.runs_dir())
    np.savez(source / run_id / "preds_test.npz", probs=np.zeros((1, 4)))
    report = runs.ingest([source])
    assert report["refused"] == [run_id] and not report["ingested"]
    assert not (paths.runs_dir() / run_id).exists()


def test_ingest_reports_incomplete_runs_and_bad_sources(tmp_path):
    run_id = runs.make_run_id(1, 1, "D", 2, now=NOW)
    source = tmp_path / "incoming"
    shutil.copytree(_write(run_id), source / run_id)
    shutil.rmtree(paths.runs_dir())
    (source / run_id / "metrics_val.json").unlink()
    assert runs.ingest([source])["incomplete"] == [run_id]
    with pytest.raises(FileNotFoundError):
        runs.ingest([tmp_path / "does-not-exist.zip"])


def test_ingest_brings_back_variant_result_files_without_overwriting(tmp_path):
    run_id = runs.make_run_id(1, 1, "A", 0, now=NOW)
    bundle = tmp_path / "bundle"
    shutil.copytree(_write(run_id), bundle / "runs" / run_id)
    shutil.rmtree(paths.runs_dir())
    result = bundle / "results" / "validation" / "m1"
    result.mkdir(parents=True)
    (result / "A.json").write_text('{"variant": "A"}')
    (result / "A_notes.md").write_text("notes")

    report = runs.ingest([bundle])
    assert report["ingested"] == [run_id] and sorted(report["result_files"]) == ["m1/A.json", "m1/A_notes.md"]
    assert (paths.results_dir() / "validation" / "m1" / "A.json").read_text() == '{"variant": "A"}'

    (result / "A.json").write_text('{"variant": "A", "edited": true}')
    again = runs.ingest([bundle])
    assert again["result_conflicts"] == ["m1/A.json"] and not again["result_files"]
    assert (paths.results_dir() / "validation" / "m1" / "A.json").read_text() == '{"variant": "A"}'


def test_git_dirty_ignores_run_outputs_but_not_code(monkeypatch):
    def status(lines):
        monkeypatch.setattr(runs, "_git", lambda *args: "\n".join(lines))
        return runs.git_dirty()

    assert status([]) is False
    assert status(["?? runs/p1-m1-A-s0-abcdef0-202610081530/config.yaml", " M results/runs_index.csv"]) is False
    assert status(["?? runs/x/config.yaml", " M src/ecgaug/runs.py"]) is True
    assert status(["?? configs/m1/E.yaml"]) is True
    assert status(["R  results/a.json -> docs/a.json"]) is True
    monkeypatch.setattr(runs, "_git", lambda *args: None)
    assert runs.git_dirty() is True  # no git: cannot vouch for the tree
