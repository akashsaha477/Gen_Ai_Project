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
    run_id = runs.make_run_id("phase1_input_representation", "A", 0, tag="lr1e-03", now=NOW)
    assert runs.RUN_ID_RE.match(run_id) and run_id.endswith("-202610081530")
    parts = runs.parse_run_id(run_id)
    assert (parts["phase"], parts["experiment"], parts["variant"], parts["tag"], parts["seed"]) == (1, "phase1_input_representation", "A", "lr1e-03", 0)
    assert parts["sha"] == runs.git_sha(short=True) and len(parts["sha"]) == 7
    plain = runs.parse_run_id(runs.make_run_id("phase3_cvae", "C", 2, now=NOW))
    assert plain["tag"] is None and plain["variant"] == "C" and plain["seed"] == 2
    with pytest.raises(ValueError):
        runs.parse_run_id("not-a-run-id")


def test_run_directory_has_the_logged_files_and_is_never_overwritten():
    run_id = runs.make_run_id("phase1_input_representation", "A", 0, now=NOW)
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
    first = runs.make_run_id("phase1_input_representation", "A", 0, tag="lr1e-03", now=NOW)
    second = runs.make_run_id("phase1_input_representation", "B", 1, now=NOW)
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
    run_id = runs.make_run_id("phase2_ptbxl_pretraining", "B", 1, now=NOW)
    _write(run_id)
    archive = shutil.make_archive(str(tmp_path / "kaggle_output"), "zip", paths.runs_dir())
    shutil.rmtree(paths.runs_dir())

    report = runs.ingest([archive])
    assert report["ingested"] == [run_id] and not report["conflicts"]
    assert (runs.find_run_dir(run_id) / "checkpoints" / "best.pt").is_file()
    with (paths.results_dir() / "runs_index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["run_id"] == run_id and rows[0]["wall_clock_s"] == "12.3" and rows[0]["selection_value"] == "0.52"

    assert runs.ingest([archive])["skipped_identical"] == [run_id]
    (runs.find_run_dir(run_id) / "metrics_val.json").write_text(json.dumps({"rare_macro_f1": 0.99}))
    assert runs.ingest([archive])["conflicts"] == [run_id]
    assert json.loads((runs.find_run_dir(run_id) / "metrics_val.json").read_text()) == {"rare_macro_f1": 0.99}


def test_ingest_refuses_runs_that_carry_test_predictions(tmp_path):
    run_id = runs.make_run_id("phase4_augmentation_policy", "A", 0, now=NOW)
    source = tmp_path / "incoming"
    shutil.copytree(_write(run_id), source / run_id)
    shutil.rmtree(paths.runs_dir())
    np.savez(source / run_id / "preds_test.npz", probs=np.zeros((1, 4)))
    report = runs.ingest([source])
    assert report["refused"] == [run_id] and not report["ingested"]
    assert runs.find_run_dir(run_id) is None


def test_ingest_reports_incomplete_runs_and_bad_sources(tmp_path):
    run_id = runs.make_run_id("phase1_input_representation", "D", 2, now=NOW)
    source = tmp_path / "incoming"
    shutil.copytree(_write(run_id), source / run_id)
    shutil.rmtree(paths.runs_dir())
    (source / run_id / "metrics_val.json").unlink()
    assert runs.ingest([source])["incomplete"] == [run_id]
    with pytest.raises(FileNotFoundError):
        runs.ingest([tmp_path / "does-not-exist.zip"])


def test_ingest_brings_back_variant_result_files_without_overwriting(tmp_path):
    run_id = runs.make_run_id("phase1_input_representation", "A", 0, now=NOW)
    bundle = tmp_path / "bundle"
    shutil.copytree(_write(run_id), bundle / "runs" / run_id)
    shutil.rmtree(paths.runs_dir())
    result = bundle / "results" / "phase1_input_representation"
    result.mkdir(parents=True)
    (result / "A.json").write_text('{"variant": "A"}')
    (result / "A_notes.md").write_text("notes")

    report = runs.ingest([bundle])
    assert report["ingested"] == [run_id] and sorted(report["result_files"]) == ["phase1_input_representation/A.json", "phase1_input_representation/A_notes.md"]
    assert (paths.results_for("phase1_input_representation") / "A.json").read_text() == '{"variant": "A"}'

    (result / "A.json").write_text('{"variant": "A", "edited": true}')
    again = runs.ingest([bundle])
    assert again["result_conflicts"] == ["phase1_input_representation/A.json"] and not again["result_files"]
    assert (paths.results_for("phase1_input_representation") / "A.json").read_text() == '{"variant": "A"}'


def test_git_dirty_ignores_run_outputs_but_not_code(monkeypatch):
    def status(lines):
        monkeypatch.setattr(runs, "_git", lambda *args: "\n".join(lines))
        return runs.git_dirty()

    assert status([]) is False
    assert status(["?? runs/p1-m1-A-s0-abcdef0-202610081530/config.yaml", " M results/runs_index.csv"]) is False
    assert status(["?? figures/phase1_input_representation/A/training_and_confusion.png"]) is False  # drawn by an earlier run
    assert status(["?? runs/x/config.yaml", " M src/ecgaug/runs.py"]) is True
    assert status(["?? configs/phase1_input_representation/E.yaml"]) is True
    assert status(["R  results/a.json -> docs/a.json"]) is True
    monkeypatch.setattr(runs, "_git", lambda *args: None)
    assert runs.git_dirty() is True  # no git: cannot vouch for the tree


def test_kaggle_bundle_round_trips_through_ingest(tmp_path):
    import importlib.util

    from conftest import REPO

    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)

    fake_repo = tmp_path / "repo"
    assert kaggle_run.bundle(tmp_path / "out", {"git_sha": "abcdef0123"}, repo=fake_repo) is None
    run_id = runs.make_run_id("phase1_input_representation", "A", 0, now=NOW)
    shutil.copytree(_write(run_id), fake_repo / "runs" / run_id)
    shutil.rmtree(paths.runs_dir())
    (fake_repo / "results" / "phase1_input_representation").mkdir(parents=True)
    (fake_repo / "results" / "phase1_input_representation" / "A.json").write_text("{}")
    (tmp_path / "out").mkdir(exist_ok=True)
    archive = kaggle_run.bundle(tmp_path / "out", {"git_sha": "abcdef0123"}, repo=fake_repo)
    assert archive.name.startswith("ecgaug_abcdef0_") and archive.suffix == ".zip"
    report = runs.ingest([archive])
    assert report["ingested"] == [run_id] and report["result_files"] == ["phase1_input_representation/A.json"]

    nested = tmp_path / "input" / "mitdb-raw" / "files"
    nested.mkdir(parents=True)
    (nested / "100.dat").write_bytes(b"x")
    assert kaggle_run.find_raw(tmp_path / "input") == nested


def test_ptbxl_report_travels_in_the_bundle_and_is_ingested(tmp_path):
    import importlib.util

    from conftest import REPO

    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)

    fake_repo = tmp_path / "repo"
    run_id = runs.make_run_id("phase2_ptbxl_pretraining", "pretrainA", 0, now=NOW)
    shutil.copytree(_write(run_id), fake_repo / "runs" / run_id)
    shutil.rmtree(paths.runs_dir())
    (fake_repo / "results" / "phase2_ptbxl_data").mkdir(parents=True)
    (fake_repo / "results" / "phase2_ptbxl_data" / "ptbxl_preprocessing.json").write_text('{"recordings_kept": 5}')
    (fake_repo / "results" / "phase2_ptbxl_data" / "something_else.json").write_text("{}")
    (tmp_path / "out").mkdir()
    archive = kaggle_run.bundle(tmp_path / "out", {"git_sha": "abcdef0123"}, repo=fake_repo)

    report = runs.ingest([archive])
    assert report["ingested"] == [run_id] and report["result_files"] == ["phase2_ptbxl_data/ptbxl_preprocessing.json"]
    assert (paths.results_for("phase2_ptbxl_data") / "ptbxl_preprocessing.json").read_text() == '{"recordings_kept": 5}'
    assert not (paths.results_for("phase2_ptbxl_data") / "something_else.json").exists()
    (paths.results_for("phase2_ptbxl_data") / "ptbxl_preprocessing.json").write_text('{"recordings_kept": 6}')
    assert runs.ingest([archive])["result_conflicts"] == ["phase2_ptbxl_data/ptbxl_preprocessing.json"]


def test_runs_are_saved_in_their_group_folder_and_found_by_id():
    run_id = runs.make_run_id("phase1_input_representation", "A", 0, now=NOW)
    run_dir = runs.write_run(run_id, {}, {}, [{"epoch": 1}], {}, {"y": np.zeros(1)}, {}, group=("phase1_input_representation", "A_short_window"))
    assert run_dir == paths.runs_dir() / "phase1_input_representation" / "A_short_window" / run_id
    assert runs.find_run_dir(run_id) == run_dir
    with pytest.raises(FileExistsError, match="already exists"):  # IDs are unique across folders
        runs.write_run(run_id, {}, {}, [{"epoch": 1}], {}, {"y": np.zeros(1)}, {}, group=("elsewhere",))
    row = runs.index_row(run_id, {}, 1.0, group=("phase1_input_representation", "A_short_window"))
    assert row["folder"] == "phase1_input_representation/A_short_window"


def test_default_group_sorts_older_flat_runs():
    assert runs.default_group(runs.make_run_id("phase1_input_representation", "B", 2, tag="lr1e-03", now=NOW)) == ("phase1_input_representation", "B")
    assert runs.default_group(runs.make_run_id("phase2_ptbxl_pretraining", "pretrainD", 0, now=NOW)) == ("phase2_ptbxl_pretraining", "D", "pretrain")
    assert runs.default_group(runs.make_run_id("phase2_ptbxl_pretraining", "finetuneC", 1, now=NOW)) == ("phase2_ptbxl_pretraining", "C", "finetune")
    assert runs.default_group(runs.make_run_id("phase2_ptbxl_pretraining", "scratchA", 1, now=NOW)) == ("phase2_ptbxl_pretraining", "A", "scratch")


def test_ingest_keeps_the_folder_layout_of_a_bundle(tmp_path):
    run_id = runs.make_run_id("phase2_ptbxl_pretraining", "finetuneD", 0, tag="lr1e-03", now=NOW)
    nested = tmp_path / "bundle" / "runs" / "phase2_ptbxl_pretraining" / "D_short_window_125hz" / "finetune"
    nested.mkdir(parents=True)
    shutil.copytree(_write(run_id), nested / run_id)
    shutil.rmtree(paths.runs_dir())
    assert runs.ingest([tmp_path / "bundle"])["ingested"] == [run_id]
    assert runs.find_run_dir(run_id) == paths.runs_dir() / "phase2_ptbxl_pretraining" / "D_short_window_125hz" / "finetune" / run_id
    with (paths.results_dir() / "runs_index.csv").open() as fh:
        assert next(csv.DictReader(fh))["folder"] == "phase2_ptbxl_pretraining/D_short_window_125hz/finetune"


def test_one_naming_scheme_everywhere():
    """Experiment names follow phase<N>_<what> and are used for configs, runners, notebooks and the Kaggle bundle."""
    import importlib.util

    from conftest import REPO
    from ecgaug import names

    for experiment in names.EXPERIMENTS:
        assert names.phase_of(experiment) in names.PHASES
        assert (REPO / "configs" / experiment).is_dir()
        assert (REPO / "src" / "ecgaug" / "experiments" / f"{experiment}.py").is_file()
        assert (REPO / "kaggle" / f"{experiment}.ipynb").is_file()
    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)
    assert kaggle_run.EXPERIMENTS == names.EXPERIMENTS and kaggle_run.RESULT_FOLDERS == names.RESULT_FOLDERS
    run_id = runs.make_run_id(names.PHASE1_INPUT_REPRESENTATION, "A", 0, tag="lr1e-03", now=NOW)
    assert run_id.startswith("phase1_input_representation-A.lr1e-03-s0-")
    with pytest.raises(ValueError):
        names.phase_of("m1_input_representation")


def test_kaggle_results_are_committed_and_pushed(tmp_path):
    """The notebook's push step: outputs go to the run's branch, or to a new branch if it moved on."""
    import importlib.util
    import subprocess

    from conftest import REPO

    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)

    def git(*args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    remote, clone = tmp_path / "remote.git", tmp_path / "clone"
    git("init", "--quiet", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    git("clone", "--quiet", str(remote), str(clone), cwd=tmp_path)
    (clone / ".gitignore").write_text("/runs/**/*.pt\n/runs/**/*.npz\n")
    (clone / "README.md").write_text("x\n")
    git("add", ".", cwd=clone)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "init", cwd=clone)
    git("push", "--quiet", "origin", "main", cwd=clone)

    run_dir = clone / "runs" / "phase1_input_representation" / "A_short_window" / "phase1_input_representation-A.lr1e-03-s0-abcdef0-202610151200"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics_val.json").write_text("{}")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"weights")
    (clone / "results" / "phase1_input_representation").mkdir(parents=True)
    (clone / "results" / "phase1_input_representation" / "A.json").write_text("{}")
    (clone / "results" / "phase1_data").mkdir(parents=True)
    (clone / "results" / "phase1_data" / "raw_checksums.json").write_text("changed on Kaggle")  # must not be committed

    assert kaggle_run.push_results("main", "add results: A", "Ann", "ann@x", repo=clone) == "main"
    pushed = git("ls-tree", "-r", "--name-only", "main", cwd=remote).split("\n")
    assert "results/phase1_input_representation/A.json" in pushed and any(p.endswith("metrics_val.json") for p in pushed)
    assert not any(p.endswith(".pt") for p in pushed) and "results/phase1_data/raw_checksums.json" not in pushed
    assert git("log", "-1", "--format=%an %s", "main", cwd=remote) == "Ann add results: A"
    assert kaggle_run.push_results("main", "again", "Ann", "ann@x", repo=clone) is None  # nothing new

    # someone else pushed to main meanwhile: the results go to a new branch instead
    other = tmp_path / "other"
    git("clone", "--quiet", str(remote), str(other), cwd=tmp_path)
    (other / "NOTE.md").write_text("teammate\n")
    git("add", ".", cwd=other)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "teammate", cwd=other)
    git("push", "--quiet", "origin", "main", cwd=other)
    (clone / "results" / "phase1_input_representation" / "B.json").write_text("{}")
    branch = kaggle_run.push_results("main", "add results: B", "Ann", "ann@x", repo=clone)
    assert branch.startswith("main-kaggle-") and "results/phase1_input_representation/B.json" in git("ls-tree", "-r", "--name-only", branch, cwd=remote)

    (run_dir / "preds_test.npz").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="test predictions"):
        kaggle_run.push_results("main", "bad", "Ann", "ann@x", repo=clone)
