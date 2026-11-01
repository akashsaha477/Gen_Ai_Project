"""phase4_leakage_audit: the intra-patient split, the locks around it, and the report (DECISIONS A4, A16)."""

import json

import numpy as np
import pytest
from conftest import REPO

from ecgaug import final_eval, paths, runs
from ecgaug.data.access import LockedTestDataError, logged_accesses
from ecgaug.experiments import phase4_leakage_audit as audit
from ecgaug.experiments.common import PreconditionError
from ecgaug.generators.bank import BankError, bank_dir, empty_part
from ecgaug.grid import runner

CONFIG = REPO / "configs" / "phase4_leakage_audit" / "audit.yaml"
STAMP = "abcdef0-202611010000"


def _part(split: str, records: list[int], labels: list[int]) -> dict:
    n = len(labels)
    part = empty_part()
    return {**part, "X": np.arange(n * 4, dtype=np.float32).reshape(n, 4), "y": np.asarray(labels, dtype=np.int8),
            "record": np.asarray(records, dtype=np.int16), "split": np.full(n, split, dtype="<U5"), "r_idx": np.arange(n, dtype=np.int32),
            "rr_prev": np.ones(n, np.float32), "rr_next": np.ones(n, np.float32), "rr_ratio": np.ones(n, np.float32),
            "is_synthetic": np.zeros(n, bool), "source": np.full(n, "real")}  # fmt: skip


def _inter_patient() -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    train = _part("train", [101] * 40 + [106] * 40, list(rng.integers(0, 4, 80)))
    val = _part("val", [118] * 30, list(rng.integers(0, 4, 30)))
    test = _part("test", [100] * 25 + [103] * 25, list(rng.integers(0, 4, 50)))
    trainval = {k: np.concatenate([train[k], val[k]]) for k in train}
    return trainval, test


def _log_evaluation() -> None:
    (paths.docs_dir()).mkdir(parents=True, exist_ok=True)
    (paths.docs_dir() / "TEST_ACCESS_LOG.md").write_text(
        "| date | kind | git SHA | command | purpose | run IDs | approved by |\n|---|---|---|---|---|---|---|\n"
        "| 2026-10-30 | evaluation | abcdef0 | `final_eval` | load_test('v1') | 12 | A |\n"
    )


def test_config_is_checked():
    cfg = audit.load_config(CONFIG)
    assert cfg["conditions"] == ["none", "cvae", "wgan_gp"] and cfg["strength"] == "medium" and cfg["seeds"] == [0, 1, 2]


def test_split_keeps_the_class_counts_of_each_part_and_mixes_patients():
    trainval, test = _inter_patient()
    pooled = audit.intra_patient_split(trainval, test, seed=0)
    for part, source in (("train", trainval["y"][trainval["split"] == "train"]), ("val", trainval["y"][trainval["split"] == "val"]), ("test", test["y"])):
        assert np.bincount(pooled["y"][pooled["split"] == part], minlength=4).tolist() == np.bincount(source, minlength=4).tolist()
    info = audit.describe_split(pooled, 0)
    assert info["records_in_train_and_test"] > 0 and info["records"] == 5
    again = audit.intra_patient_split(trainval, test, seed=0)
    assert (again["split"] == pooled["split"]).all()
    assert (audit.intra_patient_split(trainval, test, seed=1)["split"] != pooled["split"]).any()


def test_medium_strength_is_the_middle_of_the_grid():
    grid = {"policy": {"kind": "fraction_of_n"}, "conditions": {"cvae": {"strengths": [0.5, 0.1, 0.25]}, "wgan_gp": {"strengths": [0.1, 0.25]}}}
    assert audit.medium_strength(grid, "cvae") == 0.25 and audit.medium_strength(grid, "wgan_gp") == 0.25
    assert audit.medium_strength({"policy": {"kind": "balance_to_n"}, "conditions": {"cvae": {}}}, "cvae") is None


def test_nothing_runs_before_the_final_evaluation_or_without_the_audit_flag(monkeypatch):
    cfg = audit.load_config(CONFIG)
    with pytest.raises(PreconditionError, match="after the final evaluation"):
        audit.frozen_context(cfg)
    with pytest.raises(LockedTestDataError, match="ALLOW_TEST_AUDIT"):
        audit.test_arrays(None, "test")
    monkeypatch.setenv("ALLOW_TEST_AUDIT", "1")
    with pytest.raises(PreconditionError, match="after the final evaluation"):
        audit.test_arrays(None, "test")
    assert audit.main(["--config", str(CONFIG)]) == 2


def test_audit_artifacts_never_enter_the_main_pipeline(tmp_path):
    with pytest.raises(BankError):
        bank_dir("audit/phase3_cvae-A")
    run_id = f"phase4_leakage_audit-cnn.none-s0-{STAMP}"
    run_dir = paths.runs_dir() / "audit" / "phase4_leakage_audit" / "audit" / "cnn" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"weights")
    gate = {"chosen": {"cnn": {"none": {"run_ids": [run_id], "checkpoints": {run_id: runs._sha256(run_dir / "checkpoints" / "best.pt")}, "strength": None}}},
            "best_generative": {}, "approved_by": ["A"]}  # fmt: skip
    (paths.results_dir() / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase4_gate": gate}))
    (checkpoint,) = final_eval.plan()["checkpoints"]
    assert not checkpoint["present"] and not checkpoint["hash_matches"]


def test_ingest_takes_audit_runs_and_banks_only_after_the_final_evaluation(tmp_path):
    bundle = tmp_path / "bundle"
    run_id = f"phase4_leakage_audit-cnn.none-s0-{STAMP}"
    run_dir = bundle / "runs" / "audit" / "phase4_leakage_audit" / "audit" / "cnn" / run_id
    run_dir.mkdir(parents=True)
    for name, text in (("config.yaml", "{}\n"), ("env.json", '{"wall_clock_s": 1}'), ("train_log.csv", "epoch\n1\n"), ("metrics_val.json", "{}")):
        (run_dir / name).write_text(text)
    bank = bundle / "data" / "synthetic" / "audit" / "phase3_cvae-A"
    bank.mkdir(parents=True)
    (bank / "seed0.npz").write_bytes(b"bank")
    report = runs.ingest([bundle])
    assert report["refused"] == [run_id] and not report["bank_files"] and runs.find_run_dir(run_id) is None
    _log_evaluation()
    assert logged_accesses("evaluation")
    report = runs.ingest([bundle])
    assert report["ingested"] == [run_id] and report["bank_files"] == ["data/synthetic/audit/phase3_cvae-A/seed0.npz"]


def test_attached_audit_banks_stay_under_audit_on_kaggle(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)
    for folder in ("audit/phase3_cvae-A", "phase3_cvae-A"):
        bank = tmp_path / "input" / "data" / "synthetic" / folder
        bank.mkdir(parents=True)
        (bank / "bank.json").write_text("{}")
        (bank / "seed0.npz").write_bytes(folder.encode())
    copied = kaggle_run.setup_artifacts(str(tmp_path / "input"), repo=tmp_path / "repo")
    assert "data/synthetic/audit/phase3_cvae-A/seed0.npz" in copied and "data/synthetic/phase3_cvae-A/seed0.npz" in copied
    assert (tmp_path / "repo" / "data" / "synthetic" / "phase3_cvae-A" / "seed0.npz").read_bytes() == b"phase3_cvae-A"


def test_report_sets_intra_beside_inter_patient_results():
    cfg = {**audit.load_config(CONFIG), "seeds": [0, 1]}
    grid = {"experiment": audit.EXPERIMENT, "name": audit.GRID_NAME}
    values = {("none", 0): 0.80, ("none", 1): 0.70, ("cvae", 0): 0.85, ("cvae", 1): 0.80, ("wgan_gp", 0): 0.70, ("wgan_gp", 1): 0.72}
    for (condition, seed), value in values.items():
        runner.write_cell(grid, {"cell_sha": f"{condition}{seed}".ljust(16, "0"), "condition": condition, "seed": seed, "strength": None if condition == "none" else 0.25,
                                 "run_id": f"phase4_leakage_audit-cnn.{condition.replace('_', '')}-s{seed}-{STAMP}", "status": "done",
                                 "audit_test": {"rare_macro_f1": value}})  # fmt: skip
    final = {"conditions": {"cnn/none": {"rare_macro_f1": {"point": 0.30, "ci_low": 0.2, "ci_high": 0.4}, "strength": None},
                            "cnn/cvae": {"rare_macro_f1": {"point": 0.32, "ci_low": 0.2, "ci_high": 0.45}, "strength": 0.5}},
             "families": {"confirmatory": {"cnn: cvae vs none": {"point": 0.02, "ci_low": -0.05, "ci_high": 0.09}}}}  # fmt: skip
    paths.results_for("final_eval").mkdir(parents=True)
    (paths.results_for("final_eval") / "summary.json").write_text(json.dumps(final))
    summary = audit.report(cfg)
    assert summary["conditions"]["none"]["intra"]["mean"] == pytest.approx(0.75)
    assert summary["inflation_none"] == pytest.approx(0.45)
    assert summary["gains"]["cvae"]["intra"]["mean"] == pytest.approx(0.075) and summary["gains"]["cvae"]["inter"]["point"] == 0.02
    assert summary["gains"]["wgan_gp"]["inter"] is None and summary["conditions"]["wgan_gp"]["inter"] is None
    from ecgaug import paper

    tex, _ = paper.leakage_audit_table()
    assert "\\missing" not in tex and "0.750" in tex


def test_smoke_runs_end_to_end():
    out = audit.smoke(audit.load_config(CONFIG))
    assert out["cells"] == 3 and out["skipped_on_rerun"] == 3 and out["runs_under_audit"] == 5
    assert out["conditions"] == ["cvae", "none", "wgan_gp"] and out["split"]["records_in_train_and_test"] > 0


def test_ds2_is_read_as_a_logged_audit_access_and_checked_against_the_frozen_hash(fake_root, monkeypatch):
    from test_test_lock import SPEC, _write_small_frozen

    from ecgaug.data.contract import content_sha256

    _write_small_frozen(monkeypatch)
    with np.load(paths.processed_dir() / "v1" / "test.npz", allow_pickle=False) as data:
        expected = content_sha256({k: data[k] for k in data.files})
    _log_evaluation()
    paths.results_for("final_eval").mkdir(parents=True)
    (paths.results_for("final_eval") / "summary.json").write_text("{}")
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase1_gate": {"files": {"test.npz": {"content_sha256": expected}}}}))
    monkeypatch.setenv("ALLOW_TEST_AUDIT", "1")
    arrays = audit.test_arrays(SPEC, "unit test")
    assert set(np.unique(arrays["record"]).tolist()) == {100, 213}
    (paths.processed_dir() / "v1" / "test.npz").unlink()  # as on Kaggle: rebuilt from the raw records
    monkeypatch.setattr(audit, "DS2", (100, 213))
    assert content_sha256(audit.test_arrays(SPEC, "unit test")) == expected
    rows = logged_accesses("audit")
    assert len(rows) == 2 and "data/processed/v1" in rows[0] and "rebuilt from the raw DS2 records" in rows[1]
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase1_gate": {"files": {"test.npz": {"content_sha256": "0" * 64}}}}))
    with pytest.raises(PreconditionError, match="do not match"):
        audit.test_arrays(SPEC, "unit test")
