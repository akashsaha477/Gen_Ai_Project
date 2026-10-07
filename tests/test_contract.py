"""The data contract: fields, deterministic files, stable hashes, tamper detection, frozen builds."""

import json
import shutil

import numpy as np
import pytest

from conftest import FAKE_DS1
from ecgaug import paths
from ecgaug.data import load_test
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.contract import (
    FIELDS,
    ContractError,
    build_arrays,
    build_candidate,
    build_frozen,
    content_sha256,
    save_npz_deterministic,
    sha256_file,
)
from ecgaug.data.loaders import load_candidate, load_trainval
from ecgaug.data.records import DS1, DS2
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, make_split

SPEC = RepresentationSpec(name="t", window_pre_s=0.25, window_post_s=0.25)
SMALL = Split(name="small", train=FAKE_DS1[:4], val=FAKE_DS1[4:])


def test_npz_bytes_depend_only_on_the_arrays(tmp_path):
    arrays = {"b": np.arange(5, dtype=np.int16), "a": np.linspace(0, 1, 7, dtype=np.float32), "s": np.array(["x", "yy"])}
    save_npz_deterministic(tmp_path / "one.npz", arrays)
    save_npz_deterministic(tmp_path / "two.npz", dict(reversed(list(arrays.items()))))
    assert sha256_file(tmp_path / "one.npz") == sha256_file(tmp_path / "two.npz")
    with np.load(tmp_path / "one.npz", allow_pickle=False) as loaded:
        for name, value in arrays.items():
            np.testing.assert_array_equal(loaded[name], value)
    changed = {**arrays, "b": arrays["b"] + 1}
    assert content_sha256(changed) != content_sha256(arrays)


def test_fields_and_dtypes_follow_the_contract(fake_root):
    arrays, _ = build_arrays(list(FAKE_DS1), SPEC, SMALL.split_of())
    assert tuple(arrays) == FIELDS
    n = len(arrays["y"])
    assert arrays["X"].shape == (n, 180) and arrays["X"].dtype == np.float32
    assert arrays["y"].dtype == np.int8 and set(np.unique(arrays["y"]).tolist()) <= {0, 1, 2, 3}
    assert arrays["record"].dtype == np.int16 and arrays["r_idx"].dtype == np.int32
    for name in ("rr_prev", "rr_next", "rr_ratio"):
        assert arrays[name].dtype == np.float32 and np.isfinite(arrays[name]).all() and (arrays[name] > 0).all()
    assert arrays["is_synthetic"].dtype == bool and not arrays["is_synthetic"].any()
    assert (arrays["source"] == "real").all()
    assert set(np.unique(arrays["split"]).tolist()) == {"train", "val"}
    assert all(len(arrays[name]) == n for name in FIELDS)


def test_q_and_unmapped_beats_are_excluded_and_counted(fake_root):
    arrays, report = build_arrays(list(FAKE_DS1), SPEC, SMALL.split_of())
    for entry in report:
        annotated = sum(entry["annotated"].values())
        assert annotated + entry["q_beats_excluded"] + entry["unmapped_qrs_excluded"] == entry["qrs_annotations"]
        for cls in "NSVF":
            assert entry["kept"][cls] + entry["dropped_unusable"][cls] == entry["annotated"][cls]
        assert entry["q_beats_excluded"] > 0 and entry["unmapped_qrs_excluded"] > 0  # the fake data has both
    assert len(arrays["y"]) == sum(sum(e["kept"].values()) for e in report)


def test_rebuild_reproduces_file_and_content_hashes(fake_root):
    first = json.loads((build_candidate(SPEC, SMALL, require_all_ds1=False) / "contract.json").read_text())
    shutil.rmtree(paths.processed_dir())
    second = json.loads((build_candidate(SPEC, SMALL, require_all_ds1=False) / "contract.json").read_text())
    assert first["files"] == second["files"]
    assert first["files"]["trainval.npz"]["content_sha256"] == content_sha256(load_candidate("t"))


def test_contract_json_records_how_the_data_was_made(fake_root):
    contract = json.loads((build_candidate(SPEC, SMALL, require_all_ds1=False) / "contract.json").read_text())
    for key in ("source", "representation", "labels", "rr", "usable_beat_rule", "segment_windows_ms", "split",
                "counts", "per_record", "files", "git_sha", "created_utc"):  # fmt: skip
        assert key in contract, key
    assert contract["source"]["fs"] == 360 and "MLII" in contract["source"]["lead"]
    assert contract["representation"]["n_samples"] == 180
    assert contract["split"]["train"] == list(SMALL.train) and contract["split"]["val"] == list(SMALL.val)
    assert contract["segment_windows_ms"] == {"P": [-250, -100], "QRS": [-50, 50], "ST": [50, 150], "T": [150, 400]}
    assert set(contract["files"]) == {"trainval.npz"}  # a candidate never contains test data
    assert sum(contract["counts"]["train"].values()) + sum(contract["counts"]["val"].values()) == contract["files"][
        "trainval.npz"
    ]["n_beats"]


def test_loading_detects_a_modified_file(fake_root):
    out = build_candidate(SPEC, SMALL, require_all_ds1=False)
    arrays = dict(load_candidate("t"))
    arrays["y"] = arrays["y"].copy()
    arrays["y"][0] = (arrays["y"][0] + 1) % 4
    save_npz_deterministic(out / "trainval.npz", arrays)
    with pytest.raises(ContractError, match="does not match the contract"):
        load_candidate("t")
    (out / "trainval.npz").unlink()
    with pytest.raises(ContractError, match="missing"):
        load_candidate("t")


def test_frozen_build_needs_the_construction_flag_and_writes_nothing_without_it(full_fake_root):
    split = make_split("x", [201, 118, 223, 205])
    with pytest.raises(LockedTestDataError):
        build_frozen("v1", SPEC, split)
    assert not (paths.processed_dir() / "v1").exists()


def test_frozen_build_logs_the_access_and_is_never_overwritten(full_fake_root, monkeypatch):
    split = make_split("x", [201, 118, 223, 205])
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    out = build_frozen("v1", SPEC, split)
    contract = json.loads((out / "contract.json").read_text())
    assert set(contract["files"]) == {"trainval.npz", "test.npz"} and contract["kind"] == "frozen"
    assert sum(contract["counts"]["test"].values()) == contract["files"]["test.npz"]["n_beats"] > 0
    log = (paths.docs_dir() / "TEST_ACCESS_LOG.md").read_text()
    assert "| construction |" in log and "built v1/test.npz" in log and "| evaluation |" not in log
    with pytest.raises(ContractError, match="never overwritten"):
        build_frozen("v1", SPEC, split)

    # zero record overlap between the frozen train/validation and test files
    trainval = load_trainval("v1")
    monkeypatch.setenv("ALLOW_TEST", "1")
    test = load_test("v1")
    assert set(np.unique(trainval["record"]).tolist()) == set(DS1)
    assert set(np.unique(test["record"]).tolist()) == set(DS2)
    assert not set(trainval["record"].tolist()) & set(test["record"].tolist())
    assert tuple(test) == tuple(sorted(FIELDS)) or set(test) == set(FIELDS)
