"""The test lock: DS2 cannot be read without a declared, flagged and logged access."""

import re

import numpy as np
import pytest

from conftest import REPO
from ecgaug import paths
from ecgaug.data import load_test, raw
from ecgaug.data.access import LockedTestDataError, log_test_access, require_test_access
from ecgaug.data.beats import extract_beats
from ecgaug.data.contract import build_arrays, write_dataset
from ecgaug.data.counts import count_record, record_counts
from ecgaug.data.records import DS2
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split

SRC = REPO / "src" / "ecgaug"
SPEC = RepresentationSpec(name="t", window_pre_s=0.25, window_post_s=0.25)


def _write_small_frozen(monkeypatch) -> None:
    """A v1 dataset built from the fake records, test part included."""
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    split = Split(name="t", train=(101, 106, 108, 114), val=(118, 201))
    trainval = build_arrays([101, 106, 108, 114, 118, 201], SPEC, split.split_of())
    test = build_arrays([100, 213], SPEC, {100: "test", 213: "test"}, access="construction")
    write_dataset(paths.processed_dir() / "v1", "v1", "frozen", SPEC, split, trainval, test)
    monkeypatch.delenv("ALLOW_TEST_BUILD")


def test_ds2_reads_raise_without_permission(fake_root):
    for reader in (raw.read_annotations, raw.read_signal, extract_beats):
        with pytest.raises(LockedTestDataError):
            reader(100)
    with pytest.raises(LockedTestDataError):
        raw.read_annotations(213, access="construction")  # declared, but the flag is not set
    with pytest.raises(LockedTestDataError):
        record_counts(include_test=True)


def test_lock_is_checked_before_any_file_is_opened():
    # the temporary root holds no raw files at all: the lock must fire first
    for record in DS2:
        with pytest.raises(LockedTestDataError):
            raw.read_signal(record)


def test_ds1_reads_need_no_permission(fake_root):
    samples, symbols = raw.read_annotations(101)
    assert len(samples) == len(symbols) > 0


def test_flag_must_be_exactly_one(monkeypatch):
    for value in ("true", "yes", "0", ""):
        monkeypatch.setenv("ALLOW_TEST", value)
        with pytest.raises(LockedTestDataError):
            require_test_access("evaluation")
    monkeypatch.setenv("ALLOW_TEST", "1")
    require_test_access("evaluation")


def test_construction_flag_does_not_unlock_evaluation(fake_root, monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    samples, _ = raw.read_annotations(100, access="construction")
    assert len(samples) > 0
    with pytest.raises(LockedTestDataError):
        raw.read_annotations(100)  # the access kind must still be declared
    with pytest.raises(LockedTestDataError):
        raw.read_annotations(100, access="evaluation")
    with pytest.raises(LockedTestDataError):
        load_test("v1")


def test_load_test_raises_without_allow_test_even_when_the_file_exists(fake_root, monkeypatch):
    _write_small_frozen(monkeypatch)
    assert (paths.processed_dir() / "v1" / "test.npz").is_file()
    with pytest.raises(LockedTestDataError):
        load_test("v1")
    log = paths.docs_dir() / "TEST_ACCESS_LOG.md"
    assert not log.exists() or "evaluation" not in log.read_text().split("|---")[-1]


def test_load_test_with_allow_test_loads_and_logs(fake_root, monkeypatch):
    _write_small_frozen(monkeypatch)
    monkeypatch.setenv("ALLOW_TEST", "1")
    arrays = load_test("v1", run_ids="p4-m7-A-s0-abcdef0-202610270000", approved_by="team")
    assert set(np.unique(arrays["record"]).tolist()) == {100, 213}
    assert (arrays["split"] == "test").all()
    rows = [line for line in (paths.docs_dir() / "TEST_ACCESS_LOG.md").read_text().splitlines() if "| evaluation |" in line]
    assert len(rows) == 1 and "load_test('v1')" in rows[0] and "team" in rows[0]


def test_counting_ds2_labels_is_a_logged_construction_access(fake_root, monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_BUILD", "1")
    row = count_record(100, access="construction")
    assert row["dataset"] == "DS2" and row["qrs_total"] > 0
    log_test_access("construction", "per-record class counts")
    text = (paths.docs_dir() / "TEST_ACCESS_LOG.md").read_text()
    assert text.startswith("# TEST ACCESS LOG") and "| construction |" in text
    assert "| date | kind | git SHA | command | purpose | run IDs | approved by |" in text


def test_paced_and_unknown_records_are_refused():
    with pytest.raises(ValueError, match="paced"):
        raw.read_signal(102)
    with pytest.raises(ValueError):
        raw.read_annotations(999)


# --- static checks on the source tree -------------------------------------------------------------


def _sources():
    return sorted(SRC.rglob("*.py"))


def test_only_the_raw_module_reads_wfdb_files():
    readers = re.compile(r"\b(rdann|rdrecord|rdsamp|rdheader|dl_database|dl_files)\s*\(")
    for path in _sources():
        text = path.read_text()
        if path.name != "raw.py":
            assert not readers.search(text), f"{path} reads raw files; all reads must go through ecgaug.data.raw"
        if re.search(r"^\s*(import wfdb|from wfdb)", text, flags=re.MULTILINE):
            assert path.name in ("raw.py", "fake.py"), f"{path} imports wfdb"


def test_only_contract_and_loaders_touch_the_test_file():
    for path in _sources():
        text = path.read_text()
        if "test.npz" in text or "TEST_FILE" in text:
            assert path.name in ("contract.py", "loaders.py"), f"{path} refers to the test file"


def test_no_source_file_sets_a_test_access_flag():
    setter = re.compile(r"(environ\s*\[\s*['\"]ALLOW_TEST|environ\.setdefault\(\s*['\"]ALLOW_TEST|putenv\(\s*['\"]ALLOW_TEST)")
    for path in _sources():
        assert not setter.search(path.read_text()), f"{path} sets a test-access flag; only a human or /final-eval may"
