"""Leakage checks on splits and on built datasets."""

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO
from ecgaug import paths
from ecgaug.data.contract import build_candidate
from ecgaug.data.counts import count_record
from ecgaug.data.loaders import load_candidate
from ecgaug.data.records import DS1, DS2, PACED
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, load_approved_split, make_split, options_report, summarize
from ecgaug.eval.selection import FALLBACK, MIN_F_BEATS, MIN_F_RECORDS, PRIMARY, selection_metric_name

SPEC = RepresentationSpec(name="t", window_pre_s=0.25, window_post_s=0.25)


def test_make_split_covers_ds1_with_zero_overlap():
    split = make_split("x", [201, 118, 223])
    assert set(split.train) | set(split.val) == set(DS1)
    assert not set(split.train) & set(split.val)
    assert not (set(split.train) | set(split.val)) & set(split.test)
    assert split.test == DS2


@pytest.mark.parametrize(
    "train, val, test, message",
    [
        (DS1[:-1], (DS1[-1], 100), DS2, "both val/test"),  # a test record in validation
        (DS1[:-1], (DS1[-1],), DS2[:-1], "exactly DS2"),  # test set altered
        (DS1[:-1] + (102,), (DS1[-1],), DS2, "paced"),  # a paced record
        (DS1, (DS1[-1],), DS2, "both train/val"),  # a record in train and validation
        (DS1[:-1] + (DS1[0],), (DS1[-1],), DS2, "listed twice"),  # duplicate
        (DS1, (), DS2, "non-empty"),  # empty validation
        (DS1[:-2], (DS1[-1],), DS2, "unassigned"),  # a DS1 record left out
    ],
)
def test_check_split_rejects_leaky_or_malformed_splits(train, val, test, message):
    with pytest.raises(ValueError, match=message):
        check_split(Split(name="bad", train=tuple(train), val=tuple(val), test=tuple(test)))


def test_every_proposed_option_is_leak_free():
    options = yaml.safe_load((REPO / "configs" / "splits" / "options.yaml").read_text())["options"]
    assert 2 <= len(options) <= 3
    for option in options:
        split = make_split(option["name"], option["val"])
        assert not set(split.val) & (set(DS2) | set(PACED))


def test_built_dataset_has_no_record_in_two_splits(full_fake_root):
    split = make_split("x", [201, 118, 223, 205])
    build_candidate(SPEC, split)
    arrays = load_candidate("t")
    table = pd.DataFrame({"record": arrays["record"], "split": arrays["split"]})
    assert (table.groupby("record")["split"].nunique() == 1).all()
    train = set(table.loc[table["split"] == "train", "record"])
    val = set(table.loc[table["split"] == "val", "record"])
    assert not train & val and val == set(split.val) and train == set(split.train)
    assert not (train | val) & set(DS2)
    assert set(np.unique(arrays["split"]).tolist()) == {"train", "val"}


def test_selection_metric_rule_thresholds():
    assert selection_metric_name(MIN_F_BEATS, MIN_F_RECORDS) == PRIMARY
    assert selection_metric_name(MIN_F_BEATS - 1, 5) == FALLBACK
    assert selection_metric_name(10_000, MIN_F_RECORDS - 1) == FALLBACK
    assert selection_metric_name(0, 0) == FALLBACK


def _approved(tmp_root, **overrides) -> None:
    data = {
        "name": "x",
        "val": [201, 118, 223, 205],
        "selection_metric": FALLBACK,
        "approved_by": ["A", "B"],
        "approved_on": "2026-10-09",
        "decision": "docs/DECISIONS.md Gate 1",
    }
    data.update(overrides)
    path = paths.configs_dir() / "splits" / "approved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def test_approved_split_is_required_and_must_name_approvers(tmp_root):
    with pytest.raises(FileNotFoundError, match="approved"):
        load_approved_split()
    _approved(tmp_root, approved_by=[])
    with pytest.raises(ValueError, match="approved_by"):
        load_approved_split()
    _approved(tmp_root)
    split = load_approved_split()
    assert split.val == (118, 201, 205, 223) and split.selection_metric == FALLBACK


def test_approved_split_must_agree_with_the_minimum_support_rule(full_fake_root):
    counts = pd.DataFrame([count_record(r) for r in DS1])
    expected = selection_metric_name(
        int(counts.loc[counts["record"].isin([201, 118, 223, 205]), "F"].sum()),
        int((counts.loc[counts["record"].isin([201, 118, 223, 205]), "F"] > 0).sum()),
    )
    wrong = PRIMARY if expected == FALLBACK else FALLBACK
    _approved(full_fake_root[0], selection_metric=wrong)
    with pytest.raises(ValueError, match="minimum-support"):
        load_approved_split(counts)
    _approved(full_fake_root[0], selection_metric=expected)
    assert load_approved_split(counts).selection_metric == expected


def test_summary_and_report_use_ds1_rows_only(full_fake_root):
    counts = pd.DataFrame([count_record(r) for r in DS1])
    poisoned = pd.concat(
        [counts, pd.DataFrame([{**counts.iloc[0].to_dict(), "record": 213, "dataset": "DS2", "F": 10**6}])],
        ignore_index=True,
    )
    split = make_split("x", [201, 118, 223, 205])
    assert summarize(poisoned, split) == summarize(counts, split)
    option = [{"name": "x", "val": [201, 118, 223, 205], "rationale": "r"}]
    assert options_report(poisoned, option) == options_report(counts, option)
    summary = summarize(counts, split)
    assert sum(summary[p]["classes"]["N"]["beats"] for p in ("train", "val")) == int(counts["N"].sum())
