"""The record-level bootstrap: whole records are resampled, results are reproducible and correct."""

import numpy as np
import pytest

from ecgaug.eval.bootstrap import (
    _diff_summary,
    cluster_bootstrap,
    cluster_bootstrap_confusion,
    draw_record_counts,
    holm,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_confusion,
)
from ecgaug.eval.metrics import confusion_by_record, confusion_matrix, rare_macro_f1_from_cm


def accuracy(probs, y):
    return float((probs.argmax(axis=1) == y).mean())


def rare_f1(probs, y):
    return float(rare_macro_f1_from_cm(confusion_matrix(y, probs.argmax(axis=1))))


def _toy(n_records=8, beats=150, flip=0.3, seed=0, f_records=(0,)):
    """Noisy predictions for records that all hold N, S and V; F only in ``f_records``."""
    rng = np.random.default_rng(seed)
    y, records = [], []
    for r in range(n_records):
        classes = [0, 1, 2, 3] if r in f_records else [0, 1, 2]
        y.append(rng.choice(classes, size=beats))
        records.append(np.full(beats, 100 + r))
    y, records = np.concatenate(y), np.concatenate(records)
    pred = np.where(rng.random(y.size) < flip, rng.integers(0, 4, y.size), y)
    return np.eye(4)[pred] * 0.9 + 0.025, y, records


def test_records_are_resampled_whole_not_beat_by_beat():
    # four equal records, two classified perfectly and two entirely wrong:
    # every replicate's accuracy must be a multiple of 1/4
    y = np.zeros(400, dtype=int)
    records = np.repeat([1, 2, 3, 4], 100)
    pred = np.where(records <= 2, 0, 1)
    out = cluster_bootstrap(accuracy, np.eye(4)[pred], y, records, n=500, seed=1, return_replicates=True)
    assert set(np.round(out["replicates"] * 4, 9).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert len(set(out["replicates"].tolist())) == 5
    assert out["point"] == pytest.approx(0.5) and out["se"] > 0.2


def test_draws_have_one_slot_per_record():
    counts = draw_record_counts(22, 1000, seed=0)
    assert counts.shape == (1000, 22) and (counts.sum(axis=1) == 22).all()
    assert (counts == 0).mean() == pytest.approx((21 / 22) ** 22, abs=0.02)


def test_bootstrap_is_reproducible_for_a_seed():
    probs, y, records = _toy()
    a = cluster_bootstrap(accuracy, probs, y, records, n=300, seed=7, return_replicates=True)
    b = cluster_bootstrap(accuracy, probs, y, records, n=300, seed=7, return_replicates=True)
    c = cluster_bootstrap(accuracy, probs, y, records, n=300, seed=8, return_replicates=True)
    np.testing.assert_array_equal(a["replicates"], b["replicates"])
    assert not np.array_equal(a["replicates"], c["replicates"])
    assert a["ci_low"] <= a["point"] <= a["ci_high"]


def test_single_record_gives_a_zero_width_interval():
    probs, y, records = _toy(n_records=1)
    out = cluster_bootstrap(accuracy, probs, y, records, n=100)
    assert out["ci_low"] == pytest.approx(out["point"]) == pytest.approx(out["ci_high"])


def test_class_absence_matches_the_analytical_probability():
    probs, y, records = _toy(n_records=4, beats=300, f_records=(0,))
    out = cluster_bootstrap(rare_f1, probs, y, records, n=20_000, seed=3)
    expected = (3 / 4) ** 4  # record 100 is missing from a replicate of four draws
    assert out["class_absent_frac"]["F"] == pytest.approx(expected, abs=0.015)
    assert out["frac_undefined"] == pytest.approx(out["class_absent_frac"]["F"])
    assert out["class_absent_frac"]["S"] == 0.0
    dominant = out["dominant_record"]["F"]
    assert dominant["record"] == 100 and dominant["share_of_class"] == 1.0
    assert dominant["absent_frac"] == pytest.approx(expected, abs=0.015)
    assert out["n_replicates"] == 20_000 and not np.isnan(out["ci_low"])


def test_confusion_fast_path_equals_the_generic_path():
    probs, y, records = _toy(f_records=(0, 3))
    slow = cluster_bootstrap(rare_f1, probs, y, records, n=400, seed=5, return_replicates=True)
    unique, cms = confusion_by_record(y, probs.argmax(axis=1), records)
    fast = cluster_bootstrap_confusion(rare_macro_f1_from_cm, cms, unique, n=400, seed=5, return_replicates=True)
    np.testing.assert_allclose(fast["replicates"], slow["replicates"], equal_nan=True)
    for key in ("point", "ci_low", "ci_high", "frac_undefined"):
        assert fast[key] == pytest.approx(slow[key])
    assert fast["class_absent_frac"] == slow["class_absent_frac"]
    assert fast["dominant_record"] == slow["dominant_record"]


def test_seeds_are_averaged_inside_each_replicate():
    probs_a, y, records = _toy(seed=0)
    probs_b = _toy(seed=0, flip=0.5)[0]
    both = cluster_bootstrap(accuracy, [probs_a, probs_b], y, records, n=200, seed=2, return_replicates=True)
    one = cluster_bootstrap(accuracy, probs_a, y, records, n=200, seed=2, return_replicates=True)
    two = cluster_bootstrap(accuracy, probs_b, y, records, n=200, seed=2, return_replicates=True)
    np.testing.assert_allclose(both["replicates"], (one["replicates"] + two["replicates"]) / 2)
    assert both["point"] == pytest.approx((one["point"] + two["point"]) / 2)


def test_paired_difference_of_identical_conditions_is_zero():
    probs, y, records = _toy()
    out = paired_cluster_bootstrap_diff(accuracy, probs, probs, y, records, n=200)
    assert out["point"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0 and out["p_two_sided"] == 1.0


def test_paired_difference_detects_a_consistent_improvement():
    noisy, y, records = _toy(flip=0.4)
    perfect = np.eye(4)[y]
    out = paired_cluster_bootstrap_diff(accuracy, perfect, noisy, y, records, n=999, seed=4)
    assert out["point"] > 0.2 and out["ci_low"] > 0
    assert out["p_two_sided"] == pytest.approx(2 / 1000)  # no replicate at or below zero
    reverse = paired_cluster_bootstrap_diff(accuracy, noisy, perfect, y, records, n=999, seed=4)
    assert reverse["point"] == pytest.approx(-out["point"]) and reverse["ci_high"] < 0


def test_paired_p_value_formula_hand_computed():
    out = _diff_summary(0.5, np.array([-1.0, 1.0, 1.0, 1.0]), level=0.95)
    assert out["p_two_sided"] == pytest.approx(2 * 2 / 5)  # one replicate <= 0, so tail = 1 + 1
    out = _diff_summary(0.5, np.array([np.nan, -1.0, 1.0, 1.0, 1.0]), level=0.95)
    assert out["p_two_sided"] == pytest.approx(0.8) and out["frac_undefined"] == pytest.approx(0.2)
    assert np.isnan(_diff_summary(0.0, np.array([np.nan, np.nan]), level=0.95)["p_two_sided"])


def test_paired_confusion_fast_path_equals_the_generic_path():
    probs_a, y, records = _toy(flip=0.2, f_records=(0, 3))
    probs_b = _toy(flip=0.45, seed=1, f_records=(0, 3))[0]
    slow = paired_cluster_bootstrap_diff(rare_f1, probs_a, probs_b, y, records, n=300, seed=9, return_replicates=True)
    cms_a = confusion_by_record(y, probs_a.argmax(axis=1), records)[1]
    cms_b = confusion_by_record(y, probs_b.argmax(axis=1), records)[1]
    fast = paired_cluster_bootstrap_diff_confusion(
        rare_macro_f1_from_cm, cms_a, cms_b, n=300, seed=9, return_replicates=True
    )
    np.testing.assert_allclose(fast["replicates"], slow["replicates"], equal_nan=True)
    assert fast["p_two_sided"] == pytest.approx(slow["p_two_sided"])
    assert fast["point"] == pytest.approx(slow["point"])


def test_holm_adjustment_hand_computed():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d 0.005*4, a 0.01*3, c 0.03*2, b 0.04*1 -> 0.02, 0.03, 0.06, 0.04 -> running max
    assert [out[k]["p_adjusted"] for k in "dacb"] == pytest.approx([0.02, 0.03, 0.06, 0.06])
    assert [out[k]["reject"] for k in "dacb"] == [True, True, False, False]
    assert list(out) == ["a", "b", "c", "d"]
    assert holm({"x": 0.6, "y": 0.9})["y"]["p_adjusted"] == 1.0
