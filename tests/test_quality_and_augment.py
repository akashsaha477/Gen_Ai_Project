"""Diagnostics (diversity, memorization, C2ST), the composite rank, and the augmentation conditions."""

import numpy as np
import pytest

from ecgaug import augment, quality
from ecgaug.data.records import CLASS_INDEX
from ecgaug.generators import TrainOnlyError
from ecgaug.quality.diagnostics import c2st_accuracy, leave_one_out_distances, mean_pairwise_distance, prepare_reference, score_class

from test_generators import training_part


def _class(n, seed, shift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (shift + scale * rng.standard_normal((n, 20))).astype(np.float32)


def test_mean_pairwise_distance_hand_computed():
    assert mean_pairwise_distance(np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]])) == pytest.approx((5 + 4 + 3) / 3)
    assert np.isnan(mean_pairwise_distance(np.zeros((1, 3))))


def test_diagnostics_separate_good_collapsed_and_copying_generators():
    real = _class(600, seed=0)
    ref = prepare_reference(real, "S", percentile=5, max_beats=400, seed=0)
    fresh = score_class(_class(400, seed=1), ref, max_beats=400, folds=5, seed=0)  # same distribution, new beats
    assert fresh["diversity_ratio"] == pytest.approx(1.0, abs=0.05)
    assert fresh["memorization_fraction"] < 0.1  # about 0.05 expected
    assert fresh["c2st_accuracy"] < 0.65

    collapsed = score_class(np.repeat(_class(1, seed=2), 50, axis=0) + 0.01 * _class(50, seed=3), ref, 400, 5, 0)
    assert collapsed["diversity_ratio"] < 0.1

    copies = score_class(ref.pool[:200], ref, 400, 5, 0)  # beats copied from the training pool
    assert copies["memorization_fraction"] == 1.0

    shifted = score_class(_class(400, seed=4, shift=1.0), ref, 400, 5, 0)
    assert shifted["c2st_accuracy"] > 0.9


def test_memorization_threshold_is_calibrated_to_five_percent():
    fractions = []
    for seed in range(5):
        ref = prepare_reference(_class(400, seed), "F", percentile=5, max_beats=400, seed=0)
        fresh = _class(1000, 100 + seed)
        fractions.append(score_class(fresh, ref, 1000, 5, 0)["memorization_fraction"])
    assert 0.02 < np.mean(fractions) < 0.09 and max(fractions) < 0.1
    np.testing.assert_allclose(leave_one_out_distances(np.array([[0.0], [1.0], [3.0]])), [1.0, 1.0, 2.0])


def test_c2st_needs_enough_beats():
    assert np.isnan(c2st_accuracy(_class(3, 0), _class(3, 1), folds=5, seed=0))


def test_composite_rank_is_the_mean_of_per_metric_ranks(monkeypatch):
    monkeypatch.setitem(quality.METRIC_FAMILIES, "dtw", quality.MetricFamily("dtw", lambda a, b: 0.0, higher_is_better=False))
    monkeypatch.setitem(quality.METRIC_FAMILIES, "tstr", quality.MetricFamily("tstr", lambda a, b: 0.0, higher_is_better=True))
    values = {"A": {"dtw": 1.0, "tstr": 0.2}, "B": {"dtw": 2.0, "tstr": 0.9}, "C": {"dtw": 3.0, "tstr": 0.5}}
    assert quality.composite_ranks(values, ["dtw", "tstr"]) == {"A": 2.0, "B": 1.5, "C": 2.5}
    assert quality.composite_ranks(values, []) is None
    assert quality.composite_ranks(values, ["fid"]) is None
    assert quality.composite_ranks({**values, "D": {"dtw": 1.0}}, ["dtw", "tstr"]) is None


def test_targets_bring_each_minority_class_to_a_fraction_of_n():
    counts = {"N": 1000, "S": 100, "V": 600, "F": 10}
    assert augment.fraction_of_n_targets(counts, 0.5) == {"S": 400, "V": 0, "F": 490}


def test_class_weights_hand_computed():
    w = augment.class_weights({"N": 100, "S": 25, "V": 50, "F": 4}, 0.5)
    np.testing.assert_allclose(w, [1.0, 2.0, np.sqrt(2.0), 5.0])
    np.testing.assert_allclose(augment.class_weights({"N": 100, "S": 25, "V": 50, "F": 4}, 1.0), [1, 4, 2, 25])
    with pytest.raises(ValueError):
        augment.class_weights({"N": 1, "S": 0, "V": 1, "F": 1}, 1.0)


def test_random_oversampling_copies_real_training_beats():
    train = training_part()
    added = augment.random_oversample(train, {"S": 30, "V": 0, "F": 5}, seed=0)
    assert len(added["y"]) == 35 and not added["is_synthetic"].any() and (added["source"] == "random_oversampling").all()
    rows = {r.tobytes() for r in train["X"][train["y"] == CLASS_INDEX["S"]]}
    assert all(r.tobytes() in rows for r in added["X"][added["y"] == CLASS_INDEX["S"]])
    np.testing.assert_array_equal(added["X"], augment.random_oversample(train, {"S": 30, "V": 0, "F": 5}, seed=0)["X"])


def test_smote_interpolates_training_beats_and_keeps_the_contract():
    train = training_part()
    added = augment.smote_beats(train, {"S": 20, "V": 0, "F": 8}, seed=0)
    assert np.bincount(added["y"], minlength=4).tolist() == [0, 20, 0, 8]
    assert added["is_synthetic"].all() and (added["source"] == "smote").all() and (added["split"] == "train").all()
    np.testing.assert_allclose(added["X"].std(axis=1), 1, atol=1e-3)
    s_rr = train["rr_prev"][train["y"] == CLASS_INDEX["S"]]
    new_rr = added["rr_prev"][added["y"] == CLASS_INDEX["S"]]
    assert s_rr.min() - 1e-5 <= new_rr.min() and new_rr.max() <= s_rr.max() + 1e-5  # interpolations stay inside the class
    np.testing.assert_array_equal(added["X"], augment.smote_beats(train, {"S": 20, "V": 0, "F": 8}, seed=0)["X"])


def test_oversamplers_refuse_validation_beats_and_added_beats_stay_in_training():
    train = training_part()
    leaked = {**train, "split": np.where(np.arange(len(train["y"])) < 5, "val", "train")}
    for fn in (augment.random_oversample, augment.smote_beats):
        with pytest.raises(TrainOnlyError):
            fn(leaked, {"S": 5, "V": 0, "F": 0}, 0)
    combined = augment.with_added(train, augment.random_oversample(train, {"F": 4}, 0))
    assert len(combined["y"]) == len(train["y"]) + 4 and (combined["split"] == "train").all()
    bad = augment.random_oversample(train, {"F": 4}, 0)
    bad["split"][:] = "val"
    with pytest.raises(ValueError, match="training beats"):
        augment.with_added(train, bad)


def test_take_from_bank_uses_the_first_beats_of_each_class():
    train = training_part()
    bank = augment.random_oversample(train, {"S": 6, "F": 6}, seed=1)
    taken = augment.take_from_bank(bank, {"S": 2, "F": 3})
    s_rows = bank["X"][bank["y"] == CLASS_INDEX["S"]][:2]
    np.testing.assert_array_equal(taken["X"][taken["y"] == CLASS_INDEX["S"]], s_rows)
    with pytest.raises(Exception, match="holds 6 S beats"):
        augment.take_from_bank(bank, {"S": 7})
