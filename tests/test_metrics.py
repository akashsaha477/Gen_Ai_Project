"""Metric functions against hand-computed cases."""

import numpy as np
import pytest

from ecgaug.eval.metrics import (
    aami_ppv_v_from_cm,
    confusion_by_record,
    confusion_matrix,
    evaluate,
    f1_from_cm,
    macro_f1_from_cm,
    metrics_from_confusion,
    pr_auc_one_vs_rest,
    precision_from_cm,
    rare_macro_f1_from_cm,
    recall_from_cm,
)

# rows = true N, S, V, F; columns = predicted N, S, V, F
CM = np.array(
    [
        [90, 5, 3, 2],
        [4, 5, 1, 0],
        [2, 0, 17, 1],
        [3, 0, 4, 3],
    ]
)


def _labels(cm):
    y_true, y_pred = [], []
    for i in range(4):
        for j in range(4):
            y_true += [i] * int(cm[i, j])
            y_pred += [j] * int(cm[i, j])
    return np.array(y_true), np.array(y_pred)


def _one_hot(y_pred):
    return np.eye(4)[y_pred] * 0.97 + 0.01


def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    y_true, y_pred = _labels(CM)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), CM)


def test_per_class_metrics_hand_computed():
    m = metrics_from_confusion(CM)["per_class"]
    # S: TP 5, FP 5 (N->S), FN 5
    assert (m["S"]["precision"], m["S"]["recall"], m["S"]["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    # F: TP 3, FP 2 + 1 = 3, FN 7
    assert (m["F"]["precision"], m["F"]["recall"], m["F"]["f1"]) == pytest.approx((0.5, 0.3, 6 / 16))
    # V: TP 17, FP 3 + 1 + 4 = 8, FN 3
    assert (m["V"]["precision"], m["V"]["recall"], m["V"]["f1"]) == pytest.approx((17 / 25, 0.85, 34 / 45))
    # N: TP 90, FP 4 + 2 + 3 = 9, FN 10
    assert (m["N"]["precision"], m["N"]["recall"], m["N"]["f1"]) == pytest.approx((90 / 99, 0.9, 180 / 199))
    assert [m[c]["support"] for c in "NSVF"] == [100, 10, 20, 10]
    assert [m[c]["predicted"] for c in "NSVF"] == [99, 10, 25, 6]


def test_primary_and_macro_metrics_hand_computed():
    m = metrics_from_confusion(CM)
    assert m["rare_macro_f1"] == pytest.approx((0.5 + 0.375) / 2)
    assert m["macro_f1"] == pytest.approx((180 / 199 + 0.5 + 34 / 45 + 0.375) / 4)
    assert m["accuracy"] == pytest.approx(115 / 140)
    assert m["n_recall"] == pytest.approx(0.9)
    assert m["n_beats"] == 140


def test_aami_ppv_for_v_does_not_penalise_fusion_beats_called_v():
    m = metrics_from_confusion(CM)
    assert m["aami"]["se_S"] == pytest.approx(0.5) and m["aami"]["ppv_S"] == pytest.approx(0.5)
    assert m["aami"]["se_V"] == pytest.approx(0.85)
    assert m["aami"]["ppv_V"] == pytest.approx(17 / 21)  # FP = N->V (3) + S->V (1); F->V (4) is not counted
    assert m["aami"]["ppv_V"] > m["per_class"]["V"]["precision"]
    assert float(aami_ppv_v_from_cm(CM)) == pytest.approx(17 / 21)


def test_pr_auc_hand_computed():
    # class S scores rank the four beats S, N, S, N -> AP = (1/1 + 2/3) / 2
    y = np.array([1, 0, 1, 0])
    probs = np.array([[0.05, 0.90, 0.03, 0.02], [0.15, 0.80, 0.03, 0.02], [0.25, 0.70, 0.03, 0.02], [0.85, 0.10, 0.03, 0.02]])
    pr = pr_auc_one_vs_rest(probs, y)
    assert pr["S"] == pytest.approx(5 / 6)
    assert np.isnan(pr["V"]) and np.isnan(pr["F"])  # no true beats of these classes


def test_absent_class_is_undefined_not_zero():
    cm = CM.copy()
    cm[3] = 0  # no true F beats, though F is still predicted
    assert np.isnan(f1_from_cm(cm)[3]) and np.isnan(recall_from_cm(cm)[3])
    assert np.isnan(rare_macro_f1_from_cm(cm)) and np.isnan(macro_f1_from_cm(cm))
    assert not np.isnan(f1_from_cm(cm)[1])


def test_class_present_but_never_predicted_scores_zero():
    cm = CM.copy()
    cm[:, 0] += cm[:, 1]
    cm[:, 1] = 0  # nothing is predicted as S
    assert f1_from_cm(cm)[1] == 0.0 and recall_from_cm(cm)[1] == 0.0
    assert np.isnan(precision_from_cm(cm)[1])
    assert rare_macro_f1_from_cm(cm) == pytest.approx(f1_from_cm(cm)[3] / 2)


def test_batched_confusion_functions_match_the_scalar_ones():
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 50, size=(6, 3, 4, 4))
    f1 = f1_from_cm(batch)
    assert f1.shape == (6, 3, 4)
    for i in range(6):
        for j in range(3):
            np.testing.assert_allclose(f1[i, j], f1_from_cm(batch[i, j]))
            assert rare_macro_f1_from_cm(batch)[i, j] == pytest.approx(float(rare_macro_f1_from_cm(batch[i, j])))


def test_evaluate_from_probabilities():
    y_true, y_pred = _labels(CM)
    records = np.where(np.arange(len(y_true)) % 2 == 0, 101, 106)
    out = evaluate(_one_hot(y_pred), y_true, records)
    assert out["confusion_matrix"] == CM.tolist()
    assert out["rare_macro_f1"] == pytest.approx(0.4375)
    assert set(out["pr_auc"]) == {"N", "S", "V", "F"} and 0 < out["rare_macro_pr_auc"] <= 1
    assert set(out["per_record"]) == {101, 106}
    np.testing.assert_array_equal(np.sum([out["per_record"][r] for r in (101, 106)], axis=0), CM)
    unique, cms = confusion_by_record(y_true, y_pred, records)
    assert unique.tolist() == [101, 106] and cms.shape == (2, 4, 4)


def test_evaluate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        evaluate(np.zeros((5, 3)), np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        evaluate(np.zeros((5, 4)), np.zeros(4, dtype=int))
